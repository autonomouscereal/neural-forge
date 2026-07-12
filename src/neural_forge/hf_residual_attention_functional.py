"""Behavior-first functional scoring for cached parent/candidate traces.

This module intentionally ignores Q/K/V, weights, activations, residuals, and
cache similarity.  Fixed parent-driven forced trajectories are the primary
teacher signal.  Independent parent and candidate greedy trajectories are a
closed-loop retention/degeneration gate and remain valid after their prefixes
diverge.

The scorer is inference-only arithmetic over already-collected output logits
and token IDs.  It creates no optimizer, gradient, model mutation, or runtime
adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np


FUNCTIONAL_SCORE_SCHEMA = "neural-forge-residual-attention-functional-v2"
FUNCTIONAL_RANKING_SCHEMA = "neural-forge-residual-attention-functional-ranking-v2"
FUNCTIONAL_COMPARISON_SCHEMA = (
    "neural-forge-residual-attention-functional-comparison-v1"
)
FunctionalSplit = Literal[
    "calibration-selection",
    "calibration-exploratory",
    "heldout-evaluation",
]
RankingPurpose = Literal["selection", "evaluation"]


class ResidualAttentionFunctionalError(ValueError):
    """Functional trace evidence has an invalid shape, value, or pairing."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ResidualAttentionFunctionalError(
            "Functional evidence is not canonically serializable."
        ) from error


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ResidualAttentionFunctionalError(f"{name} must be positive.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ResidualAttentionFunctionalError(f"{name} must be positive.") from error
    if result <= 0:
        raise ResidualAttentionFunctionalError(f"{name} must be positive.")
    return result


def _nonnegative_token_ids(value: Any, name: str) -> tuple[int, ...]:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu().tolist()
    elif hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ResidualAttentionFunctionalError(f"{name} must be a token sequence.")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise ResidualAttentionFunctionalError(
                f"{name}[{index}] must be a non-negative token ID."
            )
        try:
            token_id = operator.index(item)
        except (TypeError, ValueError, OverflowError) as error:
            raise ResidualAttentionFunctionalError(
                f"{name}[{index}] must be a non-negative token ID."
            ) from error
        if token_id < 0:
            raise ResidualAttentionFunctionalError(
                f"{name}[{index}] must be a non-negative token ID."
            )
        result.append(token_id)
    return tuple(result)


def _logits(value: Any, name: str) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        detached = detach().cpu()
        dtype = getattr(detached, "dtype", None)
        if dtype is None or not bool(getattr(dtype, "is_floating_point", False)):
            raise ResidualAttentionFunctionalError(
                f"{name} must use a floating-point dtype."
            )
        value = detached.numpy()
    else:
        try:
            source = np.asarray(value)
        except (TypeError, ValueError) as error:
            raise ResidualAttentionFunctionalError(
                f"{name} must be a finite floating-point logit row."
            ) from error
        if source.dtype.kind != "f":
            raise ResidualAttentionFunctionalError(
                f"{name} must use a floating-point dtype."
            )
    try:
        result = np.array(value, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError) as error:
        raise ResidualAttentionFunctionalError(
            f"{name} must be a finite floating-point logit row."
        ) from error
    if result.ndim == 2 and tuple(result.shape[:1]) == (1,):
        result = result[0]
    if result.ndim != 1 or int(result.shape[0]) <= 1:
        raise ResidualAttentionFunctionalError(
            f"{name} must have shape [vocab] or [1,vocab], with vocab > 1."
        )
    if not bool(np.isfinite(result).all()):
        raise ResidualAttentionFunctionalError(f"{name} contains NaN or infinity.")
    result.setflags(write=False)
    return result


def _attribute(value: Any, name: str) -> Any:
    if not hasattr(value, name):
        raise ResidualAttentionFunctionalError(
            f"Cached trace is missing required attribute {name!r}."
        )
    return getattr(value, name)


def _token_id(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ResidualAttentionFunctionalError(
            f"{name} must be a non-negative integer token ID."
        )
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ResidualAttentionFunctionalError(
            f"{name} must be a non-negative integer token ID."
        ) from error
    if result < 0:
        raise ResidualAttentionFunctionalError(
            f"{name} must be a non-negative integer token ID."
        )
    return result


def _stable_top_token(row: np.ndarray) -> int:
    # np.argmax returns the lowest token ID on an exact tie.
    return int(np.argmax(row))


def _stable_rank(row: np.ndarray, token_id: int) -> int:
    value = float(row[token_id])
    greater = int(np.count_nonzero(row > value))
    tied_before = int(np.count_nonzero(row[:token_id] == value))
    return 1 + greater + tied_before


def _log_probability(row: np.ndarray, token_id: int) -> float:
    maximum = float(np.max(row))
    log_normalizer = maximum + math.log(float(np.exp(row - maximum).sum()))
    result = float(row[token_id]) - log_normalizer
    if not math.isfinite(result):
        raise ResidualAttentionFunctionalError(
            "Candidate parent-action log probability is non-finite."
        )
    return result


def _longest_run(values: Sequence[int]) -> int:
    longest = 0
    current = 0
    previous: int | None = None
    for value in values:
        if previous is not None and value == previous:
            current += 1
        else:
            current = 1
            previous = value
        longest = max(longest, current)
    return longest


def _repeat_fraction(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    return float(len(values) - len(set(values))) / len(values)


def _adjacent_repeat_fraction(values: Sequence[int]) -> float:
    if len(values) <= 1:
        return 0.0
    repeats = sum(left == right for left, right in zip(values, values[1:]))
    return float(repeats) / (len(values) - 1)


@dataclass(frozen=True, slots=True)
class _FunctionalTrajectory:
    sequence_id: str
    lane: str
    prompt_token_ids: tuple[int, ...]
    consumed_token_ids: tuple[int, ...]
    selected_token_ids: tuple[int, ...]
    logits: tuple[np.ndarray, ...]
    vocab_size: int
    function_sha256: str


def _trajectory(trace: Any, name: str) -> _FunctionalTrajectory:
    sequence_id = str(_attribute(trace, "sequence_id")).strip()
    lane = str(_attribute(trace, "lane")).strip().lower()
    if not sequence_id:
        raise ResidualAttentionFunctionalError(f"{name}.sequence_id cannot be blank.")
    if lane not in {"forced", "greedy"}:
        raise ResidualAttentionFunctionalError(f"{name}.lane must be forced or greedy.")
    prompt = _nonnegative_token_ids(
        _attribute(trace, "prompt_token_ids"), f"{name}.prompt_token_ids"
    )
    if not prompt:
        raise ResidualAttentionFunctionalError(
            f"{name}.prompt_token_ids cannot be empty."
        )
    consumed = _nonnegative_token_ids(
        _attribute(trace, "consumed_token_ids"), f"{name}.consumed_token_ids"
    )
    selected = _nonnegative_token_ids(
        _attribute(trace, "selected_token_ids"), f"{name}.selected_token_ids"
    )
    steps_value = _attribute(trace, "steps")
    if isinstance(steps_value, (str, bytes)) or not isinstance(steps_value, Sequence):
        raise ResidualAttentionFunctionalError(f"{name}.steps must be a sequence.")
    steps = tuple(steps_value)
    if not steps or len(selected) != len(steps):
        raise ResidualAttentionFunctionalError(
            f"{name} must have one selected token per non-empty step."
        )
    if consumed != selected[:-1]:
        raise ResidualAttentionFunctionalError(
            f"{name}.consumed_token_ids must equal selected_token_ids[:-1]."
        )
    rows: list[np.ndarray] = []
    vocab_size: int | None = None
    for step_index, (step, selected_id) in enumerate(zip(steps, selected, strict=True)):
        reported_index = _attribute(step, "step_index")
        if isinstance(reported_index, bool):
            raise ResidualAttentionFunctionalError(
                f"{name}.steps must use contiguous zero-based step_index values."
            )
        try:
            normalized_index = operator.index(reported_index)
        except (TypeError, ValueError, OverflowError) as error:
            raise ResidualAttentionFunctionalError(
                f"{name}.steps must use contiguous zero-based step_index values."
            ) from error
        if normalized_index != step_index:
            raise ResidualAttentionFunctionalError(
                f"{name}.steps must use contiguous zero-based step_index values."
            )
        if (
            _token_id(
                _attribute(step, "selected_token_id"),
                f"{name}.steps[{step_index}].selected_token_id",
            )
            != selected_id
        ):
            raise ResidualAttentionFunctionalError(
                f"{name}.steps[{step_index}] selected token disagrees with trace."
            )
        row = _logits(
            _attribute(step, "action_logits"),
            f"{name}.steps[{step_index}].action_logits",
        )
        if vocab_size is None:
            vocab_size = int(row.shape[0])
        elif int(row.shape[0]) != vocab_size:
            raise ResidualAttentionFunctionalError(
                f"{name} changes output vocabulary width across steps."
            )
        rows.append(row)
    assert vocab_size is not None
    all_ids = (*prompt, *consumed, *selected)
    if any(token_id >= vocab_size for token_id in all_ids):
        raise ResidualAttentionFunctionalError(
            f"{name} contains a token outside its output vocabulary."
        )
    if lane == "greedy":
        for step_index, (selected_id, row) in enumerate(
            zip(selected, rows, strict=True)
        ):
            if selected_id != _stable_top_token(row):
                raise ResidualAttentionFunctionalError(
                    f"{name}.steps[{step_index}] is not a true greedy action."
                )
    payload = {
        "sequence_id": sequence_id,
        "lane": lane,
        "prompt_token_ids": list(prompt),
        "consumed_token_ids": list(consumed),
        "selected_token_ids": list(selected),
        "logits": [row.tolist() for row in rows],
        "vocab_size": vocab_size,
    }
    return _FunctionalTrajectory(
        sequence_id=sequence_id,
        lane=lane,
        prompt_token_ids=prompt,
        consumed_token_ids=consumed,
        selected_token_ids=selected,
        logits=tuple(rows),
        vocab_size=vocab_size,
        function_sha256=_sha256(payload),
    )


TrajectoryKey = tuple[str, str, tuple[int, ...]]


def _deduplicated_trajectories(
    traces: Mapping[Any, Any], name: str
) -> Mapping[TrajectoryKey, _FunctionalTrajectory]:
    if not isinstance(traces, Mapping) or not traces:
        raise ResidualAttentionFunctionalError(f"{name} must be a non-empty mapping.")
    result: dict[TrajectoryKey, _FunctionalTrajectory] = {}
    for index, (_mapping_key, trace) in enumerate(traces.items()):
        value = _trajectory(trace, f"{name}[{index}]")
        key = (value.lane, value.sequence_id, value.prompt_token_ids)
        previous = result.get(key)
        if previous is not None and previous.function_sha256 != value.function_sha256:
            raise ResidualAttentionFunctionalError(
                f"{name} contains conflicting duplicate functional trajectories."
            )
        result[key] = value
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class ForcedFunctionalMetrics:
    sequence_count: int
    action_count: int
    parent_driver_top1_count: int
    parent_driver_top1_rate: float
    candidate_parent_top1_agreement_count: int
    candidate_parent_top1_agreement_rate: float
    candidate_parent_action_mean_logprob: float
    candidate_parent_action_min_logprob: float
    candidate_parent_action_topk_count: int
    candidate_parent_action_topk_rate: float
    candidate_parent_action_mean_rank: float
    candidate_parent_action_worst_rank: int

    def as_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class GreedySequenceMetrics:
    sequence_id: str
    parent_token_count: int
    candidate_token_count: int
    positional_match_count: int
    exact_token_retention_rate: float
    longest_common_prefix: int
    longest_common_prefix_rate: float
    first_divergence_step: int | None
    exact_rollout_match: bool
    parent_repeat_fraction: float
    candidate_repeat_fraction: float
    parent_adjacent_repeat_fraction: float
    candidate_adjacent_repeat_fraction: float
    parent_longest_run: int
    candidate_longest_run: int

    def as_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class GreedyFunctionalMetrics:
    sequence_count: int
    parent_token_count: int
    candidate_token_count: int
    positional_match_count: int
    exact_token_retention_rate: float
    longest_common_prefix_tokens: int
    longest_common_prefix_rate: float
    exact_rollout_match_count: int
    exact_rollout_match_rate: float
    earliest_divergence_step: int | None
    parent_mean_repeat_fraction: float
    candidate_mean_repeat_fraction: float
    candidate_repeat_fraction_excess: float
    parent_mean_adjacent_repeat_fraction: float
    candidate_mean_adjacent_repeat_fraction: float
    candidate_adjacent_repeat_fraction_excess: float
    parent_longest_run: int
    candidate_longest_run: int
    candidate_longest_run_excess: int
    sequences: tuple[GreedySequenceMetrics, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            **{
                field: getattr(self, field)
                for field in self.__dataclass_fields__
                if field != "sequences"
            },
            "sequences": [value.as_dict() for value in self.sequences],
        }


@dataclass(frozen=True, slots=True)
class ResidualAttentionFunctionalReport:
    candidate_id: str
    split: FunctionalSplit
    top_k: int
    output_vocab_size: int
    forced: ForcedFunctionalMetrics
    greedy: GreedyFunctionalMetrics
    parent_function_sha256: str
    candidate_function_sha256: str
    report_sha256: str

    @property
    def selection_allowed(self) -> bool:
        return (
            self.split == "calibration-selection"
            and self.fixed_parent_top1_anchor_valid
        )

    @property
    def fixed_parent_top1_anchor_valid(self) -> bool:
        return (
            self.forced.parent_driver_top1_count == self.forced.action_count
            and self.forced.parent_driver_top1_rate == 1.0
        )

    @property
    def heldout_evaluation(self) -> bool:
        return self.split == "heldout-evaluation"

    @property
    def calibration_exploratory(self) -> bool:
        return self.split == "calibration-exploratory"

    @property
    def ranking_key(self) -> tuple[Any, ...]:
        return functional_ranking_key(self)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": FUNCTIONAL_SCORE_SCHEMA,
            "candidate_id": self.candidate_id,
            "split": self.split,
            "split_semantics": (
                "candidate-selection-only; never promotion evidence"
                if self.selection_allowed
                else (
                    "calibration-selection-refused; forced drivers are not all "
                    "parent top-1 actions"
                    if self.split == "calibration-selection"
                    else (
                        "calibration-exploratory-only; no disjoint heldout split, "
                        "winner, selection, or promotion claim"
                        if self.calibration_exploratory
                        else (
                        "heldout-evaluation-only; excluded from candidate "
                        "fitting/selection"
                        )
                    )
                )
            ),
            "selection_allowed": self.selection_allowed,
            "calibration_exploratory": self.calibration_exploratory,
            "heldout_evaluation": self.heldout_evaluation,
            "top_k": self.top_k,
            "output_vocab_size": self.output_vocab_size,
            "forced_parent_anchor_primary": True,
            "fixed_parent_top1_anchor_valid": (
                self.fixed_parent_top1_anchor_valid
            ),
            "greedy_child_trajectory_role": (
                "closed-loop-retention-and-degeneration-gate; not sole teacher"
            ),
            "valid_after_greedy_prefix_divergence": True,
            "forced": self.forced.as_dict(),
            "greedy": self.greedy.as_dict(),
            "ranking_key": list(self.ranking_key),
            "ranking_uses_output_function_only": True,
            "excluded_similarity_inputs": [
                "weights",
                "qkv",
                "attention",
                "activations",
                "residuals",
                "cache",
            ],
            "parent_function_sha256": self.parent_function_sha256,
            "candidate_function_sha256": self.candidate_function_sha256,
            "report_sha256": self.report_sha256,
            "gradient_free": True,
            "optimizer_constructed": False,
            "training_used": False,
        }


def _forced_metrics(
    parent: Sequence[_FunctionalTrajectory],
    candidate: Sequence[_FunctionalTrajectory],
    *,
    top_k: int,
) -> ForcedFunctionalMetrics:
    action_count = 0
    driver_top1 = 0
    agreements = 0
    topk_count = 0
    logprobs: list[float] = []
    ranks: list[int] = []
    for parent_trace, candidate_trace in zip(parent, candidate, strict=True):
        if (
            parent_trace.consumed_token_ids != candidate_trace.consumed_token_ids
            or parent_trace.selected_token_ids != candidate_trace.selected_token_ids
            or len(parent_trace.logits) != len(candidate_trace.logits)
        ):
            raise ResidualAttentionFunctionalError(
                "Forced parent/candidate traces must share the exact parent-driven "
                "prefix and driver actions."
            )
        for step_index, (parent_row, candidate_row) in enumerate(
            zip(parent_trace.logits, candidate_trace.logits, strict=True)
        ):
            parent_action = _stable_top_token(parent_row)
            candidate_action = _stable_top_token(candidate_row)
            rank = _stable_rank(candidate_row, parent_action)
            action_count += 1
            driver_top1 += parent_trace.selected_token_ids[step_index] == parent_action
            agreements += candidate_action == parent_action
            topk_count += rank <= top_k
            ranks.append(rank)
            logprobs.append(_log_probability(candidate_row, parent_action))
    if action_count <= 0:
        raise ResidualAttentionFunctionalError("Forced traces contain no actions.")
    return ForcedFunctionalMetrics(
        sequence_count=len(parent),
        action_count=action_count,
        parent_driver_top1_count=driver_top1,
        parent_driver_top1_rate=driver_top1 / action_count,
        candidate_parent_top1_agreement_count=agreements,
        candidate_parent_top1_agreement_rate=agreements / action_count,
        candidate_parent_action_mean_logprob=float(np.mean(logprobs)),
        candidate_parent_action_min_logprob=float(np.min(logprobs)),
        candidate_parent_action_topk_count=topk_count,
        candidate_parent_action_topk_rate=topk_count / action_count,
        candidate_parent_action_mean_rank=float(np.mean(ranks)),
        candidate_parent_action_worst_rank=max(ranks),
    )


def _greedy_sequence(
    parent: _FunctionalTrajectory, candidate: _FunctionalTrajectory
) -> GreedySequenceMetrics:
    parent_tokens = parent.selected_token_ids
    candidate_tokens = candidate.selected_token_ids
    common_width = min(len(parent_tokens), len(candidate_tokens))
    positional_matches = sum(
        parent_tokens[index] == candidate_tokens[index] for index in range(common_width)
    )
    lcp = 0
    while lcp < common_width and parent_tokens[lcp] == candidate_tokens[lcp]:
        lcp += 1
    exact = parent_tokens == candidate_tokens
    if exact:
        divergence = None
    elif lcp < common_width:
        divergence = lcp
    else:
        divergence = common_width
    denominator = len(parent_tokens)
    if denominator <= 0:
        raise ResidualAttentionFunctionalError(
            "Parent greedy trajectory contains no output action."
        )
    return GreedySequenceMetrics(
        sequence_id=parent.sequence_id,
        parent_token_count=len(parent_tokens),
        candidate_token_count=len(candidate_tokens),
        positional_match_count=positional_matches,
        exact_token_retention_rate=positional_matches / denominator,
        longest_common_prefix=lcp,
        longest_common_prefix_rate=lcp / denominator,
        first_divergence_step=divergence,
        exact_rollout_match=exact,
        parent_repeat_fraction=_repeat_fraction(parent_tokens),
        candidate_repeat_fraction=_repeat_fraction(candidate_tokens),
        parent_adjacent_repeat_fraction=_adjacent_repeat_fraction(parent_tokens),
        candidate_adjacent_repeat_fraction=_adjacent_repeat_fraction(candidate_tokens),
        parent_longest_run=_longest_run(parent_tokens),
        candidate_longest_run=_longest_run(candidate_tokens),
    )


def _greedy_metrics(
    parent: Sequence[_FunctionalTrajectory],
    candidate: Sequence[_FunctionalTrajectory],
) -> GreedyFunctionalMetrics:
    sequences = tuple(
        _greedy_sequence(parent_trace, candidate_trace)
        for parent_trace, candidate_trace in zip(parent, candidate, strict=True)
    )
    parent_tokens = sum(value.parent_token_count for value in sequences)
    candidate_tokens = sum(value.candidate_token_count for value in sequences)
    matches = sum(value.positional_match_count for value in sequences)
    lcp = sum(value.longest_common_prefix for value in sequences)
    exact = sum(value.exact_rollout_match for value in sequences)
    divergences = [
        value.first_divergence_step
        for value in sequences
        if value.first_divergence_step is not None
    ]
    parent_repeat = float(
        np.mean([value.parent_repeat_fraction for value in sequences])
    )
    candidate_repeat = float(
        np.mean([value.candidate_repeat_fraction for value in sequences])
    )
    parent_adjacent = float(
        np.mean([value.parent_adjacent_repeat_fraction for value in sequences])
    )
    candidate_adjacent = float(
        np.mean([value.candidate_adjacent_repeat_fraction for value in sequences])
    )
    parent_run = max(value.parent_longest_run for value in sequences)
    candidate_run = max(value.candidate_longest_run for value in sequences)
    return GreedyFunctionalMetrics(
        sequence_count=len(sequences),
        parent_token_count=parent_tokens,
        candidate_token_count=candidate_tokens,
        positional_match_count=matches,
        exact_token_retention_rate=matches / parent_tokens,
        longest_common_prefix_tokens=lcp,
        longest_common_prefix_rate=lcp / parent_tokens,
        exact_rollout_match_count=exact,
        exact_rollout_match_rate=exact / len(sequences),
        earliest_divergence_step=(None if not divergences else min(divergences)),
        parent_mean_repeat_fraction=parent_repeat,
        candidate_mean_repeat_fraction=candidate_repeat,
        candidate_repeat_fraction_excess=max(0.0, candidate_repeat - parent_repeat),
        parent_mean_adjacent_repeat_fraction=parent_adjacent,
        candidate_mean_adjacent_repeat_fraction=candidate_adjacent,
        candidate_adjacent_repeat_fraction_excess=max(
            0.0, candidate_adjacent - parent_adjacent
        ),
        parent_longest_run=parent_run,
        candidate_longest_run=candidate_run,
        candidate_longest_run_excess=max(0, candidate_run - parent_run),
        sequences=sequences,
    )


def _split(value: Any) -> FunctionalSplit:
    normalized = str(value).strip().lower()
    if normalized not in {
        "calibration-selection",
        "calibration-exploratory",
        "heldout-evaluation",
    }:
        raise ResidualAttentionFunctionalError(
            "split must be calibration-selection, calibration-exploratory, or "
            "heldout-evaluation."
        )
    return normalized  # type: ignore[return-value]


def score_residual_attention_function(
    parent_traces: Mapping[Any, Any],
    candidate_traces: Mapping[Any, Any],
    *,
    candidate_id: str,
    split: FunctionalSplit,
    top_k: int = 5,
) -> ResidualAttentionFunctionalReport:
    """Score output function without using internal tensor similarity.

    ``calibration-selection`` may rank/select candidates but is never held-out
    promotion evidence. ``calibration-exploratory`` cannot rank/select because
    it has no disjoint evaluation split. ``heldout-evaluation`` must be excluded
    from fitting and selection and is reported as evaluation-only evidence.
    """

    identifier = str(candidate_id).strip()
    if not identifier:
        raise ResidualAttentionFunctionalError("candidate_id cannot be blank.")
    normalized_split = _split(split)
    requested_top_k = _positive_int(top_k, "top_k")
    parent_map = _deduplicated_trajectories(parent_traces, "parent_traces")
    candidate_map = _deduplicated_trajectories(candidate_traces, "candidate_traces")
    if set(parent_map) != set(candidate_map):
        missing = sorted(set(parent_map) - set(candidate_map), key=repr)
        extra = sorted(set(candidate_map) - set(parent_map), key=repr)
        raise ResidualAttentionFunctionalError(
            "Parent/candidate functional trajectory keys differ; "
            f"missing={missing}, extra={extra}."
        )
    ordered_keys = tuple(sorted(parent_map, key=repr))
    forced_keys = tuple(key for key in ordered_keys if key[0] == "forced")
    greedy_keys = tuple(key for key in ordered_keys if key[0] == "greedy")
    if not forced_keys or not greedy_keys:
        raise ResidualAttentionFunctionalError(
            "Functional scoring requires both forced and greedy trajectories."
        )
    parent_vocab_sizes = {parent_map[key].vocab_size for key in ordered_keys}
    candidate_vocab_sizes = {candidate_map[key].vocab_size for key in ordered_keys}
    if len(parent_vocab_sizes) != 1 or parent_vocab_sizes != candidate_vocab_sizes:
        raise ResidualAttentionFunctionalError(
            "Parent and candidate must use one identical output vocabulary width."
        )
    output_vocab = next(iter(parent_vocab_sizes))
    if requested_top_k > output_vocab:
        raise ResidualAttentionFunctionalError(
            "top_k cannot exceed the output vocabulary width."
        )
    parent_forced = tuple(parent_map[key] for key in forced_keys)
    candidate_forced = tuple(candidate_map[key] for key in forced_keys)
    parent_greedy = tuple(parent_map[key] for key in greedy_keys)
    candidate_greedy = tuple(candidate_map[key] for key in greedy_keys)
    forced_metrics = _forced_metrics(
        parent_forced,
        candidate_forced,
        top_k=requested_top_k,
    )
    greedy_metrics = _greedy_metrics(parent_greedy, candidate_greedy)
    parent_hash = _sha256([parent_map[key].function_sha256 for key in ordered_keys])
    candidate_hash = _sha256(
        [candidate_map[key].function_sha256 for key in ordered_keys]
    )
    hash_payload = {
        "schema": FUNCTIONAL_SCORE_SCHEMA,
        "candidate_id": identifier,
        "split": normalized_split,
        "top_k": requested_top_k,
        "output_vocab_size": output_vocab,
        "forced": forced_metrics.as_dict(),
        "greedy": greedy_metrics.as_dict(),
        "parent_function_sha256": parent_hash,
        "candidate_function_sha256": candidate_hash,
    }
    return ResidualAttentionFunctionalReport(
        candidate_id=identifier,
        split=normalized_split,
        top_k=requested_top_k,
        output_vocab_size=output_vocab,
        forced=forced_metrics,
        greedy=greedy_metrics,
        parent_function_sha256=parent_hash,
        candidate_function_sha256=candidate_hash,
        report_sha256=_sha256(hash_payload),
    )


def functional_ranking_key(
    report: ResidualAttentionFunctionalReport,
) -> tuple[Any, ...]:
    """Return an ascending deterministic key using output function only.

    Fixed-parent forced metrics are deliberately ordered before child-greedy
    retention and degeneration metrics. The candidate ID is the final stable
    tie breaker.
    """

    if not isinstance(report, ResidualAttentionFunctionalReport):
        raise ResidualAttentionFunctionalError(
            "report must be a ResidualAttentionFunctionalReport."
        )
    return (
        -report.forced.candidate_parent_top1_agreement_rate,
        -report.forced.candidate_parent_action_mean_logprob,
        -report.forced.candidate_parent_action_topk_rate,
        report.forced.candidate_parent_action_mean_rank,
        report.forced.candidate_parent_action_worst_rank,
        -report.greedy.exact_token_retention_rate,
        -report.greedy.longest_common_prefix_rate,
        -report.greedy.exact_rollout_match_rate,
        report.greedy.candidate_repeat_fraction_excess,
        report.greedy.candidate_adjacent_repeat_fraction_excess,
        report.greedy.candidate_longest_run_excess,
        report.candidate_id,
    )


@dataclass(frozen=True, slots=True)
class RankedFunctionalCandidate:
    rank: int
    candidate_id: str
    report_sha256: str
    ranking_key: tuple[Any, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "candidate_id": self.candidate_id,
            "report_sha256": self.report_sha256,
            "ranking_key": list(self.ranking_key),
        }


@dataclass(frozen=True, slots=True)
class FunctionalRanking:
    purpose: RankingPurpose
    split: FunctionalSplit
    candidates: tuple[RankedFunctionalCandidate, ...]
    ranking_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": FUNCTIONAL_RANKING_SCHEMA,
            "purpose": self.purpose,
            "split": self.split,
            "selection_allowed": self.purpose == "selection",
            "fixed_parent_top1_anchor_verified": self.purpose == "selection",
            "candidates": [value.as_dict() for value in self.candidates],
            "ranking_sha256": self.ranking_sha256,
            "functional_outputs_only": True,
        }


def rank_functional_reports(
    reports: Sequence[ResidualAttentionFunctionalReport],
    *,
    purpose: RankingPurpose,
) -> FunctionalRanking:
    """Rank one homogeneous calibration-selection or held-out-evaluation set."""

    values = tuple(reports)
    if not values or any(
        not isinstance(value, ResidualAttentionFunctionalReport) for value in values
    ):
        raise ResidualAttentionFunctionalError(
            "reports must be a non-empty functional-report sequence."
        )
    normalized_purpose = str(purpose).strip().lower()
    if normalized_purpose not in {"selection", "evaluation"}:
        raise ResidualAttentionFunctionalError(
            "purpose must be selection or evaluation."
        )
    splits = {value.split for value in values}
    if len(splits) != 1:
        raise ResidualAttentionFunctionalError(
            "Ranking cannot mix calibration and held-out reports."
        )
    split = next(iter(splits))
    expected_split = (
        "calibration-selection"
        if normalized_purpose == "selection"
        else "heldout-evaluation"
    )
    if split != expected_split:
        raise ResidualAttentionFunctionalError(
            f"{normalized_purpose} ranking requires {expected_split} reports."
        )
    if normalized_purpose == "selection" and any(
        not value.fixed_parent_top1_anchor_valid for value in values
    ):
        raise ResidualAttentionFunctionalError(
            "Calibration selection requires every forced driver to be the "
            "parent top-1 action (parent_driver_top1_rate == 1.0)."
        )
    candidate_ids = [value.candidate_id for value in values]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ResidualAttentionFunctionalError(
            "Functional ranking candidate IDs must be unique."
        )
    contracts = {(value.top_k, value.output_vocab_size) for value in values}
    parent_hashes = {value.parent_function_sha256 for value in values}
    if len(contracts) != 1 or len(parent_hashes) != 1:
        raise ResidualAttentionFunctionalError(
            "Ranked reports must share top-k, vocabulary, and parent evidence."
        )
    ordered = sorted(values, key=functional_ranking_key)
    candidates = tuple(
        RankedFunctionalCandidate(
            rank=index,
            candidate_id=value.candidate_id,
            report_sha256=value.report_sha256,
            ranking_key=functional_ranking_key(value),
        )
        for index, value in enumerate(ordered, start=1)
    )
    payload = {
        "schema": FUNCTIONAL_RANKING_SCHEMA,
        "purpose": normalized_purpose,
        "split": split,
        "candidates": [value.as_dict() for value in candidates],
    }
    return FunctionalRanking(
        purpose=normalized_purpose,  # type: ignore[arg-type]
        split=split,
        candidates=candidates,
        ranking_sha256=_sha256(payload),
    )


def build_residual_attention_functional_comparison(
    parent_calibration_traces: Mapping[Any, Any],
    parent_heldout_traces: Mapping[Any, Any],
    candidate_calibration_traces: Mapping[str, Mapping[Any, Any]],
    candidate_heldout_traces: Mapping[str, Mapping[Any, Any]],
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Build one leakage-resistant selection/evaluation report bundle.

    Candidate fitting happens outside this scorer.  This function scores every
    variant twice, selects and names a winner from calibration functional
    outputs only, and emits a separate held-out ordering with no winner.  The
    calibration ranking fails closed unless every forced driver is the actual
    parent top-1 action.
    """

    calibration_candidates = dict(candidate_calibration_traces)
    heldout_candidates = dict(candidate_heldout_traces)
    if not calibration_candidates or set(calibration_candidates) != set(
        heldout_candidates
    ):
        raise ResidualAttentionFunctionalError(
            "Calibration and held-out candidate variant IDs must be identical "
            "and non-empty."
        )
    raw_identifiers = tuple(calibration_candidates)
    if any(
        not isinstance(value, str) or not value.strip()
        for value in raw_identifiers
    ):
        raise ResidualAttentionFunctionalError(
            "Functional comparison candidate IDs must be non-empty strings."
        )
    identifiers = tuple(sorted(raw_identifiers))
    if len({value.strip() for value in identifiers}) != len(identifiers):
        raise ResidualAttentionFunctionalError(
            "Functional comparison candidate IDs must be unique after trimming."
        )
    calibration_reports = tuple(
        score_residual_attention_function(
            parent_calibration_traces,
            calibration_candidates[identifier],
            candidate_id=identifier,
            split="calibration-selection",
            top_k=top_k,
        )
        for identifier in identifiers
    )
    heldout_reports = tuple(
        score_residual_attention_function(
            parent_heldout_traces,
            heldout_candidates[identifier],
            candidate_id=identifier,
            split="heldout-evaluation",
            top_k=top_k,
        )
        for identifier in identifiers
    )
    calibration_ranking = rank_functional_reports(
        calibration_reports,
        purpose="selection",
    )
    heldout_ordering = rank_functional_reports(
        heldout_reports,
        purpose="evaluation",
    )
    calibration_by_id = {value.candidate_id: value for value in calibration_reports}
    heldout_by_id = {value.candidate_id: value for value in heldout_reports}
    payload: dict[str, Any] = {
        "schema": FUNCTIONAL_COMPARISON_SCHEMA,
        "variants": {
            identifier: {
                "calibration_selection": calibration_by_id[identifier].as_dict(),
                "heldout_evaluation": heldout_by_id[identifier].as_dict(),
            }
            for identifier in identifiers
        },
        "calibration_selection": {
            "winner_candidate_id": calibration_ranking.candidates[0].candidate_id,
            "ranking": calibration_ranking.as_dict(),
            "fixed_parent_top1_anchor_verified": True,
            "functional_outputs_only": True,
            "heldout_used_for_selection": False,
            "internal_similarity_used_for_selection": False,
        },
        "heldout_evaluation": {
            "ordering": heldout_ordering.as_dict(),
            "winner_named": False,
            "used_for_fitting": False,
            "used_for_selection": False,
            "evaluation_only": True,
        },
        "selection_contract": {
            "winner_named_from_calibration_only": True,
            "forced_parent_action_metrics_rank_before_greedy_metrics": True,
            "greedy_prefix_divergence_remains_scoreable": True,
            "attention_residual_cache_and_weight_similarity_excluded": True,
        },
    }
    payload["comparison_sha256"] = _sha256(payload)
    return payload


# Descriptive aliases for callers that prefer the module name in the API.
score_residual_attention_functional = score_residual_attention_function
rank_residual_attention_functional = rank_functional_reports


__all__ = [
    "FUNCTIONAL_COMPARISON_SCHEMA",
    "FUNCTIONAL_RANKING_SCHEMA",
    "FUNCTIONAL_SCORE_SCHEMA",
    "ForcedFunctionalMetrics",
    "FunctionalRanking",
    "FunctionalSplit",
    "GreedyFunctionalMetrics",
    "GreedySequenceMetrics",
    "RankedFunctionalCandidate",
    "RankingPurpose",
    "ResidualAttentionFunctionalError",
    "ResidualAttentionFunctionalReport",
    "build_residual_attention_functional_comparison",
    "functional_ranking_key",
    "rank_functional_reports",
    "rank_residual_attention_functional",
    "score_residual_attention_function",
    "score_residual_attention_functional",
]
