"""Cached, validation-only causal evidence for large LFM2 FFN folds.

This module turns the already sealed sixteen-case recovery cache into the
missing producer for :mod:`neural_forge.hf_lfm2_global_causal_allocator`.
It never calls, generates from, or rebuilds the parent.  A physical medoid
proposal is installed in a child, the child is traced with uncached
full-prefix forwards, and downstream damage is measured on rows that were not
used to choose the medoids.

The physical source evidence deliberately carries geometry-only placeholder
curve costs.  Those costs are required by the existing variable-width payload
builder but are never consumed by the global causal allocator.  Allocation is
driven only by the six measured downstream channels recorded here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from neural_forge.hf_lfm2_circuit_trace_cache import (
    FULL_PREFIX_CANONICAL_TRACE_MODE,
)
from neural_forge.hf_lfm2_global_causal_allocator import (
    Lfm2CausalDamage,
    Lfm2CausalDamageWeights,
    Lfm2CausalRemovalCandidate,
    Lfm2GlobalCausalEvidence,
    Lfm2GlobalCausalEvidenceError,
    Lfm2GlobalCausalInfeasible,
    Lfm2GlobalCausalPlan,
    lfm2_global_causal_budget,
    plan_lfm2_global_causal_target,
    seal_lfm2_global_causal_evidence,
)
from neural_forge.hf_lfm2_variable_width_fold import (
    LFM2_VARIABLE_WIDTH_LAYER_COUNT,
    Lfm2LayerWidthCurve,
    Lfm2VariableWidthEvidence,
    Lfm2WidthCostPoint,
    seal_lfm2_variable_width_evidence,
)


SPLIT_SCHEMA = "neural-forge-lfm2-cached-causal-split-v1"
ROLLOUT_SCHEMA = "neural-forge-lfm2-full-prefix-rollout-v1"
MEASUREMENT_SCHEMA = "neural-forge-lfm2-cached-causal-measurement-v1"
FRONTIER_SCHEMA = "neural-forge-lfm2-cached-causal-frontier-v1"
PHYSICAL_EVIDENCE_BUILDER_ID = (
    "lfm2.cached-causal.physical-medoid-source-rows.geometry-only-v1"
)
DEFAULT_COARSE_WIDTHS = (1792, 1280, 896, 512)
_PATH_PATTERN = re.compile(
    r"batch\.(?P<batch>\d{6})/"
    r"(?P<phase>prefill|decode\.(?P<step>\d{3}))/"
    r"row\.(?P<row>\d{4})"
)


class Lfm2CachedCausalEvidenceError(RuntimeError):
    """Cached causal evidence is missing, stale, leaked, or malformed."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2CachedCausalEvidenceError(
            "Cached causal provenance must be finite canonical JSON."
        ) from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CachedCausalEvidenceError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CachedCausalEvidenceError(f"{name} must be a positive integer.")
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2CachedCausalEvidenceError(f"{name} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CachedCausalEvidenceError(
            f"{name} must be finite and non-negative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2CachedCausalEvidenceError(f"{name} must be finite and non-negative.")
    return result


def _as_indices(value: Any, name: str, *, row_count: int) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise Lfm2CachedCausalEvidenceError(f"{name} must be an index sequence.")
    result = tuple(value)
    if not result or any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < row_count
        for index in result
    ):
        raise Lfm2CachedCausalEvidenceError(
            f"{name} must contain in-range integer row indices."
        )
    return result


@dataclass(frozen=True, slots=True)
class Lfm2CachedCausalSplit:
    """Authenticated row-disjoint proposal/fit/validation partition."""

    schema: str
    profile_name: str
    profile_sha256: str
    parent_trace_rows_sha256: str
    row_count: int
    selection_indices: tuple[int, ...]
    fit_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    selection_rows_sha256: str
    fit_rows_sha256: str
    validation_rows_sha256: str
    validation_case_ids: tuple[str, ...]
    validation_case_ids_sha256: str
    split_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "profile_name": self.profile_name,
            "profile_sha256": self.profile_sha256,
            "parent_trace_rows_sha256": self.parent_trace_rows_sha256,
            "row_count": self.row_count,
            "selection_indices": list(self.selection_indices),
            "fit_indices": list(self.fit_indices),
            "validation_indices": list(self.validation_indices),
            "selection_rows_sha256": self.selection_rows_sha256,
            "fit_rows_sha256": self.fit_rows_sha256,
            "validation_rows_sha256": self.validation_rows_sha256,
            "validation_case_ids": list(self.validation_case_ids),
            "validation_case_ids_sha256": self.validation_case_ids_sha256,
            "row_sets_disjoint": True,
            "validation_indices_unique": True,
            "manual_case_count": 0,
            "split_sha256": self.split_sha256,
        }


def _split_body(value: Lfm2CachedCausalSplit) -> dict[str, Any]:
    body = value.as_dict()
    body.pop("split_sha256", None)
    return body


def _validate_split(value: Lfm2CachedCausalSplit) -> Lfm2CachedCausalSplit:
    if not isinstance(value, Lfm2CachedCausalSplit) or value.schema != SPLIT_SCHEMA:
        raise Lfm2CachedCausalEvidenceError(
            "Cached causal split has the wrong type or schema."
        )
    row_count = _positive_int(value.row_count, "row_count")
    selection = _as_indices(
        value.selection_indices,
        "selection_indices",
        row_count=row_count,
    )
    fit = _as_indices(value.fit_indices, "fit_indices", row_count=row_count)
    validation = _as_indices(
        value.validation_indices,
        "validation_indices",
        row_count=row_count,
    )
    selection_set = set(selection)
    fit_set = set(fit)
    validation_set = set(validation)
    if (
        selection_set & fit_set
        or selection_set & validation_set
        or fit_set & validation_set
        or len(validation) != len(validation_set)
    ):
        raise Lfm2CachedCausalEvidenceError(
            "Selection, fit, and validation evidence must be row-disjoint; "
            "validation rows must also be unique."
        )
    for name in (
        "profile_sha256",
        "parent_trace_rows_sha256",
        "selection_rows_sha256",
        "fit_rows_sha256",
        "validation_rows_sha256",
        "validation_case_ids_sha256",
        "split_sha256",
    ):
        _digest(getattr(value, name), name)
    if (
        _json_sha256(list(selection)) != value.selection_rows_sha256
        or _json_sha256(list(fit)) != value.fit_rows_sha256
        or _json_sha256(list(validation)) != value.validation_rows_sha256
        or _json_sha256(list(value.validation_case_ids))
        != value.validation_case_ids_sha256
        or _json_sha256(_split_body(value)) != value.split_sha256
    ):
        raise Lfm2CachedCausalEvidenceError(
            "Cached causal split metadata changed after sealing."
        )
    return value


def build_lfm2_cached_causal_split(
    profile: Any,
    descriptors: Sequence[Mapping[str, Any]],
    *,
    parent_trace_rows_sha256: str,
) -> Lfm2CachedCausalSplit:
    """Seal the existing recovery profile and reject any row leakage."""

    values = tuple(descriptors)
    if not values:
        raise Lfm2CachedCausalEvidenceError("Trace descriptors must be nonempty.")
    row_count = len(values)
    selection = _as_indices(
        getattr(profile, "selection_indices", None),
        "selection_indices",
        row_count=row_count,
    )
    fit = _as_indices(
        getattr(profile, "fit_indices", None),
        "fit_indices",
        row_count=row_count,
    )
    validation = _as_indices(
        getattr(profile, "validation_indices", None),
        "validation_indices",
        row_count=row_count,
    )
    validation_case_ids = tuple(
        dict.fromkeys(str(values[index]["case_id"]) for index in validation)
    )
    if not validation_case_ids or any(
        not case_id.startswith(("recovery_math_", "recovery_code_"))
        for case_id in validation_case_ids
    ):
        raise Lfm2CachedCausalEvidenceError(
            "Validation evidence must contain only sealed recovery cases."
        )
    value = Lfm2CachedCausalSplit(
        schema=SPLIT_SCHEMA,
        profile_name=str(getattr(profile, "name", "")),
        profile_sha256=_digest(
            getattr(profile, "profile_sha256", None),
            "profile.profile_sha256",
        ),
        parent_trace_rows_sha256=_digest(
            parent_trace_rows_sha256,
            "parent_trace_rows_sha256",
        ),
        row_count=row_count,
        selection_indices=selection,
        fit_indices=fit,
        validation_indices=validation,
        selection_rows_sha256=_json_sha256(list(selection)),
        fit_rows_sha256=_json_sha256(list(fit)),
        validation_rows_sha256=_json_sha256(list(validation)),
        validation_case_ids=validation_case_ids,
        validation_case_ids_sha256=_json_sha256(list(validation_case_ids)),
        split_sha256="",
    )
    value = replace(value, split_sha256=_json_sha256(_split_body(value)))
    return _validate_split(value)


def lfm2_large_target_width_frontier(
    *,
    parent_unique_parameters: int,
    hidden_size: int,
    source_width: int,
    layer_count: int = LFM2_VARIABLE_WIDTH_LAYER_COUNT,
    target_fraction: float = 0.35,
    coarse_widths: Sequence[int] = DEFAULT_COARSE_WIDTHS,
) -> tuple[tuple[int, ...], ...]:
    """Return a broad frontier that contains an exact large-target solution."""

    source = _positive_int(source_width, "source_width")
    layers = _positive_int(layer_count, "layer_count")
    if layers != LFM2_VARIABLE_WIDTH_LAYER_COUNT:
        raise Lfm2CachedCausalEvidenceError(
            "The current physical LFM2 installer requires fourteen layers."
        )
    budget = lfm2_global_causal_budget(
        target_fraction,
        parent_unique_parameters=parent_unique_parameters,
        hidden_size=hidden_size,
    )
    exact_total_width = layers * source - budget.removed_ffn_neurons
    if not 0 < exact_total_width < layers * source:
        raise Lfm2CachedCausalEvidenceError(
            "The requested target is outside the physical FFN width range."
        )
    floor_width, remainder = divmod(exact_total_width, layers)
    ceil_width = floor_width + int(remainder > 0)
    if floor_width <= 0 or ceil_width >= source:
        raise Lfm2CachedCausalEvidenceError(
            "Exact feasibility widths are not strict positive contractions."
        )
    shared = {
        int(width)
        for width in coarse_widths
        if (
            not isinstance(width, bool)
            and isinstance(width, int)
            and 0 < int(width) < source
        )
    }
    shared.update((floor_width, ceil_width))
    if len(shared) < 3:
        raise Lfm2CachedCausalEvidenceError(
            "The causal frontier needs at least three contraction widths."
        )
    ordered = (source, *sorted(shared, reverse=True))
    widths = tuple(tuple(ordered) for _index in range(layers))
    # The floor/ceil combination itself proves exact feasibility while all
    # coarser points remain free for the measured global allocation.
    exact_check = remainder * ceil_width + (layers - remainder) * floor_width
    if exact_check != exact_total_width:
        raise Lfm2CachedCausalEvidenceError(
            "Large-target width frontier lost its exact solution."
        )
    return widths


def _index_rows(value: Any, indices: Sequence[int]) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for cached causal evidence.") from exc
    index = torch.tensor(tuple(indices), dtype=torch.long, device=value.device)
    return value.index_select(0, index).detach().to(device="cpu").contiguous()


def build_lfm2_causal_physical_source_evidence(
    parent: Any,
    parent_rows: Any,
    split: Lfm2CachedCausalSplit,
    widths_by_layer: Sequence[Sequence[int]],
    *,
    cache_key: str,
    heldout_corpus_sha256: str,
) -> Lfm2VariableWidthEvidence:
    """Build medoid source rows without exposing validation rows to selection."""

    _validate_split(split)
    widths = tuple(tuple(values) for values in widths_by_layer)
    if len(widths) != LFM2_VARIABLE_WIDTH_LAYER_COUNT:
        raise Lfm2CachedCausalEvidenceError(
            "Width frontier must cover all fourteen layers."
        )
    layer_rows = {}
    curves = []
    source_width: int | None = None
    for layer_index, layer_widths in enumerate(widths):
        if (
            not layer_widths
            or any(
                isinstance(width, bool) or not isinstance(width, int) or width <= 0
                for width in layer_widths
            )
            or any(left <= right for left, right in zip(layer_widths, layer_widths[1:]))
        ):
            raise Lfm2CachedCausalEvidenceError(
                "Each width frontier must be positive and strictly descending."
            )
        current_source = int(parent_rows.field(layer_index, "mlp.gate").shape[1])
        source_width = current_source if source_width is None else source_width
        if (
            current_source != source_width
            or layer_widths[0] != current_source
            or any(width > current_source for width in layer_widths)
        ):
            raise Lfm2CachedCausalEvidenceError(
                "Width frontiers do not match the cached dense parent."
            )
        gate = _index_rows(
            parent_rows.field(layer_index, "mlp.gate"),
            split.selection_indices,
        )
        up = _index_rows(
            parent_rows.field(layer_index, "mlp.up"),
            split.selection_indices,
        )
        post = _index_rows(
            parent_rows.field(layer_index, "mlp.post_swiglu"),
            split.selection_indices,
        )
        layer_rows[layer_index] = {
            "gate": gate,
            "up": up,
            "post_swiglu": post,
        }
        points = []
        for width in layer_widths:
            # Geometry-only monotonic placeholders are required solely to
            # authorize a fixed physical medoid payload.  They are never used
            # as candidate damage or as an allocation objective.
            geometry = (current_source - width) / current_source
            points.append(
                Lfm2WidthCostPoint(
                    width=width,
                    whole_trajectory_cost=geometry,
                    residual_cost=geometry,
                )
            )
        curves.append(
            Lfm2LayerWidthCurve(
                layer_index=layer_index,
                points=tuple(points),
            )
        )
    return seal_lfm2_variable_width_evidence(
        parent,
        layer_rows,
        curves,
        cache_key=_digest(cache_key, "cache_key"),
        heldout_corpus_sha256=_digest(
            heldout_corpus_sha256,
            "heldout_corpus_sha256",
        ),
        evidence_builder_id=(f"{PHYSICAL_EVIDENCE_BUILDER_ID}:{split.split_sha256}"),
    )


def _maximum_repeated_bigram_share(token_ids: Sequence[int]) -> float:
    values = tuple(int(value) for value in token_ids)
    if len(values) < 2:
        return 0.0
    counts: dict[tuple[int, int], int] = {}
    for index in range(len(values) - 1):
        pair = (values[index], values[index + 1])
        counts[pair] = counts.get(pair, 0) + 1
    repeated = sum(max(0, count - 1) for count in counts.values())
    return repeated / (len(values) - 1)


def _longest_identical_token_run(token_ids: Sequence[int]) -> int:
    values = tuple(int(value) for value in token_ids)
    longest = 0
    current = 0
    previous: int | None = None
    for value in values:
        if value == previous:
            current += 1
        else:
            previous = value
            current = 1
        longest = max(longest, current)
    return longest


def _model_device(model: Any) -> Any:
    devices = {parameter.device for parameter in model.parameters()}
    if len(devices) != 1:
        raise Lfm2CachedCausalEvidenceError(
            "Full-prefix rollout requires one materialized child device."
        )
    return next(iter(devices))


def _output_logits(output: Any) -> Any:
    logits = (
        output.get("logits")
        if isinstance(output, Mapping)
        else getattr(output, "logits", None)
    )
    if logits is None:
        raise Lfm2CachedCausalEvidenceError(
            "Child full-prefix forward did not return logits."
        )
    return logits


@dataclass(frozen=True, slots=True)
class Lfm2FullPrefixRollout:
    case_id: str
    token_ids: tuple[int, ...]
    output_text: str
    eos_reached: bool
    eos_step: int | None
    generation_limit_reached: bool
    maximum_repeated_bigram_share: float
    parent_maximum_repeated_bigram_share: float
    longest_identical_token_run: int
    parent_longest_identical_token_run: int
    task_relevant: bool | None
    task_correct: bool | None
    parent_task_relevant: bool | None
    parent_task_correct: bool | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ROLLOUT_SCHEMA,
            "case_id": self.case_id,
            "token_ids": list(self.token_ids),
            "output_text": self.output_text,
            "eos_reached": self.eos_reached,
            "eos_step": self.eos_step,
            "generation_limit_reached": self.generation_limit_reached,
            "maximum_repeated_bigram_share": (self.maximum_repeated_bigram_share),
            "parent_maximum_repeated_bigram_share": (
                self.parent_maximum_repeated_bigram_share
            ),
            "longest_identical_token_run": self.longest_identical_token_run,
            "parent_longest_identical_token_run": (
                self.parent_longest_identical_token_run
            ),
            "task_relevant": self.task_relevant,
            "task_correct": self.task_correct,
            "parent_task_relevant": self.parent_task_relevant,
            "parent_task_correct": self.parent_task_correct,
            "decode_trace_mode": FULL_PREFIX_CANONICAL_TRACE_MODE,
            "use_cache": False,
            "parent_forward_calls": 0,
        }


def generate_lfm2_full_prefix_rollouts(
    child: Any,
    recovery_rows: Sequence[Mapping[str, Any]],
    *,
    validation_case_ids: Sequence[str],
    tokenizer: Any,
    baseline: Any,
    eos_token_ids: Sequence[int],
    generation_limit: int,
    rollout_batch_size: int = 16,
) -> tuple[Lfm2FullPrefixRollout, ...]:
    """Greedily decode exact sealed prompts using only uncached child forwards."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for cached causal evidence.") from exc
    limit = _positive_int(generation_limit, "generation_limit")
    batch_size = _positive_int(rollout_batch_size, "rollout_batch_size")
    wanted = tuple(validation_case_ids)
    wanted_set = set(wanted)
    if not wanted or len(wanted_set) != len(wanted):
        raise Lfm2CachedCausalEvidenceError(
            "Validation rollout case IDs must be nonempty and unique."
        )
    row_map = {str(row["id"]): row for row in recovery_rows}
    if set(row_map).intersection(wanted_set) != wanted_set:
        raise Lfm2CachedCausalEvidenceError(
            "A validation case is absent from the sealed recovery rows."
        )
    case_map = {case.case_id: case for case in baseline.cases}
    if set(case_map).intersection(wanted_set) != wanted_set:
        raise Lfm2CachedCausalEvidenceError(
            "A validation case is absent from the cached task baseline."
        )
    eos_ids = {
        int(value)
        for value in eos_token_ids
        if not isinstance(value, bool) and isinstance(value, int) and value >= 0
    }
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
        pad_token_id = min(eos_ids) if eos_ids else 0
    device = _model_device(child)
    was_training = bool(child.training)
    child.eval()
    result = []
    try:
        with torch.inference_mode():
            for offset in range(0, len(wanted), batch_size):
                chunk_ids = wanted[offset : offset + batch_size]
                states = []
                for case_id in chunk_ids:
                    row = row_map[case_id]
                    prompt = tuple(int(value) for value in row["input_ids"])
                    mask = tuple(int(value) for value in row["attention_mask"])
                    if (
                        not prompt
                        or len(prompt) != len(mask)
                        or any(value not in (0, 1) for value in mask)
                        or mask[-1] != 1
                    ):
                        raise Lfm2CachedCausalEvidenceError(
                            f"Sealed prompt is malformed for {case_id!r}."
                        )
                    states.append(
                        {
                            "case_id": case_id,
                            "tokens": list(prompt),
                            "mask": list(mask),
                            "generated": [],
                            "eos_step": None,
                        }
                    )
                for step in range(limit):
                    active = [state for state in states if state["eos_step"] is None]
                    if not active:
                        break
                    maximum = max(len(state["tokens"]) for state in active)
                    ids = []
                    masks = []
                    for state in active:
                        padding = maximum - len(state["tokens"])
                        ids.append([pad_token_id] * padding + state["tokens"])
                        masks.append([0] * padding + state["mask"])
                    input_ids = torch.tensor(
                        ids,
                        dtype=torch.long,
                        device=device,
                    )
                    attention_mask = torch.tensor(
                        masks,
                        dtype=torch.long,
                        device=device,
                    )
                    position_ids = attention_mask.cumsum(dim=-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 0)
                    output = child(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                        return_dict=True,
                    )
                    logits = _output_logits(output)
                    if (
                        not torch.is_tensor(logits)
                        or logits.ndim != 3
                        or int(logits.shape[0]) != len(active)
                        or int(logits.shape[1]) != maximum
                    ):
                        raise Lfm2CachedCausalEvidenceError(
                            "Child full-prefix logits have an invalid shape."
                        )
                    next_ids = logits[:, -1, :].argmax(dim=-1).tolist()
                    for state, next_id in zip(active, next_ids, strict=True):
                        token = int(next_id)
                        state["generated"].append(token)
                        state["tokens"].append(token)
                        state["mask"].append(1)
                        if token in eos_ids:
                            state["eos_step"] = step
                for state in states:
                    case_id = str(state["case_id"])
                    generated = tuple(int(value) for value in state["generated"])
                    text = tokenizer.decode(
                        list(generated),
                        skip_special_tokens=True,
                    )
                    case = case_map[case_id]
                    parent_share = float(
                        case.parent_metadata.maximum_repeated_ngram_share
                    )
                    parent_output_ids = tuple(
                        int(value) for value in row_map[case_id]["parent_output_ids"]
                    )
                    result.append(
                        Lfm2FullPrefixRollout(
                            case_id=case_id,
                            token_ids=generated,
                            output_text=str(text),
                            eos_reached=state["eos_step"] is not None,
                            eos_step=state["eos_step"],
                            generation_limit_reached=(
                                state["eos_step"] is None and len(generated) == limit
                            ),
                            maximum_repeated_bigram_share=(
                                _maximum_repeated_bigram_share(generated)
                            ),
                            parent_maximum_repeated_bigram_share=parent_share,
                            longest_identical_token_run=(
                                _longest_identical_token_run(generated)
                            ),
                            parent_longest_identical_token_run=(
                                _longest_identical_token_run(parent_output_ids)
                            ),
                            task_relevant=baseline.task_score(case, str(text)),
                            task_correct=baseline.correctness_score(
                                case,
                                str(text),
                            ),
                            parent_task_relevant=baseline.task_score(
                                case,
                                case.parent_output,
                            ),
                            parent_task_correct=baseline.correctness_score(
                                case,
                                case.parent_output,
                            ),
                        )
                    )
    finally:
        child.train(was_training)
    by_case = {rollout.case_id: rollout for rollout in result}
    return tuple(by_case[case_id] for case_id in wanted)


def _bounded_relative_path_damage(parent: Any, child: Any) -> dict[str, float]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for cached causal evidence.") from exc
    left = (
        parent.detach()
        .to(device="cpu", dtype=torch.float64)
        .reshape(
            int(parent.shape[0]),
            -1,
        )
    )
    right = (
        child.detach()
        .to(device="cpu", dtype=torch.float64)
        .reshape(
            int(child.shape[0]),
            -1,
        )
    )
    if left.shape != right.shape or not bool(
        torch.isfinite(left).all() and torch.isfinite(right).all()
    ):
        raise Lfm2CachedCausalEvidenceError(
            "Parent/child causal path tensors are unaligned or non-finite."
        )
    difference_norm = torch.linalg.vector_norm(left - right, dim=1)
    parent_norm = torch.linalg.vector_norm(left, dim=1)
    child_norm = torch.linalg.vector_norm(right, dim=1)
    symmetric_relative = difference_norm / (parent_norm + child_norm).clamp_min(1e-12)
    symmetric_relative = symmetric_relative.clamp(0.0, 1.0)
    cosine = torch.nn.functional.cosine_similarity(left, right, dim=1)
    cosine_damage = ((1.0 - cosine) / 2.0).clamp(0.0, 1.0)
    combined = 0.65 * symmetric_relative + 0.35 * cosine_damage
    return {
        "maximum_damage": float(combined.max().item()),
        "mean_damage": float(combined.mean().item()),
        "maximum_symmetric_relative_l2": float(symmetric_relative.max().item()),
        "mean_cosine": float(cosine.mean().item()),
    }


def _target_margin(logits: Any, target_token_id: int) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for cached causal evidence.") from exc
    values = logits.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    target = int(target_token_id)
    if not 0 <= target < int(values.shape[0]):
        raise Lfm2CachedCausalEvidenceError(
            "A sealed target token is outside the child vocabulary."
        )
    others = values.clone()
    others[target] = -torch.inf
    return values[target] - others.max()


def _eos_probability(logits: Any, eos_token_ids: Sequence[int]) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for cached causal evidence.") from exc
    values = logits.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    eos = tuple(
        sorted(
            {
                int(value)
                for value in eos_token_ids
                if 0 <= int(value) < int(values.shape[0])
            }
        )
    )
    if not eos:
        return torch.tensor(0.0, dtype=torch.float64)
    eos_index = torch.tensor(eos, dtype=torch.long)
    eos_logit = torch.logsumexp(values.index_select(0, eos_index), dim=0)
    non_eos = values.clone()
    non_eos.index_fill_(0, eos_index, -torch.inf)
    non_eos_logit = torch.logsumexp(non_eos, dim=0)
    return torch.sigmoid(eos_logit - non_eos_logit)


def _path_logits(
    cache: Any,
    descriptor: Mapping[str, Any],
    *,
    child: bool,
) -> Any:
    match = _PATH_PATTERN.fullmatch(str(descriptor["path"]))
    if match is None:
        raise Lfm2CachedCausalEvidenceError(
            "Validation descriptor has an unsupported path."
        )
    batch_index = int(match.group("batch"))
    phase = str(match.group("phase"))
    row_index = int(match.group("row"))
    tensors = cache.load_batch(batch_index).tensors
    if phase == "prefill":
        key = "prefill.logits"
        if key not in tensors:
            raise Lfm2CachedCausalEvidenceError(
                f"Candidate evidence is missing {key!r}."
            )
        logits = tensors[key]
        if not 0 <= row_index < int(logits.shape[1]):
            raise Lfm2CachedCausalEvidenceError(
                "Prefill descriptor/logit row is unaligned."
            )
        return logits[0, row_index, :]
    canonical_key = f"{phase}.full_prefix_logits"
    key = f"{phase}.logits" if child or canonical_key not in tensors else canonical_key
    if key not in tensors:
        raise Lfm2CachedCausalEvidenceError(f"Candidate evidence is missing {key!r}.")
    logits = tensors[key]
    if row_index != 0 or int(logits.shape[1]) != 1:
        raise Lfm2CachedCausalEvidenceError("Decode descriptor/logit row is unaligned.")
    return logits[0, 0, :]


@dataclass(frozen=True, slots=True)
class Lfm2CachedCausalMeasurement:
    damage: Lfm2CausalDamage
    child_trace_rows_sha256: str
    output_event_sha256: str
    validation_case_ids_sha256: str
    measurement_case_count: int
    report: Mapping[str, Any]


def verify_lfm2_full_prefix_child_trace_cache(child_cache: Any) -> Any:
    """Reject any candidate cache containing a recurrent cached forward."""

    child_capture = child_cache.manifest.get("capture", {})
    if (
        child_capture.get("decode_trace_mode") != FULL_PREFIX_CANONICAL_TRACE_MODE
        or child_capture.get("cached_logits_accepted") is not False
        or child_capture.get("forced_decode_use_cache") is not False
    ):
        raise Lfm2CachedCausalEvidenceError(
            "Candidate damage requires uncached full-prefix child traces."
        )
    for batch in child_cache.iter_batches():
        batch_capture = batch.record.get("capture")
        if (
            not isinstance(batch_capture, Mapping)
            or batch_capture.get("cached_forward_count") != 0
        ):
            raise Lfm2CachedCausalEvidenceError(
                "Candidate trace batch used a recurrent cached forward."
            )
    return child_cache


def measure_lfm2_cached_causal_damage(
    parent_cache: Any,
    child_cache: Any,
    parent_rows: Any,
    child_rows: Any,
    descriptors: Sequence[Mapping[str, Any]],
    split: Lfm2CachedCausalSplit,
    rollouts: Sequence[Lfm2FullPrefixRollout],
    *,
    eos_token_ids: Sequence[int],
    parent_recovery_rows: Sequence[Mapping[str, Any]],
    option_id: str,
    proposal_layer_index: int,
    generation_limit: int,
) -> Lfm2CachedCausalMeasurement:
    """Measure six downstream channels on validation evidence only."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for cached causal evidence.") from exc
    _validate_split(split)
    if parent_rows.trace_rows_sha256 != split.parent_trace_rows_sha256:
        raise Lfm2CachedCausalEvidenceError(
            "The live parent rows differ from the sealed split."
        )
    parent_rows.assert_aligned(child_rows)
    verify_lfm2_full_prefix_child_trace_cache(child_cache)
    if (
        isinstance(proposal_layer_index, bool)
        or not isinstance(proposal_layer_index, int)
        or not 0 <= proposal_layer_index < LFM2_VARIABLE_WIDTH_LAYER_COUNT
    ):
        raise Lfm2CachedCausalEvidenceError(
            "proposal_layer_index must be in the fourteen-layer range."
        )
    descriptor_values = tuple(descriptors)
    if len(descriptor_values) != split.row_count:
        raise Lfm2CachedCausalEvidenceError(
            "Descriptor count differs from the sealed causal split."
        )
    eos_ids = tuple(sorted({int(value) for value in eos_token_ids}))
    eos_set = set(eos_ids)
    rollout_limit = _positive_int(generation_limit, "generation_limit")
    margin_damages = []
    eos_logit_damages = []
    predicted_rows = []
    for path_index in split.validation_indices:
        descriptor = descriptor_values[path_index]
        sealed_target = descriptor.get("predicted_token_id")
        if sealed_target is None:
            continue
        parent_logits = _path_logits(parent_cache, descriptor, child=False)
        child_logits = _path_logits(child_cache, descriptor, child=True)
        reference = int(parent_logits.argmax(dim=-1).item())
        parent_margin = _target_margin(parent_logits, reference)
        child_margin = _target_margin(child_logits, reference)
        parent_confidence = torch.sigmoid(parent_margin)
        child_confidence = torch.sigmoid(child_margin)
        margin_damage = torch.clamp(
            (parent_confidence - child_confidence) / parent_confidence.clamp_min(1e-12),
            min=0.0,
            max=1.0,
        )
        parent_eos = _eos_probability(parent_logits, eos_ids)
        child_eos = _eos_probability(child_logits, eos_ids)
        if int(sealed_target) in eos_set:
            eos_damage = torch.clamp(
                (parent_eos - child_eos) / parent_eos.clamp_min(1e-12),
                min=0.0,
                max=1.0,
            )
        else:
            eos_damage = torch.clamp(
                (child_eos - parent_eos) / (1.0 - parent_eos).clamp_min(1e-12),
                min=0.0,
                max=1.0,
            )
        margin_damages.append(float(margin_damage.item()))
        eos_logit_damages.append(float(eos_damage.item()))
        predicted_rows.append(
            {
                "path_index": path_index,
                "path": descriptor["path"],
                "case_id": descriptor["case_id"],
                "parent_reference_token_id": reference,
                "sealed_predicted_token_id": int(sealed_target),
                "parent_reference_matches_sealed_token": (
                    reference == int(sealed_target)
                ),
                "target_is_eos": int(sealed_target) in eos_set,
                "parent_target_margin": float(parent_margin.item()),
                "child_target_margin": float(child_margin.item()),
                "target_margin_damage": float(margin_damage.item()),
                "parent_eos_probability": float(parent_eos.item()),
                "child_eos_probability": float(child_eos.item()),
                "eos_logit_damage": float(eos_damage.item()),
            }
        )
    if not margin_damages:
        raise Lfm2CachedCausalEvidenceError(
            "Validation split contains no next-token prediction anchors."
        )

    validation = split.validation_indices
    residual_layers = []
    attention_layers = []
    for layer_index, layer_kind in enumerate(parent_rows.layer_kinds):
        if layer_index < proposal_layer_index:
            continue
        parent_residual = _index_rows(
            parent_rows.field(layer_index, "residual.post_layer"),
            validation,
        )
        child_residual = _index_rows(
            child_rows.field(layer_index, "residual.post_layer"),
            validation,
        )
        residual = _bounded_relative_path_damage(
            parent_residual,
            child_residual,
        )
        residual_layers.append({"layer_index": layer_index, **residual})
        if layer_kind == "full_attention" and layer_index > proposal_layer_index:
            attention_fields = {}
            for field in (
                "attention.q",
                "attention.k",
                "attention.v",
                "attention.context",
            ):
                attention_fields[field] = _bounded_relative_path_damage(
                    _index_rows(
                        parent_rows.field(layer_index, field),
                        validation,
                    ),
                    _index_rows(
                        child_rows.field(layer_index, field),
                        validation,
                    ),
                )
            operator = _bounded_relative_path_damage(
                _index_rows(
                    parent_rows.field(layer_index, "residual.post_operator"),
                    validation,
                ),
                _index_rows(
                    child_rows.field(layer_index, "residual.post_operator"),
                    validation,
                ),
            )
            attention_layers.append(
                {
                    "layer_index": layer_index,
                    "fields": attention_fields,
                    "post_operator": operator,
                    "maximum_damage": max(
                        *(
                            value["maximum_damage"]
                            for value in attention_fields.values()
                        ),
                        operator["maximum_damage"],
                    ),
                }
            )

    rollout_values = tuple(rollouts)
    if tuple(value.case_id for value in rollout_values) != split.validation_case_ids:
        raise Lfm2CachedCausalEvidenceError(
            "Full-prefix rollout cases differ from the validation split."
        )
    parent_row_map = {str(row["id"]): row for row in parent_recovery_rows}
    repetition_excesses = []
    repetition_damages = []
    relevance_regressions = []
    correctness_regressions = []
    eos_rollout_damages = []
    rollout_reports = []
    for rollout in rollout_values:
        parent_row = parent_row_map[rollout.case_id]
        parent_tokens = tuple(int(value) for value in parent_row["parent_output_ids"])
        parent_eos_step = next(
            (index for index, token in enumerate(parent_tokens) if token in eos_set),
            None,
        )
        parent_eos_reached = parent_eos_step is not None
        parent_limit_reached = (
            not parent_eos_reached and len(parent_tokens) >= rollout_limit
        )
        if parent_eos_reached != rollout.eos_reached:
            eos_rollout_damage = 1.0
        elif parent_eos_reached and rollout.eos_step is not None:
            denominator = max(1, len(parent_tokens))
            eos_rollout_damage = min(
                1.0,
                abs(int(rollout.eos_step) - int(parent_eos_step)) / denominator,
            )
        else:
            eos_rollout_damage = 0.0
        if rollout.generation_limit_reached and not parent_limit_reached:
            eos_rollout_damage = 1.0
        repetition_excess = max(
            0.0,
            rollout.maximum_repeated_bigram_share
            - rollout.parent_maximum_repeated_bigram_share,
        )
        new_degenerate = bool(
            (
                len(rollout.token_ids) >= 8
                and rollout.maximum_repeated_bigram_share >= 0.35
                and rollout.parent_maximum_repeated_bigram_share < 0.35
            )
            or (
                rollout.longest_identical_token_run >= 4
                and rollout.parent_longest_identical_token_run < 4
            )
        )
        repetition_damage = (
            1.0
            if new_degenerate
            else min(
                1.0,
                repetition_excess
                / max(
                    1e-12,
                    1.0 - rollout.parent_maximum_repeated_bigram_share,
                ),
            )
        )
        relevance_regression = bool(
            rollout.parent_task_relevant is True and rollout.task_relevant is not True
        )
        correctness_regression = bool(
            rollout.parent_task_correct is True and rollout.task_correct is not True
        )
        repetition_excesses.append(repetition_excess)
        repetition_damages.append(repetition_damage)
        relevance_regressions.append(relevance_regression)
        correctness_regressions.append(correctness_regression)
        eos_rollout_damages.append(eos_rollout_damage)
        rollout_reports.append(
            {
                **rollout.as_dict(),
                "parent_eos_reached": parent_eos_reached,
                "parent_eos_step": parent_eos_step,
                "parent_generation_limit_reached": parent_limit_reached,
                "eos_rollout_damage": eos_rollout_damage,
                "repetition_excess": repetition_excess,
                "new_degenerate_behavior": new_degenerate,
                "repetition_damage": repetition_damage,
                "task_relevance_regression": relevance_regression,
                "task_correctness_regression": correctness_regression,
            }
        )
    case_count = len(rollout_values)
    relevance_fraction = sum(relevance_regressions) / case_count
    correctness_scored = sum(
        value.parent_task_correct is True for value in rollout_values
    )
    correctness_fraction = (
        0.0
        if correctness_scored == 0
        else sum(correctness_regressions) / correctness_scored
    )
    task_damage = max(relevance_fraction, correctness_fraction)
    damage = Lfm2CausalDamage(
        final_logit_margin=max(margin_damages),
        eos=max(max(eos_logit_damages), max(eos_rollout_damages)),
        repetition=max(repetition_damages, default=0.0),
        task_anchor=task_damage,
        residual_path=max(row["maximum_damage"] for row in residual_layers),
        attention_path=max(
            (row["maximum_damage"] for row in attention_layers),
            default=0.0,
        ),
    )
    report_body = {
        "schema": MEASUREMENT_SCHEMA,
        "option_id": str(option_id),
        "split_sha256": split.split_sha256,
        "validation_rows_sha256": split.validation_rows_sha256,
        "validation_case_ids": list(split.validation_case_ids),
        "validation_case_ids_sha256": split.validation_case_ids_sha256,
        "measurement_case_count": case_count,
        "child_trace_rows_sha256": _digest(
            child_rows.trace_rows_sha256,
            "child_trace_rows_sha256",
        ),
        "child_trace_mode": FULL_PREFIX_CANONICAL_TRACE_MODE,
        "damage": damage.as_dict(),
        "channels": {
            "final_logit_margin": {
                "maximum_damage": max(margin_damages),
                "mean_damage": sum(margin_damages) / len(margin_damages),
                "predicted_row_count": len(margin_damages),
            },
            "eos": {
                "maximum_logit_damage": max(eos_logit_damages),
                "mean_logit_damage": (sum(eos_logit_damages) / len(eos_logit_damages)),
                "maximum_rollout_damage": max(eos_rollout_damages),
            },
            "repetition": {
                "maximum_damage": max(
                    repetition_damages,
                    default=0.0,
                ),
                "maximum_excess_bigram_share": max(
                    repetition_excesses,
                    default=0.0,
                ),
                "mean_excess_bigram_share": (sum(repetition_excesses) / case_count),
            },
            "task_anchor": {
                "relevance_regression_fraction": relevance_fraction,
                "correctness_regression_fraction": correctness_fraction,
                "correctness_scored_case_count": correctness_scored,
            },
            "residual_path": {
                "maximum_damage": damage.residual_path,
                "layers": residual_layers,
            },
            "attention_path": {
                "maximum_damage": damage.attention_path,
                "layers": attention_layers,
            },
        },
        "predicted_rows": predicted_rows,
        "full_prefix_rollouts": rollout_reports,
        "parent_forward_calls": 0,
        "parent_generation_calls": 0,
        "parent_cache_build_calls": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "exact_token_match_objective_used": False,
    }
    event_sha = _json_sha256(report_body)
    report = {**report_body, "output_event_sha256": event_sha}
    return Lfm2CachedCausalMeasurement(
        damage=damage,
        child_trace_rows_sha256=child_rows.trace_rows_sha256,
        output_event_sha256=event_sha,
        validation_case_ids_sha256=split.validation_case_ids_sha256,
        measurement_case_count=case_count,
        report=report,
    )


def candidate_from_cached_measurement(
    proposal: Any,
    measurement: Lfm2CachedCausalMeasurement,
) -> Lfm2CausalRemovalCandidate:
    if not isinstance(measurement, Lfm2CachedCausalMeasurement):
        raise Lfm2CachedCausalEvidenceError(
            "measurement must be Lfm2CachedCausalMeasurement."
        )
    return Lfm2CausalRemovalCandidate.from_ffn_proposal(
        proposal,
        damage=measurement.damage,
        measured_child_trace_sha256=(measurement.child_trace_rows_sha256),
        output_event_sha256=measurement.output_event_sha256,
        validation_case_ids_sha256=(measurement.validation_case_ids_sha256),
        measurement_case_count=measurement.measurement_case_count,
    )


def _candidate_from_dict(value: Mapping[str, Any]) -> Lfm2CausalRemovalCandidate:
    damage = value.get("damage")
    if not isinstance(damage, Mapping):
        raise Lfm2CachedCausalEvidenceError("Frontier candidate damage is missing.")
    return Lfm2CausalRemovalCandidate(
        schema=str(value.get("schema")),
        option_id=str(value.get("option_id")),
        site_kind=value.get("site_kind"),
        layer_index=value.get("layer_index"),
        source_units=value.get("source_units"),
        target_units=value.get("target_units"),
        parameters_saved=value.get("parameters_saved"),
        damage=Lfm2CausalDamage(
            final_logit_margin=damage.get("final_logit_margin"),
            eos=damage.get("eos"),
            repetition=damage.get("repetition"),
            task_anchor=damage.get("task_anchor"),
            residual_path=damage.get("residual_path"),
            attention_path=damage.get("attention_path"),
        ),
        measured_child_trace_sha256=value.get("measured_child_trace_sha256"),
        output_event_sha256=value.get("output_event_sha256"),
        validation_case_ids_sha256=value.get("validation_case_ids_sha256"),
        measurement_case_count=value.get("measurement_case_count"),
        proposal_sha256=value.get("proposal_sha256"),
        layer_binding_sha256=value.get("layer_binding_sha256"),
        retained_source_units=tuple(value.get("retained_source_units", ())),
        child_trace_mode=str(value.get("child_trace_mode")),
    )


def _split_from_dict(value: Mapping[str, Any]) -> Lfm2CachedCausalSplit:
    split = Lfm2CachedCausalSplit(
        schema=str(value.get("schema")),
        profile_name=str(value.get("profile_name")),
        profile_sha256=value.get("profile_sha256"),
        parent_trace_rows_sha256=value.get("parent_trace_rows_sha256"),
        row_count=value.get("row_count"),
        selection_indices=tuple(value.get("selection_indices", ())),
        fit_indices=tuple(value.get("fit_indices", ())),
        validation_indices=tuple(value.get("validation_indices", ())),
        selection_rows_sha256=value.get("selection_rows_sha256"),
        fit_rows_sha256=value.get("fit_rows_sha256"),
        validation_rows_sha256=value.get("validation_rows_sha256"),
        validation_case_ids=tuple(value.get("validation_case_ids", ())),
        validation_case_ids_sha256=value.get("validation_case_ids_sha256"),
        split_sha256=value.get("split_sha256"),
    )
    return _validate_split(split)


def _measurement_report_body(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("output_event_sha256", None)
    return body


def seal_lfm2_cached_causal_candidate_record(
    proposal: Any,
    candidate: Lfm2CausalRemovalCandidate,
    measurement_report: Mapping[str, Any],
    split: Lfm2CachedCausalSplit,
) -> dict[str, Any]:
    """Seal one resumable candidate without serializing tensor payloads."""

    _validate_split(split)
    report = dict(measurement_report)
    event_sha = _digest(
        report.get("output_event_sha256"),
        "measurement.output_event_sha256",
    )
    if (
        _json_sha256(_measurement_report_body(report)) != event_sha
        or candidate.output_event_sha256 != event_sha
        or candidate.proposal_sha256 != getattr(proposal, "proposal_sha256", None)
        or candidate.option_id != getattr(proposal, "option_id", None)
        or candidate.validation_case_ids_sha256 != split.validation_case_ids_sha256
    ):
        raise Lfm2CachedCausalEvidenceError(
            "Candidate record inputs are stale or mismatched."
        )
    body = {
        "schema": "neural-forge-lfm2-cached-causal-candidate-record-v1",
        "proposal": proposal.as_dict(),
        "candidate": candidate.as_dict(),
        "measurement": report,
        "split_sha256": split.split_sha256,
        "parent_forward_calls": 0,
        "parent_generation_calls": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    return {**body, "candidate_record_sha256": _json_sha256(body)}


def load_lfm2_cached_causal_candidate_record(
    document: Mapping[str, Any],
    *,
    expected_proposal: Any,
    split: Lfm2CachedCausalSplit,
) -> tuple[Lfm2CausalRemovalCandidate, Mapping[str, Any]]:
    """Authenticate and restore one resumable candidate result."""

    _validate_split(split)
    body = dict(document)
    seal = body.pop("candidate_record_sha256", None)
    if (
        body.get("schema") != "neural-forge-lfm2-cached-causal-candidate-record-v1"
        or _digest(seal, "candidate_record_sha256") != _json_sha256(body)
        or body.get("split_sha256") != split.split_sha256
    ):
        raise Lfm2CachedCausalEvidenceError(
            "Cached candidate record seal or split is invalid."
        )
    proposal = body.get("proposal")
    candidate_raw = body.get("candidate")
    report = body.get("measurement")
    if (
        not isinstance(proposal, Mapping)
        or not isinstance(candidate_raw, Mapping)
        or not isinstance(report, Mapping)
        or proposal != expected_proposal.as_dict()
    ):
        raise Lfm2CachedCausalEvidenceError(
            "Cached candidate record binds another physical proposal."
        )
    candidate = _candidate_from_dict(candidate_raw)
    event_sha = _digest(
        report.get("output_event_sha256"),
        "measurement.output_event_sha256",
    )
    if (
        candidate.option_id != expected_proposal.option_id
        or candidate.proposal_sha256 != expected_proposal.proposal_sha256
        or candidate.output_event_sha256 != event_sha
        or candidate.validation_case_ids_sha256 != split.validation_case_ids_sha256
        or _json_sha256(_measurement_report_body(report)) != event_sha
    ):
        raise Lfm2CachedCausalEvidenceError(
            "Cached candidate measurement changed after sealing."
        )
    return candidate, report


def seal_lfm2_cached_causal_frontier_document(
    evidence: Lfm2GlobalCausalEvidence,
    split: Lfm2CachedCausalSplit,
    widths_by_layer: Sequence[Sequence[int]],
    measurement_reports: Sequence[Mapping[str, Any]],
    exact_plan: Lfm2GlobalCausalPlan,
    *,
    parent_output_file_sha256: str,
    tokenizer_fingerprint: str,
    task_anchor_manifest: Mapping[str, Any],
    fifty_percent_infeasible_reason: str,
) -> dict[str, Any]:
    """Serialize replay instructions plus measured candidate evidence."""

    _validate_split(split)
    reports = tuple(dict(value) for value in measurement_reports)
    by_option = {candidate.option_id: candidate for candidate in evidence.candidates}
    if (
        len(reports) != len(evidence.candidates)
        or len({report.get("option_id") for report in reports}) != len(reports)
        or set(by_option) != {report.get("option_id") for report in reports}
    ):
        raise Lfm2CachedCausalEvidenceError(
            "Every candidate needs exactly one measurement report."
        )
    for report in reports:
        option_id = str(report["option_id"])
        event_sha = _digest(
            report.get("output_event_sha256"),
            "measurement.output_event_sha256",
        )
        if (
            _json_sha256(_measurement_report_body(report)) != event_sha
            or by_option[option_id].output_event_sha256 != event_sha
            or report.get("split_sha256") != split.split_sha256
        ):
            raise Lfm2CachedCausalEvidenceError(
                "A candidate measurement report is stale or mismatched."
            )
    if (
        exact_plan.evidence_sha256 != evidence.evidence_sha256
        or exact_plan.budget.requested_fraction != 0.35
        or exact_plan.stages[-1].cumulative_parameters_saved
        != exact_plan.budget.parameters_saved
    ):
        raise Lfm2CachedCausalEvidenceError(
            "The frontier document requires an exact authenticated 35% plan."
        )
    reason = str(fifty_percent_infeasible_reason).strip()
    if not reason:
        raise Lfm2CachedCausalEvidenceError(
            "The 50% infeasibility proof needs a concrete reason."
        )
    anchor_manifest = dict(task_anchor_manifest)
    if _json_sha256(anchor_manifest) != evidence.task_anchor_manifest_sha256:
        raise Lfm2CachedCausalEvidenceError(
            "Task-anchor manifest differs from the global evidence binding."
        )
    body = {
        "schema": FRONTIER_SCHEMA,
        "parent_binding": {
            "parent_identity_sha256": evidence.parent_identity_sha256,
            "parent_unique_parameters": evidence.parent_unique_parameters,
            "parent_trace_cache_key": evidence.parent_trace_cache_key,
            "parent_corpus_sha256": evidence.parent_corpus_sha256,
            "parent_trace_rows_sha256": split.parent_trace_rows_sha256,
            "parent_output_envelope_sha256": (evidence.parent_output_envelope_sha256),
            "parent_output_file_sha256": _digest(
                parent_output_file_sha256,
                "parent_output_file_sha256",
            ),
            "task_anchor_manifest_sha256": (evidence.task_anchor_manifest_sha256),
            "tokenizer_fingerprint": _digest(
                tokenizer_fingerprint,
                "tokenizer_fingerprint",
            ),
        },
        "split": split.as_dict(),
        "task_anchor_manifest": anchor_manifest,
        "widths_by_layer": [list(values) for values in widths_by_layer],
        "variable_width_evidence_sha256": (
            evidence.variable_width_evidence.evidence_sha256
        ),
        "global_evidence": evidence.as_dict(),
        "candidates": [candidate.as_dict() for candidate in evidence.candidates],
        "measurement_reports": list(reports),
        "targets": {
            "0.35": {
                "status": "exact_plan_ready",
                "plan_sha256": exact_plan.plan_sha256,
                "budget": exact_plan.budget.as_dict(),
                "stage_count": len(exact_plan.stages),
                "final_target_widths": list(exact_plan.stages[-1].target_widths),
                "selected_option_ids": list(exact_plan.selected_option_ids),
            },
            "0.50": {
                "status": "infeasible",
                "reason": reason,
            },
        },
        "physical_curve_cost_role": (
            "geometry-only payload feasibility; excluded from global selection"
        ),
        "global_selection_objective": (
            "measured final-margin/eos/repetition/task-anchor/residual/attention damage"
        ),
        "parent_cache_build_calls": 0,
        "parent_forward_calls_during_candidate_measurement": 0,
        "parent_response_generation_calls": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "exact_token_match_objective_used": False,
    }
    return {**body, "frontier_sha256": _json_sha256(body)}


def verify_lfm2_cached_causal_frontier_document(
    document: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Verify the portable document seal and every output-event seal."""

    if not isinstance(document, Mapping):
        raise Lfm2CachedCausalEvidenceError("Cached causal frontier must be a mapping.")
    body = dict(document)
    seal = body.pop("frontier_sha256", None)
    if body.get("schema") != FRONTIER_SCHEMA or _digest(
        seal, "frontier_sha256"
    ) != _json_sha256(body):
        raise Lfm2CachedCausalEvidenceError("Cached causal frontier seal is invalid.")
    split_raw = body.get("split")
    if not isinstance(split_raw, Mapping):
        raise Lfm2CachedCausalEvidenceError(
            "Cached causal frontier has no sealed split."
        )
    split = _split_from_dict(split_raw)
    binding = body.get("parent_binding")
    anchor_manifest = body.get("task_anchor_manifest")
    if (
        not isinstance(binding, Mapping)
        or not isinstance(anchor_manifest, Mapping)
        or _json_sha256(dict(anchor_manifest))
        != binding.get("task_anchor_manifest_sha256")
    ):
        raise Lfm2CachedCausalEvidenceError(
            "Cached causal task-anchor manifest is missing or stale."
        )
    candidates = body.get("candidates")
    reports = body.get("measurement_reports")
    if (
        not isinstance(candidates, list)
        or not isinstance(reports, list)
        or len(candidates) != len(reports)
        or not candidates
    ):
        raise Lfm2CachedCausalEvidenceError(
            "Cached causal frontier candidate reports are incomplete."
        )
    candidate_map = {
        str(value.get("option_id")): value
        for value in candidates
        if isinstance(value, Mapping)
    }
    if len(candidate_map) != len(candidates):
        raise Lfm2CachedCausalEvidenceError(
            "Cached causal candidate IDs are malformed or duplicated."
        )
    for report in reports:
        if not isinstance(report, Mapping):
            raise Lfm2CachedCausalEvidenceError(
                "Cached causal measurement report is not a mapping."
            )
        option_id = str(report.get("option_id"))
        event_sha = _digest(
            report.get("output_event_sha256"),
            "measurement.output_event_sha256",
        )
        candidate = candidate_map.get(option_id)
        if (
            candidate is None
            or candidate.get("output_event_sha256") != event_sha
            or report.get("split_sha256") != split.split_sha256
            or _json_sha256(_measurement_report_body(report)) != event_sha
        ):
            raise Lfm2CachedCausalEvidenceError(
                "Cached causal output event is stale or mismatched."
            )
    targets = body.get("targets")
    if (
        not isinstance(targets, Mapping)
        or not isinstance(targets.get("0.35"), Mapping)
        or targets["0.35"].get("status") != "exact_plan_ready"
        or not isinstance(targets.get("0.50"), Mapping)
        or targets["0.50"].get("status") != "infeasible"
    ):
        raise Lfm2CachedCausalEvidenceError(
            "Cached causal frontier lacks the bound 35%/50% target results."
        )
    return document


def replay_lfm2_cached_causal_frontier_document(
    parent: Any,
    parent_cache: Any,
    parent_rows: Any,
    document: Mapping[str, Any],
    *,
    maximum_stage_fraction: float = 0.10,
    max_states: int = 100_000,
) -> tuple[Lfm2GlobalCausalEvidence, Lfm2GlobalCausalPlan]:
    """Rebuild tensors from the sealed cache and authenticate the frontier."""

    verify_lfm2_cached_causal_frontier_document(document)
    binding = document["parent_binding"]
    if (
        parent_cache.cache_key != binding["parent_trace_cache_key"]
        or parent_cache.corpus_fingerprint != binding["parent_corpus_sha256"]
        or parent_rows.trace_rows_sha256 != binding["parent_trace_rows_sha256"]
    ):
        raise Lfm2CachedCausalEvidenceError(
            "Live parent cache differs from the frontier binding."
        )
    split = _split_from_dict(document["split"])
    variable = build_lfm2_causal_physical_source_evidence(
        parent,
        parent_rows,
        split,
        document["widths_by_layer"],
        cache_key=parent_cache.cache_key,
        heldout_corpus_sha256=parent_cache.corpus_fingerprint,
    )
    if variable.evidence_sha256 != document["variable_width_evidence_sha256"]:
        raise Lfm2CachedCausalEvidenceError(
            "Rebuilt physical source evidence differs from the frontier."
        )
    candidates = tuple(_candidate_from_dict(value) for value in document["candidates"])
    global_record = document["global_evidence"]
    weights = global_record["damage_weights"]
    cluster = global_record["cluster"]
    evidence = seal_lfm2_global_causal_evidence(
        parent,
        variable,
        candidates,
        parent_output_envelope_sha256=(binding["parent_output_envelope_sha256"]),
        task_anchor_manifest_sha256=(binding["task_anchor_manifest_sha256"]),
        damage_weights=Lfm2CausalDamageWeights(
            final_logit_margin=weights["final_logit_margin"],
            eos=weights["eos"],
            repetition=weights["repetition"],
            task_anchor=weights["task_anchor"],
            residual_path=weights["residual_path"],
            attention_path=weights["attention_path"],
        ),
        cluster_ridge=cluster["ridge"],
        cluster_max_iterations=cluster["max_iterations"],
        cluster_distance_chunk_size=cluster["distance_chunk_size"],
        parent_cache_build_count=0,
    )
    if evidence.evidence_sha256 != global_record["evidence_sha256"]:
        raise Lfm2CachedCausalEvidenceError(
            "Replayed global evidence differs from the frontier seal."
        )
    plan = plan_lfm2_global_causal_target(
        parent,
        evidence,
        target_fraction=0.35,
        maximum_stage_fraction=maximum_stage_fraction,
        max_states=max_states,
    )
    if plan.plan_sha256 != document["targets"]["0.35"]["plan_sha256"]:
        raise Lfm2CachedCausalEvidenceError(
            "Replayed exact 35% plan differs from the frontier."
        )
    try:
        plan_lfm2_global_causal_target(
            parent,
            evidence,
            target_fraction=0.50,
            maximum_stage_fraction=maximum_stage_fraction,
            max_states=max_states,
        )
    except Lfm2GlobalCausalInfeasible:
        pass
    else:  # pragma: no cover - guarded by the current FFN-only ceiling
        raise Lfm2CachedCausalEvidenceError(
            "Frontier claims 50% infeasibility but replay found a plan."
        )
    return evidence, plan


def build_and_seal_lfm2_cached_causal_frontier(
    parent: Any,
    variable_width_evidence: Lfm2VariableWidthEvidence,
    candidates: Sequence[Lfm2CausalRemovalCandidate],
    split: Lfm2CachedCausalSplit,
    widths_by_layer: Sequence[Sequence[int]],
    measurement_reports: Sequence[Mapping[str, Any]],
    *,
    parent_output_envelope_sha256: str,
    parent_output_file_sha256: str,
    task_anchor_manifest_sha256: str,
    task_anchor_manifest: Mapping[str, Any],
    tokenizer_fingerprint: str,
    damage_weights: Lfm2CausalDamageWeights | None = None,
    cluster_ridge: float = 1e-8,
    cluster_max_iterations: int = 16,
    cluster_distance_chunk_size: int = 512,
    maximum_stage_fraction: float = 0.10,
    max_states: int = 100_000,
) -> tuple[Lfm2GlobalCausalEvidence, Lfm2GlobalCausalPlan, dict[str, Any]]:
    """Seal the measured frontier and prove exact 35% / infeasible 50%."""

    evidence = seal_lfm2_global_causal_evidence(
        parent,
        variable_width_evidence,
        candidates,
        parent_output_envelope_sha256=parent_output_envelope_sha256,
        task_anchor_manifest_sha256=task_anchor_manifest_sha256,
        damage_weights=damage_weights,
        cluster_ridge=cluster_ridge,
        cluster_max_iterations=cluster_max_iterations,
        cluster_distance_chunk_size=cluster_distance_chunk_size,
        parent_cache_build_count=0,
    )
    exact = plan_lfm2_global_causal_target(
        parent,
        evidence,
        target_fraction=0.35,
        maximum_stage_fraction=maximum_stage_fraction,
        max_states=max_states,
    )
    try:
        plan_lfm2_global_causal_target(
            parent,
            evidence,
            target_fraction=0.50,
            maximum_stage_fraction=maximum_stage_fraction,
            max_states=max_states,
        )
    except Lfm2GlobalCausalInfeasible as exc:
        reason = str(exc)
    else:
        raise Lfm2GlobalCausalEvidenceError(
            "The FFN-only frontier unexpectedly authorized a 50% target."
        )
    document = seal_lfm2_cached_causal_frontier_document(
        evidence,
        split,
        widths_by_layer,
        measurement_reports,
        exact,
        parent_output_file_sha256=parent_output_file_sha256,
        tokenizer_fingerprint=tokenizer_fingerprint,
        task_anchor_manifest=task_anchor_manifest,
        fifty_percent_infeasible_reason=reason,
    )
    return evidence, exact, document


__all__ = [
    "DEFAULT_COARSE_WIDTHS",
    "FRONTIER_SCHEMA",
    "Lfm2CachedCausalEvidenceError",
    "Lfm2CachedCausalMeasurement",
    "Lfm2CachedCausalSplit",
    "Lfm2FullPrefixRollout",
    "build_and_seal_lfm2_cached_causal_frontier",
    "build_lfm2_cached_causal_split",
    "build_lfm2_causal_physical_source_evidence",
    "candidate_from_cached_measurement",
    "generate_lfm2_full_prefix_rollouts",
    "lfm2_large_target_width_frontier",
    "measure_lfm2_cached_causal_damage",
    "replay_lfm2_cached_causal_frontier_document",
    "load_lfm2_cached_causal_candidate_record",
    "seal_lfm2_cached_causal_candidate_record",
    "seal_lfm2_cached_causal_frontier_document",
    "verify_lfm2_cached_causal_frontier_document",
    "verify_lfm2_full_prefix_child_trace_cache",
]
