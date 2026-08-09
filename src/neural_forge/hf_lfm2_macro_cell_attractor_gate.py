"""Child-only free-rollout health gate for LFM2 macro-cell folding.

The gate is deliberately narrower than a semantic evaluator.  It protects a
closed-loop fold from introducing obvious autoregressive attractor collapse:
entropy collapse, premature/late EOS, repeated loops, or one-token dominance.
It consumes only a verified, already-tokenized parent trace cache and runs the
candidate child with ``use_cache=False``.  No parent model or tokenizer is
accepted by this API.

Exact token agreement is retained as a diagnostic.  It is never an acceptance
criterion because a coherent child is allowed to choose a different valid
continuation from its parent.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .dynamic_circuit_ir import canonical_sha256
from .dynamic_fold_executor import fingerprint_model_state
from .hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceBatch,
    Lfm2CircuitTraceCache,
)
from .hf_lfm2_tokenizer_bridge_trace_adapter import (
    Lfm2BridgeAlignedRecord,
    Lfm2StrictTokenizerBridgeTraceAdapter,
)


MACRO_CELL_ATTRACTOR_SNAPSHOT_SCHEMA = (
    "neural-forge-lfm2-macro-cell-attractor-snapshot-v1"
)
MACRO_CELL_ATTRACTOR_DECISION_SCHEMA = (
    "neural-forge-lfm2-macro-cell-attractor-decision-v1"
)
_SELECTION_SCHEMA = "neural-forge-lfm2-macro-cell-attractor-selection-v2"
_CAPTURE_AUDIT_SCHEMA = (
    "neural-forge-lfm2-macro-cell-attractor-capture-audit-v1"
)
_DECISION_AUDIT_SCHEMA = (
    "neural-forge-lfm2-macro-cell-attractor-decision-audit-v1"
)
TARGET_TOKEN_FREE_ROLLOUT_SNAPSHOT_SCHEMA = (
    "neural-forge-lfm2-target-token-free-rollout-snapshot-v1"
)
_TARGET_TOKEN_FREE_ROLLOUT_SELECTION_SCHEMA = (
    "neural-forge-lfm2-target-token-free-rollout-selection-v1"
)
_TARGET_TOKEN_FREE_ROLLOUT_CONFIG_SCHEMA = (
    "neural-forge-lfm2-target-token-free-rollout-config-v1"
)
_TARGET_TOKEN_FREE_ROLLOUT_AUDIT_SCHEMA = (
    "neural-forge-lfm2-target-token-free-rollout-audit-v1"
)


class Lfm2MacroCellAttractorGateError(RuntimeError):
    """The sealed trace or child rollout cannot support a sound gate."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise Lfm2MacroCellAttractorGateError(
            "PyTorch is required for macro-cell attractor validation."
        ) from exc
    return torch


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite nonnegative number.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be a finite nonnegative number.")
    return result


def _unit_interval(value: Any, label: str) -> float:
    result = _finite_nonnegative(value, label)
    if result > 1.0:
        raise ValueError(f"{label} must be in [0, 1].")
    return result


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return int(value)


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer.")
    return int(value)


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(
        json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _qualified_type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


@dataclass(frozen=True, slots=True)
class Lfm2MacroCellAttractorGateConfig:
    """Deterministic bounds for one pre-cell versus post-cell comparison."""

    maximum_records: int = 8
    maximum_steps: int | None = None
    split_seed: str = "neural-forge-lfm2-macro-cell-attractor-v1"
    eos_token_id: int | None = None
    maximum_entropy_parent_distance_regression: float = 0.02
    maximum_repetition_excess_regression: float = 0.05
    maximum_dominant_token_excess_regression: float = 0.05
    maximum_eos_position_error_regression: float = 0.0
    collapse_entropy_threshold: float = 0.10
    collapse_repetition_share_threshold: float = 0.80
    collapse_dominant_token_share_threshold: float = 0.80
    maximum_new_collapsed_records: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "maximum_records",
            _positive_integer(self.maximum_records, "maximum_records"),
        )
        if self.maximum_steps is not None:
            object.__setattr__(
                self,
                "maximum_steps",
                _positive_integer(self.maximum_steps, "maximum_steps"),
            )
        if not isinstance(self.split_seed, str) or not self.split_seed.strip():
            raise ValueError("split_seed must be a nonempty string.")
        object.__setattr__(self, "split_seed", self.split_seed.strip())
        if self.eos_token_id is not None:
            object.__setattr__(
                self,
                "eos_token_id",
                _nonnegative_integer(self.eos_token_id, "eos_token_id"),
            )
        for name in (
            "maximum_entropy_parent_distance_regression",
            "maximum_repetition_excess_regression",
            "maximum_dominant_token_excess_regression",
            "maximum_eos_position_error_regression",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name),
            )
        for name in (
            "collapse_entropy_threshold",
            "collapse_repetition_share_threshold",
            "collapse_dominant_token_share_threshold",
        ):
            object.__setattr__(
                self,
                name,
                _unit_interval(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "maximum_new_collapsed_records",
            _nonnegative_integer(
                self.maximum_new_collapsed_records,
                "maximum_new_collapsed_records",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "maximum_records": self.maximum_records,
            "maximum_steps": self.maximum_steps,
            "split_seed": self.split_seed,
            "eos_token_id": self.eos_token_id,
            "maximum_entropy_parent_distance_regression": (
                self.maximum_entropy_parent_distance_regression
            ),
            "maximum_repetition_excess_regression": (
                self.maximum_repetition_excess_regression
            ),
            "maximum_dominant_token_excess_regression": (
                self.maximum_dominant_token_excess_regression
            ),
            "maximum_eos_position_error_regression": (
                self.maximum_eos_position_error_regression
            ),
            "collapse_entropy_threshold": self.collapse_entropy_threshold,
            "collapse_repetition_share_threshold": (
                self.collapse_repetition_share_threshold
            ),
            "collapse_dominant_token_share_threshold": (
                self.collapse_dominant_token_share_threshold
            ),
            "maximum_new_collapsed_records": self.maximum_new_collapsed_records,
        }


@dataclass(frozen=True, slots=True)
class Lfm2MacroCellAttractorSnapshot:
    """Immutable, canonical evidence from one live child rollout."""

    _payload: Mapping[str, Any]

    @property
    def snapshot_sha256(self) -> str:
        return str(self._payload["snapshot_sha256"])

    @property
    def metrics(self) -> dict[str, Any]:
        return _json_copy(self._payload["metrics"])

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(_json_copy(record) for record in self._payload["records"])

    def as_dict(self) -> dict[str, Any]:
        return _json_copy(self._payload)


@dataclass(frozen=True, slots=True)
class Lfm2TargetTokenFreeRolloutConfig:
    """Deterministic capture bounds without any numeric acceptance policy."""

    maximum_records: int = 8
    maximum_steps: int | None = None
    split_seed: str = "neural-forge-lfm2-target-token-free-rollout-v1"
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "maximum_records",
            _positive_integer(self.maximum_records, "maximum_records"),
        )
        if self.maximum_steps is not None:
            object.__setattr__(
                self,
                "maximum_steps",
                _positive_integer(self.maximum_steps, "maximum_steps"),
            )
        if not isinstance(self.split_seed, str) or not self.split_seed.strip():
            raise ValueError("split_seed must be a nonempty string.")
        object.__setattr__(self, "split_seed", self.split_seed.strip())
        if self.eos_token_id is not None:
            object.__setattr__(
                self,
                "eos_token_id",
                _nonnegative_integer(self.eos_token_id, "eos_token_id"),
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "maximum_records": self.maximum_records,
            "maximum_steps": self.maximum_steps,
            "split_seed": self.split_seed,
            "eos_token_id": self.eos_token_id,
        }


@dataclass(frozen=True, slots=True)
class Lfm2TargetTokenFreeRolloutSnapshot:
    """Immutable target-token rollout evidence for fit planning or review."""

    _payload: Mapping[str, Any]

    @property
    def snapshot_sha256(self) -> str:
        return str(self._payload["snapshot_sha256"])

    @property
    def metrics(self) -> dict[str, Any]:
        return _json_copy(self._payload["metrics"])

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(_json_copy(record) for record in self._payload["records"])

    def as_dict(self) -> dict[str, Any]:
        return _json_copy(self._payload)


@dataclass(frozen=True, slots=True)
class Lfm2MacroCellAttractorDecision:
    """Canonical pre-cell versus post-cell health decision."""

    _payload: Mapping[str, Any]

    @property
    def accepted(self) -> bool:
        return bool(self._payload["accepted"])

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self._payload["failed_checks"])

    @property
    def decision_sha256(self) -> str:
        return str(self._payload["decision_sha256"])

    def as_dict(self) -> dict[str, Any]:
        return _json_copy(self._payload)


def _gradient_fingerprint(model: Any) -> str:
    torch = _torch()
    digest = hashlib.sha256()
    parameters = tuple(model.named_parameters())
    if not parameters:
        raise Lfm2MacroCellAttractorGateError(
            "The child must expose at least one named parameter."
        )
    for name, parameter in parameters:
        digest.update(name.encode("utf-8"))
        gradient = parameter.grad
        if gradient is None:
            digest.update(b":none")
            continue
        if (
            not isinstance(gradient, torch.Tensor)
            or gradient.device.type == "meta"
            or gradient.layout != torch.strided
        ):
            raise Lfm2MacroCellAttractorGateError(
                f"Gradient {name!r} is not a materialized dense tensor."
            )
        tensor = gradient.detach().to(device="cpu").contiguous()
        digest.update(str(tuple(int(size) for size in tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _training_flags(model: Any) -> tuple[tuple[Any, bool], ...]:
    modules = tuple(model.modules())
    if not modules:
        raise Lfm2MacroCellAttractorGateError(
            "The child must expose a PyTorch module hierarchy."
        )
    return tuple((module, bool(module.training)) for module in modules)


def _training_flags_sha256(flags: Sequence[tuple[Any, bool]]) -> str:
    return canonical_sha256(
        {
            "schema": "neural-forge-module-training-flags-v1",
            "flags": [
                {
                    "index": index,
                    "module_type": _qualified_type_name(module),
                    "training": training,
                }
                for index, (module, training) in enumerate(flags)
            ],
        }
    )


def _restore_training_flags(flags: Sequence[tuple[Any, bool]]) -> None:
    # Calling root.train(...) would destroy deliberately mixed child-module flags.
    for module, training in flags:
        module.training = training


def _model_input_device(model: Any) -> Any:
    torch = _torch()
    getter = getattr(model, "get_input_embeddings", None)
    if callable(getter):
        embeddings = getter()
        weight = getattr(embeddings, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.device.type != "meta":
            return weight.device
    for value in model.parameters():
        if value.device.type != "meta":
            return value.device
    for value in model.buffers():
        if value.device.type != "meta":
            return value.device
    raise Lfm2MacroCellAttractorGateError(
        "The child has no materialized input device."
    )


def _model_vocabulary_size(model: Any) -> int:
    value = getattr(getattr(model, "config", None), "vocab_size", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 1:
        raise Lfm2MacroCellAttractorGateError(
            "The child config must expose an integer vocab_size greater than one."
        )
    return int(value)


def _resolve_eos_token_id(
    model: Any,
    config: Lfm2MacroCellAttractorGateConfig,
    vocabulary_size: int,
) -> int:
    value = config.eos_token_id
    if value is None:
        value = getattr(getattr(model, "config", None), "eos_token_id", None)
    if isinstance(value, bool) or not isinstance(value, int):
        raise Lfm2MacroCellAttractorGateError(
            "An integer eos_token_id is required in the gate or child config."
        )
    if value < 0 or value >= vocabulary_size:
        raise Lfm2MacroCellAttractorGateError(
            "The child eos_token_id is outside its vocabulary."
        )
    return int(value)


def _extract_logits(output: Any) -> Any:
    logits = output.get("logits") if isinstance(output, Mapping) else None
    if logits is None:
        logits = getattr(output, "logits", None)
    torch = _torch()
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise Lfm2MacroCellAttractorGateError(
            "The child forward must return rank-three logits."
        )
    return logits


def _normalized_entropy(logits: Any) -> float:
    torch = _torch()
    if not isinstance(logits, torch.Tensor) or logits.ndim != 1:
        raise Lfm2MacroCellAttractorGateError(
            "Predictive entropy requires one rank-one logit vector."
        )
    vocabulary_size = int(logits.numel())
    if vocabulary_size <= 1:
        raise Lfm2MacroCellAttractorGateError(
            "Predictive entropy requires at least two vocabulary entries."
        )
    vector = logits.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(vector).all().item()):
        raise Lfm2MacroCellAttractorGateError(
            "A rollout produced non-finite logits."
        )
    log_probabilities = torch.log_softmax(vector, dim=-1)
    probabilities = log_probabilities.exp()
    entropy = float((-(probabilities * log_probabilities).sum()).item())
    return entropy / math.log(vocabulary_size)


def _maximum_repeated_bigram_share(token_ids: Sequence[int]) -> float:
    tokens = tuple(int(token_id) for token_id in token_ids)
    bigrams = tuple(zip(tokens, tokens[1:]))
    if not bigrams:
        return 0.0
    counts: dict[tuple[int, int], int] = {}
    for bigram in bigrams:
        counts[bigram] = counts.get(bigram, 0) + 1
    return max(counts.values()) / len(bigrams)


def _dominant_token_share(token_ids: Sequence[int]) -> float:
    tokens = tuple(int(token_id) for token_id in token_ids)
    if not tokens:
        return 0.0
    counts: dict[int, int] = {}
    for token_id in tokens:
        counts[token_id] = counts.get(token_id, 0) + 1
    return max(counts.values()) / len(tokens)


def _first_eos_position(token_ids: Sequence[int], eos_token_id: int) -> int | None:
    for index, token_id in enumerate(token_ids):
        if int(token_id) == eos_token_id:
            return index
    return None


def _mean(values: Sequence[float], label: str) -> float:
    if not values:
        raise Lfm2MacroCellAttractorGateError(f"{label} cannot be empty.")
    return float(sum(float(value) for value in values) / len(values))


def _sequence_health(
    token_ids: Sequence[int],
    entropies: Sequence[float],
    eos_token_id: int,
) -> dict[str, Any]:
    tokens = tuple(int(value) for value in token_ids)
    entropy_values = tuple(float(value) for value in entropies)
    if not tokens or len(tokens) != len(entropy_values):
        raise Lfm2MacroCellAttractorGateError(
            "A health trajectory must contain one entropy per generated token."
        )
    eos_position = _first_eos_position(tokens, eos_token_id)
    stop = len(tokens) if eos_position is None else eos_position + 1
    effective_tokens = tokens[:stop]
    effective_entropies = entropy_values[:stop]
    return {
        "token_ids": list(tokens),
        "evaluated_token_count": len(effective_tokens),
        "normalized_entropy": _mean(
            effective_entropies,
            "predictive entropies",
        ),
        "maximum_repeated_bigram_share": _maximum_repeated_bigram_share(
            effective_tokens
        ),
        "dominant_token_share": _dominant_token_share(effective_tokens),
        "eos_position": eos_position,
    }


def _eos_position_error(
    reference_position: int | None,
    child_position: int | None,
    horizon: int,
) -> float:
    if reference_position is None and child_position is None:
        return 0.0
    if reference_position is None or child_position is None:
        return 1.0
    return abs(reference_position - child_position) / max(1, horizon)


def _positional_match_fraction(
    reference: Sequence[int],
    child: Sequence[int],
    horizon: int,
) -> float:
    matches = sum(
        1
        for index in range(horizon)
        if index < len(reference)
        and index < len(child)
        and int(reference[index]) == int(child[index])
    )
    return matches / horizon


def _longest_common_prefix_fraction(
    reference: Sequence[int],
    child: Sequence[int],
    horizon: int,
) -> float:
    count = 0
    for index in range(horizon):
        if (
            index >= len(reference)
            or index >= len(child)
            or int(reference[index]) != int(child[index])
        ):
            break
        count += 1
    return count / horizon


def _parent_predictors(
    batch: Lfm2CircuitTraceBatch,
    row_index: int,
    horizon: int,
) -> tuple[list[int], list[float]]:
    torch = _torch()
    predicted: list[int] = []
    entropies: list[float] = []
    for step_index in range(horizon):
        key = (
            "prefill.logits"
            if step_index == 0
            else f"decode.{step_index - 1:03d}.logits"
        )
        logits = batch.tensors.get(key)
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 3
            or row_index >= int(logits.shape[0])
            or int(logits.shape[-1]) <= 1
        ):
            raise Lfm2MacroCellAttractorGateError(
                f"Sealed batch {batch.index} has invalid predictor {key!r}."
            )
        vector = logits[row_index, -1, :]
        predicted.append(int(vector.argmax(dim=-1).item()))
        entropies.append(_normalized_entropy(vector))
    return predicted, entropies


def _child_free_rollout(
    model: Any,
    prompt_ids: Any,
    horizon: int,
    eos_token_id: int,
    vocabulary_size: int,
    input_device: Any,
) -> tuple[list[int], list[float], int]:
    torch = _torch()
    prefix_ids = prompt_ids.to(device=input_device, dtype=torch.long).reshape(1, -1)
    prefix_mask = torch.ones_like(prefix_ids)
    generated: list[int] = []
    entropies: list[float] = []
    forward_calls = 0
    for _ in range(horizon):
        output = model(
            input_ids=prefix_ids,
            attention_mask=prefix_mask,
            use_cache=False,
            return_dict=True,
        )
        forward_calls += 1
        logits = _extract_logits(output)
        if int(logits.shape[0]) != 1 or int(logits.shape[-1]) != vocabulary_size:
            raise Lfm2MacroCellAttractorGateError(
                "The child rollout logits changed batch or vocabulary shape."
            )
        vector = logits[0, -1, :]
        entropies.append(_normalized_entropy(vector))
        next_token = int(vector.argmax(dim=-1).item())
        generated.append(next_token)
        if next_token == eos_token_id:
            break
        token = torch.tensor(
            [[next_token]],
            dtype=torch.long,
            device=input_device,
        )
        prefix_ids = torch.cat((prefix_ids, token), dim=1)
        prefix_mask = torch.cat(
            (
                prefix_mask,
                torch.ones(
                    (1, 1),
                    dtype=prefix_mask.dtype,
                    device=input_device,
                ),
            ),
            dim=1,
        )
    return generated, entropies, forward_calls


def _record_metrics(
    *,
    model: Any,
    batch: Lfm2CircuitTraceBatch,
    row_index: int,
    horizon: int,
    eos_token_id: int,
    vocabulary_size: int,
    input_device: Any,
    config: Lfm2MacroCellAttractorGateConfig,
) -> tuple[dict[str, Any], int]:
    torch = _torch()
    input_ids = batch.input_ids
    attention_mask = batch.attention_mask
    forced_decode_ids = batch.forced_decode_ids
    if (
        not isinstance(input_ids, torch.Tensor)
        or not isinstance(attention_mask, torch.Tensor)
        or not isinstance(forced_decode_ids, torch.Tensor)
        or input_ids.ndim != 2
        or attention_mask.shape != input_ids.shape
        or forced_decode_ids.ndim != 2
        or int(forced_decode_ids.shape[0]) != int(input_ids.shape[0])
    ):
        raise Lfm2MacroCellAttractorGateError(
            f"Sealed batch {batch.index} has invalid token tensors."
        )
    valid = attention_mask[row_index].to(dtype=torch.bool)
    prompt = input_ids[row_index][valid]
    if int(prompt.numel()) <= 0:
        raise Lfm2MacroCellAttractorGateError(
            f"Sealed batch {batch.index} row {row_index} has no prompt tokens."
        )
    if bool(((prompt < 0) | (prompt >= vocabulary_size)).any().item()):
        raise Lfm2MacroCellAttractorGateError(
            "A sealed prompt is outside the child vocabulary."
        )

    parent_tokens, parent_entropies = _parent_predictors(
        batch,
        row_index,
        horizon,
    )
    forced_tokens = [
        int(value)
        for value in forced_decode_ids[row_index, :horizon].detach().cpu().tolist()
    ]
    if parent_tokens != forced_tokens:
        raise Lfm2MacroCellAttractorGateError(
            "The selected sealed trace is not an authenticated parent-greedy "
            "rollout; free-running attractor comparison would be unsound."
        )
    child_tokens, child_entropies, forward_calls = _child_free_rollout(
        model,
        prompt,
        horizon,
        eos_token_id,
        vocabulary_size,
        input_device,
    )
    parent_health = _sequence_health(
        parent_tokens,
        parent_entropies,
        eos_token_id,
    )
    child_health = _sequence_health(
        child_tokens,
        child_entropies,
        eos_token_id,
    )
    entropy_parent_distance = abs(
        child_health["normalized_entropy"]
        - parent_health["normalized_entropy"]
    )
    repetition_excess = max(
        0.0,
        child_health["maximum_repeated_bigram_share"]
        - parent_health["maximum_repeated_bigram_share"],
    )
    dominant_token_excess = max(
        0.0,
        child_health["dominant_token_share"]
        - parent_health["dominant_token_share"],
    )
    eos_error = _eos_position_error(
        parent_health["eos_position"],
        child_health["eos_position"],
        horizon,
    )
    collapsed = (
        child_health["normalized_entropy"]
        <= config.collapse_entropy_threshold
        and (
            child_health["maximum_repeated_bigram_share"]
            >= config.collapse_repetition_share_threshold
            or child_health["dominant_token_share"]
            >= config.collapse_dominant_token_share_threshold
        )
    )
    record_id = f"batch-{batch.index:06d}-row-{row_index:06d}"
    return (
        {
            "record_id": record_id,
            "batch_index": batch.index,
            "row_index": row_index,
            "input_sha256": batch.input_sha256,
            "horizon": horizon,
            "parent": parent_health,
            "child": child_health,
            "relative_to_cached_parent": {
                "entropy_parent_distance": entropy_parent_distance,
                "repetition_excess": repetition_excess,
                "dominant_token_excess": dominant_token_excess,
                "eos_position_error": eos_error,
            },
            "collapsed": collapsed,
            "diagnostics": {
                "forced_token_ids": forced_tokens,
                "sealed_parent_greedy_rollout_authenticated": True,
                "parent_top1_exact_token_match_fraction": (
                    _positional_match_fraction(
                        parent_tokens,
                        child_tokens,
                        horizon,
                    )
                ),
                "parent_top1_longest_common_prefix_fraction": (
                    _longest_common_prefix_fraction(
                        parent_tokens,
                        child_tokens,
                        horizon,
                    )
                ),
                "forced_exact_token_match_fraction": (
                    _positional_match_fraction(
                        forced_tokens,
                        child_tokens,
                        horizon,
                    )
                ),
                "forced_longest_common_prefix_fraction": (
                    _longest_common_prefix_fraction(
                        forced_tokens,
                        child_tokens,
                        horizon,
                    )
                ),
                "exact_token_match_is_diagnostic_only": True,
            },
        },
        forward_calls,
    )


def _aggregate_record_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise Lfm2MacroCellAttractorGateError(
            "Attractor validation selected no child rows."
        )
    relative = [record["relative_to_cached_parent"] for record in records]
    diagnostics = [record["diagnostics"] for record in records]
    return {
        "record_count": len(records),
        "mean_child_normalized_entropy": _mean(
            [record["child"]["normalized_entropy"] for record in records],
            "child normalized entropies",
        ),
        "mean_parent_normalized_entropy": _mean(
            [record["parent"]["normalized_entropy"] for record in records],
            "parent normalized entropies",
        ),
        "mean_entropy_parent_distance": _mean(
            [value["entropy_parent_distance"] for value in relative],
            "entropy-parent distances",
        ),
        "mean_repetition_excess": _mean(
            [value["repetition_excess"] for value in relative],
            "repetition excesses",
        ),
        "mean_dominant_token_excess": _mean(
            [value["dominant_token_excess"] for value in relative],
            "dominant-token excesses",
        ),
        "mean_eos_position_error": _mean(
            [value["eos_position_error"] for value in relative],
            "EOS-position errors",
        ),
        "collapsed_record_count": sum(
            1 for record in records if bool(record["collapsed"])
        ),
        "collapsed_record_fraction": (
            sum(1 for record in records if bool(record["collapsed"])) / len(records)
        ),
        "diagnostic_mean_parent_top1_exact_token_match_fraction": _mean(
            [
                value["parent_top1_exact_token_match_fraction"]
                for value in diagnostics
            ],
            "parent exact-token diagnostics",
        ),
        "diagnostic_mean_parent_top1_longest_common_prefix_fraction": _mean(
            [
                value["parent_top1_longest_common_prefix_fraction"]
                for value in diagnostics
            ],
            "parent LCP diagnostics",
        ),
        "diagnostic_mean_forced_exact_token_match_fraction": _mean(
            [value["forced_exact_token_match_fraction"] for value in diagnostics],
            "forced exact-token diagnostics",
        ),
        "diagnostic_mean_forced_longest_common_prefix_fraction": _mean(
            [
                value["forced_longest_common_prefix_fraction"]
                for value in diagnostics
            ],
            "forced LCP diagnostics",
        ),
    }


class Lfm2MacroCellAttractorGate:
    """Capture and compare deterministic child-only free-rollout health."""

    def __init__(
        self,
        trace_cache: Lfm2CircuitTraceCache,
        *,
        config: Lfm2MacroCellAttractorGateConfig | None = None,
    ) -> None:
        if not isinstance(trace_cache, Lfm2CircuitTraceCache):
            raise TypeError("trace_cache must be Lfm2CircuitTraceCache.")
        selected = (
            Lfm2MacroCellAttractorGateConfig() if config is None else config
        )
        if not isinstance(selected, Lfm2MacroCellAttractorGateConfig):
            raise TypeError(
                "config must be Lfm2MacroCellAttractorGateConfig or None."
            )
        self._trace_cache = trace_cache.verify()
        self._cache_key = self._trace_cache.cache_key
        self._config = selected
        candidates: list[tuple[str, int, int, str, int]] = []
        rejected_non_greedy_records = 0
        for batch in self._trace_cache.iter_batches():
            forced_decode_ids = batch.forced_decode_ids
            torch = _torch()
            if (
                not isinstance(forced_decode_ids, torch.Tensor)
                or forced_decode_ids.ndim != 2
                or int(forced_decode_ids.shape[0]) <= 0
                or int(forced_decode_ids.shape[1]) <= 0
            ):
                raise Lfm2MacroCellAttractorGateError(
                    f"Sealed batch {batch.index} has no valid decode records."
                )
            horizon = int(forced_decode_ids.shape[1])
            if selected.maximum_steps is not None:
                horizon = min(horizon, selected.maximum_steps)
            for row_index in range(int(forced_decode_ids.shape[0])):
                parent_tokens, _ = _parent_predictors(
                    batch,
                    row_index,
                    horizon,
                )
                forced_tokens = [
                    int(value)
                    for value in forced_decode_ids[
                        row_index,
                        :horizon,
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                ]
                if parent_tokens != forced_tokens:
                    rejected_non_greedy_records += 1
                    continue
                rank = canonical_sha256(
                    {
                        "schema": _SELECTION_SCHEMA,
                        "cache_key": self._cache_key,
                        "split_seed": selected.split_seed,
                        "batch_index": batch.index,
                        "row_index": row_index,
                        "input_sha256": batch.input_sha256,
                        "horizon": horizon,
                    }
                )
                candidates.append(
                    (
                        rank,
                        batch.index,
                        row_index,
                        batch.input_sha256,
                        horizon,
                    )
                )
        candidates.sort()
        chosen = tuple(candidates[: selected.maximum_records])
        if not chosen:
            raise Lfm2MacroCellAttractorGateError(
                "The sealed parent trace cache has no authenticated "
                "parent-greedy rollout records."
            )
        self._selected_records = tuple(
            (batch_index, row_index, input_sha256, horizon)
            for _, batch_index, row_index, input_sha256, horizon in chosen
        )
        self._eligible_record_count = len(candidates)
        self._rejected_non_greedy_record_count = rejected_non_greedy_records
        self._selection_sha256 = canonical_sha256(
            {
                "schema": _SELECTION_SCHEMA,
                "cache_key": self._cache_key,
                "split_seed": selected.split_seed,
                "eligible_authenticated_record_count": (
                    self._eligible_record_count
                ),
                "rejected_non_greedy_record_count": (
                    self._rejected_non_greedy_record_count
                ),
                "selected_records": [
                    {
                        "batch_index": batch_index,
                        "row_index": row_index,
                        "input_sha256": input_sha256,
                        "horizon": horizon,
                    }
                    for (
                        batch_index,
                        row_index,
                        input_sha256,
                        horizon,
                    ) in self._selected_records
                ],
            }
        )
        self._config_sha256 = canonical_sha256(
            {
                "schema": (
                    "neural-forge-lfm2-macro-cell-attractor-gate-config-v1"
                ),
                "config": selected.as_dict(),
            }
        )

    @property
    def config(self) -> Lfm2MacroCellAttractorGateConfig:
        return self._config

    @property
    def selection_sha256(self) -> str:
        return self._selection_sha256

    def capture(
        self,
        child_model: Any,
        *,
        phase: str,
    ) -> Lfm2MacroCellAttractorSnapshot:
        """Run a state-preserving child rollout over the sealed trace split."""

        torch = _torch()
        if not isinstance(phase, str) or not phase.strip():
            raise ValueError("phase must be a nonempty string.")
        phase = phase.strip()
        if self._trace_cache.verify().cache_key != self._cache_key:
            raise Lfm2MacroCellAttractorGateError(
                "The trace-cache identity changed after gate construction."
            )
        vocabulary_size = _model_vocabulary_size(child_model)
        eos_token_id = _resolve_eos_token_id(
            child_model,
            self._config,
            vocabulary_size,
        )
        input_device = _model_input_device(child_model)
        state_before = fingerprint_model_state(child_model)
        gradients_before = _gradient_fingerprint(child_model)
        flags_before = _training_flags(child_model)
        flags_sha256_before = _training_flags_sha256(flags_before)

        records: list[dict[str, Any]] = []
        child_forward_calls = 0
        failure: Exception | None = None
        try:
            child_model.eval()
            with torch.inference_mode():
                for (
                    batch_index,
                    row_index,
                    expected_input_sha256,
                    horizon,
                ) in self._selected_records:
                    batch = self._trace_cache.load_batch(batch_index)
                    if batch.input_sha256 != expected_input_sha256:
                        raise Lfm2MacroCellAttractorGateError(
                            "A selected sealed trace changed after binding."
                        )
                    forced_decode_ids = batch.forced_decode_ids
                    if (
                        not isinstance(forced_decode_ids, torch.Tensor)
                        or forced_decode_ids.ndim != 2
                        or int(forced_decode_ids.shape[1]) <= 0
                    ):
                        raise Lfm2MacroCellAttractorGateError(
                            f"Sealed batch {batch.index} has no decode horizon."
                        )
                    if (
                        row_index >= int(forced_decode_ids.shape[0])
                        or horizon > int(forced_decode_ids.shape[1])
                    ):
                        raise Lfm2MacroCellAttractorGateError(
                            "A selected sealed rollout record changed shape "
                            "after binding."
                        )
                    record, forward_calls = _record_metrics(
                        model=child_model,
                        batch=batch,
                        row_index=row_index,
                        horizon=horizon,
                        eos_token_id=eos_token_id,
                        vocabulary_size=vocabulary_size,
                        input_device=input_device,
                        config=self._config,
                    )
                    records.append(record)
                    child_forward_calls += forward_calls
        except Exception as exc:  # state checks still run before surfacing failure
            failure = exc
        finally:
            _restore_training_flags(flags_before)

        state_after = fingerprint_model_state(child_model)
        gradients_after = _gradient_fingerprint(child_model)
        flags_after = _training_flags(child_model)
        flags_sha256_after = _training_flags_sha256(flags_after)
        state_preserved = (
            state_before == state_after
            and gradients_before == gradients_after
            and flags_sha256_before == flags_sha256_after
        )
        if not state_preserved:
            raise Lfm2MacroCellAttractorGateError(
                "Attractor validation changed child state, gradients, or "
                "module training flags."
            ) from failure
        if failure is not None:
            if isinstance(failure, Lfm2MacroCellAttractorGateError):
                raise failure
            raise Lfm2MacroCellAttractorGateError(
                "The child free rollout failed."
            ) from failure

        metrics = _aggregate_record_metrics(records)
        audit_body = {
            "schema": _CAPTURE_AUDIT_SCHEMA,
            "parent_forward_calls": 0,
            "parent_trace_regeneration_calls": 0,
            "child_forward_calls": child_forward_calls,
            "tokenizer_accessed": False,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "quantization_used": False,
            "pruning_used": False,
            "inference_mode": True,
            "child_use_cache": False,
            "child_rollout_mode": "full_prefix_greedy_argmax",
            "sealed_parent_greedy_rollouts_authenticated": True,
            "exact_token_match_is_diagnostic_only": True,
            "exact_token_metrics_used_for_acceptance": False,
            "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after,
            "gradient_sha256_before": gradients_before,
            "gradient_sha256_after": gradients_after,
            "training_flags_sha256_before": flags_sha256_before,
            "training_flags_sha256_after": flags_sha256_after,
            "state_preserved": True,
        }
        audit = {
            **audit_body,
            "audit_sha256": canonical_sha256(audit_body),
        }
        snapshot_body = {
            "schema": MACRO_CELL_ATTRACTOR_SNAPSHOT_SCHEMA,
            "phase": phase,
            "cache": {
                "cache_key": self._cache_key,
                "parent_fingerprint": self._trace_cache.parent_fingerprint,
                "corpus_fingerprint": self._trace_cache.corpus_fingerprint,
            },
            "selection": {
                "selection_sha256": self._selection_sha256,
                "split_seed": self._config.split_seed,
                "eligible_authenticated_record_count": (
                    self._eligible_record_count
                ),
                "rejected_non_greedy_record_count": (
                    self._rejected_non_greedy_record_count
                ),
                "selected_records": [
                    {
                        "batch_index": batch_index,
                        "row_index": row_index,
                        "input_sha256": input_sha256,
                        "horizon": horizon,
                    }
                    for (
                        batch_index,
                        row_index,
                        input_sha256,
                        horizon,
                    ) in self._selected_records
                ],
            },
            "gate_config": self._config.as_dict(),
            "gate_config_sha256": self._config_sha256,
            "child": {
                "model_type": _qualified_type_name(child_model),
                "model_state_sha256": state_before,
                "vocabulary_size": vocabulary_size,
                "eos_token_id": eos_token_id,
                "input_device_type": str(input_device.type),
            },
            "metrics": metrics,
            "records": records,
            "audit": audit,
        }
        payload = {
            **snapshot_body,
            "snapshot_sha256": canonical_sha256(snapshot_body),
        }
        return Lfm2MacroCellAttractorSnapshot(
            MappingProxyType(_json_copy(payload))
        )

    def compare(
        self,
        incumbent: Lfm2MacroCellAttractorSnapshot,
        candidate: Lfm2MacroCellAttractorSnapshot,
    ) -> Lfm2MacroCellAttractorDecision:
        """Reject only health regressions; token equality remains diagnostic."""

        if not isinstance(incumbent, Lfm2MacroCellAttractorSnapshot):
            raise TypeError("incumbent must be Lfm2MacroCellAttractorSnapshot.")
        if not isinstance(candidate, Lfm2MacroCellAttractorSnapshot):
            raise TypeError("candidate must be Lfm2MacroCellAttractorSnapshot.")
        incumbent_payload = incumbent.as_dict()
        candidate_payload = candidate.as_dict()
        for label, payload in (
            ("incumbent", incumbent_payload),
            ("candidate", candidate_payload),
        ):
            if payload.get("schema") != MACRO_CELL_ATTRACTOR_SNAPSHOT_SCHEMA:
                raise Lfm2MacroCellAttractorGateError(
                    f"The {label} snapshot schema is unsupported."
                )
            if payload.get("snapshot_sha256") != canonical_sha256(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "snapshot_sha256"
                }
            ):
                raise Lfm2MacroCellAttractorGateError(
                    f"The {label} snapshot canonical SHA is invalid."
                )
            if (
                payload["cache"]["cache_key"] != self._cache_key
                or payload["selection"]["selection_sha256"]
                != self._selection_sha256
                or payload["gate_config_sha256"] != self._config_sha256
            ):
                raise Lfm2MacroCellAttractorGateError(
                    f"The {label} snapshot belongs to a different sealed gate."
                )

        incumbent_records = {
            str(record["record_id"]): record
            for record in incumbent_payload["records"]
        }
        candidate_records = {
            str(record["record_id"]): record
            for record in candidate_payload["records"]
        }
        if (
            len(incumbent_records) != len(incumbent_payload["records"])
            or len(candidate_records) != len(candidate_payload["records"])
            or tuple(sorted(incumbent_records)) != tuple(sorted(candidate_records))
        ):
            raise Lfm2MacroCellAttractorGateError(
                "Incumbent and candidate record identities do not match."
            )

        metric_contracts = (
            (
                "entropy_parent_distance",
                "mean_entropy_parent_distance",
                self._config.maximum_entropy_parent_distance_regression,
            ),
            (
                "repetition_excess",
                "mean_repetition_excess",
                self._config.maximum_repetition_excess_regression,
            ),
            (
                "dominant_token_excess",
                "mean_dominant_token_excess",
                self._config.maximum_dominant_token_excess_regression,
            ),
            (
                "eos_position_error",
                "mean_eos_position_error",
                self._config.maximum_eos_position_error_regression,
            ),
        )
        checks: dict[str, dict[str, Any]] = {}
        for record_metric, aggregate_metric, limit in metric_contracts:
            mean_regression = (
                candidate_payload["metrics"][aggregate_metric]
                - incumbent_payload["metrics"][aggregate_metric]
            )
            record_regressions = [
                candidate_records[record_id]["relative_to_cached_parent"][
                    record_metric
                ]
                - incumbent_records[record_id]["relative_to_cached_parent"][
                    record_metric
                ]
                for record_id in sorted(candidate_records)
            ]
            maximum_record_regression = max(record_regressions)
            checks[f"mean_{record_metric}_regression"] = {
                "observed": mean_regression,
                "maximum": limit,
                "passed": mean_regression <= limit,
            }
            checks[f"maximum_record_{record_metric}_regression"] = {
                "observed": maximum_record_regression,
                "maximum": limit,
                "passed": maximum_record_regression <= limit,
            }

        new_collapsed_records = [
            record_id
            for record_id in sorted(candidate_records)
            if bool(candidate_records[record_id]["collapsed"])
            and not bool(incumbent_records[record_id]["collapsed"])
        ]
        checks["new_collapsed_records"] = {
            "observed": len(new_collapsed_records),
            "maximum": self._config.maximum_new_collapsed_records,
            "passed": (
                len(new_collapsed_records)
                <= self._config.maximum_new_collapsed_records
            ),
        }
        failed_checks = tuple(
            name for name, check in checks.items() if not bool(check["passed"])
        )
        accepted = not failed_checks
        audit_body = {
            "schema": _DECISION_AUDIT_SCHEMA,
            "parent_forward_calls": 0,
            "parent_trace_regeneration_calls": 0,
            "tokenizer_accessed": False,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "quantization_used": False,
            "pruning_used": False,
            "exact_token_match_is_diagnostic_only": True,
            "exact_token_metrics_used_for_acceptance": False,
            "acceptance_check_names": list(checks),
        }
        audit = {
            **audit_body,
            "audit_sha256": canonical_sha256(audit_body),
        }
        decision_body = {
            "schema": MACRO_CELL_ATTRACTOR_DECISION_SCHEMA,
            "cache_key": self._cache_key,
            "selection_sha256": self._selection_sha256,
            "gate_config_sha256": self._config_sha256,
            "incumbent_snapshot_sha256": incumbent.snapshot_sha256,
            "candidate_snapshot_sha256": candidate.snapshot_sha256,
            "accepted": accepted,
            "checks": checks,
            "failed_checks": list(failed_checks),
            "new_collapsed_record_ids": new_collapsed_records,
            "diagnostics": {
                "incumbent_exact_token_metrics": {
                    key: value
                    for key, value in incumbent_payload["metrics"].items()
                    if key.startswith("diagnostic_")
                },
                "candidate_exact_token_metrics": {
                    key: value
                    for key, value in candidate_payload["metrics"].items()
                    if key.startswith("diagnostic_")
                },
                "exact_token_match_is_diagnostic_only": True,
            },
            "audit": audit,
        }
        payload = {
            **decision_body,
            "decision_sha256": canonical_sha256(decision_body),
        }
        return Lfm2MacroCellAttractorDecision(
            MappingProxyType(_json_copy(payload))
        )


def _resolve_target_rollout_eos_token_id(
    model: Any,
    config: Lfm2TargetTokenFreeRolloutConfig,
    vocabulary_size: int,
) -> int:
    value = config.eos_token_id
    if value is None:
        value = getattr(getattr(model, "config", None), "eos_token_id", None)
    if isinstance(value, bool) or not isinstance(value, int):
        raise Lfm2MacroCellAttractorGateError(
            "An integer eos_token_id is required in the rollout source or child config."
        )
    if value < 0 or value >= vocabulary_size:
        raise Lfm2MacroCellAttractorGateError(
            "The child eos_token_id is outside the target vocabulary."
        )
    return int(value)


def _verify_bridge_cache_batch(
    batch: Lfm2CircuitTraceBatch,
    record: Lfm2BridgeAlignedRecord,
) -> None:
    torch = _torch()
    if batch.index != record.ordinal:
        raise Lfm2MacroCellAttractorGateError(
            "Tokenizer-bridge record order differs from the parent cache."
        )
    if batch.input_sha256 != record.donor_cache_input_sha256:
        raise Lfm2MacroCellAttractorGateError(
            "Tokenizer-bridge donor input does not bind to the parent cache."
        )
    parent = record.as_parent_cache_batch()
    cached_positions = batch.tensors.get("input.position_ids")
    parent_positions = parent.get("position_ids")
    if (
        not torch.equal(batch.input_ids, parent["input_ids"])
        or not torch.equal(batch.attention_mask, parent["attention_mask"])
        or not torch.equal(
            batch.forced_decode_ids,
            parent["forced_decode_ids"],
        )
        or (cached_positions is None and parent_positions is not None)
        or (
            cached_positions is not None
            and (
                parent_positions is None
                or not torch.equal(cached_positions, parent_positions)
            )
        )
    ):
        raise Lfm2MacroCellAttractorGateError(
            "Tokenizer-bridge donor tensors differ from the sealed parent cache."
        )


def _cached_parent_target_step(
    batch: Lfm2CircuitTraceBatch,
    row_index: int,
    step_index: int,
    mapping: Any,
) -> dict[str, Any]:
    torch = _torch()
    if step_index == 0:
        key = (
            "cached_prefill.logits"
            if "cached_prefill.logits" in batch.tensors
            else "prefill.logits"
        )
    else:
        key = f"decode.{step_index - 1:03d}.logits"
    logits = batch.tensors.get(key)
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 3
        or row_index >= int(logits.shape[0])
        or int(logits.shape[1]) <= 0
        or int(logits.shape[2]) <= int(mapping.max().item())
    ):
        raise Lfm2MacroCellAttractorGateError(
            f"Sealed batch {batch.index} has invalid predictor {key!r}."
        )
    donor_vector = logits[row_index, -1, :]
    target_vector = donor_vector.index_select(
        0,
        mapping.to(device=donor_vector.device),
    )
    return {
        "tensor_key": key,
        "donor_top1_token_id": int(donor_vector.argmax(dim=-1).item()),
        "donor_normalized_entropy": _normalized_entropy(donor_vector),
        "projected_target_top1_token_id": int(target_vector.argmax(dim=-1).item()),
        "projected_target_normalized_entropy": _normalized_entropy(target_vector),
    }


def _tokens_through_eos(
    token_ids: Sequence[int],
    eos_token_id: int,
) -> list[int]:
    tokens = [int(value) for value in token_ids]
    position = _first_eos_position(tokens, eos_token_id)
    return tokens if position is None else tokens[: position + 1]


def _termination_evidence(
    token_ids: Sequence[int],
    eos_token_id: int,
    horizon: int,
) -> dict[str, Any]:
    position = _first_eos_position(token_ids, eos_token_id)
    if position is not None:
        return {
            "termination_kind": "eos",
            "censored": False,
            "eos_position": position,
            "observed_token_count": position + 1,
            "configured_horizon": horizon,
        }
    return {
        "termination_kind": "horizon_censored",
        "censored": True,
        "eos_position": None,
        "observed_token_count": len(token_ids),
        "configured_horizon": horizon,
    }


def _target_rollout_match_diagnostics(
    reference: Sequence[int],
    child: Sequence[int],
) -> dict[str, Any]:
    expected = tuple(int(value) for value in reference)
    observed = tuple(int(value) for value in child)
    width = max(1, len(expected), len(observed))
    positional_matches = sum(
        1
        for index in range(width)
        if index < len(expected)
        and index < len(observed)
        and expected[index] == observed[index]
    )
    common_prefix = 0
    for index in range(min(len(expected), len(observed))):
        if expected[index] != observed[index]:
            break
        common_prefix += 1
    first_divergence = None
    if expected != observed:
        first_divergence = common_prefix
    return {
        "exact_sequence_match": expected == observed,
        "exact_token_match_fraction": positional_matches / width,
        "longest_common_prefix_token_count": common_prefix,
        "longest_common_prefix_fraction": common_prefix / width,
        "first_divergence_step": first_divergence,
        "exact_token_match_is_diagnostic_only": True,
    }


def _capture_target_token_rollout_record(
    *,
    model: Any,
    batch: Lfm2CircuitTraceBatch,
    aligned_record: Lfm2BridgeAlignedRecord,
    row_index: int,
    horizon: int,
    eos_token_id: int,
    vocabulary_size: int,
    input_device: Any,
    mapping: Any,
    boundary_order_sha256: str,
) -> tuple[dict[str, Any], int]:
    torch = _torch()
    _verify_bridge_cache_batch(batch, aligned_record)
    target_input_ids = aligned_record.target_input_ids
    attention_mask = aligned_record.attention_mask
    target_position_ids = aligned_record.position_ids
    target_forced_decode_ids = aligned_record.target_forced_decode_ids
    donor_forced_decode_ids = aligned_record.donor_forced_decode_ids
    if (
        target_input_ids.ndim != 2
        or attention_mask.shape != target_input_ids.shape
        or target_forced_decode_ids.ndim != 2
        or donor_forced_decode_ids.shape != target_forced_decode_ids.shape
        or (
            target_position_ids is not None
            and target_position_ids.shape != target_input_ids.shape
        )
        or row_index >= int(target_input_ids.shape[0])
        or horizon > int(target_forced_decode_ids.shape[1])
    ):
        raise Lfm2MacroCellAttractorGateError(
            "A selected target-token bridge record changed shape."
        )
    valid = attention_mask[row_index].to(dtype=torch.bool)
    prompt = target_input_ids[row_index][valid]
    if int(prompt.numel()) <= 0:
        raise Lfm2MacroCellAttractorGateError(
            "A selected target-token prompt has no active tokens."
        )
    if bool(((prompt < 0) | (prompt >= vocabulary_size)).any().item()):
        raise Lfm2MacroCellAttractorGateError(
            "A selected target-token prompt leaves the child vocabulary."
        )
    target_forced = [
        int(value)
        for value in target_forced_decode_ids[row_index, :horizon]
        .detach()
        .cpu()
        .tolist()
    ]
    donor_forced = [
        int(value)
        for value in donor_forced_decode_ids[row_index, :horizon]
        .detach()
        .cpu()
        .tolist()
    ]
    mapped_target_forced = [int(mapping[value].item()) for value in target_forced]
    if mapped_target_forced != donor_forced:
        raise Lfm2MacroCellAttractorGateError(
            "The target forced-prefix route is not an authenticated bridge "
            "image of the donor cache route."
        )
    cached_steps = [
        _cached_parent_target_step(batch, row_index, step_index, mapping)
        for step_index in range(horizon)
    ]
    teacher_route_tokens = _tokens_through_eos(target_forced, eos_token_id)
    batch_record = batch.record
    record_sha256 = batch_record.get("record_sha256")
    if not isinstance(record_sha256, str) or len(record_sha256) != 64:
        raise Lfm2MacroCellAttractorGateError(
            "The cached parent batch lacks its sealed record digest."
        )
    trajectory_id = canonical_sha256(
        {
            "schema": "neural-forge-lfm2-target-token-trajectory-id-v1",
            "adapter_alignment_sha256": aligned_record.alignment_sha256,
            "target_cache_input_sha256": aligned_record.target_cache_input_sha256,
            "row_index": row_index,
        }
    )

    prefix_ids = prompt.to(device=input_device, dtype=torch.long).reshape(1, -1)
    prefix_mask = torch.ones_like(prefix_ids)
    prefix_positions = (
        None
        if target_position_ids is None
        else target_position_ids[row_index][valid]
        .to(device=input_device, dtype=torch.long)
        .reshape(1, -1)
    )
    generated: list[int] = []
    child_entropies: list[float] = []
    steps: list[dict[str, Any]] = []
    prefix_aligned = True
    child_forward_calls = 0
    for step_index in range(horizon):
        teacher_route_active = step_index < len(teacher_route_tokens)
        paired_fit_eligible = prefix_aligned and teacher_route_active
        child_inputs = {
            "input_ids": prefix_ids,
            "attention_mask": prefix_mask,
            "use_cache": False,
            "return_dict": True,
        }
        if prefix_positions is not None:
            child_inputs["position_ids"] = prefix_positions
        output = model(**child_inputs)
        child_forward_calls += 1
        logits = _extract_logits(output)
        if (
            int(logits.shape[0]) != 1
            or int(logits.shape[1]) <= 0
            or int(logits.shape[2]) != vocabulary_size
        ):
            raise Lfm2MacroCellAttractorGateError(
                "The target-token child rollout changed batch or vocabulary shape."
            )
        child_vector = logits[0, -1, :]
        child_entropy = _normalized_entropy(child_vector)
        next_token = int(child_vector.argmax(dim=-1).item())
        teacher_forced_target_token_id = (
            teacher_route_tokens[step_index] if teacher_route_active else None
        )
        cached = cached_steps[step_index]
        reference_target_token_id = (
            cached["projected_target_top1_token_id"]
            if paired_fit_eligible
            else None
        )
        steps.append(
            {
                "step_index": step_index,
                "absolute_target_sequence_position": (
                    int(prompt.numel()) + step_index
                ),
                "predicted_position_id": (
                    int(prompt.numel()) + step_index
                    if prefix_positions is None
                    else int(prefix_positions[0, -1].item()) + 1
                ),
                "target_prefix_before_prediction": [
                    int(value) for value in prefix_ids[0].detach().cpu().tolist()
                ],
                "target_position_ids_before_prediction": (
                    None
                    if prefix_positions is None
                    else [
                        int(value)
                        for value in prefix_positions[0].detach().cpu().tolist()
                    ]
                ),
                "reference_target_token_id": reference_target_token_id,
                "teacher_forced_route_target_token_id": (
                    teacher_forced_target_token_id
                ),
                "teacher_forced_route_donor_token_id": (
                    donor_forced[step_index] if teacher_route_active else None
                ),
                "child_target_token_id": next_token,
                "child_normalized_entropy": child_entropy,
                "cached_parent_prefix_aligned": prefix_aligned,
                "paired_fit_eligible": paired_fit_eligible,
                "cached_parent_locator": {
                    "cache_key": batch_record["cache_key"],
                    "batch_index": batch.index,
                    "input_sha256": batch.input_sha256,
                    "record_sha256": record_sha256,
                    "boundary_order_sha256": boundary_order_sha256,
                    "tensor_key": cached["tensor_key"],
                    "row_index": row_index,
                    "sequence_index": -1,
                },
                "cached_parent_donor_top1_token_id": cached["donor_top1_token_id"],
                "cached_parent_donor_normalized_entropy": cached[
                    "donor_normalized_entropy"
                ],
                "cached_parent_projected_target_top1_token_id": cached[
                    "projected_target_top1_token_id"
                ],
                "cached_parent_projected_target_normalized_entropy": cached[
                    "projected_target_normalized_entropy"
                ],
                "child_matches_cached_parent_projected_target_top1": (
                    reference_target_token_id is not None
                    and next_token == reference_target_token_id
                ),
                "child_matches_teacher_forced_route": (
                    teacher_forced_target_token_id is not None
                    and next_token == teacher_forced_target_token_id
                ),
            }
        )
        generated.append(next_token)
        child_entropies.append(child_entropy)
        if (
            not teacher_route_active
            or next_token != teacher_forced_target_token_id
        ):
            prefix_aligned = False
        if next_token == eos_token_id:
            break
        token = torch.tensor(
            [[next_token]],
            dtype=torch.long,
            device=input_device,
        )
        prefix_ids = torch.cat((prefix_ids, token), dim=1)
        prefix_mask = torch.cat(
            (
                prefix_mask,
                torch.ones(
                    (1, 1),
                    dtype=prefix_mask.dtype,
                    device=input_device,
                ),
            ),
            dim=1,
        )
        if prefix_positions is not None:
            next_position = prefix_positions[:, -1:] + 1
            prefix_positions = torch.cat(
                (prefix_positions, next_position),
                dim=1,
            )

    child_health = _sequence_health(
        generated,
        child_entropies,
        eos_token_id,
    )
    route_diagnostics = _target_rollout_match_diagnostics(
        teacher_route_tokens,
        generated,
    )
    eligible_steps = [
        step for step in steps if bool(step["paired_fit_eligible"])
    ]
    eligible_projected_top1_matches = sum(
        1
        for step in eligible_steps
        if bool(step["child_matches_cached_parent_projected_target_top1"])
    )
    paired_fit_steps = sum(1 for step in steps if bool(step["paired_fit_eligible"]))
    first_unpaired_step = next(
        (
            int(step["step_index"])
            for step in steps
            if not bool(step["paired_fit_eligible"])
        ),
        None,
    )
    return (
        {
            "record_id": (f"{aligned_record.record_id}#row-{row_index:06d}"),
            "trajectory_id": trajectory_id,
            "adapter_record_id": aligned_record.record_id,
            "adapter_record_ordinal": aligned_record.ordinal,
            "split": aligned_record.split,
            "adapter_alignment_sha256": aligned_record.alignment_sha256,
            "batch_index": batch.index,
            "row_index": row_index,
            "donor_cache_input_sha256": batch.input_sha256,
            "target_cache_input_sha256": (aligned_record.target_cache_input_sha256),
            "horizon": horizon,
            "target_prompt_token_ids": [
                int(value) for value in prompt.detach().cpu().tolist()
            ],
            "teacher_forced_route": {
                "token_ids": teacher_route_tokens,
                "sealed_forced_target_token_ids": target_forced,
                "sealed_forced_donor_token_ids": donor_forced,
                "termination": _termination_evidence(
                    teacher_route_tokens,
                    eos_token_id,
                    horizon,
                ),
            },
            "cached_parent": {
                "conditional_context_kind": "authenticated_teacher_forced_prefix",
                "not_a_free_run_trajectory": True,
                "projected_target_top1_token_ids": [
                    int(step["projected_target_top1_token_id"])
                    for step in cached_steps
                ],
                "mean_projected_target_normalized_entropy": _mean(
                    [
                        step["projected_target_normalized_entropy"]
                        for step in cached_steps
                    ],
                    "cached projected-target entropies",
                ),
            },
            "child": {
                **child_health,
                "termination": _termination_evidence(
                    generated,
                    eos_token_id,
                    horizon,
                ),
            },
            "steps": steps,
            "causal_fit": {
                "paired_fit_step_count": paired_fit_steps,
                "evaluation_only_step_count": len(steps) - paired_fit_steps,
                "first_unpaired_step": first_unpaired_step,
                "paired_fit_requires_cached_parent_prefix_alignment": True,
                "post_divergence_steps_are_evaluation_only": True,
                "new_parent_forward_required_for_post_divergence_pairing": True,
            },
            "diagnostics": {
                **{
                    f"teacher_route_{key}": value
                    for key, value in route_diagnostics.items()
                    if key != "exact_token_match_is_diagnostic_only"
                },
                "eligible_projected_top1_match_count": (
                    eligible_projected_top1_matches
                ),
                "eligible_projected_top1_match_fraction": (
                    eligible_projected_top1_matches / max(1, len(eligible_steps))
                ),
                "exact_token_match_is_diagnostic_only": True,
            },
        },
        child_forward_calls,
    )


def _aggregate_target_rollout_metrics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise Lfm2MacroCellAttractorGateError(
            "Target-token rollout selected no child rows."
        )
    step_count = sum(len(record["steps"]) for record in records)
    paired_fit_step_count = sum(
        int(record["causal_fit"]["paired_fit_step_count"]) for record in records
    )
    return {
        "record_count": len(records),
        "observed_step_count": step_count,
        "paired_fit_step_count": paired_fit_step_count,
        "evaluation_only_step_count": step_count - paired_fit_step_count,
        "child_eos_termination_count": sum(
            1
            for record in records
            if record["child"]["termination"]["termination_kind"] == "eos"
        ),
        "child_horizon_censored_count": sum(
            1
            for record in records
            if record["child"]["termination"]["termination_kind"] == "horizon_censored"
        ),
        "teacher_route_eos_termination_count": sum(
            1
            for record in records
            if record["teacher_forced_route"]["termination"]["termination_kind"]
            == "eos"
        ),
        "teacher_route_horizon_censored_count": sum(
            1
            for record in records
            if record["teacher_forced_route"]["termination"]["termination_kind"]
            == "horizon_censored"
        ),
        "mean_child_normalized_entropy": _mean(
            [record["child"]["normalized_entropy"] for record in records],
            "child normalized entropies",
        ),
        "mean_cached_parent_projected_target_normalized_entropy": _mean(
            [
                record["cached_parent"][
                    "mean_projected_target_normalized_entropy"
                ]
                for record in records
            ],
            "cached-parent projected-target entropies",
        ),
        "diagnostic_mean_teacher_route_exact_token_match_fraction": _mean(
            [
                record["diagnostics"][
                    "teacher_route_exact_token_match_fraction"
                ]
                for record in records
            ],
            "teacher-route exact-match diagnostics",
        ),
        "diagnostic_mean_teacher_route_longest_common_prefix_fraction": _mean(
            [
                record["diagnostics"][
                    "teacher_route_longest_common_prefix_fraction"
                ]
                for record in records
            ],
            "teacher-route common-prefix diagnostics",
        ),
        "diagnostic_eligible_projected_top1_match_fraction": (
            sum(
                int(
                    record["diagnostics"][
                        "eligible_projected_top1_match_count"
                    ]
                )
                for record in records
            )
            / max(1, paired_fit_step_count)
        ),
    }


class Lfm2TargetTokenFreeRolloutSource:
    """Capture target-domain free runs backed only by sealed parent semantics.

    The source deliberately has no comparison or acceptance method.  Its
    per-step ``paired_fit_eligible`` field is true only while the live child's
    target-token prefix is the same causal prefix that produced the cached
    parent conditional.  The forced IDs are causal routing anchors, not a
    claim that the unrestricted donor argmax selected those IDs.  Once the
    child diverges from that route, later steps remain useful rollout
    observations but cannot soundly supervise a paired fit without a new
    on-policy parent cache.
    """

    def __init__(
        self,
        trace_cache: Lfm2CircuitTraceCache,
        trace_adapter: Lfm2StrictTokenizerBridgeTraceAdapter,
        *,
        config: Lfm2TargetTokenFreeRolloutConfig | None = None,
    ) -> None:
        if not isinstance(trace_cache, Lfm2CircuitTraceCache):
            raise TypeError("trace_cache must be Lfm2CircuitTraceCache.")
        if not isinstance(
            trace_adapter,
            Lfm2StrictTokenizerBridgeTraceAdapter,
        ):
            raise TypeError(
                "trace_adapter must be Lfm2StrictTokenizerBridgeTraceAdapter."
            )
        selected = Lfm2TargetTokenFreeRolloutConfig() if config is None else config
        if not isinstance(selected, Lfm2TargetTokenFreeRolloutConfig):
            raise TypeError("config must be Lfm2TargetTokenFreeRolloutConfig or None.")
        self._trace_cache = trace_cache
        self._trace_adapter = trace_adapter.verify()
        self._config = selected
        self._cache_key = trace_cache.cache_key
        self._adapter_evidence_sha256 = trace_adapter.evidence_sha256
        topology = trace_cache.manifest.get("layer_topology")
        layer_count = topology.get("layer_count") if isinstance(topology, Mapping) else None
        if (
            isinstance(layer_count, bool)
            or not isinstance(layer_count, int)
            or layer_count <= 0
            or topology.get("layer_topology_sha256")
            != trace_cache.layer_topology_fingerprint
        ):
            raise Lfm2MacroCellAttractorGateError(
                "Parent cache lacks a sealed residual-boundary topology."
            )
        boundary_body = {
            "schema": "neural-forge-lfm2-residual-boundary-order-v1",
            "layer_topology_sha256": trace_cache.layer_topology_fingerprint,
            "boundary_names": [
                "layer.000.residual.pre",
                *[
                    f"layer.{index:03d}.residual.post_layer"
                    for index in range(layer_count)
                ],
                "output.lm_head_input",
            ],
            "final_behavior_endpoint": "output.lm_head_input",
        }
        self._boundary_contract = {
            **boundary_body,
            "boundary_order_sha256": canonical_sha256(boundary_body),
        }
        if trace_cache.batch_count != trace_adapter.record_count:
            raise Lfm2MacroCellAttractorGateError(
                "Parent cache and tokenizer-bridge adapter batch counts differ."
            )
        torch = _torch()
        mapping = torch.tensor(trace_adapter.mapping, dtype=torch.long)
        if (
            mapping.ndim != 1
            or int(mapping.numel()) != trace_adapter.target_row_capacity
            or int(mapping.numel()) <= 1
            or bool((mapping < 0).any().item())
            or int(mapping.max().item()) >= trace_adapter.donor_row_capacity
        ):
            raise Lfm2MacroCellAttractorGateError(
                "Tokenizer-bridge target-to-donor mapping is invalid."
            )
        self._mapping = mapping
        self._mapping_sha256 = canonical_sha256(
            [int(value) for value in mapping.tolist()]
        )

        candidates: list[tuple[str, int, int, str, str, int]] = []
        for batch_index, aligned_record in enumerate(trace_adapter.records):
            batch = trace_cache.load_batch(batch_index)
            _verify_bridge_cache_batch(batch, aligned_record)
            target_forced = aligned_record.target_forced_decode_ids
            donor_forced = aligned_record.donor_forced_decode_ids
            if (
                target_forced.ndim != 2
                or donor_forced.shape != target_forced.shape
                or int(target_forced.shape[0]) <= 0
                or int(target_forced.shape[1]) <= 0
            ):
                raise Lfm2MacroCellAttractorGateError(
                    "Tokenizer-bridge record has no target decode trajectory."
                )
            horizon = int(target_forced.shape[1])
            if selected.maximum_steps is not None:
                horizon = min(horizon, selected.maximum_steps)
            for row_index in range(int(target_forced.shape[0])):
                donor_tokens = [
                    int(value)
                    for value in donor_forced[row_index, :horizon]
                    .detach()
                    .cpu()
                    .tolist()
                ]
                target_tokens = [
                    int(value)
                    for value in target_forced[row_index, :horizon]
                    .detach()
                    .cpu()
                    .tolist()
                ]
                mapped_target_tokens = [
                    int(mapping[value].item()) for value in target_tokens
                ]
                if mapped_target_tokens != donor_tokens:
                    raise Lfm2MacroCellAttractorGateError(
                        "Tokenizer-bridge forced-prefix mapping changed after "
                        "adapter authentication."
                    )
                rank = canonical_sha256(
                    {
                        "schema": _TARGET_TOKEN_FREE_ROLLOUT_SELECTION_SCHEMA,
                        "cache_key": self._cache_key,
                        "trace_adapter_evidence_sha256": (
                            self._adapter_evidence_sha256
                        ),
                        "split_seed": selected.split_seed,
                        "adapter_record_id": aligned_record.record_id,
                        "batch_index": batch_index,
                        "row_index": row_index,
                        "donor_cache_input_sha256": batch.input_sha256,
                        "target_cache_input_sha256": (
                            aligned_record.target_cache_input_sha256
                        ),
                        "horizon": horizon,
                    }
                )
                candidates.append(
                    (
                        rank,
                        batch_index,
                        row_index,
                        batch.input_sha256,
                        aligned_record.alignment_sha256,
                        horizon,
                    )
                )
        candidates.sort()
        chosen = tuple(candidates[: selected.maximum_records])
        if not chosen:
            raise Lfm2MacroCellAttractorGateError(
                "No target-token bridge row authenticates a cached parent "
                "teacher-forced prefix trajectory."
            )
        self._selected_records = tuple(
            (
                batch_index,
                row_index,
                donor_input_sha256,
                alignment_sha256,
                horizon,
            )
            for (
                _rank,
                batch_index,
                row_index,
                donor_input_sha256,
                alignment_sha256,
                horizon,
            ) in chosen
        )
        self._eligible_record_count = len(candidates)
        self._selection_sha256 = canonical_sha256(self._selection_body())
        self._config_sha256 = canonical_sha256(
            {
                "schema": _TARGET_TOKEN_FREE_ROLLOUT_CONFIG_SCHEMA,
                "config": selected.as_dict(),
            }
        )

    @property
    def config(self) -> Lfm2TargetTokenFreeRolloutConfig:
        return self._config

    @property
    def selection_sha256(self) -> str:
        return self._selection_sha256

    def _selection_body(self) -> dict[str, Any]:
        return {
            "schema": _TARGET_TOKEN_FREE_ROLLOUT_SELECTION_SCHEMA,
            "cache_key": self._cache_key,
            "trace_adapter_evidence_sha256": self._adapter_evidence_sha256,
            "split_seed": self._config.split_seed,
            "eligible_authenticated_record_count": self._eligible_record_count,
            "selection_requires_parent_global_greedy_tokens": False,
            "selected_records": [
                {
                    "batch_index": batch_index,
                    "row_index": row_index,
                    "donor_cache_input_sha256": donor_input_sha256,
                    "adapter_alignment_sha256": alignment_sha256,
                    "horizon": horizon,
                }
                for (
                    batch_index,
                    row_index,
                    donor_input_sha256,
                    alignment_sha256,
                    horizon,
                ) in self._selected_records
            ],
        }

    def capture(
        self,
        child_model: Any,
        *,
        phase: str,
    ) -> Lfm2TargetTokenFreeRolloutSnapshot:
        """Capture child target-token trajectories without making a decision."""

        torch = _torch()
        if not isinstance(phase, str) or not phase.strip():
            raise ValueError("phase must be a nonempty string.")
        phase = phase.strip()
        if self._trace_adapter.verify().evidence_sha256 != (
            self._adapter_evidence_sha256
        ):
            raise Lfm2MacroCellAttractorGateError(
                "Tokenizer-bridge trace identity changed after source binding."
            )
        vocabulary_size = _model_vocabulary_size(child_model)
        if vocabulary_size != self._trace_adapter.target_row_capacity:
            raise Lfm2MacroCellAttractorGateError(
                "The child vocabulary does not match the tokenizer-bridge "
                "target row capacity."
            )
        eos_token_id = _resolve_target_rollout_eos_token_id(
            child_model,
            self._config,
            vocabulary_size,
        )
        input_device = _model_input_device(child_model)
        state_before = fingerprint_model_state(child_model)
        gradients_before = _gradient_fingerprint(child_model)
        flags_before = _training_flags(child_model)
        flags_sha256_before = _training_flags_sha256(flags_before)

        records: list[dict[str, Any]] = []
        child_forward_calls = 0
        failure: Exception | None = None
        try:
            child_model.eval()
            with torch.inference_mode():
                for (
                    batch_index,
                    row_index,
                    expected_input_sha256,
                    expected_alignment_sha256,
                    horizon,
                ) in self._selected_records:
                    batch = self._trace_cache.load_batch(batch_index)
                    aligned_record = self._trace_adapter.records[batch_index]
                    if (
                        batch.input_sha256 != expected_input_sha256
                        or aligned_record.alignment_sha256 != expected_alignment_sha256
                    ):
                        raise Lfm2MacroCellAttractorGateError(
                            "A selected target-token rollout binding changed."
                        )
                    record, forward_calls = _capture_target_token_rollout_record(
                        model=child_model,
                        batch=batch,
                        aligned_record=aligned_record,
                        row_index=row_index,
                        horizon=horizon,
                        eos_token_id=eos_token_id,
                        vocabulary_size=vocabulary_size,
                        input_device=input_device,
                        mapping=self._mapping,
                        boundary_order_sha256=self._boundary_contract[
                            "boundary_order_sha256"
                        ],
                    )
                    records.append(record)
                    child_forward_calls += forward_calls
        except Exception as exc:  # state checks still run before surfacing failure
            failure = exc
        finally:
            _restore_training_flags(flags_before)

        state_after = fingerprint_model_state(child_model)
        gradients_after = _gradient_fingerprint(child_model)
        flags_after = _training_flags(child_model)
        flags_sha256_after = _training_flags_sha256(flags_after)
        state_preserved = (
            state_before == state_after
            and gradients_before == gradients_after
            and flags_sha256_before == flags_sha256_after
        )
        if not state_preserved:
            raise Lfm2MacroCellAttractorGateError(
                "Target-token rollout changed child state, gradients, or "
                "module training flags."
            ) from failure
        if failure is not None:
            if isinstance(failure, Lfm2MacroCellAttractorGateError):
                raise failure
            raise Lfm2MacroCellAttractorGateError(
                "The target-token child free rollout failed."
            ) from failure

        metrics = _aggregate_target_rollout_metrics(records)
        audit_body = {
            "schema": _TARGET_TOKEN_FREE_ROLLOUT_AUDIT_SCHEMA,
            "parent_forward_calls": 0,
            "parent_trace_regeneration_calls": 0,
            "child_forward_calls": child_forward_calls,
            "tokenizer_accessed": False,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "quantization_used": False,
            "pruning_used": False,
            "inference_mode": True,
            "child_use_cache": False,
            "child_rollout_mode": "target_token_full_prefix_greedy_argmax",
            "sealed_parent_teacher_prefix_conditionals_authenticated": True,
            "parent_global_greedy_trajectory_required": False,
            "numeric_quality_gate": False,
            "exact_token_match_is_diagnostic_only": True,
            "horizon_censored_is_failure": False,
            "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after,
            "gradient_sha256_before": gradients_before,
            "gradient_sha256_after": gradients_after,
            "training_flags_sha256_before": flags_sha256_before,
            "training_flags_sha256_after": flags_sha256_after,
            "state_preserved": True,
        }
        audit = {
            **audit_body,
            "audit_sha256": canonical_sha256(audit_body),
        }
        snapshot_body = {
            "schema": TARGET_TOKEN_FREE_ROLLOUT_SNAPSHOT_SCHEMA,
            "phase": phase,
            "cache": {
                "cache_key": self._cache_key,
                "parent_fingerprint": self._trace_cache.parent_fingerprint,
                "corpus_fingerprint": self._trace_cache.corpus_fingerprint,
            },
            "tokenizer_bridge": {
                "trace_adapter_evidence_sha256": (self._adapter_evidence_sha256),
                "bridge_artifact_sha256": (self._trace_adapter.bridge_artifact_sha256),
                "transport_profile_fingerprint": (
                    self._trace_adapter.transport_profile_fingerprint
                ),
                "mapping_sha256": self._mapping_sha256,
                "donor_row_capacity": (self._trace_adapter.donor_row_capacity),
                "target_row_capacity": (self._trace_adapter.target_row_capacity),
                "numeric_token_id_equality_required": False,
            },
            "cached_parent_residual_boundaries": self._boundary_contract,
            "selection": {
                **self._selection_body(),
                "selection_sha256": self._selection_sha256,
            },
            "rollout_config": self._config.as_dict(),
            "rollout_config_sha256": self._config_sha256,
            "child": {
                "model_type": _qualified_type_name(child_model),
                "model_state_sha256": state_before,
                "vocabulary_size": vocabulary_size,
                "eos_token_id": eos_token_id,
                "input_device_type": str(input_device.type),
            },
            "quality_contract": {
                "numeric_quality_gate": False,
                "manual_or_semantic_review_remains_authoritative": True,
                "exact_token_match_is_diagnostic_only": True,
                "horizon_censored_is_failure": False,
            },
            "causal_fit_contract": {
                "target_token_prefixes_exposed": True,
                "cached_parent_tensor_locators_exposed": True,
                "cached_parent_record_digests_exposed": True,
                "canonical_residual_boundary_order_sealed": True,
                "teacher_forced_route_is_causal_join_key": True,
                "cached_parent_targets_are_teacher_prefix_conditionals": True,
                "paired_fit_eligibility_is_per_step": True,
                "post_divergence_steps_are_evaluation_only": True,
                "parent_model_available_to_source": False,
            },
            "metrics": metrics,
            "records": records,
            "audit": audit,
        }
        payload = {
            **snapshot_body,
            "snapshot_sha256": canonical_sha256(snapshot_body),
        }
        return Lfm2TargetTokenFreeRolloutSnapshot(MappingProxyType(_json_copy(payload)))


__all__ = [
    "MACRO_CELL_ATTRACTOR_DECISION_SCHEMA",
    "MACRO_CELL_ATTRACTOR_SNAPSHOT_SCHEMA",
    "TARGET_TOKEN_FREE_ROLLOUT_SNAPSHOT_SCHEMA",
    "Lfm2MacroCellAttractorDecision",
    "Lfm2MacroCellAttractorGate",
    "Lfm2MacroCellAttractorGateConfig",
    "Lfm2MacroCellAttractorGateError",
    "Lfm2MacroCellAttractorSnapshot",
    "Lfm2TargetTokenFreeRolloutConfig",
    "Lfm2TargetTokenFreeRolloutSnapshot",
    "Lfm2TargetTokenFreeRolloutSource",
]
