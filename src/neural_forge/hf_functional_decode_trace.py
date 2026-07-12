"""Lightweight output-function traces for behavior-first fold selection.

Unlike :mod:`hf_cached_attention_trace`, this collector deliberately does not
reconstruct attention, activations, residuals, or KV tensors.  It records only
the token path and actual next-token logits produced by the model's real
cached decoding path.  It is therefore suitable for choosing projected
weights that may be numerically unrelated to their parent weights.

The collector is inference-only and bounded.  It preserves model mode,
configuration flags, and existing gradient objects, and it rejects malformed
or non-finite model output.  Internal tensor similarity is not an input.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import Tensor, nn


FUNCTIONAL_DECODE_TRACE_SCHEMA = "neural-forge-functional-decode-trace-v2"
FunctionalDecodeLane = Literal["forced", "greedy"]


class FunctionalDecodeTraceError(RuntimeError):
    """A functional decode request or model output is invalid."""


class FunctionalDecodeTraceLimitError(FunctionalDecodeTraceError):
    """A declared functional decode resource bound was exceeded."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise FunctionalDecodeTraceError(f"{name} must be positive.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise FunctionalDecodeTraceError(f"{name} must be positive.") from error
    if result <= 0:
        raise FunctionalDecodeTraceError(f"{name} must be positive.")
    return result


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FunctionalDecodeTraceError(
            "Functional trace provenance is not canonical JSON."
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    detached = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(detached.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(detached.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _qualified_class_name(value: Any) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


@dataclass(frozen=True, slots=True)
class FunctionalDecodeLimits:
    max_horizon: int = 32
    max_prompt_tokens: int = 512
    max_vocabulary: int = 1_000_000
    max_logit_bytes: int = 256 * 1024 * 1024
    max_model_devices: int = 1

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class FunctionalDecodeStep:
    step_index: int
    prefix_length: int
    selected_token_id: int
    greedy_token_id: int
    cache_length: int
    action_logits: Tensor
    action_logits_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "prefix_length": self.prefix_length,
            "selected_token_id": self.selected_token_id,
            "greedy_token_id": self.greedy_token_id,
            "cache_length": self.cache_length,
            "selected_is_greedy": self.selected_token_id == self.greedy_token_id,
            "action_logits_shape": list(self.action_logits.shape),
            "action_logits_dtype": str(self.action_logits.dtype),
            "action_logits_sha256": self.action_logits_sha256,
        }


@dataclass(frozen=True, slots=True)
class FunctionalDecodeTrace:
    sequence_id: str
    lane: FunctionalDecodeLane
    prompt_token_ids: tuple[int, ...]
    consumed_token_ids: tuple[int, ...]
    selected_token_ids: tuple[int, ...]
    steps: tuple[FunctionalDecodeStep, ...]
    vocabulary_size: int
    logit_bytes: int
    model_class: str
    cache_class: str
    trace_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": FUNCTIONAL_DECODE_TRACE_SCHEMA,
            "sequence_id": self.sequence_id,
            "lane": self.lane,
            "prompt_token_ids": list(self.prompt_token_ids),
            "consumed_token_ids": list(self.consumed_token_ids),
            "selected_token_ids": list(self.selected_token_ids),
            "steps": [value.as_dict() for value in self.steps],
            "vocabulary_size": self.vocabulary_size,
            "logit_bytes": self.logit_bytes,
            "model_class": self.model_class,
            "cache_class": self.cache_class,
            "trace_sha256": self.trace_sha256,
            "output_function_only": True,
            "excluded_similarity_inputs": [
                "weights",
                "qkv",
                "attention",
                "activations",
                "residuals",
                "cache-values",
            ],
            "gradient_free": True,
            "optimizer_constructed": False,
        }


def _model_devices(model: nn.Module) -> tuple[torch.device, ...]:
    return tuple(
        sorted(
            {value.device for value in (*model.parameters(), *model.buffers())},
            key=str,
        )
    )


def _model_device(model: nn.Module) -> torch.device:
    embeddings = getattr(model, "get_input_embeddings", None)
    if callable(embeddings):
        module = embeddings()
        weight = getattr(module, "weight", None)
        if isinstance(weight, Tensor):
            return weight.device
    for value in (*model.parameters(), *model.buffers()):
        return value.device
    raise FunctionalDecodeTraceError("model has no parameter or buffer device.")


def _output_field(output: Any, name: str) -> Any:
    value = getattr(output, name, None)
    if value is None and isinstance(output, Mapping):
        value = output.get(name)
    return value


def _output_logits(output: Any) -> Tensor:
    logits = _output_field(output, "logits")
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise FunctionalDecodeTraceError(
            "Causal model output must contain [batch,sequence,vocab] logits."
        )
    if not logits.dtype.is_floating_point or not bool(
        torch.isfinite(logits.detach()).all()
    ):
        raise FunctionalDecodeTraceError("Model logits must be finite floating point.")
    return logits


def _cache_sequence_length(cache: Any) -> int:
    """Read one structural cache length without inspecting cached values."""

    getter = getattr(cache, "get_seq_length", None)
    if callable(getter):
        for arguments in ((), (0,)):
            try:
                value = getter(*arguments)
            except TypeError:
                continue
            except Exception as error:
                raise FunctionalDecodeTraceError(
                    "Cache get_seq_length failed."
                ) from error
            if isinstance(value, Tensor):
                if value.numel() != 1:
                    raise FunctionalDecodeTraceError(
                        "Cache sequence length must be one integer."
                    )
                value = value.item()
            if isinstance(value, bool):
                raise FunctionalDecodeTraceError(
                    "Cache sequence length must be one integer."
                )
            try:
                length = operator.index(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise FunctionalDecodeTraceError(
                    "Cache sequence length must be one integer."
                ) from error
            if length < 0:
                raise FunctionalDecodeTraceError(
                    "Cache sequence length cannot be negative."
                )
            return length

    lengths: list[int] = []
    layers = getattr(cache, "layers", None)
    if isinstance(layers, Sequence):
        for layer in layers:
            keys = getattr(layer, "keys", None)
            if isinstance(keys, Tensor) and keys.ndim >= 3:
                lengths.append(int(keys.shape[-2]))
    elif isinstance(cache, (tuple, list)):
        for layer in cache:
            if (
                isinstance(layer, (tuple, list))
                and layer
                and isinstance(layer[0], Tensor)
                and layer[0].ndim >= 3
            ):
                lengths.append(int(layer[0].shape[-2]))
    if not lengths:
        raise FunctionalDecodeTraceError(
            "Cache does not expose a verifiable sequence length."
        )
    if len(set(lengths)) != 1:
        raise FunctionalDecodeTraceError(
            "Cache layers disagree on their sequence length."
        )
    return lengths[0]


def _declared_vocabulary_sizes(model: nn.Module) -> tuple[int | None, int | None]:
    config = getattr(model, "config", None)
    configured = getattr(config, "vocab_size", None)
    if configured is not None:
        configured = _positive_int(configured, "config.vocab_size")
    embedded: int | None = None
    embeddings = getattr(model, "get_input_embeddings", None)
    if callable(embeddings):
        module = embeddings()
        count = getattr(module, "num_embeddings", None)
        weight = getattr(module, "weight", None)
        if count is None and isinstance(weight, Tensor) and weight.ndim >= 1:
            count = int(weight.shape[0])
        if count is not None:
            embedded = _positive_int(count, "input embedding vocabulary")
    return configured, embedded


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


def _forward(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    past_key_values: Any,
) -> Any:
    values = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": True,
        "return_dict": True,
        "output_attentions": False,
        "output_hidden_states": False,
        "logits_to_keep": 1,
    }
    return model(**_forward_kwargs(model, values))


def _gradient_state(
    model: nn.Module,
) -> tuple[tuple[nn.Parameter, bool, Any, Any], ...]:
    result = []
    for parameter in model.parameters():
        gradient = parameter.grad
        version = None if gradient is None else getattr(gradient, "_version", None)
        result.append((parameter, bool(parameter.requires_grad), gradient, version))
    return tuple(result)


def _restore_gradient_state(
    state: Sequence[tuple[nn.Parameter, bool, Any, Any]],
) -> None:
    for parameter, requires_grad, gradient, _version in state:
        if bool(parameter.requires_grad) != requires_grad:
            parameter.requires_grad_(requires_grad)
        if parameter.grad is not gradient:
            parameter.grad = gradient


def _forced_tokens(
    values: Sequence[int] | Tensor | None,
    *,
    lane: FunctionalDecodeLane,
    horizon: int,
) -> tuple[int, ...] | None:
    if lane == "greedy":
        if values is not None:
            raise FunctionalDecodeTraceError(
                "forced_token_ids is valid only for a forced lane."
            )
        return None
    if values is None:
        raise FunctionalDecodeTraceError(
            "forced_token_ids is required for a forced lane."
        )
    tensor = torch.as_tensor(values)
    if tensor.dtype == torch.bool or tensor.is_floating_point() or tensor.is_complex():
        raise FunctionalDecodeTraceError("forced_token_ids must be integer IDs.")
    tensor = tensor.to(dtype=torch.long).reshape(-1)
    if int(tensor.numel()) < horizon:
        raise FunctionalDecodeTraceError(
            "forced_token_ids must contain at least horizon tokens."
        )
    result = tuple(int(value) for value in tensor[:horizon].tolist())
    if any(value < 0 for value in result):
        raise FunctionalDecodeTraceError(
            "forced_token_ids contains a negative token ID."
        )
    return result


def collect_functional_decode_trace(
    model: nn.Module,
    input_ids: Any,
    *,
    horizon: int,
    sequence_id: str = "prompt-0",
    lane: FunctionalDecodeLane = "greedy",
    forced_token_ids: Sequence[int] | Tensor | None = None,
    limits: FunctionalDecodeLimits | None = None,
) -> FunctionalDecodeTrace:
    """Collect one bounded real cached token/logit trajectory."""

    if not isinstance(model, nn.Module):
        raise FunctionalDecodeTraceError("model must be a torch nn.Module.")
    controls = limits or FunctionalDecodeLimits()
    if not isinstance(controls, FunctionalDecodeLimits):
        raise FunctionalDecodeTraceError("limits must be FunctionalDecodeLimits.")
    horizon_value = _positive_int(horizon, "horizon")
    if horizon_value > controls.max_horizon:
        raise FunctionalDecodeTraceLimitError(
            f"horizon exceeds max_horizon={controls.max_horizon}."
        )
    normalized_lane = str(lane).strip().lower()
    if normalized_lane not in {"forced", "greedy"}:
        raise FunctionalDecodeTraceError("lane must be forced or greedy.")
    sequence_name = str(sequence_id).strip()
    if not sequence_name:
        raise FunctionalDecodeTraceError("sequence_id cannot be blank.")
    drivers = _forced_tokens(
        forced_token_ids,
        lane=normalized_lane,  # type: ignore[arg-type]
        horizon=horizon_value,
    )
    devices = _model_devices(model)
    if not devices or len(devices) > controls.max_model_devices:
        raise FunctionalDecodeTraceLimitError(
            "Model device count exceeds max_model_devices or is empty."
        )
    device = _model_device(model)
    ids = torch.as_tensor(input_ids, device=device)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if (
        ids.ndim != 2
        or int(ids.shape[0]) != 1
        or int(ids.shape[1]) <= 0
        or int(ids.shape[1]) > controls.max_prompt_tokens
        or ids.dtype == torch.bool
        or ids.is_floating_point()
        or ids.is_complex()
    ):
        raise FunctionalDecodeTraceError(
            "input_ids must be one non-empty bounded integer prompt."
        )
    ids = ids.to(dtype=torch.long)
    if int(ids.min().item()) < 0:
        raise FunctionalDecodeTraceError("input_ids contains a negative token ID.")

    configured_vocabulary, embedding_vocabulary = _declared_vocabulary_sizes(model)
    for name, size in (
        ("Configured", configured_vocabulary),
        ("Input embedding", embedding_vocabulary),
    ):
        if size is not None and not 1 < size <= controls.max_vocabulary:
            raise FunctionalDecodeTraceLimitError(
                f"{name} vocabulary exceeds its declared bound."
            )
    prompt_maximum = int(ids.max().item())
    if embedding_vocabulary is not None and prompt_maximum >= embedding_vocabulary:
        raise FunctionalDecodeTraceError(
            "input_ids contains a token outside the input embedding vocabulary."
        )
    if configured_vocabulary is not None and prompt_maximum >= configured_vocabulary:
        raise FunctionalDecodeTraceError(
            "input_ids contains an out-of-vocabulary token."
        )
    if (
        drivers is not None
        and configured_vocabulary is not None
        and any(value >= configured_vocabulary for value in drivers)
    ):
        raise FunctionalDecodeTraceError(
            "forced_token_ids contains an out-of-vocabulary token."
        )

    config = getattr(model, "config", None)
    config_was_attached = hasattr(model, "config")
    config_values: dict[str, Any] = {}
    for name in ("use_cache", "output_attentions", "output_hidden_states"):
        if config is not None and hasattr(config, name):
            config_values[name] = getattr(config, name)
    modes = tuple((module, bool(module.training)) for module in model.modules())
    gradients_before = _gradient_state(model)
    prefix = ids.clone()
    next_input = ids
    past = None
    selected_tokens: list[int] = []
    consumed_tokens: list[int] = []
    steps: list[FunctionalDecodeStep] = []
    logit_bytes = 0
    vocabulary_size: int | None = None
    cache_class: str | None = None
    try:
        for parameter, _requires_grad, _gradient, _version in gradients_before:
            parameter.grad = None
        model.eval()
        if config is not None:
            for name, value in (
                ("use_cache", True),
                ("output_attentions", False),
                ("output_hidden_states", False),
            ):
                if hasattr(config, name):
                    setattr(config, name, value)
        with torch.inference_mode():
            for step_index in range(horizon_value):
                attention_mask = torch.ones(
                    (1, int(prefix.shape[1])), dtype=torch.long, device=device
                )
                output = _forward(
                    model,
                    next_input,
                    attention_mask,
                    past_key_values=past,
                )
                logits = _output_logits(output)
                if logits.device != device:
                    raise FunctionalDecodeTraceError(
                        "Model logits are not on the decode input device."
                    )
                if int(logits.shape[0]) != 1 or int(logits.shape[1]) <= 0:
                    raise FunctionalDecodeTraceError(
                        "Model logits do not align with a single decode row."
                    )
                action = logits[:, -1, :].detach().cpu().contiguous().clone()
                if vocabulary_size is None:
                    vocabulary_size = int(action.shape[-1])
                    if not 1 < vocabulary_size <= controls.max_vocabulary:
                        raise FunctionalDecodeTraceLimitError(
                            "Output vocabulary exceeds its declared bound."
                        )
                    if configured_vocabulary is not None and (
                        vocabulary_size != configured_vocabulary
                    ):
                        raise FunctionalDecodeTraceError(
                            "Output vocabulary disagrees with config.vocab_size."
                        )
                    if prompt_maximum >= vocabulary_size:
                        raise FunctionalDecodeTraceError(
                            "input_ids contains an out-of-vocabulary token."
                        )
                elif int(action.shape[-1]) != vocabulary_size:
                    raise FunctionalDecodeTraceError(
                        "Output vocabulary changed during cached decode."
                    )
                greedy = int(torch.argmax(action, dim=-1).item())
                selected = greedy if drivers is None else drivers[step_index]
                if selected >= vocabulary_size:
                    raise FunctionalDecodeTraceError(
                        "forced_token_ids contains an out-of-vocabulary token."
                    )
                logit_bytes += int(action.numel() * action.element_size())
                if logit_bytes > controls.max_logit_bytes:
                    raise FunctionalDecodeTraceLimitError(
                        "Functional trace exceeded max_logit_bytes."
                    )
                past = _output_field(output, "past_key_values")
                if past is None:
                    raise FunctionalDecodeTraceError(
                        "Model returned no cache under use_cache=True."
                    )
                current_cache_class = _qualified_class_name(past)
                if cache_class is None:
                    cache_class = current_cache_class
                elif current_cache_class != cache_class:
                    raise FunctionalDecodeTraceError(
                        "Cache class changed during one functional trace."
                    )
                cache_length = _cache_sequence_length(past)
                if cache_length != int(prefix.shape[1]):
                    raise FunctionalDecodeTraceError(
                        "Cache sequence length does not equal the consumed prefix."
                    )
                steps.append(
                    FunctionalDecodeStep(
                        step_index=step_index,
                        prefix_length=int(prefix.shape[1]),
                        selected_token_id=selected,
                        greedy_token_id=greedy,
                        cache_length=cache_length,
                        action_logits=action,
                        action_logits_sha256=_tensor_sha256(action),
                    )
                )
                selected_tokens.append(selected)
                if step_index + 1 < horizon_value:
                    consumed_tokens.append(selected)
                    next_input = torch.tensor(
                        [[selected]], dtype=torch.long, device=device
                    )
                    prefix = torch.cat((prefix, next_input), dim=1)
    finally:
        restoration_errors: list[Exception] = []
        for name, value in config_values.items():
            try:
                setattr(config, name, value)
            except Exception as error:  # pragma: no cover - defensive exotic config
                restoration_errors.append(error)
        if config_was_attached and getattr(model, "config", None) is not config:
            try:
                setattr(model, "config", config)
            except Exception as error:  # pragma: no cover - defensive exotic module
                restoration_errors.append(error)
        elif not config_was_attached and hasattr(model, "config"):
            try:
                delattr(model, "config")
            except Exception as error:  # pragma: no cover - defensive exotic module
                restoration_errors.append(error)
        for module, was_training in modes:
            try:
                module.training = was_training
            except Exception as error:  # pragma: no cover - defensive exotic module
                restoration_errors.append(error)
        try:
            _restore_gradient_state(gradients_before)
        except Exception as error:  # pragma: no cover - defensive exotic parameter
            restoration_errors.append(error)
        if restoration_errors:
            raise FunctionalDecodeTraceError(
                "Functional decode could not restore model execution state."
            ) from restoration_errors[0]

    if config_was_attached and getattr(model, "config", None) is not config:
        raise FunctionalDecodeTraceError(
            "Functional decode changed the model config object."
        )
    if not config_was_attached and hasattr(model, "config"):
        raise FunctionalDecodeTraceError(
            "Functional decode added a model config object."
        )
    if any(bool(module.training) != was_training for module, was_training in modes):
        raise FunctionalDecodeTraceError("Functional decode changed model modes.")
    if any(getattr(config, name) != value for name, value in config_values.items()):
        raise FunctionalDecodeTraceError(
            "Functional decode changed model configuration flags."
        )
    gradients_after = _gradient_state(model)
    if len(gradients_before) != len(gradients_after) or any(
        before_parameter is not after_parameter
        or before_requires_grad != after_requires_grad
        or before_gradient is not after_gradient
        or before_version != after_version
        for (
            before_parameter,
            before_requires_grad,
            before_gradient,
            before_version,
        ), (
            after_parameter,
            after_requires_grad,
            after_gradient,
            after_version,
        ) in zip(gradients_before, gradients_after, strict=True)
    ):
        raise FunctionalDecodeTraceError(
            "Functional decode changed model parameter gradients."
        )
    if vocabulary_size is None or cache_class is None:
        raise FunctionalDecodeTraceError("Functional trace collected no decode step.")
    prompt = tuple(int(value) for value in ids.detach().cpu().reshape(-1).tolist())
    payload = {
        "schema": FUNCTIONAL_DECODE_TRACE_SCHEMA,
        "sequence_id": sequence_name,
        "lane": normalized_lane,
        "prompt_token_ids": list(prompt),
        "consumed_token_ids": consumed_tokens,
        "selected_token_ids": selected_tokens,
        "step_logits_sha256s": [value.action_logits_sha256 for value in steps],
        "step_cache_lengths": [value.cache_length for value in steps],
        "vocabulary_size": vocabulary_size,
        "model_class": _qualified_class_name(model),
        "cache_class": cache_class,
    }
    return FunctionalDecodeTrace(
        sequence_id=sequence_name,
        lane=normalized_lane,  # type: ignore[arg-type]
        prompt_token_ids=prompt,
        consumed_token_ids=tuple(consumed_tokens),
        selected_token_ids=tuple(selected_tokens),
        steps=tuple(steps),
        vocabulary_size=vocabulary_size,
        logit_bytes=logit_bytes,
        model_class=_qualified_class_name(model),
        cache_class=cache_class,
        trace_sha256=_canonical_sha256(payload),
    )


def trace_mapping(
    traces: Sequence[FunctionalDecodeTrace],
) -> Mapping[tuple[str, str, tuple[int, ...]], FunctionalDecodeTrace]:
    """Return a collision-checked scorer-compatible trace mapping."""

    values: dict[tuple[str, str, tuple[int, ...]], FunctionalDecodeTrace] = {}
    for trace in traces:
        if not isinstance(trace, FunctionalDecodeTrace):
            raise FunctionalDecodeTraceError(
                "traces must contain FunctionalDecodeTrace values."
            )
        key = (trace.lane, trace.sequence_id, trace.prompt_token_ids)
        if key in values:
            raise FunctionalDecodeTraceError("Functional trace key is duplicated.")
        values[key] = trace
    if not values:
        raise FunctionalDecodeTraceError("traces cannot be empty.")
    return MappingProxyType(values)


__all__ = [
    "FUNCTIONAL_DECODE_TRACE_SCHEMA",
    "FunctionalDecodeLane",
    "FunctionalDecodeLimits",
    "FunctionalDecodeStep",
    "FunctionalDecodeTrace",
    "FunctionalDecodeTraceError",
    "FunctionalDecodeTraceLimitError",
    "collect_functional_decode_trace",
    "trace_mapping",
]
