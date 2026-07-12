"""Bounded cached pseudo-decode traces for Qwen-like HF causal decoders.

The ordinary transport collectors run teacher-forced forwards with
``use_cache=False``.  That cannot expose an attention fold which writes a bad
K/V state on one token and only fails when a later token consumes it.  This
module instead runs a prefill followed by true one-token cached forwards.

For each captured step it reconstructs post-RoPE attention scores from hooked
post-Q/K-normalization states and the exact layer cache.  The reconstruction is
compared against the selected heads of the real pre-O context. Separately,
every cached logit row is compared with a fresh full-prefix ``use_cache=False``
forward. ``strict`` mode requires all finite reconstructions to match declared
tolerances; explicit ``record`` mode retains their errors and validity flags as
diagnostics while preserving all structural, finite-value, and identity checks.

The collector is inference-only.  Hooks copy bounded tensors to CPU, no
gradient graph is retained, no model tensor is edited, and module/config modes
are restored exactly in ``finally`` blocks.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import Tensor, nn

from neural_forge.hf_residual_attention_fold import (
    ResidualAttentionCandidateEvidence,
    ResidualAttentionRollout,
    ResidualAttentionRolloutPair,
    ResidualAttentionStepEvidence,
)


CachedAttentionLane = Literal["forced", "greedy"]
CachedTraceValidationMode = Literal["strict", "record"]
TRACE_SCHEMA = "neural-forge-hf-cached-attention-trace-v2"


class HFCachedAttentionTraceError(RuntimeError):
    """Base failure for an unsupported or unverified cached attention trace."""


class HFCachedAttentionTraceShapeError(HFCachedAttentionTraceError):
    """A model, prompt, group, cache, or captured tensor has invalid shape."""


class HFCachedAttentionTraceStateError(HFCachedAttentionTraceError):
    """Cached and full-prefix execution or hook state could not be certified."""


class HFCachedAttentionTraceLimitError(HFCachedAttentionTraceError):
    """A declared prompt, horizon, device, or memory bound was exceeded."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise HFCachedAttentionTraceShapeError(f"{name} must be positive.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise HFCachedAttentionTraceShapeError(f"{name} must be positive.") from error
    if result <= 0:
        raise HFCachedAttentionTraceShapeError(f"{name} must be positive.")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise HFCachedAttentionTraceShapeError(f"{name} cannot be negative.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise HFCachedAttentionTraceShapeError(f"{name} cannot be negative.") from error
    if result < 0:
        raise HFCachedAttentionTraceShapeError(f"{name} cannot be negative.")
    return result


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise HFCachedAttentionTraceShapeError(f"{name} must be positive.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise HFCachedAttentionTraceShapeError(f"{name} must be positive.") from error
    if not math.isfinite(result) or result <= 0.0:
        raise HFCachedAttentionTraceShapeError(f"{name} must be positive.")
    return result


def _cached_trace_validation_mode(value: Any) -> CachedTraceValidationMode:
    if not isinstance(value, str):
        raise HFCachedAttentionTraceShapeError(
            "cached_trace_validation_mode must be strict or record."
        )
    normalized = value.strip().lower()
    if normalized not in {"strict", "record"}:
        raise HFCachedAttentionTraceShapeError(
            "cached_trace_validation_mode must be strict or record."
        )
    return normalized  # type: ignore[return-value]


def _finite_max_abs_error(left: Tensor, right: Tensor, name: str) -> float:
    """Return one finite diagnostic error without weakening shape contracts."""

    if left.shape != right.shape:
        raise HFCachedAttentionTraceShapeError(
            f"{name} tensors have different shapes."
        )
    difference = left.to(torch.float64) - right.to(torch.float64)
    if not bool(torch.isfinite(difference).all()):
        raise HFCachedAttentionTraceStateError(f"{name} difference is non-finite.")
    return float(difference.abs().max().item())


def _tensor_bytes(value: Tensor) -> int:
    return int(value.numel() * value.element_size())


def _tensor_sha256(value: Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    raw = contiguous.view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def _ids_sha256(value: Tensor) -> str:
    return _tensor_sha256(value.to(dtype=torch.int64))


def _readonly_cpu(value: Tensor, name: str, *, last_sequence: bool = False) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim == 0 or value.numel() <= 0:
        raise HFCachedAttentionTraceShapeError(
            f"Captured {name} must be a non-empty tensor."
        )
    if not bool(torch.isfinite(value.detach()).all()):
        raise HFCachedAttentionTraceShapeError(f"Captured {name} is non-finite.")
    result = value.detach()
    if last_sequence:
        if result.ndim < 2:
            raise HFCachedAttentionTraceShapeError(
                f"Captured {name} has no sequence axis."
            )
        result = result[:, -1:, ...]
    return result.to(device="cpu").contiguous().clone()


@dataclass(frozen=True, slots=True)
class HFCachedAttentionTraceLimits:
    """Hard collector-owned resource and cardinality bounds."""

    max_prompts: int = 4
    max_horizon: int = 32
    max_prompt_tokens: int = 512
    max_layers: int = 8
    max_groups: int = 8
    max_model_devices: int = 1
    max_capture_bytes: int = 256 * 1024 * 1024
    max_projection_bytes: int = 128 * 1024 * 1024
    logit_rtol: float = 5e-3
    logit_atol: float = 5e-3
    context_rtol: float = 8e-3
    context_atol: float = 8e-3
    cached_trace_validation_mode: CachedTraceValidationMode = "strict"

    def __post_init__(self) -> None:
        for name in (
            "max_prompts",
            "max_horizon",
            "max_prompt_tokens",
            "max_layers",
            "max_groups",
            "max_model_devices",
            "max_capture_bytes",
            "max_projection_bytes",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        for name in ("logit_rtol", "logit_atol", "context_rtol", "context_atol"):
            object.__setattr__(self, name, _positive_float(getattr(self, name), name))
        object.__setattr__(
            self,
            "cached_trace_validation_mode",
            _cached_trace_validation_mode(self.cached_trace_validation_mode),
        )


@dataclass(frozen=True, slots=True)
class HFCachedAttentionStep:
    """Detached CPU evidence for one prefill-last or incremental decode action."""

    step_index: int
    prefix_length: int
    selected_token_id: int
    pre_attention_residual: Tensor
    value_projection_input: Tensor
    query: Tensor
    key: Tensor
    value: Tensor
    attention_scores: Tensor
    attention_probabilities: Tensor
    full_attention_probabilities: Tensor
    selected_attention_context: Tensor
    full_attention_context: Tensor
    attention_delta: Tensor
    post_attention_residual: Tensor
    layer_key_cache: Tensor
    layer_value_cache: Tensor
    selected_key_cache: Tensor
    selected_value_cache: Tensor
    action_logits: Tensor
    cached_full_prefix_max_abs_error: float
    selected_context_max_abs_error: float
    value_projection_cache_max_abs_error: float
    captured_byte_count: int
    cached_full_prefix_top1_agreement: bool = True
    cached_full_prefix_numerically_close: bool = True
    value_projection_cache_identity_valid: bool = True
    full_context_reconstruction_valid: bool = True
    selected_context_reconstruction_valid: bool = True
    current_key_cache_append_valid: bool = True
    current_value_cache_append_valid: bool = True
    residual_add_valid: bool = True
    full_context_max_abs_error: float = 0.0
    current_key_cache_append_max_abs_error: float = 0.0
    current_value_cache_append_max_abs_error: float = 0.0
    residual_add_max_abs_error: float = 0.0

    def tensor_sha256s(self) -> Mapping[str, str]:
        values = {
            name: _tensor_sha256(getattr(self, name))
            for name in (
                "pre_attention_residual",
                "value_projection_input",
                "query",
                "key",
                "value",
                "attention_scores",
                "attention_probabilities",
                "full_attention_probabilities",
                "selected_attention_context",
                "full_attention_context",
                "attention_delta",
                "post_attention_residual",
                "layer_key_cache",
                "layer_value_cache",
                "selected_key_cache",
                "selected_value_cache",
                "action_logits",
            )
        }
        return MappingProxyType(values)

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "prefix_length": self.prefix_length,
            "selected_token_id": self.selected_token_id,
            "cached_full_prefix_max_abs_error": (self.cached_full_prefix_max_abs_error),
            "cached_full_prefix_top1_agreement": (
                self.cached_full_prefix_top1_agreement
            ),
            "cached_full_prefix_numerically_close": (
                self.cached_full_prefix_numerically_close
            ),
            "selected_context_max_abs_error": self.selected_context_max_abs_error,
            "value_projection_cache_max_abs_error": (
                self.value_projection_cache_max_abs_error
            ),
            "full_context_max_abs_error": self.full_context_max_abs_error,
            "current_key_cache_append_max_abs_error": (
                self.current_key_cache_append_max_abs_error
            ),
            "current_value_cache_append_max_abs_error": (
                self.current_value_cache_append_max_abs_error
            ),
            "residual_add_max_abs_error": self.residual_add_max_abs_error,
            "finite_validation_checks": {
                "value_projection_cache_identity_valid": (
                    self.value_projection_cache_identity_valid
                ),
                "full_context_reconstruction_valid": (
                    self.full_context_reconstruction_valid
                ),
                "selected_context_reconstruction_valid": (
                    self.selected_context_reconstruction_valid
                ),
                "current_key_cache_append_valid": (
                    self.current_key_cache_append_valid
                ),
                "current_value_cache_append_valid": (
                    self.current_value_cache_append_valid
                ),
                "residual_add_valid": self.residual_add_valid,
            },
            "captured_byte_count": self.captured_byte_count,
            "tensor_sha256s": dict(self.tensor_sha256s()),
            "tensor_shapes": {
                name: list(getattr(self, name).shape) for name in self.tensor_sha256s()
            },
        }


@dataclass(frozen=True, slots=True)
class HFCachedAttentionTrace:
    """One prompt/layer/group lane with certified cached execution."""

    sequence_id: str
    lane: CachedAttentionLane
    layer_index: int
    group_indices: tuple[int, ...]
    prompt_token_ids: tuple[int, ...]
    consumed_token_ids: tuple[int, ...]
    selected_token_ids: tuple[int, ...]
    steps: tuple[HFCachedAttentionStep, ...]
    query_heads: int
    key_value_heads: int
    query_heads_per_group: int
    head_dim: int
    capture_bytes: int
    maximum_cached_full_prefix_error: float
    maximum_selected_context_error: float
    maximum_value_projection_cache_error: float
    value_projection_weight_sha256: str
    value_projection_bias_sha256: str | None
    transformers_version: str | None
    torch_version: str
    model_class: str
    attention_backend: str
    cache_class: str
    logit_rtol: float
    logit_atol: float
    context_rtol: float
    context_atol: float
    attention_source: str
    cached_trace_validation_mode: CachedTraceValidationMode = "strict"

    @property
    def horizon(self) -> int:
        return len(self.steps)

    def as_dict(self) -> dict[str, Any]:
        top1_agreement_count = sum(
            int(step.cached_full_prefix_top1_agreement) for step in self.steps
        )
        checks = {
            "cached_full_prefix_numerically_close": all(
                step.cached_full_prefix_numerically_close for step in self.steps
            ),
            "cached_full_prefix_top1_agreement": all(
                step.cached_full_prefix_top1_agreement for step in self.steps
            ),
            "value_projection_cache_identity_valid": all(
                step.value_projection_cache_identity_valid for step in self.steps
            ),
            "full_context_reconstruction_valid": all(
                step.full_context_reconstruction_valid for step in self.steps
            ),
            "selected_context_reconstruction_valid": all(
                step.selected_context_reconstruction_valid for step in self.steps
            ),
            "current_key_cache_append_valid": all(
                step.current_key_cache_append_valid for step in self.steps
            ),
            "current_value_cache_append_valid": all(
                step.current_value_cache_append_valid for step in self.steps
            ),
            "residual_add_valid": all(step.residual_add_valid for step in self.steps),
        }
        payload = {
            "schema": TRACE_SCHEMA,
            "sequence_id": self.sequence_id,
            "lane": self.lane,
            "layer_index": self.layer_index,
            "group_indices": list(self.group_indices),
            "prompt_length": len(self.prompt_token_ids),
            "prompt_token_ids_sha256": hashlib.sha256(
                bytes(str(self.prompt_token_ids), "ascii")
            ).hexdigest(),
            "consumed_token_ids": list(self.consumed_token_ids),
            "selected_token_ids": list(self.selected_token_ids),
            "horizon": self.horizon,
            "query_heads": self.query_heads,
            "key_value_heads": self.key_value_heads,
            "query_heads_per_group": self.query_heads_per_group,
            "head_dim": self.head_dim,
            "capture_bytes": self.capture_bytes,
            "maximum_cached_full_prefix_error": (self.maximum_cached_full_prefix_error),
            "maximum_selected_context_error": self.maximum_selected_context_error,
            "maximum_value_projection_cache_error": (
                self.maximum_value_projection_cache_error
            ),
            "maximum_full_context_error": max(
                step.full_context_max_abs_error for step in self.steps
            ),
            "maximum_current_key_cache_append_error": max(
                step.current_key_cache_append_max_abs_error for step in self.steps
            ),
            "maximum_current_value_cache_append_error": max(
                step.current_value_cache_append_max_abs_error for step in self.steps
            ),
            "maximum_residual_add_error": max(
                step.residual_add_max_abs_error for step in self.steps
            ),
            "value_projection_weight_sha256": self.value_projection_weight_sha256,
            "value_projection_bias_sha256": self.value_projection_bias_sha256,
            "transformers_version": self.transformers_version,
            "torch_version": self.torch_version,
            "model_class": self.model_class,
            "attention_backend": self.attention_backend,
            "cache_class": self.cache_class,
            "logit_rtol": self.logit_rtol,
            "logit_atol": self.logit_atol,
            "context_rtol": self.context_rtol,
            "context_atol": self.context_atol,
            "cached_trace_validation_mode": self.cached_trace_validation_mode,
            "finite_validation": {
                "mode": self.cached_trace_validation_mode,
                "strict_numeric_enforcement": (
                    self.cached_trace_validation_mode == "strict"
                ),
                "finite_mismatches_are_diagnostic_only": (
                    self.cached_trace_validation_mode == "record"
                ),
                "top1_agreement_count": top1_agreement_count,
                "top1_step_count": len(self.steps),
                "top1_agreement_rate": top1_agreement_count / len(self.steps),
                "checks": checks,
            },
            "attention_source": self.attention_source,
            "attention_backend_preserved": True,
            "cache_layout": "layers[layer_index].keys-values-[batch,kv_head,sequence,head_dim]",
            "full_prefix_cache_equivalence_verified": bool(
                checks["cached_full_prefix_numerically_close"]
                and checks["cached_full_prefix_top1_agreement"]
            ),
            "exact_layer_cache_captured": True,
            "exact_value_projection_input_history_captured": True,
            "value_projection_cache_identity_verified": checks[
                "value_projection_cache_identity_valid"
            ],
            "attention_context_reconstruction_verified": bool(
                checks["full_context_reconstruction_valid"]
                and checks["selected_context_reconstruction_valid"]
            ),
            "current_kv_cache_append_verified": bool(
                checks["current_key_cache_append_valid"]
                and checks["current_value_cache_append_valid"]
            ),
            "residual_add_verified": checks["residual_add_valid"],
            "value_projection_identity": (
                "V_cache = concat(U_prefill,U_cached) @ W_v.T + bias"
            ),
            "post_rope_score_geometry": True,
            "gradient_free": True,
            "training_used": False,
            "steps": [step.as_dict() for step in self.steps],
        }
        payload["trace_sha256"] = hashlib.sha256(
            repr(payload).encode("utf-8")
        ).hexdigest()
        return payload


def _model_devices(model: nn.Module) -> tuple[torch.device, ...]:
    devices = {
        value.device
        for value in tuple(model.parameters()) + tuple(model.buffers())
        if value.device.type != "meta"
    }
    return tuple(sorted(devices, key=str))


def _base_layers(model: nn.Module) -> Sequence[nn.Module]:
    candidates = (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "model", None), "layers", None),
        getattr(model, "layers", None),
    )
    for value in candidates:
        if isinstance(value, Sequence) or isinstance(value, nn.ModuleList):
            return value
    raise HFCachedAttentionTraceShapeError(
        "Model must expose decoder layers at model.layers."
    )


def _embedding_device(model: nn.Module) -> torch.device:
    getter = getattr(model, "get_input_embeddings", None)
    if callable(getter):
        embedding = getter()
        weight = getattr(embedding, "weight", None)
        if isinstance(weight, Tensor):
            return weight.device
    for parameter in model.parameters():
        return parameter.device
    raise HFCachedAttentionTraceShapeError("Model has no parameter device.")


def _module_input(
    args: tuple[Any, ...], kwargs: Mapping[str, Any], name: str
) -> Tensor:
    if args and isinstance(args[0], Tensor):
        return args[0]
    for key in ("hidden_states", "x", "input"):
        value = kwargs.get(key)
        if isinstance(value, Tensor):
            return value
    raise HFCachedAttentionTraceStateError(
        f"Hook {name} did not receive tensor hidden states."
    )


def _tensor_output(output: Any, name: str) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    raise HFCachedAttentionTraceStateError(
        f"Hook {name} did not return a leading tensor."
    )


def _linear_projection_parts(
    projection: nn.Module,
    *,
    expected_output_width: int,
    max_projection_bytes: int,
) -> tuple[Tensor, Tensor | None]:
    """Detach one exact linear-compatible projection under a hard byte bound."""

    weight = getattr(projection, "weight", None)
    bias = getattr(projection, "bias", None)
    if not isinstance(weight, Tensor) or weight.ndim != 2:
        raise HFCachedAttentionTraceShapeError(
            "v_proj must expose one rank-two linear-compatible weight."
        )
    if int(weight.shape[0]) != expected_output_width:
        raise HFCachedAttentionTraceShapeError(
            "v_proj output width does not match the Qwen KV topology."
        )
    if bias is not None and (
        not isinstance(bias, Tensor) or tuple(bias.shape) != (expected_output_width,)
    ):
        raise HFCachedAttentionTraceShapeError(
            "v_proj bias must be absent or match its output width."
        )
    detached_weight = _readonly_cpu(weight, "value projection weight")
    detached_bias = (
        None if bias is None else _readonly_cpu(bias, "value projection bias")
    )
    projection_bytes = _tensor_bytes(detached_weight) + (
        0 if detached_bias is None else _tensor_bytes(detached_bias)
    )
    if projection_bytes > max_projection_bytes:
        raise HFCachedAttentionTraceLimitError(
            "v_proj exceeds the declared max_projection_bytes workspace bound."
        )
    return detached_weight, detached_bias


class _StepCapture:
    def __init__(self) -> None:
        self.active = False
        self.values: dict[str, Tensor | tuple[Tensor, Tensor] | None] = {}
        self.counts: dict[str, int] = {}

    def begin(self) -> None:
        if self.active:
            raise HFCachedAttentionTraceStateError("A capture is already active.")
        self.active = True
        self.values = {}
        self.counts = {}

    def suspend(self) -> None:
        self.active = False

    def record(self, name: str, value: Any) -> None:
        if not self.active:
            return
        count = self.counts.get(name, 0) + 1
        self.counts[name] = count
        if count != 1:
            raise HFCachedAttentionTraceStateError(
                f"Cached trace hook {name!r} fired {count} times."
            )
        self.values[name] = value

    def finish(self) -> Mapping[str, Any]:
        required = {
            "pre_residual",
            "position_embeddings",
            "attention_mask",
            "query_norm",
            "key_norm",
            "value_projection_input",
            "value",
            "output_context",
            "attention_delta",
            "post_residual",
        }
        missing = required - set(self.values)
        self.active = False
        if missing:
            raise HFCachedAttentionTraceStateError(
                f"Cached attention hooks did not capture {sorted(missing)}."
            )
        return MappingProxyType(dict(self.values))


class _HookSet:
    def __init__(self, layer: nn.Module, attention: nn.Module) -> None:
        self.capture = _StepCapture()
        self.handles: list[Any] = []
        self._install(layer, attention)

    def _install(self, layer: nn.Module, attention: nn.Module) -> None:
        q_norm = getattr(attention, "q_norm", None)
        k_norm = getattr(attention, "k_norm", None)
        v_proj = getattr(attention, "v_proj", None)
        o_proj = getattr(attention, "o_proj", None)
        post_norm = getattr(layer, "post_attention_layernorm", None)
        modules = {
            "q_norm": q_norm,
            "k_norm": k_norm,
            "v_proj": v_proj,
            "o_proj": o_proj,
            "post_attention_layernorm": post_norm,
        }
        missing = [
            name for name, value in modules.items() if not isinstance(value, nn.Module)
        ]
        if missing:
            raise HFCachedAttentionTraceShapeError(
                f"Selected decoder layer lacks Qwen attention modules {missing}."
            )

        def layer_pre(
            _module: nn.Module,
            args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> None:
            hidden = _module_input(args, kwargs, "decoder layer")
            self.capture.record(
                "pre_residual",
                _readonly_cpu(hidden, "pre-attention residual", last_sequence=True),
            )

        def attention_pre(
            _module: nn.Module,
            _args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> None:
            position = kwargs.get("position_embeddings")
            if (
                not isinstance(position, (tuple, list))
                or len(position) != 2
                or not all(isinstance(value, Tensor) for value in position)
            ):
                raise HFCachedAttentionTraceStateError(
                    "Attention did not expose rotary position embeddings."
                )
            cos, sin = position
            self.capture.record(
                "position_embeddings",
                (
                    _readonly_cpu(cos, "rotary cosine", last_sequence=True),
                    _readonly_cpu(sin, "rotary sine", last_sequence=True),
                ),
            )
            mask = kwargs.get("attention_mask")
            if mask is None:
                self.capture.record("attention_mask", None)
            elif isinstance(mask, Tensor):
                if mask.ndim < 2:
                    raise HFCachedAttentionTraceShapeError(
                        "Attention mask has no query/key axes."
                    )
                self.capture.record(
                    "attention_mask",
                    _readonly_cpu(mask[..., -1:, :], "attention mask"),
                )
            else:
                raise HFCachedAttentionTraceStateError(
                    "Selected attention received a non-tensor mask."
                )

        def output_hook(name: str, *, last_sequence: bool = True):
            def callback(
                _module: nn.Module,
                _args: tuple[Any, ...],
                _kwargs: Mapping[str, Any],
                output: Any,
            ) -> None:
                self.capture.record(
                    name,
                    _readonly_cpu(
                        _tensor_output(output, name),
                        name,
                        last_sequence=last_sequence,
                    ),
                )

            return callback

        def value_hook(
            _module: nn.Module,
            args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
            output: Any,
        ) -> None:
            # This is the exact tensor consumed by v_proj.  On the prefill it
            # contains the complete prompt; subsequent cached forwards append
            # one row.  Keeping the chunks (rather than duplicating every
            # cumulative prefix) makes the retained history bounded.
            self.capture.record(
                "value_projection_input",
                _readonly_cpu(
                    _module_input(args, kwargs, "v_proj"),
                    "value projection input",
                ),
            )
            self.capture.record(
                "value",
                _readonly_cpu(
                    _tensor_output(output, "value"),
                    "value",
                    last_sequence=True,
                ),
            )

        def o_hook(
            _module: nn.Module,
            args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
            output: Any,
        ) -> None:
            self.capture.record(
                "output_context",
                _readonly_cpu(
                    _module_input(args, kwargs, "o_proj"),
                    "pre-O attention context",
                    last_sequence=True,
                ),
            )
            self.capture.record(
                "attention_delta",
                _readonly_cpu(
                    _tensor_output(output, "o_proj"),
                    "attention delta",
                    last_sequence=True,
                ),
            )

        def post_norm_pre(
            _module: nn.Module,
            args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> None:
            self.capture.record(
                "post_residual",
                _readonly_cpu(
                    _module_input(args, kwargs, "post-attention norm"),
                    "post-attention residual",
                    last_sequence=True,
                ),
            )

        self.handles.append(
            layer.register_forward_pre_hook(layer_pre, with_kwargs=True)
        )
        self.handles.append(
            attention.register_forward_pre_hook(attention_pre, with_kwargs=True)
        )
        self.handles.append(
            q_norm.register_forward_hook(output_hook("query_norm"), with_kwargs=True)
        )
        self.handles.append(
            k_norm.register_forward_hook(output_hook("key_norm"), with_kwargs=True)
        )
        self.handles.append(v_proj.register_forward_hook(value_hook, with_kwargs=True))
        self.handles.append(o_proj.register_forward_hook(o_hook, with_kwargs=True))
        self.handles.append(
            post_norm.register_forward_pre_hook(post_norm_pre, with_kwargs=True)
        )

    def remove(self) -> None:
        self.capture.suspend()
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()


def _rotate_half(value: Tensor) -> Tensor:
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def _apply_rope(value: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return value * cos.unsqueeze(1) + _rotate_half(value) * sin.unsqueeze(1)


def _cache_layer(cache: Any, layer_index: int) -> tuple[Tensor, Tensor]:
    layers = getattr(cache, "layers", None)
    if isinstance(layers, Sequence) and layer_index < len(layers):
        layer = layers[layer_index]
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if isinstance(keys, Tensor) and isinstance(values, Tensor):
            return keys, values
    if isinstance(cache, (tuple, list)) and layer_index < len(cache):
        layer = cache[layer_index]
        if (
            isinstance(layer, (tuple, list))
            and len(layer) >= 2
            and isinstance(layer[0], Tensor)
            and isinstance(layer[1], Tensor)
        ):
            return layer[0], layer[1]
    raise HFCachedAttentionTraceStateError(
        "past_key_values does not expose exact layer keys/values."
    )


def _output_field(output: Any, name: str) -> Any:
    value = getattr(output, name, None)
    if value is None and isinstance(output, Mapping):
        value = output.get(name)
    return value


def _output_logits(output: Any) -> Tensor:
    logits = _output_field(output, "logits")
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise HFCachedAttentionTraceStateError(
            "Causal model output must expose [batch, sequence, vocab] logits."
        )
    if not bool(torch.isfinite(logits.detach()).all()):
        raise HFCachedAttentionTraceStateError("Model logits are non-finite.")
    return logits


def _qualified_class_name(value: Any) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _attention_backend(config: Any) -> str:
    return str(getattr(config, "_attn_implementation", None) or "eager").strip()


def _forward_kwargs(model: nn.Module, values: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return values
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return values
    return {
        name: value for name, value in values.items() if name in signature.parameters
    }


def _model_forward(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    use_cache: bool,
    past_key_values: Any = None,
) -> Any:
    values = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": use_cache,
        "past_key_values": past_key_values,
        "return_dict": True,
        "output_attentions": False,
        "output_hidden_states": False,
        "logits_to_keep": 1,
    }
    return model(**_forward_kwargs(model, values))


def _groups(
    raw: Sequence[Any], key_value_heads: int, limits: HFCachedAttentionTraceLimits
) -> tuple[int, ...]:
    values = tuple(_nonnegative_int(value, "group index") for value in raw)
    if not values or len(values) > limits.max_groups:
        raise HFCachedAttentionTraceLimitError(
            f"Select between 1 and {limits.max_groups} attention groups."
        )
    if len(set(values)) != len(values) or any(
        value >= key_value_heads for value in values
    ):
        raise HFCachedAttentionTraceShapeError(
            "Attention group indices must be unique in-range KV heads."
        )
    return tuple(sorted(values))


def _build_step(
    captures: Mapping[str, Any],
    *,
    output: Any,
    layer_index: int,
    group_indices: tuple[int, ...],
    query_heads: int,
    key_value_heads: int,
    head_dim: int,
    scaling: float,
    step_index: int,
    prefix_length: int,
    selected_token_id: int,
    full_logits: Tensor,
    value_projection_input_history: Tensor,
    value_projection_weight: Tensor,
    value_projection_bias: Tensor | None,
    limits: HFCachedAttentionTraceLimits,
) -> HFCachedAttentionStep:
    query_heads_per_group = query_heads // key_value_heads
    cache = _output_field(output, "past_key_values")
    if cache is None:
        raise HFCachedAttentionTraceStateError(
            "Model returned no past_key_values under use_cache=True."
        )
    key_cache_raw, value_cache_raw = _cache_layer(cache, layer_index)
    key_cache = _readonly_cpu(key_cache_raw, "layer key cache")
    value_cache = _readonly_cpu(value_cache_raw, "layer value cache")
    expected_cache_prefix = (1, key_value_heads, prefix_length, head_dim)
    if (
        tuple(key_cache.shape) != expected_cache_prefix
        or tuple(value_cache.shape) != expected_cache_prefix
    ):
        raise HFCachedAttentionTraceShapeError(
            "Exact layer cache shape differs from "
            f"{expected_cache_prefix}: K={tuple(key_cache.shape)}, "
            f"V={tuple(value_cache.shape)}."
        )

    value_input_chunk = captures["value_projection_input"]
    if (
        not isinstance(value_input_chunk, Tensor)
        or value_input_chunk.ndim != 3
        or int(value_input_chunk.shape[0]) != 1
        or int(value_input_chunk.shape[2]) != int(value_projection_weight.shape[1])
    ):
        raise HFCachedAttentionTraceShapeError(
            "Exact v_proj input must be [1, current_sequence, input_width]."
        )
    if (
        value_projection_input_history.ndim != 3
        or tuple(value_projection_input_history.shape[:2]) != (1, prefix_length)
        or int(value_projection_input_history.shape[2])
        != int(value_projection_weight.shape[1])
    ):
        raise HFCachedAttentionTraceShapeError(
            "Concatenated v_proj input history does not match the cache prefix."
        )
    projected_value_history = torch.nn.functional.linear(
        value_projection_input_history.to(torch.float64),
        value_projection_weight.to(torch.float64),
        (
            None
            if value_projection_bias is None
            else value_projection_bias.to(torch.float64)
        ),
    )
    projected_value_cache = projected_value_history.view(
        1, prefix_length, key_value_heads, head_dim
    ).transpose(1, 2)
    value_cache_for_projection = value_cache.to(dtype=projected_value_cache.dtype)
    value_projection_cache_error = _finite_max_abs_error(
        projected_value_cache,
        value_cache_for_projection,
        "Value projection/cache identity",
    )
    value_projection_cache_identity_valid = bool(
        torch.allclose(
            projected_value_cache,
            value_cache_for_projection,
            rtol=limits.context_rtol,
            atol=limits.context_atol,
        )
    )
    if (
        not value_projection_cache_identity_valid
        and limits.cached_trace_validation_mode == "strict"
    ):
        raise HFCachedAttentionTraceStateError(
            "The exact v_proj input history multiplied by the materialized V "
            "weight does not reproduce the post-update value cache "
            f"(maximum error {value_projection_cache_error:.6g})."
        )

    q_norm = captures["query_norm"]
    k_norm = captures["key_norm"]
    value_projection = captures["value"]
    expected_q = (1, 1, query_heads, head_dim)
    expected_kv = (1, 1, key_value_heads, head_dim)
    if tuple(q_norm.shape) != expected_q or tuple(k_norm.shape) != expected_kv:
        raise HFCachedAttentionTraceShapeError(
            "Post-Q/K-normalization tensors have unexpected Qwen head shape."
        )
    if tuple(value_projection.shape) != (1, 1, key_value_heads * head_dim):
        raise HFCachedAttentionTraceShapeError(
            "V projection has an unexpected Qwen head shape."
        )
    cos, sin = captures["position_embeddings"]
    if tuple(cos.shape) != (1, 1, head_dim) or tuple(sin.shape) != (
        1,
        1,
        head_dim,
    ):
        raise HFCachedAttentionTraceShapeError(
            "Rotary position tensors do not match the Qwen head dimension."
        )
    query_all = _apply_rope(
        q_norm.transpose(1, 2).to(torch.float32),
        cos.to(torch.float32),
        sin.to(torch.float32),
    )
    key_current_all = _apply_rope(
        k_norm.transpose(1, 2).to(torch.float32),
        cos.to(torch.float32),
        sin.to(torch.float32),
    )
    value_current_all = value_projection.view(
        1, 1, key_value_heads, head_dim
    ).transpose(1, 2)

    group_index_tensor = torch.tensor(group_indices, dtype=torch.long)
    query_index_tensor = torch.tensor(
        [
            group * query_heads_per_group + local
            for group in group_indices
            for local in range(query_heads_per_group)
        ],
        dtype=torch.long,
    )
    query = query_all.index_select(1, query_index_tensor)
    key = key_current_all.index_select(1, group_index_tensor)
    value = value_current_all.index_select(1, group_index_tensor)
    selected_key_cache = key_cache.index_select(1, group_index_tensor)
    selected_value_cache = value_cache.index_select(1, group_index_tensor)
    repeated_full_key_cache = key_cache.to(torch.float32).repeat_interleave(
        query_heads_per_group, dim=1
    )
    full_scores = (
        torch.matmul(
            query_all.to(torch.float32),
            repeated_full_key_cache.transpose(-2, -1),
        )
        * scaling
    )
    mask = captures["attention_mask"]
    if mask is not None:
        if mask.shape[-1] != prefix_length:
            raise HFCachedAttentionTraceShapeError(
                "Attention mask key length does not match the exact layer cache."
            )
        full_scores = full_scores + mask.to(torch.float32)
    if not bool(torch.isfinite(full_scores).all()):
        raise HFCachedAttentionTraceStateError(
            "Post-RoPE attention scores are non-finite."
        )
    full_probabilities = torch.softmax(full_scores, dim=-1, dtype=torch.float32)
    scores = full_scores.index_select(1, query_index_tensor)
    probabilities = full_probabilities.index_select(1, query_index_tensor)
    repeated_full_value_cache = value_cache.to(torch.float32).repeat_interleave(
        query_heads_per_group, dim=1
    )
    reconstructed_full_context = torch.matmul(
        full_probabilities, repeated_full_value_cache
    )
    reconstructed_context = reconstructed_full_context.index_select(
        1, query_index_tensor
    )

    output_context = captures["output_context"]
    if tuple(output_context.shape) != (1, 1, query_heads * head_dim):
        raise HFCachedAttentionTraceShapeError(
            "Pre-O context does not match the full Qwen query width."
        )
    output_context_heads = output_context.view(1, 1, query_heads, head_dim).transpose(
        1, 2
    )
    selected_context = output_context_heads.index_select(1, query_index_tensor)
    selected_context_float = selected_context.to(torch.float32)
    output_context_heads_float = output_context_heads.to(torch.float32)
    context_difference = _finite_max_abs_error(
        selected_context_float,
        reconstructed_context,
        "Selected attention context reconstruction",
    )
    full_context_difference = _finite_max_abs_error(
        output_context_heads_float,
        reconstructed_full_context,
        "Full attention context reconstruction",
    )
    full_context_reconstruction_valid = bool(
        torch.allclose(
            output_context_heads_float,
            reconstructed_full_context,
            rtol=limits.context_rtol,
            atol=limits.context_atol,
        )
    )
    if (
        not full_context_reconstruction_valid
        and limits.cached_trace_validation_mode == "strict"
    ):
        raise HFCachedAttentionTraceStateError(
            "Reconstructed full post-RoPE attention does not reproduce the real "
            f"pre-O context (maximum error {full_context_difference:.6g})."
        )
    selected_context_reconstruction_valid = bool(
        torch.allclose(
            selected_context_float,
            reconstructed_context,
            rtol=limits.context_rtol,
            atol=limits.context_atol,
        )
    )
    if (
        not selected_context_reconstruction_valid
        and limits.cached_trace_validation_mode == "strict"
    ):
        raise HFCachedAttentionTraceStateError(
            "Reconstructed post-RoPE attention does not reproduce the real pre-O "
            f"context (maximum error {context_difference:.6g})."
        )
    current_key = key.to(torch.float32)
    appended_key = selected_key_cache[..., -1:, :].to(torch.float32)
    current_value = value.to(torch.float32)
    appended_value = selected_value_cache[..., -1:, :].to(torch.float32)
    current_key_cache_append_error = _finite_max_abs_error(
        current_key,
        appended_key,
        "Current key/cache append",
    )
    current_value_cache_append_error = _finite_max_abs_error(
        current_value,
        appended_value,
        "Current value/cache append",
    )
    current_key_cache_append_valid = bool(
        torch.allclose(
            current_key,
            appended_key,
            rtol=limits.context_rtol,
            atol=limits.context_atol,
        )
    )
    current_value_cache_append_valid = bool(
        torch.allclose(
            current_value,
            appended_value,
            rtol=limits.context_rtol,
            atol=limits.context_atol,
        )
    )
    if (
        not (current_key_cache_append_valid and current_value_cache_append_valid)
        and limits.cached_trace_validation_mode == "strict"
    ):
        raise HFCachedAttentionTraceStateError(
            "Current post-RoPE K/V does not equal the appended exact cache entry."
        )

    logits = _readonly_cpu(_output_logits(output)[:, -1, :], "cached logits")
    full_last = _readonly_cpu(full_logits[:, -1, :], "full-prefix logits")
    if logits.shape != full_last.shape:
        raise HFCachedAttentionTraceShapeError(
            "Cached and full-prefix logits have different shapes."
        )
    logits_float = logits.to(torch.float32)
    full_last_float = full_last.to(torch.float32)
    logit_difference = _finite_max_abs_error(
        logits_float,
        full_last_float,
        "Cached/full-prefix logits",
    )
    cached_actions = torch.argmax(logits, dim=-1)
    full_actions = torch.argmax(full_last, dim=-1)
    cached_full_prefix_top1_agreement = bool(
        torch.equal(cached_actions, full_actions)
    )
    if (
        not cached_full_prefix_top1_agreement
        and limits.cached_trace_validation_mode == "strict"
    ):
        raise HFCachedAttentionTraceStateError(
            "Cached logits differ from full-prefix logits in their top-1 action."
        )
    cached_full_prefix_numerically_close = bool(
        torch.allclose(
            logits_float,
            full_last_float,
            rtol=limits.logit_rtol,
            atol=limits.logit_atol,
        )
    )
    if (
        not cached_full_prefix_numerically_close
        and limits.cached_trace_validation_mode == "strict"
    ):
        raise HFCachedAttentionTraceStateError(
            "Cached logits differ from a full-prefix use_cache=False forward "
            f"(maximum error {logit_difference:.6g})."
        )
    pre_residual = captures["pre_residual"]
    attention_delta = captures["attention_delta"]
    post_residual = captures["post_residual"]
    if pre_residual.shape != attention_delta.shape or pre_residual.shape != (
        post_residual.shape
    ):
        raise HFCachedAttentionTraceShapeError(
            "Residual and O-delta captures have incompatible shapes."
        )
    expected_post_residual = (
        pre_residual.to(torch.float32) + attention_delta.to(torch.float32)
    )
    post_residual_float = post_residual.to(torch.float32)
    residual_add_error = _finite_max_abs_error(
        post_residual_float,
        expected_post_residual,
        "Attention residual add",
    )
    residual_add_valid = bool(
        torch.allclose(
            post_residual_float,
            expected_post_residual,
            rtol=limits.context_rtol,
            atol=limits.context_atol,
        )
    )
    if not residual_add_valid and limits.cached_trace_validation_mode == "strict":
        raise HFCachedAttentionTraceStateError(
            "Captured post-attention residual is not residual + O delta."
        )
    tensors = (
        pre_residual,
        value_input_chunk,
        query,
        key,
        value,
        scores,
        probabilities,
        full_probabilities,
        selected_context,
        output_context[:, -1, :],
        attention_delta[:, -1, :],
        post_residual[:, -1, :],
        key_cache,
        value_cache,
        selected_key_cache,
        selected_value_cache,
        logits,
    )
    captured_bytes = sum(_tensor_bytes(value) for value in tensors)
    return HFCachedAttentionStep(
        step_index=step_index,
        prefix_length=prefix_length,
        selected_token_id=selected_token_id,
        pre_attention_residual=pre_residual[:, -1, :],
        value_projection_input=value_input_chunk.contiguous(),
        query=query.contiguous(),
        key=key.contiguous(),
        value=value.contiguous(),
        attention_scores=scores.contiguous(),
        attention_probabilities=probabilities.contiguous(),
        full_attention_probabilities=full_probabilities.contiguous(),
        selected_attention_context=selected_context.reshape(1, -1).contiguous(),
        full_attention_context=output_context[:, -1, :].contiguous(),
        attention_delta=attention_delta[:, -1, :].contiguous(),
        post_attention_residual=post_residual[:, -1, :].contiguous(),
        layer_key_cache=key_cache,
        layer_value_cache=value_cache,
        selected_key_cache=selected_key_cache,
        selected_value_cache=selected_value_cache,
        action_logits=logits,
        cached_full_prefix_max_abs_error=logit_difference,
        selected_context_max_abs_error=context_difference,
        value_projection_cache_max_abs_error=value_projection_cache_error,
        captured_byte_count=captured_bytes,
        cached_full_prefix_top1_agreement=cached_full_prefix_top1_agreement,
        cached_full_prefix_numerically_close=(
            cached_full_prefix_numerically_close
        ),
        value_projection_cache_identity_valid=(
            value_projection_cache_identity_valid
        ),
        full_context_reconstruction_valid=full_context_reconstruction_valid,
        selected_context_reconstruction_valid=(
            selected_context_reconstruction_valid
        ),
        current_key_cache_append_valid=current_key_cache_append_valid,
        current_value_cache_append_valid=current_value_cache_append_valid,
        residual_add_valid=residual_add_valid,
        full_context_max_abs_error=full_context_difference,
        current_key_cache_append_max_abs_error=current_key_cache_append_error,
        current_value_cache_append_max_abs_error=current_value_cache_append_error,
        residual_add_max_abs_error=residual_add_error,
    )


def collect_hf_cached_attention_trace(
    model: nn.Module,
    input_ids: Any,
    *,
    layer_index: int,
    group_indices: Sequence[int],
    horizon: int,
    sequence_id: str = "prompt-0",
    lane: CachedAttentionLane = "greedy",
    forced_token_ids: Sequence[int] | Tensor | None = None,
    limits: HFCachedAttentionTraceLimits | None = None,
) -> HFCachedAttentionTrace:
    """Run one bounded prefill plus incremental cached pseudo-decode lane."""

    if not isinstance(model, nn.Module):
        raise HFCachedAttentionTraceShapeError("model must be a torch nn.Module.")
    controls = limits or HFCachedAttentionTraceLimits()
    if not isinstance(controls, HFCachedAttentionTraceLimits):
        raise HFCachedAttentionTraceShapeError(
            "limits must be HFCachedAttentionTraceLimits."
        )
    horizon_value = _positive_int(horizon, "horizon")
    if horizon_value > controls.max_horizon:
        raise HFCachedAttentionTraceLimitError(
            f"horizon exceeds max_horizon={controls.max_horizon}."
        )
    layer_value = _nonnegative_int(layer_index, "layer_index")
    layers = _base_layers(model)
    if layer_value >= len(layers):
        raise HFCachedAttentionTraceLimitError(
            "layer_index is unavailable in the decoder."
        )
    layer = layers[layer_value]
    attention = getattr(layer, "self_attn", None)
    if not isinstance(attention, nn.Module):
        raise HFCachedAttentionTraceShapeError(
            "Selected layer has no self_attn module."
        )
    config = getattr(model, "config", getattr(attention, "config", None))
    query_heads = int(
        getattr(config, "num_attention_heads", getattr(attention, "num_heads", 0))
    )
    key_value_heads = int(
        getattr(
            config,
            "num_key_value_heads",
            getattr(attention, "num_key_value_heads", 0),
        )
    )
    head_dim = int(
        getattr(
            config,
            "head_dim",
            getattr(attention, "head_dim", 0),
        )
    )
    if (
        min(query_heads, key_value_heads, head_dim) <= 0
        or query_heads % key_value_heads
    ):
        raise HFCachedAttentionTraceShapeError(
            "Selected attention lacks a valid Qwen GQA topology."
        )
    selected_groups = _groups(group_indices, key_value_heads, controls)
    scaling = float(getattr(attention, "scaling", head_dim**-0.5))
    if not math.isfinite(scaling) or scaling <= 0.0:
        raise HFCachedAttentionTraceShapeError(
            "Selected attention has an invalid score scaling."
        )
    devices = _model_devices(model)
    if not devices or len(devices) > controls.max_model_devices:
        raise HFCachedAttentionTraceLimitError(
            "Model device count exceeds max_model_devices or is empty."
        )
    if bool(getattr(model, "is_gradient_checkpointing", False)) or bool(
        getattr(config, "gradient_checkpointing", False)
    ):
        raise HFCachedAttentionTraceStateError(
            "Gradient checkpointing must be disabled for cached tracing."
        )
    value_projection_module = getattr(attention, "v_proj", None)
    if not isinstance(value_projection_module, nn.Module):
        raise HFCachedAttentionTraceShapeError(
            "Selected attention has no linear-compatible v_proj module."
        )
    value_projection_weight, value_projection_bias = _linear_projection_parts(
        value_projection_module,
        expected_output_width=key_value_heads * head_dim,
        max_projection_bytes=controls.max_projection_bytes,
    )
    value_projection_weight_hash = _tensor_sha256(value_projection_weight)
    value_projection_bias_hash = (
        None
        if value_projection_bias is None
        else _tensor_sha256(value_projection_bias)
    )
    device = _embedding_device(model)
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
        raise HFCachedAttentionTraceShapeError(
            "input_ids must be one non-empty integer prompt within max_prompt_tokens."
        )
    ids = ids.to(dtype=torch.long)
    normalized_lane = str(lane).strip().lower()
    if normalized_lane not in {"forced", "greedy"}:
        raise HFCachedAttentionTraceShapeError("lane must be forced or greedy.")
    forced: tuple[int, ...] | None
    if normalized_lane == "forced":
        if forced_token_ids is None:
            raise HFCachedAttentionTraceShapeError(
                "forced_token_ids is required for a forced lane."
            )
        forced_tensor = torch.as_tensor(forced_token_ids, dtype=torch.long).reshape(-1)
        if int(forced_tensor.numel()) < horizon_value:
            raise HFCachedAttentionTraceShapeError(
                "forced_token_ids must contain at least horizon tokens."
            )
        forced = tuple(int(value) for value in forced_tensor[:horizon_value].tolist())
    else:
        if forced_token_ids is not None:
            raise HFCachedAttentionTraceShapeError(
                "forced_token_ids is valid only for a forced lane."
            )
        forced = None
    sequence_name = str(sequence_id).strip()
    if not sequence_name:
        raise HFCachedAttentionTraceShapeError("sequence_id cannot be blank.")

    module_modes = [(module, bool(module.training)) for module in model.modules()]
    config_values: dict[str, Any] = {}
    for name in ("use_cache", "output_attentions", "output_hidden_states"):
        if config is not None and hasattr(config, name):
            config_values[name] = getattr(config, name)
    original_attention_backend = _attention_backend(config)
    hooks = _HookSet(layer, attention)
    steps: list[HFCachedAttentionStep] = []
    selected_tokens: list[int] = []
    consumed_tokens: list[int] = []
    prefix = ids.clone()
    past = None
    next_input = ids
    capture_bytes = 0
    cache_class: str | None = None
    value_projection_input_history: Tensor | None = None
    try:
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
                    (1, int(prefix.shape[1])),
                    dtype=torch.long,
                    device=device,
                )
                hooks.capture.begin()
                output = _model_forward(
                    model,
                    next_input,
                    attention_mask,
                    use_cache=True,
                    past_key_values=past,
                )
                captures = hooks.capture.finish()
                current_value_projection_weight, current_value_projection_bias = (
                    _linear_projection_parts(
                        value_projection_module,
                        expected_output_width=key_value_heads * head_dim,
                        max_projection_bytes=controls.max_projection_bytes,
                    )
                )
                if _tensor_sha256(current_value_projection_weight) != (
                    value_projection_weight_hash
                ) or (
                    None
                    if current_value_projection_bias is None
                    else _tensor_sha256(current_value_projection_bias)
                ) != value_projection_bias_hash:
                    raise HFCachedAttentionTraceStateError(
                        "v_proj changed while one cached trace was being collected."
                    )
                value_input_chunk = captures["value_projection_input"]
                if not isinstance(value_input_chunk, Tensor):
                    raise HFCachedAttentionTraceStateError(
                        "v_proj input capture is not a tensor."
                    )
                value_projection_input_history = (
                    value_input_chunk
                    if value_projection_input_history is None
                    else torch.cat(
                        (value_projection_input_history, value_input_chunk), dim=1
                    )
                )
                if int(value_projection_input_history.shape[1]) != int(
                    prefix.shape[1]
                ):
                    raise HFCachedAttentionTraceStateError(
                        "v_proj input chunks do not concatenate to the cache prefix."
                    )
                logits = _output_logits(output)
                if forced is not None:
                    selected = forced[step_index]
                    if selected < 0 or selected >= int(logits.shape[-1]):
                        raise HFCachedAttentionTraceShapeError(
                            "forced_token_ids contains an out-of-vocabulary token."
                        )
                else:
                    selected = int(torch.argmax(logits[:, -1, :], dim=-1).item())

                hooks.capture.suspend()
                full_output = _model_forward(
                    model,
                    prefix,
                    attention_mask,
                    use_cache=False,
                    past_key_values=None,
                )
                if _output_field(full_output, "past_key_values") is not None:
                    raise HFCachedAttentionTraceStateError(
                        "Full-prefix verification returned a cache under use_cache=False."
                    )
                step = _build_step(
                    captures,
                    output=output,
                    layer_index=layer_value,
                    group_indices=selected_groups,
                    query_heads=query_heads,
                    key_value_heads=key_value_heads,
                    head_dim=head_dim,
                    scaling=scaling,
                    step_index=step_index,
                    prefix_length=int(prefix.shape[1]),
                    selected_token_id=selected,
                    full_logits=_output_logits(full_output),
                    value_projection_input_history=(
                        value_projection_input_history
                    ),
                    value_projection_weight=value_projection_weight,
                    value_projection_bias=value_projection_bias,
                    limits=controls,
                )
                steps.append(step)
                capture_bytes += step.captured_byte_count
                if capture_bytes > controls.max_capture_bytes:
                    raise HFCachedAttentionTraceLimitError(
                        "Cached attention trace exceeded max_capture_bytes."
                    )
                selected_tokens.append(selected)
                past = _output_field(output, "past_key_values")
                current_cache_class = _qualified_class_name(past)
                if cache_class is None:
                    cache_class = current_cache_class
                elif cache_class != current_cache_class:
                    raise HFCachedAttentionTraceStateError(
                        "Cache class changed during one cached attention trace."
                    )
                if step_index + 1 < horizon_value:
                    consumed_tokens.append(selected)
                    next_input = torch.tensor(
                        [[selected]], dtype=torch.long, device=device
                    )
                    prefix = torch.cat((prefix, next_input), dim=1)
    finally:
        hooks.remove()
        for name, value in config_values.items():
            setattr(config, name, value)
        for module, was_training in module_modes:
            module.training = was_training

    if _attention_backend(config) != original_attention_backend:
        raise HFCachedAttentionTraceStateError(
            "Attention backend changed during cached trace collection."
        )
    if cache_class is None:
        raise HFCachedAttentionTraceStateError(
            "Cached trace completed without identifying its cache class."
        )

    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = None
    return HFCachedAttentionTrace(
        sequence_id=sequence_name,
        lane=normalized_lane,  # type: ignore[arg-type]
        layer_index=layer_value,
        group_indices=selected_groups,
        prompt_token_ids=tuple(int(value) for value in ids.cpu().reshape(-1).tolist()),
        consumed_token_ids=tuple(consumed_tokens),
        selected_token_ids=tuple(selected_tokens),
        steps=tuple(steps),
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        query_heads_per_group=query_heads // key_value_heads,
        head_dim=head_dim,
        capture_bytes=capture_bytes,
        maximum_cached_full_prefix_error=max(
            step.cached_full_prefix_max_abs_error for step in steps
        ),
        maximum_selected_context_error=max(
            step.selected_context_max_abs_error for step in steps
        ),
        maximum_value_projection_cache_error=max(
            step.value_projection_cache_max_abs_error for step in steps
        ),
        value_projection_weight_sha256=value_projection_weight_hash,
        value_projection_bias_sha256=value_projection_bias_hash,
        transformers_version=transformers_version,
        torch_version=torch.__version__,
        model_class=_qualified_class_name(model),
        attention_backend=original_attention_backend,
        cache_class=cache_class,
        logit_rtol=controls.logit_rtol,
        logit_atol=controls.logit_atol,
        context_rtol=controls.context_rtol,
        context_atol=controls.context_atol,
        attention_source=(
            "post-rope-qk-plus-exact-layer-cache-reconstruction-verified-against-pre-o"
        ),
        cached_trace_validation_mode=controls.cached_trace_validation_mode,
    )


def build_residual_attention_candidate_evidence(
    reference_traces: Sequence[HFCachedAttentionTrace],
    folded_traces: Sequence[HFCachedAttentionTrace],
    *,
    candidate_ids: Sequence[str],
) -> ResidualAttentionCandidateEvidence:
    """Convert paired cached traces to attention-aware fold evidence.

    Folded fixed-attention V readout is computed with the parent's frozen
    attention distribution.  Thus addressing and value-content damage remain
    separable even when the tentative fold changes its own attention routing.
    """

    references = tuple(reference_traces)
    folded = tuple(folded_traces)
    if not references or len(references) != len(folded):
        raise HFCachedAttentionTraceShapeError(
            "Reference and folded trace collections must be equal and non-empty."
        )
    lanes = {trace.lane for trace in references} | {trace.lane for trace in folded}
    if len(lanes) != 1:
        raise HFCachedAttentionTraceShapeError(
            "Paired cached traces differ in lane, prompt/consumed path, or mix "
            "forced and greedy token lanes."
        )
    pairs: list[ResidualAttentionRolloutPair] = []
    for reference, candidate in zip(references, folded):
        if (
            reference.sequence_id != candidate.sequence_id
            or reference.lane != candidate.lane
            or reference.layer_index != candidate.layer_index
            or reference.group_indices != candidate.group_indices
            or reference.prompt_token_ids != candidate.prompt_token_ids
            or reference.consumed_token_ids != candidate.consumed_token_ids
            or reference.horizon != candidate.horizon
            or reference.query_heads != candidate.query_heads
            or reference.key_value_heads != candidate.key_value_heads
            or reference.query_heads_per_group != candidate.query_heads_per_group
            or reference.head_dim != candidate.head_dim
        ):
            raise HFCachedAttentionTraceShapeError(
                "Paired cached traces differ in lane, prompt/consumed path, layer, "
                "groups, or topology."
            )
        if (
            reference.lane == "forced"
            and reference.selected_token_ids != candidate.selected_token_ids
        ):
            raise HFCachedAttentionTraceShapeError(
                "Forced paired traces must use identical driving token IDs."
            )
        reference_steps: list[ResidualAttentionStepEvidence] = []
        folded_steps: list[ResidualAttentionStepEvidence] = []
        for parent_step, folded_step in zip(reference.steps, candidate.steps):
            if (
                parent_step.attention_probabilities.shape
                != folded_step.attention_probabilities.shape
                or parent_step.full_attention_probabilities.shape
                != folded_step.full_attention_probabilities.shape
                or parent_step.layer_value_cache.shape
                != folded_step.layer_value_cache.shape
                or parent_step.prefix_length != folded_step.prefix_length
            ):
                raise HFCachedAttentionTraceShapeError(
                    "Paired attention distributions/caches need one canonical shape."
                )
            q_per_group = reference.query_heads_per_group
            parent_values = parent_step.layer_value_cache.to(
                torch.float32
            ).repeat_interleave(q_per_group, dim=1)
            folded_values = folded_step.layer_value_cache.to(
                torch.float32
            ).repeat_interleave(q_per_group, dim=1)
            parent_probabilities = parent_step.full_attention_probabilities.to(
                torch.float32
            )
            parent_fixed = torch.matmul(parent_probabilities, parent_values).reshape(
                1, -1
            )
            folded_fixed = torch.matmul(parent_probabilities, folded_values).reshape(
                1, -1
            )
            if parent_fixed.shape != parent_step.full_attention_context.shape:
                raise HFCachedAttentionTraceShapeError(
                    "Full frozen-parent attention readout has the wrong shape."
                )
            if not bool(torch.isfinite(parent_fixed).all()):
                raise HFCachedAttentionTraceStateError(
                    "Full frozen-parent attention readout is non-finite."
                )
            parent_fixed_valid = bool(
                torch.allclose(
                    parent_fixed,
                    parent_step.full_attention_context.to(torch.float32),
                    rtol=reference.context_rtol,
                    atol=reference.context_atol,
                )
            )
            if (
                not parent_fixed_valid
                and reference.cached_trace_validation_mode == "strict"
            ):
                raise HFCachedAttentionTraceStateError(
                    "Full frozen-parent attention readout does not match the parent "
                    "pre-O context."
                )
            common_parent = {
                "step_index": parent_step.step_index,
                "pre_attention_residual": parent_step.pre_attention_residual,
                "query": parent_step.query,
                "key": parent_step.key,
                "value": parent_step.value,
                "attention_scores": parent_step.attention_scores,
                "attention_probabilities": parent_step.attention_probabilities,
                "fixed_attention_value_readout": parent_fixed,
                "attention_output_input": parent_step.full_attention_context,
                "attention_delta": parent_step.attention_delta,
                "post_attention_residual": parent_step.post_attention_residual,
                "key_cache": parent_step.selected_key_cache,
                "value_cache": parent_step.selected_value_cache,
                "action_logits": parent_step.action_logits,
            }
            common_folded = {
                "step_index": folded_step.step_index,
                "pre_attention_residual": folded_step.pre_attention_residual,
                "query": folded_step.query,
                "key": folded_step.key,
                "value": folded_step.value,
                "attention_scores": folded_step.attention_scores,
                "attention_probabilities": folded_step.attention_probabilities,
                "fixed_attention_value_readout": folded_fixed,
                "attention_output_input": folded_step.full_attention_context,
                "attention_delta": folded_step.attention_delta,
                "post_attention_residual": folded_step.post_attention_residual,
                "key_cache": folded_step.selected_key_cache,
                "value_cache": folded_step.selected_value_cache,
                "action_logits": folded_step.action_logits,
            }
            reference_steps.append(ResidualAttentionStepEvidence(**common_parent))
            folded_steps.append(ResidualAttentionStepEvidence(**common_folded))
        sequence_id = (
            f"{reference.sequence_id}:layer-{reference.layer_index}:groups-"
            + ",".join(str(value) for value in reference.group_indices)
            + f":lane-{reference.lane}"
        )
        pairs.append(
            ResidualAttentionRolloutPair(
                ResidualAttentionRollout(sequence_id, tuple(reference_steps)),
                ResidualAttentionRollout(sequence_id, tuple(folded_steps)),
            )
        )
    lane = next(iter(lanes))
    token_lane = (
        "forced-shared-token" if lane == "forced" else "greedy-feedback-aligned-prefix"
    )
    return ResidualAttentionCandidateEvidence(
        tuple(candidate_ids), tuple(pairs), token_lane=token_lane
    )


__all__ = [
    "CachedAttentionLane",
    "CachedTraceValidationMode",
    "HFCachedAttentionStep",
    "HFCachedAttentionTrace",
    "HFCachedAttentionTraceError",
    "HFCachedAttentionTraceLimitError",
    "HFCachedAttentionTraceLimits",
    "HFCachedAttentionTraceShapeError",
    "HFCachedAttentionTraceStateError",
    "TRACE_SCHEMA",
    "build_residual_attention_candidate_evidence",
    "collect_hf_cached_attention_trace",
]
