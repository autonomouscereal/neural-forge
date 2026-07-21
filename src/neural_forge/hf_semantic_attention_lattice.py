"""No-training attention transport across unequal semantic token lattices.

This module separates two operations that must not be conflated:

``rephased common-endpoint projection``
    Gather only target boundaries whose decoded causal prefix is exactly a
    donor boundary, apply the target RoPE at target-native positions, and
    recompute causal scores, probabilities, and context.  This is an exact
    tensor projection on the selected lattice, but contraction can change the
    donor context because omitted donor K/V nodes are genuinely absent.

``probability-preserving expansion transport``
    When tokenization only splits donor spans, duplicate each donor K/V row
    over its target subtokens and subtract ``log(r)`` from every duplicated
    teacher score, where ``r`` is the number of copies causally visible to the
    query.  The probability mass of each donor node and its V-context are then
    preserved exactly.  A donor contraction has no query-independent static
    K/V solution and therefore fails closed.

Both paths consume detached cache tensors only.  They call neither parent nor
child models, use no autograd/optimizer, and mutate no model parameters.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256


SEMANTIC_ATTENTION_CACHE_CAPABILITY_SCHEMA = (
    "neural-forge-semantic-attention-cache-capability-v1"
)
SEMANTIC_ATTENTION_HISTORY_SCHEMA = (
    "neural-forge-semantic-attention-pre-rope-history-v1"
)
SEMANTIC_ATTENTION_REPHASED_PROJECTION_SCHEMA = (
    "neural-forge-semantic-attention-rephased-projection-v1"
)
SEMANTIC_ATTENTION_EXPANSION_TRANSPORT_SCHEMA = (
    "neural-forge-semantic-attention-expansion-transport-v1"
)

LFM2_PRE_ROPE_QUERY_SUFFIX = "attention.q_normalized_pre_rope"
LFM2_PRE_ROPE_KEY_SUFFIX = "attention.k_normalized_pre_rope"
LFM2_VALUE_SUFFIX = "attention.v"
LFM2_VALUE_HISTORY_SUFFIX = "attention.value_history"
LFM2_LEGACY_QUERY_LINEAR_SUFFIX = "attention.q_linear"
LFM2_LEGACY_KEY_LINEAR_SUFFIX = "attention.k_linear"
LFM2_LEGACY_QUERY_POST_ROPE_SUFFIX = "attention.q"
LFM2_LEGACY_KEY_POST_ROPE_SUFFIX = "attention.k"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MODEL_OPERATION_AUDIT = MappingProxyType(
    {
        "parent_model_calls": 0,
        "child_model_calls": 0,
        "model_forward_calls": 0,
        "model_parameters_read": 0,
        "model_parameters_written": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "pruned_parameters": 0,
        "quantized_parameters": 0,
    }
)


class SemanticAttentionLatticeError(ValueError):
    """Cached evidence or semantic lattice geometry is not exact."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError("Semantic attention transport requires PyTorch.") from exc
    return torch


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SemanticAttentionLatticeError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SemanticAttentionLatticeError(f"{name} must be nonnegative.")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise SemanticAttentionLatticeError(f"{name} must be positive.")
    return result


def _finite_positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise SemanticAttentionLatticeError(f"{name} must be finite and positive.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SemanticAttentionLatticeError(
            f"{name} must be finite and positive."
        ) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise SemanticAttentionLatticeError(f"{name} must be finite and positive.")
    return result


def _finite_tensor(value: Any, name: str, *, ndim: int | None = None) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.numel() == 0
        or (ndim is not None and value.ndim != ndim)
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
        or value.requires_grad
        or not bool(torch.isfinite(value).all().item())
    ):
        rank = "" if ndim is None else f" rank-{ndim}"
        raise SemanticAttentionLatticeError(
            f"{name} must be a detached finite dense{rank} floating tensor."
        )
    return value.detach().to(device="cpu").contiguous().clone()


def _integer_positions(value: Any, name: str, *, batch: int) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        try:
            value = torch.tensor(value, dtype=torch.long)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise SemanticAttentionLatticeError(
                f"{name} must contain integer positions."
            ) from exc
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if (
        value.ndim != 2
        or int(value.shape[0]) not in {1, batch}
        or value.numel() == 0
        or value.dtype
        not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        or bool((value < 0).any().item())
    ):
        raise SemanticAttentionLatticeError(
            f"{name} must be nonnegative rank-two integer positions."
        )
    result = (
        value.detach().to(device="cpu", dtype=torch.long).contiguous().clone()
    )
    if int(result.shape[0]) == 1 and batch != 1:
        result = result.expand(batch, -1).contiguous()
    if int(result.shape[1]) > 1 and bool(
        (result[:, 1:] <= result[:, :-1]).any().item()
    ):
        raise SemanticAttentionLatticeError(
            f"{name} must increase strictly on every target lattice."
        )
    return result


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    raw = tensor.view(torch.uint8).reshape(-1).numpy().tobytes()
    return canonical_sha256(
        {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": [int(size) for size in tensor.shape],
            "data_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )


def _layer_tensor_name(phase: str, layer_index: int, suffix: str) -> str:
    return f"{phase}.layer.{layer_index:03d}.{suffix}"


def _phase_order(value: Sequence[str]) -> tuple[str, ...]:
    phases = tuple(value)
    expected = (
        "prefill",
        *(f"decode.{index:03d}" for index in range(max(0, len(phases) - 1))),
    )
    if not phases or phases != expected:
        raise SemanticAttentionLatticeError(
            "phase_order must be prefill followed by contiguous decode phases."
        )
    return phases


def _shape(value: Any) -> tuple[int, ...] | None:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        return None
    return tuple(int(size) for size in value.shape)


@dataclass(frozen=True, slots=True)
class Lfm2SemanticAttentionCacheCapability:
    """Fail-closed availability report for one layer and phase horizon."""

    layer_index: int
    phase_order: tuple[str, ...]
    donor_prompt_token_count: int
    selected_route: str
    prefill_captured_token_count: int
    assembled_current_row_count: int
    missing_enhanced_tensor_names: tuple[str, ...]
    missing_legacy_tensor_names: tuple[str, ...]
    missing_donor_prefix_positions: tuple[int, ...]
    blocking_reasons: tuple[str, ...]
    exact_history_supported: bool
    capability_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        layer = _nonnegative_int(self.layer_index, "layer_index")
        phases = _phase_order(self.phase_order)
        prompt = _positive_int(
            self.donor_prompt_token_count,
            "donor_prompt_token_count",
        )
        captured = _nonnegative_int(
            self.prefill_captured_token_count,
            "prefill_captured_token_count",
        )
        assembled = _nonnegative_int(
            self.assembled_current_row_count,
            "assembled_current_row_count",
        )
        if self.selected_route not in {
            "explicit_pre_rope",
            "legacy_linear_reconstruction",
            "unavailable",
        }:
            raise SemanticAttentionLatticeError("selected_route is unsupported.")
        enhanced = tuple(sorted(set(self.missing_enhanced_tensor_names)))
        legacy = tuple(sorted(set(self.missing_legacy_tensor_names)))
        missing_positions = tuple(self.missing_donor_prefix_positions)
        if missing_positions != tuple(range(max(0, prompt - captured))):
            raise SemanticAttentionLatticeError(
                "Missing donor prefix positions must name the truncated prefix."
            )
        reasons = tuple(sorted(set(self.blocking_reasons)))
        supported = bool(self.exact_history_supported)
        if supported != (not reasons and self.selected_route != "unavailable"):
            raise SemanticAttentionLatticeError(
                "Exact-history capability disagrees with blocking reasons."
            )
        object.__setattr__(self, "layer_index", layer)
        object.__setattr__(self, "phase_order", phases)
        object.__setattr__(self, "donor_prompt_token_count", prompt)
        object.__setattr__(self, "prefill_captured_token_count", captured)
        object.__setattr__(self, "assembled_current_row_count", assembled)
        object.__setattr__(self, "missing_enhanced_tensor_names", enhanced)
        object.__setattr__(self, "missing_legacy_tensor_names", legacy)
        object.__setattr__(self, "missing_donor_prefix_positions", missing_positions)
        object.__setattr__(self, "blocking_reasons", reasons)
        object.__setattr__(self, "exact_history_supported", supported)
        object.__setattr__(
            self,
            "capability_sha256",
            canonical_sha256(self._body()),
        )

    @property
    def required_external_legacy_evidence(self) -> tuple[str, ...]:
        layer = self.layer_index
        return (
            f"model.layers.{layer}.self_attn.q_layernorm.weight",
            f"model.layers.{layer}.self_attn.k_layernorm.weight",
            "config.norm_eps",
            "config.rope_parameters.rope_theta_or_exact_inv_freq",
            "input.position_ids_or_authenticated_contiguous_origin",
        )

    @property
    def model_operation_audit(self) -> dict[str, int]:
        return dict(_MODEL_OPERATION_AUDIT)

    def require_exact_history(self) -> None:
        if not self.exact_history_supported:
            rows = ",".join(str(value) for value in self.missing_donor_prefix_positions)
            raise SemanticAttentionLatticeError(
                "Exact pre-RoPE Q/K/V history is unavailable "
                f"(reasons={self.blocking_reasons}, missing_prefix_rows=[{rows}])."
            )

    def _body(self) -> dict[str, Any]:
        return {
            "schema": SEMANTIC_ATTENTION_CACHE_CAPABILITY_SCHEMA,
            "layer_index": self.layer_index,
            "phase_order": list(self.phase_order),
            "donor_prompt_token_count": self.donor_prompt_token_count,
            "selected_route": self.selected_route,
            "prefill_captured_token_count": self.prefill_captured_token_count,
            "assembled_current_row_count": self.assembled_current_row_count,
            "missing_enhanced_tensor_names": list(
                self.missing_enhanced_tensor_names
            ),
            "missing_legacy_tensor_names": list(self.missing_legacy_tensor_names),
            "missing_donor_prefix_positions": list(
                self.missing_donor_prefix_positions
            ),
            "required_external_legacy_evidence": list(
                self.required_external_legacy_evidence
            ),
            "blocking_reasons": list(self.blocking_reasons),
            "exact_history_supported": self.exact_history_supported,
            "rolling_key_value_cache_is_complete_history": False,
            "model_operations": self.model_operation_audit,
        }

    def to_evidence_dict(self) -> dict[str, Any]:
        return {**self._body(), "capability_sha256": self.capability_sha256}


def inspect_lfm2_semantic_attention_cache_capability(
    tensors: Mapping[str, Any],
    *,
    phase_order: Sequence[str],
    layer_index: int,
    donor_prompt_token_count: int,
) -> Lfm2SemanticAttentionCacheCapability:
    """Inspect exact cache rows without opening or executing either model."""

    if not isinstance(tensors, Mapping):
        raise TypeError("tensors must be a mapping.")
    phases = _phase_order(phase_order)
    layer = _nonnegative_int(layer_index, "layer_index")
    prompt = _positive_int(donor_prompt_token_count, "donor_prompt_token_count")
    enhanced_names = tuple(
        _layer_tensor_name(phase, layer, suffix)
        for phase in phases
        for suffix in (
            LFM2_PRE_ROPE_QUERY_SUFFIX,
            LFM2_PRE_ROPE_KEY_SUFFIX,
            LFM2_VALUE_SUFFIX,
        )
    ) + (
        _layer_tensor_name(phases[-1], layer, LFM2_VALUE_HISTORY_SUFFIX),
    )
    legacy_names = tuple(
        _layer_tensor_name(phase, layer, suffix)
        for phase in phases
        for suffix in (
            LFM2_LEGACY_QUERY_LINEAR_SUFFIX,
            LFM2_LEGACY_KEY_LINEAR_SUFFIX,
            LFM2_LEGACY_QUERY_POST_ROPE_SUFFIX,
            LFM2_LEGACY_KEY_POST_ROPE_SUFFIX,
            LFM2_VALUE_SUFFIX,
        )
    )
    missing_enhanced = tuple(name for name in enhanced_names if name not in tensors)
    missing_legacy = tuple(name for name in legacy_names if name not in tensors)
    if not missing_enhanced:
        route = "explicit_pre_rope"
        query_suffix = LFM2_PRE_ROPE_QUERY_SUFFIX
    elif not missing_legacy:
        route = "legacy_linear_reconstruction"
        query_suffix = LFM2_LEGACY_QUERY_LINEAR_SUFFIX
    else:
        route = "unavailable"
        query_suffix = (
            LFM2_PRE_ROPE_QUERY_SUFFIX
            if _layer_tensor_name(
                "prefill", layer, LFM2_PRE_ROPE_QUERY_SUFFIX
            )
            in tensors
            else LFM2_LEGACY_QUERY_LINEAR_SUFFIX
        )
    counts: list[int] = []
    invalid_geometry = False
    for ordinal, phase in enumerate(phases):
        shape = _shape(tensors.get(_layer_tensor_name(phase, layer, query_suffix)))
        if shape is None or len(shape) != 4:
            counts.append(0)
            invalid_geometry = True
            continue
        count = shape[2]
        counts.append(count)
        if ordinal > 0 and count != 1:
            invalid_geometry = True
    captured = counts[0] if counts else 0
    assembled = sum(counts)
    expected = prompt + len(phases) - 1
    missing_positions = tuple(range(max(0, prompt - captured)))
    reasons = []
    if route == "unavailable":
        reasons.append("missing_cached_attention_tensors")
    if captured != prompt:
        reasons.append("truncated_prefill_evidence")
    if invalid_geometry or assembled != expected:
        reasons.append("phase_row_geometry_incomplete")
    return Lfm2SemanticAttentionCacheCapability(
        layer_index=layer,
        phase_order=phases,
        donor_prompt_token_count=prompt,
        selected_route=route,
        prefill_captured_token_count=captured,
        assembled_current_row_count=assembled,
        missing_enhanced_tensor_names=missing_enhanced,
        missing_legacy_tensor_names=missing_legacy,
        missing_donor_prefix_positions=missing_positions,
        blocking_reasons=tuple(reasons),
        exact_history_supported=not reasons,
    )


@dataclass(frozen=True, slots=True)
class SemanticAttentionPreRopeHistory:
    """Complete donor Q/K/V history before positional rephasing."""

    query_normalized_pre_rope: Any = field(repr=False, compare=False)
    key_normalized_pre_rope: Any = field(repr=False, compare=False)
    value: Any = field(repr=False, compare=False)
    source_capability_sha256: str
    source_tensor_names: tuple[str, ...]
    history_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        query = _finite_tensor(
            self.query_normalized_pre_rope,
            "query_normalized_pre_rope",
            ndim=4,
        )
        key = _finite_tensor(
            self.key_normalized_pre_rope,
            "key_normalized_pre_rope",
            ndim=4,
        )
        value = _finite_tensor(self.value, "value", ndim=4)
        batch, query_heads, tokens, head_dim = map(int, query.shape)
        if (
            tuple(key.shape) != tuple(value.shape)
            or int(key.shape[0]) != batch
            or int(key.shape[2]) != tokens
            or int(key.shape[3]) != head_dim
            or query_heads % int(key.shape[1]) != 0
            or head_dim % 2 != 0
            or query.dtype != key.dtype
            or key.dtype != value.dtype
        ):
            raise SemanticAttentionLatticeError(
                "Pre-RoPE Q/K/V do not share one dense GQA history."
            )
        names = tuple(self.source_tensor_names)
        if not names or len(set(names)) != len(names):
            raise SemanticAttentionLatticeError(
                "source_tensor_names must be unique and nonempty."
            )
        capability = _digest(
            self.source_capability_sha256,
            "source capability SHA-256",
        )
        object.__setattr__(self, "query_normalized_pre_rope", query)
        object.__setattr__(self, "key_normalized_pre_rope", key)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "source_capability_sha256", capability)
        object.__setattr__(self, "source_tensor_names", names)
        object.__setattr__(self, "history_sha256", canonical_sha256(self._body()))

    @property
    def token_count(self) -> int:
        return int(self.query_normalized_pre_rope.shape[2])

    @property
    def model_operation_audit(self) -> dict[str, int]:
        return dict(_MODEL_OPERATION_AUDIT)

    def _body(self) -> dict[str, Any]:
        return {
            "schema": SEMANTIC_ATTENTION_HISTORY_SCHEMA,
            "source_capability_sha256": self.source_capability_sha256,
            "source_tensor_names": list(self.source_tensor_names),
            "query_normalized_pre_rope_sha256": _tensor_sha256(
                self.query_normalized_pre_rope
            ),
            "key_normalized_pre_rope_sha256": _tensor_sha256(
                self.key_normalized_pre_rope
            ),
            "value_sha256": _tensor_sha256(self.value),
            "token_count": self.token_count,
            "model_operations": self.model_operation_audit,
        }

    def to_evidence_dict(self) -> dict[str, Any]:
        return {**self._body(), "history_sha256": self.history_sha256}


def assemble_lfm2_explicit_pre_rope_history(
    tensors: Mapping[str, Any],
    *,
    phase_order: Sequence[str],
    layer_index: int,
    donor_prompt_token_count: int,
) -> SemanticAttentionPreRopeHistory:
    """Concatenate sealed phase-current rows; never treat rolling KV as full."""

    capability = inspect_lfm2_semantic_attention_cache_capability(
        tensors,
        phase_order=phase_order,
        layer_index=layer_index,
        donor_prompt_token_count=donor_prompt_token_count,
    )
    capability.require_exact_history()
    if capability.selected_route != "explicit_pre_rope":
        raise SemanticAttentionLatticeError(
            "Explicit pre-RoPE assembly cannot silently use a legacy route."
        )
    torch = _torch()
    query_parts = []
    key_parts = []
    value_parts = []
    names = []
    for phase in capability.phase_order:
        for suffix, target in (
            (LFM2_PRE_ROPE_QUERY_SUFFIX, query_parts),
            (LFM2_PRE_ROPE_KEY_SUFFIX, key_parts),
            (LFM2_VALUE_SUFFIX, value_parts),
        ):
            name = _layer_tensor_name(phase, capability.layer_index, suffix)
            target.append(_finite_tensor(tensors[name], name, ndim=4))
            names.append(name)
    query = torch.cat(query_parts, dim=2).contiguous()
    key = torch.cat(key_parts, dim=2).contiguous()
    value = torch.cat(value_parts, dim=2).contiguous()
    history_name = _layer_tensor_name(
        capability.phase_order[-1],
        capability.layer_index,
        LFM2_VALUE_HISTORY_SUFFIX,
    )
    sealed_history = _finite_tensor(tensors[history_name], history_name, ndim=4)
    names.append(history_name)
    if not torch.equal(value, sealed_history):
        raise SemanticAttentionLatticeError(
            "Concatenated phase V rows differ from the sealed full V history."
        )
    return SemanticAttentionPreRopeHistory(
        query_normalized_pre_rope=query,
        key_normalized_pre_rope=key,
        value=value,
        source_capability_sha256=capability.capability_sha256,
        source_tensor_names=tuple(names),
    )


def reconstruct_lfm2_normalized_pre_rope_rows(
    query_linear: Any,
    key_linear: Any,
    *,
    query_norm_weight: Any,
    key_norm_weight: Any,
    norm_epsilon: float,
) -> tuple[Any, Any]:
    """Replay only LFM2's finite-precision Q/K head RMSNorm algebra."""

    torch = _torch()
    query = _finite_tensor(query_linear, "query_linear", ndim=4)
    key = _finite_tensor(key_linear, "key_linear", ndim=4)
    q_weight = _finite_tensor(query_norm_weight, "query_norm_weight", ndim=1)
    k_weight = _finite_tensor(key_norm_weight, "key_norm_weight", ndim=1)
    epsilon = _finite_positive_float(norm_epsilon, "norm_epsilon")
    if (
        int(query.shape[-1]) != int(q_weight.shape[0])
        or int(key.shape[-1]) != int(k_weight.shape[0])
        or int(query.shape[-1]) != int(key.shape[-1])
        or query.dtype != key.dtype
    ):
        raise SemanticAttentionLatticeError(
            "Q/K linear rows and head-local norm weights are incompatible."
        )

    def normalize(value: Any, weight: Any) -> Any:
        input_dtype = value.dtype
        direction = value.to(torch.float32)
        variance = direction.square().mean(dim=-1, keepdim=True)
        direction = direction * torch.rsqrt(variance + epsilon)
        return (
            weight.to(dtype=input_dtype) * direction.to(dtype=input_dtype)
        ).contiguous()

    with torch.inference_mode():
        return normalize(query, q_weight), normalize(key, k_weight)


def _mapping_indices(
    value: Sequence[int | None],
    *,
    donor_tokens: int,
    require_complete: bool,
) -> tuple[int | None, ...]:
    result = tuple(value)
    if not result:
        raise SemanticAttentionLatticeError("Semantic boundary map cannot be empty.")
    observed = []
    for item in result:
        if item is None:
            if require_complete:
                continue
        elif (
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            or item >= donor_tokens
        ):
            raise SemanticAttentionLatticeError(
                "Semantic boundary map contains an invalid donor position."
            )
        else:
            observed.append(item)
    if any(right <= left for left, right in zip(observed, observed[1:])):
        raise SemanticAttentionLatticeError(
            "Semantic endpoint mapping must preserve strict causal order."
        )
    return result


def _rotate_half(value: Any) -> Any:
    half = int(value.shape[-1]) // 2
    return _torch().cat((-value[..., half:], value[..., :half]), dim=-1)


def _target_rope_terms(
    position_ids: Any,
    inv_freq: Any,
    *,
    head_dim: int,
    dtype: Any,
    attention_scaling: float,
) -> tuple[Any, Any, Any]:
    torch = _torch()
    positions = _integer_positions(
        position_ids,
        "target_position_ids",
        batch=int(position_ids.shape[0]) if isinstance(position_ids, torch.Tensor) and position_ids.ndim == 2 else 1,
    )
    frequencies = _finite_tensor(inv_freq, "target_rope_inv_freq", ndim=1)
    if int(frequencies.shape[0]) * 2 != head_dim:
        raise SemanticAttentionLatticeError(
            "Target RoPE inv_freq does not match the cached head dimension."
        )
    scale = _finite_positive_float(attention_scaling, "attention_scaling")
    angles = positions.to(torch.float32).unsqueeze(-1) * frequencies.to(
        torch.float32
    ).reshape(1, 1, -1)
    angles = torch.cat((angles, angles), dim=-1)
    return (
        positions,
        (angles.cos() * scale).to(dtype=dtype).contiguous(),
        (angles.sin() * scale).to(dtype=dtype).contiguous(),
    )


def _causal_mask(position_ids: Any, *, query_indices: Any | None = None) -> Any:
    positions = position_ids
    query_positions = (
        positions
        if query_indices is None
        else positions.index_select(1, query_indices)
    )
    return (
        positions[:, None, None, :]
        <= query_positions[:, None, :, None]
    ).contiguous()


def _masked_softmax(scores: Any, mask: Any) -> Any:
    torch = _torch()
    expanded = mask.expand(-1, int(scores.shape[1]), -1, -1)
    probabilities = torch.softmax(
        scores.to(torch.float64).masked_fill(~expanded, -torch.inf),
        dim=-1,
        dtype=torch.float64,
    )
    return torch.where(expanded, probabilities, torch.zeros_like(probabilities))


@dataclass(frozen=True, slots=True)
class SemanticAttentionRephasedProjection:
    """Target-native attention over semantically equal donor endpoints."""

    semantic_alignment_sha256: str
    source_history_sha256: str
    donor_gather_indices: tuple[int, ...]
    target_position_ids: Any = field(repr=False, compare=False)
    target_rope_cos: Any = field(repr=False, compare=False)
    target_rope_sin: Any = field(repr=False, compare=False)
    query: Any = field(repr=False, compare=False)
    key: Any = field(repr=False, compare=False)
    value: Any = field(repr=False, compare=False)
    scores: Any = field(repr=False, compare=False)
    causal_mask: Any = field(repr=False, compare=False)
    probabilities: Any = field(repr=False, compare=False)
    context: Any = field(repr=False, compare=False)
    projection_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _digest(self.semantic_alignment_sha256, "semantic alignment SHA-256")
        _digest(self.source_history_sha256, "source history SHA-256")
        indices = tuple(self.donor_gather_indices)
        if not indices or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in indices
        ):
            raise SemanticAttentionLatticeError("donor_gather_indices are invalid.")
        tensors = {
            "target_position_ids": self.target_position_ids,
            "target_rope_cos": self.target_rope_cos,
            "target_rope_sin": self.target_rope_sin,
            "query": self.query,
            "key": self.key,
            "value": self.value,
            "scores": self.scores,
            "probabilities": self.probabilities,
            "context": self.context,
        }
        torch = _torch()
        positions = _integer_positions(
            tensors.pop("target_position_ids"),
            "target_position_ids",
            batch=int(self.query.shape[0]),
        )
        validated = {
            name: _finite_tensor(value, name, ndim=3 if "rope" in name else 4)
            for name, value in tensors.items()
        }
        mask = self.causal_mask
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim != 4:
            raise SemanticAttentionLatticeError(
                "causal_mask must be a detached rank-four bool tensor."
            )
        mask = mask.detach().to(device="cpu").contiguous()
        batch, query_heads, tokens, head_dim = map(int, validated["query"].shape)
        kv_heads = int(validated["key"].shape[1])
        if (
            len(indices) != tokens
            or int(positions.shape[1]) != tokens
            or query_heads % kv_heads != 0
            or tuple(validated["key"].shape) != (batch, kv_heads, tokens, head_dim)
            or tuple(validated["value"].shape) != tuple(validated["key"].shape)
            or tuple(validated["scores"].shape) != (batch, query_heads, tokens, tokens)
            or tuple(validated["probabilities"].shape) != tuple(validated["scores"].shape)
            or tuple(validated["context"].shape) != (batch, query_heads, tokens, head_dim)
            or tuple(validated["target_rope_cos"].shape) != (batch, tokens, head_dim)
            or tuple(validated["target_rope_sin"].shape) != (batch, tokens, head_dim)
            or tuple(mask.shape) != (batch, 1, tokens, tokens)
        ):
            raise SemanticAttentionLatticeError(
                "Rephased projection tensors do not share one causal geometry."
            )
        expanded_mask = mask.expand_as(validated["probabilities"])
        if not torch.equal(
            validated["probabilities"].masked_select(~expanded_mask),
            torch.zeros_like(
                validated["probabilities"].masked_select(~expanded_mask)
            ),
        ):
            raise SemanticAttentionLatticeError(
                "Rephased projection assigns probability outside causality."
            )
        object.__setattr__(self, "donor_gather_indices", indices)
        object.__setattr__(self, "target_position_ids", positions)
        for name, value in validated.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "causal_mask", mask)
        object.__setattr__(self, "projection_sha256", canonical_sha256(self._body()))

    def endpoint_residual_evidence(self) -> dict[str, Any]:
        return {
            "parent_probabilities": self.probabilities,
            "parent_causal_mask": self.causal_mask,
            "parent_context": self.context,
            "query_rope_cos": self.target_rope_cos,
            "query_rope_sin": self.target_rope_sin,
            "key_rope_cos": self.target_rope_cos,
            "key_rope_sin": self.target_rope_sin,
            "parent_probability_provenance": "explicit_parent_probability_tensor",
            "semantic_attention_projection_sha256": self.projection_sha256,
        }

    def _body(self) -> dict[str, Any]:
        tensors = {
            name: _tensor_sha256(getattr(self, name))
            for name in (
                "target_position_ids",
                "target_rope_cos",
                "target_rope_sin",
                "query",
                "key",
                "value",
                "scores",
                "causal_mask",
                "probabilities",
                "context",
            )
        }
        return {
            "schema": SEMANTIC_ATTENTION_REPHASED_PROJECTION_SCHEMA,
            "semantic_alignment_sha256": self.semantic_alignment_sha256,
            "source_history_sha256": self.source_history_sha256,
            "donor_gather_indices": list(self.donor_gather_indices),
            "tensors": tensors,
            "semantic_equal_endpoints_only": True,
            "target_position_rephasing": True,
            "target_lattice_softmax_recomputed": True,
            "donor_context_preservation_claimed": False,
            "model_operations": dict(_MODEL_OPERATION_AUDIT),
        }

    def to_evidence_dict(self) -> dict[str, Any]:
        return {**self._body(), "projection_sha256": self.projection_sha256}


def project_rephased_semantic_attention_lattice(
    history: SemanticAttentionPreRopeHistory,
    *,
    semantic_alignment_sha256: str,
    target_to_donor_boundary_positions: Sequence[int | None],
    target_position_ids: Any,
    target_rope_inv_freq: Any,
    target_rope_attention_scaling: float = 1.0,
) -> SemanticAttentionRephasedProjection:
    """Rephase exact common endpoints and analytically replay target attention."""

    if not isinstance(history, SemanticAttentionPreRopeHistory):
        raise TypeError("history must be SemanticAttentionPreRopeHistory.")
    alignment = _digest(semantic_alignment_sha256, "semantic alignment SHA-256")
    mapping = _mapping_indices(
        target_to_donor_boundary_positions,
        donor_tokens=history.token_count,
        require_complete=True,
    )
    missing = tuple(index for index, value in enumerate(mapping) if value is None)
    if missing:
        raise SemanticAttentionLatticeError(
            "Target lattice contains boundaries without semantically equal donor "
            f"endpoints: {missing}."
        )
    indices_tuple = tuple(int(value) for value in mapping if value is not None)
    torch = _torch()
    indices = torch.tensor(indices_tuple, dtype=torch.long)
    query_pre = history.query_normalized_pre_rope.index_select(2, indices)
    key_pre = history.key_normalized_pre_rope.index_select(2, indices)
    value = history.value.index_select(2, indices)
    positions = _integer_positions(
        target_position_ids,
        "target_position_ids",
        batch=int(query_pre.shape[0]),
    )
    if int(positions.shape[1]) != len(indices_tuple):
        raise SemanticAttentionLatticeError(
            "Target position count differs from the semantic endpoint map."
        )
    positions, cosine, sine = _target_rope_terms(
        positions,
        target_rope_inv_freq,
        head_dim=int(query_pre.shape[-1]),
        dtype=query_pre.dtype,
        attention_scaling=target_rope_attention_scaling,
    )
    with torch.inference_mode():
        query = query_pre * cosine.unsqueeze(1) + _rotate_half(query_pre) * sine.unsqueeze(1)
        key = key_pre * cosine.unsqueeze(1) + _rotate_half(key_pre) * sine.unsqueeze(1)
        groups = int(query.shape[1]) // int(key.shape[1])
        repeated_key = key.repeat_interleave(groups, dim=1)
        repeated_value = value.repeat_interleave(groups, dim=1)
        scores = torch.matmul(
            query.to(torch.float64),
            repeated_key.transpose(2, 3).to(torch.float64),
        ) / math.sqrt(int(query.shape[-1]))
        mask = _causal_mask(positions)
        probabilities = _masked_softmax(scores, mask)
        context = torch.matmul(probabilities, repeated_value.to(torch.float64))
    return SemanticAttentionRephasedProjection(
        semantic_alignment_sha256=alignment,
        source_history_sha256=history.history_sha256,
        donor_gather_indices=indices_tuple,
        target_position_ids=positions,
        target_rope_cos=cosine,
        target_rope_sin=sine,
        query=query,
        key=key,
        value=value,
        scores=scores,
        causal_mask=mask,
        probabilities=probabilities,
        context=context,
    )


@dataclass(frozen=True, slots=True)
class SemanticAttentionExpansionTransport:
    """Exact teacher probability/context transport for token splitting."""

    semantic_alignment_sha256: str
    query_mode: str
    target_to_donor_boundary_positions: tuple[int | None, ...]
    target_to_donor_span_owner_positions: tuple[int, ...]
    donor_span_multiplicities: tuple[int, ...]
    query_target_positions: tuple[int, ...]
    query_donor_positions: tuple[int, ...]
    target_position_ids: Any = field(repr=False, compare=False)
    query_rope_cos: Any = field(repr=False, compare=False)
    query_rope_sin: Any = field(repr=False, compare=False)
    key_rope_cos: Any = field(repr=False, compare=False)
    key_rope_sin: Any = field(repr=False, compare=False)
    probabilities: Any = field(repr=False, compare=False)
    causal_mask: Any = field(repr=False, compare=False)
    context: Any = field(repr=False, compare=False)
    expanded_value: Any = field(repr=False, compare=False)
    maximum_probability_mass_error: float
    maximum_context_absolute_error: float
    transport_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _digest(self.semantic_alignment_sha256, "semantic alignment SHA-256")
        if self.query_mode not in {"semantic_endpoints", "owner_surrogate_full_lattice"}:
            raise SemanticAttentionLatticeError("query_mode is unsupported.")
        torch = _torch()
        probability = _finite_tensor(self.probabilities, "probabilities", ndim=4)
        context = _finite_tensor(self.context, "context", ndim=4)
        value = _finite_tensor(self.expanded_value, "expanded_value", ndim=4)
        mask = self.causal_mask
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim != 4:
            raise SemanticAttentionLatticeError("causal_mask must be rank-four bool.")
        mask = mask.detach().to(device="cpu").contiguous()
        batch, heads, queries, keys = map(int, probability.shape)
        if (
            len(self.query_target_positions) != queries
            or len(self.query_donor_positions) != queries
            or len(self.target_to_donor_span_owner_positions) != keys
            or len(self.target_to_donor_boundary_positions) != keys
            or tuple(mask.shape) != (batch, 1, queries, keys)
            or int(context.shape[0]) != batch
            or int(context.shape[1]) != heads
            or int(context.shape[2]) != queries
            or int(value.shape[0]) != batch
            or int(value.shape[2]) != keys
        ):
            raise SemanticAttentionLatticeError(
                "Expansion transport tensors do not share one query/key lattice."
            )
        positions = _integer_positions(
            self.target_position_ids,
            "target_position_ids",
            batch=batch,
        )
        query_cos = _finite_tensor(self.query_rope_cos, "query_rope_cos", ndim=3)
        query_sin = _finite_tensor(self.query_rope_sin, "query_rope_sin", ndim=3)
        key_cos = _finite_tensor(self.key_rope_cos, "key_rope_cos", ndim=3)
        key_sin = _finite_tensor(self.key_rope_sin, "key_rope_sin", ndim=3)
        if (
            int(positions.shape[1]) != keys
            or tuple(query_cos.shape) != tuple(query_sin.shape)
            or tuple(key_cos.shape) != tuple(key_sin.shape)
            or tuple(query_cos.shape[:2]) != (batch, queries)
            or tuple(key_cos.shape[:2]) != (batch, keys)
            or int(query_cos.shape[-1]) != int(key_cos.shape[-1])
        ):
            raise SemanticAttentionLatticeError("Expansion target RoPE geometry changed.")
        mass_error = float(self.maximum_probability_mass_error)
        context_error = float(self.maximum_context_absolute_error)
        if (
            not math.isfinite(mass_error)
            or mass_error < 0.0
            or not math.isfinite(context_error)
            or context_error < 0.0
        ):
            raise SemanticAttentionLatticeError(
                "Expansion preservation errors must be finite and nonnegative."
            )
        object.__setattr__(self, "target_position_ids", positions)
        object.__setattr__(self, "query_rope_cos", query_cos)
        object.__setattr__(self, "query_rope_sin", query_sin)
        object.__setattr__(self, "key_rope_cos", key_cos)
        object.__setattr__(self, "key_rope_sin", key_sin)
        object.__setattr__(self, "probabilities", probability)
        object.__setattr__(self, "causal_mask", mask)
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "expanded_value", value)
        object.__setattr__(self, "maximum_probability_mass_error", mass_error)
        object.__setattr__(self, "maximum_context_absolute_error", context_error)
        object.__setattr__(self, "transport_sha256", canonical_sha256(self._body()))

    @property
    def direct_endpoint_case_suffix_compatible(self) -> bool:
        keys = len(self.target_to_donor_span_owner_positions)
        queries = len(self.query_target_positions)
        return self.query_target_positions == tuple(range(keys - queries, keys))

    @property
    def inserted_query_rows_are_semantically_exact(self) -> bool:
        return self.query_mode == "semantic_endpoints"

    def endpoint_residual_evidence(self) -> dict[str, Any]:
        return {
            "parent_probabilities": self.probabilities,
            "parent_causal_mask": self.causal_mask,
            "parent_context": self.context,
            "query_rope_cos": self.query_rope_cos,
            "query_rope_sin": self.query_rope_sin,
            "key_rope_cos": self.key_rope_cos,
            "key_rope_sin": self.key_rope_sin,
            "parent_probability_provenance": "explicit_parent_probability_tensor",
            "semantic_attention_transport_sha256": self.transport_sha256,
        }

    def _body(self) -> dict[str, Any]:
        return {
            "schema": SEMANTIC_ATTENTION_EXPANSION_TRANSPORT_SCHEMA,
            "semantic_alignment_sha256": self.semantic_alignment_sha256,
            "query_mode": self.query_mode,
            "target_to_donor_boundary_positions": list(
                self.target_to_donor_boundary_positions
            ),
            "target_to_donor_span_owner_positions": list(
                self.target_to_donor_span_owner_positions
            ),
            "donor_span_multiplicities": list(self.donor_span_multiplicities),
            "query_target_positions": list(self.query_target_positions),
            "query_donor_positions": list(self.query_donor_positions),
            "tensors": {
                name: _tensor_sha256(getattr(self, name))
                for name in (
                    "target_position_ids",
                    "query_rope_cos",
                    "query_rope_sin",
                    "key_rope_cos",
                    "key_rope_sin",
                    "probabilities",
                    "causal_mask",
                    "context",
                    "expanded_value",
                )
            },
            "maximum_probability_mass_error": self.maximum_probability_mass_error,
            "maximum_context_absolute_error": self.maximum_context_absolute_error,
            "probability_mass_preserved": self.maximum_probability_mass_error <= 1e-12,
            "context_preserved": self.maximum_context_absolute_error <= 1e-12,
            "contraction_supported": False,
            "score_rule": "source_score_minus_log_causally_visible_owner_copies",
            "inserted_query_rows_are_semantically_exact": (
                self.inserted_query_rows_are_semantically_exact
            ),
            "direct_endpoint_case_suffix_compatible": (
                self.direct_endpoint_case_suffix_compatible
            ),
            "model_operations": dict(_MODEL_OPERATION_AUDIT),
        }

    def to_evidence_dict(self) -> dict[str, Any]:
        return {**self._body(), "transport_sha256": self.transport_sha256}


def transport_probability_preserving_attention_expansion(
    *,
    semantic_alignment_sha256: str,
    target_to_donor_boundary_positions: Sequence[int | None],
    target_to_donor_span_owner_positions: Sequence[int],
    donor_span_multiplicities: Sequence[int],
    parent_query_post_rope: Any,
    parent_key_post_rope: Any,
    parent_value: Any,
    target_position_ids: Any,
    target_rope_inv_freq: Any,
    target_rope_attention_scaling: float = 1.0,
    query_mode: str = "semantic_endpoints",
) -> SemanticAttentionExpansionTransport:
    """Duplicate split-span K/V and preserve source probability/context exactly."""

    torch = _torch()
    alignment = _digest(semantic_alignment_sha256, "semantic alignment SHA-256")
    query = _finite_tensor(parent_query_post_rope, "parent_query_post_rope", ndim=4)
    key = _finite_tensor(parent_key_post_rope, "parent_key_post_rope", ndim=4)
    value = _finite_tensor(parent_value, "parent_value", ndim=4)
    donor_tokens = int(query.shape[2])
    if (
        int(key.shape[2]) != donor_tokens
        or tuple(key.shape) != tuple(value.shape)
        or int(query.shape[0]) != int(key.shape[0])
        or int(query.shape[-1]) != int(key.shape[-1])
        or int(query.shape[1]) % int(key.shape[1]) != 0
    ):
        raise SemanticAttentionLatticeError(
            "Parent post-RoPE Q/K/V do not share one complete donor lattice."
        )
    boundary = _mapping_indices(
        target_to_donor_boundary_positions,
        donor_tokens=donor_tokens,
        require_complete=False,
    )
    owner = tuple(target_to_donor_span_owner_positions)
    if len(owner) != len(boundary) or any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or item < 0
        or item >= donor_tokens
        for item in owner
    ):
        raise SemanticAttentionLatticeError("Span-owner map is invalid.")
    if any(right < left for left, right in zip(owner, owner[1:])):
        raise SemanticAttentionLatticeError(
            "Span-owner map must preserve nondecreasing donor order."
        )
    multiplicities = tuple(owner.count(index) for index in range(donor_tokens))
    expected_multiplicities = tuple(donor_span_multiplicities)
    if multiplicities != expected_multiplicities or any(value <= 0 for value in multiplicities):
        raise SemanticAttentionLatticeError(
            "Attention expansion requires every donor span exactly once or more; "
            "contraction is not a query-independent static K/V transport."
        )
    exact_target = tuple(index for index, item in enumerate(boundary) if item is not None)
    exact_donor = tuple(int(item) for item in boundary if item is not None)
    if exact_donor != tuple(range(donor_tokens)) or any(
        owner[target_index] != donor_index
        or target_index != max(
            index for index, candidate in enumerate(owner) if candidate == donor_index
        )
        for target_index, donor_index in zip(exact_target, exact_donor, strict=True)
    ):
        raise SemanticAttentionLatticeError(
            "Every donor endpoint must equal the terminal boundary of its target span."
        )
    if query_mode == "semantic_endpoints":
        query_targets = exact_target
        query_donors = exact_donor
    elif query_mode == "owner_surrogate_full_lattice":
        query_targets = tuple(range(len(owner)))
        query_donors = owner
    else:
        raise SemanticAttentionLatticeError("query_mode is unsupported.")
    positions = _integer_positions(
        target_position_ids,
        "target_position_ids",
        batch=int(query.shape[0]),
    )
    if int(positions.shape[1]) != len(owner):
        raise SemanticAttentionLatticeError(
            "Target positions do not cover the span-owner lattice."
        )
    positions, full_cos, full_sin = _target_rope_terms(
        positions,
        target_rope_inv_freq,
        head_dim=int(query.shape[-1]),
        dtype=query.dtype,
        attention_scaling=target_rope_attention_scaling,
    )
    owner_indices = torch.tensor(owner, dtype=torch.long)
    query_target_indices = torch.tensor(query_targets, dtype=torch.long)
    query_donor_indices = torch.tensor(query_donors, dtype=torch.long)
    groups = int(query.shape[1]) // int(key.shape[1])
    repeated_key = key.repeat_interleave(groups, dim=1)
    repeated_value = value.repeat_interleave(groups, dim=1)
    with torch.inference_mode():
        source_scores = torch.matmul(
            query.to(torch.float64),
            repeated_key.transpose(2, 3).to(torch.float64),
        ) / math.sqrt(int(query.shape[-1]))
        donor_index = torch.arange(donor_tokens)
        source_mask = donor_index.reshape(1, 1, 1, -1) <= donor_index.reshape(
            1, 1, -1, 1
        )
        source_probabilities = _masked_softmax(source_scores, source_mask)
        source_context = torch.matmul(
            source_probabilities,
            repeated_value.to(torch.float64),
        )
        selected_scores = source_scores.index_select(2, query_donor_indices)
        selected_scores = selected_scores.index_select(3, owner_indices)
        mask = _causal_mask(positions, query_indices=query_target_indices)
        visible = mask[:, 0]
        corrections = torch.zeros(
            (int(query.shape[0]), len(query_targets), len(owner)),
            dtype=torch.float64,
        )
        for query_ordinal in range(len(query_targets)):
            for donor_index_value in range(donor_tokens):
                keys = torch.tensor(
                    [index for index, item in enumerate(owner) if item == donor_index_value],
                    dtype=torch.long,
                )
                available = visible[:, query_ordinal].index_select(1, keys).sum(dim=1)
                for batch_index in range(int(query.shape[0])):
                    count = int(available[batch_index].item())
                    if count:
                        corrections[
                            batch_index,
                            query_ordinal,
                            keys,
                        ] = math.log(count)
        transported_scores = torch.where(
            mask.expand(-1, int(query.shape[1]), -1, -1),
            selected_scores - corrections.unsqueeze(1),
            torch.zeros_like(selected_scores),
        )
        probabilities = _masked_softmax(transported_scores, mask)
        expanded_value = value.index_select(2, owner_indices)
        expanded_repeated_value = expanded_value.repeat_interleave(groups, dim=1)
        context = torch.matmul(
            probabilities,
            expanded_repeated_value.to(torch.float64),
        )
        expected_probabilities = source_probabilities.index_select(
            2, query_donor_indices
        )
        owner_mass = torch.zeros_like(expected_probabilities)
        for donor_index_value in range(donor_tokens):
            keys = torch.tensor(
                [index for index, item in enumerate(owner) if item == donor_index_value],
                dtype=torch.long,
            )
            owner_mass[..., donor_index_value] = probabilities.index_select(
                3, keys
            ).sum(dim=3)
        expected_context = source_context.index_select(2, query_donor_indices)
        mass_error = float((owner_mass - expected_probabilities).abs().max().item())
        context_error = float((context - expected_context).abs().max().item())
    if mass_error > 1e-12 or context_error > 1e-12:
        raise SemanticAttentionLatticeError(
            "Probability-preserving expansion failed its exact FP64 audit "
            f"(mass={mass_error:.6g}, context={context_error:.6g})."
        )
    return SemanticAttentionExpansionTransport(
        semantic_alignment_sha256=alignment,
        query_mode=query_mode,
        target_to_donor_boundary_positions=boundary,
        target_to_donor_span_owner_positions=owner,
        donor_span_multiplicities=multiplicities,
        query_target_positions=query_targets,
        query_donor_positions=query_donors,
        target_position_ids=positions,
        query_rope_cos=full_cos.index_select(1, query_target_indices),
        query_rope_sin=full_sin.index_select(1, query_target_indices),
        key_rope_cos=full_cos,
        key_rope_sin=full_sin,
        probabilities=probabilities,
        causal_mask=mask,
        context=context,
        expanded_value=expanded_value,
        maximum_probability_mass_error=mass_error,
        maximum_context_absolute_error=context_error,
    )


def project_lfm2_phase_rephased_semantic_attention(
    alignment: Any,
    history: SemanticAttentionPreRopeHistory,
    *,
    target_rope_inv_freq: Any,
    target_rope_attention_scaling: float = 1.0,
) -> SemanticAttentionRephasedProjection:
    """Use the authenticated LFM2 semantic phase alignment directly."""

    from neural_forge.hf_lfm2_semantic_span_cpcf_alignment import (
        Lfm2SemanticSpanCpcfPhaseAlignment,
    )

    if not isinstance(alignment, Lfm2SemanticSpanCpcfPhaseAlignment):
        raise TypeError(
            "alignment must be Lfm2SemanticSpanCpcfPhaseAlignment."
        )
    if history.token_count != alignment.donor_full_prefix_tokens:
        raise SemanticAttentionLatticeError(
            "Accumulated parent history does not cover the aligned donor prefix."
        )
    alignment.require_endpoint_row_gather_support()
    return project_rephased_semantic_attention_lattice(
        history,
        semantic_alignment_sha256=alignment.alignment_sha256,
        target_to_donor_boundary_positions=(
            alignment.target_to_donor_boundary_positions
        ),
        target_position_ids=alignment.canonical_target_position_ids,
        target_rope_inv_freq=target_rope_inv_freq,
        target_rope_attention_scaling=target_rope_attention_scaling,
    )


def transport_lfm2_phase_probability_preserving_attention(
    alignment: Any,
    *,
    parent_query_post_rope: Any,
    parent_key_post_rope: Any,
    parent_value: Any,
    target_rope_inv_freq: Any,
    target_rope_attention_scaling: float = 1.0,
    query_mode: str = "semantic_endpoints",
) -> SemanticAttentionExpansionTransport:
    """Use one sealed LFM2 span-owner map for exact split-token transport."""

    from neural_forge.hf_lfm2_semantic_span_cpcf_alignment import (
        Lfm2SemanticSpanCpcfPhaseAlignment,
    )

    if not isinstance(alignment, Lfm2SemanticSpanCpcfPhaseAlignment):
        raise TypeError(
            "alignment must be Lfm2SemanticSpanCpcfPhaseAlignment."
        )
    alignment.require_probability_preserving_attention_transport()
    owners = alignment.target_to_donor_span_owner_positions
    if any(value is None for value in owners):  # pragma: no cover - method proves it
        raise SemanticAttentionLatticeError(
            "Authenticated span-owner alignment unexpectedly contains a gap."
        )
    return transport_probability_preserving_attention_expansion(
        semantic_alignment_sha256=alignment.alignment_sha256,
        target_to_donor_boundary_positions=(
            alignment.target_to_donor_boundary_positions
        ),
        target_to_donor_span_owner_positions=tuple(
            int(value) for value in owners if value is not None
        ),
        donor_span_multiplicities=alignment.donor_span_multiplicities,
        parent_query_post_rope=parent_query_post_rope,
        parent_key_post_rope=parent_key_post_rope,
        parent_value=parent_value,
        target_position_ids=alignment.canonical_target_position_ids,
        target_rope_inv_freq=target_rope_inv_freq,
        target_rope_attention_scaling=target_rope_attention_scaling,
        query_mode=query_mode,
    )


__all__ = [
    "LFM2_LEGACY_KEY_LINEAR_SUFFIX",
    "LFM2_LEGACY_KEY_POST_ROPE_SUFFIX",
    "LFM2_LEGACY_QUERY_LINEAR_SUFFIX",
    "LFM2_LEGACY_QUERY_POST_ROPE_SUFFIX",
    "LFM2_PRE_ROPE_KEY_SUFFIX",
    "LFM2_PRE_ROPE_QUERY_SUFFIX",
    "LFM2_VALUE_HISTORY_SUFFIX",
    "LFM2_VALUE_SUFFIX",
    "Lfm2SemanticAttentionCacheCapability",
    "SEMANTIC_ATTENTION_CACHE_CAPABILITY_SCHEMA",
    "SEMANTIC_ATTENTION_EXPANSION_TRANSPORT_SCHEMA",
    "SEMANTIC_ATTENTION_HISTORY_SCHEMA",
    "SEMANTIC_ATTENTION_REPHASED_PROJECTION_SCHEMA",
    "SemanticAttentionExpansionTransport",
    "SemanticAttentionLatticeError",
    "SemanticAttentionPreRopeHistory",
    "SemanticAttentionRephasedProjection",
    "assemble_lfm2_explicit_pre_rope_history",
    "inspect_lfm2_semantic_attention_cache_capability",
    "project_lfm2_phase_rephased_semantic_attention",
    "project_rephased_semantic_attention_lattice",
    "reconstruct_lfm2_normalized_pre_rope_rows",
    "transport_probability_preserving_attention_expansion",
    "transport_lfm2_phase_probability_preserving_attention",
]
