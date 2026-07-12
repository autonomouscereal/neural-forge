"""Causal residual-attention interventions for Qwen-like HF decoders.

This module answers a narrower question than a quality benchmark: *which state
transition is causing a folded child to leave the parent's trajectory?*  A
dense parent is first traced on a bounded cached decode.  The candidate is then
replayed on the same token path while an ephemeral Hugging Face ``DynamicCache``
substitutes parent K, V, or both, and/or an ephemeral ``o_proj`` hook substitutes
the parent's attention residual delta.

The interventions are deliberately post-RoPE and post-cache-update.  The real
Qwen query normalization, RoPE, GQA head expansion, causal mask, softmax, output
projection, residual addition, MLP, and every non-target layer still execute.
Calling the official cache update before substitution preserves cache length and
metadata.  No parameter or persistent buffer is edited.

Two feedback contracts are explicit:

``forced-cache-recurrent``
    Every replay consumes the parent trace's next token.  Cache errors can recur,
    but token-choice feedback is held fixed, making intervention comparisons
    causal over the complete horizon.

``autoregressive``
    The candidate consumes its own greedy token.  Parent intervention evidence
    remains valid only through the first action divergence, so replay stops
    there instead of applying a parent cache from a different prefix.

Raw substitution requires an exact vocabulary chart plus shared hidden width,
layer index, GQA topology, head dimension, and cache layout. Cross-architecture
children need a separately certified coordinate adapter before this oracle is
applicable.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import Tensor, nn

from neural_forge.hf_cached_attention_trace import (
    HFCachedAttentionTrace,
    HFCachedAttentionTraceLimits,
    HFCachedAttentionTraceShapeError,
    collect_hf_cached_attention_trace,
)


ResidualAttentionIntervention = Literal[
    "none",
    "k_cache",
    "v_cache",
    "attention_delta",
    "kv_cache",
    "kv_cache_and_attention_delta",
]
ResidualAttentionFeedback = Literal["forced-cache-recurrent", "autoregressive"]

ORACLE_TRACE_SCHEMA = "neural-forge-residual-attention-oracle-trace-v2"
ORACLE_REPLAY_SCHEMA = "neural-forge-residual-attention-oracle-replay-v2"
ORACLE_SUITE_SCHEMA = "neural-forge-residual-attention-oracle-suite-v2"
ORACLE_COORDINATE_SCHEMA = "neural-forge-residual-attention-coordinate-contract-v1"

_INTERVENTIONS: tuple[ResidualAttentionIntervention, ...] = (
    "none",
    "k_cache",
    "v_cache",
    "attention_delta",
    "kv_cache",
    "kv_cache_and_attention_delta",
)


class HFResidualAttentionOracleError(RuntimeError):
    """Base failure for an unsupported or uncertified intervention."""


class HFResidualAttentionOracleShapeError(HFResidualAttentionOracleError):
    """Parent evidence and candidate runtime are not shape compatible."""


class HFResidualAttentionOracleStateError(HFResidualAttentionOracleError):
    """A model, trace, hook, cache, or identity invariant was violated."""


class HFResidualAttentionOracleLimitError(HFResidualAttentionOracleError):
    """A declared compute, memory, or cardinality bound was exceeded."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise HFResidualAttentionOracleShapeError(f"{name} must be positive.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise HFResidualAttentionOracleShapeError(
            f"{name} must be positive."
        ) from error
    if result <= 0:
        raise HFResidualAttentionOracleShapeError(f"{name} must be positive.")
    return result


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise HFResidualAttentionOracleShapeError(f"{name} must be positive.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise HFResidualAttentionOracleShapeError(
            f"{name} must be positive."
        ) from error
    if not math.isfinite(result) or result <= 0.0:
        raise HFResidualAttentionOracleShapeError(f"{name} must be positive.")
    return result


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise HFResidualAttentionOracleStateError(
            "Identity material is not canonical JSON."
        ) from error


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_sha256(value: Any, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise HFResidualAttentionOracleShapeError(
            f"{name} must be a lowercase SHA-256 string."
        )
    return normalized


def _tensor_sha256(value: Tensor) -> str:
    if (
        not isinstance(value, Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise HFResidualAttentionOracleStateError(
            "Oracle tensors must be materialized dense tensors."
        )
    canonical = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(canonical.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json(list(canonical.shape)))
    digest.update(b"\0")
    digest.update(canonical.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _readonly_cpu(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim == 0 or value.numel() <= 0:
        raise HFResidualAttentionOracleShapeError(f"{name} must be a non-empty tensor.")
    if not bool(torch.isfinite(value.detach()).all()):
        raise HFResidualAttentionOracleStateError(f"{name} is non-finite.")
    return value.detach().to(device="cpu").contiguous().clone()


def _tensor_bytes(value: Tensor) -> int:
    return int(value.numel() * value.element_size())


@dataclass(frozen=True, slots=True)
class HFResidualAttentionOracleLimits:
    """Hard limits for hashing, trace capture, and candidate replay."""

    max_horizon: int = 32
    max_prompt_tokens: int = 512
    max_model_devices: int = 1
    max_trace_bytes: int = 512 * 1024 * 1024
    max_replay_bytes: int = 256 * 1024 * 1024
    fingerprint_chunk_bytes: int = 16 * 1024 * 1024
    cache_rtol: float = 1e-4
    cache_atol: float = 1e-5
    residual_rtol: float = 8e-3
    residual_atol: float = 8e-3
    logit_rtol: float = 5e-3
    logit_atol: float = 5e-3

    def __post_init__(self) -> None:
        for name in (
            "max_horizon",
            "max_prompt_tokens",
            "max_model_devices",
            "max_trace_bytes",
            "max_replay_bytes",
            "fingerprint_chunk_bytes",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        for name in (
            "cache_rtol",
            "cache_atol",
            "residual_rtol",
            "residual_atol",
            "logit_rtol",
            "logit_atol",
        ):
            object.__setattr__(self, name, _positive_float(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class HFOracleModelIdentity:
    """Complete, streaming SHA-256 identity of model config and state."""

    model_class: str
    config_sha256: str
    state_dict_sha256: str
    state_tensor_count: int
    state_byte_count: int
    identity_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_class": self.model_class,
            "config_sha256": self.config_sha256,
            "state_dict_sha256": self.state_dict_sha256,
            "state_tensor_count": self.state_tensor_count,
            "state_byte_count": self.state_byte_count,
            "identity_sha256": self.identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class HFOracleVocabularyChart:
    """Exact child-logit-index to dense-parent-logit-row mapping."""

    child_to_parent_rows: Tensor
    parent_vocab_size: int
    child_vocab_size: int
    identity_sha256: str

    @property
    def is_identity(self) -> bool:
        expected = torch.arange(self.child_vocab_size, dtype=torch.int64)
        return self.parent_vocab_size == self.child_vocab_size and bool(
            torch.equal(self.child_to_parent_rows, expected)
        )

    def as_dict(self) -> dict[str, Any]:
        body = _vocabulary_chart_body(self)
        body["identity_sha256"] = self.identity_sha256
        return body


@dataclass(frozen=True, slots=True)
class HFOracleAttentionCoordinateContract:
    """Execution coordinates that raw parent K/V/delta substitution requires."""

    model_class: str
    decoder_layer_class: str
    attention_class: str
    rotary_embedding_class: str
    layer_count: int
    layer_index: int
    attention_layer_index: int
    hidden_size: int
    query_heads: int
    key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rope_parameters_json: str
    rotary_rope_type: str
    rotary_attention_scaling: float
    use_sliding_window: bool
    sliding_window: int | None
    max_window_layers: int | None
    layer_types: tuple[str, ...]
    selected_layer_type: str
    attention_sliding_window: int | None
    q_norm_class: str
    q_norm_epsilon: float
    k_norm_class: str
    k_norm_epsilon: float
    input_norm_class: str
    input_norm_epsilon: float
    post_attention_norm_class: str
    post_attention_norm_epsilon: float
    rms_norm_epsilon: float
    attention_bias: bool
    projection_bias_presence: tuple[bool, bool, bool, bool]
    attention_dropout: float
    attention_scaling: float
    identity_sha256: str

    def as_dict(self) -> dict[str, Any]:
        body = _attention_coordinate_contract_body(self)
        body["identity_sha256"] = self.identity_sha256
        return body


def _vocabulary_chart_body(chart: HFOracleVocabularyChart) -> dict[str, Any]:
    return {
        "schema": "neural-forge-residual-attention-vocabulary-chart-v1",
        "parent_vocab_size": chart.parent_vocab_size,
        "child_vocab_size": chart.child_vocab_size,
        "child_to_parent_rows_sha256": _tensor_sha256(chart.child_to_parent_rows),
        "is_identity": chart.is_identity,
        "rows_unique": True,
        "rows_in_parent_range": True,
    }


def build_hf_oracle_vocabulary_chart(
    child_to_parent_rows: Sequence[int] | Tensor,
    *,
    parent_vocab_size: int,
) -> HFOracleVocabularyChart:
    """Build a strict injective vocabulary chart for parent trace projection."""

    parent_size = _positive_int(parent_vocab_size, "parent_vocab_size")
    raw = torch.as_tensor(child_to_parent_rows, device="cpu")
    if (
        raw.ndim != 1
        or raw.numel() <= 0
        or raw.dtype == torch.bool
        or raw.is_floating_point()
        or raw.is_complex()
    ):
        raise HFResidualAttentionOracleShapeError(
            "child_to_parent_rows must be one non-empty integer vector."
        )
    rows = raw.to(dtype=torch.int64).contiguous().clone()
    if int(rows.min().item()) < 0 or int(rows.max().item()) >= parent_size:
        raise HFResidualAttentionOracleShapeError(
            "Vocabulary chart contains an out-of-range parent row."
        )
    if int(torch.unique(rows).numel()) != int(rows.numel()):
        raise HFResidualAttentionOracleShapeError(
            "Vocabulary chart parent rows must be unique."
        )
    provisional = HFOracleVocabularyChart(
        child_to_parent_rows=rows,
        parent_vocab_size=parent_size,
        child_vocab_size=int(rows.numel()),
        identity_sha256="0" * 64,
    )
    result = replace(
        provisional,
        identity_sha256=_json_sha256(_vocabulary_chart_body(provisional)),
    )
    _validate_vocabulary_chart(result)
    return result


def _validate_vocabulary_chart(chart: HFOracleVocabularyChart) -> None:
    if not isinstance(chart, HFOracleVocabularyChart):
        raise HFResidualAttentionOracleShapeError(
            "vocabulary chart has an unsupported type."
        )
    rows = chart.child_to_parent_rows
    if (
        not isinstance(rows, Tensor)
        or rows.device.type != "cpu"
        or rows.dtype != torch.int64
        or not rows.is_contiguous()
        or rows.ndim != 1
        or int(rows.numel()) != chart.child_vocab_size
        or chart.child_vocab_size <= 0
        or chart.parent_vocab_size <= 0
        or int(rows.min().item()) < 0
        or int(rows.max().item()) >= chart.parent_vocab_size
        or int(torch.unique(rows).numel()) != int(rows.numel())
    ):
        raise HFResidualAttentionOracleStateError(
            "Vocabulary chart tensor or cardinality changed."
        )
    if _json_sha256(_vocabulary_chart_body(chart)) != chart.identity_sha256:
        raise HFResidualAttentionOracleStateError(
            "Vocabulary chart identity or mapping rows changed."
        )


def _qualified_class_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise HFResidualAttentionOracleShapeError(
            f"{name} must be a finite number."
        ) from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive " if positive else ""
        raise HFResidualAttentionOracleShapeError(
            f"{name} must be a finite {qualifier}number."
        )
    return result


def _optional_nonnegative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise HFResidualAttentionOracleShapeError(
            f"{name} must be a nonnegative integer or None."
        )
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise HFResidualAttentionOracleShapeError(
            f"{name} must be a nonnegative integer or None."
        ) from error
    if result < 0:
        raise HFResidualAttentionOracleShapeError(
            f"{name} must be a nonnegative integer or None."
        )
    return result


def _norm_contract(module: Any, name: str) -> tuple[str, float]:
    if not isinstance(module, nn.Module):
        raise HFResidualAttentionOracleShapeError(f"Selected Qwen layer lacks {name}.")
    epsilon = None
    for attribute in ("variance_epsilon", "eps", "epsilon"):
        candidate = getattr(module, attribute, None)
        if candidate is not None:
            epsilon = candidate
            break
    if epsilon is None:
        raise HFResidualAttentionOracleShapeError(
            f"Selected Qwen {name} does not expose its epsilon."
        )
    return _qualified_class_name(module), _finite_float(
        epsilon, f"{name} epsilon", positive=True
    )


def _attention_coordinate_contract_body(
    contract: HFOracleAttentionCoordinateContract,
) -> dict[str, Any]:
    return {
        "schema": ORACLE_COORDINATE_SCHEMA,
        "model_class": contract.model_class,
        "decoder_layer_class": contract.decoder_layer_class,
        "attention_class": contract.attention_class,
        "rotary_embedding_class": contract.rotary_embedding_class,
        "layer_count": contract.layer_count,
        "layer_index": contract.layer_index,
        "attention_layer_index": contract.attention_layer_index,
        "hidden_size": contract.hidden_size,
        "query_heads": contract.query_heads,
        "key_value_heads": contract.key_value_heads,
        "head_dim": contract.head_dim,
        "max_position_embeddings": contract.max_position_embeddings,
        "rope_parameters": json.loads(contract.rope_parameters_json),
        "rotary_rope_type": contract.rotary_rope_type,
        "rotary_attention_scaling": contract.rotary_attention_scaling,
        "use_sliding_window": contract.use_sliding_window,
        "sliding_window": contract.sliding_window,
        "max_window_layers": contract.max_window_layers,
        "layer_types": list(contract.layer_types),
        "selected_layer_type": contract.selected_layer_type,
        "attention_sliding_window": contract.attention_sliding_window,
        "q_norm_class": contract.q_norm_class,
        "q_norm_epsilon": contract.q_norm_epsilon,
        "k_norm_class": contract.k_norm_class,
        "k_norm_epsilon": contract.k_norm_epsilon,
        "input_norm_class": contract.input_norm_class,
        "input_norm_epsilon": contract.input_norm_epsilon,
        "post_attention_norm_class": contract.post_attention_norm_class,
        "post_attention_norm_epsilon": contract.post_attention_norm_epsilon,
        "rms_norm_epsilon": contract.rms_norm_epsilon,
        "attention_bias": contract.attention_bias,
        "projection_bias_presence": list(contract.projection_bias_presence),
        "attention_dropout": contract.attention_dropout,
        "attention_scaling": contract.attention_scaling,
    }


def _build_attention_coordinate_contract(
    model: nn.Module, layer_index: int
) -> HFOracleAttentionCoordinateContract:
    layers = _base_layers(model)
    if layer_index < 0 or layer_index >= len(layers):
        raise HFResidualAttentionOracleShapeError(
            "Attention coordinate layer index is unavailable."
        )
    layer = layers[layer_index]
    attention = getattr(layer, "self_attn", None)
    if not isinstance(attention, nn.Module):
        raise HFResidualAttentionOracleShapeError(
            "Selected Qwen layer lacks self_attn."
        )
    config = getattr(model, "config", getattr(attention, "config", None))
    if config is None:
        raise HFResidualAttentionOracleShapeError(
            "Attention coordinate contract requires a model config."
        )
    base = getattr(model, "model", None)
    if base is not None and not isinstance(getattr(base, "layers", None), Sequence):
        nested = getattr(base, "model", None)
        if nested is not None:
            base = nested
    rotary = getattr(base, "rotary_emb", None)
    if not isinstance(rotary, nn.Module):
        rotary = getattr(attention, "rotary_emb", None)
    if not isinstance(rotary, nn.Module):
        raise HFResidualAttentionOracleShapeError(
            "Qwen runtime lacks a certifiable rotary embedding module."
        )

    q_norm_class, q_norm_epsilon = _norm_contract(
        getattr(attention, "q_norm", None), "q_norm"
    )
    k_norm_class, k_norm_epsilon = _norm_contract(
        getattr(attention, "k_norm", None), "k_norm"
    )
    input_norm_class, input_norm_epsilon = _norm_contract(
        getattr(layer, "input_layernorm", None), "input_layernorm"
    )
    post_norm_class, post_norm_epsilon = _norm_contract(
        getattr(layer, "post_attention_layernorm", None),
        "post_attention_layernorm",
    )
    layer_types_raw = getattr(config, "layer_types", None)
    if layer_types_raw is None:
        layer_types = tuple("full_attention" for _ in layers)
    elif isinstance(layer_types_raw, Sequence) and not isinstance(
        layer_types_raw, (str, bytes)
    ):
        layer_types = tuple(str(value) for value in layer_types_raw)
    else:
        raise HFResidualAttentionOracleShapeError(
            "config.layer_types must be a sequence when present."
        )
    if len(layer_types) != len(layers):
        raise HFResidualAttentionOracleShapeError(
            "config.layer_types does not match the decoder layer count."
        )
    rope_payload = {
        "rope_parameters": getattr(config, "rope_parameters", None),
        "rope_theta": getattr(config, "rope_theta", None),
        "rope_scaling": getattr(config, "rope_scaling", None),
    }
    rope_json = _canonical_json(rope_payload).decode("ascii")
    projections = tuple(
        getattr(attention, name, None)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj")
    )
    if any(not isinstance(value, nn.Module) for value in projections):
        raise HFResidualAttentionOracleShapeError(
            "Selected Qwen attention lacks Q/K/V/O projection modules."
        )
    attention_bias = bool(getattr(config, "attention_bias", False))
    projection_bias_presence = tuple(
        isinstance(value._parameters.get("bias"), Tensor)
        if "bias" in value._parameters
        else attention_bias
        for value in projections
        if isinstance(value, nn.Module)
    )
    provisional = HFOracleAttentionCoordinateContract(
        model_class=_qualified_class_name(model),
        decoder_layer_class=_qualified_class_name(layer),
        attention_class=_qualified_class_name(attention),
        rotary_embedding_class=_qualified_class_name(rotary),
        layer_count=len(layers),
        layer_index=layer_index,
        attention_layer_index=int(getattr(attention, "layer_idx", -1)),
        hidden_size=_positive_int(getattr(config, "hidden_size", 0), "hidden_size"),
        query_heads=_positive_int(
            getattr(config, "num_attention_heads", 0), "num_attention_heads"
        ),
        key_value_heads=_positive_int(
            getattr(config, "num_key_value_heads", 0), "num_key_value_heads"
        ),
        head_dim=_positive_int(
            getattr(config, "head_dim", getattr(attention, "head_dim", 0)),
            "head_dim",
        ),
        max_position_embeddings=_positive_int(
            getattr(config, "max_position_embeddings", 0),
            "max_position_embeddings",
        ),
        rope_parameters_json=rope_json,
        rotary_rope_type=str(getattr(rotary, "rope_type", "")).strip(),
        rotary_attention_scaling=_finite_float(
            getattr(rotary, "attention_scaling", 1.0),
            "rotary attention scaling",
            positive=True,
        ),
        use_sliding_window=bool(getattr(config, "use_sliding_window", False)),
        sliding_window=_optional_nonnegative_int(
            getattr(config, "sliding_window", None), "sliding_window"
        ),
        max_window_layers=_optional_nonnegative_int(
            getattr(config, "max_window_layers", None), "max_window_layers"
        ),
        layer_types=layer_types,
        selected_layer_type=layer_types[layer_index],
        attention_sliding_window=_optional_nonnegative_int(
            getattr(attention, "sliding_window", None),
            "attention sliding_window",
        ),
        q_norm_class=q_norm_class,
        q_norm_epsilon=q_norm_epsilon,
        k_norm_class=k_norm_class,
        k_norm_epsilon=k_norm_epsilon,
        input_norm_class=input_norm_class,
        input_norm_epsilon=input_norm_epsilon,
        post_attention_norm_class=post_norm_class,
        post_attention_norm_epsilon=post_norm_epsilon,
        rms_norm_epsilon=_finite_float(
            getattr(config, "rms_norm_eps", 0.0),
            "rms_norm_eps",
            positive=True,
        ),
        attention_bias=attention_bias,
        projection_bias_presence=projection_bias_presence,  # type: ignore[arg-type]
        attention_dropout=_finite_float(
            getattr(attention, "attention_dropout", 0.0), "attention dropout"
        ),
        attention_scaling=_finite_float(
            getattr(attention, "scaling", 0.0),
            "attention scaling",
            positive=True,
        ),
        identity_sha256="0" * 64,
    )
    result = replace(
        provisional,
        identity_sha256=_json_sha256(_attention_coordinate_contract_body(provisional)),
    )
    _validate_attention_coordinate_contract(result)
    return result


def _validate_attention_coordinate_contract(
    contract: HFOracleAttentionCoordinateContract,
) -> None:
    if not isinstance(contract, HFOracleAttentionCoordinateContract):
        raise HFResidualAttentionOracleShapeError(
            "attention coordinate contract has an unsupported type."
        )
    if _json_sha256(_attention_coordinate_contract_body(contract)) != (
        contract.identity_sha256
    ):
        raise HFResidualAttentionOracleStateError(
            "Attention coordinate contract identity or fields changed."
        )


def _coordinate_contract_mismatch(
    parent: HFOracleAttentionCoordinateContract,
    candidate: HFOracleAttentionCoordinateContract,
) -> str | None:
    parent_body = _attention_coordinate_contract_body(parent)
    candidate_body = _attention_coordinate_contract_body(candidate)
    for name in parent_body:
        if parent_body[name] != candidate_body.get(name):
            return name
    return None


def fingerprint_hf_oracle_model(
    model: nn.Module,
    *,
    chunk_bytes: int = 16 * 1024 * 1024,
) -> HFOracleModelIdentity:
    """Hash every model state byte without materializing the whole state on CPU."""

    if not isinstance(model, nn.Module):
        raise HFResidualAttentionOracleShapeError("model must be a torch module.")
    chunk_limit = _positive_int(chunk_bytes, "chunk_bytes")
    config = getattr(model, "config", None)
    to_dict = getattr(config, "to_dict", None)
    if not callable(to_dict):
        raise HFResidualAttentionOracleStateError(
            "Model config must expose to_dict() for identity binding."
        )
    config_sha256 = _json_sha256(to_dict())
    state = model.state_dict()
    if not isinstance(state, Mapping) or not state:
        raise HFResidualAttentionOracleStateError(
            "Model state_dict must be a non-empty mapping."
        )
    digest = hashlib.sha256()
    digest.update(b"neural-forge-residual-attention-oracle-state-v1\0")
    state_bytes = 0
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise HFResidualAttentionOracleStateError(
                "State entries must be string-named tensors."
            )
        if (
            value.layout != torch.strided
            or value.device.type == "meta"
            or not value.is_contiguous()
        ):
            raise HFResidualAttentionOracleStateError(
                f"State tensor {name!r} must be materialized, dense, and contiguous."
            )
        detached = value.detach()
        byte_view = detached.reshape(-1).view(torch.uint8).reshape(-1)
        total_bytes = int(byte_view.numel())
        state_bytes += total_bytes
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(detached.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(_canonical_json(list(detached.shape)))
        digest.update(b"\0")
        for offset in range(0, total_bytes, chunk_limit):
            chunk = byte_view[offset : offset + chunk_limit]
            digest.update(chunk.to(device="cpu").numpy().tobytes(order="C"))
        digest.update(b"\0")
    state_sha256 = digest.hexdigest()
    model_class = f"{type(model).__module__}.{type(model).__qualname__}"
    body = {
        "schema": "neural-forge-residual-attention-oracle-model-v1",
        "model_class": model_class,
        "config_sha256": config_sha256,
        "state_dict_sha256": state_sha256,
        "state_tensor_count": len(state),
        "state_byte_count": state_bytes,
    }
    return HFOracleModelIdentity(
        model_class=model_class,
        config_sha256=config_sha256,
        state_dict_sha256=state_sha256,
        state_tensor_count=len(state),
        state_byte_count=state_bytes,
        identity_sha256=_json_sha256(body),
    )


def fingerprint_hf_oracle_tokenizer(tokenizer: Any) -> str:
    """Bind exact token strings, IDs, special IDs, and tokenizer class."""

    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise HFResidualAttentionOracleShapeError("tokenizer must expose get_vocab().")
    vocab = get_vocab()
    if not isinstance(vocab, Mapping) or not vocab:
        raise HFResidualAttentionOracleShapeError(
            "tokenizer vocabulary must be a non-empty mapping."
        )
    normalized_vocab: list[tuple[str, int]] = []
    for token, token_id in vocab.items():
        if isinstance(token_id, bool):
            raise HFResidualAttentionOracleShapeError("tokenizer IDs must be integers.")
        try:
            normalized_id = operator.index(token_id)
        except (TypeError, ValueError, OverflowError) as error:
            raise HFResidualAttentionOracleShapeError(
                "tokenizer IDs must be integers."
            ) from error
        normalized_vocab.append((str(token), normalized_id))
    specials = {
        name: getattr(tokenizer, name, None)
        for name in (
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "unk_token_id",
            "mask_token_id",
        )
    }
    return _json_sha256(
        {
            "schema": "neural-forge-residual-attention-oracle-tokenizer-v1",
            "tokenizer_class": (
                f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
            ),
            "vocab": sorted(normalized_vocab),
            "special_token_ids": specials,
        }
    )


def _output_field(output: Any, name: str) -> Any:
    value = getattr(output, name, None)
    if value is None and isinstance(output, Mapping):
        value = output.get(name)
    return value


def _output_logits(output: Any) -> Tensor:
    logits = _output_field(output, "logits")
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise HFResidualAttentionOracleStateError(
            "Causal model output must expose [batch, sequence, vocab] logits."
        )
    if not bool(torch.isfinite(logits.detach()).all()):
        raise HFResidualAttentionOracleStateError("Model logits are non-finite.")
    return logits


def _base_layers(model: nn.Module) -> Sequence[nn.Module]:
    candidates = (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "model", None), "layers", None),
        getattr(model, "layers", None),
    )
    for value in candidates:
        if isinstance(value, (Sequence, nn.ModuleList)):
            return value
    raise HFResidualAttentionOracleShapeError(
        "Model must expose decoder layers at model.layers."
    )


def _embedding_device(model: nn.Module) -> torch.device:
    getter = getattr(model, "get_input_embeddings", None)
    if callable(getter):
        weight = getattr(getter(), "weight", None)
        if isinstance(weight, Tensor):
            return weight.device
    for parameter in model.parameters():
        return parameter.device
    raise HFResidualAttentionOracleShapeError("Model has no parameter device.")


def _model_vocabulary_sizes(model: nn.Module) -> tuple[int, int]:
    input_getter = getattr(model, "get_input_embeddings", None)
    output_getter = getattr(model, "get_output_embeddings", None)
    if not callable(input_getter) or not callable(output_getter):
        raise HFResidualAttentionOracleShapeError(
            "Model must expose input and output embeddings."
        )
    input_weight = getattr(input_getter(), "weight", None)
    output_weight = getattr(output_getter(), "weight", None)
    if (
        not isinstance(input_weight, Tensor)
        or not isinstance(output_weight, Tensor)
        or input_weight.ndim != 2
        or output_weight.ndim != 2
    ):
        raise HFResidualAttentionOracleShapeError(
            "Model vocabulary weights must be rank-two tensors."
        )
    return int(input_weight.shape[0]), int(output_weight.shape[0])


def _model_devices(model: nn.Module) -> tuple[torch.device, ...]:
    devices = {
        value.device
        for value in tuple(model.parameters()) + tuple(model.buffers())
        if value.device.type != "meta"
    }
    return tuple(sorted(devices, key=str))


def _hook_inventory(model: nn.Module) -> tuple[tuple[int, ...], ...]:
    return tuple(
        (
            id(module),
            *tuple(int(value) for value in module._forward_pre_hooks),
            -1,
            *tuple(int(value) for value in module._forward_hooks),
            -2,
            *tuple(int(value) for value in module._backward_hooks),
        )
        for module in model.modules()
    )


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


def _model_forward(
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
    }
    return model(**_forward_kwargs(model, values))


def _cache_layer(cache: Any, layer_index: int) -> tuple[Tensor, Tensor]:
    layers = getattr(cache, "layers", None)
    if isinstance(layers, Sequence) and layer_index < len(layers):
        layer = layers[layer_index]
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if isinstance(keys, Tensor) and isinstance(values, Tensor):
            return keys, values
    raise HFResidualAttentionOracleStateError(
        "Cache does not expose mutable layers[layer].keys/values."
    )


def _new_dynamic_cache(model: nn.Module) -> Any:
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError as error:  # pragma: no cover - optional dependency
        raise HFResidualAttentionOracleStateError(
            "Hugging Face DynamicCache is required for Qwen intervention."
        ) from error
    try:
        return DynamicCache(config=getattr(model, "config", None))
    except TypeError:
        return DynamicCache()


def _module_input(
    args: tuple[Any, ...], kwargs: Mapping[str, Any], name: str
) -> Tensor:
    if args and isinstance(args[0], Tensor):
        return args[0]
    for key in ("hidden_states", "x", "input"):
        value = kwargs.get(key)
        if isinstance(value, Tensor):
            return value
    raise HFResidualAttentionOracleStateError(
        f"Hook {name} did not receive hidden states."
    )


class _FullDeltaCapture:
    def __init__(self, output_projection: nn.Module) -> None:
        self.active = False
        self.count = 0
        self.value: Tensor | None = None

        def hook(
            _module: nn.Module,
            _args: tuple[Any, ...],
            _kwargs: Mapping[str, Any],
            output: Any,
        ) -> None:
            if not self.active:
                return
            self.count += 1
            if self.count != 1 or not isinstance(output, Tensor):
                raise HFResidualAttentionOracleStateError(
                    "Parent o_proj hook did not fire exactly once with a tensor."
                )
            self.value = _readonly_cpu(output, "parent attention delta")

        self.handle = output_projection.register_forward_hook(hook, with_kwargs=True)

    def begin(self) -> None:
        if self.active:
            raise HFResidualAttentionOracleStateError(
                "Parent delta capture is already active."
            )
        self.active = True
        self.count = 0
        self.value = None

    def finish(self) -> Tensor:
        self.active = False
        if self.count != 1 or self.value is None:
            raise HFResidualAttentionOracleStateError(
                "Parent o_proj hook did not capture exactly one delta."
            )
        return self.value

    def remove(self) -> None:
        self.active = False
        self.handle.remove()


def _runtime_restore_context(
    model: nn.Module,
) -> tuple[list[tuple[nn.Module, bool]], Any, dict[str, Any]]:
    modes = [(module, bool(module.training)) for module in model.modules()]
    config = getattr(model, "config", None)
    values: dict[str, Any] = {}
    for name in ("use_cache", "output_attentions", "output_hidden_states"):
        if config is not None and hasattr(config, name):
            values[name] = getattr(config, name)
    return modes, config, values


def _set_inference_runtime(model: nn.Module, config: Any) -> None:
    model.eval()
    if config is not None:
        for name, value in (
            ("use_cache", True),
            ("output_attentions", False),
            ("output_hidden_states", False),
        ):
            if hasattr(config, name):
                setattr(config, name, value)


def _restore_runtime(
    modes: Sequence[tuple[nn.Module, bool]], config: Any, values: Mapping[str, Any]
) -> None:
    for name, value in values.items():
        setattr(config, name, value)
    for module, was_training in modes:
        module.training = was_training


@dataclass(frozen=True, slots=True)
class HFParentResidualAttentionOracleTrace:
    """Certified dense-parent evidence, detached to CPU for sequential replay."""

    cached_trace: HFCachedAttentionTrace
    full_attention_deltas: tuple[Tensor, ...]
    parent_model_identity: HFOracleModelIdentity
    vocabulary_chart: HFOracleVocabularyChart
    attention_coordinate_contract: HFOracleAttentionCoordinateContract
    tokenizer_identity_sha256: str
    parent_prompt_token_ids: tuple[int, ...]
    parent_selected_token_ids: tuple[int, ...]
    trace_byte_count: int
    identity_sha256: str

    def as_dict(self) -> dict[str, Any]:
        body = _parent_trace_identity_body(self)
        body["identity_sha256"] = self.identity_sha256
        return body


def _parent_trace_identity_body(
    trace: HFParentResidualAttentionOracleTrace,
) -> dict[str, Any]:
    cached = trace.cached_trace.as_dict()
    return {
        "schema": ORACLE_TRACE_SCHEMA,
        "cached_trace_sha256": cached["trace_sha256"],
        "parent_model_identity": trace.parent_model_identity.as_dict(),
        "vocabulary_chart": trace.vocabulary_chart.as_dict(),
        "attention_coordinate_contract": (
            trace.attention_coordinate_contract.as_dict()
        ),
        "tokenizer_identity_sha256": trace.tokenizer_identity_sha256,
        "layer_index": trace.cached_trace.layer_index,
        "group_indices": list(trace.cached_trace.group_indices),
        "lane": trace.cached_trace.lane,
        "horizon": trace.cached_trace.horizon,
        "prompt_token_ids": list(trace.cached_trace.prompt_token_ids),
        "selected_token_ids": list(trace.cached_trace.selected_token_ids),
        "parent_prompt_token_ids": list(trace.parent_prompt_token_ids),
        "parent_selected_token_ids": list(trace.parent_selected_token_ids),
        "prompt_rows_identity": (
            trace.parent_prompt_token_ids == trace.cached_trace.prompt_token_ids
        ),
        "parent_logits_projected_to_child_vocabulary": True,
        "full_attention_delta_sha256s": [
            _tensor_sha256(value) for value in trace.full_attention_deltas
        ],
        "full_attention_delta_shapes": [
            list(value.shape) for value in trace.full_attention_deltas
        ],
        "trace_byte_count": trace.trace_byte_count,
        "vocabulary_chart_byte_count": _tensor_bytes(
            trace.vocabulary_chart.child_to_parent_rows
        ),
        "parent_state_unchanged": True,
        "gradient_free": True,
        "training_used": False,
    }


def _validate_parent_trace(trace: HFParentResidualAttentionOracleTrace) -> None:
    if not isinstance(trace, HFParentResidualAttentionOracleTrace):
        raise HFResidualAttentionOracleShapeError(
            "parent_trace must be HFParentResidualAttentionOracleTrace."
        )
    if len(trace.full_attention_deltas) != trace.cached_trace.horizon:
        raise HFResidualAttentionOracleStateError(
            "Parent full-delta count differs from cached trace horizon."
        )
    _validate_vocabulary_chart(trace.vocabulary_chart)
    _validate_attention_coordinate_contract(trace.attention_coordinate_contract)
    if (
        trace.attention_coordinate_contract.layer_index
        != trace.cached_trace.layer_index
        or trace.attention_coordinate_contract.query_heads
        != trace.cached_trace.query_heads
        or trace.attention_coordinate_contract.key_value_heads
        != trace.cached_trace.key_value_heads
        or trace.attention_coordinate_contract.head_dim != trace.cached_trace.head_dim
    ):
        raise HFResidualAttentionOracleStateError(
            "Parent trace and attention coordinate contract disagree."
        )
    if (
        len(trace.parent_prompt_token_ids) != len(trace.cached_trace.prompt_token_ids)
        or len(trace.parent_selected_token_ids) != trace.cached_trace.horizon
    ):
        raise HFResidualAttentionOracleStateError(
            "Parent and canonical token paths have incompatible lengths."
        )
    expected_parent_prompt = tuple(
        int(value)
        for value in _canonical_to_parent_ids(
            trace.cached_trace.prompt_token_ids,
            trace.vocabulary_chart,
            name="canonical prompt trace",
        )
        .reshape(-1)
        .tolist()
    )
    expected_parent_selected = tuple(
        int(value)
        for value in _canonical_to_parent_ids(
            trace.cached_trace.selected_token_ids,
            trace.vocabulary_chart,
            name="canonical selected-token trace",
        )
        .reshape(-1)
        .tolist()
    )
    if expected_parent_prompt != trace.parent_prompt_token_ids or (
        expected_parent_selected != trace.parent_selected_token_ids
    ):
        raise HFResidualAttentionOracleStateError(
            "Canonical and parent token paths violate the vocabulary chart."
        )
    for step in trace.cached_trace.steps:
        if int(step.action_logits.shape[-1]) != trace.vocabulary_chart.child_vocab_size:
            raise HFResidualAttentionOracleStateError(
                "Canonical parent logits do not match vocabulary chart width."
            )
    expected = _json_sha256(_parent_trace_identity_body(trace))
    if expected != trace.identity_sha256:
        raise HFResidualAttentionOracleStateError(
            "Parent oracle trace identity or a trace tensor changed."
        )
    expected_bytes = (
        trace.cached_trace.capture_bytes
        + sum(_tensor_bytes(value) for value in trace.full_attention_deltas)
        + _tensor_bytes(trace.vocabulary_chart.child_to_parent_rows)
    )
    if expected_bytes != trace.trace_byte_count:
        raise HFResidualAttentionOracleStateError(
            "Parent oracle trace byte count is not exact."
        )


def _recapture_parent_full_deltas(
    model: nn.Module,
    cached: HFCachedAttentionTrace,
    limits: HFResidualAttentionOracleLimits,
) -> tuple[tuple[Tensor, ...], int]:
    layer = _base_layers(model)[cached.layer_index]
    attention = getattr(layer, "self_attn", None)
    output_projection = getattr(attention, "o_proj", None)
    if not isinstance(output_projection, nn.Module):
        raise HFResidualAttentionOracleShapeError(
            "Selected Qwen attention has no o_proj module."
        )
    device = _embedding_device(model)
    prompt = torch.tensor([cached.prompt_token_ids], dtype=torch.long, device=device)
    prefix = prompt.clone()
    next_input = prompt
    past = _new_dynamic_cache(model)
    deltas: list[Tensor] = []
    captured_bytes = 0
    modes, config, config_values = _runtime_restore_context(model)
    hooks_before = _hook_inventory(model)
    hook = _FullDeltaCapture(output_projection)
    try:
        _set_inference_runtime(model, config)
        with torch.inference_mode():
            for step_index, cached_step in enumerate(cached.steps):
                mask = torch.ones(
                    (1, int(prefix.shape[1])), dtype=torch.long, device=device
                )
                hook.begin()
                output = _model_forward(model, next_input, mask, past_key_values=past)
                delta = hook.finish()
                logits = _readonly_cpu(
                    _output_logits(output)[:, -1, :], "parent replay logits"
                )
                if not bool(
                    torch.allclose(
                        logits.to(torch.float32),
                        cached_step.action_logits.to(torch.float32),
                        rtol=limits.logit_rtol,
                        atol=limits.logit_atol,
                    )
                ):
                    raise HFResidualAttentionOracleStateError(
                        "Parent full-delta replay logits differ from certified trace."
                    )
                if (
                    delta.shape[0] != 1
                    or delta.shape[-1] != (cached_step.attention_delta.shape[-1])
                ):
                    raise HFResidualAttentionOracleShapeError(
                        "Parent full attention delta has an incompatible shape."
                    )
                if not bool(
                    torch.allclose(
                        delta[:, -1, :].to(torch.float32),
                        cached_step.attention_delta.to(torch.float32),
                        rtol=limits.residual_rtol,
                        atol=limits.residual_atol,
                    )
                ):
                    raise HFResidualAttentionOracleStateError(
                        "Parent full delta does not end in the certified trace delta."
                    )
                key_cache, value_cache = _cache_layer(
                    _output_field(output, "past_key_values"), cached.layer_index
                )
                for actual, expected, name in (
                    (key_cache, cached_step.layer_key_cache, "K"),
                    (value_cache, cached_step.layer_value_cache, "V"),
                ):
                    if actual.shape != expected.shape or not bool(
                        torch.allclose(
                            actual.detach().to(device="cpu", dtype=torch.float32),
                            expected.to(torch.float32),
                            rtol=limits.cache_rtol,
                            atol=limits.cache_atol,
                        )
                    ):
                        raise HFResidualAttentionOracleStateError(
                            f"Parent full-delta replay {name} cache differs from trace."
                        )
                deltas.append(delta)
                captured_bytes += _tensor_bytes(delta)
                if captured_bytes > limits.max_trace_bytes:
                    raise HFResidualAttentionOracleLimitError(
                        "Parent full deltas exceeded max_trace_bytes."
                    )
                past = _output_field(output, "past_key_values")
                if step_index + 1 < cached.horizon:
                    token = cached.selected_token_ids[step_index]
                    next_input = torch.tensor(
                        [[token]], dtype=torch.long, device=device
                    )
                    prefix = torch.cat((prefix, next_input), dim=1)
    finally:
        hook.remove()
        _restore_runtime(modes, config, config_values)
    if _hook_inventory(model) != hooks_before:
        raise HFResidualAttentionOracleStateError(
            "Parent full-delta capture did not restore hooks exactly."
        )
    return tuple(deltas), captured_bytes


def _canonical_to_parent_ids(
    values: Any,
    chart: HFOracleVocabularyChart,
    *,
    name: str,
) -> Tensor:
    raw = torch.as_tensor(values, device="cpu")
    if (
        raw.numel() <= 0
        or raw.dtype == torch.bool
        or raw.is_floating_point()
        or raw.is_complex()
    ):
        raise HFResidualAttentionOracleShapeError(
            f"{name} must contain integer child-vocabulary IDs."
        )
    canonical = raw.to(dtype=torch.int64)
    if int(canonical.min().item()) < 0 or int(canonical.max().item()) >= (
        chart.child_vocab_size
    ):
        raise HFResidualAttentionOracleShapeError(
            f"{name} contains an ID outside the child vocabulary chart."
        )
    flat_parent = chart.child_to_parent_rows.index_select(0, canonical.reshape(-1))
    return flat_parent.reshape(canonical.shape).contiguous()


def _parent_inverse_chart(chart: HFOracleVocabularyChart) -> Tensor:
    inverse = torch.full(
        (chart.parent_vocab_size,), -1, dtype=torch.int64, device="cpu"
    )
    inverse.index_copy_(
        0,
        chart.child_to_parent_rows,
        torch.arange(chart.child_vocab_size, dtype=torch.int64),
    )
    return inverse


def _parent_path_to_canonical(
    values: Sequence[int],
    chart: HFOracleVocabularyChart,
    *,
    name: str,
) -> tuple[int, ...]:
    raw = torch.tensor(tuple(values), dtype=torch.int64)
    if (
        raw.numel() <= 0
        or int(raw.min().item()) < 0
        or int(raw.max().item()) >= (chart.parent_vocab_size)
    ):
        raise HFResidualAttentionOracleStateError(
            f"Parent {name} contains an out-of-range vocabulary row."
        )
    canonical = _parent_inverse_chart(chart).index_select(0, raw)
    if bool(torch.any(canonical < 0)):
        raise HFResidualAttentionOracleStateError(
            f"Parent {name} selected an unmapped padded vocabulary row."
        )
    return tuple(int(value) for value in canonical.tolist())


def _canonicalize_parent_cached_trace(
    raw: HFCachedAttentionTrace,
    chart: HFOracleVocabularyChart,
) -> HFCachedAttentionTrace:
    if any(
        int(step.action_logits.shape[-1]) != chart.parent_vocab_size
        for step in raw.steps
    ):
        raise HFResidualAttentionOracleShapeError(
            "Dense-parent logit width differs from vocabulary chart parent width."
        )
    canonical_prompt = _parent_path_to_canonical(
        raw.prompt_token_ids, chart, name="prompt path"
    )
    if raw.lane == "greedy":
        canonical_selected = _parent_path_to_canonical(
            raw.selected_token_ids, chart, name="greedy action path"
        )
    else:
        canonical_selected = _parent_path_to_canonical(
            raw.selected_token_ids, chart, name="forced action path"
        )
    canonical_consumed = (
        _parent_path_to_canonical(
            raw.consumed_token_ids, chart, name="consumed token path"
        )
        if raw.consumed_token_ids
        else ()
    )
    rows = chart.child_to_parent_rows
    steps = []
    for step_index, step in enumerate(raw.steps):
        projected_logits = step.action_logits.index_select(-1, rows).contiguous()
        if (
            raw.lane == "greedy"
            and int(torch.argmax(projected_logits, dim=-1).item())
            != canonical_selected[step_index]
        ):
            raise HFResidualAttentionOracleStateError(
                "Vocabulary projection changed the parent greedy action."
            )
        old_logit_bytes = _tensor_bytes(step.action_logits)
        new_logit_bytes = _tensor_bytes(projected_logits)
        steps.append(
            replace(
                step,
                selected_token_id=canonical_selected[step_index],
                action_logits=projected_logits,
                captured_byte_count=(
                    step.captured_byte_count - old_logit_bytes + new_logit_bytes
                ),
            )
        )
    return replace(
        raw,
        prompt_token_ids=canonical_prompt,
        consumed_token_ids=canonical_consumed,
        selected_token_ids=canonical_selected,
        steps=tuple(steps),
        capture_bytes=sum(step.captured_byte_count for step in steps),
    )


def capture_parent_residual_attention_oracle_trace(
    model: nn.Module,
    input_ids: Any,
    *,
    tokenizer_identity_sha256: str,
    layer_index: int,
    group_indices: Sequence[int],
    horizon: int,
    sequence_id: str = "prompt-0",
    lane: Literal["forced", "greedy"] = "greedy",
    forced_token_ids: Sequence[int] | Tensor | None = None,
    parent_token_rows: Sequence[int] | Tensor | None = None,
    limits: HFResidualAttentionOracleLimits | None = None,
) -> HFParentResidualAttentionOracleTrace:
    """Capture a bound parent trace and prove that capture changed no state."""

    controls = limits or HFResidualAttentionOracleLimits()
    if not isinstance(controls, HFResidualAttentionOracleLimits):
        raise HFResidualAttentionOracleShapeError(
            "limits must be HFResidualAttentionOracleLimits."
        )
    tokenizer_sha = _validate_sha256(
        tokenizer_identity_sha256, "tokenizer_identity_sha256"
    )
    horizon_value = _positive_int(horizon, "horizon")
    if horizon_value > controls.max_horizon:
        raise HFResidualAttentionOracleLimitError(
            f"horizon exceeds max_horizon={controls.max_horizon}."
        )
    devices = _model_devices(model)
    if not devices or len(devices) > controls.max_model_devices:
        raise HFResidualAttentionOracleLimitError(
            "Parent device count exceeds max_model_devices or is empty."
        )
    parent_input_vocab, parent_output_vocab = _model_vocabulary_sizes(model)
    if parent_input_vocab != parent_output_vocab:
        raise HFResidualAttentionOracleShapeError(
            "Dense parent input and output vocabulary widths differ."
        )
    chart_rows: Sequence[int] | Tensor
    if parent_token_rows is None:
        chart_rows = torch.arange(parent_output_vocab, dtype=torch.int64)
    else:
        chart_rows = parent_token_rows
    vocabulary_chart = build_hf_oracle_vocabulary_chart(
        chart_rows, parent_vocab_size=parent_output_vocab
    )
    coordinate_contract = _build_attention_coordinate_contract(
        model, operator.index(layer_index)
    )
    canonical_input = torch.as_tensor(input_ids, device="cpu")
    parent_input = _canonical_to_parent_ids(
        canonical_input, vocabulary_chart, name="input_ids"
    )
    parent_forced = (
        None
        if forced_token_ids is None
        else _canonical_to_parent_ids(
            forced_token_ids, vocabulary_chart, name="forced_token_ids"
        ).reshape(-1)
    )
    identity_before = fingerprint_hf_oracle_model(
        model, chunk_bytes=controls.fingerprint_chunk_bytes
    )
    trace_limits = HFCachedAttentionTraceLimits(
        max_horizon=controls.max_horizon,
        max_prompt_tokens=controls.max_prompt_tokens,
        max_model_devices=controls.max_model_devices,
        max_capture_bytes=controls.max_trace_bytes,
        logit_rtol=controls.logit_rtol,
        logit_atol=controls.logit_atol,
        context_rtol=controls.residual_rtol,
        context_atol=controls.residual_atol,
    )
    try:
        raw_cached = collect_hf_cached_attention_trace(
            model,
            parent_input,
            layer_index=layer_index,
            group_indices=group_indices,
            horizon=horizon_value,
            sequence_id=sequence_id,
            lane=lane,
            forced_token_ids=parent_forced,
            limits=trace_limits,
        )
    except HFCachedAttentionTraceShapeError as error:
        raise HFResidualAttentionOracleShapeError(str(error)) from error
    full_deltas, delta_bytes = _recapture_parent_full_deltas(
        model, raw_cached, controls
    )
    cached = _canonicalize_parent_cached_trace(raw_cached, vocabulary_chart)
    identity_after = fingerprint_hf_oracle_model(
        model, chunk_bytes=controls.fingerprint_chunk_bytes
    )
    if identity_before != identity_after:
        raise HFResidualAttentionOracleStateError(
            "Parent model config or state changed during oracle capture."
        )
    total_bytes = (
        cached.capture_bytes
        + delta_bytes
        + _tensor_bytes(vocabulary_chart.child_to_parent_rows)
    )
    if total_bytes > controls.max_trace_bytes:
        raise HFResidualAttentionOracleLimitError(
            "Combined parent oracle trace exceeded max_trace_bytes."
        )
    provisional = HFParentResidualAttentionOracleTrace(
        cached_trace=cached,
        full_attention_deltas=full_deltas,
        parent_model_identity=identity_before,
        vocabulary_chart=vocabulary_chart,
        attention_coordinate_contract=coordinate_contract,
        tokenizer_identity_sha256=tokenizer_sha,
        parent_prompt_token_ids=raw_cached.prompt_token_ids,
        parent_selected_token_ids=raw_cached.selected_token_ids,
        trace_byte_count=total_bytes,
        identity_sha256="0" * 64,
    )
    result = replace(
        provisional,
        identity_sha256=_json_sha256(_parent_trace_identity_body(provisional)),
    )
    _validate_parent_trace(result)
    return result


def _normalize_intervention(value: Any) -> ResidualAttentionIntervention:
    normalized = str(value).strip().lower()
    if normalized not in _INTERVENTIONS:
        raise HFResidualAttentionOracleShapeError(
            f"intervention must be one of {_INTERVENTIONS}."
        )
    return normalized  # type: ignore[return-value]


def _normalize_feedback(value: Any) -> ResidualAttentionFeedback:
    normalized = str(value).strip().lower()
    if normalized not in {"forced-cache-recurrent", "autoregressive"}:
        raise HFResidualAttentionOracleShapeError(
            "feedback must be forced-cache-recurrent or autoregressive."
        )
    return normalized  # type: ignore[return-value]


def _uses_key(intervention: ResidualAttentionIntervention) -> bool:
    return intervention in {"k_cache", "kv_cache", "kv_cache_and_attention_delta"}


def _uses_value(intervention: ResidualAttentionIntervention) -> bool:
    return intervention in {"v_cache", "kv_cache", "kv_cache_and_attention_delta"}


def _uses_delta(intervention: ResidualAttentionIntervention) -> bool:
    return intervention in {"attention_delta", "kv_cache_and_attention_delta"}


class _CacheInterventionController:
    def __init__(
        self,
        parent_trace: HFParentResidualAttentionOracleTrace,
        intervention: ResidualAttentionIntervention,
        limits: HFResidualAttentionOracleLimits,
    ) -> None:
        self.parent_trace = parent_trace
        self.intervention = intervention
        self.limits = limits
        self.step_index: int | None = None
        self.target_update_count = 0
        self.key_max_abs_error = 0.0
        self.value_max_abs_error = 0.0

    def begin(self, step_index: int) -> None:
        if self.step_index is not None:
            raise HFResidualAttentionOracleStateError(
                "Cache intervention step is already active."
            )
        self.step_index = step_index
        self.target_update_count = 0
        self.key_max_abs_error = 0.0
        self.value_max_abs_error = 0.0

    def apply(
        self,
        cache: Any,
        keys: Tensor,
        values: Tensor,
        layer_index: int,
    ) -> tuple[Tensor, Tensor]:
        target_layer = self.parent_trace.cached_trace.layer_index
        if layer_index != target_layer:
            return keys, values
        if self.step_index is None:
            raise HFResidualAttentionOracleStateError(
                "Target cache updated outside an active oracle step."
            )
        self.target_update_count += 1
        if self.target_update_count != 1:
            raise HFResidualAttentionOracleStateError(
                "Target cache updated more than once during one model forward."
            )
        reference_step = self.parent_trace.cached_trace.steps[self.step_index]
        if keys.shape != reference_step.layer_key_cache.shape or (
            values.shape != reference_step.layer_value_cache.shape
        ):
            raise HFResidualAttentionOracleShapeError(
                "Candidate cache and parent cache have different Qwen/GQA shapes."
            )
        layers = getattr(cache, "layers", None)
        if not isinstance(layers, Sequence) or target_layer >= len(layers):
            raise HFResidualAttentionOracleStateError(
                "DynamicCache lost its target cache layer."
            )
        cache_layer = layers[target_layer]
        stored_keys = getattr(cache_layer, "keys", None)
        stored_values = getattr(cache_layer, "values", None)
        if stored_keys is not keys or stored_values is not values:
            raise HFResidualAttentionOracleStateError(
                "Official cache update did not return its stored target K/V tensors."
            )
        group_indices = torch.tensor(
            self.parent_trace.cached_trace.group_indices,
            dtype=torch.long,
            device=keys.device,
        )
        result_keys = keys
        result_values = values
        if _uses_key(self.intervention):
            parent_keys = reference_step.layer_key_cache.to(
                device=keys.device, dtype=keys.dtype
            )
            selected_parent_keys = parent_keys.index_select(1, group_indices)
            result_keys = keys.clone()
            result_keys.index_copy_(1, group_indices, selected_parent_keys)
            cache_layer.keys = result_keys
            self.key_max_abs_error = float(
                (
                    result_keys.index_select(1, group_indices).to(torch.float32)
                    - selected_parent_keys.to(torch.float32)
                )
                .abs()
                .max()
                .item()
            )
        if _uses_value(self.intervention):
            parent_values = reference_step.layer_value_cache.to(
                device=values.device, dtype=values.dtype
            )
            selected_parent_values = parent_values.index_select(1, group_indices)
            result_values = values.clone()
            result_values.index_copy_(1, group_indices, selected_parent_values)
            cache_layer.values = result_values
            self.value_max_abs_error = float(
                (
                    result_values.index_select(1, group_indices).to(torch.float32)
                    - selected_parent_values.to(torch.float32)
                )
                .abs()
                .max()
                .item()
            )
        return result_keys, result_values

    def finish(self) -> tuple[float, float]:
        if self.step_index is None or self.target_update_count != 1:
            raise HFResidualAttentionOracleStateError(
                "Target cache did not update exactly once during oracle step."
            )
        if _uses_key(self.intervention) and self.key_max_abs_error > (
            self.limits.cache_atol
        ):
            raise HFResidualAttentionOracleStateError(
                "Injected parent K cache was not retained exactly."
            )
        if _uses_value(self.intervention) and self.value_max_abs_error > (
            self.limits.cache_atol
        ):
            raise HFResidualAttentionOracleStateError(
                "Injected parent V cache was not retained exactly."
            )
        result = self.key_max_abs_error, self.value_max_abs_error
        self.step_index = None
        return result

    def abort(self) -> None:
        self.step_index = None


def _new_intervention_cache(
    model: nn.Module, controller: _CacheInterventionController
) -> Any:
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError as error:  # pragma: no cover - optional dependency
        raise HFResidualAttentionOracleStateError(
            "Hugging Face DynamicCache is required for Qwen intervention."
        ) from error

    class _OracleDynamicCache(DynamicCache):
        def update(
            self,
            key_states: Tensor,
            value_states: Tensor,
            layer_idx: int,
            *args: Any,
            **kwargs: Any,
        ) -> tuple[Tensor, Tensor]:
            keys, values = super().update(
                key_states, value_states, layer_idx, *args, **kwargs
            )
            return controller.apply(self, keys, values, int(layer_idx))

    try:
        return _OracleDynamicCache(config=getattr(model, "config", None))
    except TypeError:
        return _OracleDynamicCache()


class _ResidualDeltaInterventionHook:
    def __init__(
        self,
        layer: nn.Module,
        parent_trace: HFParentResidualAttentionOracleTrace,
        intervention: ResidualAttentionIntervention,
        limits: HFResidualAttentionOracleLimits,
    ) -> None:
        attention = getattr(layer, "self_attn", None)
        output_projection = getattr(attention, "o_proj", None)
        post_norm = getattr(layer, "post_attention_layernorm", None)
        if not isinstance(output_projection, nn.Module) or not isinstance(
            post_norm, nn.Module
        ):
            raise HFResidualAttentionOracleShapeError(
                "Selected Qwen layer lacks o_proj or post_attention_layernorm."
            )
        self.parent_trace = parent_trace
        self.intervention = intervention
        self.limits = limits
        self.active_step: int | None = None
        self.counts: dict[str, int] = {}
        self.pre_residual: Tensor | None = None
        self.original_delta: Tensor | None = None
        self.effective_delta: Tensor | None = None
        self.post_residual: Tensor | None = None

        def layer_pre(
            _module: nn.Module,
            args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> None:
            if self.active_step is None:
                return
            self._count("pre_residual")
            self.pre_residual = _module_input(args, kwargs, "decoder layer").detach()

        def output_hook(
            _module: nn.Module,
            _args: tuple[Any, ...],
            _kwargs: Mapping[str, Any],
            output: Any,
        ) -> Tensor | None:
            if self.active_step is None:
                return None
            self._count("attention_delta")
            if not isinstance(output, Tensor):
                raise HFResidualAttentionOracleStateError(
                    "Qwen o_proj did not return a tensor."
                )
            self.original_delta = output.detach()
            if _uses_delta(self.intervention):
                reference = self.parent_trace.full_attention_deltas[
                    self.active_step
                ].to(device=output.device, dtype=output.dtype)
                if reference.shape != output.shape:
                    raise HFResidualAttentionOracleShapeError(
                        "Candidate residual delta shape differs from parent trace."
                    )
                self.effective_delta = reference
                return reference
            self.effective_delta = output.detach()
            return None

        def post_norm_pre(
            _module: nn.Module,
            args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> None:
            if self.active_step is None:
                return
            self._count("post_residual")
            self.post_residual = _module_input(
                args, kwargs, "post-attention norm"
            ).detach()

        self.handles = [
            layer.register_forward_pre_hook(layer_pre, with_kwargs=True),
            output_projection.register_forward_hook(output_hook, with_kwargs=True),
            post_norm.register_forward_pre_hook(post_norm_pre, with_kwargs=True),
        ]

    def _count(self, name: str) -> None:
        count = self.counts.get(name, 0) + 1
        self.counts[name] = count
        if count != 1:
            raise HFResidualAttentionOracleStateError(
                f"Residual intervention hook {name!r} fired {count} times."
            )

    def begin(self, step_index: int) -> None:
        if self.active_step is not None:
            raise HFResidualAttentionOracleStateError(
                "Residual intervention step is already active."
            )
        self.active_step = step_index
        self.counts = {}
        self.pre_residual = None
        self.original_delta = None
        self.effective_delta = None
        self.post_residual = None

    def finish(self) -> tuple[Tensor, float]:
        required = {"pre_residual", "attention_delta", "post_residual"}
        if set(self.counts) != required or any(
            self.counts[name] != 1 for name in required
        ):
            raise HFResidualAttentionOracleStateError(
                "Residual intervention hooks did not each fire exactly once."
            )
        if (
            self.pre_residual is None
            or self.original_delta is None
            or self.effective_delta is None
            or self.post_residual is None
        ):
            raise HFResidualAttentionOracleStateError(
                "Residual intervention hook state is incomplete."
            )
        if self.pre_residual.shape != self.effective_delta.shape or (
            self.post_residual.shape != self.pre_residual.shape
        ):
            raise HFResidualAttentionOracleShapeError(
                "Candidate residual tensors have incompatible shapes."
            )
        expected = self.pre_residual.to(torch.float32) + self.effective_delta.to(
            torch.float32
        )
        error = float(
            (self.post_residual.to(torch.float32) - expected).abs().max().item()
        )
        if not bool(
            torch.allclose(
                self.post_residual.to(torch.float32),
                expected,
                rtol=self.limits.residual_rtol,
                atol=self.limits.residual_atol,
            )
        ):
            raise HFResidualAttentionOracleStateError(
                "Candidate did not add the effective attention delta to residual."
            )
        original = _readonly_cpu(
            self.original_delta[:, -1, :], "candidate original attention delta"
        )
        self.active_step = None
        return original, error

    def abort(self) -> None:
        self.active_step = None

    def remove(self) -> None:
        self.active_step = None
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()


@dataclass(frozen=True, slots=True)
class HFResidualAttentionOracleReplayStep:
    """One valid-prefix candidate action under a causal intervention."""

    step_index: int
    prefix_length: int
    parent_greedy_token_id: int
    parent_driver_token_id: int
    candidate_greedy_token_id: int
    consumed_token_id: int | None
    parent_logits: Tensor
    candidate_logits: Tensor
    candidate_original_attention_delta: Tensor
    injected_key_max_abs_error: float
    injected_value_max_abs_error: float
    residual_add_max_abs_error: float

    def as_dict(self) -> dict[str, Any]:
        parent = self.parent_logits.to(torch.float32)
        candidate = self.candidate_logits.to(torch.float32)
        parent_centered = parent - parent.mean(dim=-1, keepdim=True)
        candidate_centered = candidate - candidate.mean(dim=-1, keepdim=True)
        centered_rmse = float(
            torch.sqrt(torch.mean((parent_centered - candidate_centered) ** 2)).item()
        )
        parent_log_probs = torch.log_softmax(parent, dim=-1)
        candidate_log_probs = torch.log_softmax(candidate, dim=-1)
        parent_probs = torch.exp(parent_log_probs)
        kl = float(
            torch.sum(parent_probs * (parent_log_probs - candidate_log_probs), dim=-1)
            .mean()
            .item()
        )
        return {
            "step_index": self.step_index,
            "prefix_length": self.prefix_length,
            "parent_greedy_token_id": self.parent_greedy_token_id,
            "parent_driver_token_id": self.parent_driver_token_id,
            "candidate_greedy_token_id": self.candidate_greedy_token_id,
            "consumed_token_id": self.consumed_token_id,
            "top1_agreement": (
                self.parent_greedy_token_id == self.candidate_greedy_token_id
            ),
            "centered_logit_rmse": centered_rmse,
            "parent_to_candidate_kl": kl,
            "candidate_logits_sha256": _tensor_sha256(self.candidate_logits),
            "candidate_original_attention_delta_sha256": _tensor_sha256(
                self.candidate_original_attention_delta
            ),
            "injected_key_max_abs_error": self.injected_key_max_abs_error,
            "injected_value_max_abs_error": self.injected_value_max_abs_error,
            "residual_add_max_abs_error": self.residual_add_max_abs_error,
        }


@dataclass(frozen=True, slots=True)
class HFResidualAttentionOracleReplay:
    """Candidate replay result for exactly one intervention."""

    intervention: ResidualAttentionIntervention
    feedback: ResidualAttentionFeedback
    parent_trace_identity_sha256: str
    candidate_model_identity: HFOracleModelIdentity
    tokenizer_identity_sha256: str
    steps: tuple[HFResidualAttentionOracleReplayStep, ...]
    requested_horizon: int
    first_action_divergence_step: int | None
    replay_byte_count: int
    candidate_state_unchanged: bool

    @property
    def completed_horizon(self) -> int:
        return len(self.steps)

    @property
    def full_horizon_valid(self) -> bool:
        return self.completed_horizon == self.requested_horizon

    def summary_metrics(self) -> Mapping[str, float]:
        if not self.steps:
            raise HFResidualAttentionOracleStateError(
                "A replay must contain at least one valid-prefix step."
            )
        rows = [step.as_dict() for step in self.steps]
        return MappingProxyType(
            {
                "top1_agreement": sum(bool(row["top1_agreement"]) for row in rows)
                / len(rows),
                "centered_logit_rmse": sum(
                    float(row["centered_logit_rmse"]) for row in rows
                )
                / len(rows),
                "parent_to_candidate_kl": sum(
                    float(row["parent_to_candidate_kl"]) for row in rows
                )
                / len(rows),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema": ORACLE_REPLAY_SCHEMA,
            "intervention": self.intervention,
            "feedback": self.feedback,
            "parent_trace_identity_sha256": self.parent_trace_identity_sha256,
            "candidate_model_identity": self.candidate_model_identity.as_dict(),
            "tokenizer_identity_sha256": self.tokenizer_identity_sha256,
            "requested_horizon": self.requested_horizon,
            "completed_horizon": self.completed_horizon,
            "full_horizon_valid": self.full_horizon_valid,
            "first_action_divergence_step": self.first_action_divergence_step,
            "feedback_intervention_stopped_at_divergence": (
                self.feedback == "autoregressive"
                and self.first_action_divergence_step is not None
                and self.completed_horizon < self.requested_horizon
            ),
            "replay_byte_count": self.replay_byte_count,
            "candidate_state_unchanged": self.candidate_state_unchanged,
            "summary_metrics": dict(self.summary_metrics()),
            "steps": [step.as_dict() for step in self.steps],
            "raw_substitution_requires_shared_coordinates": True,
            "gradient_free": True,
            "training_used": False,
        }
        body["replay_sha256"] = _json_sha256(body)
        return body


def _validate_candidate_topology(
    model: nn.Module, parent_trace: HFParentResidualAttentionOracleTrace
) -> nn.Module:
    cached = parent_trace.cached_trace
    model_class = f"{type(model).__module__}.{type(model).__qualname__}"
    if model_class != cached.model_class:
        raise HFResidualAttentionOracleShapeError(
            "Candidate model class differs from the certified parent trace path."
        )
    config = getattr(model, "config", None)
    backend = str(getattr(config, "_attn_implementation", None) or "eager").strip()
    if backend != cached.attention_backend:
        raise HFResidualAttentionOracleStateError(
            "Candidate attention backend differs from parent trace."
        )
    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = None
    if transformers_version != cached.transformers_version:
        raise HFResidualAttentionOracleStateError(
            "Transformers runtime version differs from parent trace."
        )
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError as error:  # pragma: no cover - optional dependency
        raise HFResidualAttentionOracleStateError(
            "Hugging Face DynamicCache is unavailable."
        ) from error
    cache_class = f"{DynamicCache.__module__}.{DynamicCache.__qualname__}"
    if cache_class != cached.cache_class:
        raise HFResidualAttentionOracleStateError(
            "Candidate DynamicCache path differs from parent trace cache path."
        )
    chart = parent_trace.vocabulary_chart
    candidate_input_vocab, candidate_output_vocab = _model_vocabulary_sizes(model)
    if (
        candidate_input_vocab != chart.child_vocab_size
        or candidate_output_vocab != chart.child_vocab_size
    ):
        raise HFResidualAttentionOracleShapeError(
            "Candidate vocabulary width differs from the canonical child chart."
        )
    mapping_buffer = getattr(model, "parent_token_rows", None)
    if mapping_buffer is None:
        if not chart.is_identity:
            raise HFResidualAttentionOracleStateError(
                "Mapped-vocabulary candidate lacks parent_token_rows provenance."
            )
    elif (
        not isinstance(mapping_buffer, Tensor)
        or mapping_buffer.ndim != 1
        or int(mapping_buffer.numel()) != chart.child_vocab_size
        or not bool(
            torch.equal(
                mapping_buffer.detach().to(device="cpu", dtype=torch.int64),
                chart.child_to_parent_rows,
            )
        )
    ):
        raise HFResidualAttentionOracleStateError(
            "Candidate parent_token_rows differs from parent trace vocabulary chart."
        )
    layers = _base_layers(model)
    if cached.layer_index >= len(layers):
        raise HFResidualAttentionOracleShapeError(
            "Candidate does not contain the traced parent layer index."
        )
    layer = layers[cached.layer_index]
    attention = getattr(layer, "self_attn", None)
    config = getattr(model, "config", getattr(attention, "config", None))
    topology = (
        int(getattr(config, "num_attention_heads", 0)),
        int(getattr(config, "num_key_value_heads", 0)),
        int(getattr(config, "head_dim", getattr(attention, "head_dim", 0))),
    )
    expected = (cached.query_heads, cached.key_value_heads, cached.head_dim)
    if topology != expected:
        raise HFResidualAttentionOracleShapeError(
            f"Candidate Qwen/GQA topology {topology} differs from parent {expected}."
        )
    if max(cached.group_indices) >= topology[1]:
        raise HFResidualAttentionOracleShapeError(
            "Parent group indices are unavailable in candidate."
        )
    hidden_size = int(getattr(config, "hidden_size", 0))
    parent_hidden = int(cached.steps[0].attention_delta.shape[-1])
    if hidden_size != parent_hidden:
        raise HFResidualAttentionOracleShapeError(
            "Candidate hidden width differs from parent residual coordinates."
        )
    candidate_contract = _build_attention_coordinate_contract(model, cached.layer_index)
    mismatch = _coordinate_contract_mismatch(
        parent_trace.attention_coordinate_contract, candidate_contract
    )
    if mismatch is not None:
        raise HFResidualAttentionOracleShapeError(
            "Candidate attention-coordinate contract differs from parent at "
            f"{mismatch}."
        )
    return layer


def _replay_core(
    model: nn.Module,
    parent_trace: HFParentResidualAttentionOracleTrace,
    *,
    candidate_identity: HFOracleModelIdentity,
    tokenizer_identity_sha256: str,
    intervention: ResidualAttentionIntervention,
    feedback: ResidualAttentionFeedback,
    limits: HFResidualAttentionOracleLimits,
) -> HFResidualAttentionOracleReplay:
    cached = parent_trace.cached_trace
    if feedback == "autoregressive" and cached.lane != "greedy":
        raise HFResidualAttentionOracleShapeError(
            "autoregressive replay requires a greedy parent trace."
        )
    layer = _validate_candidate_topology(model, parent_trace)
    devices = _model_devices(model)
    if not devices or len(devices) > limits.max_model_devices:
        raise HFResidualAttentionOracleLimitError(
            "Candidate device count exceeds max_model_devices or is empty."
        )
    if bool(getattr(model, "is_gradient_checkpointing", False)) or bool(
        getattr(getattr(model, "config", None), "gradient_checkpointing", False)
    ):
        raise HFResidualAttentionOracleStateError(
            "Gradient checkpointing must be disabled for oracle replay."
        )
    device = _embedding_device(model)
    prompt = torch.tensor([cached.prompt_token_ids], dtype=torch.long, device=device)
    prefix = prompt.clone()
    next_input = prompt
    controller = _CacheInterventionController(parent_trace, intervention, limits)
    past = _new_intervention_cache(model, controller)
    hooks_before = _hook_inventory(model)
    residual_hook = _ResidualDeltaInterventionHook(
        layer, parent_trace, intervention, limits
    )
    modes, config, config_values = _runtime_restore_context(model)
    results: list[HFResidualAttentionOracleReplayStep] = []
    replay_bytes = 0
    divergence: int | None = None
    try:
        _set_inference_runtime(model, config)
        with torch.inference_mode():
            for step_index, parent_step in enumerate(cached.steps):
                mask = torch.ones(
                    (1, int(prefix.shape[1])), dtype=torch.long, device=device
                )
                controller.begin(step_index)
                residual_hook.begin(step_index)
                try:
                    output = _model_forward(
                        model, next_input, mask, past_key_values=past
                    )
                    key_error, value_error = controller.finish()
                    original_delta, residual_error = residual_hook.finish()
                except Exception:
                    controller.abort()
                    residual_hook.abort()
                    raise
                logits = _readonly_cpu(
                    _output_logits(output)[:, -1, :], "candidate oracle logits"
                )
                if logits.shape != parent_step.action_logits.shape:
                    raise HFResidualAttentionOracleShapeError(
                        "Candidate and canonical parent logits have different shapes."
                    )
                candidate_greedy = int(torch.argmax(logits, dim=-1).item())
                parent_greedy = int(
                    torch.argmax(parent_step.action_logits, dim=-1).item()
                )
                parent_driver = int(parent_step.selected_token_id)
                if candidate_greedy != parent_greedy and divergence is None:
                    divergence = step_index
                if feedback == "forced-cache-recurrent":
                    consumed = (
                        parent_driver if step_index + 1 < cached.horizon else None
                    )
                else:
                    consumed = (
                        candidate_greedy
                        if divergence is None and step_index + 1 < cached.horizon
                        else None
                    )
                row = HFResidualAttentionOracleReplayStep(
                    step_index=step_index,
                    prefix_length=int(prefix.shape[1]),
                    parent_greedy_token_id=parent_greedy,
                    parent_driver_token_id=parent_driver,
                    candidate_greedy_token_id=candidate_greedy,
                    consumed_token_id=consumed,
                    parent_logits=parent_step.action_logits.clone(),
                    candidate_logits=logits,
                    candidate_original_attention_delta=original_delta,
                    injected_key_max_abs_error=key_error,
                    injected_value_max_abs_error=value_error,
                    residual_add_max_abs_error=residual_error,
                )
                results.append(row)
                replay_bytes += sum(
                    _tensor_bytes(value)
                    for value in (
                        row.parent_logits,
                        row.candidate_logits,
                        row.candidate_original_attention_delta,
                    )
                )
                if replay_bytes > limits.max_replay_bytes:
                    raise HFResidualAttentionOracleLimitError(
                        "Candidate replay exceeded max_replay_bytes."
                    )
                if divergence is not None and feedback == "autoregressive":
                    break
                past = _output_field(output, "past_key_values")
                if step_index + 1 < cached.horizon:
                    assert consumed is not None
                    next_input = torch.tensor(
                        [[consumed]], dtype=torch.long, device=device
                    )
                    prefix = torch.cat((prefix, next_input), dim=1)
    finally:
        controller.abort()
        residual_hook.remove()
        _restore_runtime(modes, config, config_values)
    if _hook_inventory(model) != hooks_before:
        raise HFResidualAttentionOracleStateError(
            "Candidate replay did not restore hooks exactly."
        )
    if not results:
        raise HFResidualAttentionOracleStateError(
            "Candidate replay produced no valid-prefix evidence."
        )
    return HFResidualAttentionOracleReplay(
        intervention=intervention,
        feedback=feedback,
        parent_trace_identity_sha256=parent_trace.identity_sha256,
        candidate_model_identity=candidate_identity,
        tokenizer_identity_sha256=tokenizer_identity_sha256,
        steps=tuple(results),
        requested_horizon=cached.horizon,
        first_action_divergence_step=divergence,
        replay_byte_count=replay_bytes,
        candidate_state_unchanged=True,
    )


def replay_residual_attention_oracle_intervention(
    model: nn.Module,
    parent_trace: HFParentResidualAttentionOracleTrace,
    *,
    tokenizer_identity_sha256: str,
    intervention: ResidualAttentionIntervention,
    feedback: ResidualAttentionFeedback = "forced-cache-recurrent",
    limits: HFResidualAttentionOracleLimits | None = None,
) -> HFResidualAttentionOracleReplay:
    """Replay one intervention and prove candidate state restoration exactly."""

    controls = limits or HFResidualAttentionOracleLimits()
    if not isinstance(controls, HFResidualAttentionOracleLimits):
        raise HFResidualAttentionOracleShapeError(
            "limits must be HFResidualAttentionOracleLimits."
        )
    _validate_parent_trace(parent_trace)
    tokenizer_sha = _validate_sha256(
        tokenizer_identity_sha256, "tokenizer_identity_sha256"
    )
    if tokenizer_sha != parent_trace.tokenizer_identity_sha256:
        raise HFResidualAttentionOracleStateError(
            "Candidate tokenizer identity differs from parent trace."
        )
    normalized_intervention = _normalize_intervention(intervention)
    normalized_feedback = _normalize_feedback(feedback)
    identity_before = fingerprint_hf_oracle_model(
        model, chunk_bytes=controls.fingerprint_chunk_bytes
    )
    result = _replay_core(
        model,
        parent_trace,
        candidate_identity=identity_before,
        tokenizer_identity_sha256=tokenizer_sha,
        intervention=normalized_intervention,
        feedback=normalized_feedback,
        limits=controls,
    )
    identity_after = fingerprint_hf_oracle_model(
        model, chunk_bytes=controls.fingerprint_chunk_bytes
    )
    if identity_before != identity_after:
        raise HFResidualAttentionOracleStateError(
            "Candidate model config or state changed during oracle replay."
        )
    return result


@dataclass(frozen=True, slots=True)
class HFResidualAttentionOracleComparison:
    """Damage and recovery of one intervention relative to no intervention."""

    intervention: ResidualAttentionIntervention
    top1_agreement: float
    centered_logit_rmse: float
    parent_to_candidate_kl: float
    top1_agreement_gain: float
    centered_logit_rmse_recovery: float
    parent_to_candidate_kl_recovery: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "intervention": self.intervention,
            "independent_causal_evidence": self.intervention
            not in {"none", "kv_cache_and_attention_delta"},
            "top1_agreement": self.top1_agreement,
            "centered_logit_rmse": self.centered_logit_rmse,
            "parent_to_candidate_kl": self.parent_to_candidate_kl,
            "top1_agreement_gain": self.top1_agreement_gain,
            "centered_logit_rmse_recovery": self.centered_logit_rmse_recovery,
            "parent_to_candidate_kl_recovery": (self.parent_to_candidate_kl_recovery),
        }


@dataclass(frozen=True, slots=True)
class HFResidualAttentionOracleSuite:
    """Baseline plus the five requested parent-state causal interventions."""

    parent_trace_identity_sha256: str
    candidate_model_identity: HFOracleModelIdentity
    tokenizer_identity_sha256: str
    feedback: ResidualAttentionFeedback
    replays: tuple[HFResidualAttentionOracleReplay, ...]
    comparisons: tuple[HFResidualAttentionOracleComparison, ...]
    candidate_state_unchanged: bool

    def replay(self, intervention: str) -> HFResidualAttentionOracleReplay:
        normalized = _normalize_intervention(intervention)
        for replay in self.replays:
            if replay.intervention == normalized:
                return replay
        raise KeyError(normalized)

    def as_dict(self) -> dict[str, Any]:
        full_horizon = all(replay.full_horizon_valid for replay in self.replays)
        body = {
            "schema": ORACLE_SUITE_SCHEMA,
            "parent_trace_identity_sha256": self.parent_trace_identity_sha256,
            "candidate_model_identity": self.candidate_model_identity.as_dict(),
            "tokenizer_identity_sha256": self.tokenizer_identity_sha256,
            "feedback": self.feedback,
            "causal_recovery_comparison_valid": (
                self.feedback == "forced-cache-recurrent" and full_horizon
            ),
            "all_replays_full_horizon": full_horizon,
            "candidate_state_unchanged": self.candidate_state_unchanged,
            "comparisons": [value.as_dict() for value in self.comparisons],
            "replays": [value.as_dict() for value in self.replays],
            "interpretation": {
                "k_cache": "post-RoPE selected-group addressing state",
                "v_cache": "selected-group value memory state",
                "attention_delta": "full selected-layer residual write",
                "kv_cache": "selected-group address and value memory",
                "kv_cache_and_attention_delta": (
                    "redundancy/consistency control only: the full residual-write "
                    "substitution causally dominates target-layer K/V effects"
                ),
            },
            "independent_causal_interventions": [
                "k_cache",
                "v_cache",
                "attention_delta",
                "kv_cache",
            ],
            "redundancy_controls": ["kv_cache_and_attention_delta"],
            "raw_cross_architecture_intervention_supported": False,
            "gradient_free": True,
            "training_used": False,
        }
        body["suite_sha256"] = _json_sha256(body)
        return body


def _error_recovery(baseline: float, intervention: float) -> float:
    if baseline <= 1e-12:
        return 0.0 if intervention <= 1e-12 else -intervention / 1e-12
    return (baseline - intervention) / baseline


def run_residual_attention_oracle_suite(
    model: nn.Module,
    parent_trace: HFParentResidualAttentionOracleTrace,
    *,
    tokenizer_identity_sha256: str,
    feedback: ResidualAttentionFeedback = "forced-cache-recurrent",
    interventions: Sequence[ResidualAttentionIntervention] = _INTERVENTIONS,
    limits: HFResidualAttentionOracleLimits | None = None,
) -> HFResidualAttentionOracleSuite:
    """Run a fresh cache for every control while hashing candidate only twice."""

    controls = limits or HFResidualAttentionOracleLimits()
    if not isinstance(controls, HFResidualAttentionOracleLimits):
        raise HFResidualAttentionOracleShapeError(
            "limits must be HFResidualAttentionOracleLimits."
        )
    _validate_parent_trace(parent_trace)
    tokenizer_sha = _validate_sha256(
        tokenizer_identity_sha256, "tokenizer_identity_sha256"
    )
    if tokenizer_sha != parent_trace.tokenizer_identity_sha256:
        raise HFResidualAttentionOracleStateError(
            "Candidate tokenizer identity differs from parent trace."
        )
    normalized_feedback = _normalize_feedback(feedback)
    normalized_interventions = tuple(
        _normalize_intervention(value) for value in interventions
    )
    if (
        not normalized_interventions
        or normalized_interventions[0] != "none"
        or len(set(normalized_interventions)) != len(normalized_interventions)
    ):
        raise HFResidualAttentionOracleShapeError(
            "interventions must be unique and begin with the none baseline."
        )
    identity_before = fingerprint_hf_oracle_model(
        model, chunk_bytes=controls.fingerprint_chunk_bytes
    )
    replays = tuple(
        _replay_core(
            model,
            parent_trace,
            candidate_identity=identity_before,
            tokenizer_identity_sha256=tokenizer_sha,
            intervention=intervention,
            feedback=normalized_feedback,
            limits=controls,
        )
        for intervention in normalized_interventions
    )
    identity_after = fingerprint_hf_oracle_model(
        model, chunk_bytes=controls.fingerprint_chunk_bytes
    )
    if identity_before != identity_after:
        raise HFResidualAttentionOracleStateError(
            "Candidate model config or state changed during oracle suite."
        )
    baseline = replays[0].summary_metrics()
    comparisons: list[HFResidualAttentionOracleComparison] = []
    for replay in replays:
        metrics = replay.summary_metrics()
        comparisons.append(
            HFResidualAttentionOracleComparison(
                intervention=replay.intervention,
                top1_agreement=metrics["top1_agreement"],
                centered_logit_rmse=metrics["centered_logit_rmse"],
                parent_to_candidate_kl=metrics["parent_to_candidate_kl"],
                top1_agreement_gain=(
                    metrics["top1_agreement"] - baseline["top1_agreement"]
                ),
                centered_logit_rmse_recovery=_error_recovery(
                    baseline["centered_logit_rmse"],
                    metrics["centered_logit_rmse"],
                ),
                parent_to_candidate_kl_recovery=_error_recovery(
                    baseline["parent_to_candidate_kl"],
                    metrics["parent_to_candidate_kl"],
                ),
            )
        )
    return HFResidualAttentionOracleSuite(
        parent_trace_identity_sha256=parent_trace.identity_sha256,
        candidate_model_identity=identity_before,
        tokenizer_identity_sha256=tokenizer_sha,
        feedback=normalized_feedback,
        replays=replays,
        comparisons=tuple(comparisons),
        candidate_state_unchanged=True,
    )


__all__ = [
    "HFOracleModelIdentity",
    "HFOracleVocabularyChart",
    "HFParentResidualAttentionOracleTrace",
    "HFResidualAttentionOracleComparison",
    "HFResidualAttentionOracleError",
    "HFResidualAttentionOracleLimitError",
    "HFResidualAttentionOracleLimits",
    "HFResidualAttentionOracleReplay",
    "HFResidualAttentionOracleReplayStep",
    "HFResidualAttentionOracleShapeError",
    "HFResidualAttentionOracleStateError",
    "HFResidualAttentionOracleSuite",
    "ORACLE_REPLAY_SCHEMA",
    "ORACLE_SUITE_SCHEMA",
    "ORACLE_TRACE_SCHEMA",
    "ResidualAttentionFeedback",
    "ResidualAttentionIntervention",
    "build_hf_oracle_vocabulary_chart",
    "capture_parent_residual_attention_oracle_trace",
    "fingerprint_hf_oracle_model",
    "fingerprint_hf_oracle_tokenizer",
    "replay_residual_attention_oracle_intervention",
    "run_residual_attention_oracle_suite",
]
