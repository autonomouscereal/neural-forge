"""Fail-closed Hugging Face causal-decoder architecture layouts.

The transport proof supports three explicit operator topologies:

* ``GPT2LMHeadModel``-style GPT-2 decoders with learned positions, fused QKV,
  dense GELU-family MLPs, and LayerNorm.
* ``LlamaForCausalLM``-style Llama decoders with RoPE, split Q/K/V (including
  grouped-query attention), gated MLPs, and RMSNorm.
* ``Qwen3ForCausalLM`` dense decoders with RoPE, split grouped-query attention,
  per-head Q/K RMSNorm, gated SwiGLU MLPs, and residual RMSNorm.

This module deliberately does not discover operators by suffix similarity. A
recognized ``model_type`` must satisfy the complete expected path, type, shape,
activation, residual, and tying contract or construction fails.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal


ArchitectureFamily = Literal["gpt2", "llama", "qwen3"]
PositionKind = Literal["learned_absolute", "rope"]
NormKind = Literal["layer_norm", "rms_norm"]
NormPlacement = Literal[
    "pre_attention",
    "query_head",
    "key_head",
    "pre_mlp",
    "final",
]
ProjectionStyle = Literal["fused_qkv", "split_qkv"]
MLPKind = Literal["dense", "gated"]


class UnsupportedDecoderLayoutError(ValueError):
    """Raised when a decoder cannot satisfy a complete supported layout."""


# Short alias for callers that do not need to qualify the failure domain.
UnsupportedLayoutError = UnsupportedDecoderLayoutError


@dataclass(frozen=True, slots=True)
class EmbeddingLayout:
    token_path: str
    position_path: str | None
    dropout_path: str | None
    vocab_size: int
    hidden_size: int


@dataclass(frozen=True, slots=True)
class PositionLayout:
    kind: PositionKind
    operator_paths: tuple[str, ...]
    max_positions: int
    rotary_dim: int | None = None
    rope_theta: float | None = None
    rope_type: str | None = None


@dataclass(frozen=True, slots=True)
class NormLayout:
    path: str
    kind: NormKind
    placement: NormPlacement
    epsilon: float
    has_bias: bool


@dataclass(frozen=True, slots=True)
class AttentionLayout:
    module_path: str
    projection_style: ProjectionStyle
    fused_qkv_path: str | None
    query_path: str | None
    key_path: str | None
    value_path: str | None
    output_path: str
    query_heads: int
    key_value_heads: int
    head_dim: int
    hidden_size: int
    projection_weight_layout: Literal["input_output", "output_input"]
    projection_bias: bool
    attention_dropout: float
    query_norm: NormLayout | None = None
    key_norm: NormLayout | None = None

    @property
    def grouped_query_attention(self) -> bool:
        return self.query_heads != self.key_value_heads

    @property
    def query_width(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def key_value_width(self) -> int:
        return self.key_value_heads * self.head_dim


@dataclass(frozen=True, slots=True)
class MLPLayout:
    module_path: str
    kind: MLPKind
    input_projection_path: str | None
    gate_projection_path: str | None
    up_projection_path: str | None
    down_projection_path: str
    activation_path: str
    activation_name: str
    activation_module_type: str
    hidden_size: int
    intermediate_size: int
    projection_weight_layout: Literal["input_output", "output_input"]
    projection_bias: bool


@dataclass(frozen=True, slots=True)
class ResidualLayout:
    kind: Literal["serial_pre_norm"]
    attention_residual_source: Literal["block_input"]
    mlp_residual_source: Literal["post_attention_residual"]
    attention_norm_position: Literal["pre"]
    mlp_norm_position: Literal["pre"]


SERIAL_PRE_NORM_RESIDUAL = ResidualLayout(
    kind="serial_pre_norm",
    attention_residual_source="block_input",
    mlp_residual_source="post_attention_residual",
    attention_norm_position="pre",
    mlp_norm_position="pre",
)


@dataclass(frozen=True, slots=True)
class DecoderBlockLayout:
    index: int
    path: str
    attention_input_norm: NormLayout
    attention: AttentionLayout
    mlp_input_norm: NormLayout
    mlp: MLPLayout
    residual: ResidualLayout


@dataclass(frozen=True, slots=True)
class DecoderLayout:
    family: ArchitectureFamily
    model_type: str
    model_class: str
    hidden_size: int
    intermediate_size: int
    blocks_path: str
    embeddings: EmbeddingLayout
    position: PositionLayout
    blocks: tuple[DecoderBlockLayout, ...]
    final_norm: NormLayout
    lm_head_path: str
    tie_word_embeddings: bool

    @classmethod
    def from_model(cls, model: Any) -> "DecoderLayout":
        return build_decoder_layout(model)

    @property
    def layer_count(self) -> int:
        return len(self.blocks)

    def iter_operator_paths(self) -> Iterator[tuple[str, str]]:
        """Yield every required operator role and its already-validated path."""

        yield "token_embedding", self.embeddings.token_path
        if self.embeddings.position_path is not None:
            yield "position_embedding", self.embeddings.position_path
        if self.embeddings.dropout_path is not None:
            yield "embedding_dropout", self.embeddings.dropout_path
        for index, path in enumerate(self.position.operator_paths):
            yield f"position_operator.{index}", path
        yield "blocks", self.blocks_path
        for block in self.blocks:
            prefix = f"block.{block.index}"
            yield prefix, block.path
            yield f"{prefix}.attention_norm", block.attention_input_norm.path
            yield f"{prefix}.attention", block.attention.module_path
            if block.attention.fused_qkv_path is not None:
                yield f"{prefix}.attention.qkv", block.attention.fused_qkv_path
            else:
                assert block.attention.query_path is not None
                assert block.attention.key_path is not None
                assert block.attention.value_path is not None
                yield f"{prefix}.attention.q", block.attention.query_path
                yield f"{prefix}.attention.k", block.attention.key_path
                yield f"{prefix}.attention.v", block.attention.value_path
            yield f"{prefix}.attention.output", block.attention.output_path
            if block.attention.query_norm is not None:
                yield f"{prefix}.attention.q_norm", block.attention.query_norm.path
            if block.attention.key_norm is not None:
                yield f"{prefix}.attention.k_norm", block.attention.key_norm.path
            yield f"{prefix}.mlp_norm", block.mlp_input_norm.path
            yield f"{prefix}.mlp", block.mlp.module_path
            if block.mlp.input_projection_path is not None:
                yield f"{prefix}.mlp.input", block.mlp.input_projection_path
            if block.mlp.gate_projection_path is not None:
                yield f"{prefix}.mlp.gate", block.mlp.gate_projection_path
            if block.mlp.up_projection_path is not None:
                yield f"{prefix}.mlp.up", block.mlp.up_projection_path
            yield f"{prefix}.mlp.down", block.mlp.down_projection_path
            yield f"{prefix}.mlp.activation", block.mlp.activation_path
        yield "final_norm", self.final_norm.path
        yield "lm_head", self.lm_head_path

    @property
    def required_operator_paths(self) -> tuple[str, ...]:
        """All required paths, de-duplicated while preserving role order."""

        return tuple(dict.fromkeys(path for _, path in self.iter_operator_paths()))


_ACTIVATION_MODULE_TYPES: Mapping[str, frozenset[str]] = {
    "gelu": frozenset({"GELUActivation"}),
    "gelu_10": frozenset({"ClippedGELUActivation"}),
    "gelu_accurate": frozenset({"AccurateGELUActivation"}),
    "gelu_fast": frozenset({"FastGELUActivation"}),
    "gelu_new": frozenset({"NewGELUActivation"}),
    "gelu_python": frozenset({"GELUActivation"}),
    "gelu_python_tanh": frozenset({"GELUTanh"}),
    "gelu_pytorch_tanh": frozenset({"GELUTanh"}),
    "hardswish": frozenset({"Hardswish"}),
    "leaky_relu": frozenset({"LeakyReLU"}),
    "linear": frozenset({"LinearActivation"}),
    "mish": frozenset({"MishActivation"}),
    "prelu": frozenset({"PReLU"}),
    "quick_gelu": frozenset({"QuickGELUActivation"}),
    "relu": frozenset({"ReLU"}),
    "relu2": frozenset({"ReLUSquaredActivation"}),
    "relu6": frozenset({"ReLU6"}),
    "sigmoid": frozenset({"Sigmoid"}),
    "silu": frozenset({"SiLUActivation"}),
    "sqrtsoftplus": frozenset({"SqrtSoftplusActivation"}),
    "swish": frozenset({"SiLU"}),
    "tanh": frozenset({"Tanh"}),
}


def _unsupported(family: str, detail: str) -> UnsupportedDecoderLayoutError:
    return UnsupportedDecoderLayoutError(
        f"Unsupported {family} decoder layout: {detail}"
    )


def _required_attr(value: Any, name: str, family: str) -> Any:
    if not hasattr(value, name):
        raise _unsupported(family, f"required configuration field {name!r} is missing.")
    result = getattr(value, name)
    if result is None:
        raise _unsupported(family, f"required configuration field {name!r} is null.")
    return result


def _positive_int(value: Any, name: str, family: str) -> int:
    if isinstance(value, bool):
        raise _unsupported(family, f"{name} must be a positive integer, not bool.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _unsupported(family, f"{name} must be a positive integer.") from exc
    if result != value or result <= 0:
        raise _unsupported(family, f"{name} must be a positive integer.")
    return result


def _finite_float(
    value: Any, name: str, family: str, *, nonnegative: bool = False
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _unsupported(family, f"{name} must be a finite number.") from exc
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise _unsupported(family, f"{name} must be a finite nonnegative number.")
    return result


def resolve_module(model: Any, path: str, *, family: str = "decoder") -> Any:
    """Resolve an exact module path or raise a layout-domain exception."""

    current = model
    for segment in path.split("."):
        if segment.isdecimal():
            try:
                current = current[int(segment)]
            except (IndexError, KeyError, TypeError) as exc:
                raise _unsupported(
                    family, f"required operator {path!r} is missing."
                ) from exc
        else:
            try:
                current = getattr(current, segment)
            except AttributeError as exc:
                raise _unsupported(
                    family, f"required operator {path!r} is missing."
                ) from exc
    return current


def _require_module_type(model: Any, path: str, expected: str, family: str) -> Any:
    module = resolve_module(model, path, family=family)
    actual = type(module).__name__
    if actual != expected:
        raise _unsupported(
            family,
            f"required operator {path!r} must be {expected}, got {actual}.",
        )
    return module


def _require_exact_type(
    value: Any,
    *,
    expected_name: str,
    expected_module: str,
    label: str,
    family: str,
) -> Any:
    actual = type(value)
    if actual.__name__ != expected_name or actual.__module__ != expected_module:
        raise _unsupported(
            family,
            f"{label} must be exactly {expected_module}.{expected_name}, got "
            f"{actual.__module__}.{actual.__name__}.",
        )
    return value


def _weight_shape(module: Any, path: str, family: str) -> tuple[int, ...]:
    weight = getattr(module, "weight", None)
    shape = getattr(weight, "shape", None)
    if shape is None:
        raise _unsupported(family, f"required operator {path!r} has no weight tensor.")
    return tuple(int(size) for size in shape)


def _require_weight_shape(
    module: Any,
    path: str,
    expected: tuple[int, ...],
    family: str,
) -> None:
    actual = _weight_shape(module, path, family)
    if actual != expected:
        raise _unsupported(
            family,
            f"operator {path!r} weight shape must be {expected}, got {actual}.",
        )


def _require_bias(
    module: Any, path: str, expected: bool, width: int, family: str
) -> None:
    bias = getattr(module, "bias", None)
    actual = bias is not None
    if actual != expected:
        state = "have" if expected else "not have"
        raise _unsupported(family, f"operator {path!r} must {state} a bias.")
    if expected:
        shape = tuple(int(size) for size in bias.shape)
        if shape != (width,):
            raise _unsupported(
                family,
                f"operator {path!r} bias shape must be {(width,)}, got {shape}.",
            )


def _norm_layout(
    model: Any,
    *,
    path: str,
    family: ArchitectureFamily,
    kind: NormKind,
    placement: NormPlacement,
    hidden_size: int,
    epsilon: float,
    expected_type: str | None = None,
    expected_module: str | None = None,
) -> NormLayout:
    if expected_type is None:
        expected_type = "LayerNorm" if kind == "layer_norm" else "LlamaRMSNorm"
    if expected_module is None:
        module = _require_module_type(model, path, expected_type, family)
    else:
        module = _require_exact_type(
            resolve_module(model, path, family=family),
            expected_name=expected_type,
            expected_module=expected_module,
            label=f"operator {path!r}",
            family=family,
        )
    _require_weight_shape(module, path, (hidden_size,), family)
    has_bias = getattr(module, "bias", None) is not None
    expected_bias = kind == "layer_norm"
    if has_bias != expected_bias:
        raise _unsupported(
            family,
            f"operator {path!r} has an incompatible {kind} bias contract.",
        )
    if has_bias:
        _require_bias(module, path, True, hidden_size, family)
    actual_epsilon = getattr(module, "eps", None)
    if actual_epsilon is None:
        actual_epsilon = getattr(module, "variance_epsilon", None)
    if actual_epsilon is None or not math.isclose(
        float(actual_epsilon), epsilon, rel_tol=0.0, abs_tol=0.0
    ):
        raise _unsupported(
            family,
            f"operator {path!r} epsilon does not match configuration value {epsilon}.",
        )
    return NormLayout(
        path=path,
        kind=kind,
        placement=placement,
        epsilon=epsilon,
        has_bias=has_bias,
    )


def _activation_layout(
    model: Any,
    *,
    path: str,
    configured_name: Any,
    family: ArchitectureFamily,
) -> tuple[str, str]:
    if not isinstance(configured_name, str) or not configured_name:
        raise _unsupported(family, "configured activation must be a non-empty string.")
    expected_types = _ACTIVATION_MODULE_TYPES.get(configured_name)
    if expected_types is None:
        raise _unsupported(
            family, f"activation {configured_name!r} is not supported explicitly."
        )
    module = resolve_module(model, path, family=family)
    actual_type = type(module).__name__
    if actual_type not in expected_types:
        raise _unsupported(
            family,
            f"activation operator {path!r} is {actual_type}, but configuration "
            f"requires {configured_name!r} ({sorted(expected_types)}).",
        )
    return configured_name, actual_type


def _validate_causal_wrapper(model: Any, family: ArchitectureFamily) -> Any:
    config = getattr(model, "config", None)
    if config is None:
        raise _unsupported(family, "model has no Hugging Face configuration.")
    if bool(getattr(config, "is_encoder_decoder", False)):
        raise _unsupported(
            family, "encoder-decoder models are outside this proof slice."
        )
    return config


def _validate_tying(
    model: Any,
    *,
    embedding_path: str,
    head_path: str,
    configured: Any,
    family: ArchitectureFamily,
) -> bool:
    if not isinstance(configured, bool):
        raise _unsupported(family, "tie_word_embeddings must be a bool.")
    embedding = resolve_module(model, embedding_path, family=family)
    head = resolve_module(model, head_path, family=family)
    observed = getattr(embedding, "weight", None) is getattr(head, "weight", None)
    if observed != configured:
        raise _unsupported(
            family,
            "configured and observed input/output weight tying disagree "
            f"(configured={configured}, observed={observed}).",
        )
    return observed


def _build_gpt2_layout(model: Any) -> DecoderLayout:
    family: ArchitectureFamily = "gpt2"
    config = _validate_causal_wrapper(model, family)
    if bool(getattr(config, "add_cross_attention", False)):
        raise _unsupported(
            family, "cross-attention blocks are outside this proof slice."
        )

    hidden_size = _positive_int(
        _required_attr(config, "n_embd", family), "n_embd", family
    )
    layer_count = _positive_int(
        _required_attr(config, "n_layer", family), "n_layer", family
    )
    query_heads = _positive_int(
        _required_attr(config, "n_head", family), "n_head", family
    )
    if hidden_size % query_heads:
        raise _unsupported(family, "n_embd must be divisible by n_head.")
    head_dim = hidden_size // query_heads
    configured_inner = getattr(config, "n_inner", None)
    intermediate_size = (
        hidden_size * 4
        if configured_inner is None
        else _positive_int(configured_inner, "n_inner", family)
    )
    vocab_size = _positive_int(
        _required_attr(config, "vocab_size", family), "vocab_size", family
    )
    max_positions = _positive_int(
        _required_attr(config, "n_positions", family), "n_positions", family
    )
    epsilon = _finite_float(
        _required_attr(config, "layer_norm_epsilon", family),
        "layer_norm_epsilon",
        family,
        nonnegative=True,
    )
    attention_dropout = _finite_float(
        _required_attr(config, "attn_pdrop", family),
        "attn_pdrop",
        family,
        nonnegative=True,
    )

    embedding = _require_module_type(model, "transformer.wte", "Embedding", family)
    position_embedding = _require_module_type(
        model, "transformer.wpe", "Embedding", family
    )
    _require_module_type(model, "transformer.drop", "Dropout", family)
    _require_weight_shape(
        embedding, "transformer.wte", (vocab_size, hidden_size), family
    )
    _require_weight_shape(
        position_embedding,
        "transformer.wpe",
        (max_positions, hidden_size),
        family,
    )
    blocks_module = _require_module_type(model, "transformer.h", "ModuleList", family)
    if len(blocks_module) != layer_count:
        raise _unsupported(
            family,
            f"transformer.h must contain {layer_count} blocks, got {len(blocks_module)}.",
        )

    activation_name = _required_attr(config, "activation_function", family)
    blocks: list[DecoderBlockLayout] = []
    for index in range(layer_count):
        prefix = f"transformer.h.{index}"
        _require_module_type(model, prefix, "GPT2Block", family)
        attention_path = f"{prefix}.attn"
        _require_module_type(model, attention_path, "GPT2Attention", family)
        fused_path = f"{attention_path}.c_attn"
        output_path = f"{attention_path}.c_proj"
        fused = _require_module_type(model, fused_path, "Conv1D", family)
        output = _require_module_type(model, output_path, "Conv1D", family)
        _require_weight_shape(fused, fused_path, (hidden_size, hidden_size * 3), family)
        _require_weight_shape(output, output_path, (hidden_size, hidden_size), family)
        _require_bias(fused, fused_path, True, hidden_size * 3, family)
        _require_bias(output, output_path, True, hidden_size, family)
        _require_module_type(model, f"{attention_path}.attn_dropout", "Dropout", family)
        _require_module_type(
            model, f"{attention_path}.resid_dropout", "Dropout", family
        )

        attention_module = resolve_module(model, attention_path, family=family)
        if int(getattr(attention_module, "num_heads", -1)) != query_heads:
            raise _unsupported(
                family, f"operator {attention_path!r} head count disagrees with config."
            )
        if int(getattr(attention_module, "head_dim", -1)) != head_dim:
            raise _unsupported(
                family,
                f"operator {attention_path!r} head dimension disagrees with config.",
            )

        mlp_path = f"{prefix}.mlp"
        _require_module_type(model, mlp_path, "GPT2MLP", family)
        input_path = f"{mlp_path}.c_fc"
        down_path = f"{mlp_path}.c_proj"
        input_projection = _require_module_type(model, input_path, "Conv1D", family)
        down_projection = _require_module_type(model, down_path, "Conv1D", family)
        _require_weight_shape(
            input_projection,
            input_path,
            (hidden_size, intermediate_size),
            family,
        )
        _require_weight_shape(
            down_projection,
            down_path,
            (intermediate_size, hidden_size),
            family,
        )
        _require_bias(input_projection, input_path, True, intermediate_size, family)
        _require_bias(down_projection, down_path, True, hidden_size, family)
        activation_path = f"{mlp_path}.act"
        actual_activation, activation_type = _activation_layout(
            model,
            path=activation_path,
            configured_name=activation_name,
            family=family,
        )
        _require_module_type(model, f"{mlp_path}.dropout", "Dropout", family)

        attention_norm = _norm_layout(
            model,
            path=f"{prefix}.ln_1",
            family=family,
            kind="layer_norm",
            placement="pre_attention",
            hidden_size=hidden_size,
            epsilon=epsilon,
        )
        mlp_norm = _norm_layout(
            model,
            path=f"{prefix}.ln_2",
            family=family,
            kind="layer_norm",
            placement="pre_mlp",
            hidden_size=hidden_size,
            epsilon=epsilon,
        )
        blocks.append(
            DecoderBlockLayout(
                index=index,
                path=prefix,
                attention_input_norm=attention_norm,
                attention=AttentionLayout(
                    module_path=attention_path,
                    projection_style="fused_qkv",
                    fused_qkv_path=fused_path,
                    query_path=None,
                    key_path=None,
                    value_path=None,
                    output_path=output_path,
                    query_heads=query_heads,
                    key_value_heads=query_heads,
                    head_dim=head_dim,
                    hidden_size=hidden_size,
                    projection_weight_layout="input_output",
                    projection_bias=True,
                    attention_dropout=attention_dropout,
                ),
                mlp_input_norm=mlp_norm,
                mlp=MLPLayout(
                    module_path=mlp_path,
                    kind="dense",
                    input_projection_path=input_path,
                    gate_projection_path=None,
                    up_projection_path=None,
                    down_projection_path=down_path,
                    activation_path=activation_path,
                    activation_name=actual_activation,
                    activation_module_type=activation_type,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    projection_weight_layout="input_output",
                    projection_bias=True,
                ),
                residual=SERIAL_PRE_NORM_RESIDUAL,
            )
        )

    final_norm = _norm_layout(
        model,
        path="transformer.ln_f",
        family=family,
        kind="layer_norm",
        placement="final",
        hidden_size=hidden_size,
        epsilon=epsilon,
    )
    lm_head = _require_module_type(model, "lm_head", "Linear", family)
    _require_weight_shape(lm_head, "lm_head", (vocab_size, hidden_size), family)
    _require_bias(lm_head, "lm_head", False, vocab_size, family)
    tied = _validate_tying(
        model,
        embedding_path="transformer.wte",
        head_path="lm_head",
        configured=_required_attr(config, "tie_word_embeddings", family),
        family=family,
    )
    return DecoderLayout(
        family=family,
        model_type="gpt2",
        model_class=type(model).__name__,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        blocks_path="transformer.h",
        embeddings=EmbeddingLayout(
            token_path="transformer.wte",
            position_path="transformer.wpe",
            dropout_path="transformer.drop",
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        ),
        position=PositionLayout(
            kind="learned_absolute",
            operator_paths=("transformer.wpe",),
            max_positions=max_positions,
        ),
        blocks=tuple(blocks),
        final_norm=final_norm,
        lm_head_path="lm_head",
        tie_word_embeddings=tied,
    )


def _llama_rope_layout(
    model: Any,
    config: Any,
    *,
    layer_count: int,
    head_dim: int,
) -> PositionLayout:
    family: ArchitectureFamily = "llama"
    shared_path = "model.rotary_emb"
    try:
        _require_module_type(model, shared_path, "LlamaRotaryEmbedding", family)
    except UnsupportedDecoderLayoutError:
        paths = tuple(
            f"model.layers.{index}.self_attn.rotary_emb" for index in range(layer_count)
        )
        for path in paths:
            _require_module_type(model, path, "LlamaRotaryEmbedding", family)
    else:
        paths = (shared_path,)

    rope_parameters = getattr(config, "rope_parameters", None)
    rope_scaling = getattr(config, "rope_scaling", None)
    rope_theta_value = getattr(config, "rope_theta", None)
    rope_type_value: Any = None
    if isinstance(rope_parameters, Mapping):
        if rope_theta_value is None:
            rope_theta_value = rope_parameters.get("rope_theta")
        rope_type_value = rope_parameters.get("rope_type")
    if isinstance(rope_scaling, Mapping) and rope_type_value is None:
        rope_type_value = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if rope_theta_value is None:
        rope_theta_value = 10_000.0
    if rope_type_value is None:
        rope_type_value = "default"
    if not isinstance(rope_type_value, str) or not rope_type_value:
        raise _unsupported(family, "RoPE type must be a non-empty string.")
    rope_theta = _finite_float(rope_theta_value, "rope_theta", family)
    if rope_theta <= 0.0:
        raise _unsupported(family, "rope_theta must be positive.")
    max_positions = _positive_int(
        _required_attr(config, "max_position_embeddings", family),
        "max_position_embeddings",
        family,
    )
    return PositionLayout(
        kind="rope",
        operator_paths=paths,
        max_positions=max_positions,
        rotary_dim=head_dim,
        rope_theta=rope_theta,
        rope_type=rope_type_value,
    )


def _build_llama_layout(model: Any) -> DecoderLayout:
    family: ArchitectureFamily = "llama"
    config = _validate_causal_wrapper(model, family)
    hidden_size = _positive_int(
        _required_attr(config, "hidden_size", family), "hidden_size", family
    )
    intermediate_size = _positive_int(
        _required_attr(config, "intermediate_size", family),
        "intermediate_size",
        family,
    )
    layer_count = _positive_int(
        _required_attr(config, "num_hidden_layers", family),
        "num_hidden_layers",
        family,
    )
    query_heads = _positive_int(
        _required_attr(config, "num_attention_heads", family),
        "num_attention_heads",
        family,
    )
    key_value_heads_value = getattr(config, "num_key_value_heads", None)
    key_value_heads = (
        query_heads
        if key_value_heads_value is None
        else _positive_int(key_value_heads_value, "num_key_value_heads", family)
    )
    if query_heads % key_value_heads:
        raise _unsupported(
            family, "num_attention_heads must be divisible by num_key_value_heads."
        )
    configured_head_dim = getattr(config, "head_dim", None)
    if configured_head_dim is None:
        if hidden_size % query_heads:
            raise _unsupported(
                family, "hidden_size must be divisible by num_attention_heads."
            )
        head_dim = hidden_size // query_heads
    else:
        head_dim = _positive_int(configured_head_dim, "head_dim", family)
    vocab_size = _positive_int(
        _required_attr(config, "vocab_size", family), "vocab_size", family
    )
    epsilon = _finite_float(
        _required_attr(config, "rms_norm_eps", family),
        "rms_norm_eps",
        family,
        nonnegative=True,
    )
    attention_dropout = _finite_float(
        getattr(config, "attention_dropout", 0.0),
        "attention_dropout",
        family,
        nonnegative=True,
    )
    attention_bias = getattr(config, "attention_bias", False)
    mlp_bias = getattr(config, "mlp_bias", False)
    if not isinstance(attention_bias, bool) or not isinstance(mlp_bias, bool):
        raise _unsupported(family, "attention_bias and mlp_bias must be bool values.")

    embedding = _require_module_type(model, "model.embed_tokens", "Embedding", family)
    _require_weight_shape(
        embedding, "model.embed_tokens", (vocab_size, hidden_size), family
    )
    blocks_module = _require_module_type(model, "model.layers", "ModuleList", family)
    if len(blocks_module) != layer_count:
        raise _unsupported(
            family,
            f"model.layers must contain {layer_count} blocks, got {len(blocks_module)}.",
        )
    position = _llama_rope_layout(
        model,
        config,
        layer_count=layer_count,
        head_dim=head_dim,
    )

    query_width = query_heads * head_dim
    key_value_width = key_value_heads * head_dim
    activation_name = _required_attr(config, "hidden_act", family)
    blocks: list[DecoderBlockLayout] = []
    for index in range(layer_count):
        prefix = f"model.layers.{index}"
        _require_module_type(model, prefix, "LlamaDecoderLayer", family)
        attention_path = f"{prefix}.self_attn"
        attention_module = _require_module_type(
            model, attention_path, "LlamaAttention", family
        )
        q_path = f"{attention_path}.q_proj"
        k_path = f"{attention_path}.k_proj"
        v_path = f"{attention_path}.v_proj"
        output_path = f"{attention_path}.o_proj"
        projections = (
            (q_path, query_width, hidden_size),
            (k_path, key_value_width, hidden_size),
            (v_path, key_value_width, hidden_size),
            (output_path, hidden_size, query_width),
        )
        for path, rows, columns in projections:
            projection = _require_module_type(model, path, "Linear", family)
            _require_weight_shape(projection, path, (rows, columns), family)
            _require_bias(projection, path, attention_bias, rows, family)
        if int(getattr(attention_module, "head_dim", -1)) != head_dim:
            raise _unsupported(
                family,
                f"operator {attention_path!r} head dimension disagrees with config.",
            )
        expected_groups = query_heads // key_value_heads
        if (
            int(getattr(attention_module, "num_key_value_groups", -1))
            != expected_groups
        ):
            raise _unsupported(
                family,
                f"operator {attention_path!r} GQA grouping disagrees with config.",
            )

        mlp_path = f"{prefix}.mlp"
        _require_module_type(model, mlp_path, "LlamaMLP", family)
        gate_path = f"{mlp_path}.gate_proj"
        up_path = f"{mlp_path}.up_proj"
        down_path = f"{mlp_path}.down_proj"
        for path, rows, columns in (
            (gate_path, intermediate_size, hidden_size),
            (up_path, intermediate_size, hidden_size),
            (down_path, hidden_size, intermediate_size),
        ):
            projection = _require_module_type(model, path, "Linear", family)
            _require_weight_shape(projection, path, (rows, columns), family)
            _require_bias(projection, path, mlp_bias, rows, family)
        activation_path = f"{mlp_path}.act_fn"
        actual_activation, activation_type = _activation_layout(
            model,
            path=activation_path,
            configured_name=activation_name,
            family=family,
        )
        attention_norm = _norm_layout(
            model,
            path=f"{prefix}.input_layernorm",
            family=family,
            kind="rms_norm",
            placement="pre_attention",
            hidden_size=hidden_size,
            epsilon=epsilon,
        )
        mlp_norm = _norm_layout(
            model,
            path=f"{prefix}.post_attention_layernorm",
            family=family,
            kind="rms_norm",
            placement="pre_mlp",
            hidden_size=hidden_size,
            epsilon=epsilon,
        )
        blocks.append(
            DecoderBlockLayout(
                index=index,
                path=prefix,
                attention_input_norm=attention_norm,
                attention=AttentionLayout(
                    module_path=attention_path,
                    projection_style="split_qkv",
                    fused_qkv_path=None,
                    query_path=q_path,
                    key_path=k_path,
                    value_path=v_path,
                    output_path=output_path,
                    query_heads=query_heads,
                    key_value_heads=key_value_heads,
                    head_dim=head_dim,
                    hidden_size=hidden_size,
                    projection_weight_layout="output_input",
                    projection_bias=attention_bias,
                    attention_dropout=attention_dropout,
                ),
                mlp_input_norm=mlp_norm,
                mlp=MLPLayout(
                    module_path=mlp_path,
                    kind="gated",
                    input_projection_path=None,
                    gate_projection_path=gate_path,
                    up_projection_path=up_path,
                    down_projection_path=down_path,
                    activation_path=activation_path,
                    activation_name=actual_activation,
                    activation_module_type=activation_type,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    projection_weight_layout="output_input",
                    projection_bias=mlp_bias,
                ),
                residual=SERIAL_PRE_NORM_RESIDUAL,
            )
        )

    final_norm = _norm_layout(
        model,
        path="model.norm",
        family=family,
        kind="rms_norm",
        placement="final",
        hidden_size=hidden_size,
        epsilon=epsilon,
    )
    lm_head = _require_module_type(model, "lm_head", "Linear", family)
    _require_weight_shape(lm_head, "lm_head", (vocab_size, hidden_size), family)
    _require_bias(lm_head, "lm_head", False, vocab_size, family)
    tied = _validate_tying(
        model,
        embedding_path="model.embed_tokens",
        head_path="lm_head",
        configured=_required_attr(config, "tie_word_embeddings", family),
        family=family,
    )
    return DecoderLayout(
        family=family,
        model_type="llama",
        model_class=type(model).__name__,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        blocks_path="model.layers",
        embeddings=EmbeddingLayout(
            token_path="model.embed_tokens",
            position_path=None,
            dropout_path=None,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        ),
        position=position,
        blocks=tuple(blocks),
        final_norm=final_norm,
        lm_head_path="lm_head",
        tie_word_embeddings=tied,
    )


_QWEN3_CONFIG_MODULE = "transformers.models.qwen3.configuration_qwen3"
_QWEN3_MODELING_MODULE = "transformers.models.qwen3.modeling_qwen3"


def _qwen3_rope_layout(
    model: Any,
    config: Any,
    *,
    head_dim: int,
) -> PositionLayout:
    family: ArchitectureFamily = "qwen3"
    path = "model.rotary_emb"
    rotary = _require_exact_type(
        resolve_module(model, path, family=family),
        expected_name="Qwen3RotaryEmbedding",
        expected_module=_QWEN3_MODELING_MODULE,
        label=f"operator {path!r}",
        family=family,
    )
    if head_dim % 2:
        raise _unsupported(family, "head_dim must be even for full-head RoPE.")

    rope_parameters = getattr(config, "rope_parameters", None)
    rope_scaling = getattr(config, "rope_scaling", None)
    rope_theta_value = getattr(config, "rope_theta", None)
    rope_type_value: Any = None
    if isinstance(rope_parameters, Mapping):
        if rope_theta_value is None:
            rope_theta_value = rope_parameters.get("rope_theta")
        rope_type_value = rope_parameters.get("rope_type")
    if isinstance(rope_scaling, Mapping):
        if rope_theta_value is None:
            rope_theta_value = rope_scaling.get("rope_theta")
        if rope_type_value is None:
            rope_type_value = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if rope_theta_value is None:
        rope_theta_value = 10_000.0
    if rope_type_value is None:
        rope_type_value = "default"
    if rope_type_value != "default":
        raise _unsupported(
            family,
            f"RoPE type must be 'default', got {rope_type_value!r}.",
        )
    rope_theta = _finite_float(rope_theta_value, "rope_theta", family)
    if rope_theta <= 0.0:
        raise _unsupported(family, "rope_theta must be positive.")
    if getattr(rotary, "rope_type", None) != rope_type_value:
        raise _unsupported(
            family,
            f"operator {path!r} RoPE type disagrees with configuration.",
        )
    inv_freq = getattr(rotary, "inv_freq", None)
    inv_freq_shape = getattr(inv_freq, "shape", None)
    expected_inv_freq_shape = (head_dim // 2,)
    if inv_freq_shape is None or tuple(int(size) for size in inv_freq_shape) != (
        expected_inv_freq_shape
    ):
        actual = None if inv_freq_shape is None else tuple(inv_freq_shape)
        raise _unsupported(
            family,
            f"operator {path!r} inv_freq shape must be "
            f"{expected_inv_freq_shape}, got {actual}.",
        )
    max_positions = _positive_int(
        _required_attr(config, "max_position_embeddings", family),
        "max_position_embeddings",
        family,
    )
    if int(getattr(rotary, "max_seq_len_cached", -1)) != max_positions:
        raise _unsupported(
            family,
            f"operator {path!r} cached position limit disagrees with config.",
        )
    return PositionLayout(
        kind="rope",
        operator_paths=(path,),
        max_positions=max_positions,
        rotary_dim=head_dim,
        rope_theta=rope_theta,
        rope_type=rope_type_value,
    )


def _build_qwen3_layout(model: Any) -> DecoderLayout:
    family: ArchitectureFamily = "qwen3"
    _require_exact_type(
        model,
        expected_name="Qwen3ForCausalLM",
        expected_module=_QWEN3_MODELING_MODULE,
        label="model",
        family=family,
    )
    config = _validate_causal_wrapper(model, family)
    _require_exact_type(
        config,
        expected_name="Qwen3Config",
        expected_module=_QWEN3_CONFIG_MODULE,
        label="configuration",
        family=family,
    )
    _require_exact_type(
        resolve_module(model, "model", family=family),
        expected_name="Qwen3Model",
        expected_module=_QWEN3_MODELING_MODULE,
        label="operator 'model'",
        family=family,
    )

    hidden_size = _positive_int(
        _required_attr(config, "hidden_size", family), "hidden_size", family
    )
    intermediate_size = _positive_int(
        _required_attr(config, "intermediate_size", family),
        "intermediate_size",
        family,
    )
    layer_count = _positive_int(
        _required_attr(config, "num_hidden_layers", family),
        "num_hidden_layers",
        family,
    )
    query_heads = _positive_int(
        _required_attr(config, "num_attention_heads", family),
        "num_attention_heads",
        family,
    )
    key_value_heads = _positive_int(
        _required_attr(config, "num_key_value_heads", family),
        "num_key_value_heads",
        family,
    )
    if not 1 < key_value_heads < query_heads or query_heads % key_value_heads:
        raise _unsupported(
            family,
            "dense Qwen3 requires grouped-query attention with "
            "more than one num_key_value_heads strictly dividing "
            "num_attention_heads.",
        )
    head_dim = _positive_int(
        _required_attr(config, "head_dim", family), "head_dim", family
    )
    vocab_size = _positive_int(
        _required_attr(config, "vocab_size", family), "vocab_size", family
    )
    epsilon = _finite_float(
        _required_attr(config, "rms_norm_eps", family),
        "rms_norm_eps",
        family,
        nonnegative=True,
    )
    attention_dropout = _finite_float(
        _required_attr(config, "attention_dropout", family),
        "attention_dropout",
        family,
        nonnegative=True,
    )
    if attention_dropout >= 1.0:
        raise _unsupported(family, "attention_dropout must be less than one.")
    if _required_attr(config, "hidden_act", family) != "silu":
        raise _unsupported(family, "dense Qwen3 requires hidden_act='silu'.")
    if _required_attr(config, "attention_bias", family) is not False:
        raise _unsupported(family, "dense Qwen3 requires bias-free attention.")
    if _required_attr(config, "use_sliding_window", family) is not False:
        raise _unsupported(
            family, "sliding-window Qwen3 variants are outside this proof slice."
        )
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None and (
        not isinstance(layer_types, (list, tuple))
        or len(layer_types) != layer_count
        or any(layer_type != "full_attention" for layer_type in layer_types)
    ):
        raise _unsupported(
            family, "all Qwen3 layers must use full_attention in this proof slice."
        )

    embedding = _require_module_type(model, "model.embed_tokens", "Embedding", family)
    _require_weight_shape(
        embedding, "model.embed_tokens", (vocab_size, hidden_size), family
    )
    blocks_module = _require_module_type(model, "model.layers", "ModuleList", family)
    if len(blocks_module) != layer_count:
        raise _unsupported(
            family,
            f"model.layers must contain {layer_count} blocks, got {len(blocks_module)}.",
        )
    position = _qwen3_rope_layout(model, config, head_dim=head_dim)

    query_width = query_heads * head_dim
    key_value_width = key_value_heads * head_dim
    blocks: list[DecoderBlockLayout] = []
    for index in range(layer_count):
        prefix = f"model.layers.{index}"
        _require_exact_type(
            resolve_module(model, prefix, family=family),
            expected_name="Qwen3DecoderLayer",
            expected_module=_QWEN3_MODELING_MODULE,
            label=f"operator {prefix!r}",
            family=family,
        )
        attention_path = f"{prefix}.self_attn"
        attention_module = _require_exact_type(
            resolve_module(model, attention_path, family=family),
            expected_name="Qwen3Attention",
            expected_module=_QWEN3_MODELING_MODULE,
            label=f"operator {attention_path!r}",
            family=family,
        )
        q_path = f"{attention_path}.q_proj"
        k_path = f"{attention_path}.k_proj"
        v_path = f"{attention_path}.v_proj"
        output_path = f"{attention_path}.o_proj"
        for path, rows, columns in (
            (q_path, query_width, hidden_size),
            (k_path, key_value_width, hidden_size),
            (v_path, key_value_width, hidden_size),
            (output_path, hidden_size, query_width),
        ):
            projection = _require_module_type(model, path, "Linear", family)
            _require_weight_shape(projection, path, (rows, columns), family)
            _require_bias(projection, path, False, rows, family)
        if int(getattr(attention_module, "head_dim", -1)) != head_dim:
            raise _unsupported(
                family,
                f"operator {attention_path!r} head dimension disagrees with config.",
            )
        expected_groups = query_heads // key_value_heads
        if int(getattr(attention_module, "num_key_value_groups", -1)) != expected_groups:
            raise _unsupported(
                family,
                f"operator {attention_path!r} GQA grouping disagrees with config.",
            )
        if not math.isclose(
            float(getattr(attention_module, "scaling", math.nan)),
            head_dim**-0.5,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise _unsupported(
                family, f"operator {attention_path!r} attention scaling is incompatible."
            )
        if getattr(attention_module, "is_causal", None) is not True:
            raise _unsupported(
                family, f"operator {attention_path!r} must be causal."
            )
        if getattr(attention_module, "sliding_window", None) is not None:
            raise _unsupported(
                family,
                f"operator {attention_path!r} unexpectedly enables sliding attention.",
            )

        q_norm = _norm_layout(
            model,
            path=f"{attention_path}.q_norm",
            family=family,
            kind="rms_norm",
            placement="query_head",
            hidden_size=head_dim,
            epsilon=epsilon,
            expected_type="Qwen3RMSNorm",
            expected_module=_QWEN3_MODELING_MODULE,
        )
        k_norm = _norm_layout(
            model,
            path=f"{attention_path}.k_norm",
            family=family,
            kind="rms_norm",
            placement="key_head",
            hidden_size=head_dim,
            epsilon=epsilon,
            expected_type="Qwen3RMSNorm",
            expected_module=_QWEN3_MODELING_MODULE,
        )

        mlp_path = f"{prefix}.mlp"
        _require_exact_type(
            resolve_module(model, mlp_path, family=family),
            expected_name="Qwen3MLP",
            expected_module=_QWEN3_MODELING_MODULE,
            label=f"operator {mlp_path!r}",
            family=family,
        )
        gate_path = f"{mlp_path}.gate_proj"
        up_path = f"{mlp_path}.up_proj"
        down_path = f"{mlp_path}.down_proj"
        for path, rows, columns in (
            (gate_path, intermediate_size, hidden_size),
            (up_path, intermediate_size, hidden_size),
            (down_path, hidden_size, intermediate_size),
        ):
            projection = _require_module_type(model, path, "Linear", family)
            _require_weight_shape(projection, path, (rows, columns), family)
            _require_bias(projection, path, False, rows, family)
        activation_path = f"{mlp_path}.act_fn"
        actual_activation, activation_type = _activation_layout(
            model,
            path=activation_path,
            configured_name="silu",
            family=family,
        )
        attention_norm = _norm_layout(
            model,
            path=f"{prefix}.input_layernorm",
            family=family,
            kind="rms_norm",
            placement="pre_attention",
            hidden_size=hidden_size,
            epsilon=epsilon,
            expected_type="Qwen3RMSNorm",
            expected_module=_QWEN3_MODELING_MODULE,
        )
        mlp_norm = _norm_layout(
            model,
            path=f"{prefix}.post_attention_layernorm",
            family=family,
            kind="rms_norm",
            placement="pre_mlp",
            hidden_size=hidden_size,
            epsilon=epsilon,
            expected_type="Qwen3RMSNorm",
            expected_module=_QWEN3_MODELING_MODULE,
        )
        blocks.append(
            DecoderBlockLayout(
                index=index,
                path=prefix,
                attention_input_norm=attention_norm,
                attention=AttentionLayout(
                    module_path=attention_path,
                    projection_style="split_qkv",
                    fused_qkv_path=None,
                    query_path=q_path,
                    key_path=k_path,
                    value_path=v_path,
                    output_path=output_path,
                    query_heads=query_heads,
                    key_value_heads=key_value_heads,
                    head_dim=head_dim,
                    hidden_size=hidden_size,
                    projection_weight_layout="output_input",
                    projection_bias=False,
                    attention_dropout=attention_dropout,
                    query_norm=q_norm,
                    key_norm=k_norm,
                ),
                mlp_input_norm=mlp_norm,
                mlp=MLPLayout(
                    module_path=mlp_path,
                    kind="gated",
                    input_projection_path=None,
                    gate_projection_path=gate_path,
                    up_projection_path=up_path,
                    down_projection_path=down_path,
                    activation_path=activation_path,
                    activation_name=actual_activation,
                    activation_module_type=activation_type,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    projection_weight_layout="output_input",
                    projection_bias=False,
                ),
                residual=SERIAL_PRE_NORM_RESIDUAL,
            )
        )

    final_norm = _norm_layout(
        model,
        path="model.norm",
        family=family,
        kind="rms_norm",
        placement="final",
        hidden_size=hidden_size,
        epsilon=epsilon,
        expected_type="Qwen3RMSNorm",
        expected_module=_QWEN3_MODELING_MODULE,
    )
    lm_head = _require_module_type(model, "lm_head", "Linear", family)
    _require_weight_shape(lm_head, "lm_head", (vocab_size, hidden_size), family)
    _require_bias(lm_head, "lm_head", False, vocab_size, family)
    tied = _validate_tying(
        model,
        embedding_path="model.embed_tokens",
        head_path="lm_head",
        configured=_required_attr(config, "tie_word_embeddings", family),
        family=family,
    )
    return DecoderLayout(
        family=family,
        model_type="qwen3",
        model_class=type(model).__name__,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        blocks_path="model.layers",
        embeddings=EmbeddingLayout(
            token_path="model.embed_tokens",
            position_path=None,
            dropout_path=None,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        ),
        position=position,
        blocks=tuple(blocks),
        final_norm=final_norm,
        lm_head_path="lm_head",
        tie_word_embeddings=tied,
    )


def build_decoder_layout(model: Any) -> DecoderLayout:
    """Build and validate a complete GPT-2, Llama, or Qwen3 decoder layout."""

    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    if model_type == "gpt2":
        return _build_gpt2_layout(model)
    if model_type == "llama":
        return _build_llama_layout(model)
    if model_type == "qwen3":
        return _build_qwen3_layout(model)
    shown = "<missing>" if model_type is None else repr(model_type)
    raise UnsupportedDecoderLayoutError(
        "Unsupported causal decoder architecture: model_type "
        f"{shown}; supported model types are 'gpt2', 'llama', and 'qwen3'."
    )


discover_decoder_layout = build_decoder_layout


__all__ = [
    "AttentionLayout",
    "DecoderBlockLayout",
    "DecoderLayout",
    "EmbeddingLayout",
    "MLPLayout",
    "NormLayout",
    "PositionLayout",
    "ResidualLayout",
    "UnsupportedDecoderLayoutError",
    "UnsupportedLayoutError",
    "build_decoder_layout",
    "discover_decoder_layout",
    "resolve_module",
]
