"""Exact Gemma2 and text-only Gemma3 decoder layout validation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from neural_forge.hf_arch_layout import (
    AttentionLayout,
    DecoderBlockLayout,
    DecoderLayout,
    EmbeddingLayout,
    MLPLayout,
    PositionLayout,
    SERIAL_PRE_POST_NORM_RESIDUAL,
    _activation_layout,
    _finite_float,
    _norm_layout,
    _positive_int,
    _require_bias,
    _require_exact_type,
    _require_module_type,
    _require_weight_shape,
    _required_attr,
    _unsupported,
    _validate_causal_wrapper,
    _validate_tying,
    resolve_module,
)


_GEMMA2_CONFIG_MODULE = "transformers.models.gemma2.configuration_gemma2"
_GEMMA2_MODELING_MODULE = "transformers.models.gemma2.modeling_gemma2"
_GEMMA3_CONFIG_MODULE = "transformers.models.gemma3.configuration_gemma3"
_GEMMA3_MODELING_MODULE = "transformers.models.gemma3.modeling_gemma3"
_LAYER_TYPES = frozenset({"full_attention", "sliding_attention"})


@dataclass(frozen=True, slots=True)
class _GemmaSpec:
    family: str
    model_type: str
    config_name: str
    config_module: str
    wrapper_name: str
    model_name: str
    embedding_name: str
    decoder_name: str
    attention_name: str
    mlp_name: str
    norm_name: str
    rotary_name: str
    modeling_module: str
    has_query_key_norms: bool


_GEMMA2 = _GemmaSpec(
    family="gemma2",
    model_type="gemma2",
    config_name="Gemma2Config",
    config_module=_GEMMA2_CONFIG_MODULE,
    wrapper_name="Gemma2ForCausalLM",
    model_name="Gemma2Model",
    embedding_name="Gemma2TextScaledWordEmbedding",
    decoder_name="Gemma2DecoderLayer",
    attention_name="Gemma2Attention",
    mlp_name="Gemma2MLP",
    norm_name="Gemma2RMSNorm",
    rotary_name="Gemma2RotaryEmbedding",
    modeling_module=_GEMMA2_MODELING_MODULE,
    has_query_key_norms=False,
)
_GEMMA3 = _GemmaSpec(
    family="gemma3",
    model_type="gemma3_text",
    config_name="Gemma3TextConfig",
    config_module=_GEMMA3_CONFIG_MODULE,
    wrapper_name="Gemma3ForCausalLM",
    model_name="Gemma3TextModel",
    embedding_name="Gemma3TextScaledWordEmbedding",
    decoder_name="Gemma3DecoderLayer",
    attention_name="Gemma3Attention",
    mlp_name="Gemma3MLP",
    norm_name="Gemma3RMSNorm",
    rotary_name="Gemma3RotaryEmbedding",
    modeling_module=_GEMMA3_MODELING_MODULE,
    has_query_key_norms=True,
)


def _spec_for_model(model: Any) -> _GemmaSpec:
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type == _GEMMA2.model_type:
        return _GEMMA2
    if model_type == _GEMMA3.model_type:
        return _GEMMA3
    raise _unsupported("gemma", f"unexpected model_type {model_type!r}.")


def _optional_positive_float(value: Any, name: str, family: str) -> float | None:
    if value is None:
        return None
    result = _finite_float(value, name, family)
    if result <= 0.0:
        raise _unsupported(family, f"{name} must be positive when configured.")
    return result


def _exact_float(actual: Any, expected: float, label: str, family: str) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _unsupported(family, f"{label} must be a finite number.") from exc
    if not math.isfinite(value) or not math.isclose(
        value, expected, rel_tol=0.0, abs_tol=0.0
    ):
        raise _unsupported(
            family,
            f"{label} must equal {expected}, got {actual!r}.",
        )


def _validate_embedding(
    model: Any,
    spec: _GemmaSpec,
    *,
    vocab_size: int,
    hidden_size: int,
) -> tuple[float, str]:
    family = spec.family
    path = "model.embed_tokens"
    embedding = _require_exact_type(
        resolve_module(model, path, family=family),
        expected_name=spec.embedding_name,
        expected_module=spec.modeling_module,
        label=f"operator {path!r}",
        family=family,
    )
    _require_weight_shape(embedding, path, (vocab_size, hidden_size), family)
    parameters = {name for name, _ in embedding.named_parameters(recurse=False)}
    buffers = {name for name, _ in embedding.named_buffers(recurse=False)}
    if parameters != {"weight"} or buffers != {"embed_scale"}:
        raise _unsupported(
            family,
            f"operator {path!r} must expose exactly weight and embed_scale state.",
        )
    scale_tensor = getattr(embedding, "embed_scale", None)
    if getattr(scale_tensor, "ndim", None) != 0 or not bool(
        scale_tensor.isfinite().item()
    ):
        raise _unsupported(family, f"operator {path!r} embed_scale must be finite scalar.")
    expected_scale = hidden_size**0.5
    scale = float(scale_tensor.item())
    if not math.isclose(scale, expected_scale, rel_tol=1e-6, abs_tol=1e-7):
        raise _unsupported(
            family,
            f"operator {path!r} embed_scale must equal sqrt(hidden_size).",
        )
    return scale, f"{path}.embed_scale"


def _rope_mapping(config: Any, family: str) -> Mapping[str, Any]:
    value = getattr(config, "rope_parameters", None)
    if value is None:
        value = getattr(config, "rope_scaling", None)
    if not isinstance(value, Mapping):
        raise _unsupported(family, "rope_parameters must be a mapping.")
    return value


def _validate_rope_buffer(
    rotary: Any,
    name: str,
    *,
    head_dim: int,
    family: str,
) -> None:
    value = getattr(rotary, name, None)
    expected_shape = (head_dim // 2,)
    shape = getattr(value, "shape", None)
    if shape is None or tuple(int(size) for size in shape) != expected_shape:
        actual = None if shape is None else tuple(int(size) for size in shape)
        raise _unsupported(
            family,
            f"RoPE buffer {name!r} must have shape {expected_shape}, got {actual}.",
        )
    if not bool(value.isfinite().all().item()):
        raise _unsupported(family, f"RoPE buffer {name!r} contains non-finite values.")


def _build_position_layout(
    model: Any,
    config: Any,
    spec: _GemmaSpec,
    *,
    head_dim: int,
    layer_types: tuple[str, ...],
    max_positions: int,
) -> PositionLayout:
    family = spec.family
    if head_dim % 2:
        raise _unsupported(family, "head_dim must be even for full-head RoPE.")
    path = "model.rotary_emb"
    rotary = _require_exact_type(
        resolve_module(model, path, family=family),
        expected_name=spec.rotary_name,
        expected_module=spec.modeling_module,
        label=f"operator {path!r}",
        family=family,
    )
    if int(getattr(rotary, "max_seq_len_cached", -1)) != max_positions:
        raise _unsupported(
            family,
            f"operator {path!r} cached position limit disagrees with config.",
        )
    parameters = tuple(rotary.named_parameters(recurse=False))
    if parameters:
        raise _unsupported(family, f"operator {path!r} must not have parameters.")

    configured = _rope_mapping(config, family)
    used_layer_types = tuple(dict.fromkeys(layer_types))
    variants: list[tuple[str, float, str]] = []
    expected_buffers: set[str] = set()
    if spec is _GEMMA2:
        rope_type = configured.get("rope_type", "default")
        if rope_type != "default":
            raise _unsupported(family, "only default Gemma2 RoPE is supported.")
        theta = _optional_positive_float(
            configured.get("rope_theta", 10_000.0), "rope_theta", family
        )
        assert theta is not None
        if getattr(rotary, "rope_type", None) != rope_type:
            raise _unsupported(family, "rotary RoPE type disagrees with config.")
        _exact_float(
            getattr(rotary, "attention_scaling", None),
            1.0,
            "rotary attention scaling",
            family,
        )
        expected_buffers.update({"inv_freq", "original_inv_freq"})
        variants.append(("all", theta, rope_type))
    else:
        observed_rope_types = getattr(rotary, "rope_type", None)
        if not isinstance(observed_rope_types, Mapping):
            raise _unsupported(family, "Gemma3 rotary rope_type must be per-layer mapping.")
        if set(observed_rope_types) != set(used_layer_types):
            raise _unsupported(
                family, "Gemma3 rotary layer types disagree with configured layers."
            )
        if set(getattr(rotary, "layer_types", ())) != set(used_layer_types):
            raise _unsupported(
                family, "Gemma3 rotary layer types disagree with configured layers."
            )
        for layer_type in used_layer_types:
            parameters = configured.get(layer_type)
            if not isinstance(parameters, Mapping):
                raise _unsupported(
                    family, f"RoPE parameters for {layer_type!r} must be a mapping."
                )
            rope_type = parameters.get("rope_type", "default")
            if rope_type != "default":
                raise _unsupported(
                    family, "only default Gemma3 RoPE variants are supported."
                )
            theta = _optional_positive_float(
                parameters.get("rope_theta"),
                f"rope_parameters.{layer_type}.rope_theta",
                family,
            )
            if theta is None:
                raise _unsupported(
                    family, f"RoPE theta for {layer_type!r} must be configured."
                )
            if observed_rope_types.get(layer_type) != rope_type:
                raise _unsupported(
                    family, f"rotary RoPE type for {layer_type!r} disagrees with config."
                )
            _exact_float(
                getattr(rotary, f"{layer_type}_attention_scaling", None),
                1.0,
                f"{layer_type} rotary attention scaling",
                family,
            )
            expected_buffers.update(
                {
                    f"{layer_type}_inv_freq",
                    f"{layer_type}_original_inv_freq",
                }
            )
            variants.append((layer_type, theta, rope_type))

    observed_buffers = {name for name, _ in rotary.named_buffers(recurse=False)}
    if observed_buffers != expected_buffers:
        raise _unsupported(
            family,
            "rotary buffers disagree with the configured RoPE variants "
            f"(expected={sorted(expected_buffers)}, got={sorted(observed_buffers)}).",
        )
    for name in sorted(expected_buffers):
        _validate_rope_buffer(rotary, name, head_dim=head_dim, family=family)
    for name in sorted(expected_buffers):
        if "original_inv_freq" not in name:
            continue
        current_name = name.replace("original_inv_freq", "inv_freq")
        current = getattr(rotary, current_name)
        original = getattr(rotary, name)
        if not bool((current == original).all().item()):
            raise _unsupported(
                family, f"RoPE buffers {current_name!r} and {name!r} disagree."
            )

    theta = variants[0][1] if len(variants) == 1 else None
    rope_type = variants[0][2] if all(v[2] == variants[0][2] for v in variants) else None
    return PositionLayout(
        kind="rope",
        operator_paths=(path,),
        max_positions=max_positions,
        rotary_dim=head_dim,
        rope_theta=theta,
        rope_type=rope_type,
        rope_variants=tuple(variants),
    )


def _gemma_norm(
    model: Any,
    spec: _GemmaSpec,
    *,
    path: str,
    placement: str,
    width: int,
    epsilon: float,
):
    return _norm_layout(
        model,
        path=path,
        family=spec.family,
        kind="rms_norm",
        placement=placement,
        hidden_size=width,
        epsilon=epsilon,
        expected_type=spec.norm_name,
        expected_module=spec.modeling_module,
        weight_kind="unit_offset",
    )


def _validate_complete_state_claims(model: Any, layout: DecoderLayout) -> None:
    family = layout.family
    parameter_claims: set[str] = set()
    buffer_claims: set[str] = set()
    for path in layout.required_operator_paths:
        module = resolve_module(model, path, family=family)
        parameter_claims.update(
            f"{path}.{name}" for name, _ in module.named_parameters(recurse=False)
        )
        buffer_claims.update(
            f"{path}.{name}" for name, _ in module.named_buffers(recurse=False)
        )
    actual_parameters = {
        name for name, _ in model.named_parameters(remove_duplicate=False)
    }
    actual_buffers = {name for name, _ in model.named_buffers(remove_duplicate=False)}
    if parameter_claims != actual_parameters:
        raise _unsupported(
            family,
            "explicit operator paths do not claim exactly all parameters "
            f"(missing={sorted(actual_parameters - parameter_claims)}, "
            f"extra={sorted(parameter_claims - actual_parameters)}).",
        )
    if buffer_claims != actual_buffers:
        raise _unsupported(
            family,
            "explicit operator paths do not claim exactly all buffers "
            f"(missing={sorted(actual_buffers - buffer_claims)}, "
            f"extra={sorted(buffer_claims - actual_buffers)}).",
        )


def build_gemma_decoder_layout(model: Any) -> DecoderLayout:
    """Build a fail-closed current-Transformers Gemma text decoder layout."""

    spec = _spec_for_model(model)
    family = spec.family
    _require_exact_type(
        model,
        expected_name=spec.wrapper_name,
        expected_module=spec.modeling_module,
        label="model",
        family=family,
    )
    config = _validate_causal_wrapper(model, family)
    _require_exact_type(
        config,
        expected_name=spec.config_name,
        expected_module=spec.config_module,
        label="configuration",
        family=family,
    )
    _require_exact_type(
        resolve_module(model, "model", family=family),
        expected_name=spec.model_name,
        expected_module=spec.modeling_module,
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
    if key_value_heads > query_heads or query_heads % key_value_heads:
        raise _unsupported(
            family,
            "num_key_value_heads must evenly divide num_attention_heads.",
        )
    head_dim = _positive_int(
        _required_attr(config, "head_dim", family), "head_dim", family
    )
    vocab_size = _positive_int(
        _required_attr(config, "vocab_size", family), "vocab_size", family
    )
    max_positions = _positive_int(
        _required_attr(config, "max_position_embeddings", family),
        "max_position_embeddings",
        family,
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
    attention_bias = _required_attr(config, "attention_bias", family)
    if not isinstance(attention_bias, bool):
        raise _unsupported(family, "attention_bias must be a bool.")
    if getattr(config, "use_bidirectional_attention", False) not in {None, False}:
        raise _unsupported(family, "bidirectional attention is not causal decoding.")
    query_pre_attn_scalar = _optional_positive_float(
        _required_attr(config, "query_pre_attn_scalar", family),
        "query_pre_attn_scalar",
        family,
    )
    assert query_pre_attn_scalar is not None
    attention_scaling = query_pre_attn_scalar**-0.5
    attention_softcap = _optional_positive_float(
        getattr(config, "attn_logit_softcapping", None),
        "attn_logit_softcapping",
        family,
    )
    final_softcap = _optional_positive_float(
        getattr(config, "final_logit_softcapping", None),
        "final_logit_softcapping",
        family,
    )

    configured_layer_types = _required_attr(config, "layer_types", family)
    if not isinstance(configured_layer_types, (list, tuple)):
        raise _unsupported(family, "layer_types must be a list or tuple.")
    layer_types = tuple(configured_layer_types)
    if len(layer_types) != layer_count or any(
        layer_type not in _LAYER_TYPES for layer_type in layer_types
    ):
        raise _unsupported(
            family,
            "layer_types must name full_attention or sliding_attention for every layer.",
        )
    sliding_window_value = getattr(config, "sliding_window", None)
    sliding_window = None
    if "sliding_attention" in layer_types:
        sliding_window = _positive_int(
            sliding_window_value, "sliding_window", family
        )

    token_scale, token_scale_buffer_name = _validate_embedding(
        model,
        spec,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
    )
    blocks_module = _require_module_type(model, "model.layers", "ModuleList", family)
    if len(blocks_module) != layer_count:
        raise _unsupported(
            family,
            f"model.layers must contain {layer_count} blocks, got {len(blocks_module)}.",
        )
    position = _build_position_layout(
        model,
        config,
        spec,
        head_dim=head_dim,
        layer_types=layer_types,
        max_positions=max_positions,
    )

    query_width = query_heads * head_dim
    key_value_width = key_value_heads * head_dim
    activation_name = _required_attr(config, "hidden_activation", family)
    blocks: list[DecoderBlockLayout] = []
    for index, layer_type in enumerate(layer_types):
        prefix = f"model.layers.{index}"
        _require_exact_type(
            resolve_module(model, prefix, family=family),
            expected_name=spec.decoder_name,
            expected_module=spec.modeling_module,
            label=f"operator {prefix!r}",
            family=family,
        )
        attention_path = f"{prefix}.self_attn"
        attention_module = _require_exact_type(
            resolve_module(model, attention_path, family=family),
            expected_name=spec.attention_name,
            expected_module=spec.modeling_module,
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
            _require_bias(projection, path, attention_bias, rows, family)
        if int(getattr(attention_module, "head_dim", -1)) != head_dim:
            raise _unsupported(
                family, f"operator {attention_path!r} head dimension disagrees."
            )
        expected_groups = query_heads // key_value_heads
        if int(getattr(attention_module, "num_key_value_groups", -1)) != expected_groups:
            raise _unsupported(family, f"operator {attention_path!r} GQA grouping disagrees.")
        _exact_float(
            getattr(attention_module, "scaling", None),
            attention_scaling,
            f"operator {attention_path!r} scaling",
            family,
        )
        _exact_float(
            getattr(attention_module, "attention_dropout", None),
            attention_dropout,
            f"operator {attention_path!r} dropout",
            family,
        )
        if getattr(attention_module, "is_causal", None) is not True:
            raise _unsupported(family, f"operator {attention_path!r} must be causal.")
        if int(getattr(attention_module, "layer_idx", -1)) != index:
            raise _unsupported(family, f"operator {attention_path!r} layer index disagrees.")
        if getattr(attention_module, "layer_type", None) != layer_type:
            raise _unsupported(family, f"operator {attention_path!r} layer type disagrees.")
        expected_window = sliding_window if layer_type == "sliding_attention" else None
        if getattr(attention_module, "sliding_window", None) != expected_window:
            raise _unsupported(
                family, f"operator {attention_path!r} sliding-window policy disagrees."
            )
        if getattr(attention_module, "attn_logit_softcapping", None) != attention_softcap:
            raise _unsupported(
                family, f"operator {attention_path!r} logit softcap disagrees."
            )

        query_norm = None
        key_norm = None
        if spec.has_query_key_norms:
            query_norm = _gemma_norm(
                model,
                spec,
                path=f"{attention_path}.q_norm",
                placement="query_head",
                width=head_dim,
                epsilon=epsilon,
            )
            key_norm = _gemma_norm(
                model,
                spec,
                path=f"{attention_path}.k_norm",
                placement="key_head",
                width=head_dim,
                epsilon=epsilon,
            )
        elif hasattr(attention_module, "q_norm") or hasattr(attention_module, "k_norm"):
            raise _unsupported(family, "Gemma2 attention unexpectedly exposes Q/K norms.")

        mlp_path = f"{prefix}.mlp"
        _require_exact_type(
            resolve_module(model, mlp_path, family=family),
            expected_name=spec.mlp_name,
            expected_module=spec.modeling_module,
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
            configured_name=activation_name,
            family=family,
        )

        attention_input_norm = _gemma_norm(
            model,
            spec,
            path=f"{prefix}.input_layernorm",
            placement="pre_attention",
            width=hidden_size,
            epsilon=epsilon,
        )
        attention_output_norm = _gemma_norm(
            model,
            spec,
            path=f"{prefix}.post_attention_layernorm",
            placement="post_attention",
            width=hidden_size,
            epsilon=epsilon,
        )
        mlp_input_norm = _gemma_norm(
            model,
            spec,
            path=f"{prefix}.pre_feedforward_layernorm",
            placement="pre_mlp",
            width=hidden_size,
            epsilon=epsilon,
        )
        mlp_output_norm = _gemma_norm(
            model,
            spec,
            path=f"{prefix}.post_feedforward_layernorm",
            placement="post_mlp",
            width=hidden_size,
            epsilon=epsilon,
        )
        blocks.append(
            DecoderBlockLayout(
                index=index,
                path=prefix,
                attention_input_norm=attention_input_norm,
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
                    query_norm=query_norm,
                    key_norm=key_norm,
                    layer_type=layer_type,
                    sliding_window=expected_window,
                    scaling=attention_scaling,
                    logit_softcap=attention_softcap,
                ),
                mlp_input_norm=mlp_input_norm,
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
                residual=SERIAL_PRE_POST_NORM_RESIDUAL,
                attention_output_norm=attention_output_norm,
                mlp_output_norm=mlp_output_norm,
            )
        )

    final_norm = _gemma_norm(
        model,
        spec,
        path="model.norm",
        placement="final",
        width=hidden_size,
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
    layout = DecoderLayout(
        family=family,
        model_type=spec.model_type,
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
            token_scale=token_scale,
            token_scale_buffer_name=token_scale_buffer_name,
        ),
        position=position,
        blocks=tuple(blocks),
        final_norm=final_norm,
        lm_head_path="lm_head",
        tie_word_embeddings=tied,
        final_logit_softcap=final_softcap,
    )
    _validate_complete_state_claims(model, layout)
    return layout


__all__ = ["build_gemma_decoder_layout"]
