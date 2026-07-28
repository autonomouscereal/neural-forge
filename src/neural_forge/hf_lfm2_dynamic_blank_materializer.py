"""Direct-local physical materialization of nonuniform dense LFM2 children.

The layout descriptor is deliberately separate from this runtime adapter.  A
normal locally constructed :class:`transformers.Lfm2ForCausalLM` starts with
uniform feed-forward widths and short-convolution channels equal to the
residual width.  This module replaces only those physical modules before any
folded state is loaded.  The resulting model retains the native LFM2 state-key
namespace while owning independent, dense FP32 storage for every parameter.

No checkpoint or Hub loader is used here.  The sealed descriptor is embedded
in ``config.json``-serializable metadata so a direct-local loader can repeat
the materialization before performing one strict state-dict load.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from .hf_lfm2_dynamic_physical_layout import (
    Lfm2DynamicPhysicalLayoutDescriptor,
    Lfm2DynamicPhysicalLayoutError,
    authenticate_lfm2_dynamic_physical_layout,
    authenticate_lfm2_physical_materialization,
    lfm2_expected_physical_state_shapes,
)


LFM2_DYNAMIC_BLANK_MATERIALIZATION_SCHEMA = (
    "neural-forge-lfm2-dynamic-blank-materialization-v2"
)
LFM2_DYNAMIC_BLANK_CONFIG_SCHEMA = "neural-forge-lfm2-dynamic-blank-config-binding-v2"

_CONFIG_SCHEMA_FIELD = "neural_forge_dynamic_blank_config_schema"
_CONFIG_LAYOUT_FIELD = "neural_forge_dynamic_physical_layout"
_CONFIG_LAYOUT_SHA_FIELD = "neural_forge_dynamic_physical_layout_sha256"
_MODEL_RUNTIME_SHA_FIELD = "_neural_forge_dynamic_blank_runtime_sha256"
_MODEL_CACHE_OWNER_FIELD = "_neural_forge_dynamic_blank_cache_owner"
_MISSING = object()
_DYNAMIC_SHORT_CONV_CLASS: type[Any] | None = None
_DYNAMIC_MLP_CLASS: type[Any] | None = None
_DYNAMIC_ATTENTION_CLASS: type[Any] | None = None
_CACHE_OWNER_FIELD = "_neural_forge_dynamic_blank_model_owner"


class Lfm2DynamicBlankMaterializationError(Lfm2DynamicPhysicalLayoutError):
    """A local target cannot be authenticated as the requested blank child."""


def _dynamic_mlp_class() -> type[Any]:
    """Return a behavior-identical native subclass marking physical replacement."""

    global _DYNAMIC_MLP_CLASS
    if _DYNAMIC_MLP_CLASS is not None:
        return _DYNAMIC_MLP_CLASS
    Lfm2MLP = _dependencies()[8]

    class Lfm2DynamicWidthMLP(Lfm2MLP):
        """Native LFM2 SwiGLU with a descriptor-selected physical width."""

    Lfm2DynamicWidthMLP.__name__ = "Lfm2DynamicWidthMLP"
    Lfm2DynamicWidthMLP.__qualname__ = "Lfm2DynamicWidthMLP"
    _DYNAMIC_MLP_CLASS = Lfm2DynamicWidthMLP
    return Lfm2DynamicWidthMLP


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
        raise Lfm2DynamicBlankMaterializationError(
            "Dynamic blank-child metadata must be finite canonical JSON."
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _dependencies() -> tuple[Any, ...]:
    try:
        import torch
        import transformers
        from transformers import Lfm2Config, Lfm2ForCausalLM
        from transformers.cache_utils import (
            DynamicCache,
            LinearAttentionCacheLayerMixin,
        )
        from transformers.models.lfm2.modeling_lfm2 import (
            Lfm2Attention,
            Lfm2DecoderLayer,
            Lfm2MLP,
            Lfm2Model,
            Lfm2RMSNorm,
            Lfm2RotaryEmbedding,
            Lfm2ShortConv,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise Lfm2DynamicBlankMaterializationError(
            "Concrete Transformers LFM2 and PyTorch are required for physical "
            "blank-child materialization."
        ) from exc
    return (
        torch,
        transformers,
        Lfm2Config,
        Lfm2ForCausalLM,
        DynamicCache,
        LinearAttentionCacheLayerMixin,
        Lfm2Attention,
        Lfm2DecoderLayer,
        Lfm2MLP,
        Lfm2Model,
        Lfm2RMSNorm,
        Lfm2RotaryEmbedding,
        Lfm2ShortConv,
    )


def _authenticate_cache_owner(
    cache: Any,
    owner_token: object,
    *,
    torch: Any,
    dynamic_cache_class: type[Any],
) -> None:
    if cache is None:
        return
    if type(cache) is not dynamic_cache_class:
        raise Lfm2DynamicBlankMaterializationError(
            "Dynamic LFM2 operators accept only Transformers DynamicCache."
        )
    for layer_index, layer in enumerate(cache.layers):
        layer_tensors: dict[str, Any] = {}
        for name in (
            "keys",
            "values",
            "key_cache",
            "value_cache",
            "conv_states",
            "recurrent_states",
        ):
            state = getattr(layer, name, None)
            if state is None:
                continue
            if (
                type(state) is not torch.Tensor
                or state.device.type == "meta"
                or state.layout != torch.strided
                or bool(getattr(state, "is_quantized", False))
                or int(state.storage_offset()) != 0
                or state.requires_grad
                or int(state.untyped_storage().nbytes())
                != int(state.numel() * state.element_size())
            ):
                raise Lfm2DynamicBlankMaterializationError(
                    "LFM2 DynamicCache physical cache contract contains a "
                    "noncanonical tensor at "
                    f"layer {layer_index} field {name!r}."
                )
            declared_device = getattr(layer, "device", None)
            declared_dtype = getattr(layer, "dtype", None)
            if (
                declared_device is not None
                and state.device != torch.device(declared_device)
            ) or (declared_dtype is not None and state.dtype != declared_dtype):
                raise Lfm2DynamicBlankMaterializationError(
                    "LFM2 DynamicCache tensor disagrees with its declared "
                    f"device or dtype at layer {layer_index} field {name!r}."
                )
            layer_tensors[name] = state
        if ("keys" in layer_tensors) is not ("values" in layer_tensors):
            raise Lfm2DynamicBlankMaterializationError(
                f"LFM2 DynamicCache attention pair is incomplete at layer {layer_index}."
            )
        if "keys" in layer_tensors and (
            layer_tensors["keys"].shape != layer_tensors["values"].shape
            or layer_tensors["keys"].device != layer_tensors["values"].device
            or layer_tensors["keys"].dtype != layer_tensors["values"].dtype
        ):
            raise Lfm2DynamicBlankMaterializationError(
                f"LFM2 DynamicCache attention pair drifted at layer {layer_index}."
            )
    observed_owner = getattr(cache, _CACHE_OWNER_FIELD, _MISSING)
    if observed_owner is _MISSING:
        initialized = False
        for layer in cache.layers:
            initialized = initialized or any(
                bool(getattr(layer, name, False))
                for name in (
                    "is_initialized",
                    "is_conv_states_initialized",
                    "is_recurrent_states_initialized",
                )
            )
            initialized = initialized or any(
                type(getattr(layer, name, None)) is torch.Tensor
                and int(getattr(layer, name).numel()) > 0
                for name in (
                    "keys",
                    "values",
                    "key_cache",
                    "value_cache",
                    "conv_states",
                    "recurrent_states",
                )
            )
        if initialized:
            raise Lfm2DynamicBlankMaterializationError(
                "LFM2 DynamicCache was not freshly created for this physical "
                "materialization cache contract."
            )
        try:
            setattr(cache, _CACHE_OWNER_FIELD, owner_token)
        except Exception as exc:
            raise Lfm2DynamicBlankMaterializationError(
                "LFM2 DynamicCache cannot authenticate materialization ownership."
            ) from exc
    elif observed_owner is not owner_token:
        raise Lfm2DynamicBlankMaterializationError(
            "LFM2 DynamicCache belongs to a different physical materialization."
        )


def _dynamic_short_conv_class() -> type[Any]:
    """Build the native-derived variable-channel ShortConv runtime lazily."""

    global _DYNAMIC_SHORT_CONV_CLASS
    if _DYNAMIC_SHORT_CONV_CLASS is not None:
        return _DYNAMIC_SHORT_CONV_CLASS
    (
        torch,
        _,
        _,
        _,
        DynamicCache,
        LinearAttentionCacheLayerMixin,
        _,
        _,
        _,
        _,
        _,
        _,
        Lfm2ShortConv,
    ) = _dependencies()

    class Lfm2DynamicWidthShortConv(Lfm2ShortConv):
        """Native LFM2 ShortConv with independent physical channel width."""

        def __init__(
            self,
            config: Any,
            layer_idx: int,
            *,
            internal_channel_width: int,
            expected_layer_count: int,
            cache_owner_token: object,
        ) -> None:
            super().__init__(config, layer_idx)
            hidden = int(config.hidden_size)
            internal = int(internal_channel_width)
            kernel = int(config.conv_L_cache)
            self.external_hidden_size = hidden
            self.internal_channel_width = internal
            self.expected_layer_count = int(expected_layer_count)
            self.cache_owner_token = cache_owner_token
            self.layer_type = str(config.layer_types[layer_idx])
            self.in_proj = torch.nn.Linear(
                hidden,
                3 * internal,
                bias=False,
                dtype=torch.float32,
            )
            self.conv = torch.nn.Conv1d(
                in_channels=internal,
                out_channels=internal,
                kernel_size=kernel,
                groups=internal,
                bias=False,
                padding=kernel - 1,
                dtype=torch.float32,
            )
            self.out_proj = torch.nn.Linear(
                internal,
                hidden,
                bias=False,
                dtype=torch.float32,
            )

        def _authenticate_cache(self, hidden_states: Any, cache: Any) -> None:
            if cache is None:
                return
            _authenticate_cache_owner(
                cache,
                self.cache_owner_token,
                torch=torch,
                dynamic_cache_class=DynamicCache,
            )
            if len(cache.layers) != self.expected_layer_count:
                raise Lfm2DynamicBlankMaterializationError(
                    "LFM2 DynamicCache layer count disagrees with the physical layout."
                )
            cache_layer = cache.layers[self.layer_idx]
            if not isinstance(cache_layer, LinearAttentionCacheLayerMixin):
                raise Lfm2DynamicBlankMaterializationError(
                    "LFM2 DynamicCache layer kind disagrees with ShortConv."
                )
            initialized = bool(
                getattr(cache_layer, "is_conv_states_initialized", False)
            )
            if not initialized:
                return
            state = getattr(cache_layer, "conv_states", None)
            kernel_width = getattr(
                self,
                "L_cache",
                getattr(self, "conv_kernel_size", None),
            )
            expected = (
                int(hidden_states.shape[0]),
                self.internal_channel_width,
                kernel_width,
            )
            if (
                type(state) is not torch.Tensor
                or tuple(state.shape) != expected
                or state.device != hidden_states.device
                or state.dtype != hidden_states.dtype
            ):
                raise Lfm2DynamicBlankMaterializationError(
                    "LFM2 DynamicCache ShortConv state disagrees with the "
                    f"physical cache contract; expected {expected}."
                )

        def forward(
            self,
            hidden_states: Any,
            past_key_values: Any = None,
            attention_mask: Any = None,
            seq_idx: Any = None,
        ) -> Any:
            if (
                not isinstance(hidden_states, torch.Tensor)
                or hidden_states.ndim != 3
                or int(hidden_states.shape[-1]) != self.external_hidden_size
            ):
                raise Lfm2DynamicBlankMaterializationError(
                    "Dynamic-width LFM2 ShortConv input disagrees with its "
                    "external residual width."
                )
            self._authenticate_cache(hidden_states, past_key_values)
            return super().forward(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                seq_idx=seq_idx,
            )

    Lfm2DynamicWidthShortConv.__name__ = "Lfm2DynamicWidthShortConv"
    Lfm2DynamicWidthShortConv.__qualname__ = "Lfm2DynamicWidthShortConv"
    _DYNAMIC_SHORT_CONV_CLASS = Lfm2DynamicWidthShortConv
    return Lfm2DynamicWidthShortConv


def _dynamic_attention_class() -> type[Any]:
    """Build native attention with pre-mutation cache ownership checks."""

    global _DYNAMIC_ATTENTION_CLASS
    if _DYNAMIC_ATTENTION_CLASS is not None:
        return _DYNAMIC_ATTENTION_CLASS
    torch, _, _, _, DynamicCache, _, Lfm2Attention, *_ = _dependencies()

    class Lfm2DynamicAttention(Lfm2Attention):
        """Native LFM2 attention owned by one physical materialization."""

        def __init__(
            self,
            config: Any,
            layer_idx: int,
            *,
            cache_owner_token: object,
        ) -> None:
            super().__init__(config, layer_idx)
            self.cache_owner_token = cache_owner_token

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            cache = kwargs.get("past_key_values")
            if cache is None and len(args) >= 4:
                cache = args[3]
            _authenticate_cache_owner(
                cache,
                self.cache_owner_token,
                torch=torch,
                dynamic_cache_class=DynamicCache,
            )
            return super().forward(*args, **kwargs)

    Lfm2DynamicAttention.__name__ = "Lfm2DynamicAttention"
    Lfm2DynamicAttention.__qualname__ = "Lfm2DynamicAttention"
    _DYNAMIC_ATTENTION_CLASS = Lfm2DynamicAttention
    return Lfm2DynamicAttention


def _qualified_class(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _config_value(config: Any, name: str) -> Any:
    value = getattr(config, name, _MISSING)
    if value is _MISSING:
        raise Lfm2DynamicBlankMaterializationError(
            f"Concrete LFM2 config is missing {name!r}."
        )
    return value


def _same_semantic_scalar(actual: Any, expected: Any) -> bool:
    if type(expected) is bool:
        return type(actual) is bool and actual is expected
    if isinstance(expected, int):
        return type(actual) is int and actual == expected
    if isinstance(expected, float):
        return type(actual) in {int, float} and float(actual) == expected
    return type(actual) is type(expected) and actual == expected


def _config_snapshot(config: Any) -> Mapping[str, Any]:
    to_dict = getattr(config, "to_dict", None)
    if not callable(to_dict):
        raise Lfm2DynamicBlankMaterializationError(
            "Concrete LFM2 config must expose to_dict()."
        )
    try:
        raw = to_dict()
    except Exception as exc:
        raise Lfm2DynamicBlankMaterializationError(
            "Concrete LFM2 config cannot be serialized safely."
        ) from exc
    if not isinstance(raw, Mapping):
        raise Lfm2DynamicBlankMaterializationError(
            "Concrete LFM2 config serialization must be one mapping."
        )
    snapshot = dict(raw)
    try:
        baseline_raw = type(config)().to_dict()
    except Exception as exc:
        raise Lfm2DynamicBlankMaterializationError(
            "Could not derive the installed LFM2 config namespace."
        ) from exc
    if not isinstance(baseline_raw, Mapping):
        raise Lfm2DynamicBlankMaterializationError(
            "Installed LFM2 config namespace is malformed."
        )
    binding_fields = {
        _CONFIG_SCHEMA_FIELD,
        _CONFIG_LAYOUT_FIELD,
        _CONFIG_LAYOUT_SHA_FIELD,
    }
    present_bindings = binding_fields.intersection(snapshot)
    if present_bindings not in (set(), binding_fields):
        raise Lfm2DynamicBlankMaterializationError(
            "Concrete LFM2 config has incomplete dynamic-layout metadata."
        )
    required_fields = set(baseline_raw) | {"head_dim"} | present_bindings
    optional_aliases = {"block_ff_dim", "torch_dtype"}
    if not required_fields.issubset(snapshot) or not (
        set(snapshot) - required_fields
    ).issubset(optional_aliases):
        raise Lfm2DynamicBlankMaterializationError(
            "Concrete LFM2 config namespace contains missing or unsealed fields."
        )
    fixed_values = {
        "model_type": "lfm2",
        "is_encoder_decoder": False,
        "chunk_size_feed_forward": 0,
        "output_attentions": False,
        "output_hidden_states": False,
        "return_dict": True,
        "full_attn_idxs": None,
        "problem_type": None,
        "id2label": {0: "LABEL_0", 1: "LABEL_1"},
        "label2id": {"LABEL_0": 0, "LABEL_1": 1},
    }
    if any(snapshot.get(name) != expected for name, expected in fixed_values.items()):
        raise Lfm2DynamicBlankMaterializationError(
            "Concrete LFM2 config metadata is not the sealed causal runtime profile."
        )
    for name in ("intermediate_size", "block_multiple_of"):
        if type(snapshot.get(name)) is not int or snapshot[name] <= 0:
            raise Lfm2DynamicBlankMaterializationError(
                f"Concrete LFM2 config {name} must be one positive integer."
            )
    for name in ("initializer_range", "block_ffn_dim_multiplier"):
        value = snapshot.get(name)
        if (
            type(value) not in {int, float}
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise Lfm2DynamicBlankMaterializationError(
                f"Concrete LFM2 config {name} must be finite and positive."
            )
    block_alias = snapshot.pop("block_ff_dim", None)
    if block_alias is not None and (
        type(block_alias) is not int or block_alias != snapshot["intermediate_size"]
    ):
        raise Lfm2DynamicBlankMaterializationError(
            "Concrete LFM2 block_ff_dim alias conflicts with intermediate_size."
        )
    if snapshot.get("architectures") not in (None, ["Lfm2ForCausalLM"]):
        raise Lfm2DynamicBlankMaterializationError(
            "Concrete LFM2 config architectures field is not canonical."
        )
    for dtype_field in ("dtype", "torch_dtype"):
        dtype_value = snapshot.pop(dtype_field, None)
        if dtype_value not in (None, "float32"):
            raise Lfm2DynamicBlankMaterializationError(
                "Concrete LFM2 config dtype must be absent or float32."
            )
    snapshot["architectures"] = ["Lfm2ForCausalLM"]
    snapshot.pop("_name_or_path", None)
    snapshot.pop("transformers_version", None)
    _canonical_json(snapshot)
    return MappingProxyType(snapshot)


def _runtime_semantic_binding(
    config: Any,
    layout: Lfm2DynamicPhysicalLayoutDescriptor,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "norm_eps": float(_config_value(config, "norm_eps")),
            "max_position_embeddings": int(
                _config_value(config, "max_position_embeddings")
            ),
            "use_cache": _config_value(config, "use_cache"),
            "attention_implementation": _config_value(
                config,
                "_attn_implementation",
            ),
            "bos_token_id": _config_value(config, "bos_token_id"),
            "eos_token_id": _config_value(config, "eos_token_id"),
            "pad_token_id": _config_value(config, "pad_token_id"),
            "block_auto_adjust_ff_dim": _config_value(
                config,
                "block_auto_adjust_ff_dim",
            ),
            "descriptor_sha256": layout.descriptor_sha256,
        }
    )


def _generation_config_snapshot(
    model: Any,
    config: Any,
    transformers_module: Any,
) -> Mapping[str, Any]:
    generation_class = transformers_module.GenerationConfig
    generation_config = getattr(model, "generation_config", _MISSING)
    if type(generation_config) is not generation_class:
        raise Lfm2DynamicBlankMaterializationError(
            "Materialized LFM2 generation config has the wrong exact type."
        )
    try:
        actual = generation_config.to_dict()
        expected = generation_class.from_model_config(config).to_dict()
    except Exception as exc:
        raise Lfm2DynamicBlankMaterializationError(
            "Materialized LFM2 generation config is not serializable."
        ) from exc
    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        raise Lfm2DynamicBlankMaterializationError(
            "Materialized LFM2 generation config is malformed."
        )
    actual_snapshot = dict(actual)
    expected_snapshot = dict(expected)
    _canonical_json(actual_snapshot)
    _canonical_json(expected_snapshot)
    if actual_snapshot != expected_snapshot:
        raise Lfm2DynamicBlankMaterializationError(
            "Materialized LFM2 generation config drifted from model semantics."
        )
    return MappingProxyType(actual_snapshot)


def _validate_config(
    config: Any,
    layout: Lfm2DynamicPhysicalLayoutDescriptor,
    lfm2_config_class: type[Any],
) -> None:
    if type(config) is not lfm2_config_class or config.model_type != "lfm2":
        raise Lfm2DynamicBlankMaterializationError(
            "Target config must be one concrete local Lfm2Config."
        )
    expected_scalars = {
        "hidden_size": layout.hidden_size,
        "vocab_size": layout.vocab_size,
        "head_dim": layout.head_dim,
        "num_attention_heads": layout.num_attention_heads,
        "num_key_value_heads": layout.num_key_value_heads,
        "num_hidden_layers": layout.layer_count,
        "conv_L_cache": layout.conv_kernel_size,
        "norm_eps": layout.norm_eps,
        "max_position_embeddings": layout.max_position_embeddings,
        "use_cache": layout.use_cache,
        "_attn_implementation": layout.attention_implementation,
        "bos_token_id": layout.bos_token_id,
        "eos_token_id": layout.eos_token_id,
        "pad_token_id": layout.pad_token_id,
        "block_auto_adjust_ff_dim": layout.block_auto_adjust_ff_dim,
    }
    for name, expected in expected_scalars.items():
        if not _same_semantic_scalar(_config_value(config, name), expected):
            raise Lfm2DynamicBlankMaterializationError(
                f"Concrete LFM2 config {name} disagrees with the descriptor."
            )
    if tuple(_config_value(config, "layer_types")) != tuple(
        layer.layer_type for layer in layout.layers
    ):
        raise Lfm2DynamicBlankMaterializationError(
            "Concrete LFM2 layer_types disagree with the descriptor."
        )
    if bool(_config_value(config, "conv_bias")):
        raise Lfm2DynamicBlankMaterializationError(
            "Dynamic dense LFM2 materialization requires bias-free ShortConv."
        )
    if bool(_config_value(config, "tie_word_embeddings")):
        raise Lfm2DynamicBlankMaterializationError(
            "Dynamic dense LFM2 materialization requires untied token and "
            "readout parameters."
        )
    if _config_value(config, "_name_or_path") not in {"", None}:
        raise Lfm2DynamicBlankMaterializationError(
            "Materialization requires a caller-supplied local blank config with "
            "an empty _name_or_path; origin remains a caller precondition."
        )
    rope = _config_value(config, "rope_parameters")
    try:
        rope_theta = (
            float(rope.get("rope_theta")) if isinstance(rope, Mapping) else None
        )
    except (TypeError, ValueError):
        rope_theta = None
    if (
        not isinstance(rope, Mapping)
        or set(rope) != {"rope_type", "rope_theta"}
        or rope.get("rope_type") != "default"
        or rope_theta != layout.rope_theta
    ):
        raise Lfm2DynamicBlankMaterializationError(
            "Concrete LFM2 config disagrees with the sealed default-RoPE contract."
        )
    _config_snapshot(config)


def _saved_config_layout(config: Any) -> Any:
    return getattr(config, _CONFIG_LAYOUT_FIELD, _MISSING)


def extract_lfm2_dynamic_physical_layout_from_config(
    config: Any,
) -> Lfm2DynamicPhysicalLayoutDescriptor:
    """Authenticate the physical-layout metadata recovered from local JSON."""

    if getattr(config, _CONFIG_SCHEMA_FIELD, _MISSING) != (
        LFM2_DYNAMIC_BLANK_CONFIG_SCHEMA
    ):
        raise Lfm2DynamicBlankMaterializationError(
            "Local LFM2 config has no authenticated dynamic-layout binding."
        )
    raw = _saved_config_layout(config)
    expected_sha = getattr(config, _CONFIG_LAYOUT_SHA_FIELD, _MISSING)
    if raw is _MISSING or expected_sha is _MISSING:
        raise Lfm2DynamicBlankMaterializationError(
            "Local LFM2 config has incomplete dynamic-layout metadata."
        )
    return authenticate_lfm2_dynamic_physical_layout(
        raw,
        expected_descriptor_sha256=expected_sha,
    )


def _validate_pristine_model(
    model: Any,
    config: Any,
    layout: Lfm2DynamicPhysicalLayoutDescriptor,
    dependencies: tuple[Any, ...],
) -> Any:
    (
        torch,
        _,
        _,
        Lfm2ForCausalLM,
        _,
        _,
        Lfm2Attention,
        Lfm2DecoderLayer,
        Lfm2MLP,
        _,
        _,
        _,
        Lfm2ShortConv,
    ) = dependencies
    if type(model) is not Lfm2ForCausalLM:
        raise Lfm2DynamicBlankMaterializationError(
            "Target must be one concrete locally instantiated Lfm2ForCausalLM."
        )
    if model.config is not config or model.model.config is not config:
        raise Lfm2DynamicBlankMaterializationError(
            "Target model and nested LFM2 core must share the supplied config."
        )
    if len(model.model.layers) != layout.layer_count:
        raise Lfm2DynamicBlankMaterializationError(
            "Target model layer count disagrees with the physical layout."
        )
    if model.model.embed_tokens.weight is model.lm_head.weight:
        raise Lfm2DynamicBlankMaterializationError(
            "Target token embedding and readout are tied."
        )
    expected_names = set(lfm2_expected_physical_state_shapes(layout))
    if set(model.state_dict()) != expected_names:
        raise Lfm2DynamicBlankMaterializationError(
            "Pristine LFM2 target has missing, extra, or sidecar state keys."
        )
    devices: set[Any] = set()
    object_ids: set[int] = set()
    storage_ids: set[tuple[str, int]] = set()
    try:
        named = tuple(model.named_parameters(remove_duplicate=False))
    except TypeError:  # pragma: no cover - old optional PyTorch
        named = tuple(model.named_parameters())
    if {name for name, _ in named} != expected_names:
        raise Lfm2DynamicBlankMaterializationError(
            "Pristine LFM2 target parameter namespace is not canonical."
        )
    for name, parameter in named:
        if (
            type(parameter) is not torch.nn.Parameter
            or parameter.device.type == "meta"
            or parameter.dtype != torch.float32
            or parameter.layout != torch.strided
            or not parameter.is_contiguous()
            or parameter.storage_offset() != 0
            or parameter.untyped_storage().nbytes()
            != parameter.numel() * parameter.element_size()
        ):
            raise Lfm2DynamicBlankMaterializationError(
                f"Pristine LFM2 parameter {name} is not exact dense FP32 storage."
            )
        identity = id(parameter)
        storage = (str(parameter.device), int(parameter.untyped_storage().data_ptr()))
        if identity in object_ids or storage in storage_ids:
            raise Lfm2DynamicBlankMaterializationError(
                "Pristine LFM2 target contains parameter aliases or sharing."
            )
        object_ids.add(identity)
        storage_ids.add(storage)
        devices.add(parameter.device)
    if len(devices) != 1:
        raise Lfm2DynamicBlankMaterializationError(
            "Physical blank-child materialization requires one concrete device."
        )
    for physical, layer in zip(layout.layers, model.model.layers, strict=True):
        if type(layer) is not Lfm2DecoderLayer:
            raise Lfm2DynamicBlankMaterializationError(
                f"LFM2 layer {physical.layer_index} is not a native decoder layer."
            )
        if type(layer.feed_forward) is not Lfm2MLP:
            raise Lfm2DynamicBlankMaterializationError(
                f"LFM2 layer {physical.layer_index} MLP is not pristine/native."
            )
        if physical.layer_type == "conv":
            if type(layer.conv) is not Lfm2ShortConv or layer.is_attention_layer:
                raise Lfm2DynamicBlankMaterializationError(
                    f"LFM2 layer {physical.layer_index} ShortConv is malformed."
                )
        elif type(layer.self_attn) is not Lfm2Attention or not layer.is_attention_layer:
            raise Lfm2DynamicBlankMaterializationError(
                f"LFM2 layer {physical.layer_index} attention is malformed."
            )
    return next(iter(devices))


def _validate_materialized_runtime(
    model: Any,
    layout: Lfm2DynamicPhysicalLayoutDescriptor,
    dependencies: tuple[Any, ...],
) -> None:
    (
        torch,
        _,
        _,
        _,
        _,
        _,
        Lfm2Attention,
        Lfm2DecoderLayer,
        _,
        _,
        Lfm2RMSNorm,
        Lfm2RotaryEmbedding,
        _,
    ) = dependencies
    config = model.config
    dynamic_conv_class = _dynamic_short_conv_class()
    dynamic_mlp_class = _dynamic_mlp_class()
    dynamic_attention_class = _dynamic_attention_class()
    cache_owner_token = getattr(model, _MODEL_CACHE_OWNER_FIELD, _MISSING)
    _generation_config_snapshot(model, config, dependencies[1])
    if (
        model.model.config is not config
        or getattr(model, _MODEL_RUNTIME_SHA_FIELD, _MISSING)
        != layout.descriptor_sha256
        or type(cache_owner_token) is not object
        or type(model.model.rotary_emb) is not Lfm2RotaryEmbedding
        or model.model.rotary_emb.config is not config
        or model.vocab_size != layout.vocab_size
        or model.model.vocab_size != layout.vocab_size
        or model.model.padding_idx != layout.pad_token_id
        or len(model.model.layers) != layout.layer_count
    ):
        raise Lfm2DynamicBlankMaterializationError(
            "Materialized LFM2 root/core runtime semantics drifted."
        )
    rotary = model.model.rotary_emb
    if (
        rotary.max_seq_len_cached != layout.max_position_embeddings
        or rotary.original_max_seq_len != layout.max_position_embeddings
        or rotary.rope_type != "default"
        or type(rotary.attention_scaling) not in {int, float}
        or float(rotary.attention_scaling) != 1.0
    ):
        raise Lfm2DynamicBlankMaterializationError(
            "Materialized LFM2 rotary runtime semantics drifted."
        )
    modules = tuple(model.named_modules(remove_duplicate=False))
    if any(
        type(module.training) is not bool or module.training for _, module in modules
    ):
        raise Lfm2DynamicBlankMaterializationError(
            "Materialized LFM2 runtime must remain entirely in eval mode."
        )
    if bool(getattr(model, "is_gradient_checkpointing", False)) or any(
        bool(getattr(module, "gradient_checkpointing", False)) for _, module in modules
    ):
        raise Lfm2DynamicBlankMaterializationError(
            "Materialized LFM2 runtime cannot enable gradient checkpointing."
        )
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise Lfm2DynamicBlankMaterializationError(
            "Materialized LFM2 runtime parameters must remain frozen."
        )
    for module_path, module in modules:
        if type(module) is Lfm2RMSNorm and (
            type(module.variance_epsilon) not in {int, float}
            or float(module.variance_epsilon) != layout.norm_eps
        ):
            raise Lfm2DynamicBlankMaterializationError(
                f"Materialized LFM2 RMSNorm {module_path!r} epsilon drifted."
            )

    def validate_linear(
        module: Any,
        *,
        in_features: int,
        out_features: int,
        label: str,
    ) -> None:
        if (
            type(module) is not torch.nn.Linear
            or type(module.in_features) is not int
            or module.in_features != in_features
            or type(module.out_features) is not int
            or module.out_features != out_features
            or module.bias is not None
            or tuple(module.weight.shape) != (out_features, in_features)
        ):
            raise Lfm2DynamicBlankMaterializationError(
                f"Materialized LFM2 {label} linear geometry drifted."
            )

    embedding = model.model.embed_tokens
    if (
        type(embedding) is not torch.nn.Embedding
        or type(embedding.num_embeddings) is not int
        or embedding.num_embeddings != layout.vocab_size
        or type(embedding.embedding_dim) is not int
        or embedding.embedding_dim != layout.hidden_size
        or embedding.padding_idx != layout.pad_token_id
        or embedding.max_norm is not None
        or type(embedding.norm_type) not in {int, float}
        or float(embedding.norm_type) != 2.0
        or embedding.scale_grad_by_freq is not False
        or embedding.sparse is not False
        or tuple(embedding.weight.shape) != (layout.vocab_size, layout.hidden_size)
    ):
        raise Lfm2DynamicBlankMaterializationError(
            "Materialized LFM2 token embedding runtime geometry drifted."
        )
    validate_linear(
        model.lm_head,
        in_features=layout.hidden_size,
        out_features=layout.vocab_size,
        label="language-model readout",
    )

    for physical, layer in zip(layout.layers, model.model.layers, strict=True):
        if (
            type(layer) is not Lfm2DecoderLayer
            or type(layer.feed_forward) is not dynamic_mlp_class
            or type(layer.is_attention_layer) is not bool
            or layer.is_attention_layer is not (physical.layer_type == "full_attention")
        ):
            raise Lfm2DynamicBlankMaterializationError(
                f"Materialized LFM2 layer {physical.layer_index} runtime is malformed."
            )
        intermediate = physical.mlp_intermediate_size
        if (
            tuple(layer.feed_forward.w1.weight.shape)
            != (intermediate, layout.hidden_size)
            or tuple(layer.feed_forward.w3.weight.shape)
            != (intermediate, layout.hidden_size)
            or tuple(layer.feed_forward.w2.weight.shape)
            != (layout.hidden_size, intermediate)
        ):
            raise Lfm2DynamicBlankMaterializationError(
                f"Materialized LFM2 layer {physical.layer_index} SwiGLU width drifted."
            )
        validate_linear(
            layer.feed_forward.w1,
            in_features=layout.hidden_size,
            out_features=intermediate,
            label=f"layer {physical.layer_index} MLP w1",
        )
        validate_linear(
            layer.feed_forward.w3,
            in_features=layout.hidden_size,
            out_features=intermediate,
            label=f"layer {physical.layer_index} MLP w3",
        )
        validate_linear(
            layer.feed_forward.w2,
            in_features=intermediate,
            out_features=layout.hidden_size,
            label=f"layer {physical.layer_index} MLP w2",
        )
        if physical.layer_type == "conv":
            conv = layer.conv
            kernel_values = [
                getattr(conv, name)
                for name in ("L_cache", "conv_kernel_size")
                if hasattr(conv, name)
            ]
            internal = physical.short_conv_internal_width
            assert internal is not None
            if (
                type(conv) is not dynamic_conv_class
                or conv.config is not config
                or type(conv.layer_idx) is not int
                or conv.layer_idx != physical.layer_index
                or not kernel_values
                or any(
                    type(value) is not int or value != layout.conv_kernel_size
                    for value in kernel_values
                )
                or conv.bias is not False
                or conv.layer_type != "conv"
                or type(conv.internal_channel_width) is not int
                or conv.internal_channel_width != internal
                or type(conv.external_hidden_size) is not int
                or conv.external_hidden_size != layout.hidden_size
                or type(conv.expected_layer_count) is not int
                or conv.expected_layer_count != layout.layer_count
                or conv.cache_owner_token is not cache_owner_token
                or type(conv.conv) is not torch.nn.Conv1d
                or conv.conv.in_channels != internal
                or conv.conv.out_channels != internal
                or conv.conv.kernel_size != (layout.conv_kernel_size,)
                or conv.conv.stride != (1,)
                or conv.conv.padding != (layout.conv_kernel_size - 1,)
                or conv.conv.dilation != (1,)
                or conv.conv.groups != internal
                or conv.conv.bias is not None
                or conv.conv.padding_mode != "zeros"
            ):
                raise Lfm2DynamicBlankMaterializationError(
                    f"Materialized LFM2 layer {physical.layer_index} ShortConv drifted."
                )
            validate_linear(
                conv.in_proj,
                in_features=layout.hidden_size,
                out_features=3 * internal,
                label=f"layer {physical.layer_index} ShortConv input",
            )
            validate_linear(
                conv.out_proj,
                in_features=internal,
                out_features=layout.hidden_size,
                label=f"layer {physical.layer_index} ShortConv output",
            )
        else:
            attention = layer.self_attn
            query_width = layout.num_attention_heads * layout.head_dim
            key_value_width = layout.num_key_value_heads * layout.head_dim
            if (
                type(attention) is not dynamic_attention_class
                or attention.config is not config
                or attention.cache_owner_token is not cache_owner_token
                or type(attention.layer_idx) is not int
                or attention.layer_idx != physical.layer_index
                or type(attention.head_dim) is not int
                or attention.head_dim != layout.head_dim
                or type(attention.num_key_value_groups) is not int
                or attention.num_key_value_groups
                != layout.num_attention_heads // layout.num_key_value_heads
                or type(attention.scaling) not in {int, float}
                or float(attention.scaling) != layout.head_dim**-0.5
                or attention.is_causal is not True
            ):
                raise Lfm2DynamicBlankMaterializationError(
                    f"Materialized LFM2 layer {physical.layer_index} attention drifted."
                )
            validate_linear(
                attention.q_proj,
                in_features=layout.hidden_size,
                out_features=query_width,
                label=f"layer {physical.layer_index} attention query",
            )
            validate_linear(
                attention.k_proj,
                in_features=layout.hidden_size,
                out_features=key_value_width,
                label=f"layer {physical.layer_index} attention key",
            )
            validate_linear(
                attention.v_proj,
                in_features=layout.hidden_size,
                out_features=key_value_width,
                label=f"layer {physical.layer_index} attention value",
            )
            validate_linear(
                attention.out_proj,
                in_features=query_width,
                out_features=layout.hidden_size,
                label=f"layer {physical.layer_index} attention output",
            )


def _expected_module_type_names(
    layout: Lfm2DynamicPhysicalLayoutDescriptor,
    dependencies: tuple[Any, ...],
) -> Mapping[str, str]:
    """Build the exact native module graph independently of the live graph."""

    (
        torch,
        _,
        _,
        Lfm2ForCausalLM,
        _,
        _,
        Lfm2Attention,
        Lfm2DecoderLayer,
        _,
        Lfm2Model,
        Lfm2RMSNorm,
        Lfm2RotaryEmbedding,
        _,
    ) = dependencies

    def name(cls: type[Any]) -> str:
        return f"{cls.__module__}.{cls.__qualname__}"

    dynamic_conv = _dynamic_short_conv_class()
    dynamic_mlp = _dynamic_mlp_class()
    dynamic_attention = _dynamic_attention_class()
    result = {
        "": name(Lfm2ForCausalLM),
        "model": name(Lfm2Model),
        "model.embed_tokens": name(torch.nn.Embedding),
        "model.layers": name(torch.nn.ModuleList),
        "model.rotary_emb": name(Lfm2RotaryEmbedding),
        "model.embedding_norm": name(Lfm2RMSNorm),
        "lm_head": name(torch.nn.Linear),
    }
    for physical in layout.layers:
        prefix = f"model.layers.{physical.layer_index}"
        result[prefix] = name(Lfm2DecoderLayer)
        result[f"{prefix}.operator_norm"] = name(Lfm2RMSNorm)
        result[f"{prefix}.ffn_norm"] = name(Lfm2RMSNorm)
        result[f"{prefix}.feed_forward"] = name(dynamic_mlp)
        for projection in ("w1", "w3", "w2"):
            result[f"{prefix}.feed_forward.{projection}"] = name(torch.nn.Linear)
        if physical.layer_type == "conv":
            result[f"{prefix}.conv"] = name(dynamic_conv)
            result[f"{prefix}.conv.conv"] = name(torch.nn.Conv1d)
            result[f"{prefix}.conv.in_proj"] = name(torch.nn.Linear)
            result[f"{prefix}.conv.out_proj"] = name(torch.nn.Linear)
        else:
            result[f"{prefix}.self_attn"] = name(dynamic_attention)
            for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
                result[f"{prefix}.self_attn.{projection}"] = name(torch.nn.Linear)
            result[f"{prefix}.self_attn.q_layernorm"] = name(Lfm2RMSNorm)
            result[f"{prefix}.self_attn.k_layernorm"] = name(Lfm2RMSNorm)
    return MappingProxyType(result)


def _physical_audit(
    model: Any,
    layout: Lfm2DynamicPhysicalLayoutDescriptor,
    dependencies: tuple[Any, ...],
) -> Mapping[str, Any]:
    return authenticate_lfm2_physical_materialization(
        model,
        layout,
        expected_descriptor_sha256=layout.descriptor_sha256,
        expected_runtime_root_class=dependencies[3],
        expected_module_type_names=_expected_module_type_names(layout, dependencies),
    )


def _receipt(
    model: Any,
    config: Any,
    layout: Lfm2DynamicPhysicalLayoutDescriptor,
    audit: Mapping[str, Any],
    transformers_module: Any,
) -> Mapping[str, Any]:
    expected = lfm2_expected_physical_state_shapes(layout)
    config_snapshot = _config_snapshot(config)
    generation_snapshot = _generation_config_snapshot(
        model,
        config,
        transformers_module,
    )
    semantic_binding = _runtime_semantic_binding(config, layout)
    namespace_rows = [
        {"name": name, "shape": list(expected[name])} for name in sorted(expected)
    ]
    body = {
        "schema": LFM2_DYNAMIC_BLANK_MATERIALIZATION_SCHEMA,
        "status": "authenticated-dense-fp32-blank-child-materialization",
        "descriptor_sha256": layout.descriptor_sha256,
        "model_class": _qualified_class(model),
        "config_class": _qualified_class(config),
        "transformers_version": transformers_module.__version__,
        "parameter_state_key_count": len(expected),
        "parameter_state_shape_sha256": _canonical_sha256(namespace_rows),
        "parameter_elements": layout.accounting.total_parameter_elements,
        "physical_materialization_audit": dict(audit),
        "config_binding": {
            "schema_field": _CONFIG_SCHEMA_FIELD,
            "layout_field": _CONFIG_LAYOUT_FIELD,
            "layout_sha256_field": _CONFIG_LAYOUT_SHA_FIELD,
            "descriptor_embedded_in_serialized_config": True,
            "runtime_semantics": dict(semantic_binding),
            "runtime_semantics_sha256": _canonical_sha256(dict(semantic_binding)),
            "serialized_config_json_safe": True,
            "normalized_config_sha256": _canonical_sha256(dict(config_snapshot)),
            "generation_config_sha256": _canonical_sha256(dict(generation_snapshot)),
        },
        "origin_authentication": {
            "caller_must_supply_fresh_local_native_blank": True,
            "empty_name_or_path_required": True,
            "origin_proven_by_materialization_receipt": False,
        },
        "fresh_reload_contract": {
            "source": "caller-supplied-local-config-and-state-only",
            "hub_loader_allowed": False,
            "instantiate_native_lfm2_before_load": True,
            "extract_embedded_descriptor": True,
            "materialize_before_state_load": True,
            "strict_state_load_required": True,
            "fresh_dynamic_cache_required": True,
            "runtime_parent_or_sidecar_required": False,
        },
    }
    return MappingProxyType({**body, "receipt_sha256": _canonical_sha256(body)})


def materialize_lfm2_dynamic_blank_child(
    model: Any,
    config: Any,
    descriptor: Lfm2DynamicPhysicalLayoutDescriptor | Mapping[str, Any],
    *,
    expected_descriptor_sha256: str,
) -> Mapping[str, Any]:
    """Physically materialize an authenticated nonuniform blank LFM2 child.

    ``model`` must be a fresh, local, concrete ``Lfm2ForCausalLM`` built from
    ``config`` without a Hub loader.  Repeating this call on the same already
    authenticated materialization is idempotent; any drift fails closed.
    """

    layout = authenticate_lfm2_dynamic_physical_layout(
        descriptor,
        expected_descriptor_sha256=expected_descriptor_sha256,
    )
    dependencies = _dependencies()
    (
        _,
        transformers,
        Lfm2Config,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = dependencies
    _validate_config(config, layout, Lfm2Config)
    if model.config is not config or model.model.config is not config:
        raise Lfm2DynamicBlankMaterializationError(
            "Target model, nested core, and supplied config must share identity."
        )

    saved_layout = _saved_config_layout(config)
    if saved_layout is not _MISSING:
        recovered = extract_lfm2_dynamic_physical_layout_from_config(config)
        if recovered != layout:
            raise Lfm2DynamicBlankMaterializationError(
                "Target config is already bound to a different physical layout."
            )
        runtime_marker = getattr(model, _MODEL_RUNTIME_SHA_FIELD, _MISSING)
        if runtime_marker is not _MISSING:
            if runtime_marker != layout.descriptor_sha256:
                raise Lfm2DynamicBlankMaterializationError(
                    "Target runtime is bound to a different physical layout."
                )
            _validate_materialized_runtime(model, layout, dependencies)
            audit = _physical_audit(model, layout, dependencies)
            return _receipt(model, config, layout, audit, transformers)
        # A direct-local reload first reconstructs the ordinary native model
        # from the serialized config.  The model-local runtime marker is not
        # serialized, so only that genuinely fresh object reaches replacement.

    device = _validate_pristine_model(model, config, layout, dependencies)
    dynamic_conv_class = _dynamic_short_conv_class()
    dynamic_mlp_class = _dynamic_mlp_class()
    dynamic_attention_class = _dynamic_attention_class()
    cache_owner_token = object()
    replacements: list[tuple[Any, Any, Any | None, Any | None]] = []
    for physical, layer in zip(layout.layers, model.model.layers, strict=True):
        mlp_config = copy.deepcopy(config)
        mlp_config.intermediate_size = physical.mlp_intermediate_size
        if hasattr(mlp_config, "block_ff_dim"):
            mlp_config.block_ff_dim = physical.mlp_intermediate_size
        mlp_config.block_auto_adjust_ff_dim = False
        new_mlp = dynamic_mlp_class(mlp_config)
        new_mlp.apply(model._init_weights)
        new_mlp.to(device=device, dtype=dependencies[0].float32)
        new_conv = None
        new_attention = None
        if physical.layer_type == "conv":
            assert physical.short_conv_internal_width is not None
            new_conv = dynamic_conv_class(
                config,
                physical.layer_index,
                internal_channel_width=physical.short_conv_internal_width,
                expected_layer_count=layout.layer_count,
                cache_owner_token=cache_owner_token,
            )
            new_conv.apply(model._init_weights)
            new_conv.to(device=device, dtype=dependencies[0].float32)
        else:
            new_attention = dynamic_attention_class(
                config,
                physical.layer_index,
                cache_owner_token=cache_owner_token,
            )
            new_attention.apply(model._init_weights)
            new_attention.to(device=device, dtype=dependencies[0].float32)
        replacements.append((layer, new_mlp, new_conv, new_attention))

    old_modules = [
        (
            layer,
            layer.feed_forward,
            getattr(layer, "conv", None),
            getattr(layer, "self_attn", None),
        )
        for layer, _, _, _ in replacements
    ]
    previous_config = {
        name: getattr(config, name, _MISSING)
        for name in (
            _CONFIG_SCHEMA_FIELD,
            _CONFIG_LAYOUT_FIELD,
            _CONFIG_LAYOUT_SHA_FIELD,
        )
    }
    previous_runtime_marker = getattr(model, _MODEL_RUNTIME_SHA_FIELD, _MISSING)
    previous_cache_owner = getattr(model, _MODEL_CACHE_OWNER_FIELD, _MISSING)
    was_training = bool(model.training)
    original_requires_grad = {
        id(parameter): bool(parameter.requires_grad) for parameter in model.parameters()
    }
    try:
        for layer, new_mlp, new_conv, new_attention in replacements:
            layer.feed_forward = new_mlp
            if new_conv is not None:
                layer.conv = new_conv
            if new_attention is not None:
                layer.self_attn = new_attention
        setattr(config, _CONFIG_SCHEMA_FIELD, LFM2_DYNAMIC_BLANK_CONFIG_SCHEMA)
        setattr(config, _CONFIG_LAYOUT_FIELD, layout.as_dict())
        setattr(config, _CONFIG_LAYOUT_SHA_FIELD, layout.descriptor_sha256)
        setattr(model, _MODEL_RUNTIME_SHA_FIELD, layout.descriptor_sha256)
        setattr(model, _MODEL_CACHE_OWNER_FIELD, cache_owner_token)
        model.requires_grad_(False)
        model.eval()
        _validate_materialized_runtime(model, layout, dependencies)
        audit = _physical_audit(model, layout, dependencies)
    except Exception:
        for layer, old_mlp, old_conv, old_attention in old_modules:
            layer.feed_forward = old_mlp
            if old_conv is not None:
                layer.conv = old_conv
            if old_attention is not None:
                layer.self_attn = old_attention
        for parameter in model.parameters():
            parameter.requires_grad_(original_requires_grad[id(parameter)])
        model.train(was_training)
        for name, previous in previous_config.items():
            if previous is _MISSING:
                try:
                    delattr(config, name)
                except AttributeError:
                    pass
            else:
                setattr(config, name, previous)
        if previous_runtime_marker is _MISSING:
            try:
                delattr(model, _MODEL_RUNTIME_SHA_FIELD)
            except AttributeError:
                pass
        else:
            setattr(model, _MODEL_RUNTIME_SHA_FIELD, previous_runtime_marker)
        if previous_cache_owner is _MISSING:
            try:
                delattr(model, _MODEL_CACHE_OWNER_FIELD)
            except AttributeError:
                pass
        else:
            setattr(model, _MODEL_CACHE_OWNER_FIELD, previous_cache_owner)
        raise
    return _receipt(model, config, layout, audit, transformers)


__all__ = [
    "LFM2_DYNAMIC_BLANK_CONFIG_SCHEMA",
    "LFM2_DYNAMIC_BLANK_MATERIALIZATION_SCHEMA",
    "Lfm2DynamicBlankMaterializationError",
    "extract_lfm2_dynamic_physical_layout_from_config",
    "materialize_lfm2_dynamic_blank_child",
]
