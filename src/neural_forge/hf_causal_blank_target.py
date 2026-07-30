"""Local, family-neutral construction of dense Hugging Face causal blanks.

This module is intentionally narrower than checkpoint bootstrapping.  It only
turns an explicit architecture/configuration contract into a fresh causal LM
whose learned parameters are exact zero.  It never resolves a model ID, calls
``from_pretrained``, reads parent weights, or creates an auxiliary runtime.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


SUPPORTED_HF_CAUSAL_BLANK_FAMILIES = ("gpt2", "llama", "qwen3")

_ARCHITECTURE_BY_FAMILY = {
    "gpt2": "GPT2LMHeadModel",
    "llama": "LlamaForCausalLM",
    "qwen3": "Qwen3ForCausalLM",
}
_FAMILY_IDENTIFIERS = {
    "gpt2": "gpt2",
    "gpt2lmheadmodel": "gpt2",
    "llama": "llama",
    "llamaforcausallm": "llama",
    "qwen3": "qwen3",
    "qwen3forcausallm": "qwen3",
}
_SHAPE_ALIASES = {
    "n_embd": "hidden_size",
    "n_inner": "intermediate_size",
    "n_layer": "num_hidden_layers",
    "num_layers": "num_hidden_layers",
    "n_head": "num_attention_heads",
    "n_positions": "max_position_embeddings",
    "n_ctx": "max_position_embeddings",
}
_REQUIRED_CONFIGURATION_KEYS = frozenset(
    {
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "tie_word_embeddings",
        "attention_bias",
        "mlp_bias",
    }
)
_IDENTITY_AND_TOKEN_CONFIGURATION_KEYS = frozenset(
    {
        "model_type",
        "architecture",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
        "hidden_act",
        "attention_dropout",
        "use_cache",
    }
)
_FAMILY_OPTIONAL_CONFIGURATION_KEYS = {
    "gpt2": _IDENTITY_AND_TOKEN_CONFIGURATION_KEYS,
    "llama": _IDENTITY_AND_TOKEN_CONFIGURATION_KEYS
    | frozenset({"rms_norm_eps", "rope_theta"}),
    "qwen3": _IDENTITY_AND_TOKEN_CONFIGURATION_KEYS
    | frozenset({"rms_norm_eps", "rope_theta", "use_sliding_window"}),
}
_ALL_OPTIONAL_CONFIGURATION_KEYS = frozenset().union(
    *_FAMILY_OPTIONAL_CONFIGURATION_KEYS.values()
)


class HFCausalBlankTargetError(ValueError):
    """An explicit HF causal blank contract could not be materialized."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HFCausalBlankTargetError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int_or_none(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HFCausalBlankTargetError(
            f"{name} must be a non-negative integer or None."
        )
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise HFCausalBlankTargetError(f"{name} must be boolean.")
    return value


def _positive_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise HFCausalBlankTargetError(f"{name} must be positive and finite.")
    return float(value)


def _family_identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HFCausalBlankTargetError(f"{name} must be a nonempty string.")
    normalized = value.strip().lower().replace("_", "").replace("-", "")
    try:
        return _FAMILY_IDENTIFIERS[normalized]
    except KeyError as exc:
        raise HFCausalBlankTargetError(
            f"{name} does not identify a supported GPT-2, Llama, or Qwen3 "
            "causal architecture."
        ) from exc


def _normalize_family(
    target_family: Any,
    configuration: Mapping[str, Any],
) -> str:
    candidates: list[tuple[str, str]] = []
    if not isinstance(target_family, str) or not target_family.strip():
        raise HFCausalBlankTargetError("target_family must be a nonempty string.")
    requested = target_family.strip().lower()
    if requested not in {"any", "auto"}:
        candidates.append(
            ("target_family", _family_identifier(requested, "target_family"))
        )
    for key in ("model_type", "architecture"):
        if key in configuration:
            candidates.append((key, _family_identifier(configuration[key], key)))
    if not candidates:
        raise HFCausalBlankTargetError(
            "An auto/any target requires model_type or architecture."
        )
    family = candidates[0][1]
    conflicts = [name for name, value in candidates if value != family]
    if conflicts:
        raise HFCausalBlankTargetError(
            "target_family, model_type, and architecture identify different "
            "families."
        )
    return family


def _canonical_configuration(
    target_family: Any,
    configuration: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    if not isinstance(configuration, Mapping):
        raise HFCausalBlankTargetError("configuration must be a mapping.")
    if any(not isinstance(key, str) or not key.strip() for key in configuration):
        raise HFCausalBlankTargetError(
            "configuration keys must be nonempty strings."
        )
    values = dict(configuration)
    family = _normalize_family(target_family, values)
    allowed = (
        _REQUIRED_CONFIGURATION_KEYS
        | _ALL_OPTIONAL_CONFIGURATION_KEYS
        | frozenset(_SHAPE_ALIASES)
    )
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise HFCausalBlankTargetError(
            "Unknown HF causal blank configuration keys: " + ", ".join(unknown)
        )
    inapplicable = sorted(
        set(values).intersection(_ALL_OPTIONAL_CONFIGURATION_KEYS)
        - _FAMILY_OPTIONAL_CONFIGURATION_KEYS[family]
    )
    if inapplicable:
        raise HFCausalBlankTargetError(
            f"Configuration keys are not applicable to {family}: "
            + ", ".join(inapplicable)
        )
    for alias, canonical in _SHAPE_ALIASES.items():
        if alias not in values:
            continue
        if canonical in values and values[canonical] != values[alias]:
            raise HFCausalBlankTargetError(
                f"Conflicting {canonical!r} and alias {alias!r}."
            )
        values[canonical] = values.pop(alias)
    missing = sorted(_REQUIRED_CONFIGURATION_KEYS - set(values))
    if missing:
        raise HFCausalBlankTargetError(
            "Missing HF causal blank configuration keys: " + ", ".join(missing)
        )

    for name in (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
    ):
        values[name] = _positive_int(values[name], name)
    for name in ("tie_word_embeddings", "attention_bias", "mlp_bias"):
        values[name] = _strict_bool(values[name], name)
    if values["tie_word_embeddings"]:
        raise HFCausalBlankTargetError(
            "HF causal blank targets must set tie_word_embeddings=false."
        )
    for name in ("use_cache", "use_sliding_window"):
        if name in values:
            values[name] = _strict_bool(values[name], name)
    if family == "qwen3" and values.get("use_sliding_window", False):
        raise HFCausalBlankTargetError(
            "The dense Qwen3 blank seam does not support sliding-window layers."
        )
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        if name in values:
            values[name] = _nonnegative_int_or_none(values[name], name)
            token_id = values[name]
            if token_id is not None and token_id >= values["vocab_size"]:
                raise HFCausalBlankTargetError(
                    f"{name} must be smaller than vocab_size."
                )
    if "hidden_act" in values and (
        not isinstance(values["hidden_act"], str)
        or not values["hidden_act"].strip()
    ):
        raise HFCausalBlankTargetError("hidden_act must be a nonempty string.")
    if "rms_norm_eps" in values:
        values["rms_norm_eps"] = _positive_float(
            values["rms_norm_eps"], "rms_norm_eps"
        )
    if "rope_theta" in values:
        values["rope_theta"] = _positive_float(values["rope_theta"], "rope_theta")
    if "attention_dropout" in values:
        dropout = values["attention_dropout"]
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not math.isfinite(float(dropout))
            or not 0.0 <= float(dropout) < 1.0
        ):
            raise HFCausalBlankTargetError(
                "attention_dropout must be finite in the interval [0, 1)."
            )
        values["attention_dropout"] = float(dropout)

    heads = values["num_attention_heads"]
    kv_heads = values["num_key_value_heads"]
    head_dim = values["head_dim"]
    if heads % kv_heads:
        raise HFCausalBlankTargetError(
            "num_attention_heads must be divisible by num_key_value_heads."
        )
    if family == "gpt2":
        if kv_heads != heads:
            raise HFCausalBlankTargetError(
                "GPT-2 requires num_key_value_heads == num_attention_heads."
            )
        if values["hidden_size"] != heads * head_dim:
            raise HFCausalBlankTargetError(
                "GPT-2 requires hidden_size == num_attention_heads * head_dim."
            )
        if not values["attention_bias"] or not values["mlp_bias"]:
            raise HFCausalBlankTargetError(
                "Stock GPT-2 requires attention_bias=true and mlp_bias=true."
            )
    else:
        if head_dim % 2:
            raise HFCausalBlankTargetError(
                "Llama and Qwen3 require an even head_dim for RoPE."
            )
        if family == "qwen3" and values["mlp_bias"]:
            raise HFCausalBlankTargetError(
                "Stock Qwen3 does not expose MLP bias parameters."
            )

    values["model_type"] = family
    values["architecture"] = _ARCHITECTURE_BY_FAMILY[family]
    values.setdefault("use_cache", False)
    if family == "qwen3":
        values["use_sliding_window"] = False
    return family, values


def normalize_hf_causal_blank_target_configuration(
    *,
    target_family: str,
    configuration: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    """Return the unambiguous family and canonical strict constructor config."""

    family, values = _canonical_configuration(target_family, configuration)
    return family, MappingProxyType(values)


@dataclass(frozen=True, slots=True)
class HFCausalBlankTargetMaterialization:
    """A verified, fresh, dense, untied causal blank held only in its model."""

    model: Any
    family: str
    architecture: str
    constructor: str
    configuration: Mapping[str, Any]
    logical_parameter_count: int
    unique_parameter_count: int
    logical_parameter_tensors: int
    unique_parameter_tensors: int
    unique_parameter_storages: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "configuration",
            MappingProxyType(dict(self.configuration)),
        )


def _model_class_and_config(
    transformers: Any,
    family: str,
    values: Mapping[str, Any],
) -> tuple[Any, Any]:
    common = {
        "vocab_size": values["vocab_size"],
        "bos_token_id": values.get("bos_token_id"),
        "eos_token_id": values.get("eos_token_id"),
        "pad_token_id": values.get("pad_token_id"),
        "unk_token_id": values.get("unk_token_id"),
        "tie_word_embeddings": False,
        "use_cache": values["use_cache"],
    }
    if family == "gpt2":
        config = transformers.GPT2Config(
            **common,
            n_embd=values["hidden_size"],
            n_inner=values["intermediate_size"],
            n_layer=values["num_hidden_layers"],
            n_head=values["num_attention_heads"],
            n_positions=values["max_position_embeddings"],
            n_ctx=values["max_position_embeddings"],
            activation_function=values.get("hidden_act", "gelu_new"),
            attn_pdrop=values.get("attention_dropout", 0.0),
        )
        return transformers.GPT2LMHeadModel, config
    rotary = {}
    if "rope_theta" in values:
        rotary["rope_theta"] = values["rope_theta"]
    if family == "llama":
        config = transformers.LlamaConfig(
            **common,
            **rotary,
            hidden_size=values["hidden_size"],
            intermediate_size=values["intermediate_size"],
            num_hidden_layers=values["num_hidden_layers"],
            num_attention_heads=values["num_attention_heads"],
            num_key_value_heads=values["num_key_value_heads"],
            head_dim=values["head_dim"],
            max_position_embeddings=values["max_position_embeddings"],
            hidden_act=values.get("hidden_act", "silu"),
            rms_norm_eps=values.get("rms_norm_eps", 1e-6),
            attention_bias=values["attention_bias"],
            mlp_bias=values["mlp_bias"],
            attention_dropout=values.get("attention_dropout", 0.0),
        )
        return transformers.LlamaForCausalLM, config
    if not hasattr(transformers, "Qwen3ForCausalLM"):
        raise HFCausalBlankTargetError(
            "The installed Transformers build lacks Qwen3 support."
        )
    config = transformers.Qwen3Config(
        **common,
        **rotary,
        hidden_size=values["hidden_size"],
        intermediate_size=values["intermediate_size"],
        num_hidden_layers=values["num_hidden_layers"],
        num_attention_heads=values["num_attention_heads"],
        num_key_value_heads=values["num_key_value_heads"],
        head_dim=values["head_dim"],
        max_position_embeddings=values["max_position_embeddings"],
        hidden_act=values.get("hidden_act", "silu"),
        rms_norm_eps=values.get("rms_norm_eps", 1e-6),
        attention_bias=values["attention_bias"],
        attention_dropout=values.get("attention_dropout", 0.0),
        use_sliding_window=False,
    )
    return transformers.Qwen3ForCausalLM, config


def _storage_identity(tensor: Any) -> tuple[str, int | None, int, int]:
    storage = tensor.untyped_storage()
    return (
        str(tensor.device.type),
        tensor.device.index,
        int(storage.data_ptr()),
        int(storage.nbytes()),
    )


def _verify_applied_configuration(
    model: Any,
    *,
    family: str,
    values: Mapping[str, Any],
) -> None:
    config = model.config
    if (
        type(model).__name__ != _ARCHITECTURE_BY_FAMILY[family]
        or getattr(config, "model_type", None) != family
    ):
        raise HFCausalBlankTargetError(
            "The constructed model does not match its normalized family/architecture."
        )
    required_attribute_by_key = {
        "vocab_size": "vocab_size",
        "hidden_size": "n_embd" if family == "gpt2" else "hidden_size",
        "intermediate_size": (
            "n_inner" if family == "gpt2" else "intermediate_size"
        ),
        "num_hidden_layers": (
            "n_layer" if family == "gpt2" else "num_hidden_layers"
        ),
        "num_attention_heads": (
            "n_head" if family == "gpt2" else "num_attention_heads"
        ),
        "max_position_embeddings": (
            "n_positions" if family == "gpt2" else "max_position_embeddings"
        ),
        "tie_word_embeddings": "tie_word_embeddings",
    }
    for key, attribute in required_attribute_by_key.items():
        if getattr(config, attribute, object()) != values[key]:
            raise HFCausalBlankTargetError(
                f"Constructed {family} config did not preserve required {key}."
            )
    configured_key_value_heads = (
        getattr(config, "n_head", object())
        if family == "gpt2"
        else getattr(config, "num_key_value_heads", object())
    )
    configured_head_dim = (
        getattr(config, "n_embd", 0) // getattr(config, "n_head", 1)
        if family == "gpt2"
        else getattr(config, "head_dim", object())
    )
    if configured_key_value_heads != values["num_key_value_heads"]:
        raise HFCausalBlankTargetError(
            f"Constructed {family} config did not preserve required "
            "num_key_value_heads."
        )
    if configured_head_dim != values["head_dim"]:
        raise HFCausalBlankTargetError(
            f"Constructed {family} config did not preserve required head_dim."
        )

    if family == "gpt2":
        layers = tuple(model.transformer.h)
        attention_projections = tuple(
            projection
            for layer in layers
            for projection in (layer.attn.c_attn, layer.attn.c_proj)
        )
        mlp_projections = tuple(
            projection
            for layer in layers
            for projection in (layer.mlp.c_fc, layer.mlp.c_proj)
        )
    else:
        layers = tuple(model.model.layers)
        attention_projections = tuple(
            projection
            for layer in layers
            for projection in (
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
            )
        )
        mlp_projections = tuple(
            projection
            for layer in layers
            for projection in (
                layer.mlp.gate_proj,
                layer.mlp.up_proj,
                layer.mlp.down_proj,
            )
        )
    if len(layers) != values["num_hidden_layers"]:
        raise HFCausalBlankTargetError(
            f"Constructed {family} model did not preserve required "
            "num_hidden_layers."
        )
    observed_attention_bias = any(
        getattr(projection, "bias", None) is not None
        for projection in attention_projections
    )
    observed_mlp_bias = any(
        getattr(projection, "bias", None) is not None
        for projection in mlp_projections
    )
    if observed_attention_bias != values["attention_bias"]:
        raise HFCausalBlankTargetError(
            f"Constructed {family} model did not preserve required "
            "attention_bias."
        )
    if observed_mlp_bias != values["mlp_bias"]:
        raise HFCausalBlankTargetError(
            f"Constructed {family} model did not preserve required mlp_bias."
        )
    attribute_by_key = {
        "bos_token_id": "bos_token_id",
        "eos_token_id": "eos_token_id",
        "pad_token_id": "pad_token_id",
        "unk_token_id": "unk_token_id",
        "use_cache": "use_cache",
        "hidden_act": (
            "activation_function" if family == "gpt2" else "hidden_act"
        ),
        "attention_dropout": (
            "attn_pdrop" if family == "gpt2" else "attention_dropout"
        ),
        "rms_norm_eps": "rms_norm_eps",
        "use_sliding_window": "use_sliding_window",
    }
    for key in _FAMILY_OPTIONAL_CONFIGURATION_KEYS[family]:
        if key not in values or key in {"model_type", "architecture", "rope_theta"}:
            continue
        attribute = attribute_by_key[key]
        if getattr(config, attribute, object()) != values[key]:
            raise HFCausalBlankTargetError(
                f"Constructed {family} config did not preserve {key}."
            )
    if "rope_theta" in values:
        rope_parameters = getattr(config, "rope_parameters", None)
        if (
            not isinstance(rope_parameters, Mapping)
            or rope_parameters.get("rope_theta") != values["rope_theta"]
        ):
            raise HFCausalBlankTargetError(
                f"Constructed {family} config did not preserve rope_theta."
            )


def _verify_fresh_dense_untied_zero_model(
    torch: Any,
    model: Any,
    *,
    exact_unique_parameters: int,
) -> tuple[int, int, int, int, int]:
    try:
        parameters = tuple(
            model.named_parameters(recurse=True, remove_duplicate=False)
        )
    except TypeError:  # pragma: no cover - older supported Torch boundary
        parameters = tuple(model.named_parameters())
    if not parameters:
        raise HFCausalBlankTargetError(
            "The materialized causal target has no learned parameters."
        )
    object_ids: set[int] = set()
    storages: set[tuple[str, int | None, int, int]] = set()
    logical_count = 0
    for name, parameter in parameters:
        if not isinstance(parameter, torch.nn.Parameter):
            raise HFCausalBlankTargetError(
                f"Target state {name!r} is not a Parameter."
            )
        if parameter.device.type == "meta" or parameter.layout != torch.strided:
            raise HFCausalBlankTargetError(
                f"Target Parameter {name!r} is not dense materialized storage."
            )
        logical_count += int(parameter.numel())
        object_ids.add(id(parameter))
        storages.add(_storage_identity(parameter))
        if parameter.grad is not None or parameter.grad_fn is not None:
            raise HFCausalBlankTargetError(
                f"Target Parameter {name!r} retains autograd state."
            )
        if int(torch.count_nonzero(parameter.detach()).item()) != 0:
            raise HFCausalBlankTargetError(
                f"Target Parameter {name!r} is not exact zero."
            )
    if len(object_ids) != len(parameters) or len(storages) != len(parameters):
        raise HFCausalBlankTargetError(
            "The target contains tied, shared, or storage-aliased Parameters."
        )
    unique_count = sum(
        int(parameter.numel())
        for parameter in {id(value): value for _name, value in parameters}.values()
    )
    if logical_count != unique_count:
        raise HFCausalBlankTargetError(
            "Logical and unique target parameter counts differ."
        )
    if unique_count != exact_unique_parameters:
        raise HFCausalBlankTargetError(
            "Materialized target misses the exact unique-parameter budget: "
            f"expected {exact_unique_parameters}, observed {unique_count}."
        )
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    input_weight = getattr(input_embeddings, "weight", None)
    output_weight = getattr(output_embeddings, "weight", None)
    if (
        not isinstance(input_weight, torch.nn.Parameter)
        or not isinstance(output_weight, torch.nn.Parameter)
        or input_weight is output_weight
        or _storage_identity(input_weight) == _storage_identity(output_weight)
    ):
        raise HFCausalBlankTargetError(
            "Input embeddings and output head must own independent Parameters."
        )
    return (
        logical_count,
        unique_count,
        len(parameters),
        len(object_ids),
        len(storages),
    )


def materialize_hf_causal_blank_target(
    *,
    target_family: str,
    configuration: Mapping[str, Any],
    exact_unique_parameters: int,
    seed: int = 0,
    device: Any = "cpu",
    dtype: Any | None = None,
) -> HFCausalBlankTargetMaterialization:
    """Construct and prove a fresh exact-zero dense untied causal LM locally.

    ``configuration`` is a strict architecture contract.  Common GPT-2 aliases
    are normalized explicitly; contradictory aliases and unknown keys are
    rejected.  The function never accepts a parent model or checkpoint path,
    which makes pretrained or network-backed construction impossible here.
    """

    expected = _positive_int(
        exact_unique_parameters,
        "exact_unique_parameters",
    )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise HFCausalBlankTargetError("seed must be a non-negative integer.")
    family, normalized = normalize_hf_causal_blank_target_configuration(
        target_family=target_family,
        configuration=configuration,
    )
    values = dict(normalized)
    try:
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise HFCausalBlankTargetError(
            "PyTorch and Transformers are required for HF causal blank construction."
        ) from exc
    try:
        resolved_device = torch.device(device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HFCausalBlankTargetError(f"Invalid device {device!r}.") from exc
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise HFCausalBlankTargetError(
            "A CUDA target was requested but CUDA is unavailable."
        )
    resolved_dtype = torch.float32 if dtype is None else dtype
    if resolved_dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }:
        raise HFCausalBlankTargetError(
            "dtype must be a floating-point torch dtype."
        )
    model_class, config = _model_class_and_config(transformers, family, values)
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(seed)
        with torch.no_grad():
            model = model_class(config)
            model.to(device=resolved_device, dtype=resolved_dtype)
            for parameter in model.parameters():
                parameter.zero_()
                parameter.grad = None
    model.eval()
    _verify_applied_configuration(
        model,
        family=family,
        values=values,
    )
    counts = _verify_fresh_dense_untied_zero_model(
        torch,
        model,
        exact_unique_parameters=expected,
    )
    return HFCausalBlankTargetMaterialization(
        model=model,
        family=family,
        architecture=_ARCHITECTURE_BY_FAMILY[family],
        constructor=f"{model_class.__name__}({type(config).__name__}(...))",
        configuration=values,
        logical_parameter_count=counts[0],
        unique_parameter_count=counts[1],
        logical_parameter_tensors=counts[2],
        unique_parameter_tensors=counts[3],
        unique_parameter_storages=counts[4],
    )


__all__ = [
    "HFCausalBlankTargetError",
    "HFCausalBlankTargetMaterialization",
    "SUPPORTED_HF_CAUSAL_BLANK_FAMILIES",
    "materialize_hf_causal_blank_target",
    "normalize_hf_causal_blank_target_configuration",
]
