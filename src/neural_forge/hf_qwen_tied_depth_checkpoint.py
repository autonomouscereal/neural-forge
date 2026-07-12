"""Standalone, pickle-free Qwen3 tied-depth checkpoints.

The JSON manifest describes the complete logical Qwen3 topology and its alias
graph.  The tensor payload stores each physical tensor exactly once.  Reload
constructs a fresh Qwen3 on the meta device and materializes one physical
tensor at a time, so neither the dense donor nor an expanded logical state dict
is required in memory.
"""

from __future__ import annotations

import hashlib
import json
import operator
import os
import shutil
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from neural_forge.hf_arch_layout import build_decoder_layout
from neural_forge.hf_tied_depth import (
    ATTENTION_PROJECTIONS,
    MLP_PROJECTIONS,
    TiedDepthError,
    TiedDepthReport,
    exact_token_row_mapping,
    prepare_qwen3_tied_depth_model,
)


CHECKPOINT_SCHEMA = "neural-forge-qwen3-tied-depth-checkpoint-v1"
MANIFEST_FILENAME = "neural_forge_tied_depth_manifest.json"
CONFIG_FILENAME = "config.json"
GENERATION_CONFIG_FILENAME = "generation_config.json"
SAFETENSORS_FILENAME = "model.safetensors"
RAW_TENSORS_FILENAME = "model.nfraw"
RAW_TENSOR_FORMAT = "neural-forge-raw-v1"
_MANIFEST_HASH_KEY = "manifest_sha256"
_NORM_ROLES = (
    "self_attn.q_norm",
    "self_attn.k_norm",
    "input_layernorm",
    "post_attention_layernorm",
)
_SUPPORTED_DTYPES = {
    str(dtype): dtype
    for dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
}


class Qwen3TiedDepthCheckpointError(ValueError):
    """Base class for invalid source models, options, or artifacts."""


class Qwen3TiedDepthBudgetError(Qwen3TiedDepthCheckpointError):
    """A requested parameter or memory cap cannot contain the checkpoint."""


class Qwen3TiedDepthFormatError(Qwen3TiedDepthCheckpointError):
    """A checkpoint artifact is incomplete, inconsistent, or corrupted."""


@dataclass(frozen=True, slots=True)
class _Qwen3Dimensions:
    layers: int
    hidden: int
    intermediate: int
    query_width: int
    key_value_width: int
    head_dim: int
    config_vocab: int

    @property
    def shared_attention_parameters(self) -> int:
        return 2 * self.query_width * self.hidden + (
            2 * self.key_value_width * self.hidden
        )

    @property
    def shared_mlp_parameters(self) -> int:
        return 3 * self.intermediate * self.hidden

    @property
    def shared_projection_parameters(self) -> int:
        return self.shared_attention_parameters + self.shared_mlp_parameters

    @property
    def layer_specific_norm_parameters(self) -> int:
        return self.layers * (2 * self.hidden + 2 * self.head_dim)

    @property
    def final_norm_parameters(self) -> int:
        return self.hidden


@dataclass(frozen=True, slots=True)
class Qwen3TiedDepthCheckpointPlan:
    """Closed-form physical and logical accounting for a tied-depth child."""

    parameter_budget: int | None
    budget_headroom: int | None
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    query_width: int
    key_value_width: int
    head_dim: int
    source_vocab_size: int
    target_vocab_size: int
    shared_attention_parameter_count: int
    shared_mlp_parameter_count: int
    layer_specific_norm_parameter_count: int
    final_norm_parameter_count: int
    vocabulary_unique_parameter_count: int
    vocabulary_logical_parameter_count: int
    checkpoint_unique_parameter_count: int
    checkpoint_logical_parameter_count: int
    dense_source_unique_parameter_count: int
    dense_source_logical_parameter_count: int

    @property
    def unique_parameter_count(self) -> int:
        return self.checkpoint_unique_parameter_count

    @property
    def logical_parameter_count(self) -> int:
        return self.checkpoint_logical_parameter_count

    @property
    def shared_projection_parameter_count(self) -> int:
        return self.shared_attention_parameter_count + self.shared_mlp_parameter_count

    @property
    def retained_norm_parameter_count(self) -> int:
        return (
            self.layer_specific_norm_parameter_count + self.final_norm_parameter_count
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "parameter_budget": self.parameter_budget,
            "budget_headroom": self.budget_headroom,
            "dimensions": {
                "num_hidden_layers": self.num_hidden_layers,
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "query_width": self.query_width,
                "key_value_width": self.key_value_width,
                "head_dim": self.head_dim,
                "source_vocab_size": self.source_vocab_size,
                "target_vocab_size": self.target_vocab_size,
            },
            "shared_attention_parameter_count": (self.shared_attention_parameter_count),
            "shared_mlp_parameter_count": self.shared_mlp_parameter_count,
            "shared_projection_parameter_count": (
                self.shared_projection_parameter_count
            ),
            "layer_specific_norm_parameter_count": (
                self.layer_specific_norm_parameter_count
            ),
            "final_norm_parameter_count": self.final_norm_parameter_count,
            "retained_norm_parameter_count": self.retained_norm_parameter_count,
            "vocabulary_unique_parameter_count": (
                self.vocabulary_unique_parameter_count
            ),
            "vocabulary_logical_parameter_count": (
                self.vocabulary_logical_parameter_count
            ),
            "unique_parameter_count": self.checkpoint_unique_parameter_count,
            "logical_parameter_count": self.checkpoint_logical_parameter_count,
            "dense_source_unique_parameter_count": (
                self.dense_source_unique_parameter_count
            ),
            "dense_source_logical_parameter_count": (
                self.dense_source_logical_parameter_count
            ),
            "tied_vocabulary": True,
            "logical_layers": self.num_hidden_layers,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise Qwen3TiedDepthCheckpointError(f"{name} must be an integer.")
    try:
        return operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Qwen3TiedDepthCheckpointError(f"{name} must be an integer.") from exc


def _positive_integer(value: Any, name: str) -> int:
    result = _integer(value, name)
    if result <= 0:
        raise Qwen3TiedDepthCheckpointError(f"{name} must be positive.")
    return result


def _optional_nonnegative_integer(value: Any, name: str) -> int | None:
    if value is None:
        return None
    result = _integer(value, name)
    if result < 0:
        raise Qwen3TiedDepthCheckpointError(f"{name} must be nonnegative or None.")
    return result


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _qwen3_dimensions(config: Any) -> _Qwen3Dimensions:
    if _config_value(config, "model_type") != "qwen3":
        raise Qwen3TiedDepthCheckpointError(
            "Qwen3 tied-depth checkpoints require model_type='qwen3'."
        )
    if _config_value(config, "tie_word_embeddings") is not True:
        raise Qwen3TiedDepthCheckpointError(
            "Qwen3 tied-depth checkpoints require tied word embeddings."
        )
    if _config_value(config, "attention_bias") is not False:
        raise Qwen3TiedDepthCheckpointError(
            "Qwen3 tied-depth checkpoints require bias-free attention."
        )
    if _config_value(config, "hidden_act") != "silu":
        raise Qwen3TiedDepthCheckpointError(
            "Qwen3 tied-depth checkpoints require hidden_act='silu'."
        )
    if _config_value(config, "use_sliding_window", False) is not False:
        raise Qwen3TiedDepthCheckpointError(
            "Sliding-window Qwen3 is not supported by this checkpoint schema."
        )

    layers = _positive_integer(
        _config_value(config, "num_hidden_layers"),
        "num_hidden_layers",
    )
    hidden = _positive_integer(_config_value(config, "hidden_size"), "hidden_size")
    intermediate = _positive_integer(
        _config_value(config, "intermediate_size"),
        "intermediate_size",
    )
    query_heads = _positive_integer(
        _config_value(config, "num_attention_heads"),
        "num_attention_heads",
    )
    key_value_heads = _positive_integer(
        _config_value(config, "num_key_value_heads"),
        "num_key_value_heads",
    )
    head_dim = _positive_integer(_config_value(config, "head_dim"), "head_dim")
    config_vocab = _positive_integer(
        _config_value(config, "vocab_size"),
        "vocab_size",
    )
    if not 1 < key_value_heads < query_heads or query_heads % key_value_heads != 0:
        raise Qwen3TiedDepthCheckpointError(
            "Qwen3 requires grouped-query attention with more than one key/value "
            "head strictly dividing the query-head count."
        )
    if head_dim % 2:
        raise Qwen3TiedDepthCheckpointError("Qwen3 head_dim must be even for RoPE.")
    layer_types = _config_value(config, "layer_types")
    if layer_types is not None and (
        not isinstance(layer_types, (list, tuple))
        or len(layer_types) != layers
        or any(value != "full_attention" for value in layer_types)
    ):
        raise Qwen3TiedDepthCheckpointError(
            "Every Qwen3 logical layer must use full_attention."
        )
    return _Qwen3Dimensions(
        layers=layers,
        hidden=hidden,
        intermediate=intermediate,
        query_width=query_heads * head_dim,
        key_value_width=key_value_heads * head_dim,
        head_dim=head_dim,
        config_vocab=config_vocab,
    )


def plan_qwen3_tied_depth_checkpoint(
    config: Any,
    target_vocab_size: int,
    *,
    source_vocab_size: int | None = None,
    parameter_budget: int | None = None,
) -> Qwen3TiedDepthCheckpointPlan:
    """Return exact physical/logical counts without constructing a model."""

    dimensions = _qwen3_dimensions(config)
    target_vocab = _positive_integer(target_vocab_size, "target_vocab_size")
    source_vocab = (
        dimensions.config_vocab
        if source_vocab_size is None
        else _positive_integer(source_vocab_size, "source_vocab_size")
    )
    if target_vocab > source_vocab:
        raise Qwen3TiedDepthCheckpointError(
            "target_vocab_size cannot exceed source_vocab_size."
        )
    budget = _optional_nonnegative_integer(parameter_budget, "parameter_budget")

    projection = dimensions.shared_projection_parameters
    retained_norms = (
        dimensions.layer_specific_norm_parameters + dimensions.final_norm_parameters
    )
    vocabulary_unique = target_vocab * dimensions.hidden
    vocabulary_logical = 2 * vocabulary_unique
    unique = vocabulary_unique + projection + retained_norms
    logical = (
        vocabulary_logical
        + dimensions.layers
        * (projection + 2 * dimensions.hidden + 2 * dimensions.head_dim)
        + dimensions.final_norm_parameters
    )
    dense_source_unique = (
        source_vocab * dimensions.hidden
        + dimensions.layers
        * (projection + 2 * dimensions.hidden + 2 * dimensions.head_dim)
        + dimensions.final_norm_parameters
    )
    dense_source_logical = dense_source_unique + source_vocab * dimensions.hidden
    if budget is not None and unique > budget:
        raise Qwen3TiedDepthBudgetError(
            f"Tied-depth checkpoint requires {unique} parameters, exceeding "
            f"parameter_budget={budget}."
        )
    return Qwen3TiedDepthCheckpointPlan(
        parameter_budget=budget,
        budget_headroom=None if budget is None else budget - unique,
        num_hidden_layers=dimensions.layers,
        hidden_size=dimensions.hidden,
        intermediate_size=dimensions.intermediate,
        query_width=dimensions.query_width,
        key_value_width=dimensions.key_value_width,
        head_dim=dimensions.head_dim,
        source_vocab_size=source_vocab,
        target_vocab_size=target_vocab,
        shared_attention_parameter_count=(dimensions.shared_attention_parameters),
        shared_mlp_parameter_count=dimensions.shared_mlp_parameters,
        layer_specific_norm_parameter_count=(dimensions.layer_specific_norm_parameters),
        final_norm_parameter_count=dimensions.final_norm_parameters,
        vocabulary_unique_parameter_count=vocabulary_unique,
        vocabulary_logical_parameter_count=vocabulary_logical,
        checkpoint_unique_parameter_count=unique,
        checkpoint_logical_parameter_count=logical,
        dense_source_unique_parameter_count=dense_source_unique,
        dense_source_logical_parameter_count=dense_source_logical,
    )


def count_qwen3_tied_depth_parameters(
    config: Any,
    target_vocab_size: int,
    *,
    source_vocab_size: int | None = None,
) -> int:
    """Return the exact physical parameter count for a tied-depth child."""

    return plan_qwen3_tied_depth_checkpoint(
        config,
        target_vocab_size,
        source_vocab_size=source_vocab_size,
    ).unique_parameter_count


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise Qwen3TiedDepthCheckpointError(
            "Checkpoint metadata must be finite JSON data."
        ) from exc
    return encoded + b"\n"


def _sha256_bytes(value: bytes | bytearray | memoryview[Any]) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping_fingerprint(mapping: Sequence[int]) -> str:
    payload = json.dumps(
        tuple(mapping),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return _sha256_bytes(payload)


def _write_bytes(path: Path, value: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _artifact_entry(path: Path) -> dict[str, Any]:
    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _tensor_cpu_view(tensor: Tensor) -> tuple[Tensor, memoryview[Any]]:
    cpu = tensor.detach().to(device="cpu").contiguous()
    byte_tensor = cpu.view(torch.uint8).reshape(-1)
    return cpu, memoryview(byte_tensor.numpy())


def _storage_key(tensor: Tensor) -> tuple[Any, ...]:
    if tensor.layout != torch.strided or tensor.device.type == "meta":
        raise Qwen3TiedDepthCheckpointError(
            "Checkpoint parameters must be materialized dense strided tensors."
        )
    return (
        tensor.device.type,
        tensor.device.index,
        str(tensor.dtype),
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


def _same_storage(left: Tensor, right: Tensor) -> bool:
    return _storage_key(left) == _storage_key(right)


def _parameter_counts(model: nn.Module) -> tuple[int, int]:
    unique: dict[tuple[Any, ...], nn.Parameter] = {}
    logical = 0
    for _, parameter in model.named_parameters(
        recurse=True,
        remove_duplicate=False,
    ):
        logical += int(parameter.numel())
        unique[_storage_key(parameter)] = parameter
    return sum(int(value.numel()) for value in unique.values()), logical


def _layer_index(value: Any, name: str, layer_count: int) -> int:
    result = _integer(value, name)
    if not 0 <= result < layer_count:
        raise Qwen3TiedDepthCheckpointError(
            f"{name} must be in [0, {layer_count - 1}]."
        )
    return result


def _tokenizer_rows(tokenizer: Any, label: str) -> tuple[str, ...]:
    try:
        size = len(tokenizer)
        vocabulary = tokenizer.get_vocab()
    except Exception as exc:
        raise Qwen3TiedDepthCheckpointError(
            f"{label} tokenizer must expose get_vocab() and __len__()."
        ) from exc
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise Qwen3TiedDepthCheckpointError(
            f"{label} tokenizer must have a positive integer size."
        )
    if not isinstance(vocabulary, dict) or len(vocabulary) != size:
        raise Qwen3TiedDepthCheckpointError(
            f"{label} tokenizer vocabulary size is inconsistent."
        )
    if set(vocabulary.values()) != set(range(size)):
        raise Qwen3TiedDepthCheckpointError(
            f"{label} tokenizer IDs must be exactly contiguous from zero."
        )
    rows: list[str] = []
    for token_id in range(size):
        try:
            token = tokenizer.convert_ids_to_tokens(token_id)
        except Exception as exc:
            raise Qwen3TiedDepthCheckpointError(
                f"{label} tokenizer could not resolve token ID {token_id}."
            ) from exc
        if not isinstance(token, str) or vocabulary.get(token) != token_id:
            raise Qwen3TiedDepthCheckpointError(
                f"{label} tokenizer row {token_id} does not round-trip exactly."
            )
        rows.append(token)
    return tuple(rows)


def _special_token_ids(tokenizer: Any, target_size: int) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, name, None)
        if value is not None:
            value = _integer(value, name)
            if not 0 <= value < target_size:
                raise Qwen3TiedDepthCheckpointError(
                    f"{name} must be within the target vocabulary."
                )
        result[name] = value
    return result


def _projection_key(layer_index: int, group: str, role: str) -> str:
    return f"model.layers.{layer_index}.{group}.{role}.weight"


def _norm_key(layer_index: int, role: str) -> str:
    return f"model.layers.{layer_index}.{role}.weight"


def _expected_aliases(
    layers: int,
    attention_source_layer: int,
    mlp_source_layer: int,
) -> dict[str, str]:
    aliases = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "lm_head.weight": "model.embed_tokens.weight",
        "model.norm.weight": "model.norm.weight",
    }
    for layer_index in range(layers):
        for role in ATTENTION_PROJECTIONS:
            aliases[_projection_key(layer_index, "self_attn", role)] = _projection_key(
                attention_source_layer, "self_attn", role
            )
        for role in MLP_PROJECTIONS:
            aliases[_projection_key(layer_index, "mlp", role)] = _projection_key(
                mlp_source_layer,
                "mlp",
                role,
            )
        for role in _NORM_ROLES:
            key = _norm_key(layer_index, role)
            aliases[key] = key
    return aliases


def _expected_shapes(
    dimensions: _Qwen3Dimensions,
    target_vocab_size: int,
) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (target_vocab_size, dimensions.hidden),
        "lm_head.weight": (target_vocab_size, dimensions.hidden),
        "model.norm.weight": (dimensions.hidden,),
    }
    for layer_index in range(dimensions.layers):
        result[_projection_key(layer_index, "self_attn", "q_proj")] = (
            dimensions.query_width,
            dimensions.hidden,
        )
        for role in ("k_proj", "v_proj"):
            result[_projection_key(layer_index, "self_attn", role)] = (
                dimensions.key_value_width,
                dimensions.hidden,
            )
        result[_projection_key(layer_index, "self_attn", "o_proj")] = (
            dimensions.hidden,
            dimensions.query_width,
        )
        for role in ("gate_proj", "up_proj"):
            result[_projection_key(layer_index, "mlp", role)] = (
                dimensions.intermediate,
                dimensions.hidden,
            )
        result[_projection_key(layer_index, "mlp", "down_proj")] = (
            dimensions.hidden,
            dimensions.intermediate,
        )
        for role in ("self_attn.q_norm", "self_attn.k_norm"):
            result[_norm_key(layer_index, role)] = (dimensions.head_dim,)
        for role in ("input_layernorm", "post_attention_layernorm"):
            result[_norm_key(layer_index, role)] = (dimensions.hidden,)
    return result


def _model_layers(model: nn.Module) -> nn.ModuleList:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, nn.ModuleList) or not layers:
        raise Qwen3TiedDepthCheckpointError(
            "Model must expose a non-empty model.layers ModuleList."
        )
    return layers


def _parameter_at(model: nn.Module, state_key: str) -> nn.Parameter:
    owner_path, _, name = state_key.rpartition(".")
    owner: Any = model
    for part in owner_path.split("."):
        owner = getattr(owner, part)
    value = getattr(owner, name)
    if not isinstance(value, nn.Parameter):
        raise Qwen3TiedDepthCheckpointError(f"Expected Parameter at {state_key!r}.")
    return value


def _assert_state_surface(
    model: nn.Module,
    aliases: Mapping[str, str],
) -> None:
    expected = set(aliases)
    state_keys = set(model.state_dict())
    parameter_keys = {
        name
        for name, _ in model.named_parameters(
            recurse=True,
            remove_duplicate=False,
        )
    }
    if state_keys != expected or parameter_keys != expected:
        raise Qwen3TiedDepthCheckpointError(
            "Qwen3 state topology differs from the strict tied-depth schema: "
            f"state_missing={sorted(expected - state_keys)!r}, "
            f"state_unexpected={sorted(state_keys - expected)!r}, "
            f"parameter_missing={sorted(expected - parameter_keys)!r}, "
            f"parameter_unexpected={sorted(parameter_keys - expected)!r}."
        )


def _validate_dense_source(
    model: nn.Module,
    plan: Qwen3TiedDepthCheckpointPlan,
) -> None:
    if getattr(model, "neural_forge_tied_depth_report", None) is not None:
        raise Qwen3TiedDepthCheckpointError(
            "Export requires an unprepared dense Qwen3 source model."
        )
    try:
        layout = build_decoder_layout(model)
    except Exception as exc:
        raise Qwen3TiedDepthCheckpointError(
            "Source model is not a supported exact dense Qwen3 layout."
        ) from exc
    if layout.family != "qwen3":
        raise Qwen3TiedDepthCheckpointError("Source model must be Qwen3.")

    layers = _model_layers(model)
    aliases = _expected_aliases(len(layers), 0, 0)
    _assert_state_surface(model, aliases)
    embedding = model.get_input_embeddings()
    output = model.get_output_embeddings()
    if (
        not isinstance(embedding, nn.Embedding)
        or not isinstance(output, nn.Linear)
        or embedding.weight is not output.weight
    ):
        raise Qwen3TiedDepthCheckpointError(
            "Dense source input and output vocabulary weights must be tied."
        )

    physical_keys: set[tuple[Any, ...]] = {_storage_key(embedding.weight)}
    for layer_index in range(len(layers)):
        for group, roles in (
            ("self_attn", ATTENTION_PROJECTIONS),
            ("mlp", MLP_PROJECTIONS),
        ):
            for role in roles:
                parameter = _parameter_at(
                    model,
                    _projection_key(layer_index, group, role),
                )
                storage = _storage_key(parameter)
                if storage in physical_keys:
                    raise Qwen3TiedDepthCheckpointError(
                        "Export requires distinct dense projection parameters."
                    )
                physical_keys.add(storage)
        for role in _NORM_ROLES:
            parameter = _parameter_at(model, _norm_key(layer_index, role))
            storage = _storage_key(parameter)
            if storage in physical_keys:
                raise Qwen3TiedDepthCheckpointError(
                    "Export requires every layer-specific norm to be distinct."
                )
            physical_keys.add(storage)
    final_storage = _storage_key(_parameter_at(model, "model.norm.weight"))
    if final_storage in physical_keys:
        raise Qwen3TiedDepthCheckpointError("Final norm must have distinct storage.")

    unique, logical = _parameter_counts(model)
    if (
        unique != plan.dense_source_unique_parameter_count
        or logical != plan.dense_source_logical_parameter_count
    ):
        raise Qwen3TiedDepthCheckpointError(
            "Dense source parameter counts disagree with the closed-form plan: "
            f"expected unique/logical "
            f"{plan.dense_source_unique_parameter_count}/"
            f"{plan.dense_source_logical_parameter_count}, got {unique}/{logical}."
        )


def _expected_report(
    plan: Qwen3TiedDepthCheckpointPlan,
    attention_source_layer: int,
    mlp_source_layer: int,
    token_row_fingerprint: str,
) -> TiedDepthReport:
    return TiedDepthReport(
        schema="neural-forge-qwen3-tied-depth-v1",
        method="exact-token-rows-plus-coupled-real-projection-sharing",
        logical_layers=plan.num_hidden_layers,
        attention_source_layer=attention_source_layer,
        mlp_source_layer=mlp_source_layer,
        source_vocab_size=plan.source_vocab_size,
        target_vocab_size=plan.target_vocab_size,
        token_row_fingerprint=token_row_fingerprint,
        source_unique_parameter_count=plan.dense_source_unique_parameter_count,
        source_logical_parameter_count=plan.dense_source_logical_parameter_count,
        child_unique_parameter_count=plan.unique_parameter_count,
        child_logical_parameter_count=plan.logical_parameter_count,
        unique_compression_ratio=(
            plan.dense_source_unique_parameter_count / plan.unique_parameter_count
        ),
        retained_layer_specific_norm_parameter_count=(
            plan.retained_norm_parameter_count
        ),
        shared_projection_parameter_count=(plan.shared_projection_parameter_count),
        tied_projection_alias_count=(
            plan.num_hidden_layers * (len(ATTENTION_PROJECTIONS) + len(MLP_PROJECTIONS))
        ),
        zero_gradient=True,
        full_logical_depth=True,
        exact_parent_token_rows=True,
    )


def _validate_prepared_topology(
    model: nn.Module,
    plan: Qwen3TiedDepthCheckpointPlan,
    attention_source_layer: int,
    mlp_source_layer: int,
    *,
    expected_report: TiedDepthReport | None,
) -> None:
    layers = _model_layers(model)
    aliases = _expected_aliases(
        len(layers),
        attention_source_layer,
        mlp_source_layer,
    )
    _assert_state_surface(model, aliases)
    embedding = model.get_input_embeddings()
    output = model.get_output_embeddings()
    if (
        not isinstance(embedding, nn.Embedding)
        or not isinstance(output, nn.Linear)
        or embedding.weight is not output.weight
    ):
        raise Qwen3TiedDepthCheckpointError(
            "Prepared vocabulary weights are not tied by Parameter identity."
        )

    canonical_storages: set[tuple[Any, ...]] = {_storage_key(embedding.weight)}
    for group, roles, source_index in (
        ("self_attn", ATTENTION_PROJECTIONS, attention_source_layer),
        ("mlp", MLP_PROJECTIONS, mlp_source_layer),
    ):
        for role in roles:
            source = _parameter_at(
                model,
                _projection_key(source_index, group, role),
            )
            storage = _storage_key(source)
            if storage in canonical_storages:
                raise Qwen3TiedDepthCheckpointError(
                    "Shared projection roles must use distinct physical tensors."
                )
            canonical_storages.add(storage)
            for layer_index in range(len(layers)):
                target = _parameter_at(
                    model,
                    _projection_key(layer_index, group, role),
                )
                if target is not source:
                    raise Qwen3TiedDepthCheckpointError(
                        f"Projection {group}.{role} is not shared over every layer."
                    )

    for layer_index in range(len(layers)):
        for role in _NORM_ROLES:
            parameter = _parameter_at(model, _norm_key(layer_index, role))
            storage = _storage_key(parameter)
            if storage in canonical_storages:
                raise Qwen3TiedDepthCheckpointError(
                    "Every layer-specific norm must retain distinct storage."
                )
            canonical_storages.add(storage)
    final_norm = _parameter_at(model, "model.norm.weight")
    if _storage_key(final_norm) in canonical_storages:
        raise Qwen3TiedDepthCheckpointError("Final norm must retain distinct storage.")

    unique, logical = _parameter_counts(model)
    if unique != plan.unique_parameter_count or logical != plan.logical_parameter_count:
        raise Qwen3TiedDepthCheckpointError(
            "Prepared parameter counts disagree with the tied-depth plan: "
            f"expected {plan.unique_parameter_count}/{plan.logical_parameter_count}, "
            f"got {unique}/{logical}."
        )
    if expected_report is not None:
        actual_report = getattr(model, "neural_forge_tied_depth_report", None)
        if not isinstance(actual_report, TiedDepthReport) or (
            actual_report.as_dict() != expected_report.as_dict()
        ):
            raise Qwen3TiedDepthCheckpointError(
                "In-memory tied-depth report disagrees with the checkpoint plan."
            )
    try:
        layout = build_decoder_layout(model)
    except Exception as exc:
        raise Qwen3TiedDepthCheckpointError(
            "Prepared model no longer has an exact Qwen3 layout."
        ) from exc
    if layout.layer_count != plan.num_hidden_layers:
        raise Qwen3TiedDepthCheckpointError("Prepared logical depth changed.")


def _canonical_state(
    model: nn.Module,
    aliases: Mapping[str, str],
    plan: Qwen3TiedDepthCheckpointPlan,
) -> tuple[
    dict[str, Tensor],
    dict[str, dict[str, Any]],
    dict[str, bool],
]:
    state = model.state_dict(keep_vars=True)
    requires_grad = {
        name: bool(parameter.requires_grad)
        for name, parameter in model.named_parameters(
            recurse=True,
            remove_duplicate=False,
        )
    }
    if set(state) != set(aliases) or set(requires_grad) != set(aliases):
        raise Qwen3TiedDepthCheckpointError(
            "State changed while constructing the checkpoint payload."
        )
    canonical_names = set(aliases.values())
    canonical_storage: dict[tuple[Any, ...], str] = {}
    tensors: dict[str, Tensor] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for canonical_name in sorted(canonical_names):
        value = state[canonical_name]
        storage = _storage_key(value)
        previous = canonical_storage.get(storage)
        if previous is not None:
            raise Qwen3TiedDepthCheckpointError(
                f"Canonical tensors {previous!r} and {canonical_name!r} alias."
            )
        canonical_storage[storage] = canonical_name
        if not value.is_contiguous():
            raise Qwen3TiedDepthCheckpointError(
                f"Canonical tensor {canonical_name!r} is not contiguous."
            )
        tensors[canonical_name] = value.detach()
        cpu, raw = _tensor_cpu_view(value)
        metadata[canonical_name] = {
            "shape": list(cpu.shape),
            "dtype": str(cpu.dtype),
            "numel": int(cpu.numel()),
            "nbytes": raw.nbytes,
            "sha256": _sha256_bytes(raw),
            "is_parameter": True,
        }
    for state_key, canonical_name in aliases.items():
        if not _same_storage(state[state_key], state[canonical_name]):
            raise Qwen3TiedDepthCheckpointError(
                f"Declared alias {state_key!r} does not share its canonical tensor."
            )
        if requires_grad[state_key] != requires_grad[canonical_name]:
            raise Qwen3TiedDepthCheckpointError(
                f"Alias group for {canonical_name!r} has mixed requires_grad values."
            )
    stored_parameters = sum(entry["numel"] for entry in metadata.values())
    if stored_parameters != plan.unique_parameter_count:
        raise Qwen3TiedDepthCheckpointError(
            "Canonical payload parameter count disagrees with the plan."
        )
    return tensors, metadata, requires_grad


def _select_tensor_format(requested: str | None) -> str:
    if requested not in {None, "safetensors", "raw"}:
        raise Qwen3TiedDepthCheckpointError(
            "tensor_format must be 'safetensors', 'raw', or None."
        )
    if requested == "raw":
        return "raw"
    try:
        import safetensors.torch  # noqa: F401
    except ImportError:
        if requested == "safetensors":
            raise Qwen3TiedDepthCheckpointError(
                "safetensors was requested but is not installed."
            ) from None
        return "raw"
    return "safetensors"


def _save_tensor_payload(
    directory: Path,
    tensors: Mapping[str, Tensor],
    metadata: dict[str, dict[str, Any]],
    tensor_format: str,
) -> tuple[str, Path]:
    ordered_names = sorted(tensors)
    if tensor_format == "safetensors":
        from safetensors.torch import save_file

        path = directory / SAFETENSORS_FILENAME
        save_file(
            {name: tensors[name] for name in ordered_names},
            str(path),
            metadata={"format": CHECKPOINT_SCHEMA},
        )
        _fsync_file(path)
        return "safetensors", path

    path = directory / RAW_TENSORS_FILENAME
    offset = 0
    with path.open("xb") as handle:
        for name in ordered_names:
            _, raw = _tensor_cpu_view(tensors[name])
            metadata[name]["byte_offset"] = offset
            written = handle.write(raw)
            if written != raw.nbytes:
                raise OSError(f"Short raw tensor write for {name!r}.")
            offset += raw.nbytes
        handle.flush()
        os.fsync(handle.fileno())
    return RAW_TENSOR_FORMAT, path


def _sanitized_config(model: nn.Module) -> dict[str, Any]:
    data = dict(model.config.to_dict())
    data["_name_or_path"] = ""
    data.pop("name_or_path", None)
    data.pop("auto_map", None)
    data["architectures"] = ["Qwen3ForCausalLM"]
    data["model_type"] = "qwen3"
    data["tie_word_embeddings"] = True
    return data


def _topology_manifest(
    plan: Qwen3TiedDepthCheckpointPlan,
    attention_source_layer: int,
    mlp_source_layer: int,
) -> dict[str, Any]:
    return {
        "model_class": "transformers.Qwen3ForCausalLM",
        "logical_layers": plan.num_hidden_layers,
        "attention": {
            "source_layer": attention_source_layer,
            "roles": list(ATTENTION_PROJECTIONS),
            "coupled_real_projection_set": True,
            "shared_over_every_logical_layer": True,
        },
        "mlp": {
            "source_layer": mlp_source_layer,
            "roles": list(MLP_PROJECTIONS),
            "coupled_real_projection_set": True,
            "shared_over_every_logical_layer": True,
        },
        "norms": {
            "roles": list(_NORM_ROLES),
            "layer_specific": True,
            "distinct_layer_tensor_count": (plan.num_hidden_layers * len(_NORM_ROLES)),
            "final_norm_retained": True,
        },
        "vocabulary": {
            "tied_input_output": True,
            "exact_reduced_donor_rows": True,
        },
        "parent_module_reference": False,
    }


def export_qwen3_tied_depth_checkpoint(
    model: nn.Module,
    output_directory: str | os.PathLike[str],
    donor_tokenizer: Any,
    target_tokenizer: Any,
    *,
    attention_source_layer: int,
    mlp_source_layer: int,
    parameter_budget: int | None = None,
    tensor_format: str | None = None,
) -> dict[str, Any]:
    """Fold and atomically export a standalone exact Qwen3 tied-depth child."""

    if not isinstance(model, nn.Module) or not hasattr(model, "config"):
        raise Qwen3TiedDepthCheckpointError(
            "model must be a configured torch.nn.Module."
        )
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise Qwen3TiedDepthCheckpointError(
            f"Refusing to replace existing checkpoint path {output}."
        )
    selected_format = _select_tensor_format(tensor_format)
    donor_tokens = _tokenizer_rows(donor_tokenizer, "donor")
    target_tokens = _tokenizer_rows(target_tokenizer, "target")
    source_embedding = model.get_input_embeddings()
    if not isinstance(source_embedding, nn.Embedding):
        raise Qwen3TiedDepthCheckpointError(
            "Source model must expose an nn.Embedding input table."
        )
    if len(donor_tokens) != source_embedding.num_embeddings:
        raise Qwen3TiedDepthCheckpointError(
            "Donor tokenizer size must equal the dense source vocabulary size."
        )
    try:
        donor_rows = exact_token_row_mapping(donor_tokenizer, target_tokenizer)
    except TiedDepthError as exc:
        raise Qwen3TiedDepthCheckpointError(str(exc)) from exc
    special_ids = _special_token_ids(target_tokenizer, len(target_tokens))
    plan = plan_qwen3_tied_depth_checkpoint(
        model.config,
        len(target_tokens),
        source_vocab_size=len(donor_tokens),
        parameter_budget=parameter_budget,
    )
    attention_index = _layer_index(
        attention_source_layer,
        "attention_source_layer",
        plan.num_hidden_layers,
    )
    mlp_index = _layer_index(
        mlp_source_layer,
        "mlp_source_layer",
        plan.num_hidden_layers,
    )
    _validate_dense_source(model, plan)
    token_fingerprint = _mapping_fingerprint(donor_rows)
    expected_report = _expected_report(
        plan,
        attention_index,
        mlp_index,
        token_fingerprint,
    )

    try:
        report = prepare_qwen3_tied_depth_model(
            model,
            donor_tokenizer,
            target_tokenizer,
            attention_source_layer=attention_index,
            mlp_source_layer=mlp_index,
        )
    except TiedDepthError as exc:
        raise Qwen3TiedDepthCheckpointError(str(exc)) from exc
    if report.as_dict() != expected_report.as_dict():
        raise Qwen3TiedDepthCheckpointError(
            "Tied-depth preparation produced an unexpected report."
        )
    _validate_prepared_topology(
        model,
        plan,
        attention_index,
        mlp_index,
        expected_report=expected_report,
    )
    aliases = _expected_aliases(
        plan.num_hidden_layers,
        attention_index,
        mlp_index,
    )
    tensors, tensor_metadata, requires_grad = _canonical_state(
        model,
        aliases,
        plan,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        config_path = temporary / CONFIG_FILENAME
        _write_bytes(config_path, _canonical_json_bytes(_sanitized_config(model)))
        generation_path = temporary / GENERATION_CONFIG_FILENAME
        generation_config = getattr(model, "generation_config", None)
        generation_data = (
            {} if generation_config is None else generation_config.to_dict()
        )
        _write_bytes(generation_path, _canonical_json_bytes(generation_data))
        stored_format, tensor_path = _save_tensor_payload(
            temporary,
            tensors,
            tensor_metadata,
            selected_format,
        )
        payload_bytes = sum(int(value["nbytes"]) for value in tensor_metadata.values())
        largest_tensor_bytes = max(
            int(value["nbytes"]) for value in tensor_metadata.values()
        )
        manifest_payload: dict[str, Any] = {
            "schema": CHECKPOINT_SCHEMA,
            "serialization": {
                "python_pickle": False,
                "serialized_modules": False,
                "serialized_weakrefs": False,
                "tensor_format": stored_format,
                "byte_order": sys.byteorder,
            },
            "artifacts": {
                "config": _artifact_entry(config_path),
                "generation_config": _artifact_entry(generation_path),
                "tensors": _artifact_entry(tensor_path),
            },
            "topology": _topology_manifest(plan, attention_index, mlp_index),
            "vocabulary": {
                "source_vocab_size": plan.source_vocab_size,
                "target_vocab_size": plan.target_vocab_size,
                "donor_row_indices": list(donor_rows),
                "donor_row_sha256": token_fingerprint,
                "target_raw_token_sha256": _sha256_bytes(
                    _canonical_json_bytes(list(target_tokens))
                ),
                "special_token_ids": special_ids,
            },
            "parameter_counts": plan.as_dict(),
            "tied_depth_report": report.as_dict(),
            "state_dict": {
                "key_to_tensor": dict(aliases),
                "logical_tensor_count": len(aliases),
                "stored_tensor_count": len(tensors),
                "alias_count": len(aliases) - len(tensors),
                "projection_alias_count": (
                    (plan.num_hidden_layers - 1)
                    * (len(ATTENTION_PROJECTIONS) + len(MLP_PROJECTIONS))
                ),
                "vocabulary_alias_count": 1,
                "parameter_requires_grad": requires_grad,
            },
            "tensors": tensor_metadata,
            "reload_memory": {
                "topology_construction_device": "meta",
                "streamed_physical_tensors": True,
                "model_parameter_bytes": payload_bytes,
                "largest_staging_tensor_bytes": largest_tensor_bytes,
                "bounded_peak_parameter_and_staging_bytes": (
                    payload_bytes + largest_tensor_bytes
                ),
            },
            "model_training": bool(model.training),
            "standalone": True,
            "parent_model_required": False,
            "external_weight_dependencies": [],
        }
        manifest_hash = _sha256_bytes(_canonical_json_bytes(manifest_payload))
        manifest = {**manifest_payload, _MANIFEST_HASH_KEY: manifest_hash}
        _write_bytes(
            temporary / MANIFEST_FILENAME,
            _canonical_json_bytes(manifest),
        )
        os.replace(temporary, output)
    except Exception:
        resolved = temporary.resolve()
        if resolved.parent == output.parent and resolved.exists():
            shutil.rmtree(resolved)
        raise

    setattr(model, "neural_forge_checkpoint_manifest", manifest)
    return manifest


def _reject_duplicate_json_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Qwen3TiedDepthFormatError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except Qwen3TiedDepthFormatError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Qwen3TiedDepthFormatError(
            f"Cannot read JSON artifact {path.name}."
        ) from exc
    if not isinstance(value, dict):
        raise Qwen3TiedDepthFormatError(f"{path.name} must contain a JSON object.")
    return value


def _strict_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Qwen3TiedDepthFormatError(f"{label} must be a JSON object.")
    actual = set(value)
    if actual != expected:
        raise Qwen3TiedDepthFormatError(
            f"{label} keys differ: missing={sorted(expected - actual)!r}, "
            f"unexpected={sorted(actual - expected)!r}."
        )
    return value


def _checked_artifact_path(root: Path, entry: Any, label: str) -> Path:
    value = _strict_keys(entry, {"file", "size_bytes", "sha256"}, label)
    filename = value["file"]
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise Qwen3TiedDepthFormatError(f"Artifact {label!r} has an unsafe filename.")
    path = root / filename
    if path.is_symlink():
        raise Qwen3TiedDepthFormatError(
            f"Artifact {filename!r} cannot be a symbolic link."
        )
    if not path.is_file():
        raise Qwen3TiedDepthFormatError(f"Artifact {filename!r} is missing.")
    expected_size = value["size_bytes"]
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
    ):
        raise Qwen3TiedDepthFormatError(f"Artifact {label!r} has no valid size.")
    if path.stat().st_size != expected_size:
        raise Qwen3TiedDepthFormatError(f"Artifact {filename!r} has the wrong size.")
    expected_hash = value["sha256"]
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise Qwen3TiedDepthFormatError(f"Artifact {label!r} has no valid SHA-256.")
    if _sha256_file(path) != expected_hash:
        raise Qwen3TiedDepthFormatError(f"Artifact {filename!r} failed SHA-256.")
    return path


def _format_integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Qwen3TiedDepthFormatError(
            f"Manifest field {name!r} must be an integer >= {minimum}."
        )
    return value


def _dtype_from_name(name: Any) -> torch.dtype:
    if not isinstance(name, str) or name not in _SUPPORTED_DTYPES:
        raise Qwen3TiedDepthFormatError(f"Unsupported tensor dtype {name!r}.")
    return _SUPPORTED_DTYPES[name]


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _checked_limit(value: Any, name: str) -> int | None:
    if value is None:
        return None
    try:
        return _optional_nonnegative_integer(value, name)
    except Qwen3TiedDepthCheckpointError as exc:
        raise Qwen3TiedDepthBudgetError(str(exc)) from exc


def _validate_vocabulary_manifest(
    value: Any,
    config_data: Mapping[str, Any],
) -> tuple[int, int, tuple[int, ...], str]:
    vocabulary = _strict_keys(
        value,
        {
            "source_vocab_size",
            "target_vocab_size",
            "donor_row_indices",
            "donor_row_sha256",
            "target_raw_token_sha256",
            "special_token_ids",
        },
        "vocabulary",
    )
    source_vocab = _format_integer(
        vocabulary["source_vocab_size"],
        "source_vocab_size",
        minimum=1,
    )
    target_vocab = _format_integer(
        vocabulary["target_vocab_size"],
        "target_vocab_size",
        minimum=1,
    )
    if target_vocab > source_vocab or config_data.get("vocab_size") != target_vocab:
        raise Qwen3TiedDepthFormatError(
            "Vocabulary sizes disagree with the standalone config."
        )
    raw_rows = vocabulary["donor_row_indices"]
    if not isinstance(raw_rows, list) or len(raw_rows) != target_vocab:
        raise Qwen3TiedDepthFormatError("Donor row mapping has the wrong length.")
    rows = tuple(
        _format_integer(value, f"donor_row_indices[{index}]")
        for index, value in enumerate(raw_rows)
    )
    if len(set(rows)) != len(rows) or any(value >= source_vocab for value in rows):
        raise Qwen3TiedDepthFormatError(
            "Donor row mapping must contain unique in-range rows."
        )
    fingerprint = vocabulary["donor_row_sha256"]
    if fingerprint != _mapping_fingerprint(rows):
        raise Qwen3TiedDepthFormatError("Donor row mapping SHA-256 mismatch.")
    target_token_hash = vocabulary["target_raw_token_sha256"]
    if not isinstance(target_token_hash, str) or len(target_token_hash) != 64:
        raise Qwen3TiedDepthFormatError("Target raw-token fingerprint is invalid.")
    special = _strict_keys(
        vocabulary["special_token_ids"],
        {"bos_token_id", "eos_token_id", "pad_token_id"},
        "special_token_ids",
    )
    for name, special_id in special.items():
        if special_id is not None and (
            isinstance(special_id, bool)
            or not isinstance(special_id, int)
            or not 0 <= special_id < target_vocab
        ):
            raise Qwen3TiedDepthFormatError(
                f"Special token {name!r} is outside the target vocabulary."
            )
        if config_data.get(name) != special_id:
            raise Qwen3TiedDepthFormatError(
                f"Special token {name!r} disagrees with config.json."
            )
    return source_vocab, target_vocab, rows, fingerprint


def _validate_topology_manifest(
    value: Any,
    plan: Qwen3TiedDepthCheckpointPlan,
) -> tuple[int, int]:
    topology = _strict_keys(
        value,
        {
            "model_class",
            "logical_layers",
            "attention",
            "mlp",
            "norms",
            "vocabulary",
            "parent_module_reference",
        },
        "topology",
    )
    if (
        topology["model_class"] != "transformers.Qwen3ForCausalLM"
        or topology["logical_layers"] != plan.num_hidden_layers
        or topology["parent_module_reference"] is not False
    ):
        raise Qwen3TiedDepthFormatError("Top-level topology is inconsistent.")
    attention = _strict_keys(
        topology["attention"],
        {
            "source_layer",
            "roles",
            "coupled_real_projection_set",
            "shared_over_every_logical_layer",
        },
        "attention topology",
    )
    mlp = _strict_keys(
        topology["mlp"],
        {
            "source_layer",
            "roles",
            "coupled_real_projection_set",
            "shared_over_every_logical_layer",
        },
        "MLP topology",
    )
    attention_index = _format_integer(
        attention["source_layer"],
        "attention.source_layer",
    )
    mlp_index = _format_integer(mlp["source_layer"], "mlp.source_layer")
    if attention_index >= plan.num_hidden_layers or mlp_index >= plan.num_hidden_layers:
        raise Qwen3TiedDepthFormatError("Projection source layer is out of range.")
    for group, expected_roles in (
        (attention, ATTENTION_PROJECTIONS),
        (mlp, MLP_PROJECTIONS),
    ):
        if (
            group["roles"] != list(expected_roles)
            or group["coupled_real_projection_set"] is not True
            or group["shared_over_every_logical_layer"] is not True
        ):
            raise Qwen3TiedDepthFormatError(
                "Projection topology does not describe coupled exact sharing."
            )
    norms = _strict_keys(
        topology["norms"],
        {
            "roles",
            "layer_specific",
            "distinct_layer_tensor_count",
            "final_norm_retained",
        },
        "norm topology",
    )
    if (
        norms["roles"] != list(_NORM_ROLES)
        or norms["layer_specific"] is not True
        or norms["distinct_layer_tensor_count"]
        != plan.num_hidden_layers * len(_NORM_ROLES)
        or norms["final_norm_retained"] is not True
    ):
        raise Qwen3TiedDepthFormatError("Norm topology is inconsistent.")
    vocabulary = _strict_keys(
        topology["vocabulary"],
        {"tied_input_output", "exact_reduced_donor_rows"},
        "vocabulary topology",
    )
    if (
        vocabulary["tied_input_output"] is not True
        or vocabulary["exact_reduced_donor_rows"] is not True
    ):
        raise Qwen3TiedDepthFormatError("Vocabulary topology is inconsistent.")
    return attention_index, mlp_index


def _validate_plan_counts(
    config_data: Mapping[str, Any],
    manifest: Mapping[str, Any],
    source_vocab_size: int,
    target_vocab_size: int,
) -> Qwen3TiedDepthCheckpointPlan:
    counts = manifest.get("parameter_counts")
    if not isinstance(counts, Mapping):
        raise Qwen3TiedDepthFormatError("Manifest parameter counts are missing.")
    budget = counts.get("parameter_budget")
    try:
        plan = plan_qwen3_tied_depth_checkpoint(
            config_data,
            target_vocab_size,
            source_vocab_size=source_vocab_size,
            parameter_budget=budget,
        )
    except Qwen3TiedDepthCheckpointError as exc:
        raise Qwen3TiedDepthFormatError(
            "Manifest cannot produce a valid tied-depth count plan."
        ) from exc
    if dict(counts) != plan.as_dict():
        raise Qwen3TiedDepthFormatError(
            "Manifest parameter counts disagree with the closed-form plan."
        )
    return plan


def _validate_report_manifest(
    value: Any,
    expected: TiedDepthReport,
) -> None:
    if not isinstance(value, Mapping) or dict(value) != expected.as_dict():
        raise Qwen3TiedDepthFormatError(
            "Tied-depth report disagrees with topology or counts."
        )


def _validate_state_and_tensors(
    manifest: Mapping[str, Any],
    plan: Qwen3TiedDepthCheckpointPlan,
    dimensions: _Qwen3Dimensions,
    attention_source_layer: int,
    mlp_source_layer: int,
    tensor_format: str,
    tensor_artifact_size: int,
) -> tuple[dict[str, str], dict[str, bool], int, int]:
    aliases = _expected_aliases(
        plan.num_hidden_layers,
        attention_source_layer,
        mlp_source_layer,
    )
    state = _strict_keys(
        manifest.get("state_dict"),
        {
            "key_to_tensor",
            "logical_tensor_count",
            "stored_tensor_count",
            "alias_count",
            "projection_alias_count",
            "vocabulary_alias_count",
            "parameter_requires_grad",
        },
        "state_dict",
    )
    raw_aliases = state["key_to_tensor"]
    if not isinstance(raw_aliases, Mapping) or dict(raw_aliases) != aliases:
        raise Qwen3TiedDepthFormatError(
            "State alias table disagrees with the tied-depth topology."
        )
    canonical_names = set(aliases.values())
    expected_projection_aliases = (plan.num_hidden_layers - 1) * (
        len(ATTENTION_PROJECTIONS) + len(MLP_PROJECTIONS)
    )
    if (
        state["logical_tensor_count"] != len(aliases)
        or state["stored_tensor_count"] != len(canonical_names)
        or state["alias_count"] != len(aliases) - len(canonical_names)
        or state["projection_alias_count"] != expected_projection_aliases
        or state["vocabulary_alias_count"] != 1
    ):
        raise Qwen3TiedDepthFormatError("State tensor counts are inconsistent.")
    raw_requires_grad = state["parameter_requires_grad"]
    if not isinstance(raw_requires_grad, Mapping) or set(raw_requires_grad) != set(
        aliases
    ):
        raise Qwen3TiedDepthFormatError(
            "Parameter requires-grad table has the wrong keys."
        )
    requires_grad: dict[str, bool] = {}
    for name, required in raw_requires_grad.items():
        if not isinstance(name, str) or not isinstance(required, bool):
            raise Qwen3TiedDepthFormatError(
                "Parameter requires-grad entries must map strings to booleans."
            )
        requires_grad[name] = required
    for name, canonical in aliases.items():
        if requires_grad[name] != requires_grad[canonical]:
            raise Qwen3TiedDepthFormatError(
                f"Alias group for {canonical!r} has mixed requires-grad metadata."
            )

    tensors = manifest.get("tensors")
    if not isinstance(tensors, Mapping) or set(tensors) != canonical_names:
        raise Qwen3TiedDepthFormatError(
            "Tensor metadata keys disagree with the canonical alias set."
        )
    shapes = _expected_shapes(dimensions, plan.target_vocab_size)
    payload_bytes = 0
    largest_bytes = 0
    logical_parameters = 0
    expected_offset = 0
    metadata_keys = {
        "shape",
        "dtype",
        "numel",
        "nbytes",
        "sha256",
        "is_parameter",
    }
    if tensor_format == RAW_TENSOR_FORMAT:
        metadata_keys = metadata_keys | {"byte_offset"}
    for name in sorted(canonical_names):
        entry = _strict_keys(tensors[name], metadata_keys, f"tensor {name!r}")
        shape = entry["shape"]
        if not isinstance(shape, list) or tuple(shape) != shapes[name]:
            raise Qwen3TiedDepthFormatError(
                f"Tensor {name!r} has the wrong manifest shape."
            )
        dtype = _dtype_from_name(entry["dtype"])
        numel = _format_integer(entry["numel"], f"{name}.numel")
        expected_numel = 1
        for dimension in shape:
            expected_numel *= _format_integer(
                dimension,
                f"{name}.shape",
                minimum=1,
            )
        nbytes = _format_integer(entry["nbytes"], f"{name}.nbytes", minimum=1)
        digest = entry["sha256"]
        if (
            numel != expected_numel
            or nbytes != numel * _dtype_nbytes(dtype)
            or not isinstance(digest, str)
            or len(digest) != 64
            or entry["is_parameter"] is not True
        ):
            raise Qwen3TiedDepthFormatError(
                f"Tensor metadata for {name!r} is internally inconsistent."
            )
        if tensor_format == RAW_TENSOR_FORMAT:
            offset = _format_integer(entry["byte_offset"], f"{name}.byte_offset")
            if offset != expected_offset:
                raise Qwen3TiedDepthFormatError(
                    "Raw tensor ranges must be sorted, contiguous, and non-overlapping."
                )
            expected_offset += nbytes
        payload_bytes += nbytes
        largest_bytes = max(largest_bytes, nbytes)
    for canonical in aliases.values():
        logical_parameters += int(tensors[canonical]["numel"])
    if (
        payload_bytes <= 0
        or sum(int(tensors[name]["numel"]) for name in tensors)
        != plan.unique_parameter_count
        or logical_parameters != plan.logical_parameter_count
    ):
        raise Qwen3TiedDepthFormatError(
            "Tensor metadata counts disagree with the parameter plan."
        )
    if tensor_format == RAW_TENSOR_FORMAT and payload_bytes != tensor_artifact_size:
        raise Qwen3TiedDepthFormatError(
            "Raw payload size disagrees with its tensor ranges."
        )

    reload_memory = _strict_keys(
        manifest.get("reload_memory"),
        {
            "topology_construction_device",
            "streamed_physical_tensors",
            "model_parameter_bytes",
            "largest_staging_tensor_bytes",
            "bounded_peak_parameter_and_staging_bytes",
        },
        "reload_memory",
    )
    if (
        reload_memory["topology_construction_device"] != "meta"
        or reload_memory["streamed_physical_tensors"] is not True
        or reload_memory["model_parameter_bytes"] != payload_bytes
        or reload_memory["largest_staging_tensor_bytes"] != largest_bytes
        or reload_memory["bounded_peak_parameter_and_staging_bytes"]
        != payload_bytes + largest_bytes
    ):
        raise Qwen3TiedDepthFormatError("Reload memory accounting is inconsistent.")
    return aliases, requires_grad, payload_bytes, largest_bytes


def _build_meta_topology(
    config_data: Mapping[str, Any],
    generation_data: Mapping[str, Any],
    plan: Qwen3TiedDepthCheckpointPlan,
    attention_source_layer: int,
    mlp_source_layer: int,
    target_device: torch.device,
    aliases: Mapping[str, str],
    model_training: bool,
) -> nn.Module:
    try:
        from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM
    except ImportError as exc:
        raise Qwen3TiedDepthCheckpointError(
            "Loading requires Transformers with Qwen3 support."
        ) from exc
    config = Qwen3Config.from_dict(dict(config_data))
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    rotary_type = type(model.model.rotary_emb)
    model.model.rotary_emb = rotary_type(config, device=target_device)
    model.generation_config = GenerationConfig.from_dict(dict(generation_data))
    layers = _model_layers(model)
    for group, roles, source_index in (
        ("self_attn", ATTENTION_PROJECTIONS, attention_source_layer),
        ("mlp", MLP_PROJECTIONS, mlp_source_layer),
    ):
        source_owner = getattr(layers[source_index], group)
        for role in roles:
            source = getattr(source_owner, role).weight
            for layer in layers:
                getattr(getattr(layer, group), role).weight = source
    model.get_output_embeddings().weight = model.get_input_embeddings().weight
    _assert_state_surface(model, aliases)
    unique, logical = _parameter_counts_meta(model)
    if unique != plan.unique_parameter_count or logical != plan.logical_parameter_count:
        raise Qwen3TiedDepthFormatError(
            "Meta topology parameter counts disagree with the manifest plan."
        )
    model.train(model_training)
    return model


def _parameter_counts_meta(model: nn.Module) -> tuple[int, int]:
    unique_ids: dict[int, nn.Parameter] = {}
    logical = 0
    for _, parameter in model.named_parameters(
        recurse=True,
        remove_duplicate=False,
    ):
        logical += int(parameter.numel())
        unique_ids[id(parameter)] = parameter
    return sum(int(value.numel()) for value in unique_ids.values()), logical


def _validate_loaded_tensor(
    name: str,
    tensor: Tensor,
    entry: Mapping[str, Any],
) -> None:
    if (
        tensor.device.type != "cpu"
        or tensor.layout != torch.strided
        or not tensor.is_contiguous()
        or list(tensor.shape) != entry["shape"]
        or str(tensor.dtype) != entry["dtype"]
        or tensor.numel() != entry["numel"]
        or tensor.numel() * tensor.element_size() != entry["nbytes"]
    ):
        raise Qwen3TiedDepthFormatError(
            f"Payload tensor {name!r} disagrees with its metadata."
        )
    _, raw = _tensor_cpu_view(tensor)
    if _sha256_bytes(raw) != entry["sha256"]:
        raise Qwen3TiedDepthFormatError(f"Payload tensor {name!r} failed SHA-256.")


def _install_parameter_group(
    model: nn.Module,
    logical_keys: Sequence[str],
    tensor: Tensor,
    requires_grad: Mapping[str, bool],
) -> None:
    if not logical_keys:
        raise Qwen3TiedDepthFormatError("Canonical tensor has no logical aliases.")
    required = requires_grad[logical_keys[0]]
    parameter = nn.Parameter(tensor, requires_grad=required)
    for state_key in logical_keys:
        if requires_grad[state_key] != required:
            raise Qwen3TiedDepthFormatError(
                "Alias group changed requires-grad during reload."
            )
        owner_path, _, name = state_key.rpartition(".")
        owner: Any = model
        for part in owner_path.split("."):
            owner = getattr(owner, part)
        setattr(owner, name, parameter)


def _materialize_safetensors(
    path: Path,
    model: nn.Module,
    aliases: Mapping[str, str],
    requires_grad: Mapping[str, bool],
    tensor_metadata: Mapping[str, Any],
    target_device: torch.device,
) -> None:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise Qwen3TiedDepthCheckpointError(
            "This artifact requires the safetensors package."
        ) from exc
    logical_by_tensor: dict[str, list[str]] = {name: [] for name in tensor_metadata}
    for logical_name, canonical_name in aliases.items():
        logical_by_tensor[canonical_name].append(logical_name)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(tensor_metadata):
            raise Qwen3TiedDepthFormatError(
                "Safetensors keys disagree with the tensor manifest."
            )
        if handle.metadata() != {"format": CHECKPOINT_SCHEMA}:
            raise Qwen3TiedDepthFormatError(
                "Safetensors format metadata is missing or invalid."
            )
        for name in sorted(tensor_metadata):
            source = handle.get_tensor(name)
            entry = tensor_metadata[name]
            _validate_loaded_tensor(name, source, entry)
            materialized = (
                source.clone()
                if target_device.type == "cpu"
                else source.to(device=target_device, copy=True)
            )
            _install_parameter_group(
                model,
                logical_by_tensor[name],
                materialized,
                requires_grad,
            )


def _materialize_raw(
    path: Path,
    model: nn.Module,
    aliases: Mapping[str, str],
    requires_grad: Mapping[str, bool],
    tensor_metadata: Mapping[str, Any],
    target_device: torch.device,
) -> None:
    logical_by_tensor: dict[str, list[str]] = {name: [] for name in tensor_metadata}
    for logical_name, canonical_name in aliases.items():
        logical_by_tensor[canonical_name].append(logical_name)
    with path.open("rb", buffering=0) as handle:
        for name in sorted(tensor_metadata):
            entry = tensor_metadata[name]
            nbytes = int(entry["nbytes"])
            buffer = bytearray(nbytes)
            view = memoryview(buffer)
            handle.seek(int(entry["byte_offset"]))
            if handle.readinto(view) != nbytes:
                raise Qwen3TiedDepthFormatError(
                    f"Raw tensor {name!r} ended unexpectedly."
                )
            if _sha256_bytes(view) != entry["sha256"]:
                raise Qwen3TiedDepthFormatError(f"Raw tensor {name!r} failed SHA-256.")
            source = torch.frombuffer(
                buffer,
                dtype=_dtype_from_name(entry["dtype"]),
            ).reshape(tuple(entry["shape"]))
            _validate_loaded_tensor(name, source, entry)
            materialized = (
                source
                if target_device.type == "cpu"
                else source.to(device=target_device, copy=True)
            )
            _install_parameter_group(
                model,
                logical_by_tensor[name],
                materialized,
                requires_grad,
            )


def load_qwen3_tied_depth_checkpoint(
    checkpoint_directory: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
    max_parameter_count: int | None = None,
    max_model_bytes: int | None = None,
    max_tensor_bytes: int | None = None,
    max_memory_bytes: int | None = None,
) -> nn.Module:
    """Strictly reload a parent-free tied-depth Qwen3 with bounded staging."""

    root = Path(checkpoint_directory).expanduser().resolve()
    if not root.is_dir():
        raise Qwen3TiedDepthFormatError(f"Checkpoint directory {root} does not exist.")
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise Qwen3TiedDepthFormatError(f"Missing {MANIFEST_FILENAME}.")
    manifest = _read_json(manifest_path)
    expected_top_keys = {
        "schema",
        "serialization",
        "artifacts",
        "topology",
        "vocabulary",
        "parameter_counts",
        "tied_depth_report",
        "state_dict",
        "tensors",
        "reload_memory",
        "model_training",
        "standalone",
        "parent_model_required",
        "external_weight_dependencies",
        _MANIFEST_HASH_KEY,
    }
    _strict_keys(manifest, expected_top_keys, "manifest")
    if manifest["schema"] != CHECKPOINT_SCHEMA:
        raise Qwen3TiedDepthFormatError("Unsupported checkpoint schema.")
    claimed_hash = manifest[_MANIFEST_HASH_KEY]
    hash_payload = dict(manifest)
    hash_payload.pop(_MANIFEST_HASH_KEY)
    if (
        not isinstance(claimed_hash, str)
        or len(claimed_hash) != 64
        or claimed_hash != _sha256_bytes(_canonical_json_bytes(hash_payload))
    ):
        raise Qwen3TiedDepthFormatError("Manifest SHA-256 mismatch.")
    serialization = _strict_keys(
        manifest["serialization"],
        {
            "python_pickle",
            "serialized_modules",
            "serialized_weakrefs",
            "tensor_format",
            "byte_order",
        },
        "serialization",
    )
    if any(
        serialization[name] is not False
        for name in ("python_pickle", "serialized_modules", "serialized_weakrefs")
    ):
        raise Qwen3TiedDepthFormatError(
            "Manifest permits unsafe Python object serialization."
        )
    tensor_format = serialization["tensor_format"]
    if tensor_format not in {"safetensors", RAW_TENSOR_FORMAT}:
        raise Qwen3TiedDepthFormatError(f"Unsupported tensor format {tensor_format!r}.")
    if serialization["byte_order"] not in {"little", "big"} or (
        tensor_format == RAW_TENSOR_FORMAT
        and serialization["byte_order"] != sys.byteorder
    ):
        raise Qwen3TiedDepthFormatError(
            "Raw tensor byte order differs from this runtime."
        )
    if (
        manifest["standalone"] is not True
        or manifest["parent_model_required"] is not False
        or manifest["external_weight_dependencies"] != []
    ):
        raise Qwen3TiedDepthFormatError(
            "Manifest does not describe a parent-free standalone checkpoint."
        )
    if not isinstance(manifest["model_training"], bool):
        raise Qwen3TiedDepthFormatError("model_training must be boolean.")

    artifacts = _strict_keys(
        manifest["artifacts"],
        {"config", "generation_config", "tensors"},
        "artifacts",
    )
    config_path = _checked_artifact_path(root, artifacts["config"], "config")
    generation_path = _checked_artifact_path(
        root,
        artifacts["generation_config"],
        "generation_config",
    )
    tensor_path = _checked_artifact_path(root, artifacts["tensors"], "tensors")
    expected_files = {
        MANIFEST_FILENAME,
        config_path.name,
        generation_path.name,
        tensor_path.name,
    }
    actual_entries = {path.name for path in root.iterdir()}
    if actual_entries != expected_files:
        raise Qwen3TiedDepthFormatError(
            "Checkpoint directory contains missing or unclaimed artifacts."
        )
    config_data = _read_json(config_path)
    generation_data = _read_json(generation_path)
    if config_data.get("_name_or_path") not in {None, ""} or "auto_map" in config_data:
        raise Qwen3TiedDepthFormatError(
            "Standalone config contains a parent or remote-code reference."
        )
    source_vocab, target_vocab, _, token_fingerprint = _validate_vocabulary_manifest(
        manifest["vocabulary"], config_data
    )
    plan = _validate_plan_counts(
        config_data,
        manifest,
        source_vocab,
        target_vocab,
    )
    attention_index, mlp_index = _validate_topology_manifest(
        manifest["topology"],
        plan,
    )
    expected_report = _expected_report(
        plan,
        attention_index,
        mlp_index,
        token_fingerprint,
    )
    _validate_report_manifest(manifest["tied_depth_report"], expected_report)
    dimensions = _qwen3_dimensions(config_data)
    aliases, requires_grad, model_bytes, largest_tensor_bytes = (
        _validate_state_and_tensors(
            manifest,
            plan,
            dimensions,
            attention_index,
            mlp_index,
            tensor_format,
            tensor_path.stat().st_size,
        )
    )

    parameter_limit = _checked_limit(max_parameter_count, "max_parameter_count")
    model_byte_limit = _checked_limit(max_model_bytes, "max_model_bytes")
    tensor_byte_limit = _checked_limit(max_tensor_bytes, "max_tensor_bytes")
    memory_limit = _checked_limit(max_memory_bytes, "max_memory_bytes")
    limits = (
        (parameter_limit, plan.unique_parameter_count, "max_parameter_count"),
        (model_byte_limit, model_bytes, "max_model_bytes"),
        (tensor_byte_limit, largest_tensor_bytes, "max_tensor_bytes"),
        (memory_limit, model_bytes + largest_tensor_bytes, "max_memory_bytes"),
    )
    for limit, required, name in limits:
        if limit is not None and required > limit:
            raise Qwen3TiedDepthBudgetError(
                f"Checkpoint requires {required} for {name}, exceeding {limit}."
            )

    target_device = torch.device(device)
    if target_device.type == "meta":
        raise Qwen3TiedDepthCheckpointError(
            "device='meta' is not a materialized reload target."
        )
    model = _build_meta_topology(
        config_data,
        generation_data,
        plan,
        attention_index,
        mlp_index,
        target_device,
        aliases,
        manifest["model_training"],
    )
    tensor_metadata = manifest["tensors"]
    if tensor_format == "safetensors":
        _materialize_safetensors(
            tensor_path,
            model,
            aliases,
            requires_grad,
            tensor_metadata,
            target_device,
        )
    else:
        _materialize_raw(
            tensor_path,
            model,
            aliases,
            requires_grad,
            tensor_metadata,
            target_device,
        )
    if any(parameter.is_meta for parameter in model.parameters()) or any(
        buffer.is_meta for buffer in model.buffers()
    ):
        raise Qwen3TiedDepthFormatError(
            "Reload left meta tensors in the materialized model."
        )
    setattr(model, "neural_forge_tied_depth_report", expected_report)
    _validate_prepared_topology(
        model,
        plan,
        attention_index,
        mlp_index,
        expected_report=expected_report,
    )
    setattr(model, "neural_forge_checkpoint_manifest", manifest)
    setattr(
        model,
        "neural_forge_load_report",
        {
            "missing_keys": [],
            "unexpected_keys": [],
            "stored_tensor_count": len(tensor_metadata),
            "logical_state_tensor_count": len(aliases),
            "manifest_sha256": claimed_hash,
            "constructed_on_meta": True,
            "streamed_physical_tensors": True,
            "model_parameter_bytes": model_bytes,
            "largest_staging_tensor_bytes": largest_tensor_bytes,
            "bounded_peak_parameter_and_staging_bytes": (
                model_bytes + largest_tensor_bytes
            ),
        },
    )
    return model


export_qwen_tied_depth_checkpoint = export_qwen3_tied_depth_checkpoint
load_qwen_tied_depth_checkpoint = load_qwen3_tied_depth_checkpoint
plan_qwen_tied_depth_checkpoint = plan_qwen3_tied_depth_checkpoint
count_qwen_tied_depth_parameters = count_qwen3_tied_depth_parameters


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CONFIG_FILENAME",
    "GENERATION_CONFIG_FILENAME",
    "MANIFEST_FILENAME",
    "RAW_TENSORS_FILENAME",
    "SAFETENSORS_FILENAME",
    "Qwen3TiedDepthBudgetError",
    "Qwen3TiedDepthCheckpointError",
    "Qwen3TiedDepthCheckpointPlan",
    "Qwen3TiedDepthFormatError",
    "count_qwen3_tied_depth_parameters",
    "count_qwen_tied_depth_parameters",
    "export_qwen3_tied_depth_checkpoint",
    "export_qwen_tied_depth_checkpoint",
    "load_qwen3_tied_depth_checkpoint",
    "load_qwen_tied_depth_checkpoint",
    "plan_qwen3_tied_depth_checkpoint",
    "plan_qwen_tied_depth_checkpoint",
]
