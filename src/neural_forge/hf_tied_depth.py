from __future__ import annotations

import hashlib
import json
import operator
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


class TiedDepthError(ValueError):
    """Raised when an exact projection-sharing fold cannot be constructed."""


@dataclass(frozen=True)
class TiedDepthReport:
    schema: str
    method: str
    logical_layers: int
    attention_source_layer: int
    mlp_source_layer: int
    source_vocab_size: int
    target_vocab_size: int
    token_row_fingerprint: str
    source_unique_parameter_count: int
    source_logical_parameter_count: int
    child_unique_parameter_count: int
    child_logical_parameter_count: int
    unique_compression_ratio: float
    retained_layer_specific_norm_parameter_count: int
    shared_projection_parameter_count: int
    tied_projection_alias_count: int
    zero_gradient: bool
    full_logical_depth: bool
    exact_parent_token_rows: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TiedDepthError(f"{name} must be an integer.")
    try:
        return operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TiedDepthError(f"{name} must be an integer.") from exc


def _parameter_counts(model: nn.Module) -> tuple[int, int]:
    unique = sum(parameter.numel() for parameter in model.parameters())
    logical = sum(
        parameter.numel()
        for _, parameter in model.named_parameters(remove_duplicate=False)
    )
    return int(unique), int(logical)


def _raw_token(tokenizer: Any, token_id: int) -> str:
    try:
        token = tokenizer.convert_ids_to_tokens(token_id)
    except Exception as exc:
        raise TiedDepthError(
            f"Tokenizer could not resolve token ID {token_id}."
        ) from exc
    if not isinstance(token, str):
        raise TiedDepthError(f"Tokenizer returned no raw token for ID {token_id}.")
    return token


def exact_token_row_mapping(
    donor_tokenizer: Any,
    target_tokenizer: Any,
) -> tuple[int, ...]:
    """Map every target raw token to exactly one donor vocabulary row."""

    try:
        donor_vocab = donor_tokenizer.get_vocab()
        target_size = len(target_tokenizer)
    except Exception as exc:
        raise TiedDepthError(
            "Both tokenizers must expose get_vocab() and __len__()."
        ) from exc
    if not isinstance(donor_vocab, dict) or not donor_vocab or target_size <= 0:
        raise TiedDepthError("Tokenizer vocabularies must be non-empty.")

    mapping: list[int] = []
    seen: set[int] = set()
    for target_id in range(target_size):
        token = _raw_token(target_tokenizer, target_id)
        donor_id = donor_vocab.get(token)
        if isinstance(donor_id, bool) or not isinstance(donor_id, int):
            raise TiedDepthError(f"Target raw token {token!r} has no exact donor row.")
        if donor_id in seen:
            raise TiedDepthError(
                f"Target tokenizer maps multiple rows to donor row {donor_id}."
            )
        if _raw_token(donor_tokenizer, donor_id) != token:
            raise TiedDepthError(
                f"Donor row {donor_id} does not round-trip raw token {token!r}."
            )
        seen.add(donor_id)
        mapping.append(donor_id)
    return tuple(mapping)


def _mapping_fingerprint(mapping: tuple[int, ...]) -> str:
    payload = json.dumps(
        mapping,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _model_layers(model: nn.Module) -> nn.ModuleList:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, nn.ModuleList) or not layers:
        raise TiedDepthError("Model must expose a non-empty model.layers ModuleList.")
    return layers


def _layer_index(value: Any, name: str, layer_count: int) -> int:
    result = _integer(value, name)
    if not 0 <= result < layer_count:
        raise TiedDepthError(f"{name} must be in [0, {layer_count - 1}].")
    return result


def _projection(layer: nn.Module, group: str, role: str) -> nn.Linear:
    owner = getattr(layer, group, None)
    value = getattr(owner, role, None)
    if not isinstance(value, nn.Linear):
        raise TiedDepthError(f"Layer lacks nn.Linear {group}.{role}.")
    if value.bias is not None:
        raise TiedDepthError(f"{group}.{role} must be bias-free for exact tying.")
    return value


def _special_token_ids(target_tokenizer: Any) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(target_tokenizer, name, None)
        if value is not None:
            value = _integer(value, name)
        result[name] = value
    return result


def _resize_vocabulary_exactly(
    model: nn.Module,
    donor_rows: tuple[int, ...],
    target_tokenizer: Any,
) -> None:
    embedding = model.get_input_embeddings()
    output = model.get_output_embeddings()
    if not isinstance(embedding, nn.Embedding) or not isinstance(output, nn.Linear):
        raise TiedDepthError(
            "Model must expose nn.Embedding input and nn.Linear output modules."
        )
    if embedding.weight is not output.weight:
        raise TiedDepthError("Source input and output vocabulary weights must be tied.")
    if max(donor_rows) >= embedding.num_embeddings:
        raise TiedDepthError("Token row mapping exceeds the source embedding table.")

    row_indices = torch.tensor(
        donor_rows,
        device=embedding.weight.device,
        dtype=torch.long,
    )
    selected = embedding.weight.detach().index_select(0, row_indices).clone()
    target_embedding = nn.Embedding(
        len(donor_rows),
        embedding.embedding_dim,
        padding_idx=getattr(target_tokenizer, "pad_token_id", None),
        max_norm=embedding.max_norm,
        norm_type=embedding.norm_type,
        scale_grad_by_freq=embedding.scale_grad_by_freq,
        sparse=embedding.sparse,
        device=selected.device,
        dtype=selected.dtype,
    )
    target_embedding.weight = nn.Parameter(selected)
    target_output = nn.Linear(
        output.in_features,
        len(donor_rows),
        bias=False,
        device=selected.device,
        dtype=selected.dtype,
    )
    target_output.weight = target_embedding.weight
    target_embedding.train(embedding.training)
    target_output.train(output.training)
    model.set_input_embeddings(target_embedding)
    model.set_output_embeddings(target_output)

    model.config.vocab_size = len(donor_rows)
    model.config.tie_word_embeddings = True
    backbone = getattr(model, "model", None)
    if hasattr(backbone, "vocab_size"):
        backbone.vocab_size = len(donor_rows)
    if hasattr(model, "vocab_size"):
        model.vocab_size = len(donor_rows)
    for name, value in _special_token_ids(target_tokenizer).items():
        setattr(model.config, name, value)
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            setattr(generation_config, name, value)


def prepare_qwen3_tied_depth_model(
    model: nn.Module,
    donor_tokenizer: Any,
    target_tokenizer: Any,
    *,
    attention_source_layer: int,
    mlp_source_layer: int,
) -> TiedDepthReport:
    """Fold Qwen3 depth by sharing complete real projection sets.

    Every logical decoder layer remains active. Attention and gated-MLP
    projections are aliased to selected real parent layers as coupled groups;
    all per-layer input, post-attention, Q, and K norms remain layer-specific.
    The vocabulary is replaced with exact donor rows for a strict tokenizer
    subset. No optimization or synthetic target weights are involved.
    """

    if not isinstance(model, nn.Module) or not hasattr(model, "config"):
        raise TiedDepthError("model must be a configured nn.Module.")
    if getattr(model, "neural_forge_tied_depth_report", None) is not None:
        raise TiedDepthError("Model is already prepared as a tied-depth child.")
    if getattr(model.config, "model_type", None) != "qwen3":
        raise TiedDepthError("This proof lane currently requires model_type='qwen3'.")
    layers = _model_layers(model)
    layer_count = len(layers)
    attention_index = _layer_index(
        attention_source_layer,
        "attention_source_layer",
        layer_count,
    )
    mlp_index = _layer_index(mlp_source_layer, "mlp_source_layer", layer_count)
    source_unique, source_logical = _parameter_counts(model)
    source_vocab_size = int(model.get_input_embeddings().num_embeddings)
    donor_rows = exact_token_row_mapping(donor_tokenizer, target_tokenizer)

    attention_source = layers[attention_index]
    mlp_source = layers[mlp_index]
    shared_parameter_ids: set[int] = set()
    alias_count = 0
    with torch.no_grad():
        for layer in layers:
            for role in ATTENTION_PROJECTIONS:
                source = _projection(attention_source, "self_attn", role)
                target = _projection(layer, "self_attn", role)
                target.weight = source.weight
                shared_parameter_ids.add(id(source.weight))
                alias_count += 1
            for role in MLP_PROJECTIONS:
                source = _projection(mlp_source, "mlp", role)
                target = _projection(layer, "mlp", role)
                target.weight = source.weight
                shared_parameter_ids.add(id(source.weight))
                alias_count += 1
        _resize_vocabulary_exactly(model, donor_rows, target_tokenizer)

    child_unique, child_logical = _parameter_counts(model)
    shared_projection_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if id(parameter) in shared_parameter_ids
    )
    embedding_parameters = model.get_input_embeddings().weight.numel()
    retained_norm_parameters = (
        child_unique - shared_projection_parameters - embedding_parameters
    )
    if retained_norm_parameters < 0:
        raise TiedDepthError("Tied-depth parameter accounting became negative.")
    report = TiedDepthReport(
        schema="neural-forge-qwen3-tied-depth-v1",
        method="exact-token-rows-plus-coupled-real-projection-sharing",
        logical_layers=layer_count,
        attention_source_layer=attention_index,
        mlp_source_layer=mlp_index,
        source_vocab_size=source_vocab_size,
        target_vocab_size=len(donor_rows),
        token_row_fingerprint=_mapping_fingerprint(donor_rows),
        source_unique_parameter_count=source_unique,
        source_logical_parameter_count=source_logical,
        child_unique_parameter_count=child_unique,
        child_logical_parameter_count=child_logical,
        unique_compression_ratio=source_unique / child_unique,
        retained_layer_specific_norm_parameter_count=retained_norm_parameters,
        shared_projection_parameter_count=shared_projection_parameters,
        tied_projection_alias_count=alias_count,
        zero_gradient=True,
        full_logical_depth=True,
        exact_parent_token_rows=True,
    )
    setattr(model, "neural_forge_tied_depth_report", report)
    return report


__all__ = [
    "ATTENTION_PROJECTIONS",
    "MLP_PROJECTIONS",
    "TiedDepthError",
    "TiedDepthReport",
    "exact_token_row_mapping",
    "prepare_qwen3_tied_depth_model",
]
