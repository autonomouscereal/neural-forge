"""Standalone, alias-aware checkpoints for partially shared LFM2 FFNs.

The checkpoint stores every physical parameter exactly once and records a
complete mapping from the ordinary Transformers LFM2 state surface to those
stored tensors.  Reload constructs the ordinary ``Lfm2ForCausalLM`` topology
on ``meta``, restores arbitrary disjoint FFN sharing groups by exact
``Parameter`` identity, and streams the physical tensors into that topology.

No Python objects, parent model, tokenizer, adapter, or sidecar weights are
needed to reload the artifact.
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


CHECKPOINT_SCHEMA = "neural-forge-lfm2-shared-ffn-checkpoint-v1"
MANIFEST_FILENAME = "neural_forge_lfm2_shared_manifest.json"
CONFIG_FILENAME = "config.json"
GENERATION_CONFIG_FILENAME = "generation_config.json"
SAFETENSORS_FILENAME = "model.safetensors"
RAW_TENSORS_FILENAME = "model.nfraw"
RAW_TENSOR_FORMAT = "neural-forge-raw-v1"
FFN_PROJECTION_ROLES = ("w1", "w3", "w2")
_MANIFEST_HASH_KEY = "manifest_sha256"
_LFM2_MODEL_MODULE = "transformers.models.lfm2.modeling_lfm2"
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


class Lfm2SharedCheckpointError(ValueError):
    """Base error for invalid source models, options, or artifacts."""


class Lfm2SharedCheckpointBudgetError(Lfm2SharedCheckpointError):
    """A requested parameter or memory cap cannot contain the artifact."""


class Lfm2SharedCheckpointFormatError(Lfm2SharedCheckpointError):
    """A checkpoint is incomplete, inconsistent, or corrupted."""


@dataclass(frozen=True, slots=True)
class Lfm2SharedCheckpointPlan:
    """Exact physical/logical accounting for an already-shared LFM2."""

    logical_layer_count: int
    shared_ffn_layer_groups: tuple[tuple[int, ...], ...]
    unique_parameter_count: int
    logical_parameter_count: int
    ffn_shared_parameter_savings: int
    vocabulary_shared_parameter_savings: int
    parameter_budget: int | None
    budget_headroom: int | None

    @property
    def total_parameter_alias_savings(self) -> int:
        return self.logical_parameter_count - self.unique_parameter_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "logical_layer_count": self.logical_layer_count,
            "shared_ffn_layer_groups": [
                list(group) for group in self.shared_ffn_layer_groups
            ],
            "unique_parameter_count": self.unique_parameter_count,
            "logical_parameter_count": self.logical_parameter_count,
            "ffn_shared_parameter_savings": self.ffn_shared_parameter_savings,
            "vocabulary_shared_parameter_savings": (
                self.vocabulary_shared_parameter_savings
            ),
            "total_parameter_alias_savings": self.total_parameter_alias_savings,
            "parameter_budget": self.parameter_budget,
            "budget_headroom": self.budget_headroom,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise Lfm2SharedCheckpointError(f"{name} must be an integer.")
    try:
        return operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2SharedCheckpointError(f"{name} must be an integer.") from exc


def _optional_nonnegative_integer(value: Any, name: str) -> int | None:
    if value is None:
        return None
    result = _integer(value, name)
    if result < 0:
        raise Lfm2SharedCheckpointError(f"{name} must be nonnegative or None.")
    return result


def _model_layers(model: nn.Module) -> nn.ModuleList:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, nn.ModuleList) or not layers:
        raise Lfm2SharedCheckpointError(
            "LFM2 model must expose a non-empty model.layers ModuleList."
        )
    return layers


def _validate_lfm2_model(model: nn.Module) -> nn.ModuleList:
    if not isinstance(model, nn.Module) or not hasattr(model, "config"):
        raise Lfm2SharedCheckpointError(
            "model must be a configured torch.nn.Module."
        )
    model_type = type(model)
    if (
        model_type.__name__ != "Lfm2ForCausalLM"
        or model_type.__module__ != _LFM2_MODEL_MODULE
        or getattr(model.config, "model_type", None) != "lfm2"
    ):
        raise Lfm2SharedCheckpointError(
            "Shared FFN checkpoints require Transformers Lfm2ForCausalLM."
        )
    if getattr(model.config, "tie_word_embeddings", None) is not True:
        raise Lfm2SharedCheckpointError(
            "LFM2 shared checkpoints require tied word embeddings."
        )
    layers = _model_layers(model)
    configured_layers = getattr(model.config, "num_hidden_layers", None)
    layer_types = getattr(model.config, "layer_types", None)
    if (
        configured_layers != len(layers)
        or not isinstance(layer_types, (list, tuple))
        or len(layer_types) != len(layers)
        or any(value not in {"conv", "full_attention"} for value in layer_types)
    ):
        raise Lfm2SharedCheckpointError(
            "LFM2 config depth/layer_types disagree with the instantiated model."
        )
    embedding = model.get_input_embeddings()
    output = model.get_output_embeddings()
    if (
        not isinstance(embedding, nn.Embedding)
        or not isinstance(output, nn.Linear)
        or embedding.weight is not output.weight
    ):
        raise Lfm2SharedCheckpointError(
            "Input and output embeddings must share exact Parameter identity."
        )
    for index, layer in enumerate(layers):
        feed_forward = getattr(layer, "feed_forward", None)
        if feed_forward is None:
            raise Lfm2SharedCheckpointError(
                f"LFM2 layer {index} has no feed_forward module."
            )
        for role in FFN_PROJECTION_ROLES:
            projection = getattr(feed_forward, role, None)
            if (
                not isinstance(projection, nn.Linear)
                or not isinstance(projection.weight, nn.Parameter)
                or projection.bias is not None
            ):
                raise Lfm2SharedCheckpointError(
                    f"LFM2 layer {index} FFN role {role} is not a bias-free Linear."
                )
    return layers


def _normalize_groups(
    groups: Sequence[Sequence[int]],
    layer_count: int,
    *,
    error_type: type[Lfm2SharedCheckpointError] = Lfm2SharedCheckpointError,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
        raise error_type("shared_ffn_layer_groups must be a sequence of groups.")
    normalized: list[tuple[int, ...]] = []
    occupied: set[int] = set()
    for group_index, values in enumerate(groups):
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise error_type(
                f"shared_ffn_layer_groups[{group_index}] must be a sequence."
            )
        group = tuple(values)
        if len(group) < 2:
            raise error_type("Every shared FFN group must contain at least two layers.")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in group):
            raise error_type("Shared FFN layer indices must be integers.")
        if tuple(sorted(group)) != group or len(set(group)) != len(group):
            raise error_type(
                "Every shared FFN group must be strictly increasing and unique."
            )
        if any(value < 0 or value >= layer_count for value in group):
            raise error_type("Shared FFN group contains an out-of-range layer index.")
        overlap = occupied.intersection(group)
        if overlap:
            raise error_type(
                "Shared FFN layer groups must be disjoint; overlap at "
                f"{sorted(overlap)!r}."
            )
        occupied.update(group)
        normalized.append(group)
    ordered = tuple(sorted(normalized, key=lambda value: value[0]))
    if tuple(normalized) != ordered:
        raise error_type("Shared FFN groups must be ordered by first layer index.")
    return ordered


def _identity_partition(
    layers: Sequence[nn.Module],
    role: str,
) -> tuple[tuple[int, ...], ...]:
    buckets: dict[int, list[int]] = {}
    for index, layer in enumerate(layers):
        parameter = getattr(layer.feed_forward, role).weight
        buckets.setdefault(id(parameter), []).append(index)
    return tuple(
        sorted(
            (tuple(indices) for indices in buckets.values() if len(indices) > 1),
            key=lambda value: value[0],
        )
    )


def _infer_and_validate_groups(
    model: nn.Module,
    requested: Sequence[Sequence[int]] | None,
) -> tuple[tuple[int, ...], ...]:
    layers = _validate_lfm2_model(model)
    partitions = {
        role: _identity_partition(layers, role) for role in FFN_PROJECTION_ROLES
    }
    first = partitions[FFN_PROJECTION_ROLES[0]]
    if any(partition != first for partition in partitions.values()):
        raise Lfm2SharedCheckpointError(
            "Partial FFN aliasing is forbidden: w1, w3, and w2 must declare "
            "the same complete layer partition."
        )
    actual = _normalize_groups(first, len(layers))
    if requested is not None:
        declared = _normalize_groups(requested, len(layers))
        if declared != actual:
            raise Lfm2SharedCheckpointError(
                "Declared shared_ffn_layer_groups differ from exact live "
                "Parameter identity."
            )
    else:
        declared = actual

    # A weight may be shared only with the same role in its declared group.
    owners: dict[int, list[tuple[int, str]]] = {}
    for layer_index, layer in enumerate(layers):
        for role in FFN_PROJECTION_ROLES:
            parameter = getattr(layer.feed_forward, role).weight
            owners.setdefault(id(parameter), []).append((layer_index, role))
    for aliases in owners.values():
        if len({role for _, role in aliases}) != 1:
            raise Lfm2SharedCheckpointError(
                "Different FFN projection roles cannot share one Parameter."
            )
    return declared


def _ffn_key(layer_index: int, role: str) -> str:
    return f"model.layers.{layer_index}.feed_forward.{role}.weight"


def _expected_aliases(
    logical_names: Sequence[str],
    groups: Sequence[Sequence[int]],
    *,
    error_type: type[Lfm2SharedCheckpointError] = Lfm2SharedCheckpointError,
) -> dict[str, str]:
    aliases = {name: name for name in logical_names}
    required = {"model.embed_tokens.weight", "lm_head.weight"}
    if not required.issubset(aliases):
        raise error_type(
            "LFM2 state surface lacks tied input/output embedding weights."
        )
    aliases["lm_head.weight"] = "model.embed_tokens.weight"
    for group in groups:
        source = group[0]
        for role in FFN_PROJECTION_ROLES:
            canonical = _ffn_key(source, role)
            if canonical not in aliases:
                raise error_type(
                    f"LFM2 state surface lacks expected FFN tensor {canonical!r}."
                )
            for layer_index in group:
                logical = _ffn_key(layer_index, role)
                if logical not in aliases:
                    raise error_type(
                        f"LFM2 state surface lacks expected FFN tensor {logical!r}."
                    )
                aliases[logical] = canonical
    return aliases


def _storage_key(tensor: Tensor) -> tuple[Any, ...]:
    if tensor.layout != torch.strided or tensor.device.type == "meta":
        raise Lfm2SharedCheckpointError(
            "Checkpoint tensors must be materialized dense strided tensors."
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


def _logical_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    return dict(
        model.named_parameters(
            recurse=True,
            remove_duplicate=False,
        )
    )


def _validate_alias_surface(
    model: nn.Module,
    groups: tuple[tuple[int, ...], ...],
) -> tuple[dict[str, str], dict[str, nn.Parameter]]:
    parameters = _logical_parameters(model)
    state = model.state_dict()
    if set(state) != set(parameters):
        raise Lfm2SharedCheckpointError(
            "Standard LFM2 persistent state must consist only of its parameters."
        )
    logical_names = tuple(state)
    aliases = _expected_aliases(logical_names, groups)

    canonical_parameters: dict[str, nn.Parameter] = {}
    storage_owners: dict[tuple[Any, ...], str] = {}
    for logical, canonical in aliases.items():
        parameter = parameters[logical]
        canonical_parameter = parameters[canonical]
        if parameter is not canonical_parameter:
            raise Lfm2SharedCheckpointError(
                f"Declared alias {logical!r} lacks exact Parameter identity with "
                f"{canonical!r}."
            )
        if not _same_storage(state[logical], state[canonical]):
            raise Lfm2SharedCheckpointError(
                f"Declared alias {logical!r} lacks exact tensor storage identity."
            )
        canonical_parameters.setdefault(canonical, canonical_parameter)

    for canonical, parameter in canonical_parameters.items():
        storage = _storage_key(parameter)
        prior = storage_owners.get(storage)
        if prior is not None and prior != canonical:
            raise Lfm2SharedCheckpointError(
                f"Undeclared storage alias between {prior!r} and {canonical!r}."
            )
        storage_owners[storage] = canonical

    expected_ids: dict[int, str] = {}
    for logical, canonical in aliases.items():
        identifier = id(parameters[logical])
        prior = expected_ids.setdefault(identifier, canonical)
        if prior != canonical:
            raise Lfm2SharedCheckpointError(
                f"Undeclared Parameter alias between {prior!r} and {canonical!r}."
            )
    return aliases, parameters


def _parameter_counts(model: nn.Module) -> tuple[int, int]:
    logical = 0
    unique: dict[int, nn.Parameter] = {}
    for _, parameter in model.named_parameters(
        recurse=True,
        remove_duplicate=False,
    ):
        logical += int(parameter.numel())
        unique[id(parameter)] = parameter
    return sum(int(parameter.numel()) for parameter in unique.values()), logical


def plan_lfm2_shared_checkpoint(
    model: nn.Module,
    shared_ffn_layer_groups: Sequence[Sequence[int]] | None = None,
    *,
    parameter_budget: int | None = None,
) -> Lfm2SharedCheckpointPlan:
    """Validate an already-tied LFM2 and return exact physical/logical counts."""

    groups = _infer_and_validate_groups(model, shared_ffn_layer_groups)
    aliases, parameters = _validate_alias_surface(model, groups)
    unique, logical = _parameter_counts(model)
    canonical_names = set(aliases.values())
    canonical_count = sum(int(parameters[name].numel()) for name in canonical_names)
    if canonical_count != unique:
        raise Lfm2SharedCheckpointError(
            "Alias-derived and live unique parameter counts disagree."
        )
    embedding_savings = int(parameters["model.embed_tokens.weight"].numel())
    ffn_savings = 0
    for group in groups:
        source = group[0]
        group_width = len(group) - 1
        ffn_savings += group_width * sum(
            int(parameters[_ffn_key(source, role)].numel())
            for role in FFN_PROJECTION_ROLES
        )
    if logical - unique != embedding_savings + ffn_savings:
        raise Lfm2SharedCheckpointError(
            "Live alias savings are not explained by embeddings plus declared FFNs."
        )
    budget = _optional_nonnegative_integer(parameter_budget, "parameter_budget")
    if budget is not None and unique > budget:
        raise Lfm2SharedCheckpointBudgetError(
            f"Shared LFM2 checkpoint requires {unique} parameters, exceeding "
            f"parameter_budget={budget}."
        )
    return Lfm2SharedCheckpointPlan(
        logical_layer_count=len(_model_layers(model)),
        shared_ffn_layer_groups=groups,
        unique_parameter_count=unique,
        logical_parameter_count=logical,
        ffn_shared_parameter_savings=ffn_savings,
        vocabulary_shared_parameter_savings=embedding_savings,
        parameter_budget=budget,
        budget_headroom=None if budget is None else budget - unique,
    )


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
        raise Lfm2SharedCheckpointError(
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


def _canonical_state(
    model: nn.Module,
    aliases: Mapping[str, str],
    plan: Lfm2SharedCheckpointPlan,
) -> tuple[
    dict[str, Tensor],
    dict[str, dict[str, Any]],
    dict[str, bool],
]:
    state = model.state_dict()
    parameters = _logical_parameters(model)
    canonical_names = set(aliases.values())
    tensors: dict[str, Tensor] = {}
    metadata: dict[str, dict[str, Any]] = {}
    requires_grad = {
        name: bool(parameter.requires_grad) for name, parameter in parameters.items()
    }
    for canonical in sorted(canonical_names):
        cpu, raw = _tensor_cpu_view(state[canonical])
        tensors[canonical] = cpu
        metadata[canonical] = {
            "shape": list(cpu.shape),
            "dtype": str(cpu.dtype),
            "numel": int(cpu.numel()),
            "nbytes": raw.nbytes,
            "sha256": _sha256_bytes(raw),
            "is_parameter": True,
        }
    for logical, canonical in aliases.items():
        if not _same_storage(state[logical], state[canonical]):
            raise Lfm2SharedCheckpointError(
                f"Alias {logical!r} changed storage during export."
            )
        if requires_grad[logical] != requires_grad[canonical]:
            raise Lfm2SharedCheckpointError(
                f"Alias group for {canonical!r} has mixed requires_grad."
            )
    if (
        sum(int(entry["numel"]) for entry in metadata.values())
        != plan.unique_parameter_count
    ):
        raise Lfm2SharedCheckpointError(
            "Stored tensor count disagrees with the exact parameter plan."
        )
    return tensors, metadata, requires_grad


def _select_tensor_format(requested: str | None) -> str:
    if requested not in {None, "safetensors", "raw"}:
        raise Lfm2SharedCheckpointError(
            "tensor_format must be 'safetensors', 'raw', or None."
        )
    if requested == "raw":
        return "raw"
    try:
        import safetensors.torch  # noqa: F401
    except ImportError:
        if requested == "safetensors":
            raise Lfm2SharedCheckpointError(
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
            if handle.write(raw) != raw.nbytes:
                raise OSError(f"Short raw tensor write for {name!r}.")
            offset += raw.nbytes
        handle.flush()
        os.fsync(handle.fileno())
    return RAW_TENSOR_FORMAT, path


def _sanitized_config(model: nn.Module) -> dict[str, Any]:
    try:
        data = dict(model.config.to_dict())
    except Exception as exc:
        raise Lfm2SharedCheckpointError(
            "LFM2 config must support finite JSON serialization."
        ) from exc
    data["_name_or_path"] = ""
    data.pop("name_or_path", None)
    data.pop("auto_map", None)
    data["architectures"] = ["Lfm2ForCausalLM"]
    data["model_type"] = "lfm2"
    data["tie_word_embeddings"] = True
    return data


def export_lfm2_shared_checkpoint(
    model: nn.Module,
    output_directory: str | os.PathLike[str],
    *,
    shared_ffn_layer_groups: Sequence[Sequence[int]] | None = None,
    parameter_budget: int | None = None,
    max_tensor_payload_bytes: int | None = None,
    tensor_format: str | None = None,
) -> dict[str, Any]:
    """Atomically export an already-tied, parent-free LFM2 checkpoint."""

    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise Lfm2SharedCheckpointError(
            f"Refusing to replace existing checkpoint path {output}."
        )
    selected_format = _select_tensor_format(tensor_format)
    plan = plan_lfm2_shared_checkpoint(
        model,
        shared_ffn_layer_groups,
        parameter_budget=parameter_budget,
    )
    aliases, _ = _validate_alias_surface(model, plan.shared_ffn_layer_groups)
    tensors, tensor_metadata, requires_grad = _canonical_state(model, aliases, plan)
    payload_bytes = sum(int(value["nbytes"]) for value in tensor_metadata.values())
    payload_limit = _optional_nonnegative_integer(
        max_tensor_payload_bytes,
        "max_tensor_payload_bytes",
    )
    if payload_limit is not None and payload_bytes > payload_limit:
        raise Lfm2SharedCheckpointBudgetError(
            f"Shared LFM2 payload requires {payload_bytes} bytes, exceeding "
            f"max_tensor_payload_bytes={payload_limit}."
        )
    largest_tensor_bytes = max(
        int(value["nbytes"]) for value in tensor_metadata.values()
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
        ffn_aliases = sum(
            (len(group) - 1) * len(FFN_PROJECTION_ROLES)
            for group in plan.shared_ffn_layer_groups
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
            "topology": {
                "model_class": "transformers.Lfm2ForCausalLM",
                "model_type": "lfm2",
                "logical_layer_count": plan.logical_layer_count,
                "layer_types": list(model.config.layer_types),
                "shared_ffn_layer_groups": [
                    list(group) for group in plan.shared_ffn_layer_groups
                ],
                "ffn_projection_roles": list(FFN_PROJECTION_ROLES),
                "complete_coupled_ffn_groups": True,
                "exact_parameter_identity_on_reload": True,
                "tied_input_output_embeddings": True,
            },
            "parameter_counts": plan.as_dict(),
            "state_dict": {
                "logical_to_stored_tensor": dict(aliases),
                "logical_tensor_count": len(aliases),
                "stored_tensor_count": len(tensors),
                "alias_count": len(aliases) - len(tensors),
                "ffn_alias_count": ffn_aliases,
                "vocabulary_alias_count": 1,
                "complete_logical_state_mapping": True,
                "parameter_requires_grad": requires_grad,
            },
            "tensors": tensor_metadata,
            "storage": {
                "physical_tensor_bytes": payload_bytes,
                "physical_parameter_bytes": payload_bytes,
                "largest_physical_tensor_bytes": largest_tensor_bytes,
                "tensor_file_bytes": tensor_path.stat().st_size,
            },
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
            "tokenizer_required_for_reload": False,
            "tokenizer_serialized": False,
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
        if temporary.exists() and temporary.resolve().parent == output.parent:
            shutil.rmtree(temporary)
        raise
    return manifest


def _reject_duplicate_json_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Lfm2SharedCheckpointFormatError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda constant: (_raise_nonfinite_json(constant)),
        )
    except Lfm2SharedCheckpointFormatError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Lfm2SharedCheckpointFormatError(
            f"Could not read strict JSON artifact {path.name!r}."
        ) from exc
    if not isinstance(value, dict):
        raise Lfm2SharedCheckpointFormatError(
            f"JSON artifact {path.name!r} must contain an object."
        )
    return value


def _raise_nonfinite_json(value: str) -> None:
    raise Lfm2SharedCheckpointFormatError(
        f"Non-finite JSON constant {value!r} is forbidden."
    )


def _strict_keys(
    value: Any,
    expected: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise Lfm2SharedCheckpointFormatError(
            f"{label} keys differ from the schema: expected {sorted(expected)!r}, "
            f"got {actual!r}."
        )
    return value


def _format_integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Lfm2SharedCheckpointFormatError(
            f"{name} must be an integer >= {minimum}."
        )
    return value


def _checked_limit(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2SharedCheckpointError(f"{name} must be nonnegative or None.")
    return value


def _checked_artifact_path(
    root: Path,
    entry_value: Any,
    label: str,
) -> Path:
    entry = _strict_keys(
        entry_value,
        {"file", "size_bytes", "sha256"},
        f"{label} artifact",
    )
    filename = entry["file"]
    digest = entry["sha256"]
    size = _format_integer(entry["size_bytes"], f"{label}.size_bytes", minimum=1)
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
    ):
        raise Lfm2SharedCheckpointFormatError(
            f"{label} artifact filename is unsafe."
        )
    if not isinstance(digest, str) or len(digest) != 64:
        raise Lfm2SharedCheckpointFormatError(
            f"{label} artifact SHA-256 is invalid."
        )
    path = root / filename
    if not path.is_file() or path.is_symlink():
        raise Lfm2SharedCheckpointFormatError(
            f"{label} artifact {filename!r} is missing or unsafe."
        )
    if path.stat().st_size != size or _sha256_file(path) != digest:
        raise Lfm2SharedCheckpointFormatError(
            f"{label} artifact failed size/SHA-256 verification."
        )
    return path


def _dtype_from_name(value: Any) -> torch.dtype:
    if not isinstance(value, str) or value not in _SUPPORTED_DTYPES:
        raise Lfm2SharedCheckpointFormatError(
            f"Unsupported tensor dtype {value!r}."
        )
    return _SUPPORTED_DTYPES[value]


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _validate_topology_and_aliases(
    manifest: Mapping[str, Any],
    config_data: Mapping[str, Any],
) -> tuple[
    tuple[tuple[int, ...], ...],
    dict[str, str],
    dict[str, bool],
]:
    topology = _strict_keys(
        manifest["topology"],
        {
            "model_class",
            "model_type",
            "logical_layer_count",
            "layer_types",
            "shared_ffn_layer_groups",
            "ffn_projection_roles",
            "complete_coupled_ffn_groups",
            "exact_parameter_identity_on_reload",
            "tied_input_output_embeddings",
        },
        "topology",
    )
    layer_count = _format_integer(
        topology["logical_layer_count"],
        "topology.logical_layer_count",
        minimum=1,
    )
    layer_types = topology["layer_types"]
    if (
        topology["model_class"] != "transformers.Lfm2ForCausalLM"
        or topology["model_type"] != "lfm2"
        or topology["ffn_projection_roles"] != list(FFN_PROJECTION_ROLES)
        or topology["complete_coupled_ffn_groups"] is not True
        or topology["exact_parameter_identity_on_reload"] is not True
        or topology["tied_input_output_embeddings"] is not True
        or not isinstance(layer_types, list)
        or len(layer_types) != layer_count
        or any(value not in {"conv", "full_attention"} for value in layer_types)
    ):
        raise Lfm2SharedCheckpointFormatError(
            "Topology does not describe a strict shared-FFN LFM2."
        )
    if (
        config_data.get("model_type") != "lfm2"
        or config_data.get("tie_word_embeddings") is not True
        or config_data.get("num_hidden_layers") != layer_count
        or config_data.get("layer_types") != layer_types
        or config_data.get("_name_or_path") not in {None, ""}
        or "auto_map" in config_data
    ):
        raise Lfm2SharedCheckpointFormatError(
            "Standalone LFM2 config differs from the declared topology."
        )
    groups = _normalize_groups(
        topology["shared_ffn_layer_groups"],
        layer_count,
        error_type=Lfm2SharedCheckpointFormatError,
    )

    state = _strict_keys(
        manifest["state_dict"],
        {
            "logical_to_stored_tensor",
            "logical_tensor_count",
            "stored_tensor_count",
            "alias_count",
            "ffn_alias_count",
            "vocabulary_alias_count",
            "complete_logical_state_mapping",
            "parameter_requires_grad",
        },
        "state_dict",
    )
    aliases_value = state["logical_to_stored_tensor"]
    requires_grad_value = state["parameter_requires_grad"]
    if not isinstance(aliases_value, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in aliases_value.items()
    ):
        raise Lfm2SharedCheckpointFormatError(
            "logical_to_stored_tensor must be a string mapping."
        )
    aliases = dict(aliases_value)
    if not isinstance(requires_grad_value, Mapping) or set(requires_grad_value) != set(
        aliases
    ) or not all(isinstance(value, bool) for value in requires_grad_value.values()):
        raise Lfm2SharedCheckpointFormatError(
            "parameter_requires_grad must cover every logical state name."
        )
    requires_grad = dict(requires_grad_value)
    expected = _expected_aliases(
        tuple(aliases),
        groups,
        error_type=Lfm2SharedCheckpointFormatError,
    )
    if aliases != expected:
        raise Lfm2SharedCheckpointFormatError(
            "Logical alias map differs from the complete declared FFN groups."
        )
    canonical_names = set(aliases.values())
    ffn_alias_count = sum(
        (len(group) - 1) * len(FFN_PROJECTION_ROLES) for group in groups
    )
    if (
        state["complete_logical_state_mapping"] is not True
        or state["logical_tensor_count"] != len(aliases)
        or state["stored_tensor_count"] != len(canonical_names)
        or state["alias_count"] != len(aliases) - len(canonical_names)
        or state["ffn_alias_count"] != ffn_alias_count
        or state["vocabulary_alias_count"] != 1
        or "model.embed_tokens.weight" not in aliases
        or aliases.get("lm_head.weight") != "model.embed_tokens.weight"
    ):
        raise Lfm2SharedCheckpointFormatError(
            "Logical state alias counts are inconsistent."
        )
    for logical, canonical in aliases.items():
        if canonical not in aliases or aliases[canonical] != canonical:
            raise Lfm2SharedCheckpointFormatError(
                f"Alias {logical!r} has a non-canonical target."
            )
        if requires_grad[logical] != requires_grad[canonical]:
            raise Lfm2SharedCheckpointFormatError(
                f"Alias group for {canonical!r} has mixed requires_grad."
            )
    return groups, aliases, requires_grad


def _validate_counts_and_tensors(
    manifest: Mapping[str, Any],
    groups: tuple[tuple[int, ...], ...],
    aliases: Mapping[str, str],
    tensor_format: str,
    tensor_file_size: int,
) -> tuple[Lfm2SharedCheckpointPlan, int, int]:
    counts = _strict_keys(
        manifest["parameter_counts"],
        {
            "logical_layer_count",
            "shared_ffn_layer_groups",
            "unique_parameter_count",
            "logical_parameter_count",
            "ffn_shared_parameter_savings",
            "vocabulary_shared_parameter_savings",
            "total_parameter_alias_savings",
            "parameter_budget",
            "budget_headroom",
        },
        "parameter_counts",
    )
    if (
        counts["logical_layer_count"] != manifest["topology"]["logical_layer_count"]
        or counts["shared_ffn_layer_groups"]
        != [list(group) for group in groups]
    ):
        raise Lfm2SharedCheckpointFormatError(
            "Parameter counts differ from declared topology."
        )
    unique = _format_integer(
        counts["unique_parameter_count"],
        "unique_parameter_count",
        minimum=1,
    )
    logical = _format_integer(
        counts["logical_parameter_count"],
        "logical_parameter_count",
        minimum=1,
    )
    ffn_savings = _format_integer(
        counts["ffn_shared_parameter_savings"],
        "ffn_shared_parameter_savings",
    )
    vocabulary_savings = _format_integer(
        counts["vocabulary_shared_parameter_savings"],
        "vocabulary_shared_parameter_savings",
        minimum=1,
    )
    total_savings = _format_integer(
        counts["total_parameter_alias_savings"],
        "total_parameter_alias_savings",
        minimum=1,
    )
    budget = counts["parameter_budget"]
    headroom = counts["budget_headroom"]
    if budget is not None:
        budget = _format_integer(budget, "parameter_budget")
        headroom = _format_integer(headroom, "budget_headroom")
    elif headroom is not None:
        raise Lfm2SharedCheckpointFormatError(
            "budget_headroom must be null when parameter_budget is null."
        )
    if (
        logical - unique != total_savings
        or ffn_savings + vocabulary_savings != total_savings
        or (budget is not None and (unique > budget or headroom != budget - unique))
    ):
        raise Lfm2SharedCheckpointFormatError(
            "Parameter count arithmetic is inconsistent."
        )
    plan = Lfm2SharedCheckpointPlan(
        logical_layer_count=counts["logical_layer_count"],
        shared_ffn_layer_groups=groups,
        unique_parameter_count=unique,
        logical_parameter_count=logical,
        ffn_shared_parameter_savings=ffn_savings,
        vocabulary_shared_parameter_savings=vocabulary_savings,
        parameter_budget=budget,
        budget_headroom=headroom,
    )

    tensors_value = manifest["tensors"]
    canonical_names = set(aliases.values())
    if not isinstance(tensors_value, Mapping) or set(tensors_value) != canonical_names:
        raise Lfm2SharedCheckpointFormatError(
            "Tensor metadata differs from the canonical alias set."
        )
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
    payload_bytes = 0
    largest_bytes = 0
    stored_parameters = 0
    expected_offset = 0
    for name in sorted(canonical_names):
        entry = _strict_keys(tensors_value[name], metadata_keys, f"tensor {name!r}")
        shape = entry["shape"]
        if (
            not isinstance(shape, list)
            or not shape
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in shape
            )
        ):
            raise Lfm2SharedCheckpointFormatError(
                f"Tensor {name!r} shape is invalid."
            )
        dtype = _dtype_from_name(entry["dtype"])
        numel = _format_integer(entry["numel"], f"{name}.numel", minimum=1)
        expected_numel = 1
        for dimension in shape:
            expected_numel *= dimension
        nbytes = _format_integer(entry["nbytes"], f"{name}.nbytes", minimum=1)
        digest = entry["sha256"]
        if (
            numel != expected_numel
            or nbytes != numel * _dtype_nbytes(dtype)
            or not isinstance(digest, str)
            or len(digest) != 64
            or entry["is_parameter"] is not True
        ):
            raise Lfm2SharedCheckpointFormatError(
                f"Tensor metadata for {name!r} is inconsistent."
            )
        if tensor_format == RAW_TENSOR_FORMAT:
            offset = _format_integer(entry["byte_offset"], f"{name}.byte_offset")
            if offset != expected_offset:
                raise Lfm2SharedCheckpointFormatError(
                    "Raw tensor ranges must be sorted, contiguous, and non-overlapping."
                )
            expected_offset += nbytes
        payload_bytes += nbytes
        largest_bytes = max(largest_bytes, nbytes)
        stored_parameters += numel
    logical_parameters = sum(
        int(tensors_value[canonical]["numel"]) for canonical in aliases.values()
    )
    expected_vocabulary_savings = int(
        tensors_value["model.embed_tokens.weight"]["numel"]
    )
    expected_ffn_savings = sum(
        (len(group) - 1)
        * sum(
            int(tensors_value[_ffn_key(group[0], role)]["numel"])
            for role in FFN_PROJECTION_ROLES
        )
        for group in groups
    )
    if (
        stored_parameters != unique
        or logical_parameters != logical
        or vocabulary_savings != expected_vocabulary_savings
        or ffn_savings != expected_ffn_savings
        or (
            tensor_format == RAW_TENSOR_FORMAT
            and payload_bytes != tensor_file_size
        )
    ):
        raise Lfm2SharedCheckpointFormatError(
            "Tensor metadata counts disagree with the parameter plan."
        )
    storage = _strict_keys(
        manifest["storage"],
        {
            "physical_tensor_bytes",
            "physical_parameter_bytes",
            "largest_physical_tensor_bytes",
            "tensor_file_bytes",
        },
        "storage",
    )
    reload_memory = _strict_keys(
        manifest["reload_memory"],
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
        storage["physical_tensor_bytes"] != payload_bytes
        or storage["physical_parameter_bytes"] != payload_bytes
        or storage["largest_physical_tensor_bytes"] != largest_bytes
        or storage["tensor_file_bytes"] != tensor_file_size
        or reload_memory["topology_construction_device"] != "meta"
        or reload_memory["streamed_physical_tensors"] is not True
        or reload_memory["model_parameter_bytes"] != payload_bytes
        or reload_memory["largest_staging_tensor_bytes"] != largest_bytes
        or reload_memory["bounded_peak_parameter_and_staging_bytes"]
        != payload_bytes + largest_bytes
    ):
        raise Lfm2SharedCheckpointFormatError(
            "Physical storage or reload-memory accounting is inconsistent."
        )
    return plan, payload_bytes, largest_bytes


def _parameter_counts_meta(model: nn.Module) -> tuple[int, int]:
    logical = 0
    unique: dict[int, nn.Parameter] = {}
    for _, parameter in model.named_parameters(
        recurse=True,
        remove_duplicate=False,
    ):
        logical += int(parameter.numel())
        unique[id(parameter)] = parameter
    return sum(int(parameter.numel()) for parameter in unique.values()), logical


def _build_meta_topology(
    config_data: Mapping[str, Any],
    generation_data: Mapping[str, Any],
    groups: tuple[tuple[int, ...], ...],
    aliases: Mapping[str, str],
    plan: Lfm2SharedCheckpointPlan,
    target_device: torch.device,
    runtime_dtype: torch.dtype,
    model_training: bool,
) -> nn.Module:
    try:
        from transformers import GenerationConfig, Lfm2Config, Lfm2ForCausalLM
    except ImportError as exc:
        raise Lfm2SharedCheckpointError(
            "Loading requires Transformers with LFM2 support."
        ) from exc
    config = Lfm2Config.from_dict(dict(config_data))
    with torch.device("meta"):
        model = Lfm2ForCausalLM(config)
    rotary = model.model.rotary_emb
    model.model.rotary_emb = type(rotary)(
        config,
        device=target_device,
    ).to(dtype=runtime_dtype)
    model.generation_config = GenerationConfig.from_dict(dict(generation_data))
    layers = _model_layers(model)
    for group in groups:
        source = layers[group[0]].feed_forward
        for role in FFN_PROJECTION_ROLES:
            parameter = getattr(source, role).weight
            for layer_index in group[1:]:
                getattr(layers[layer_index].feed_forward, role).weight = parameter
    model.get_output_embeddings().weight = model.get_input_embeddings().weight
    if set(model.state_dict()) != set(aliases) or set(_logical_parameters(model)) != set(
        aliases
    ):
        raise Lfm2SharedCheckpointFormatError(
            "Fresh Transformers LFM2 state surface differs from the manifest."
        )
    unique, logical = _parameter_counts_meta(model)
    if (
        unique != plan.unique_parameter_count
        or logical != plan.logical_parameter_count
    ):
        raise Lfm2SharedCheckpointFormatError(
            "Meta topology parameter counts disagree with the manifest."
        )
    model.train(model_training)
    return model


def _validate_meta_tensor_shapes(
    model: nn.Module,
    aliases: Mapping[str, str],
    tensor_metadata: Mapping[str, Any],
) -> None:
    """Bind every sealed physical shape to the fresh standard LFM2 topology."""

    state = model.state_dict()
    parameters = _logical_parameters(model)
    for logical, canonical in aliases.items():
        expected_shape = tuple(tensor_metadata[canonical]["shape"])
        if (
            tuple(state[logical].shape) != expected_shape
            or tuple(parameters[logical].shape) != expected_shape
        ):
            raise Lfm2SharedCheckpointFormatError(
                f"Tensor shape for {logical!r} differs from fresh LFM2 topology."
            )
        if parameters[logical] is not parameters[canonical]:
            raise Lfm2SharedCheckpointFormatError(
                f"Meta topology did not restore alias {logical!r} by identity."
            )


def _resolve_owner(model: nn.Module, state_key: str) -> tuple[Any, str]:
    owner_path, separator, name = state_key.rpartition(".")
    if not separator:
        raise Lfm2SharedCheckpointFormatError(
            f"Invalid logical state name {state_key!r}."
        )
    owner: Any = model
    try:
        for part in owner_path.split("."):
            owner = getattr(owner, part)
    except (AttributeError, KeyError, IndexError) as exc:
        raise Lfm2SharedCheckpointFormatError(
            f"Logical state owner {owner_path!r} does not exist."
        ) from exc
    return owner, name


def _install_parameter_group(
    model: nn.Module,
    logical_names: Sequence[str],
    tensor: Tensor,
    requires_grad: Mapping[str, bool],
) -> None:
    if not logical_names:
        raise Lfm2SharedCheckpointFormatError(
            "Canonical tensor has no logical state names."
        )
    required = requires_grad[logical_names[0]]
    parameter = nn.Parameter(tensor, requires_grad=required)
    for logical in logical_names:
        if requires_grad[logical] != required:
            raise Lfm2SharedCheckpointFormatError(
                "Alias group changed requires_grad during reload."
            )
        owner, name = _resolve_owner(model, logical)
        setattr(owner, name, parameter)


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
        raise Lfm2SharedCheckpointFormatError(
            f"Payload tensor {name!r} differs from its metadata."
        )
    _, raw = _tensor_cpu_view(tensor)
    if _sha256_bytes(raw) != entry["sha256"]:
        raise Lfm2SharedCheckpointFormatError(
            f"Payload tensor {name!r} failed SHA-256."
        )


def _logical_by_canonical(
    aliases: Mapping[str, str],
    tensor_metadata: Mapping[str, Any],
) -> dict[str, list[str]]:
    result = {name: [] for name in tensor_metadata}
    for logical, canonical in aliases.items():
        result[canonical].append(logical)
    return result


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
        raise Lfm2SharedCheckpointError(
            "This artifact requires the safetensors package."
        ) from exc
    logical_groups = _logical_by_canonical(aliases, tensor_metadata)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(tensor_metadata):
            raise Lfm2SharedCheckpointFormatError(
                "Safetensors keys differ from the manifest."
            )
        if handle.metadata() != {"format": CHECKPOINT_SCHEMA}:
            raise Lfm2SharedCheckpointFormatError(
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
                logical_groups[name],
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
    logical_groups = _logical_by_canonical(aliases, tensor_metadata)
    with path.open("rb", buffering=0) as handle:
        for name in sorted(tensor_metadata):
            entry = tensor_metadata[name]
            nbytes = int(entry["nbytes"])
            buffer = bytearray(nbytes)
            view = memoryview(buffer)
            handle.seek(int(entry["byte_offset"]))
            if handle.readinto(view) != nbytes:
                raise Lfm2SharedCheckpointFormatError(
                    f"Raw tensor {name!r} ended unexpectedly."
                )
            if _sha256_bytes(view) != entry["sha256"]:
                raise Lfm2SharedCheckpointFormatError(
                    f"Raw tensor {name!r} failed SHA-256."
                )
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
                logical_groups[name],
                materialized,
                requires_grad,
            )


def load_lfm2_shared_checkpoint(
    checkpoint_directory: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
    max_parameter_count: int | None = None,
    max_model_bytes: int | None = None,
    max_tensor_bytes: int | None = None,
    max_memory_bytes: int | None = None,
) -> nn.Module:
    """Strictly reload a standalone shared-FFN LFM2 with bounded staging."""

    root = Path(checkpoint_directory).expanduser().resolve()
    if not root.is_dir():
        raise Lfm2SharedCheckpointFormatError(
            f"Checkpoint directory {root} does not exist."
        )
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise Lfm2SharedCheckpointFormatError(f"Missing {MANIFEST_FILENAME}.")
    manifest = _read_json(manifest_path)
    _strict_keys(
        manifest,
        {
            "schema",
            "serialization",
            "artifacts",
            "topology",
            "parameter_counts",
            "state_dict",
            "tensors",
            "storage",
            "reload_memory",
            "model_training",
            "standalone",
            "parent_model_required",
            "tokenizer_required_for_reload",
            "tokenizer_serialized",
            "external_weight_dependencies",
            _MANIFEST_HASH_KEY,
        },
        "manifest",
    )
    if manifest["schema"] != CHECKPOINT_SCHEMA:
        raise Lfm2SharedCheckpointFormatError("Unsupported checkpoint schema.")
    claimed_hash = manifest[_MANIFEST_HASH_KEY]
    hash_payload = dict(manifest)
    hash_payload.pop(_MANIFEST_HASH_KEY)
    if (
        not isinstance(claimed_hash, str)
        or len(claimed_hash) != 64
        or claimed_hash != _sha256_bytes(_canonical_json_bytes(hash_payload))
    ):
        raise Lfm2SharedCheckpointFormatError("Manifest SHA-256 mismatch.")
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
        raise Lfm2SharedCheckpointFormatError(
            "Manifest permits unsafe Python object serialization."
        )
    tensor_format = serialization["tensor_format"]
    if tensor_format not in {"safetensors", RAW_TENSOR_FORMAT}:
        raise Lfm2SharedCheckpointFormatError(
            f"Unsupported tensor format {tensor_format!r}."
        )
    if serialization["byte_order"] not in {"little", "big"} or (
        tensor_format == RAW_TENSOR_FORMAT
        and serialization["byte_order"] != sys.byteorder
    ):
        raise Lfm2SharedCheckpointFormatError(
            "Raw tensor byte order differs from this runtime."
        )
    if (
        manifest["standalone"] is not True
        or manifest["parent_model_required"] is not False
        or manifest["tokenizer_required_for_reload"] is not False
        or manifest["tokenizer_serialized"] is not False
        or manifest["external_weight_dependencies"] != []
        or not isinstance(manifest["model_training"], bool)
    ):
        raise Lfm2SharedCheckpointFormatError(
            "Manifest does not describe a parent/tokenizer-free standalone checkpoint."
        )

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
    if {path.name for path in root.iterdir()} != expected_files:
        raise Lfm2SharedCheckpointFormatError(
            "Checkpoint directory contains missing or unclaimed artifacts."
        )
    config_data = _read_json(config_path)
    generation_data = _read_json(generation_path)
    groups, aliases, requires_grad = _validate_topology_and_aliases(
        manifest,
        config_data,
    )
    plan, model_bytes, largest_tensor_bytes = _validate_counts_and_tensors(
        manifest,
        groups,
        aliases,
        tensor_format,
        tensor_path.stat().st_size,
    )

    limits = (
        (
            _checked_limit(max_parameter_count, "max_parameter_count"),
            plan.unique_parameter_count,
            "max_parameter_count",
        ),
        (
            _checked_limit(max_model_bytes, "max_model_bytes"),
            model_bytes,
            "max_model_bytes",
        ),
        (
            _checked_limit(max_tensor_bytes, "max_tensor_bytes"),
            largest_tensor_bytes,
            "max_tensor_bytes",
        ),
        (
            _checked_limit(max_memory_bytes, "max_memory_bytes"),
            model_bytes + largest_tensor_bytes,
            "max_memory_bytes",
        ),
    )
    for limit, required, name in limits:
        if limit is not None and required > limit:
            raise Lfm2SharedCheckpointBudgetError(
                f"Checkpoint requires {required} for {name}, exceeding {limit}."
            )

    target_device = torch.device(device)
    if target_device.type == "meta":
        raise Lfm2SharedCheckpointError(
            "device='meta' is not a materialized reload target."
        )
    tensor_metadata = manifest["tensors"]
    embedding_name = aliases.get("model.embed_tokens.weight")
    if (
        not isinstance(embedding_name, str)
        or embedding_name not in tensor_metadata
    ):
        raise Lfm2SharedCheckpointFormatError(
            "Manifest lacks the canonical tied embedding dtype."
        )
    runtime_dtype = _dtype_from_name(tensor_metadata[embedding_name]["dtype"])
    model = _build_meta_topology(
        config_data,
        generation_data,
        groups,
        aliases,
        plan,
        target_device,
        runtime_dtype,
        manifest["model_training"],
    )
    _validate_meta_tensor_shapes(
        model,
        aliases,
        tensor_metadata,
    )
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
        raise Lfm2SharedCheckpointFormatError(
            "Reload left meta tensors in the materialized model."
        )
    reloaded_plan = plan_lfm2_shared_checkpoint(model, groups)
    if (
        reloaded_plan.unique_parameter_count != plan.unique_parameter_count
        or reloaded_plan.logical_parameter_count != plan.logical_parameter_count
        or reloaded_plan.ffn_shared_parameter_savings
        != plan.ffn_shared_parameter_savings
    ):
        raise Lfm2SharedCheckpointFormatError(
            "Reloaded topology differs from the sealed parameter plan."
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
            "shared_ffn_layer_groups": [list(group) for group in groups],
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


# Descriptive aliases for callers that prefer to spell out the shared unit.
export_lfm2_shared_ffn_checkpoint = export_lfm2_shared_checkpoint
load_lfm2_shared_ffn_checkpoint = load_lfm2_shared_checkpoint
plan_lfm2_shared_ffn_checkpoint = plan_lfm2_shared_checkpoint


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CONFIG_FILENAME",
    "FFN_PROJECTION_ROLES",
    "GENERATION_CONFIG_FILENAME",
    "MANIFEST_FILENAME",
    "RAW_TENSORS_FILENAME",
    "SAFETENSORS_FILENAME",
    "Lfm2SharedCheckpointBudgetError",
    "Lfm2SharedCheckpointError",
    "Lfm2SharedCheckpointFormatError",
    "Lfm2SharedCheckpointPlan",
    "export_lfm2_shared_checkpoint",
    "export_lfm2_shared_ffn_checkpoint",
    "load_lfm2_shared_checkpoint",
    "load_lfm2_shared_ffn_checkpoint",
    "plan_lfm2_shared_checkpoint",
    "plan_lfm2_shared_ffn_checkpoint",
]
