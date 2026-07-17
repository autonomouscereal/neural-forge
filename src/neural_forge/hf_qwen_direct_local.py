"""Concrete, local-only Qwen3 loading without Auto classes or hub access."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


class Qwen3DirectLocalError(RuntimeError):
    """A local Qwen3 file set or concrete runtime failed its closed contract."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def read_json_object(path: str | os.PathLike[str], label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Qwen3DirectLocalError(f"Cannot read {label}: {source}.") from exc
    if not isinstance(value, dict):
        raise Qwen3DirectLocalError(f"{label} must be a JSON object.")
    return value


def resolve_local_directory(
    path: str | os.PathLike[str],
    label: str,
) -> Path:
    source = Path(path).expanduser().absolute()
    if not source.is_dir() or source.is_symlink():
        raise Qwen3DirectLocalError(
            f"{label} must be a real local directory: {source}."
        )
    return source.resolve(strict=True)


@dataclass(frozen=True, slots=True)
class LocalSafetensorLayout:
    shards: tuple[Path, ...]
    weight_map: Mapping[str, str] | None


def local_safetensor_layout(
    model_directory: str | os.PathLike[str],
) -> LocalSafetensorLayout:
    directory = resolve_local_directory(model_directory, "model directory")
    index_path = directory / "model.safetensors.index.json"
    weight_map: dict[str, str] | None = None
    if index_path.is_file():
        index = read_json_object(index_path, "safetensors index")
        raw_map = index.get("weight_map")
        if not isinstance(raw_map, Mapping) or not raw_map:
            raise Qwen3DirectLocalError("Local safetensors index has no weight_map.")
        weight_map = {}
        for raw_name, raw_filename in raw_map.items():
            name = str(raw_name)
            filename = str(raw_filename)
            relative = Path(filename)
            if (
                not name
                or not filename
                or relative.is_absolute()
                or relative.name != filename
            ):
                raise Qwen3DirectLocalError(
                    "Safetensors index contains an unsafe key or shard path."
                )
            weight_map[name] = filename
        shards = tuple(
            directory / filename for filename in sorted(set(weight_map.values()))
        )
    else:
        single = directory / "model.safetensors"
        shards = (
            (single,)
            if single.is_file()
            else tuple(sorted(directory.glob("model-*.safetensors")))
        )
    if not shards or any(not shard.is_file() for shard in shards):
        raise Qwen3DirectLocalError(
            f"Local Qwen3 safetensors are incomplete in {directory}."
        )
    return LocalSafetensorLayout(shards=shards, weight_map=weight_map)


def load_local_qwen3_config(model_directory: str | os.PathLike[str]) -> Any:
    """Construct concrete ``Qwen3Config`` directly from local JSON."""

    from transformers import Qwen3Config

    directory = resolve_local_directory(model_directory, "Qwen3 model")
    payload = read_json_object(directory / "config.json", "Qwen3 config")
    if payload.get("model_type") != "qwen3":
        raise Qwen3DirectLocalError("Local config.json is not a Qwen3 configuration.")
    payload.pop("model_type", None)
    return Qwen3Config(**payload)


@contextmanager
def temporary_default_dtype(dtype: Any) -> Iterator[None]:
    import torch

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def rebuild_qwen3_runtime_buffers(model: Any, device: Any) -> None:
    backbone = getattr(model, "model", None)
    rotary = getattr(backbone, "rotary_emb", None)
    config = getattr(model, "config", None)
    if rotary is None or config is None:
        raise Qwen3DirectLocalError(
            "Concrete Qwen3 runtime lacks its rotary embedding."
        )
    try:
        replacement = type(rotary)(config, device=device)
    except Exception as exc:
        raise Qwen3DirectLocalError(
            "Could not reconstruct Qwen3 non-persistent rotary buffers."
        ) from exc
    backbone.rotary_emb = replacement


def _storage_identity(value: Any) -> tuple[int, int, tuple[int, ...]]:
    return (
        int(value.untyped_storage().data_ptr()),
        int(value.storage_offset()),
        tuple(int(size) for size in value.shape),
    )


def _tensor_equal_across_devices(left: Any, right: Any) -> bool:
    import torch

    if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
        return False
    return bool(
        torch.equal(
            left.detach().to(device="cpu"),
            right.detach().to(device="cpu"),
        )
    )


def stream_local_safetensors_into_model(
    model: Any,
    model_directory: str | os.PathLike[str],
) -> None:
    """Copy every indexed local tensor into an already materialized runtime."""

    import torch
    from safetensors import safe_open

    layout = local_safetensor_layout(model_directory)
    target_state = model.state_dict()
    loaded: set[str] = set()
    unexpected: list[str] = []
    loaded_storage: dict[tuple[int, int, tuple[int, ...]], str] = {}
    with torch.no_grad():
        for shard in layout.shards:
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if name in loaded:
                        raise Qwen3DirectLocalError(
                            f"Duplicate tensor across local shards: {name}."
                        )
                    if (
                        layout.weight_map is not None
                        and layout.weight_map.get(name) != shard.name
                    ):
                        raise Qwen3DirectLocalError(
                            f"Safetensors index maps {name!r} to the wrong shard."
                        )
                    if name not in target_state:
                        unexpected.append(name)
                        continue
                    target = target_state[name]
                    identity = _storage_identity(target)
                    source = handle.get_tensor(name)
                    if identity in loaded_storage:
                        if not _tensor_equal_across_devices(target, source):
                            raise Qwen3DirectLocalError(
                                "Serialized aliases disagree for tied Qwen3 "
                                f"storage: {loaded_storage[identity]!r}, {name!r}."
                            )
                        loaded.add(name)
                        continue
                    if tuple(source.shape) != tuple(target.shape):
                        raise Qwen3DirectLocalError(
                            f"Local tensor shape mismatch for {name}: "
                            f"{tuple(source.shape)} != {tuple(target.shape)}."
                        )
                    target.copy_(
                        source.to(device=target.device, dtype=target.dtype),
                    )
                    loaded.add(name)
                    loaded_storage[identity] = name
    if unexpected:
        raise Qwen3DirectLocalError(
            f"Local safetensors contain unexpected Qwen3 keys: {unexpected[:8]}."
        )
    if layout.weight_map is not None and set(layout.weight_map) != loaded:
        raise Qwen3DirectLocalError(
            "Safetensors index key inventory differs from loaded shard keys."
        )
    missing = sorted(
        name
        for name, value in target_state.items()
        if name not in loaded and _storage_identity(value) not in loaded_storage
    )
    if missing:
        raise Qwen3DirectLocalError(
            f"Local safetensors are missing non-aliased Qwen3 keys: {missing[:8]}."
        )


def load_local_qwen3_model(
    model_directory: str | os.PathLike[str],
    *,
    device: str | Any,
    dtype: Any,
) -> Any:
    """Stream local weights into concrete ``Qwen3ForCausalLM``."""

    import torch
    from transformers import GenerationConfig, Qwen3ForCausalLM

    directory = resolve_local_directory(model_directory, "Qwen3 model")
    config = load_local_qwen3_config(directory)
    # Mirror the concrete configuration identity produced by the authenticated
    # local load.  These are runtime load facts, not serialized model metadata.
    config._name_or_path = str(directory)
    config.dtype = dtype
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise Qwen3DirectLocalError("A CUDA Qwen3 device was requested without CUDA.")
    with temporary_default_dtype(dtype), torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    model.to_empty(device=target_device)
    model.tie_weights()
    rebuild_qwen3_runtime_buffers(model, target_device)
    stream_local_safetensors_into_model(model, directory)
    model.tie_weights()
    generation_path = directory / "generation_config.json"
    if generation_path.is_file():
        model.generation_config = GenerationConfig(
            **read_json_object(generation_path, "generation config")
        )
    meta_parameters = [
        name for name, value in model.named_parameters() if value.device.type == "meta"
    ]
    meta_buffers = [
        name for name, value in model.named_buffers() if value.device.type == "meta"
    ]
    if meta_parameters or meta_buffers:
        raise Qwen3DirectLocalError(
            "Direct Qwen3 loading left meta state: "
            f"parameters={meta_parameters[:4]}, buffers={meta_buffers[:4]}."
        )
    model.eval()
    return model


def _special_token_value(value: Any) -> Any:
    if isinstance(value, Mapping) and "content" in value:
        return str(value["content"])
    return value


def load_local_tokenizer(model_directory: str | os.PathLike[str]) -> Any:
    """Construct a fast tokenizer directly from local ``tokenizer.json``."""

    from transformers import AddedToken, Qwen2TokenizerFast

    directory = resolve_local_directory(model_directory, "Qwen3 tokenizer")
    tokenizer_path = directory / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise Qwen3DirectLocalError(
            f"Local tokenizer.json is missing: {tokenizer_path}."
        )
    kwargs: dict[str, Any] = {}
    config_path = directory / "tokenizer_config.json"
    if config_path.is_file():
        kwargs.update(read_json_object(config_path, "tokenizer config"))
    special_path = directory / "special_tokens_map.json"
    if special_path.is_file():
        kwargs.update(read_json_object(special_path, "special-token map"))
    raw_added_tokens = kwargs.pop("added_tokens_decoder", None)
    if raw_added_tokens is not None:
        if not isinstance(raw_added_tokens, Mapping):
            raise Qwen3DirectLocalError(
                "Tokenizer added_tokens_decoder must be a mapping."
            )
        added_tokens: dict[int, Any] = {}
        allowed_fields = {
            "content",
            "single_word",
            "lstrip",
            "rstrip",
            "normalized",
            "special",
        }
        for raw_index, raw_token in raw_added_tokens.items():
            if not isinstance(raw_token, Mapping) or "content" not in raw_token:
                raise Qwen3DirectLocalError(
                    "Tokenizer added-token records must contain content."
                )
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise Qwen3DirectLocalError(
                    "Tokenizer added-token IDs must be integers."
                ) from exc
            if index < 0 or index in added_tokens:
                raise Qwen3DirectLocalError(
                    "Tokenizer added-token IDs must be unique and non-negative."
                )
            token_kwargs = {
                str(name): value
                for name, value in raw_token.items()
                if str(name) in allowed_fields
            }
            added_tokens[index] = AddedToken(**token_kwargs)
        kwargs["added_tokens_decoder"] = added_tokens
    for name in (
        "auto_map",
        "backend",
        "is_local",
        "sp_model_kwargs",
        "tokenizer_class",
        "tokenizer_file",
        "use_fast",
    ):
        kwargs.pop(name, None)
    for name in (
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
    ):
        if name in kwargs:
            kwargs[name] = _special_token_value(kwargs[name])
    tokenizer = Qwen2TokenizerFast(
        tokenizer_file=str(tokenizer_path),
        **kwargs,
    )
    chat_template_path = directory / "chat_template.jinja"
    if chat_template_path.is_file():
        tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")
    return tokenizer


def tensor_sha256(value: Any, *, chunk_bytes: int = 8 * 1024**2) -> str:
    import torch

    tensor = value.detach()
    if tensor.device.type == "meta":
        raise Qwen3DirectLocalError("Cannot fingerprint a meta tensor.")
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    raw = tensor.view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(value.dtype), "shape": list(value.shape)},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )
    for start in range(0, int(raw.numel()), max(1, int(chunk_bytes))):
        piece = raw[start : start + chunk_bytes].to(device="cpu").contiguous()
        digest.update(piece.numpy().tobytes())
    return digest.hexdigest()


def state_identity(model: Any) -> dict[str, Any]:
    """Reproduce the authenticated compensated-runner state identity."""

    import torch

    tensors = []
    for name, value in sorted(model.state_dict().items()):
        if not isinstance(value, torch.Tensor):
            raise Qwen3DirectLocalError(f"State entry is not a tensor: {name}.")
        tensors.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": tensor_sha256(value),
            }
        )
    digest = _canonical_sha256(tensors)
    return {
        "tensor_count": len(tensors),
        "state_sha256": digest,
        "tensor_manifest_sha256": digest,
    }


__all__ = [
    "LocalSafetensorLayout",
    "Qwen3DirectLocalError",
    "file_sha256",
    "load_local_qwen3_config",
    "load_local_qwen3_model",
    "load_local_tokenizer",
    "local_safetensor_layout",
    "read_json_object",
    "rebuild_qwen3_runtime_buffers",
    "resolve_local_directory",
    "state_identity",
    "stream_local_safetensors_into_model",
    "temporary_default_dtype",
    "tensor_sha256",
]
