"""Immutable, content-addressed Hugging Face parent activation traces.

The cache accepts an already-loaded model and already-tokenized calibration
batches.  It never resolves model names, tokenizers, URLs, or remote files.
Each missing batch is produced by exactly one evaluation/inference forward and
published as an atomic directory containing a safetensors file plus a strict
JSON record.  Completed batch directories are never rewritten.

Hugging Face's standard ``hidden_states`` output is the only layer interface
used here.  Consequently, hidden-state indices describe model-defined layer
boundaries without requiring an architecture-specific module layout.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sys
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal


CACHE_SCHEMA = "neural-forge-parent-trace-cache-v1"
BATCH_SCHEMA = "neural-forge-parent-trace-batch-v1"
COMPLETION_SCHEMA = "neural-forge-parent-trace-completion-v1"
MANIFEST_FILENAME = "manifest.json"
COMPLETION_FILENAME = "complete.json"
TRACE_FILENAME = "trace.safetensors"
BATCH_RECORD_FILENAME = "record.json"

_LOCK_FILENAME = ".cache.lock"
_BATCH_NAME = re.compile(r"batch-(\d{6})\Z")
_STAGE_NAME = re.compile(r"\.batch-(\d{6})\.stage-[A-Za-z0-9-]+\Z")
_METADATA_STAGE_NAME = re.compile(
    r"\.(?:manifest|complete)\.json\.stage-[0-9a-f]{32}\Z"
)
_INPUT_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RESERVED_FORWARD_INPUTS = frozenset(
    {
        "cache_position",
        "encoder_attention_mask",
        "encoder_hidden_states",
        "labels",
        "output_attentions",
        "output_hidden_states",
        "past_key_value",
        "past_key_values",
        "return_dict",
        "use_cache",
    }
)


class ParentTraceCacheError(RuntimeError):
    """Base error for parent trace capture and persistence failures."""


class ParentTraceCaptureError(ParentTraceCacheError):
    """Raised when a parent cannot produce the requested deterministic trace."""


class ParentTraceCacheCorruptionError(ParentTraceCacheError):
    """Raised when a committed cache artifact fails strict verification."""


class ParentTraceCacheConflictError(ParentTraceCacheError):
    """Raised when existing cache contents do not match the requested identity."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - requires missing HF extras
        raise RuntimeError(
            "Parent trace caching requires the 'hf' optional dependencies."
        ) from exc
    return torch


def _safetensors() -> tuple[Any, Any]:
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - requires missing HF extras
        raise RuntimeError(
            "Parent trace caching requires the 'hf' optional dependencies."
        ) from exc
    return safe_open, save_file


def _canonical_json(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ParentTraceCacheError("Trace metadata must be canonical JSON.") from exc
    return text.encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_canonical_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=_strict_json_pairs)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ParentTraceCacheCorruptionError(
            f"Cannot parse cache metadata {path}."
        ) from exc
    if not isinstance(value, dict):
        raise ParentTraceCacheCorruptionError(
            f"Cache metadata {path} must contain a JSON object."
        )
    expected = _canonical_json(value) + b"\n"
    if raw != expected:
        raise ParentTraceCacheCorruptionError(
            f"Cache metadata {path} is not canonical or was modified."
        )
    return value


def _write_new_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_json(dict(value)) + b"\n"
    temporary = path.with_name(f".{path.name}.stage-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise ParentTraceCacheConflictError(
                f"Immutable cache metadata already exists at {path}."
            )
        os.rename(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ParentTraceCacheCorruptionError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _dense_cpu_copy(torch: Any, value: Any, name: str) -> Any:
    if not torch.is_tensor(value):
        try:
            value = torch.as_tensor(value)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ParentTraceCaptureError(f"{name} must be tensor-like.") from exc
    if value.layout != torch.strided or value.device.type == "meta":
        raise ParentTraceCaptureError(f"{name} must be a materialized dense tensor.")
    result = value.detach().to(device="cpu").contiguous().clone()
    if result.requires_grad or result.grad_fn is not None:
        raise ParentTraceCaptureError(f"{name} retained a gradient graph.")
    if (result.is_floating_point() or result.is_complex()) and not bool(
        torch.isfinite(result).all().item()
    ):
        raise ParentTraceCaptureError(f"{name} contains NaN or infinity.")
    return result


def _tensor_bytes(torch: Any, tensor: Any) -> bytes:
    value = tensor.detach().to(device="cpu").contiguous()
    return value.reshape(-1).view(torch.uint8).numpy().tobytes()


def _tensor_sha256(torch: Any, tensor: Any) -> str:
    digest = hashlib.sha256()
    digest.update(b"neural-forge-tensor-v1\0")
    digest.update(str(tensor.dtype).replace("torch.", "").encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json([int(size) for size in tensor.shape]))
    digest.update(b"\0")
    digest.update(_tensor_bytes(torch, tensor))
    return digest.hexdigest()


def _tensor_descriptor(torch: Any, tensor: Any) -> dict[str, Any]:
    return {
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "shape": [int(size) for size in tensor.shape],
        "numel": int(tensor.numel()),
        "size_bytes": int(tensor.numel() * tensor.element_size()),
        "sha256": _tensor_sha256(torch, tensor),
    }


def _named_tensor_sha256(torch: Any, tensors: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(b"neural-forge-named-tensors-v1\0")
    for name in sorted(tensors):
        tensor = tensors[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_tensor_sha256(torch, tensor).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ParentTraceCaptureConfig:
    """Architecture-neutral selection for cached parent outputs.

    ``hidden_state_indices=None`` retains every state in the Hugging Face
    ``hidden_states`` tuple.  Explicit indices must be unique, increasing, and
    nonnegative.  ``topk`` stores sorted values, token indices, and per-position
    full-vocabulary log-normalizers; ``full`` stores the complete logits tensor.
    """

    hidden_state_indices: tuple[int, ...] | None = None
    logits_mode: Literal["full", "topk"] = "topk"
    top_k: int = 64
    seed: int = 0

    def __post_init__(self) -> None:
        indices = self.hidden_state_indices
        if indices is not None:
            if isinstance(indices, (str, bytes)):
                raise TypeError("hidden_state_indices must be an integer sequence.")
            normalized = tuple(indices)
            if not normalized:
                raise ValueError("hidden_state_indices cannot be empty.")
            for index in normalized:
                if isinstance(index, bool) or not isinstance(index, int):
                    raise TypeError("hidden_state_indices must contain integers.")
                if index < 0:
                    raise ValueError("hidden_state_indices must be nonnegative.")
            if tuple(sorted(set(normalized))) != normalized:
                raise ValueError(
                    "hidden_state_indices must be unique and strictly increasing."
                )
            object.__setattr__(self, "hidden_state_indices", normalized)
        if self.logits_mode not in {"full", "topk"}:
            raise ValueError("logits_mode must be 'full' or 'topk'.")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise TypeError("top_k must be an integer.")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer.")
        if self.seed < 0 or self.seed >= 2**63:
            raise ValueError("seed must be in [0, 2**63).")

    def as_dict(self) -> dict[str, Any]:
        return {
            "hidden_state_indices": (
                None
                if self.hidden_state_indices is None
                else list(self.hidden_state_indices)
            ),
            "hidden_state_semantics": "hugging-face-output-hidden-states",
            "logits_mode": self.logits_mode,
            "top_k": self.top_k if self.logits_mode == "topk" else None,
            "seed": self.seed,
            "deterministic_algorithms": True,
            "inference_mode": True,
            "model_eval_mode": True,
            "use_cache": False,
        }


@dataclass(frozen=True, slots=True)
class ParentModelFingerprint:
    """Strict identity of parent class, configuration, and state tensors."""

    model_class: str
    config_sha256: str
    state_dict_sha256: str
    identity_sha256: str
    state_tensor_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_class": self.model_class,
            "config_sha256": self.config_sha256,
            "state_dict_sha256": self.state_dict_sha256,
            "identity_sha256": self.identity_sha256,
            "state_tensor_count": self.state_tensor_count,
        }


def _model_config(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    to_dict = getattr(config, "to_dict", None)
    if config is None or not callable(to_dict):
        raise ParentTraceCaptureError(
            "The parent must expose a Hugging Face-style config.to_dict()."
        )
    try:
        value = to_dict()
    except Exception as exc:
        raise ParentTraceCaptureError("Could not serialize the parent config.") from exc
    if not isinstance(value, Mapping):
        raise ParentTraceCaptureError("Parent config.to_dict() must return a mapping.")
    result = dict(value)
    _canonical_json(result)
    return result


def fingerprint_parent_model(model: Any) -> ParentModelFingerprint:
    """Hash a loaded parent's complete state dict and forward configuration."""

    torch = _torch()
    model_class = f"{type(model).__module__}.{type(model).__qualname__}"
    config_sha256 = _json_sha256(_model_config(model))
    try:
        state = model.state_dict()
    except Exception as exc:
        raise ParentTraceCaptureError("Could not read the parent state dict.") from exc
    if not isinstance(state, Mapping) or not state:
        raise ParentTraceCaptureError("The parent state dict must be nonempty.")

    digest = hashlib.sha256()
    digest.update(b"neural-forge-parent-state-v1\0")
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise ParentTraceCaptureError(
                "Parent state dict entries must be string-named tensors."
            )
        if tensor.layout != torch.strided or tensor.device.type == "meta":
            raise ParentTraceCaptureError(
                f"Parent state tensor {name!r} must be materialized and dense."
            )
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_tensor_sha256(torch, tensor).encode("ascii"))
        digest.update(b"\0")
    state_sha256 = digest.hexdigest()
    identity = {
        "schema": "neural-forge-parent-model-fingerprint-v1",
        "model_class": model_class,
        "config_sha256": config_sha256,
        "state_dict_sha256": state_sha256,
        "state_tensor_count": len(state),
    }
    return ParentModelFingerprint(
        model_class=model_class,
        config_sha256=config_sha256,
        state_dict_sha256=state_sha256,
        identity_sha256=_json_sha256(identity),
        state_tensor_count=len(state),
    )


@dataclass(frozen=True, slots=True)
class _PreparedBatch:
    index: int
    tensors: Mapping[str, Any]
    forward_input_names: tuple[str, ...]
    input_sha256: str


def _validate_binary_mask(
    torch: Any, tensor: Any, shape: tuple[int, int], name: str
) -> None:
    if tensor.ndim != 2 or tuple(int(size) for size in tensor.shape) != shape:
        raise ParentTraceCaptureError(f"{name} must have shape {shape}.")
    if tensor.is_complex() or not bool(((tensor == 0) | (tensor == 1)).all().item()):
        raise ParentTraceCaptureError(
            f"{name} must contain only boolean or binary values."
        )


def _prepare_batch(torch: Any, value: Any, index: int) -> _PreparedBatch:
    if not isinstance(value, Mapping):
        raise ParentTraceCaptureError(
            f"Calibration batch {index} must be a mapping of model input tensors."
        )
    if "input_ids" not in value:
        raise ParentTraceCaptureError(
            f"Calibration batch {index} must contain input_ids."
        )
    invalid_names = [
        name
        for name in value
        if not isinstance(name, str) or _INPUT_NAME.fullmatch(name) is None
    ]
    if invalid_names:
        raise ParentTraceCaptureError(
            f"Calibration batch {index} has invalid input names: {invalid_names!r}."
        )
    reserved = sorted(set(value).intersection(_RESERVED_FORWARD_INPUTS))
    if reserved:
        raise ParentTraceCaptureError(
            f"Calibration batch {index} supplies reserved forward controls: {reserved}."
        )

    tensors = {
        name: _dense_cpu_copy(torch, tensor, f"batch {index} input {name!r}")
        for name, tensor in value.items()
    }
    input_ids = tensors["input_ids"]
    if input_ids.ndim != 2 or min(int(size) for size in input_ids.shape) <= 0:
        raise ParentTraceCaptureError(
            f"Calibration batch {index} input_ids must have nonempty [batch, sequence] shape."
        )
    if (
        input_ids.dtype == torch.bool
        or input_ids.is_floating_point()
        or input_ids.is_complex()
    ):
        raise ParentTraceCaptureError(
            f"Calibration batch {index} input_ids must contain integer token IDs."
        )
    shape = tuple(int(size) for size in input_ids.shape)

    if "attention_mask" not in tensors:
        tensors["attention_mask"] = torch.ones(shape, dtype=torch.int64)
    _validate_binary_mask(torch, tensors["attention_mask"], shape, "attention_mask")
    if "valid_token_mask" not in tensors:
        tensors["valid_token_mask"] = tensors["attention_mask"].to(torch.bool)
    _validate_binary_mask(
        torch,
        tensors["valid_token_mask"],
        shape,
        "valid_token_mask",
    )
    attention = tensors["attention_mask"].to(torch.bool)
    valid = tensors["valid_token_mask"].to(torch.bool)
    if bool((valid & ~attention).any().item()):
        raise ParentTraceCaptureError(
            f"Calibration batch {index} valid_token_mask selects masked tokens."
        )
    if not bool(valid.any().item()):
        raise ParentTraceCaptureError(
            f"Calibration batch {index} must select at least one valid token."
        )
    tensors["valid_token_mask"] = valid.contiguous()
    forward_names = tuple(sorted(set(tensors) - {"valid_token_mask"}))
    input_sha256 = _named_tensor_sha256(torch, tensors)
    return _PreparedBatch(
        index=index,
        tensors=MappingProxyType(dict(tensors)),
        forward_input_names=forward_names,
        input_sha256=input_sha256,
    )


def _prepare_batches(
    torch: Any, batches: Iterable[Mapping[str, Any]]
) -> tuple[_PreparedBatch, ...]:
    if isinstance(batches, (str, bytes, Mapping)):
        raise ParentTraceCaptureError(
            "calibration_batches must be an iterable of input mappings."
        )
    try:
        prepared = tuple(
            _prepare_batch(torch, batch, index) for index, batch in enumerate(batches)
        )
    except TypeError as exc:
        raise ParentTraceCaptureError("calibration_batches must be iterable.") from exc
    if not prepared:
        raise ParentTraceCaptureError("calibration_batches cannot be empty.")
    if len(prepared) >= 1_000_000:
        raise ParentTraceCaptureError(
            "At most 999999 calibration batches are supported."
        )
    return prepared


def _calibration_fingerprint(prepared: Sequence[_PreparedBatch]) -> str:
    return _json_sha256(
        {
            "schema": "neural-forge-calibration-inputs-v1",
            "ordered_batch_input_sha256": [item.input_sha256 for item in prepared],
        }
    )


def _model_devices(model: Any) -> list[str]:
    devices: set[str] = set()
    for tensor in model.state_dict().values():
        device = getattr(tensor, "device", None)
        if device is not None:
            devices.add(str(device))
    return sorted(devices)


def _default_input_device(torch: Any, model: Any) -> Any:
    get_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_embeddings):
        try:
            embeddings = get_embeddings()
            weight = getattr(embeddings, "weight", None)
            if torch.is_tensor(weight) and weight.device.type != "meta":
                return weight.device
        except Exception:
            pass
    try:
        parameter = next(model.parameters())
    except (StopIteration, AttributeError) as exc:
        raise ParentTraceCaptureError(
            "Could not infer the parent input device."
        ) from exc
    if parameter.device.type == "meta":
        raise ParentTraceCaptureError("The parent input device cannot be meta.")
    return parameter.device


def _resolve_input_device(torch: Any, model: Any, value: Any | None) -> Any:
    if value is None:
        return _default_input_device(torch, model)
    try:
        device = torch.device(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ParentTraceCaptureError(f"Invalid input_device {value!r}.") from exc
    if device.type == "meta":
        raise ParentTraceCaptureError("input_device cannot be meta.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ParentTraceCaptureError(
            "A CUDA input device was requested but CUDA is unavailable."
        )
    return device


def _runtime_identity(torch: Any, model: Any, input_device: Any) -> dict[str, Any]:
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            cuda_devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    return {
        "python": platform.python_version(),
        "python_byteorder": sys.byteorder,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "torch": str(torch.__version__),
        "transformers": _package_version("transformers"),
        "safetensors": _package_version("safetensors"),
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "cuda_devices": cuda_devices,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "input_device": str(input_device),
        "model_state_devices": _model_devices(model),
        "local_files_only": True,
        "network_access_performed": False,
    }


def _manifest_for(
    parent: ParentModelFingerprint,
    prepared: Sequence[_PreparedBatch],
    config: ParentTraceCaptureConfig,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    selection = config.as_dict()
    selection["selection_sha256"] = _json_sha256(selection)
    calibration_sha256 = _calibration_fingerprint(prepared)
    body = {
        "schema": CACHE_SCHEMA,
        "hash_algorithm": "sha256",
        "storage": {
            "format": "safetensors",
            "batch_publication": "atomic-directory-rename",
            "committed_batches_immutable": True,
        },
        "parent": parent.as_dict(),
        "calibration": {
            "batch_count": len(prepared),
            "calibration_sha256": calibration_sha256,
            "ordered_batch_input_sha256": [item.input_sha256 for item in prepared],
        },
        "capture": selection,
        "runtime": dict(runtime),
    }
    result = dict(body)
    result["cache_key"] = _json_sha256(body)
    return result


def _validate_manifest(value: Mapping[str, Any]) -> None:
    expected_keys = {
        "cache_key",
        "calibration",
        "capture",
        "hash_algorithm",
        "parent",
        "runtime",
        "schema",
        "storage",
    }
    if set(value) != expected_keys or value.get("schema") != CACHE_SCHEMA:
        raise ParentTraceCacheCorruptionError("Cache manifest schema is invalid.")
    if value.get("hash_algorithm") != "sha256":
        raise ParentTraceCacheCorruptionError("Cache hash algorithm is invalid.")
    key = _validate_sha256(value.get("cache_key"), "cache_key")
    body = {name: item for name, item in value.items() if name != "cache_key"}
    if _json_sha256(body) != key:
        raise ParentTraceCacheCorruptionError("Cache manifest fingerprint is invalid.")
    calibration = value.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ParentTraceCacheCorruptionError("Cache calibration metadata is invalid.")
    batch_count = calibration.get("batch_count")
    hashes = calibration.get("ordered_batch_input_sha256")
    if (
        isinstance(batch_count, bool)
        or not isinstance(batch_count, int)
        or batch_count <= 0
        or not isinstance(hashes, list)
        or len(hashes) != batch_count
    ):
        raise ParentTraceCacheCorruptionError("Cache batch inventory is invalid.")
    for index, digest in enumerate(hashes):
        _validate_sha256(digest, f"calibration batch {index} input hash")
    expected_calibration = _json_sha256(
        {
            "schema": "neural-forge-calibration-inputs-v1",
            "ordered_batch_input_sha256": hashes,
        }
    )
    if calibration.get("calibration_sha256") != expected_calibration:
        raise ParentTraceCacheCorruptionError(
            "Cache calibration fingerprint is invalid."
        )
    capture = value.get("capture")
    if not isinstance(capture, Mapping):
        raise ParentTraceCacheCorruptionError("Cache capture metadata is invalid.")
    selection = dict(capture)
    selection_sha256 = selection.pop("selection_sha256", None)
    _validate_sha256(selection_sha256, "selection_sha256")
    if _json_sha256(selection) != selection_sha256:
        raise ParentTraceCacheCorruptionError("Cache selection fingerprint is invalid.")
    parent = value.get("parent")
    if not isinstance(parent, Mapping):
        raise ParentTraceCacheCorruptionError("Cache parent metadata is invalid.")
    for name in ("config_sha256", "state_dict_sha256", "identity_sha256"):
        _validate_sha256(parent.get(name), f"parent {name}")


def _manifest_matches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if _canonical_json(dict(actual)) != _canonical_json(dict(expected)):
        raise ParentTraceCacheConflictError(
            "Existing cache manifest does not match the requested parent, "
            "calibration inputs, capture selection, and runtime."
        )


@contextmanager
def _exclusive_cache_lock(path: Path, timeout_seconds: float = 300.0) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = time.monotonic() + timeout_seconds
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if time.monotonic() >= deadline:
                        raise ParentTraceCacheError(
                            f"Timed out waiting for cache lock {path}."
                        ) from exc
                    time.sleep(0.05)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as exc:
                    if time.monotonic() >= deadline:
                        raise ParentTraceCacheError(
                            f"Timed out waiting for cache lock {path}."
                        ) from exc
                    time.sleep(0.05)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _batch_dir(cache_path: Path, index: int) -> Path:
    return cache_path / "batches" / f"batch-{index:06d}"


def _clean_staging_directories(cache_path: Path) -> None:
    batches_path = cache_path / "batches"
    if not batches_path.exists():
        return
    for path in batches_path.iterdir():
        if _STAGE_NAME.fullmatch(path.name) is None:
            continue
        if path.is_symlink() or not path.is_dir():
            raise ParentTraceCacheCorruptionError(
                f"Invalid staging artifact in cache: {path}."
            )
        shutil.rmtree(path)


def _clean_metadata_staging_files(cache_path: Path) -> None:
    for path in cache_path.iterdir():
        if _METADATA_STAGE_NAME.fullmatch(path.name) is None:
            continue
        if path.is_symlink() or not path.is_file():
            raise ParentTraceCacheCorruptionError(
                f"Invalid metadata staging artifact in cache: {path}."
            )
        path.unlink()


def _assert_cache_layout(cache_path: Path, *, complete: bool) -> None:
    allowed_top = {MANIFEST_FILENAME, "batches", _LOCK_FILENAME}
    if complete:
        allowed_top.add(COMPLETION_FILENAME)
    for path in cache_path.iterdir():
        if path.name not in allowed_top or path.is_symlink():
            raise ParentTraceCacheCorruptionError(
                f"Unexpected cache artifact {path.name!r}."
            )
    batches_path = cache_path / "batches"
    if not batches_path.is_dir() or batches_path.is_symlink():
        raise ParentTraceCacheCorruptionError("Cache batches directory is missing.")
    for path in batches_path.iterdir():
        match = _BATCH_NAME.fullmatch(path.name)
        if match is None or not path.is_dir() or path.is_symlink():
            raise ParentTraceCacheCorruptionError(
                f"Unexpected committed batch artifact {path.name!r}."
            )


def _state_versions(model: Any) -> dict[str, tuple[Any, ...]]:
    versions: dict[str, tuple[Any, ...]] = {}
    for name, tensor in model.state_dict().items():
        versions[name] = (
            tuple(int(size) for size in tensor.shape),
            str(tensor.dtype),
            str(tensor.device),
            int(getattr(tensor, "_version", -1)),
        )
    return versions


@contextmanager
def _deterministic_parent_execution(
    torch: Any,
    model: Any,
    seed: int,
) -> Iterator[None]:
    modes = [(module, bool(module.training)) for module in model.modules()]
    algorithms_enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cuda_devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    try:
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.manual_seed(seed)
            model.eval()
            yield
    finally:
        torch.use_deterministic_algorithms(algorithms_enabled, warn_only=warn_only)
        for module, was_training in modes:
            module.training = was_training


def _batch_seed(config: ParentTraceCaptureConfig, batch: _PreparedBatch) -> int:
    digest = hashlib.sha256()
    digest.update(b"neural-forge-parent-trace-seed-v1\0")
    digest.update(str(config.seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(batch.input_sha256.encode("ascii"))
    return int.from_bytes(digest.digest()[:8], "big") % (2**63)


def _output_field(output: Any, name: str) -> Any:
    value = getattr(output, name, None)
    if value is None and isinstance(output, Mapping):
        value = output.get(name)
    return value


def _validate_output_tensor(
    torch: Any,
    tensor: Any,
    name: str,
    input_shape: tuple[int, int],
    *,
    rank: int,
) -> Any:
    if not torch.is_tensor(tensor) or tensor.layout != torch.strided:
        raise ParentTraceCaptureError(f"Parent {name} must be a dense tensor.")
    if (
        tensor.ndim != rank
        or tuple(int(size) for size in tensor.shape[:2]) != input_shape
    ):
        raise ParentTraceCaptureError(
            f"Parent {name} must have rank {rank} and begin with shape {input_shape}."
        )
    if not tensor.is_floating_point():
        raise ParentTraceCaptureError(f"Parent {name} must be floating point.")
    return _dense_cpu_copy(torch, tensor, f"parent {name}")


def _extract_trace_tensors(
    torch: Any,
    output: Any,
    batch: _PreparedBatch,
    config: ParentTraceCaptureConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cache = _output_field(output, "past_key_values")
    if cache is not None:
        raise ParentTraceCaptureError(
            "Parent returned past_key_values despite use_cache=False."
        )
    hidden_states = _output_field(output, "hidden_states")
    if not isinstance(hidden_states, (tuple, list)) or not hidden_states:
        raise ParentTraceCaptureError(
            "Parent output must contain a nonempty hidden_states sequence."
        )
    input_shape = tuple(int(size) for size in batch.tensors["input_ids"].shape)
    selected = (
        tuple(range(len(hidden_states)))
        if config.hidden_state_indices is None
        else config.hidden_state_indices
    )
    if selected[-1] >= len(hidden_states):
        raise ParentTraceCaptureError(
            "Requested hidden_state_indices exceed the parent hidden_states sequence."
        )

    tensors = {
        f"input.{name}": tensor.clone() for name, tensor in batch.tensors.items()
    }
    for index in selected:
        tensors[f"hidden_state.{index:06d}"] = _validate_output_tensor(
            torch,
            hidden_states[index],
            f"hidden_states[{index}]",
            input_shape,
            rank=3,
        )

    logits = _validate_output_tensor(
        torch,
        _output_field(output, "logits"),
        "logits",
        input_shape,
        rank=3,
    )
    vocabulary_size = int(logits.shape[-1])
    if vocabulary_size <= 0:
        raise ParentTraceCaptureError("Parent logits vocabulary cannot be empty.")
    if config.logits_mode == "full":
        tensors["logits.full"] = logits
        logits_metadata = {
            "mode": "full",
            "vocabulary_size": vocabulary_size,
            "requested_top_k": None,
            "effective_top_k": None,
        }
    else:
        effective_top_k = min(config.top_k, vocabulary_size)
        values, indices = torch.topk(
            logits,
            k=effective_top_k,
            dim=-1,
            largest=True,
            sorted=True,
        )
        tensors["logits.topk_values"] = values.contiguous()
        tensors["logits.topk_indices"] = indices.to(torch.int64).contiguous()
        tensors["logits.logsumexp"] = torch.logsumexp(
            logits.to(torch.float32), dim=-1
        ).contiguous()
        logits_metadata = {
            "mode": "topk",
            "vocabulary_size": vocabulary_size,
            "requested_top_k": config.top_k,
            "effective_top_k": effective_top_k,
        }
    metadata = {
        "available_hidden_state_count": len(hidden_states),
        "resolved_hidden_state_indices": list(selected),
        "logits": logits_metadata,
    }
    return tensors, metadata


def _capture_batch(
    torch: Any,
    model: Any,
    batch: _PreparedBatch,
    config: ParentTraceCaptureConfig,
    input_device: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    forward_inputs = {
        name: batch.tensors[name].to(device=input_device)
        for name in batch.forward_input_names
    }
    forward_inputs.update(
        use_cache=False,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=True,
    )
    versions_before = _state_versions(model)
    config_before = _json_sha256(_model_config(model))
    seed = _batch_seed(config, batch)
    try:
        with _deterministic_parent_execution(torch, model, seed):
            with torch.inference_mode():
                output = model(**forward_inputs)
    except ParentTraceCacheError:
        raise
    except Exception as exc:
        raise ParentTraceCaptureError(
            f"Parent forward failed for calibration batch {batch.index}."
        ) from exc
    versions_after = _state_versions(model)
    config_after = _json_sha256(_model_config(model))
    if versions_after != versions_before or config_after != config_before:
        raise ParentTraceCaptureError(
            "Parent state or configuration changed during trace collection."
        )
    return _extract_trace_tensors(torch, output, batch, config)


def _publish_batch(
    torch: Any,
    cache_path: Path,
    manifest: Mapping[str, Any],
    batch: _PreparedBatch,
    tensors: Mapping[str, Any],
    capture_metadata: Mapping[str, Any],
) -> Path:
    _safe_open, save_file = _safetensors()
    batches_path = cache_path / "batches"
    destination = _batch_dir(cache_path, batch.index)
    if destination.exists():
        raise ParentTraceCacheConflictError(
            f"Immutable batch {batch.index} already exists."
        )
    stage = batches_path / (
        f".batch-{batch.index:06d}.stage-{os.getpid()}-{uuid.uuid4().hex}"
    )
    stage.mkdir()
    try:
        trace_path = stage / TRACE_FILENAME
        tensor_dict = {name: tensor.contiguous() for name, tensor in tensors.items()}
        save_file(
            tensor_dict,
            str(trace_path),
            metadata={
                "schema": BATCH_SCHEMA,
                "cache_key": str(manifest["cache_key"]),
                "batch_index": str(batch.index),
                "input_sha256": batch.input_sha256,
                "selection_sha256": str(manifest["capture"]["selection_sha256"]),
            },
        )
        with trace_path.open("r+b") as handle:
            os.fsync(handle.fileno())
        inventory = {
            name: _tensor_descriptor(torch, tensor)
            for name, tensor in sorted(tensor_dict.items())
        }
        record_body = {
            "schema": BATCH_SCHEMA,
            "cache_key": manifest["cache_key"],
            "batch_index": batch.index,
            "input_sha256": batch.input_sha256,
            "forward_input_names": list(batch.forward_input_names),
            "available_hidden_state_count": capture_metadata[
                "available_hidden_state_count"
            ],
            "resolved_hidden_state_indices": capture_metadata[
                "resolved_hidden_state_indices"
            ],
            "logits": capture_metadata["logits"],
            "trace_file": {
                "name": TRACE_FILENAME,
                "size_bytes": trace_path.stat().st_size,
                "sha256": _file_sha256(trace_path),
            },
            "tensors": inventory,
        }
        record = dict(record_body)
        record["record_sha256"] = _json_sha256(record_body)
        _write_new_canonical_json(stage / BATCH_RECORD_FILENAME, record)
        os.rename(stage, destination)
        _fsync_directory(batches_path)
        return destination
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _record_body(record: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in record.items() if name != "record_sha256"}


def _validate_record_shape(
    record: Mapping[str, Any], manifest: Mapping[str, Any], index: int
) -> None:
    expected_keys = {
        "available_hidden_state_count",
        "batch_index",
        "cache_key",
        "forward_input_names",
        "input_sha256",
        "logits",
        "record_sha256",
        "resolved_hidden_state_indices",
        "schema",
        "tensors",
        "trace_file",
    }
    if set(record) != expected_keys or record.get("schema") != BATCH_SCHEMA:
        raise ParentTraceCacheCorruptionError(
            f"Batch {index} record schema is invalid."
        )
    if record.get("cache_key") != manifest["cache_key"]:
        raise ParentTraceCacheCorruptionError(
            f"Batch {index} belongs to a different cache."
        )
    if record.get("batch_index") != index:
        raise ParentTraceCacheCorruptionError(f"Batch {index} index is invalid.")
    expected_input = manifest["calibration"]["ordered_batch_input_sha256"][index]
    if record.get("input_sha256") != expected_input:
        raise ParentTraceCacheCorruptionError(
            f"Batch {index} input fingerprint is invalid."
        )
    digest = _validate_sha256(record.get("record_sha256"), "record_sha256")
    if _json_sha256(_record_body(record)) != digest:
        raise ParentTraceCacheCorruptionError(
            f"Batch {index} record fingerprint is invalid."
        )


def _validate_loaded_inputs(
    torch: Any,
    tensors: Mapping[str, Any],
    record: Mapping[str, Any],
    index: int,
) -> None:
    names = record.get("forward_input_names")
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ParentTraceCacheCorruptionError(
            f"Batch {index} forward input inventory is invalid."
        )
    input_tensors = {
        name.removeprefix("input."): tensor
        for name, tensor in tensors.items()
        if name.startswith("input.")
    }
    if set(input_tensors) != set(names) | {"valid_token_mask"}:
        raise ParentTraceCacheCorruptionError(
            f"Batch {index} persisted input inventory is invalid."
        )
    if _named_tensor_sha256(torch, input_tensors) != record["input_sha256"]:
        raise ParentTraceCacheCorruptionError(
            f"Batch {index} persisted inputs do not match their fingerprint."
        )
    input_ids = input_tensors.get("input_ids")
    attention = input_tensors.get("attention_mask")
    valid = input_tensors.get("valid_token_mask")
    if input_ids is None or input_ids.ndim != 2:
        raise ParentTraceCacheCorruptionError(f"Batch {index} input_ids are invalid.")
    shape = tuple(int(size) for size in input_ids.shape)
    try:
        _validate_binary_mask(torch, attention, shape, "attention_mask")
        _validate_binary_mask(torch, valid, shape, "valid_token_mask")
    except ParentTraceCaptureError as exc:
        raise ParentTraceCacheCorruptionError(
            f"Batch {index} token masks are invalid."
        ) from exc
    if bool((valid.to(torch.bool) & ~attention.to(torch.bool)).any().item()):
        raise ParentTraceCacheCorruptionError(
            f"Batch {index} valid token mask exceeds its attention mask."
        )


def _validate_loaded_trace_semantics(
    tensors: Mapping[str, Any],
    record: Mapping[str, Any],
    index: int,
) -> None:
    selected = record.get("resolved_hidden_state_indices")
    if (
        not isinstance(selected, list)
        or not selected
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in selected
        )
        or selected != sorted(set(selected))
    ):
        raise ParentTraceCacheCorruptionError(
            f"Batch {index} hidden-state selection is invalid."
        )
    expected_hidden = {f"hidden_state.{item:06d}" for item in selected}
    actual_hidden = {name for name in tensors if name.startswith("hidden_state.")}
    if actual_hidden != expected_hidden:
        raise ParentTraceCacheCorruptionError(
            f"Batch {index} hidden-state inventory is invalid."
        )
    logits = record.get("logits")
    if not isinstance(logits, Mapping):
        raise ParentTraceCacheCorruptionError(
            f"Batch {index} logits metadata is invalid."
        )
    mode = logits.get("mode")
    actual_logits = {name for name in tensors if name.startswith("logits.")}
    expected_logits = (
        {"logits.full"}
        if mode == "full"
        else {"logits.topk_values", "logits.topk_indices", "logits.logsumexp"}
    )
    if mode not in {"full", "topk"} or actual_logits != expected_logits:
        raise ParentTraceCacheCorruptionError(
            f"Batch {index} logits tensor inventory is invalid."
        )


def _read_verified_batch(
    cache_path: Path,
    manifest: Mapping[str, Any],
    index: int,
    completion_entry: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = _torch()
    safe_open, _save_file = _safetensors()
    directory = _batch_dir(cache_path, index)
    if not directory.is_dir() or directory.is_symlink():
        raise ParentTraceCacheCorruptionError(f"Committed batch {index} is missing.")
    children = {path.name for path in directory.iterdir()}
    if children != {TRACE_FILENAME, BATCH_RECORD_FILENAME}:
        raise ParentTraceCacheCorruptionError(
            f"Batch {index} contains unexpected artifacts."
        )
    trace_path = directory / TRACE_FILENAME
    record_path = directory / BATCH_RECORD_FILENAME
    if trace_path.is_symlink() or record_path.is_symlink():
        raise ParentTraceCacheCorruptionError(
            f"Batch {index} cannot contain symbolic links."
        )
    record = _read_canonical_json(record_path)
    _validate_record_shape(record, manifest, index)
    if completion_entry is not None:
        if completion_entry.get("record_file_sha256") != _file_sha256(record_path):
            raise ParentTraceCacheCorruptionError(
                f"Batch {index} record file hash differs from completion metadata."
            )
        if completion_entry.get("record_sha256") != record["record_sha256"]:
            raise ParentTraceCacheCorruptionError(
                f"Batch {index} record hash differs from completion metadata."
            )
    trace_file = record.get("trace_file")
    if not isinstance(trace_file, Mapping) or trace_file.get("name") != TRACE_FILENAME:
        raise ParentTraceCacheCorruptionError(
            f"Batch {index} trace file metadata is invalid."
        )
    try:
        trace_size = trace_path.stat().st_size
    except OSError as exc:
        raise ParentTraceCacheCorruptionError(
            f"Cannot stat batch {index} trace file."
        ) from exc
    trace_sha256 = _file_sha256(trace_path)
    if trace_size != trace_file.get("size_bytes") or trace_sha256 != trace_file.get(
        "sha256"
    ):
        raise ParentTraceCacheCorruptionError(
            f"Batch {index} safetensors file is truncated or corrupt."
        )
    if (
        completion_entry is not None
        and completion_entry.get("trace_file_sha256") != trace_sha256
    ):
        raise ParentTraceCacheCorruptionError(
            f"Batch {index} trace hash differs from completion metadata."
        )

    try:
        with safe_open(str(trace_path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            keys = list(handle.keys())
            tensors = {name: handle.get_tensor(name) for name in keys}
    except Exception as exc:
        raise ParentTraceCacheCorruptionError(
            f"Cannot decode batch {index} safetensors file."
        ) from exc
    expected_metadata = {
        "schema": BATCH_SCHEMA,
        "cache_key": str(manifest["cache_key"]),
        "batch_index": str(index),
        "input_sha256": str(record["input_sha256"]),
        "selection_sha256": str(manifest["capture"]["selection_sha256"]),
    }
    if metadata != expected_metadata:
        raise ParentTraceCacheCorruptionError(
            f"Batch {index} safetensors metadata is invalid."
        )
    inventory = record.get("tensors")
    if not isinstance(inventory, Mapping) or set(inventory) != set(tensors):
        raise ParentTraceCacheCorruptionError(
            f"Batch {index} tensor inventory is invalid."
        )
    for name, tensor in tensors.items():
        if inventory[name] != _tensor_descriptor(torch, tensor):
            raise ParentTraceCacheCorruptionError(
                f"Batch {index} tensor {name!r} failed integrity verification."
            )
    _validate_loaded_inputs(torch, tensors, record, index)
    _validate_loaded_trace_semantics(tensors, record, index)
    return tensors, record


def _completion_for(cache_path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    batches = []
    count = int(manifest["calibration"]["batch_count"])
    for index in range(count):
        directory = _batch_dir(cache_path, index)
        record_path = directory / BATCH_RECORD_FILENAME
        record = _read_canonical_json(record_path)
        batches.append(
            {
                "batch_index": index,
                "input_sha256": record["input_sha256"],
                "record_sha256": record["record_sha256"],
                "record_file_sha256": _file_sha256(record_path),
                "trace_file_sha256": record["trace_file"]["sha256"],
            }
        )
    body = {
        "schema": COMPLETION_SCHEMA,
        "cache_key": manifest["cache_key"],
        "calibration_sha256": manifest["calibration"]["calibration_sha256"],
        "batch_count": count,
        "batches": batches,
    }
    result = dict(body)
    result["completion_sha256"] = _json_sha256(body)
    return result


def _validate_completion(
    completion: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    expected_keys = {
        "batch_count",
        "batches",
        "cache_key",
        "calibration_sha256",
        "completion_sha256",
        "schema",
    }
    if (
        set(completion) != expected_keys
        or completion.get("schema") != COMPLETION_SCHEMA
    ):
        raise ParentTraceCacheCorruptionError("Cache completion schema is invalid.")
    body = {
        name: value for name, value in completion.items() if name != "completion_sha256"
    }
    digest = _validate_sha256(completion.get("completion_sha256"), "completion_sha256")
    if _json_sha256(body) != digest:
        raise ParentTraceCacheCorruptionError(
            "Cache completion fingerprint is invalid."
        )
    expected_count = int(manifest["calibration"]["batch_count"])
    batches = completion.get("batches")
    if (
        completion.get("cache_key") != manifest["cache_key"]
        or completion.get("calibration_sha256")
        != manifest["calibration"]["calibration_sha256"]
        or completion.get("batch_count") != expected_count
        or not isinstance(batches, list)
        or len(batches) != expected_count
    ):
        raise ParentTraceCacheCorruptionError(
            "Cache completion does not match the manifest."
        )
    for index, entry in enumerate(batches):
        expected_entry_keys = {
            "batch_index",
            "input_sha256",
            "record_file_sha256",
            "record_sha256",
            "trace_file_sha256",
        }
        if (
            not isinstance(entry, Mapping)
            or set(entry) != expected_entry_keys
            or entry.get("batch_index") != index
        ):
            raise ParentTraceCacheCorruptionError(
                f"Cache completion batch {index} is invalid."
            )
        if (
            entry.get("input_sha256")
            != manifest["calibration"]["ordered_batch_input_sha256"][index]
        ):
            raise ParentTraceCacheCorruptionError(
                f"Cache completion input hash for batch {index} is invalid."
            )
        for name in (
            "record_sha256",
            "record_file_sha256",
            "trace_file_sha256",
        ):
            _validate_sha256(entry.get(name), f"completion batch {index} {name}")
    return tuple(batches)


@dataclass(frozen=True, slots=True)
class ParentTraceBatch:
    """One verified, detached CPU batch loaded from the immutable cache."""

    index: int
    input_sha256: str
    _tensors: Mapping[str, Any]
    _record: Mapping[str, Any]

    @property
    def tensors(self) -> Mapping[str, Any]:
        return self._tensors

    @property
    def record(self) -> dict[str, Any]:
        return json.loads(_canonical_json(dict(self._record)))

    @property
    def model_inputs(self) -> Mapping[str, Any]:
        names = self._record["forward_input_names"]
        return MappingProxyType(
            {name: self._tensors[f"input.{name}"] for name in names}
        )

    @property
    def input_ids(self) -> Any:
        return self._tensors["input.input_ids"]

    @property
    def attention_mask(self) -> Any:
        return self._tensors["input.attention_mask"]

    @property
    def valid_token_mask(self) -> Any:
        return self._tensors["input.valid_token_mask"]

    @property
    def hidden_states(self) -> Mapping[int, Any]:
        values = {
            int(name.removeprefix("hidden_state.")): tensor
            for name, tensor in self._tensors.items()
            if name.startswith("hidden_state.")
        }
        return MappingProxyType(dict(sorted(values.items())))

    @property
    def layer_boundary_states(self) -> Mapping[int, Any]:
        return self.hidden_states

    @property
    def logits(self) -> Any | None:
        return self._tensors.get("logits.full")

    @property
    def topk_values(self) -> Any | None:
        return self._tensors.get("logits.topk_values")

    @property
    def topk_indices(self) -> Any | None:
        return self._tensors.get("logits.topk_indices")

    @property
    def logits_logsumexp(self) -> Any | None:
        return self._tensors.get("logits.logsumexp")


@dataclass(frozen=True, slots=True)
class ParentTraceCache:
    """Read-only handle for a verified parent activation cache."""

    path: Path
    _manifest: Mapping[str, Any]
    _completion: Mapping[str, Any] | None

    @classmethod
    def open(
        cls,
        path: str | os.PathLike[str],
        *,
        verify: bool = True,
        require_complete: bool = True,
    ) -> ParentTraceCache:
        cache_path = Path(path).expanduser().absolute()
        if not cache_path.is_dir() or cache_path.is_symlink():
            raise ParentTraceCacheCorruptionError(
                f"Trace cache directory does not exist: {cache_path}."
            )
        manifest = _read_canonical_json(cache_path / MANIFEST_FILENAME)
        _validate_manifest(manifest)
        completion_path = cache_path / COMPLETION_FILENAME
        completion = (
            _read_canonical_json(completion_path) if completion_path.exists() else None
        )
        if require_complete and completion is None:
            raise ParentTraceCacheCorruptionError(
                f"Trace cache {cache_path} is incomplete."
            )
        result = cls(
            path=cache_path,
            _manifest=MappingProxyType(manifest),
            _completion=(None if completion is None else MappingProxyType(completion)),
        )
        if verify:
            result.verify()
        return result

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(_canonical_json(dict(self._manifest)))

    @property
    def cache_key(self) -> str:
        return str(self._manifest["cache_key"])

    @property
    def parent_fingerprint(self) -> str:
        return str(self._manifest["parent"]["identity_sha256"])

    @property
    def calibration_fingerprint(self) -> str:
        return str(self._manifest["calibration"]["calibration_sha256"])

    @property
    def batch_count(self) -> int:
        return int(self._manifest["calibration"]["batch_count"])

    @property
    def is_complete(self) -> bool:
        return self._completion is not None

    def __len__(self) -> int:
        return self.batch_count

    def _completion_entries(self) -> tuple[Mapping[str, Any], ...] | None:
        if self._completion is None:
            if (self.path / COMPLETION_FILENAME).exists():
                raise ParentTraceCacheCorruptionError(
                    "Cache completion state changed after the cache was opened."
                )
            return None
        completion = _read_canonical_json(self.path / COMPLETION_FILENAME)
        if completion.get("completion_sha256") != self._completion.get(
            "completion_sha256"
        ):
            raise ParentTraceCacheCorruptionError(
                "Cache completion metadata changed after the cache was opened."
            )
        return _validate_completion(completion, self._manifest)

    def verify(self) -> ParentTraceCache:
        fresh_manifest = _read_canonical_json(self.path / MANIFEST_FILENAME)
        _validate_manifest(fresh_manifest)
        if fresh_manifest["cache_key"] != self.cache_key:
            raise ParentTraceCacheCorruptionError(
                "Cache manifest changed after the cache was opened."
            )
        entries = self._completion_entries()
        _assert_cache_layout(self.path, complete=entries is not None)
        if entries is None:
            for path in (self.path / "batches").iterdir():
                match = _BATCH_NAME.fullmatch(path.name)
                if match is None:
                    raise ParentTraceCacheCorruptionError(
                        f"Unexpected incomplete cache artifact {path.name!r}."
                    )
                index = int(match.group(1))
                if index >= self.batch_count:
                    raise ParentTraceCacheCorruptionError(
                        f"Unexpected batch index {index}."
                    )
                _read_verified_batch(self.path, self._manifest, index, None)
            return self
        if len(list((self.path / "batches").iterdir())) != self.batch_count:
            raise ParentTraceCacheCorruptionError(
                "Completed cache batch count differs from its manifest."
            )
        for index, entry in enumerate(entries):
            _read_verified_batch(self.path, self._manifest, index, entry)
        return self

    def load_batch(self, index: int) -> ParentTraceBatch:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("batch index must be an integer.")
        if index < 0 or index >= self.batch_count:
            raise IndexError(
                f"batch index must be in [0, {self.batch_count}), got {index}."
            )
        entries = self._completion_entries()
        entry = None if entries is None else entries[index]
        tensors, record = _read_verified_batch(
            self.path,
            self._manifest,
            index,
            entry,
        )
        return ParentTraceBatch(
            index=index,
            input_sha256=str(record["input_sha256"]),
            _tensors=MappingProxyType(tensors),
            _record=MappingProxyType(record),
        )

    def iter_batches(self) -> Iterator[ParentTraceBatch]:
        for index in range(self.batch_count):
            yield self.load_batch(index)


def open_parent_trace_cache(
    path: str | os.PathLike[str],
    *,
    verify: bool = True,
) -> ParentTraceCache:
    """Open a complete local cache without loading or running its parent."""

    return ParentTraceCache.open(path, verify=verify, require_complete=True)


def _rollback_new_batches(cache_path: Path, paths: Sequence[Path]) -> None:
    batches_root = (cache_path / "batches").resolve()
    for path in reversed(paths):
        resolved = path.resolve()
        if resolved.parent != batches_root or _BATCH_NAME.fullmatch(path.name) is None:
            raise ParentTraceCacheError(
                f"Refusing to roll back an unexpected batch path: {path}."
            )
        if resolved.is_dir() and not resolved.is_symlink():
            shutil.rmtree(resolved)


def build_parent_trace_cache(
    cache_root: str | os.PathLike[str],
    model: Any,
    calibration_batches: Iterable[Mapping[str, Any]],
    *,
    config: ParentTraceCaptureConfig | None = None,
    input_device: Any | None = None,
) -> ParentTraceCache:
    """Create or resume a content-addressed cache and return a verified handle.

    The returned directory is ``cache_root / cache_key``.  A second call with
    the same parent, ordered inputs, selection, and runtime performs no parent
    forwards.  An interrupted call verifies and skips every committed batch,
    then captures only missing indices.  Any corruption fails closed and is
    never silently replaced.
    """

    torch = _torch()
    _safetensors()
    if config is None:
        selected_config = ParentTraceCaptureConfig()
    elif isinstance(config, ParentTraceCaptureConfig):
        selected_config = config
    else:
        raise TypeError("config must be ParentTraceCaptureConfig or None.")
    prepared = _prepare_batches(torch, calibration_batches)
    parent_before = fingerprint_parent_model(model)
    resolved_input_device = _resolve_input_device(torch, model, input_device)
    manifest = _manifest_for(
        parent_before,
        prepared,
        selected_config,
        _runtime_identity(torch, model, resolved_input_device),
    )
    root = Path(cache_root).expanduser().absolute()
    if root.exists() and (not root.is_dir() or root.is_symlink()):
        raise ParentTraceCacheError(f"cache_root must be a local directory: {root}.")
    root.mkdir(parents=True, exist_ok=True)
    cache_path = root / str(manifest["cache_key"])
    cache_path.mkdir(exist_ok=True)
    if not cache_path.is_dir() or cache_path.is_symlink():
        raise ParentTraceCacheError(
            f"Content-addressed cache path must be a local directory: {cache_path}."
        )

    published: list[Path] = []
    with _exclusive_cache_lock(cache_path / _LOCK_FILENAME):
        _clean_metadata_staging_files(cache_path)
        manifest_path = cache_path / MANIFEST_FILENAME
        if manifest_path.exists():
            existing_manifest = _read_canonical_json(manifest_path)
            _validate_manifest(existing_manifest)
            _manifest_matches(existing_manifest, manifest)
        else:
            unexpected = {
                path.name
                for path in cache_path.iterdir()
                if path.name != _LOCK_FILENAME
            }
            if unexpected:
                raise ParentTraceCacheCorruptionError(
                    "Uninitialized cache directory contains unexpected artifacts."
                )
            _write_new_canonical_json(manifest_path, manifest)
        batches_path = cache_path / "batches"
        batches_path.mkdir(exist_ok=True)
        _clean_staging_directories(cache_path)

        completion_path = cache_path / COMPLETION_FILENAME
        if completion_path.exists():
            cache = ParentTraceCache.open(cache_path, verify=True)
            _manifest_matches(cache._manifest, manifest)
            return cache

        _assert_cache_layout(cache_path, complete=False)
        for path in batches_path.iterdir():
            match = _BATCH_NAME.fullmatch(path.name)
            if match is None or int(match.group(1)) >= len(prepared):
                raise ParentTraceCacheCorruptionError(
                    f"Unexpected committed batch artifact {path.name!r}."
                )

        try:
            for batch in prepared:
                destination = _batch_dir(cache_path, batch.index)
                if destination.exists():
                    _read_verified_batch(
                        cache_path,
                        manifest,
                        batch.index,
                        None,
                    )
                    continue
                tensors, capture_metadata = _capture_batch(
                    torch,
                    model,
                    batch,
                    selected_config,
                    resolved_input_device,
                )
                published.append(
                    _publish_batch(
                        torch,
                        cache_path,
                        manifest,
                        batch,
                        tensors,
                        capture_metadata,
                    )
                )
            parent_after = fingerprint_parent_model(model)
            if parent_after != parent_before:
                _rollback_new_batches(cache_path, published)
                published.clear()
                raise ParentTraceCaptureError(
                    "Parent fingerprint changed during cache construction."
                )
            for batch in prepared:
                _read_verified_batch(
                    cache_path,
                    manifest,
                    batch.index,
                    None,
                )
            completion = _completion_for(cache_path, manifest)
            _write_new_canonical_json(completion_path, completion)
        except Exception:
            _clean_staging_directories(cache_path)
            raise

        return ParentTraceCache.open(cache_path, verify=True)


capture_parent_trace_cache = build_parent_trace_cache


__all__ = [
    "BATCH_RECORD_FILENAME",
    "BATCH_SCHEMA",
    "CACHE_SCHEMA",
    "COMPLETION_FILENAME",
    "COMPLETION_SCHEMA",
    "MANIFEST_FILENAME",
    "TRACE_FILENAME",
    "ParentModelFingerprint",
    "ParentTraceBatch",
    "ParentTraceCache",
    "ParentTraceCacheConflictError",
    "ParentTraceCacheCorruptionError",
    "ParentTraceCacheError",
    "ParentTraceCaptureConfig",
    "ParentTraceCaptureError",
    "build_parent_trace_cache",
    "capture_parent_trace_cache",
    "fingerprint_parent_model",
    "open_parent_trace_cache",
]
