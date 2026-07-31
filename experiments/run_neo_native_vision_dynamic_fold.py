#!/usr/bin/env python3
"""Fold the validated local NEO native-vision parent without training.

The runner imports the checkpoint's concrete Python classes from its local
snapshot, streams local safetensors into them, authenticates the existing
256-output parent cache, captures calibration traces once, and delegates the
actual one-action fold/publication transaction to the generic dynamic system.
It never uses an automatic model/tokenizer loader, a hub identifier, an
optimizer, backward, or regenerated parent answer outputs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import shutil
import sys
import tempfile
import threading
import types
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import torch
from torch import Tensor, nn

try:
    from experiments import eval_neo_native_vision as native_eval
except ImportError:  # Direct execution from the experiments directory.
    import eval_neo_native_vision as native_eval

from neural_forge.dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from neural_forge.dynamic_fold_registry import (
    DurableExperimentLedger,
    ExperimentLedger,
)
from neural_forge.dynamic_fold_snapshot import DynamicFoldSnapshotConfig
from neural_forge.dynamic_model_inspector import inspect_model_circuits
from neural_forge.dynamic_neo_adapter import NeoNativeVisionTargetSynthesisAdapter
from neural_forge.dynamic_neo_fold import (
    METHOD_ID,
    NeoNativeVisionBehaviorGate,
    NeoNativeVisionFoldCache,
    NeoNativeVisionFoldStackAdapter,
    NeoNativeVisionStackResources,
    NeoVisionBehaviorRecord,
    build_neo_native_vision_fold_cache,
    derive_neo_native_vision_fold_cache,
    load_neo_parent_behavior_cache,
    write_neo_dynamic_fold_receipt,
)
from neural_forge.dynamic_production_catalog import (
    ProductionDynamicFoldCatalog,
    ProductionFoldBundle,
    ProductionFoldCandidateSpec,
    ProductionFoldCapability,
)
from neural_forge.dynamic_search import DynamicFoldSearchObjective
from neural_forge.dynamic_target_synthesis import (
    DynamicTargetRequest,
)


PARENT_UNIQUE_PARAMETERS = 2_896_618_496
TARGET_UNIQUE_PARAMETERS = 2_896_372_736
SOURCE_INTERMEDIATE_SIZE = 6144
TARGET_INTERMEDIATE_SIZE = 6143
EXACT_PARAMETER_DELTA = 245_760
DECODER_LAYERS = 40
DECODER_HIDDEN_SIZE = 2048
PARENT_NAME = "neo1.5-2b-parent"
PINNED_MANIFEST_SHA256 = (
    "76152804adba32da5e51b4ffccf5b9fe18c9b079dec8a6ee72553e88e1ac4811"
)
PINNED_PARENT_OUTPUTS_SHA256 = (
    "d02b28bbc91b5206dc7f86b48ba7743aa66a334a037ea390fc702259fd2d9e7f"
)
DIRECT_SAVER_ID = "neo-local-concrete-sharded-safetensors-v1"
DIRECT_LOADER_ID = "neo-local-concrete-sharded-safetensors-v1"
FAILURE_RECEIPT_SCHEMA = "neural-forge-production-neo-failed-closed-v1"
PINNED_RUNTIME = {
    "python": "3.12.3",
    "torch": "2.11.0+cu128",
    "torch_cuda": "12.8",
    "transformers": "4.57.3",
    "pillow": "12.2.0",
    "numpy": "2.5.1",
}
_DEFAULT_DTYPE_LOCK = threading.RLock()
_RUNTIME_FILES = (
    "__init__.py",
    "configuration_neo_chat.py",
    "configuration_neo_vit.py",
    "conversation.py",
    "modeling_neo_chat.py",
    "modeling_neo_vit.py",
    "modeling_qwen3.py",
)
_TOKENIZER_FILES = (
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)


class ProductionNeoFoldError(RuntimeError):
    """The concrete local NEO run cannot be authenticated or completed."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProductionNeoFoldError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProductionNeoFoldError(f"{name} must be a non-negative integer.")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionNeoFoldError(f"Could not read {name}: {path}.") from exc
    if not isinstance(value, Mapping):
        raise ProductionNeoFoldError(f"{name} must contain one JSON object.")
    return dict(value)


def _resolve_local_directory(
    value: str | os.PathLike[str],
    name: str,
) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise ProductionNeoFoldError(
            f"{name} must be an explicit local directory, not a URL."
        )
    path = Path(raw).expanduser()
    if not path.is_dir() or path.is_symlink():
        raise ProductionNeoFoldError(f"{name} directory is missing or unsafe.")
    return path.resolve(strict=True)


def _resolve_local_file(value: str | os.PathLike[str], name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise ProductionNeoFoldError(
            f"{name} must be an explicit local file, not a URL."
        )
    path = Path(raw).expanduser()
    if not path.is_file() or path.is_symlink():
        raise ProductionNeoFoldError(f"{name} file is missing or unsafe.")
    return path.resolve(strict=True)


def _write_atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().absolute()
    if destination.exists():
        raise ProductionNeoFoldError(f"Refusing to overwrite receipt: {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(dict(value)) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _serialized_model_config(config: Any) -> dict[str, Any]:
    """Use the concrete Transformers JSON encoder before canonical writing."""

    to_json_string = getattr(config, "to_json_string", None)
    if not callable(to_json_string):
        raise ProductionNeoFoldError(
            "NEO configuration does not expose to_json_string()."
        )
    try:
        encoded = to_json_string(use_diff=False)
    except TypeError:
        encoded = to_json_string()
    if not isinstance(encoded, str):
        raise ProductionNeoFoldError(
            "NEO configuration JSON serializer returned non-text."
        )
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ProductionNeoFoldError(
            "NEO configuration serializer returned invalid JSON."
        ) from exc
    if not isinstance(payload, Mapping):
        raise ProductionNeoFoldError("NEO serialized configuration must be one object.")
    normalized = dict(payload)
    canonical_json_bytes(normalized)
    return normalized


def _standalone_neo_model_config(
    model: nn.Module,
    source_snapshot: Path,
) -> dict[str, Any]:
    """Preserve the pinned custom config's constructor input representation."""

    source = _read_json_object(
        source_snapshot / "config.json",
        "source NEO config",
    )
    source_llm = source.get("llm_config")
    current_llm = getattr(getattr(model, "config", None), "llm_config", None)
    source_width = (
        source_llm.get("intermediate_size") if isinstance(source_llm, Mapping) else None
    )
    target_width = getattr(current_llm, "intermediate_size", None)
    if (
        isinstance(source_width, bool)
        or not isinstance(source_width, int)
        or isinstance(target_width, bool)
        or not isinstance(target_width, int)
        or target_width <= 0
        or target_width > source_width
    ):
        raise ProductionNeoFoldError(
            "Standalone NEO config requires a positive nonexpanding decoder width."
        )
    payload = json.loads(json.dumps(source))
    payload["llm_config"]["intermediate_size"] = target_width
    if (
        not isinstance(payload.get("vision_config"), Mapping)
        or isinstance(payload["vision_config"].get("downsample_ratio"), list)
        or isinstance(payload["vision_config"].get("llm_hidden_size"), list)
    ):
        raise ProductionNeoFoldError(
            "Pinned NEO source config lost its scalar custom-constructor fields."
        )
    canonical_json_bytes(payload)
    return payload


@contextmanager
def _temporary_default_dtype(dtype: torch.dtype):
    with _DEFAULT_DTYPE_LOCK:
        previous = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            yield
        finally:
            torch.set_default_dtype(previous)


def _torch_dtype(name: str) -> torch.dtype:
    normalized = str(name).strip().lower()
    if normalized != "bfloat16":
        raise ProductionNeoFoldError(
            "The pinned parent-output evidence was generated in bfloat16; "
            "production folding must use bfloat16."
        )
    return torch.bfloat16


def validate_pinned_production_runtime() -> Mapping[str, str]:
    """Require the exact software runtime that produced the cached baseline."""

    observed = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_cuda": str(getattr(torch.version, "cuda", None)),
        "transformers": importlib.metadata.version("transformers"),
        "pillow": importlib.metadata.version("Pillow"),
        "numpy": importlib.metadata.version("numpy"),
    }
    if observed != PINNED_RUNTIME:
        differences = {
            name: {
                "expected": PINNED_RUNTIME[name],
                "observed": observed[name],
            }
            for name in PINNED_RUNTIME
            if observed[name] != PINNED_RUNTIME[name]
        }
        raise ProductionNeoFoldError(
            "Production NEO runtime differs from the cached-parent baseline: "
            + json.dumps(differences, sort_keys=True)
        )
    return observed


@dataclass(frozen=True, slots=True)
class LocalNeoRuntime:
    directory: Path
    package_name: str
    config_class: type
    model_class: type
    source_fingerprint: str


def load_local_neo_runtime(
    model_directory: str | os.PathLike[str],
) -> LocalNeoRuntime:
    """Import concrete checkpoint classes from one explicit local directory."""

    directory = _resolve_local_directory(model_directory, "NEO model")
    files = []
    for filename in _RUNTIME_FILES:
        path = directory / filename
        if not path.is_file() or path.is_symlink():
            raise ProductionNeoFoldError(
                f"Local NEO runtime file is missing or unsafe: {path}."
            )
        files.append(
            {
                "path": filename,
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    source_fingerprint = canonical_sha256(
        {
            "schema": "neural-forge-local-neo-runtime-v1",
            "files": files,
        }
    )
    package_name = (
        "_neural_forge_neo_"
        + canonical_sha256(
            {
                "directory": str(directory),
                "source_fingerprint": source_fingerprint,
            }
        )[:20]
    )
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__file__ = str(directory / "__init__.py")
        package.__package__ = package_name
        package.__path__ = [str(directory)]
        sys.modules[package_name] = package
    config_module = importlib.import_module(f"{package_name}.configuration_neo_chat")
    model_module = importlib.import_module(f"{package_name}.modeling_neo_chat")
    config_class = getattr(config_module, "NEOChatConfig", None)
    model_class = getattr(model_module, "NEOChatModel", None)
    if not isinstance(config_class, type) or not isinstance(model_class, type):
        raise ProductionNeoFoldError(
            "Local NEO runtime does not expose its concrete config/model classes."
        )
    return LocalNeoRuntime(
        directory=directory,
        package_name=package_name,
        config_class=config_class,
        model_class=model_class,
        source_fingerprint=source_fingerprint,
    )


def load_local_neo_config(runtime: LocalNeoRuntime) -> Any:
    payload = _read_json_object(runtime.directory / "config.json", "NEO config")
    if (
        payload.get("model_type") != "neo_chat"
        or not isinstance(payload.get("llm_config"), Mapping)
        or payload["llm_config"].get("model_type") != "qwen3"
    ):
        raise ProductionNeoFoldError(
            "Local config is not the native neo_chat/Qwen3 architecture."
        )
    return runtime.config_class(**payload)


@dataclass(frozen=True, slots=True)
class LocalSafetensorLayout:
    shards: tuple[Path, ...]
    weight_map: Mapping[str, str] | None


def _local_safetensor_layout(directory: Path) -> LocalSafetensorLayout:
    index_path = directory / "model.safetensors.index.json"
    weight_map: dict[str, str] | None = None
    if index_path.is_file():
        index = _read_json_object(index_path, "NEO safetensors index")
        raw_map = index.get("weight_map")
        if not isinstance(raw_map, Mapping) or not raw_map:
            raise ProductionNeoFoldError(
                "Local NEO safetensors index has no weight_map."
            )
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
                raise ProductionNeoFoldError(
                    "NEO safetensors index contains an unsafe entry."
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
    if not shards or any(not shard.is_file() or shard.is_symlink() for shard in shards):
        raise ProductionNeoFoldError(
            f"Local NEO safetensors are incomplete in {directory}."
        )
    return LocalSafetensorLayout(shards=shards, weight_map=weight_map)


def _storage_identity(value: Tensor) -> tuple[int, int, tuple[int, ...]]:
    return (
        int(value.untyped_storage().data_ptr()),
        int(value.storage_offset()),
        tuple(int(size) for size in value.shape),
    )


def _tensor_equal_across_devices(target: Tensor, source: Tensor) -> bool:
    if tuple(target.shape) != tuple(source.shape) or target.dtype != source.dtype:
        return False
    target_flat = target.detach().reshape(-1)
    source_flat = source.detach().reshape(-1)
    elements = max(1, (32 * 1024**2) // max(1, target.element_size()))
    for start in range(0, int(target_flat.numel()), elements):
        stop = min(int(target_flat.numel()), start + elements)
        if not torch.equal(
            target_flat[start:stop].to(device="cpu"),
            source_flat[start:stop].to(device="cpu"),
        ):
            return False
    return True


def _rebuild_runtime_buffers(model: nn.Module, device: torch.device) -> None:
    try:
        embeddings = model.vision_model.embeddings
        layers = tuple(model.language_model.model.layers)
    except (AttributeError, TypeError) as exc:
        raise ProductionNeoFoldError(
            "Concrete NEO runtime lacks its vision/decoder buffer owners."
        ) from exc
    vision_module = sys.modules.get(type(embeddings).__module__)
    precompute = getattr(vision_module, "precompute_rope_freqs_sincos", None)
    if not callable(precompute):
        raise ProductionNeoFoldError(
            "Concrete NEO vision runtime lacks rotary precomputation."
        )
    config = embeddings.config
    for axis in ("x", "y"):
        cos, sin = precompute(
            embeddings.rope_dim_part,
            config.max_position_embeddings_vision,
            base=config.rope_theta_vision,
            device=device,
        )
        setattr(embeddings, f"cos_cached_{axis}", cos)
        setattr(embeddings, f"sin_cached_{axis}", sin)
    for layer in layers:
        attention = layer.self_attn
        for name in ("rotary_emb", "rotary_emb_hw"):
            incumbent = getattr(attention, name, None)
            incumbent_config = getattr(incumbent, "config", None)
            if incumbent is None or incumbent_config is None:
                raise ProductionNeoFoldError(
                    "Concrete NEO attention lacks temporal/spatial rotary state."
                )
            setattr(
                attention,
                name,
                type(incumbent)(incumbent_config, device=device),
            )


def load_local_neo_model(
    model_directory: str | os.PathLike[str],
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """Stream local safetensors into the concrete checkpoint class."""

    from safetensors import safe_open

    runtime = load_local_neo_runtime(model_directory)
    config = load_local_neo_config(runtime)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ProductionNeoFoldError("CUDA was requested but is unavailable.")
    with _temporary_default_dtype(dtype), torch.device("meta"):
        model = runtime.model_class(config)
    model.to_empty(device=target_device)
    _rebuild_runtime_buffers(model, target_device)
    target_state = model.state_dict()
    layout = _local_safetensor_layout(runtime.directory)
    loaded: set[str] = set()
    unexpected = []
    loaded_storage: dict[tuple[int, int, tuple[int, ...]], str] = {}
    with torch.no_grad():
        for shard in layout.shards:
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if name in loaded:
                        raise ProductionNeoFoldError(
                            f"Duplicate NEO tensor across shards: {name}."
                        )
                    if (
                        layout.weight_map is not None
                        and layout.weight_map.get(name) != shard.name
                    ):
                        raise ProductionNeoFoldError(
                            f"NEO index maps {name!r} to the wrong shard."
                        )
                    if name not in target_state:
                        unexpected.append(name)
                        continue
                    target = target_state[name]
                    identity = _storage_identity(target)
                    source = handle.get_tensor(name)
                    if identity in loaded_storage:
                        if not _tensor_equal_across_devices(target, source):
                            raise ProductionNeoFoldError(
                                "Serialized NEO aliases disagree for "
                                f"{loaded_storage[identity]!r} and {name!r}."
                            )
                        loaded.add(name)
                        continue
                    if (
                        tuple(source.shape) != tuple(target.shape)
                        or source.dtype != target.dtype
                    ):
                        raise ProductionNeoFoldError(
                            f"Local NEO tensor contract mismatch for {name}: "
                            f"{tuple(source.shape)}/{source.dtype} != "
                            f"{tuple(target.shape)}/{target.dtype}."
                        )
                    target.copy_(source.to(device=target.device))
                    loaded.add(name)
                    loaded_storage[identity] = name
    if unexpected:
        raise ProductionNeoFoldError(
            f"Local NEO safetensors contain unexpected keys: {unexpected[:8]}."
        )
    if layout.weight_map is not None and set(layout.weight_map) != loaded:
        raise ProductionNeoFoldError(
            "NEO safetensors index differs from the loaded key inventory."
        )
    missing = sorted(
        name
        for name, value in target_state.items()
        if name not in loaded and _storage_identity(value) not in loaded_storage
    )
    if missing:
        raise ProductionNeoFoldError(
            f"Local NEO safetensors are missing keys: {missing[:8]}."
        )
    if any(parameter.device.type == "meta" for parameter in model.parameters()) or any(
        buffer.device.type == "meta" for buffer in model.buffers()
    ):
        raise ProductionNeoFoldError("Direct NEO loading left meta tensors.")
    native_eval._force_neo_eager_prefill_attention(model)
    # Fold-only runtime: this loader is the published artifact's inference
    # contract, so it must return frozen parameters. eval() only sets module
    # training mode and leaves requires_grad untouched, which fails
    # _assert_frozen_inference_parameters during artifact publication.
    model.requires_grad_(False)
    model.eval()
    return model


def _added_token(value: Any) -> Any:
    from transformers import AddedToken

    if not isinstance(value, Mapping):
        return value
    if "content" not in value:
        raise ProductionNeoFoldError("Added-token object lacks content.")
    return AddedToken(
        str(value["content"]),
        single_word=bool(value.get("single_word", False)),
        lstrip=bool(value.get("lstrip", False)),
        rstrip=bool(value.get("rstrip", False)),
        normalized=bool(value.get("normalized", True)),
        special=bool(value.get("special", False)),
    )


def load_local_neo_tokenizer(
    model_directory: str | os.PathLike[str],
) -> Any:
    """Construct Qwen2Tokenizer directly from local vocab/merges/config."""

    # Import the concrete implementation module instead of the Transformers
    # root lazy-module facade.  The facade can retain a previously resolved
    # class across unrelated model-family work in the same process.
    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

    directory = _resolve_local_directory(model_directory, "NEO tokenizer")
    vocab = directory / "vocab.json"
    merges = directory / "merges.txt"
    if not vocab.is_file() or not merges.is_file():
        raise ProductionNeoFoldError(
            "Local NEO tokenizer lacks vocab.json or merges.txt."
        )
    config = _read_json_object(
        directory / "tokenizer_config.json",
        "NEO tokenizer config",
    )
    if config.get("tokenizer_class") != "Qwen2Tokenizer":
        raise ProductionNeoFoldError(
            "Pinned NEO tokenizer is not the concrete Qwen2 tokenizer."
        )
    kwargs: dict[str, Any] = {}
    for name in (
        "errors",
        "unk_token",
        "bos_token",
        "eos_token",
        "pad_token",
        "add_prefix_space",
        "add_bos_token",
        "add_eos_token",
        "model_max_length",
        "clean_up_tokenization_spaces",
        "split_special_tokens",
        "chat_template",
    ):
        if name in config:
            kwargs[name] = config[name]
    decoder = config.get("added_tokens_decoder", {})
    if not isinstance(decoder, Mapping):
        raise ProductionNeoFoldError("NEO tokenizer added_tokens_decoder is malformed.")
    kwargs["added_tokens_decoder"] = {
        int(index): _added_token(value) for index, value in decoder.items()
    }
    additional = config.get("additional_special_tokens", ())
    if not isinstance(additional, Sequence) or isinstance(
        additional,
        (str, bytes),
    ):
        raise ProductionNeoFoldError(
            "NEO tokenizer additional_special_tokens is malformed."
        )
    kwargs["additional_special_tokens"] = [_added_token(value) for value in additional]
    parameters = inspect.signature(Qwen2Tokenizer.__init__).parameters
    if {"vocab_file", "merges_file"} <= set(parameters):
        return Qwen2Tokenizer(
            vocab_file=str(vocab),
            merges_file=str(merges),
            **kwargs,
        )
    if {"vocab", "merges"} <= set(parameters):
        return Qwen2Tokenizer(
            vocab=str(vocab),
            merges=str(merges),
            **kwargs,
        )
    raise ProductionNeoFoldError(
        "Installed concrete Qwen2Tokenizer has an unsupported constructor."
    )


@dataclass(frozen=True, slots=True)
class LocalTokenizerIdentity:
    fingerprint: str
    vocab_size: int
    special_token_ids: Mapping[str, int]
    file_sha256: Mapping[str, str]


def fingerprint_local_neo_tokenizer(
    model_directory: str | os.PathLike[str],
    tokenizer: Any,
) -> LocalTokenizerIdentity:
    directory = _resolve_local_directory(model_directory, "NEO tokenizer")
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise ProductionNeoFoldError("NEO tokenizer does not expose get_vocab().")
    vocabulary = get_vocab()
    if not isinstance(vocabulary, Mapping) or not vocabulary:
        raise ProductionNeoFoldError("NEO tokenizer vocabulary is malformed.")
    normalized = {str(token): int(index) for token, index in vocabulary.items()}
    if len(set(normalized.values())) != len(normalized):
        raise ProductionNeoFoldError("NEO tokenizer IDs are not unique.")
    file_hashes = {
        name: _file_sha256(directory / name)
        for name in _TOKENIZER_FILES
        if (directory / name).is_file()
    }
    if set(file_hashes) != set(_TOKENIZER_FILES):
        raise ProductionNeoFoldError("NEO tokenizer artifact inventory is incomplete.")
    special_ids = {
        token: int(tokenizer.convert_tokens_to_ids(token))
        for token in native_eval.SPECIAL_TOKEN_IDS
    }
    body = {
        "schema": "neural-forge-local-neo-tokenizer-v1",
        "vocab_size": len(normalized),
        "vocab_sha256": hashlib.sha256(canonical_json_bytes(normalized)).hexdigest(),
        "special_token_ids": special_ids,
        "file_sha256": file_hashes,
    }
    return LocalTokenizerIdentity(
        fingerprint=canonical_sha256(body),
        vocab_size=len(normalized),
        special_token_ids=special_ids,
        file_sha256=file_hashes,
    )


def _model_device_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameters = tuple(model.parameters())
    if not parameters:
        raise ProductionNeoFoldError("NEO model has no parameters.")
    devices = {parameter.device for parameter in parameters}
    dtypes = {parameter.dtype for parameter in parameters}
    if len(devices) != 1 or len(dtypes) != 1:
        raise ProductionNeoFoldError(
            "Production NEO models must use one device and one parameter dtype."
        )
    return next(iter(devices)), next(iter(dtypes))


class NativeVisionRecordRunner:
    """Run one real image/prompt against a parent or child NEO model."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        seed: int,
        max_new_tokens: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.seed = _nonnegative_int(seed, "seed")
        self.max_new_tokens = _positive_int(
            max_new_tokens,
            "max_new_tokens",
        )

    def __call__(self, model: nn.Module, record: NeoVisionBehaviorRecord) -> str:
        tokenizer = getattr(
            model,
            "_neural_forge_local_tokenizer",
            self.tokenizer,
        )
        device, dtype = _model_device_dtype(model)
        pixel_values, grid_hw = native_eval.preprocess_neo_image(
            Path(record.image_path),
            torch_module=torch,
        )
        torch.manual_seed(self.seed + record.index)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed + record.index)
        with torch.inference_mode():
            output = model.chat(
                tokenizer,
                pixel_values=pixel_values.to(device=device, dtype=dtype),
                grid_hw=grid_hw.to(device=device),
                question=record.prompt,
                generation_config={
                    "do_sample": False,
                    "max_new_tokens": self.max_new_tokens,
                    "num_beams": 1,
                },
                verbose=False,
            )
        if not isinstance(output, str):
            raise ProductionNeoFoldError(
                f"NEO model returned non-text for {record.probe_id}."
            )
        return output


class DirectNeoCheckpointIO:
    """Standalone sharded safetensor save/reload for the local NEO runtime."""

    def __init__(
        self,
        *,
        source_snapshot: Path,
        device: torch.device,
        dtype: torch.dtype,
        tokenizer_fingerprint: str,
        shard_bytes: int,
    ) -> None:
        self.source_snapshot = _resolve_local_directory(
            source_snapshot,
            "source NEO snapshot",
        )
        self.device = device
        self.dtype = dtype
        self.tokenizer_fingerprint = tokenizer_fingerprint
        self.shard_bytes = _positive_int(shard_bytes, "shard_bytes")

    def _copy_runtime(self, destination: Path) -> None:
        for filename in (*_RUNTIME_FILES, *_TOKENIZER_FILES):
            source = self.source_snapshot / filename
            if not source.is_file() or source.is_symlink():
                raise ProductionNeoFoldError(
                    f"Cannot package missing/unsafe local artifact {source}."
                )
            shutil.copy2(source, destination / filename)

    def _save_state(self, model: nn.Module, destination: Path) -> None:
        from safetensors.torch import save_file

        state = model.state_dict()
        groups: list[list[str]] = []
        current: list[str] = []
        current_bytes = 0
        for name, value in state.items():
            size = int(value.numel()) * int(value.element_size())
            if current and current_bytes + size > self.shard_bytes:
                groups.append(current)
                current = []
                current_bytes = 0
            current.append(name)
            current_bytes += size
        if current:
            groups.append(current)
        weight_map: dict[str, str] = {}
        total_size = 0
        for index, names in enumerate(groups, start=1):
            filename = f"model-{index:05d}-of-{len(groups):05d}.safetensors"
            tensors = {
                name: state[name].detach().to(device="cpu").contiguous()
                for name in names
            }
            save_file(
                tensors,
                str(destination / filename),
                metadata={"format": "pt"},
            )
            for name, tensor in tensors.items():
                weight_map[name] = filename
                total_size += int(tensor.numel()) * int(tensor.element_size())
            del tensors
            gc.collect()
        _write_atomic_json(
            destination / "model.safetensors.index.json",
            {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            },
        )

    def save(self, model: nn.Module, destination: Path) -> None:
        destination = destination.resolve()
        if any(destination.iterdir()):
            raise ProductionNeoFoldError(
                "Direct NEO saver requires an empty staging directory."
            )
        device, dtype = _model_device_dtype(model)
        if device != self.device or dtype != self.dtype:
            raise ProductionNeoFoldError(
                "Accepted NEO child differs from checkpoint I/O device/dtype."
            )
        self._copy_runtime(destination)
        _write_atomic_json(
            destination / "config.json",
            _standalone_neo_model_config(
                model,
                self.source_snapshot,
            ),
        )
        self._save_state(model, destination)

    def load(self, directory: Path) -> nn.Module:
        model = load_local_neo_model(
            directory,
            device=self.device,
            dtype=self.dtype,
        )
        tokenizer = load_local_neo_tokenizer(directory)
        identity = fingerprint_local_neo_tokenizer(directory, tokenizer)
        if identity.fingerprint != self.tokenizer_fingerprint:
            raise ProductionNeoFoldError(
                "Standalone child tokenizer differs from the parent tokenizer."
            )
        model._neural_forge_local_tokenizer = tokenizer
        return model


@dataclass(frozen=True, slots=True)
class ProductionNeoFoldConfig:
    parent_snapshot: Path
    manifest: Path
    parent_outputs: Path
    fold_cache: Path
    snapshot_root: Path
    publish_destination: Path
    receipt_path: Path
    ledger_path: Path
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    seed: int = 0
    blank_seed: int = 17
    max_new_tokens: int = 16
    sketch_dim: int = 64
    candidate_count: int = 8
    snapshot_chunk_bytes: int = 64 * 1024**2
    artifact_shard_bytes: int = 1024 * 1024**2
    derived_cache_root: Path | None = None
    anchor_count: int = 1
    portfolio_candidate_counts: tuple[int, ...] = ()
    maximum_attempts: int = 1
    include_source_cache_candidate: bool = False

    def __post_init__(self) -> None:
        for name in (
            "parent_snapshot",
            "manifest",
            "parent_outputs",
            "fold_cache",
            "snapshot_root",
            "publish_destination",
            "receipt_path",
            "ledger_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.derived_cache_root is not None:
            object.__setattr__(
                self,
                "derived_cache_root",
                Path(self.derived_cache_root),
            )
        for name in (
            "max_new_tokens",
            "sketch_dim",
            "candidate_count",
            "snapshot_chunk_bytes",
            "artifact_shard_bytes",
            "anchor_count",
            "maximum_attempts",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name),
            )
        candidate_counts = tuple(
            _positive_int(value, "portfolio_candidate_counts")
            for value in (self.portfolio_candidate_counts or (self.candidate_count,))
        )
        if len(set(candidate_counts)) != len(candidate_counts):
            raise ProductionNeoFoldError("Portfolio candidate counts must be unique.")
        object.__setattr__(
            self,
            "portfolio_candidate_counts",
            candidate_counts,
        )
        if self.derived_cache_root is not None and self.anchor_count <= 1:
            raise ProductionNeoFoldError(
                "Derived NEO portfolio caches require multiple anchors."
            )
        if not isinstance(self.include_source_cache_candidate, bool):
            raise ProductionNeoFoldError(
                "include_source_cache_candidate must be boolean."
            )
        for name in ("seed", "blank_seed"):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name),
            )
        _torch_dtype(self.dtype)


def write_neo_failed_closed_receipt(
    config: ProductionNeoFoldConfig,
    error: BaseException,
) -> Mapping[str, Any]:
    """Seal one non-success receipt without claiming an accepted artifact."""

    if not isinstance(config, ProductionNeoFoldConfig):
        raise ProductionNeoFoldError(
            "Failure receipt requires ProductionNeoFoldConfig."
        )
    if not isinstance(error, BaseException):
        raise ProductionNeoFoldError("Failure receipt requires the caught exception.")
    input_hashes = {}
    for name, path in (
        ("manifest_sha256", config.manifest),
        ("parent_outputs_sha256", config.parent_outputs),
    ):
        source = path.expanduser().absolute()
        input_hashes[name] = _file_sha256(source) if source.is_file() else None
    fold_cache = config.fold_cache.expanduser().absolute()
    publication = config.publish_destination.expanduser().absolute()
    body = {
        "schema": FAILURE_RECEIPT_SCHEMA,
        "status": "failed_closed",
        "failure_phase": "run_production_neo_fold",
        "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
        "error_message": str(error),
        "runner_sha256": _file_sha256(Path(__file__).resolve(strict=True)),
        "parent_snapshot": str(config.parent_snapshot.expanduser().absolute()),
        "manifest": str(config.manifest.expanduser().absolute()),
        "parent_outputs": str(config.parent_outputs.expanduser().absolute()),
        "input_hashes": input_hashes,
        "fold_cache": str(fold_cache),
        "fold_cache_present_after_failure": fold_cache.is_dir(),
        "fold_cache_manifest_present": (fold_cache / "manifest.json").is_file(),
        "publication_destination": str(publication),
        "publication_present_after_failure": publication.exists(),
        "accepted_child_published": False,
        "success_receipt_issued": False,
        "runner_uses_hub_loader": False,
        "runner_constructs_optimizer": False,
        "runner_calls_backward": False,
        "parent_outputs_regenerated": False,
        "no_success_claim": True,
    }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    _write_atomic_json(config.receipt_path, receipt)
    return receipt


def _load_ledger(path: Path) -> ExperimentLedger:
    return DurableExperimentLedger(path)


def run_production_neo_fold(
    config: ProductionNeoFoldConfig,
) -> tuple[Any, Mapping[str, Any]]:
    """Execute the exact 6144-to-6143 native-vision fold and certify it."""

    if not isinstance(config, ProductionNeoFoldConfig):
        raise ProductionNeoFoldError("config must be ProductionNeoFoldConfig.")
    validate_pinned_production_runtime()
    parent_snapshot = _resolve_local_directory(
        config.parent_snapshot,
        "parent snapshot",
    )
    manifest = _resolve_local_file(config.manifest, "probe manifest")
    outputs = _resolve_local_file(
        config.parent_outputs,
        "cached parent outputs",
    )
    publish = config.publish_destination.expanduser().absolute()
    receipt_path = config.receipt_path.expanduser().absolute()
    if publish.exists() or receipt_path.exists():
        raise ProductionNeoFoldError(
            "Publication and receipt destinations must be new."
        )
    config.snapshot_root.mkdir(parents=True, exist_ok=True)
    if config.snapshot_root.is_symlink():
        raise ProductionNeoFoldError("Snapshot root cannot be a symlink.")
    behavior_cache = load_neo_parent_behavior_cache(
        manifest,
        outputs,
        expected_manifest_sha256=PINNED_MANIFEST_SHA256,
        expected_outputs_sha256=PINNED_PARENT_OUTPUTS_SHA256,
    )
    if len(behavior_cache.records) != 256:
        raise ProductionNeoFoldError(
            "Pinned NEO behavior cache must contain exactly 256 records."
        )
    dtype = _torch_dtype(config.dtype)
    device = torch.device(config.device)
    parent = load_local_neo_model(
        parent_snapshot,
        device=device,
        dtype=dtype,
    )
    tokenizer = load_local_neo_tokenizer(parent_snapshot)
    tokenizer_identity = fingerprint_local_neo_tokenizer(
        parent_snapshot,
        tokenizer,
    )
    loaded_report = native_eval._validate_loaded_model(
        tokenizer,
        parent,
        torch=torch,
        torch_dtype=dtype,
        device=device,
    )
    if loaded_report["parameter_count"] != PARENT_UNIQUE_PARAMETERS:
        raise ProductionNeoFoldError(
            "Concrete local NEO parent misses the validated parameter count."
        )
    parent_graph = inspect_model_circuits(
        parent,
        model_name=PARENT_NAME,
        tokenizer_fingerprint=tokenizer_identity.fingerprint,
    )
    if (
        parent_graph.unique_parameter_count != PARENT_UNIQUE_PARAMETERS
        or len(parent_graph.layer_indices) != DECODER_LAYERS
        or parent.config.llm_config.hidden_size != DECODER_HIDDEN_SIZE
        or parent.config.llm_config.intermediate_size != SOURCE_INTERMEDIATE_SIZE
        or PARENT_UNIQUE_PARAMETERS - TARGET_UNIQUE_PARAMETERS != EXACT_PARAMETER_DELTA
    ):
        raise ProductionNeoFoldError(
            "Loaded parent differs from the validated NEO folding contract."
        )
    record_runner = NativeVisionRecordRunner(
        tokenizer,
        seed=config.seed,
        max_new_tokens=config.max_new_tokens,
    )
    cache_reused = config.fold_cache.exists()
    if cache_reused:
        source_cache = NeoNativeVisionFoldCache.open(
            config.fold_cache,
            behavior_cache=behavior_cache,
        )
        source_cache.verify(parent_model=parent, parent_graph=parent_graph)
    else:
        if config.derived_cache_root is not None:
            raise ProductionNeoFoldError(
                "A derived portfolio requires an existing sealed source cache."
            )
        config.fold_cache.parent.mkdir(parents=True, exist_ok=True)
        source_cache = build_neo_native_vision_fold_cache(
            parent,
            parent_graph,
            behavior_cache,
            trace_runner=record_runner,
            destination=config.fold_cache,
            sketch_dim=config.sketch_dim,
            candidate_count=config.candidate_count,
        )
    caches: list[NeoNativeVisionFoldCache] = []
    if config.derived_cache_root is None:
        caches.append(source_cache)
    else:
        if config.include_source_cache_candidate:
            caches.append(source_cache)
        derived_root = config.derived_cache_root.expanduser().absolute()
        if derived_root.exists() and (
            not derived_root.is_dir() or derived_root.is_symlink()
        ):
            raise ProductionNeoFoldError(
                "Derived-cache root must be a regular local directory."
            )
        derived_root.mkdir(parents=True, exist_ok=True)
        for candidate_count in config.portfolio_candidate_counts:
            candidate_path = derived_root / f"a{config.anchor_count}-c{candidate_count}"
            if candidate_path.exists():
                derived = NeoNativeVisionFoldCache.open(
                    candidate_path,
                    behavior_cache=behavior_cache,
                )
                if (
                    derived.source_cache_key != source_cache.cache_key
                    or derived.anchor_count != config.anchor_count
                    or derived.candidate_count != candidate_count
                ):
                    raise ProductionNeoFoldError(
                        "Existing derived NEO cache differs from its portfolio arm."
                    )
            else:
                derived = derive_neo_native_vision_fold_cache(
                    source_cache,
                    destination=candidate_path,
                    candidate_count=candidate_count,
                    anchor_count=config.anchor_count,
                )
            derived.verify(parent_model=parent, parent_graph=parent_graph)
            caches.append(derived)
    for cache in caches:
        if len(cache.selections) != DECODER_LAYERS or any(
            selection.source_neurons != SOURCE_INTERMEDIATE_SIZE
            or selection.target_neurons != TARGET_INTERMEDIATE_SIZE
            for selection in cache.selections
        ):
            raise ProductionNeoFoldError(
                "Sealed NEO evidence does not describe exact 6144-to-6143 folding."
            )
    checkpoint_io = DirectNeoCheckpointIO(
        source_snapshot=parent_snapshot,
        device=device,
        dtype=dtype,
        tokenizer_fingerprint=tokenizer_identity.fingerprint,
        shard_bytes=config.artifact_shard_bytes,
    )
    snapshot_config = DynamicFoldSnapshotConfig.disk(
        config.snapshot_root,
        chunk_bytes=config.snapshot_chunk_bytes,
    )
    behavior_gates: dict[str, NeoNativeVisionBehaviorGate] = {}
    stack_adapters = []
    candidate_specs = []
    caches_by_candidate: dict[str, NeoNativeVisionFoldCache] = {}
    for index, cache in enumerate(caches):
        candidate_id = f"a{cache.anchor_count}-c{cache.candidate_count}"
        behavior_gate = NeoNativeVisionBehaviorGate(
            behavior_cache,
            child_runner=record_runner,
        )
        behavior_gates[candidate_id] = behavior_gate
        caches_by_candidate[candidate_id] = cache
        resources = NeoNativeVisionStackResources(
            cache=cache,
            behavior_gate=behavior_gate,
            model_saver=checkpoint_io.save,
            model_loader=checkpoint_io.load,
            saver_id=DIRECT_SAVER_ID,
            loader_id=DIRECT_LOADER_ID,
            snapshot_config=snapshot_config,
        )
        stack_adapters.append(NeoNativeVisionFoldStackAdapter(resources))
        candidate_specs.append(
            ProductionFoldCandidateSpec(
                candidate_id=candidate_id,
                method_id=METHOD_ID,
                evidence_fingerprint=cache.cache_key,
                target_adapter_id=(NeoNativeVisionTargetSynthesisAdapter.adapter_id),
                priority=len(caches) - index,
                blank_seed=config.blank_seed,
                configuration=(("fold_cache_key", cache.cache_key),),
                metadata=(
                    ("anchor_count", cache.anchor_count),
                    ("candidate_count", cache.candidate_count),
                    ("heldout_samples_used_for_selection", 0),
                    ("parent_trace_calls_this_run", 0 if cache_reused else 192),
                ),
            )
        )
    ledger = _load_ledger(config.ledger_path)
    request = DynamicTargetRequest(
        child_name="neo1.5-2b-native-folded-6143",
        target_family="neo_chat",
        target_modalities=("image", "text"),
        exact_unique_parameters=TARGET_UNIQUE_PARAMETERS,
        maximum_parameter_bytes=TARGET_UNIQUE_PARAMETERS * 2,
        tokenizer_fingerprint=tokenizer_identity.fingerprint,
        blank_seed=config.blank_seed,
        configuration=(
            ("decoder_intermediate_size", TARGET_INTERMEDIATE_SIZE),
            ("source_parent", PARENT_NAME),
        ),
    )
    capability = ProductionFoldCapability(
        capability_id="neo-native-vision-exact-2896372736",
        source_families=("neo_chat",),
        target_family="neo_chat",
        target_modalities=("image", "text"),
        minimum_target_unique_parameters=TARGET_UNIQUE_PARAMETERS,
        maximum_target_unique_parameters=TARGET_UNIQUE_PARAMETERS,
    )
    bundle = ProductionFoldBundle(
        capability=capability,
        target_adapters=(
            NeoNativeVisionTargetSynthesisAdapter(
                device=device,
                dtype=dtype,
            ),
        ),
        stack_adapters=tuple(stack_adapters),
        candidate_specs=tuple(candidate_specs),
        maximum_attempts=min(config.maximum_attempts, len(candidate_specs)),
        portfolio_id="neural-forge.production.neo-native-vision-2896372736",
        version=1,
        priority=250,
    )
    catalog = ProductionDynamicFoldCatalog(
        (bundle,),
        ledger=ledger,
        snapshot_config=snapshot_config,
        require_positive_compaction=True,
    )
    objective = DynamicFoldSearchObjective(
        target=request,
        maximum_attempts=min(config.maximum_attempts, len(candidate_specs)),
        portfolio_id=bundle.portfolio_id,
        stop_after_first_accept=True,
    )
    search_result = catalog.assemble().run(
        parent_models={PARENT_NAME: parent},
        objective=objective,
        destination_root=publish,
        parent_tokenizer_fingerprints={PARENT_NAME: tokenizer_identity.fingerprint},
    )
    if not search_result.complete:
        attempt_summary = [
            {
                "candidate_id": attempt.candidate.candidate_id,
                "status": attempt.status,
                "reason": attempt.reason,
            }
            for attempt in search_result.attempts
        ]
        raise ProductionNeoFoldError(
            "Bounded NEO production portfolio found no strict behavior "
            "acceptance: "
            + json.dumps(
                attempt_summary,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    result = search_result.selected_result
    selected_id = str(search_result.selected_candidate_id)
    cache = caches_by_candidate[selected_id]
    behavior_gate = behavior_gates[selected_id]
    if (
        not result.complete
        or result.target.graph.unique_parameter_count != TARGET_UNIQUE_PARAMETERS
    ):
        raise ProductionNeoFoldError(
            "Dynamic NEO fold did not reach its exact accepted target."
        )
    receipt = write_neo_dynamic_fold_receipt(
        result,
        cache,
        behavior_gate,
        destination=receipt_path,
        cache_reused=cache_reused,
        search_result=search_result,
    )
    return result, receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-snapshot",
        type=Path,
        default=Path(
            "/mnt/ai_data/neural-forge/native_parent_cache_20260710/"
            "neo1_5_2b_materialized_105970970eccd7cddf105224a4930342c31733e4"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("runs/native_vision_probe_v1_20260710/manifest.json"),
    )
    parser.add_argument(
        "--parent-outputs",
        type=Path,
        default=Path(
            "runs/native_vision_parent_20260710/"
            "neo1p5_2b_sft_probe256_bf16_gpu0_r6_eagerprefill_tf4573_20260710/"
            "parent_outputs.jsonl"
        ),
    )
    parser.add_argument(
        "--fold-cache",
        type=Path,
        default=Path("runs/neo_native_dynamic_fold_cache_20260716"),
    )
    parser.add_argument(
        "--derived-cache-root",
        type=Path,
        default=None,
        help=(
            "Optional root for heldout-blind caches derived from --fold-cache "
            "without new parent traces."
        ),
    )
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=Path("runs/neo_native_dynamic_snapshots"),
    )
    parser.add_argument(
        "--publish-destination",
        type=Path,
        default=Path("runs/neo_native_dynamic_child_6143_20260716"),
    )
    parser.add_argument(
        "--receipt-path",
        type=Path,
        default=Path("runs/neo_native_dynamic_child_6143_20260716.receipt.json"),
    )
    parser.add_argument(
        "--ledger-path",
        type=Path,
        default=Path("runs/neo_native_dynamic_ledger.jsonl"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blank-seed", type=int, default=17)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--sketch-dim", type=int, default=64)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument(
        "--anchor-count",
        type=int,
        default=1,
        help="Sparse closed-form survivor anchors per removed neuron.",
    )
    parser.add_argument(
        "--portfolio-candidate-count",
        action="append",
        type=int,
        default=None,
        help=(
            "Repeat to register deterministic low-importance candidate-pool "
            "sizes in the bounded production catalog."
        ),
    )
    parser.add_argument("--maximum-attempts", type=int, default=1)
    parser.add_argument(
        "--include-source-cache-candidate",
        action="store_true",
        help=(
            "Include the source cache as a catalog arm so an existing exact "
            "ledger rejection is classified before derived arms."
        ),
    )
    parser.add_argument(
        "--snapshot-chunk-bytes",
        type=int,
        default=64 * 1024**2,
    )
    parser.add_argument(
        "--artifact-shard-bytes",
        type=int,
        default=1024 * 1024**2,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = ProductionNeoFoldConfig(
        parent_snapshot=args.parent_snapshot,
        manifest=args.manifest,
        parent_outputs=args.parent_outputs,
        fold_cache=args.fold_cache,
        snapshot_root=args.snapshot_root,
        publish_destination=args.publish_destination,
        receipt_path=args.receipt_path,
        ledger_path=args.ledger_path,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        blank_seed=args.blank_seed,
        max_new_tokens=args.max_new_tokens,
        sketch_dim=args.sketch_dim,
        candidate_count=args.candidate_count,
        snapshot_chunk_bytes=args.snapshot_chunk_bytes,
        artifact_shard_bytes=args.artifact_shard_bytes,
        derived_cache_root=args.derived_cache_root,
        anchor_count=args.anchor_count,
        portfolio_candidate_counts=tuple(args.portfolio_candidate_count or ()),
        maximum_attempts=args.maximum_attempts,
        include_source_cache_candidate=args.include_source_cache_candidate,
    )
    try:
        result, receipt = run_production_neo_fold(config)
    except BaseException as exc:
        try:
            failure = write_neo_failed_closed_receipt(config, exc)
            print(json.dumps(dict(failure), sort_keys=True), file=sys.stderr)
        except BaseException as receipt_error:
            print(
                "Could not seal NEO failed-closed receipt: "
                f"{type(receipt_error).__name__}: {receipt_error}",
                file=sys.stderr,
            )
        raise
    print(
        json.dumps(
            {
                "complete": result.complete,
                "target_unique_parameters": (
                    result.target.graph.unique_parameter_count
                ),
                "artifact": result.published.artifact.path,
                "receipt_sha256": receipt["receipt_sha256"],
                "zero_training": receipt["zero_training"],
                "parent_outputs_regenerated_after_cache": receipt[
                    "parent_outputs_regenerated_after_cache"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
