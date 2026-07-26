"""Direct local-only LFM2 checkpoint and tokenizer I/O.

This module is the production I/O boundary for LFM2 dynamic folding.  It
constructs the concrete LFM2 classes from local JSON, copies tensors from
local safetensors, and reconstructs the fast tokenizer from tokenizer.json.
It has no AutoModel/AutoTokenizer, ``from_pretrained``, hub, URL, or network
fallback.

The checkpoint callable additionally copies the exact tokenizer package bound
to a verified strict tokenizer bridge.  Publication can therefore prove that
the standalone child contains the same target token domain that was reviewed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from .hf_tokenizer_bridge import (
    LOCAL_TOKENIZER_PACKAGE_FILENAMES,
    StrictTokenizerBridgeArtifact,
    strict_tokenizer_bridge_target_package_fingerprint,
)


SCHEMA = "neural-forge-lfm2-direct-local-checkpoint-io-v1"
BRIDGE_REFERENCE_FILENAME = "neural_forge_tokenizer_bridge_reference.json"
_RESERVED_MODEL_FILES = frozenset(
    {
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "model.safetensors.index.json",
    }
)


class Lfm2DirectLocalError(RuntimeError):
    """A direct local LFM2 artifact is missing, unsafe, or inconsistent."""


def _local_directory(
    value: str | os.PathLike[str],
    name: str,
) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2DirectLocalError(f"{name} must be a local directory.")
    try:
        path = Path(raw).expanduser().resolve(strict=True)
    except OSError as exc:
        raise Lfm2DirectLocalError(f"{name} does not exist.") from exc
    if not path.is_dir():
        raise Lfm2DirectLocalError(f"{name} must be a local directory.")
    return path


def _read_json(path: Path, name: str) -> dict[str, Any]:
    # A local model-cache snapshot may point at its local blob store through
    # symlinks.  That remains direct local I/O.  Published child artifacts are
    # checked separately and are required to contain only real files.
    if not path.is_file():
        raise Lfm2DirectLocalError(f"Local {name} is missing or unsafe.")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2DirectLocalError(f"Could not read local {name}.") from exc
    if not isinstance(value, Mapping):
        raise Lfm2DirectLocalError(f"Local {name} must be a JSON object.")
    return dict(value)


def _write_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Lfm2DirectLocalError(
            f"Refusing to overwrite checkpoint file {path.name!r}."
        )
    path.write_bytes(canonical_json_bytes(dict(value)) + b"\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: Any) -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise Lfm2DirectLocalError("PyTorch is required for LFM2 I/O.") from exc
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2DirectLocalError(
            "LFM2 state contains a non-materialized or non-strided tensor."
        )
    tensor = value.detach().contiguous()
    raw = tensor.view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    digest.update(
        canonical_json_bytes(
            {
                "dtype": str(tensor.dtype),
                "shape": [int(size) for size in tensor.shape],
            }
        )
    )
    for start in range(0, int(raw.numel()), 8 * 1024**2):
        digest.update(
            raw[start : start + 8 * 1024**2]
            .to(device="cpu")
            .contiguous()
            .numpy()
            .tobytes()
        )
    return digest.hexdigest()


def fingerprint_lfm2_direct_local_state(model: Any) -> str:
    """Hash every named state tensor, including tied state-dict aliases."""

    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise Lfm2DirectLocalError("LFM2 model has no state_dict().")
    rows = []
    for name, value in sorted(state_dict().items()):
        rows.append(
            {
                "name": str(name),
                "shape": [int(size) for size in value.shape],
                "dtype": str(value.dtype),
                "sha256": _tensor_sha256(value),
            }
        )
    if not rows:
        raise Lfm2DirectLocalError("LFM2 model state is empty.")
    return canonical_sha256(rows)


def _storage_identity(value: Any) -> tuple[int, int, tuple[int, ...]]:
    return (
        int(value.untyped_storage().data_ptr()),
        int(value.storage_offset()),
        tuple(int(size) for size in value.shape),
    )


def _safe_indexed_safetensor_shard(
    directory: Path,
    value: Any,
) -> Path:
    if not isinstance(value, str) or not value:
        raise Lfm2DirectLocalError(
            "Local safetensors index contains an unsafe shard path."
        )
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part in {".", ".."} for part in posix_path.parts)
        or any(part in {".", ".."} for part in windows_path.parts)
        or posix_path.suffix != ".safetensors"
        or windows_path.suffix != ".safetensors"
    ):
        raise Lfm2DirectLocalError(
            "Local safetensors index contains an unsafe shard path."
        )
    try:
        local_shard = directory / Path(value)
        shard = local_shard.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise Lfm2DirectLocalError(
            "Local safetensors index shard escapes its checkpoint directory."
        ) from exc
    if not shard.is_file():
        raise Lfm2DirectLocalError(
            "Local safetensors index shard is not a regular local file."
        )
    try:
        shard.relative_to(directory)
    except ValueError as exc:
        if not _is_expected_hf_cache_blob_link(
            directory,
            local_shard,
            shard,
        ):
            raise Lfm2DirectLocalError(
                "Local safetensors index shard escapes its checkpoint "
                "directory."
            ) from exc
    return shard


def _is_lower_hex_digest(value: str, *, lengths: set[int]) -> bool:
    return (
        len(value) in lengths
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_expected_hf_cache_blob_link(
    directory: Path,
    local_shard: Path,
    resolved_shard: Path,
) -> bool:
    """Recognize only the content-addressed HF cache snapshot link contract."""

    snapshots = directory.parent
    model_root = snapshots.parent
    if (
        not local_shard.is_symlink()
        or snapshots.name != "snapshots"
        or not model_root.name.startswith("models--")
        or model_root.name == "models--"
        or not _is_lower_hex_digest(directory.name, lengths={40, 64})
    ):
        return False
    blobs_path = model_root / "blobs"
    if blobs_path.is_symlink() or not blobs_path.is_dir():
        return False
    try:
        blobs = blobs_path.resolve(strict=True)
        if blobs.parent != model_root:
            return False
        relative_blob = resolved_shard.relative_to(blobs)
    except (OSError, RuntimeError, ValueError):
        return False
    return (
        len(relative_blob.parts) == 1
        and _is_lower_hex_digest(relative_blob.name, lengths={40, 64})
    )


def _local_safetensor_shards(directory: Path) -> tuple[Path, ...]:
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        index = _read_json(index_path, "safetensors index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise Lfm2DirectLocalError(
                "Local safetensors index has no weight_map."
            )
        raw_names = tuple(weight_map.values())
        if any(not isinstance(name, str) for name in raw_names):
            raise Lfm2DirectLocalError(
                "Local safetensors index contains an unsafe shard path."
            )
        names = tuple(sorted(set(raw_names)))
        shards = tuple(
            _safe_indexed_safetensor_shard(directory, name)
            for name in names
        )
    else:
        single = directory / "model.safetensors"
        shards = (
            (single,)
            if single.is_file()
            else tuple(sorted(directory.glob("model-*.safetensors")))
        )
    if not shards or any(not path.is_file() for path in shards):
        raise Lfm2DirectLocalError(
            f"Local LFM2 safetensors are incomplete in {directory}."
        )
    return shards


def load_lfm2_direct_local_config(
    model_directory: str | os.PathLike[str],
) -> Any:
    """Construct concrete ``Lfm2Config`` from local config.json."""

    try:
        from transformers import Lfm2Config
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise Lfm2DirectLocalError(
            "The concrete transformers LFM2 classes are required."
        ) from exc
    directory = _local_directory(model_directory, "model_directory")
    payload = _read_json(directory / "config.json", "LFM2 config")
    if payload.get("model_type") != "lfm2":
        raise Lfm2DirectLocalError(
            "Local config.json is not an LFM2 configuration."
        )
    # Older Neural Forge LFM2 children serialized ``block_ff_dim`` as an
    # explicit compatibility alias for ``intermediate_size``.  Current
    # Transformers consumes that constructor field by assigning it to
    # ``intermediate_size`` without retaining the alias.  Validate the two
    # serialized fields before construction so a conflicting alias cannot
    # silently override the concrete topology.
    block_ff_dim = None
    if "block_ff_dim" in payload:
        block_ff_dim = payload["block_ff_dim"]
        serialized_intermediate_size = payload.get("intermediate_size")
        if (
            isinstance(block_ff_dim, bool)
            or not isinstance(block_ff_dim, int)
            or block_ff_dim <= 0
            or isinstance(serialized_intermediate_size, bool)
            or not isinstance(serialized_intermediate_size, int)
            or block_ff_dim != serialized_intermediate_size
        ):
            raise Lfm2DirectLocalError(
                "Local LFM2 block_ff_dim compatibility alias must be one "
                "positive integer equal to intermediate_size."
            )
    try:
        config = Lfm2Config(**payload)
    except (TypeError, ValueError) as exc:
        raise Lfm2DirectLocalError(
            "Local config.json could not construct a concrete LFM2 config."
        ) from exc

    # Restore the validated alias so direct-local load/save is byte-stable and
    # a tensor-only treatment cannot appear to mutate config.json.
    if block_ff_dim is not None:
        intermediate_size = getattr(config, "intermediate_size", None)
        if block_ff_dim != intermediate_size:
            raise Lfm2DirectLocalError(
                "Concrete LFM2 config disagrees with block_ff_dim."
            )
        config.block_ff_dim = block_ff_dim
        if config.to_dict().get("block_ff_dim") != block_ff_dim:
            raise Lfm2DirectLocalError(
                "Concrete LFM2 config did not retain block_ff_dim."
            )
    return config


def load_lfm2_direct_local_model(
    model_directory: str | os.PathLike[str],
    *,
    device: Any,
    dtype: Any,
) -> Any:
    """Load concrete LFM2 from local JSON+safetensors without HF loaders."""

    try:
        import torch
        from safetensors import safe_open
        from transformers import GenerationConfig, Lfm2ForCausalLM
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise Lfm2DirectLocalError(
            "PyTorch, safetensors, and concrete LFM2 classes are required."
        ) from exc
    directory = _local_directory(model_directory, "model_directory")
    config = load_lfm2_direct_local_config(directory)
    model = Lfm2ForCausalLM(config)
    model.to(dtype=dtype)
    model.tie_weights()
    target_state = model.state_dict()
    loaded: set[str] = set()
    unexpected: list[str] = []
    with torch.no_grad():
        for shard in _local_safetensor_shards(directory):
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if name in loaded:
                        raise Lfm2DirectLocalError(
                            f"Duplicate local tensor across shards: {name}."
                        )
                    if name not in target_state:
                        unexpected.append(name)
                        continue
                    source = handle.get_tensor(name)
                    target = target_state[name]
                    if tuple(source.shape) != tuple(target.shape):
                        raise Lfm2DirectLocalError(
                            f"Local tensor shape mismatch for {name}: "
                            f"{tuple(source.shape)} != {tuple(target.shape)}."
                        )
                    target.copy_(source.to(dtype=target.dtype))
                    loaded.add(name)
    if unexpected:
        raise Lfm2DirectLocalError(
            f"Local safetensors contain unexpected state keys: {unexpected}."
        )
    loaded_storage = {_storage_identity(target_state[name]) for name in loaded}
    missing = sorted(
        name
        for name, value in target_state.items()
        if name not in loaded and _storage_identity(value) not in loaded_storage
    )
    if missing:
        raise Lfm2DirectLocalError(
            f"Local safetensors are missing non-aliased state keys: {missing}."
        )
    generation_path = directory / "generation_config.json"
    if generation_path.is_file():
        model.generation_config = GenerationConfig(
            **_read_json(generation_path, "generation config")
        )
    model.to(device=torch.device(device), dtype=dtype)
    model.requires_grad_(False)
    model.eval()
    return model


def _special_token_value(value: Any) -> Any:
    if isinstance(value, Mapping) and "content" in value:
        return str(value["content"])
    return value


def load_lfm2_direct_local_tokenizer(
    tokenizer_directory: str | os.PathLike[str],
) -> Any:
    """Construct the fast tokenizer directly from local tokenizer artifacts."""

    try:
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise Lfm2DirectLocalError(
            "The concrete fast tokenizer class is required."
        ) from exc
    directory = _local_directory(
        tokenizer_directory,
        "tokenizer_directory",
    )
    tokenizer_path = directory / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise Lfm2DirectLocalError(
            "Local tokenizer.json is missing or unsafe."
        )
    kwargs: dict[str, Any] = {}
    config_path = directory / "tokenizer_config.json"
    if config_path.is_file():
        kwargs.update(_read_json(config_path, "tokenizer config"))
    special_path = directory / "special_tokens_map.json"
    if special_path.is_file():
        kwargs.update(_read_json(special_path, "special-token map"))
    for name in (
        "tokenizer_class",
        "backend",
        "is_local",
        "use_fast",
        "sp_model_kwargs",
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
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        **kwargs,
    )
    chat_template_path = directory / "chat_template.jinja"
    if chat_template_path.is_file():
        tokenizer.chat_template = chat_template_path.read_text(
            encoding="utf-8"
        )
    return tokenizer


def _checkpoint_state(model: Any) -> dict[str, Any]:
    stored: dict[str, Any] = {}
    seen: set[tuple[int, int, tuple[int, ...]]] = set()
    for name, value in model.state_dict().items():
        identity = _storage_identity(value)
        if identity in seen:
            continue
        seen.add(identity)
        stored[str(name)] = value.detach().to(device="cpu").contiguous()
    if not stored:
        raise Lfm2DirectLocalError("Cannot save an empty LFM2 state.")
    return stored


def _verify_target_tokenizer_files(
    directory: Path,
    bridge: StrictTokenizerBridgeArtifact,
) -> None:
    records = {
        record.relative_path: record for record in bridge.target_files
    }
    for name in LOCAL_TOKENIZER_PACKAGE_FILENAMES:
        record = records.get(name)
        if record is None:
            continue
        path = directory / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != record.size_bytes
            or _file_sha256(path) != record.sha256
        ):
            raise Lfm2DirectLocalError(
                f"Checkpoint tokenizer file {name!r} differs from the bridge."
            )
    if "tokenizer.json" not in records:
        raise Lfm2DirectLocalError(
            "Strict tokenizer bridge lacks tokenizer.json."
        )


@dataclass(frozen=True, slots=True)
class Lfm2DirectLocalCheckpointIO:
    """Exact direct-local saver/loader bound to one reviewed token bridge."""

    tokenizer_bridge: StrictTokenizerBridgeArtifact
    device: Any
    dtype: Any
    saver_id: str = "neural-forge.lfm2.direct-local-safetensors-v1"
    loader_id: str = "neural-forge.lfm2.direct-local-safetensors-v1"
    resource_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        bridge = self.tokenizer_bridge
        if not isinstance(bridge, StrictTokenizerBridgeArtifact):
            raise TypeError(
                "tokenizer_bridge must be StrictTokenizerBridgeArtifact."
            )
        bridge.verify_files()
        collisions = sorted(
            record.relative_path
            for record in bridge.target_files
            if record.relative_path in _RESERVED_MODEL_FILES
        )
        if collisions:
            raise Lfm2DirectLocalError(
                "Tokenizer bridge collides with model checkpoint files: "
                f"{collisions!r}."
            )
        for name in ("saver_id", "loader_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise Lfm2DirectLocalError(f"{name} must be nonempty.")
        object.__setattr__(
            self,
            "resource_fingerprint",
            canonical_sha256(
                {
                    "schema": SCHEMA,
                    "bridge_artifact_sha256": bridge.artifact_sha256,
                    "bridge_profile_sha256": bridge.profile_fingerprint,
                    "target_tokenizer_package_fingerprint": (
                        strict_tokenizer_bridge_target_package_fingerprint(
                            bridge
                        )
                    ),
                    "target_tokenizer_semantics_sha256": (
                        bridge.target_tokenizer_semantics_sha256
                    ),
                    "device": str(self.device),
                    "dtype": str(self.dtype),
                    "saver_id": self.saver_id,
                    "loader_id": self.loader_id,
                    "auto_loader_used": False,
                    "from_pretrained_used": False,
                    "network_allowed": False,
                }
            ),
        )

    def save(self, model: Any, destination: Path) -> None:
        """Write model JSON+safetensors plus the exact target tokenizer."""

        try:
            from safetensors.torch import save_file
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise Lfm2DirectLocalError(
                "safetensors is required for direct LFM2 saving."
            ) from exc
        path = Path(destination).expanduser().absolute()
        if (
            not path.is_dir()
            or path.is_symlink()
            or any(path.iterdir())
        ):
            raise Lfm2DirectLocalError(
                "Direct checkpoint destination must be a real empty directory."
            )
        config = getattr(model, "config", None)
        if config is None or not callable(getattr(config, "to_dict", None)):
            raise Lfm2DirectLocalError(
                "LFM2 model has no serializable config."
            )
        if getattr(config, "model_type", None) != "lfm2":
            raise Lfm2DirectLocalError("Only LFM2 models can be saved.")
        _write_canonical_json(path / "config.json", config.to_dict())
        generation = getattr(model, "generation_config", None)
        if generation is not None:
            to_payload = getattr(generation, "to_diff_dict", None)
            if not callable(to_payload):
                to_payload = getattr(generation, "to_dict", None)
            if not callable(to_payload):
                raise Lfm2DirectLocalError(
                    "Generation config is not serializable."
                )
            _write_canonical_json(
                path / "generation_config.json",
                to_payload(),
            )
        save_file(
            _checkpoint_state(model),
            str(path / "model.safetensors"),
            metadata={"format": "pt"},
        )
        bridge = self.tokenizer_bridge
        bridge.verify_files()
        for record in bridge.target_files:
            source = bridge.root_dir / record.relative_path
            target = path / record.relative_path
            if target.exists() or target.is_symlink():
                raise Lfm2DirectLocalError(
                    "Tokenizer package attempted to overwrite a model file."
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if (
                target.stat().st_size != record.size_bytes
                or _file_sha256(target) != record.sha256
            ):
                raise Lfm2DirectLocalError(
                    f"Copied tokenizer file {record.relative_path!r} changed."
                )
        _write_canonical_json(
            path / BRIDGE_REFERENCE_FILENAME,
            {
                "schema": SCHEMA,
                "bridge_artifact_sha256": bridge.artifact_sha256,
                "bridge_profile_sha256": bridge.profile_fingerprint,
                "target_tokenizer_semantics_sha256": (
                    bridge.target_tokenizer_semantics_sha256
                ),
                "target_tokenizer_package_fingerprint": (
                    strict_tokenizer_bridge_target_package_fingerprint(bridge)
                ),
                "resource_fingerprint": self.resource_fingerprint,
                "strict_bridge_manifest_is_external_provenance": True,
            },
        )
        _verify_target_tokenizer_files(path, bridge)
        observed = fingerprint_lfm2_direct_local_state(model)
        if not isinstance(observed, str):
            raise Lfm2DirectLocalError("Could not seal saved LFM2 state.")

    def load(self, source: Path) -> Any:
        """Load an exact disjoint child and reauthenticate its tokenizer."""

        directory = _local_directory(source, "checkpoint_directory")
        required_real_files = (
            "config.json",
            "model.safetensors",
            BRIDGE_REFERENCE_FILENAME,
        )
        if any(
            not (directory / name).is_file()
            or (directory / name).is_symlink()
            for name in required_real_files
        ):
            raise Lfm2DirectLocalError(
                "Published checkpoint model files are missing or symlinked."
            )
        generation_path = directory / "generation_config.json"
        if generation_path.exists() and (
            not generation_path.is_file() or generation_path.is_symlink()
        ):
            raise Lfm2DirectLocalError(
                "Published generation config is not a real file."
            )
        reference = _read_json(
            directory / BRIDGE_REFERENCE_FILENAME,
            "tokenizer bridge reference",
        )
        bridge = self.tokenizer_bridge
        expected = {
            "bridge_artifact_sha256": bridge.artifact_sha256,
            "bridge_profile_sha256": bridge.profile_fingerprint,
            "target_tokenizer_semantics_sha256": (
                bridge.target_tokenizer_semantics_sha256
            ),
            "target_tokenizer_package_fingerprint": (
                strict_tokenizer_bridge_target_package_fingerprint(bridge)
            ),
            "resource_fingerprint": self.resource_fingerprint,
        }
        if any(reference.get(name) != value for name, value in expected.items()):
            raise Lfm2DirectLocalError(
                "Checkpoint tokenizer bridge reference is missing or stale."
            )
        _verify_target_tokenizer_files(directory, bridge)
        tokenizer = load_lfm2_direct_local_tokenizer(directory)
        try:
            from .hf_tokenizer_bridge import fingerprint_hf_tokenizer_semantics

            semantics = fingerprint_hf_tokenizer_semantics(tokenizer)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise Lfm2DirectLocalError(
                "Checkpoint tokenizer semantics could not be authenticated."
            ) from exc
        if semantics != bridge.target_tokenizer_semantics_sha256:
            raise Lfm2DirectLocalError(
                "Checkpoint tokenizer semantics differ from the bridge."
            )
        return load_lfm2_direct_local_model(
            directory,
            device=self.device,
            dtype=self.dtype,
        )


__all__ = [
    "BRIDGE_REFERENCE_FILENAME",
    "Lfm2DirectLocalCheckpointIO",
    "Lfm2DirectLocalError",
    "SCHEMA",
    "fingerprint_lfm2_direct_local_state",
    "load_lfm2_direct_local_config",
    "load_lfm2_direct_local_model",
    "load_lfm2_direct_local_tokenizer",
]
