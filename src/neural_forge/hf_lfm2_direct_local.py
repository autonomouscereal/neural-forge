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

import copy
import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any

from .dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from .hf_tokenizer_bridge import (
    LOCAL_TOKENIZER_PACKAGE_FILENAMES,
    StrictTokenizerBridgeArtifact,
    strict_tokenizer_bridge_target_package_fingerprint,
)


SCHEMA = "neural-forge-lfm2-direct-local-checkpoint-io-v1"
PORTABLE_LOAD_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-direct-local-portable-load-receipt-v1"
)
CANONICAL_FP32_RUNTIME_RELOCATION_SCHEMA = (
    "neural-forge-lfm2-canonical-fp32-runtime-relocation-v1"
)
RUNTIME_TENSOR_ALIAS_MANIFEST_SCHEMA = (
    "neural-forge-lfm2-direct-local-runtime-tensor-alias-manifest-v1"
)
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


def count_lfm2_direct_local_unique_parameter_elements(model: Any) -> int:
    """Count each concrete Parameter object exactly once."""

    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        raise Lfm2DirectLocalError("LFM2 model has no parameters().")
    seen: set[int] = set()
    total = 0
    for parameter in parameters():
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        total += int(parameter.numel())
    if total <= 0:
        raise Lfm2DirectLocalError("LFM2 model has no materialized parameters.")
    return total


def _named_lfm2_runtime_tensors(
    model: Any,
) -> tuple[tuple[str, str, bool | None, Any], ...]:
    """Return every parameter and buffer, including non-persistent buffers."""

    named_parameters = getattr(model, "named_parameters", None)
    named_buffers = getattr(model, "named_buffers", None)
    get_submodule = getattr(model, "get_submodule", None)
    if (
        not callable(named_parameters)
        or not callable(named_buffers)
        or not callable(get_submodule)
    ):
        raise Lfm2DirectLocalError(
            "LFM2 runtime state requires named parameters, named buffers, "
            "and submodule lookup."
        )
    try:
        parameters = tuple(named_parameters(remove_duplicate=False))
        buffers = tuple(named_buffers(remove_duplicate=False))
    except TypeError as exc:
        raise Lfm2DirectLocalError(
            "LFM2 runtime state requires duplicate-preserving tensor naming."
        ) from exc
    rows: list[tuple[str, str, bool | None, Any]] = []
    observed_names: set[tuple[str, str]] = set()
    for kind, values in (("parameter", parameters), ("buffer", buffers)):
        for raw_name, value in values:
            name = str(raw_name)
            key = (kind, name)
            if not name or key in observed_names:
                raise Lfm2DirectLocalError(
                    "LFM2 runtime tensor names must be nonempty and unique "
                    "within their state kind."
                )
            observed_names.add(key)
            if (
                not hasattr(value, "untyped_storage")
                or str(value.layout) != "torch.strided"
                or value.device.type == "meta"
            ):
                raise Lfm2DirectLocalError(
                    "LFM2 runtime state contains a non-materialized or "
                    "non-strided tensor."
                )
            persistent = None
            if kind == "buffer":
                module_path, _, leaf_name = name.rpartition(".")
                owner = model if not module_path else get_submodule(module_path)
                non_persistent = getattr(
                    owner,
                    "_non_persistent_buffers_set",
                    None,
                )
                if not isinstance(non_persistent, set):
                    raise Lfm2DirectLocalError(
                        "LFM2 buffer persistence metadata is unavailable."
                    )
                persistent = leaf_name not in non_persistent
            rows.append((name, kind, persistent, value))
    if not rows:
        raise Lfm2DirectLocalError("LFM2 model has no named runtime tensors.")
    return tuple(sorted(rows, key=lambda row: (row[1], row[0])))


def _lfm2_direct_local_state_alias_abi_from_entries(
    entries: tuple[tuple[str, str, bool | None, Any], ...],
) -> dict[str, Any]:
    """Build the legacy alias ABI from one already-enumerated runtime state."""
    identities: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    tensors = []
    for name, kind, persistent, value in entries:
        member = {
            "name": name,
            "kind": kind,
            "persistent": persistent,
            "shape": [int(size) for size in value.shape],
            "stride": [int(size) for size in value.stride()],
            "storage_offset": int(value.storage_offset()),
        }
        tensors.append(member)
        if int(value.numel()) <= 0:
            continue
        storage = value.untyped_storage()
        # Storage offset belongs to the member, not the storage identity.  Two
        # overlapping views at different offsets must remain in one group.
        identity = (
            str(value.device),
            int(storage.data_ptr()),
            int(storage.nbytes()),
        )
        identities.setdefault(identity, []).append(member)
    shared_storage_groups = sorted(
        (
            {
                "members": sorted(
                    members,
                    key=lambda row: (row["kind"], row["name"]),
                )
            }
            for members in identities.values()
            if len(members) > 1
        ),
        key=lambda group: [
            (row["kind"], row["name"]) for row in group["members"]
        ],
    )
    body = {
        "tensors": tensors,
        "shared_storage_groups": shared_storage_groups,
        "member_offsets_strides_and_shapes_sealed": True,
        "partial_and_overlapping_storage_views_sealed": True,
        "non_persistent_buffers_included": True,
    }
    return {**body, "abi_sha256": canonical_sha256(body)}


def _lfm2_direct_local_state_alias_abi(model: Any) -> dict[str, Any]:
    """Seal names, views, and complete shared-storage/overlap topology."""

    return _lfm2_direct_local_state_alias_abi_from_entries(
        _named_lfm2_runtime_tensors(model)
    )


def _runtime_tensor_storage_byte_span(value: Any) -> dict[str, Any]:
    """Return the storage-relative bounding byte interval of one strided view."""

    element_size = int(value.element_size())
    storage_offset = int(value.storage_offset())
    storage_nbytes = int(value.untyped_storage().nbytes())
    if int(value.numel()) == 0:
        start = storage_offset * element_size
        end = start
    else:
        minimum_offset = storage_offset
        maximum_offset = storage_offset
        for size, stride in zip(value.shape, value.stride(), strict=True):
            delta = (int(size) - 1) * int(stride)
            minimum_offset += min(0, delta)
            maximum_offset += max(0, delta)
        start = minimum_offset * element_size
        end = (maximum_offset + 1) * element_size
    if start < 0 or end < start or end > storage_nbytes:
        raise Lfm2DirectLocalError(
            "LFM2 runtime tensor view has an invalid storage-relative byte span."
        )
    return {
        "start": start,
        "end_exclusive": end,
        "bounding_byte_count": end - start,
        "empty": int(value.numel()) == 0,
    }


def _runtime_tensor_storage_identity(value: Any) -> tuple[Any, ...]:
    storage = value.untyped_storage()
    storage_nbytes = int(storage.nbytes())
    if storage_nbytes > 0:
        return (
            str(value.device),
            int(storage.data_ptr()),
            storage_nbytes,
        )
    # Empty tensors have no addressable bytes and commonly report data_ptr=0.
    # PyTorch's internal storage handle is nevertheless stable across distinct
    # views of the same zero-byte storage.  Use it only for in-process grouping;
    # it is never serialized into the manifest.
    storage_handle = getattr(storage, "_cdata", None)
    if isinstance(storage_handle, bool) or not isinstance(storage_handle, int):
        raise Lfm2DirectLocalError(
            "An empty LFM2 runtime tensor has no stable storage identity."
        )
    return (str(value.device), "empty", storage_handle)


def build_lfm2_direct_local_runtime_alias_manifest(model: Any) -> dict[str, Any]:
    """Describe every live state view and its complete shared-storage topology.

    Storage addresses are used only to discover membership and are never
    serialized. Stable group identifiers derive solely from canonical member
    names, so repeated manifests for unchanged runtime state are byte-stable.
    Byte spans are storage-relative bounding intervals; for non-contiguous
    views they may include holes, which is declared explicitly in the manifest.
    """

    entries = _named_lfm2_runtime_tensors(model)
    legacy_alias_abi = _lfm2_direct_local_state_alias_abi_from_entries(entries)
    state_entries: list[dict[str, Any]] = []
    storage_members: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for name, kind, persistent, value in entries:
        byte_span = _runtime_tensor_storage_byte_span(value)
        entry = {
            "name": name,
            "kind": kind,
            "persistent": persistent,
            "shape": [int(size) for size in value.shape],
            "stride": [int(size) for size in value.stride()],
            "storage_offset": int(value.storage_offset()),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "numel": int(value.numel()),
            "element_size_bytes": int(value.element_size()),
            "storage_nbytes": int(value.untyped_storage().nbytes()),
            "storage_byte_span": byte_span,
            "contiguous": bool(value.is_contiguous()),
        }
        state_entries.append(entry)
        storage_members.setdefault(_runtime_tensor_storage_identity(value), []).append(
            entry
        )

    ordered_groups = sorted(
        storage_members.values(),
        key=lambda members: sorted((row["kind"], row["name"]) for row in members),
    )
    entry_group_ids: dict[tuple[str, str], str] = {}
    storage_groups: list[dict[str, Any]] = []
    overlap_pairs: list[dict[str, Any]] = []
    for group_index, members in enumerate(ordered_groups):
        group_id = f"storage-{group_index:06d}"
        ordered_members = sorted(members, key=lambda row: (row["kind"], row["name"]))
        member_references = []
        for member in ordered_members:
            entry_group_ids[(member["kind"], member["name"])] = group_id
            member_references.append(
                {
                    "kind": member["kind"],
                    "name": member["name"],
                    "storage_byte_span": dict(member["storage_byte_span"]),
                }
            )
        group_overlap_pairs = []
        for left_index, left in enumerate(ordered_members):
            left_span = left["storage_byte_span"]
            for right in ordered_members[left_index + 1 :]:
                right_span = right["storage_byte_span"]
                overlap_start = max(left_span["start"], right_span["start"])
                overlap_end = min(
                    left_span["end_exclusive"],
                    right_span["end_exclusive"],
                )
                if overlap_start >= overlap_end:
                    continue
                if left_span == right_span:
                    overlap_kind = "exact_span"
                elif (
                    left_span["start"] <= right_span["start"]
                    and left_span["end_exclusive"] >= right_span["end_exclusive"]
                ) or (
                    right_span["start"] <= left_span["start"]
                    and right_span["end_exclusive"] >= left_span["end_exclusive"]
                ):
                    overlap_kind = "containment"
                else:
                    overlap_kind = "partial_span"
                pair = {
                    "storage_group_id": group_id,
                    "left": {"kind": left["kind"], "name": left["name"]},
                    "right": {"kind": right["kind"], "name": right["name"]},
                    "overlap_kind": overlap_kind,
                    "overlap_start": overlap_start,
                    "overlap_end_exclusive": overlap_end,
                    "overlap_byte_count": overlap_end - overlap_start,
                }
                group_overlap_pairs.append(pair)
                overlap_pairs.append(pair)
        storage_groups.append(
            {
                "storage_group_id": group_id,
                "device": ordered_members[0]["device"],
                "storage_nbytes": ordered_members[0]["storage_nbytes"],
                "members": member_references,
                "member_count": len(member_references),
                "shared": len(member_references) > 1,
                "byte_span_overlap_pair_count": len(group_overlap_pairs),
            }
        )

    canonical_entries = [
        {
            **entry,
            "storage_group_id": entry_group_ids[(entry["kind"], entry["name"])],
        }
        for entry in state_entries
    ]
    body = {
        "schema": RUNTIME_TENSOR_ALIAS_MANIFEST_SCHEMA,
        "state_entries": canonical_entries,
        "storage_groups": storage_groups,
        "overlapping_byte_span_pairs": overlap_pairs,
        "state_entry_count": len(canonical_entries),
        "storage_group_count": len(storage_groups),
        "shared_storage_group_count": sum(
            bool(group["shared"]) for group in storage_groups
        ),
        "overlapping_byte_span_pair_count": len(overlap_pairs),
        "legacy_alias_abi_sha256": legacy_alias_abi["abi_sha256"],
        "storage_addresses_serialized": False,
        "storage_byte_spans_are_relative": True,
        "noncontiguous_byte_spans_are_bounding_intervals": True,
        "partial_and_overlapping_storage_views_sealed": True,
        "non_persistent_buffers_included": True,
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def _lfm2_direct_local_state_storage_pointers(model: Any) -> set[int]:
    return {
        int(value.untyped_storage().data_ptr())
        for _name, _kind, _persistent, value in (
            _named_lfm2_runtime_tensors(model)
        )
        if int(value.numel()) > 0
    }


def _unique_runtime_tensor_elements(
    entries: tuple[tuple[str, str, bool | None, Any], ...],
    *,
    kind: str,
) -> int:
    seen: set[int] = set()
    total = 0
    for _name, observed_kind, _persistent, value in entries:
        if observed_kind != kind or id(value) in seen:
            continue
        seen.add(id(value))
        total += int(value.numel())
    return total


def _runtime_tensor_value_rows(
    entries: tuple[tuple[str, str, bool | None, Any], ...],
    *,
    cast_floating_to: Any | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for name, kind, persistent, value in entries:
        tensor = value.detach().to(device="cpu")
        if cast_floating_to is not None and (
            tensor.is_floating_point() or tensor.is_complex()
        ):
            tensor = tensor.to(dtype=cast_floating_to)
        rows.append(
            {
                "name": name,
                "kind": kind,
                "persistent": persistent,
                "shape": [int(size) for size in value.shape],
                "stride": [int(size) for size in value.stride()],
                "dtype": str(tensor.dtype),
                "sha256": _tensor_sha256(tensor),
            }
        )
    return rows


def fingerprint_lfm2_direct_local_runtime_tensor_state(
    model: Any,
) -> dict[str, Any]:
    """Seal all parameter/buffer bytes and storage ABI for live execution."""

    entries = _named_lfm2_runtime_tensors(model)
    alias_abi = _lfm2_direct_local_state_alias_abi(model)
    body = {
        "tensor_values": _runtime_tensor_value_rows(entries),
        "alias_abi_sha256": alias_abi["abi_sha256"],
        "unique_parameter_elements": _unique_runtime_tensor_elements(
            entries,
            kind="parameter",
        ),
        "unique_buffer_elements": _unique_runtime_tensor_elements(
            entries,
            kind="buffer",
        ),
        "named_tensor_occurrence_count": len(entries),
        "non_persistent_buffers_included": True,
    }
    return {**body, "state_sha256": canonical_sha256(body)}


def _expected_lfm2_post_cast_tensor_state(
    model: Any,
    *,
    runtime_dtype: Any,
) -> dict[str, Any]:
    entries = _named_lfm2_runtime_tensors(model)
    alias_abi = _lfm2_direct_local_state_alias_abi(model)
    body = {
        "tensor_values": _runtime_tensor_value_rows(
            entries,
            cast_floating_to=runtime_dtype,
        ),
        "alias_abi_sha256": alias_abi["abi_sha256"],
        "unique_parameter_elements": _unique_runtime_tensor_elements(
            entries,
            kind="parameter",
        ),
        "unique_buffer_elements": _unique_runtime_tensor_elements(
            entries,
            kind="buffer",
        ),
        "named_tensor_occurrence_count": len(entries),
        "non_persistent_buffers_included": True,
    }
    return {**body, "state_sha256": canonical_sha256(body)}


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
    # A nonuniform child cannot accept its state through the native uniform
    # module graph.  The authenticated descriptor is part of config.json, so
    # reconstruct the physical blank *before* enumerating or copying state.
    # Ordinary LFM2 checkpoints have no descriptor and retain the exact legacy
    # path below.
    try:
        from .hf_lfm2_dynamic_blank_materializer import (
            extract_lfm2_dynamic_physical_layout_from_config,
            materialize_lfm2_dynamic_blank_child,
        )

        dynamic_binding_fields = (
            "neural_forge_dynamic_blank_config_schema",
            "neural_forge_dynamic_physical_layout",
            "neural_forge_dynamic_physical_layout_sha256",
        )
        has_dynamic_binding = any(
            hasattr(config, name) for name in dynamic_binding_fields
        )
        dynamic_layout = (
            extract_lfm2_dynamic_physical_layout_from_config(config)
            if has_dynamic_binding
            else None
        )
        if dynamic_layout is not None:
            materialize_lfm2_dynamic_blank_child(
                model,
                config,
                dynamic_layout,
                expected_descriptor_sha256=dynamic_layout.descriptor_sha256,
            )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2DirectLocalError(
            "Local dynamic LFM2 physical layout could not be authenticated "
            "and materialized before state loading."
        ) from exc
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
class Lfm2DirectLocalPortableLoadReceipt:
    """Audit record for strict legacy-policy authentication then relocation."""

    provenance_resource_fingerprint: str
    provenance_device: str
    provenance_dtype: str
    staging_device: str
    runtime_device: str
    runtime_dtype: str
    staged_state_sha256: str
    runtime_state_sha256: str
    device_relocation_performed: bool
    dtype_conversion_performed: bool

    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema": PORTABLE_LOAD_RECEIPT_SCHEMA,
            "authentication": (
                "strict-v1-reference-at-declared-provenance-before-relocation"
            ),
            "provenance_resource_fingerprint": (
                self.provenance_resource_fingerprint
            ),
            "provenance_device": self.provenance_device,
            "provenance_dtype": self.provenance_dtype,
            "staging_device": self.staging_device,
            "runtime_device": self.runtime_device,
            "runtime_dtype": self.runtime_dtype,
            "staged_state_sha256": self.staged_state_sha256,
            "runtime_state_sha256": self.runtime_state_sha256,
            "device_relocation_performed": self.device_relocation_performed,
            "dtype_conversion_performed": self.dtype_conversion_performed,
            "runtime_policy_used_to_authenticate_legacy_reference": False,
            "tokenizer_bridge_reference_authenticated": True,
            "target_tokenizer_files_authenticated": True,
            "target_tokenizer_semantics_authenticated": True,
            "network_allowed": False,
        }
        return {**body, "receipt_sha256": canonical_sha256(body)}


@dataclass(frozen=True, slots=True)
class Lfm2CanonicalFp32RuntimeRelocationReceipt:
    """Proof that an authenticated FP32 model was only relocated for use."""

    provenance_resource_fingerprint: str
    canonical_device: str
    runtime_device: str
    runtime_dtype: str
    canonical_state_sha256: str
    expected_runtime_state_sha256: str
    runtime_state_sha256: str
    canonical_state_alias_abi_sha256: str
    runtime_state_alias_abi_sha256: str
    canonical_unique_parameter_elements: int
    runtime_unique_parameter_elements: int
    executable_forward_smoke: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema": CANONICAL_FP32_RUNTIME_RELOCATION_SCHEMA,
            "authentication": (
                "strict-caller-authentication-at-canonical-fp32-before-"
                "runtime-relocation"
            ),
            "provenance_resource_fingerprint": (
                self.provenance_resource_fingerprint
            ),
            "canonical_device": self.canonical_device,
            "canonical_dtype": "torch.float32",
            "runtime_device": self.runtime_device,
            "runtime_dtype": self.runtime_dtype,
            "canonical_state_sha256": self.canonical_state_sha256,
            "expected_runtime_state_sha256": (
                self.expected_runtime_state_sha256
            ),
            "runtime_state_sha256": self.runtime_state_sha256,
            "canonical_state_alias_abi_sha256": (
                self.canonical_state_alias_abi_sha256
            ),
            "runtime_state_alias_abi_sha256": (
                self.runtime_state_alias_abi_sha256
            ),
            "deterministic_post_cast_state_match": (
                self.expected_runtime_state_sha256
                == self.runtime_state_sha256
            ),
            "state_names_shapes_and_alias_abi_preserved": (
                self.canonical_state_alias_abi_sha256
                == self.runtime_state_alias_abi_sha256
            ),
            "canonical_unique_parameter_elements": (
                self.canonical_unique_parameter_elements
            ),
            "runtime_unique_parameter_elements": (
                self.runtime_unique_parameter_elements
            ),
            "executable_forward_smoke": dict(
                self.executable_forward_smoke
            ),
            "exact_parameter_count_invariant": (
                self.canonical_unique_parameter_elements
                == self.runtime_unique_parameter_elements
            ),
            "canonical_artifact_rewritten": False,
            "canonical_disk_artifact_mutated": False,
            "fresh_loaded_runtime_copy_required": True,
            "caller_owned_canonical_copy_mutated": False,
            "transactional_runtime_copy_created": True,
            "returned_runtime_copy_mutated_in_place": True,
            "failed_runtime_copy_must_be_discarded": True,
            "runtime_dtype_parameter_compaction_credit": 0,
            "runtime_device_parameter_compaction_credit": 0,
            "runtime_precision_changes_counted_as_folding": False,
            "runtime_precision_changes_counted_as_quantized_compaction": False,
            "parent_forward_calls": 0,
            "training_steps": 0,
            "network_allowed": False,
            "silent_device_or_dtype_fallback_allowed": False,
        }
        return {**body, "receipt_sha256": canonical_sha256(body)}


def relocate_authenticated_lfm2_canonical_fp32_model(
    model: Any,
    *,
    provenance_resource_fingerprint: str,
    runtime_device: Any,
    runtime_dtype: Any,
    fresh_loaded_runtime_copy: bool,
) -> tuple[Any, Lfm2CanonicalFp32RuntimeRelocationReceipt]:
    """Relocate one already-authenticated canonical FP32 model in memory.

    The caller must complete its strict checkpoint/package authentication
    before entering this boundary.  This function refuses non-FP32 provenance,
    performs exactly the requested ``Module.to`` operation, and gives device or
    dtype conversion zero parameter-compaction credit.  ``model`` must be a
    freshly loaded CPU runtime source.  This boundary deep-copies that source,
    applies ``Module.to`` only to the private transaction, and never writes the
    canonical disk artifact or mutates the caller-owned CPU source.  If any
    conversion, byte/ABI postcondition, or executable-forward check fails, the
    private runtime transaction is discarded and no success receipt is
    returned.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise Lfm2DirectLocalError(
            "PyTorch is required for portable LFM2 relocation."
        ) from exc
    if (
        not isinstance(provenance_resource_fingerprint, str)
        or not provenance_resource_fingerprint.strip()
    ):
        raise Lfm2DirectLocalError(
            "Canonical FP32 relocation requires a nonempty provenance "
            "resource fingerprint."
        )
    if fresh_loaded_runtime_copy is not True:
        raise Lfm2DirectLocalError(
            "Canonical FP32 relocation requires an explicitly attested fresh "
            "loaded runtime copy."
        )
    supported_dtypes = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
    if not isinstance(runtime_dtype, torch.dtype) or runtime_dtype not in (
        supported_dtypes
    ):
        raise Lfm2DirectLocalError(
            "Runtime dtype must be explicitly FP16, BF16, FP32, or FP64."
        )
    try:
        requested_runtime_device = torch.device(runtime_device)
    except (TypeError, RuntimeError) as exc:
        raise Lfm2DirectLocalError(
            "Canonical FP32 relocation received an invalid runtime device."
        ) from exc
    if (
        requested_runtime_device.type == "cuda"
        and requested_runtime_device.index is None
    ):
        raise Lfm2DirectLocalError(
            "CUDA runtime relocation requires an explicit accelerator index "
            "such as cuda:0; generic cuda is rejected without fallback."
        )
    runtime_cuda_compute_capability = None
    if (
        requested_runtime_device.type == "cuda"
        and runtime_dtype is torch.bfloat16
    ):
        try:
            major, minor = torch.cuda.get_device_capability(
                requested_runtime_device
            )
        except (AssertionError, TypeError, ValueError, RuntimeError) as exc:
            raise Lfm2DirectLocalError(
                "CUDA BF16 runtime capability could not be authenticated; "
                "no fallback was attempted."
            ) from exc
        runtime_cuda_compute_capability = f"{int(major)}.{int(minor)}"
        if int(major) < 8:
            raise Lfm2DirectLocalError(
                "CUDA BF16 runtime requires native compute capability 8.0 "
                "or newer; no fallback was attempted."
            )
    if not callable(getattr(model, "to", None)):
        raise Lfm2DirectLocalError(
            "Canonical FP32 relocation requires a materialized torch module."
        )
    canonical_entries = _named_lfm2_runtime_tensors(model)
    canonical_values = tuple(row[3] for row in canonical_entries)
    canonical_devices = {value.device for value in canonical_values}
    canonical_floating_dtypes = {
        value.dtype
        for value in canonical_values
        if value.is_floating_point() or value.is_complex()
    }
    if (
        len(canonical_devices) != 1
        or any(value.device.type == "meta" for value in canonical_values)
        or canonical_floating_dtypes != {torch.float32}
    ):
        raise Lfm2DirectLocalError(
            "Portable runtime relocation requires one fully materialized "
            "canonical FP32 model state."
    )
    canonical_device = next(iter(canonical_devices))
    if canonical_device.type != "cpu":
        raise Lfm2DirectLocalError(
            "Portable runtime relocation must authenticate and stage the "
            "canonical FP32 source on CPU before touching an accelerator."
        )
    canonical_state = fingerprint_lfm2_direct_local_runtime_tensor_state(model)
    canonical_state_sha256 = str(canonical_state["state_sha256"])
    canonical_state_alias_abi = _lfm2_direct_local_state_alias_abi(model)
    canonical_parameter_elements = (
        count_lfm2_direct_local_unique_parameter_elements(model)
    )
    expected_runtime_state = _expected_lfm2_post_cast_tensor_state(
        model,
        runtime_dtype=runtime_dtype,
    )
    expected_runtime_state_sha256 = str(expected_runtime_state["state_sha256"])
    canonical_storage_pointers = (
        _lfm2_direct_local_state_storage_pointers(model)
    )
    try:
        runtime_model = copy.deepcopy(model)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2DirectLocalError(
            "Canonical FP32 runtime copy could not be created transactionally."
        ) from exc
    if runtime_model is model:
        raise Lfm2DirectLocalError(
            "Canonical FP32 runtime copy aliases its caller-owned source."
        )
    if (
        fingerprint_lfm2_direct_local_runtime_tensor_state(runtime_model)
        != canonical_state
        or _lfm2_direct_local_state_alias_abi(runtime_model)["abi_sha256"]
        != canonical_state_alias_abi["abi_sha256"]
        or count_lfm2_direct_local_unique_parameter_elements(runtime_model)
        != canonical_parameter_elements
        or canonical_storage_pointers
        & _lfm2_direct_local_state_storage_pointers(runtime_model)
    ):
        raise Lfm2DirectLocalError(
            "Transactional runtime copy changed or shared canonical FP32 "
            "state, names, shapes, aliases, or parameter count."
        )
    try:
        relocated_result = runtime_model.to(
            device=requested_runtime_device,
            dtype=runtime_dtype,
        )
    except (AssertionError, TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2DirectLocalError(
            "Canonical FP32 model could not be relocated to the exact "
            "requested runtime policy; no fallback was attempted."
        ) from exc
    if relocated_result is not runtime_model:
        raise Lfm2DirectLocalError(
            "Canonical FP32 relocation did not mutate exactly the supplied "
            "fresh runtime copy."
        )
    runtime_model.requires_grad_(False)
    runtime_model.eval()

    runtime_entries = _named_lfm2_runtime_tensors(runtime_model)
    runtime_values = tuple(row[3] for row in runtime_entries)
    runtime_devices = {value.device for value in runtime_values}
    runtime_floating_dtypes = {
        value.dtype
        for value in runtime_values
        if value.is_floating_point() or value.is_complex()
    }
    if len(runtime_devices) != 1:
        raise Lfm2DirectLocalError(
            "Relocated LFM2 state spans more than one runtime device."
        )
    observed_runtime_device = next(iter(runtime_devices))
    if (
        observed_runtime_device.type != requested_runtime_device.type
        or (
            requested_runtime_device.index is not None
            and observed_runtime_device.index != requested_runtime_device.index
        )
    ):
        raise Lfm2DirectLocalError(
            "Relocated LFM2 state is not on the exact requested device."
        )
    if runtime_floating_dtypes != {runtime_dtype}:
        raise Lfm2DirectLocalError(
            "Relocated LFM2 floating state is not in the exact requested dtype."
        )
    runtime_parameter_elements = (
        count_lfm2_direct_local_unique_parameter_elements(runtime_model)
    )
    if runtime_parameter_elements != canonical_parameter_elements:
        raise Lfm2DirectLocalError(
            "Runtime relocation changed the exact unique parameter count."
        )
    pre_forward_runtime_state = (
        fingerprint_lfm2_direct_local_runtime_tensor_state(runtime_model)
    )
    pre_forward_runtime_state_sha256 = str(
        pre_forward_runtime_state["state_sha256"]
    )
    runtime_state_alias_abi = _lfm2_direct_local_state_alias_abi(
        runtime_model
    )
    if (
        pre_forward_runtime_state_sha256 != expected_runtime_state_sha256
        or runtime_state_alias_abi["abi_sha256"]
        != canonical_state_alias_abi["abi_sha256"]
    ):
        raise Lfm2DirectLocalError(
            "Runtime relocation differs from the deterministic CPU FP32 "
            "post-cast bytes, names, shapes, or alias ABI."
        )
    if getattr(getattr(runtime_model, "config", None), "model_type", None) != (
        "lfm2"
    ):
        raise Lfm2DirectLocalError(
            "Executable portable relocation requires a concrete LFM2 causal "
            "model configuration."
        )
    input_embeddings = getattr(runtime_model, "get_input_embeddings", None)
    if not callable(input_embeddings) or not callable(runtime_model):
        raise Lfm2DirectLocalError(
            "Relocated runtime copy does not expose an executable causal LFM "
            "forward interface."
        )
    input_table_module = input_embeddings()
    input_table = getattr(input_table_module, "weight", None)
    if (
        not isinstance(input_table, torch.Tensor)
        or input_table.ndim != 2
        or int(input_table.shape[0]) <= 0
    ):
        raise Lfm2DirectLocalError(
            "Relocated runtime copy has no usable input-token table."
        )
    second_token = min(1, int(input_table.shape[0]) - 1)
    smoke_input_ids = torch.tensor(
        [[0, second_token]],
        device=requested_runtime_device,
        dtype=torch.int64,
    )
    smoke_attention_mask = torch.ones_like(smoke_input_ids)
    try:
        with torch.no_grad():
            smoke_output = runtime_model(
                input_ids=smoke_input_ids,
                attention_mask=smoke_attention_mask,
                use_cache=False,
                return_dict=True,
            )
    except (AssertionError, TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2DirectLocalError(
            "Relocated runtime storage succeeded but its minimal no-grad "
            "use_cache=False LFM forward failed; the fresh runtime copy must "
            "be discarded and no fallback was attempted."
        ) from exc
    smoke_logits = getattr(smoke_output, "logits", None)
    if smoke_logits is None and isinstance(smoke_output, Mapping):
        smoke_logits = smoke_output.get("logits")
    if (
        not isinstance(smoke_logits, torch.Tensor)
        or smoke_logits.ndim != 3
        or tuple(int(size) for size in smoke_logits.shape[:2]) != (1, 2)
        or int(smoke_logits.shape[-1]) <= 0
        or smoke_logits.device.type != requested_runtime_device.type
        or (
            requested_runtime_device.index is not None
            and smoke_logits.device.index != requested_runtime_device.index
        )
        or smoke_logits.dtype is not runtime_dtype
        or not smoke_logits.is_floating_point()
        or smoke_logits.is_complex()
        or not bool(torch.isfinite(smoke_logits.detach()).all())
    ):
        raise Lfm2DirectLocalError(
            "Relocated minimal LFM forward did not return finite logits in "
            "the exact requested runtime device and dtype."
        )
    post_forward_runtime_parameter_elements = (
        count_lfm2_direct_local_unique_parameter_elements(runtime_model)
    )
    runtime_state = fingerprint_lfm2_direct_local_runtime_tensor_state(
        runtime_model
    )
    runtime_state_sha256 = str(runtime_state["state_sha256"])
    if (
        post_forward_runtime_parameter_elements != runtime_parameter_elements
        or runtime_state != pre_forward_runtime_state
    ):
        raise Lfm2DirectLocalError(
            "Minimal executable LFM forward mutated runtime model state or "
            "its exact parameter count."
        )
    forward_body = {
        "schema": "neural-forge-lfm2-portable-runtime-forward-smoke-v1",
        "input_shape": [1, 2],
        "input_dtype": "torch.int64",
        "runtime_device": str(smoke_logits.device),
        "runtime_dtype": str(smoke_logits.dtype),
        "logits_shape": [int(size) for size in smoke_logits.shape],
        "logits_finite": True,
        "torch_no_grad": True,
        "use_cache": False,
        "return_dict": True,
        "runtime_state_unchanged_by_forward": True,
        "runtime_parameter_count_unchanged_by_forward": True,
        "runtime_cuda_compute_capability": runtime_cuda_compute_capability,
        "silent_device_or_dtype_fallback_used": False,
    }
    executable_forward_smoke = MappingProxyType(
        {
            **forward_body,
            "smoke_sha256": canonical_sha256(forward_body),
        }
    )
    if (
        runtime_dtype is torch.float32
        and runtime_state_sha256 != canonical_state_sha256
    ):
        raise Lfm2DirectLocalError(
            "Device-only canonical FP32 relocation changed state bytes."
        )
    receipt = Lfm2CanonicalFp32RuntimeRelocationReceipt(
        provenance_resource_fingerprint=provenance_resource_fingerprint,
        canonical_device=str(canonical_device),
        runtime_device=str(observed_runtime_device),
        runtime_dtype=str(runtime_dtype),
        canonical_state_sha256=canonical_state_sha256,
        expected_runtime_state_sha256=expected_runtime_state_sha256,
        runtime_state_sha256=runtime_state_sha256,
        canonical_state_alias_abi_sha256=str(
            canonical_state_alias_abi["abi_sha256"]
        ),
        runtime_state_alias_abi_sha256=str(
            runtime_state_alias_abi["abi_sha256"]
        ),
        canonical_unique_parameter_elements=canonical_parameter_elements,
        runtime_unique_parameter_elements=runtime_parameter_elements,
        executable_forward_smoke=executable_forward_smoke,
    )
    if receipt.as_dict()["exact_parameter_count_invariant"] is not True:
        raise Lfm2DirectLocalError(
            "Canonical FP32 runtime relocation lost parameter-count invariance."
        )
    if (
        fingerprint_lfm2_direct_local_runtime_tensor_state(model)
        != canonical_state
        or count_lfm2_direct_local_unique_parameter_elements(model)
        != canonical_parameter_elements
        or _lfm2_direct_local_state_alias_abi(model)["abi_sha256"]
        != canonical_state_alias_abi["abi_sha256"]
    ):
        raise Lfm2DirectLocalError(
            "Transactional relocation mutated its caller-owned canonical "
            "FP32 source."
        )
    return runtime_model, receipt


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

    def _authenticate_checkpoint(self, source: Path) -> Path:
        """Authenticate one checkpoint against this exact legacy v1 policy."""

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
        return directory

    def load(self, source: Path) -> Any:
        """Strictly load using the same device/dtype policy sealed at save."""

        directory = self._authenticate_checkpoint(source)
        return load_lfm2_direct_local_model(
            directory,
            device=self.device,
            dtype=self.dtype,
        )

    def load_authenticated_portable(
        self,
        source: Path,
        *,
        runtime_device: Any,
        runtime_dtype: Any,
    ) -> tuple[Any, Lfm2DirectLocalPortableLoadReceipt]:
        """Authenticate v1 provenance, then explicitly relocate/cast for use.

        ``self.device`` and ``self.dtype`` remain the legacy save-time policy
        sealed into the checkpoint reference.  They are never replaced by the
        requested runtime policy during authentication.  The model is replayed
        on CPU at the authenticated provenance dtype before a separately
        receipted ``Module.to`` operation.  This preserves strict legacy
        authentication while allowing an artifact saved for CPU to run on a
        GPU (or vice versa) without forging a new reference fingerprint.
        """

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise Lfm2DirectLocalError(
                "PyTorch is required for portable LFM2 loading."
            ) from exc
        supported_dtypes = {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        }
        for value, name in (
            (self.dtype, "authenticated provenance dtype"),
            (runtime_dtype, "runtime dtype"),
        ):
            if not isinstance(value, torch.dtype) or value not in supported_dtypes:
                raise Lfm2DirectLocalError(
                    f"{name} must be FP16, BF16, FP32, or FP64."
                )
        try:
            requested_runtime_device = torch.device(runtime_device)
            provenance_device = torch.device(self.device)
        except (TypeError, RuntimeError) as exc:
            raise Lfm2DirectLocalError(
                "Portable LFM2 load received an invalid device policy."
            ) from exc

        directory = self._authenticate_checkpoint(source)
        staging_device = torch.device("cpu")
        model = load_lfm2_direct_local_model(
            directory,
            device=staging_device,
            dtype=self.dtype,
        )
        staged_state_sha256 = fingerprint_lfm2_direct_local_state(model)
        try:
            model.to(
                device=requested_runtime_device,
                dtype=runtime_dtype,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise Lfm2DirectLocalError(
                "Authenticated checkpoint could not be relocated to the "
                "requested runtime policy."
            ) from exc
        model.requires_grad_(False)
        model.eval()

        state_values = tuple(model.state_dict().values())
        if not state_values:
            raise Lfm2DirectLocalError(
                "Relocated LFM2 model unexpectedly has no state."
            )
        observed_devices = {value.device for value in state_values}
        if len(observed_devices) != 1:
            raise Lfm2DirectLocalError(
                "Relocated LFM2 state spans multiple devices."
            )
        observed_runtime_device = next(iter(observed_devices))
        if (
            observed_runtime_device.type != requested_runtime_device.type
            or (
                requested_runtime_device.index is not None
                and observed_runtime_device.index
                != requested_runtime_device.index
            )
        ):
            raise Lfm2DirectLocalError(
                "Relocated LFM2 state is not on the requested device."
            )
        floating_dtypes = {
            value.dtype
            for value in state_values
            if value.is_floating_point() or value.is_complex()
        }
        if floating_dtypes != {runtime_dtype}:
            raise Lfm2DirectLocalError(
                "Relocated LFM2 floating state is not in the requested dtype."
            )

        runtime_state_sha256 = fingerprint_lfm2_direct_local_state(model)
        dtype_conversion_performed = str(self.dtype) != str(runtime_dtype)
        if (
            not dtype_conversion_performed
            and runtime_state_sha256 != staged_state_sha256
        ):
            raise Lfm2DirectLocalError(
                "Device-only checkpoint relocation changed authenticated "
                "state bytes."
            )
        receipt = Lfm2DirectLocalPortableLoadReceipt(
            provenance_resource_fingerprint=self.resource_fingerprint,
            provenance_device=str(provenance_device),
            provenance_dtype=str(self.dtype),
            staging_device=str(staging_device),
            runtime_device=str(observed_runtime_device),
            runtime_dtype=str(runtime_dtype),
            staged_state_sha256=staged_state_sha256,
            runtime_state_sha256=runtime_state_sha256,
            device_relocation_performed=(
                str(provenance_device) != str(observed_runtime_device)
            ),
            dtype_conversion_performed=dtype_conversion_performed,
        )
        return model, receipt


__all__ = [
    "BRIDGE_REFERENCE_FILENAME",
    "CANONICAL_FP32_RUNTIME_RELOCATION_SCHEMA",
    "Lfm2CanonicalFp32RuntimeRelocationReceipt",
    "Lfm2DirectLocalCheckpointIO",
    "Lfm2DirectLocalError",
    "Lfm2DirectLocalPortableLoadReceipt",
    "PORTABLE_LOAD_RECEIPT_SCHEMA",
    "RUNTIME_TENSOR_ALIAS_MANIFEST_SCHEMA",
    "SCHEMA",
    "build_lfm2_direct_local_runtime_alias_manifest",
    "count_lfm2_direct_local_unique_parameter_elements",
    "fingerprint_lfm2_direct_local_runtime_tensor_state",
    "fingerprint_lfm2_direct_local_state",
    "load_lfm2_direct_local_config",
    "load_lfm2_direct_local_model",
    "load_lfm2_direct_local_tokenizer",
    "relocate_authenticated_lfm2_canonical_fp32_model",
]
