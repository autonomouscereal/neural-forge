"""Tiny safetensors packs for assignment-local Q/K/V/O residual factors.

The pack contains exactly eight learned factor tensors. Site topology, rank,
role-scale provenance, base/fixture hashes, and the complete selected adapter
state hash live in a canonical JSON manifest. No optimizer state, parent
tensor, base model tensor, or full-model checkpoint can enter this format.
"""

from __future__ import annotations

import hashlib
import json
import os
import operator
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn


QKVO_FACTOR_PACK_SCHEMA = "neural-forge-qkvo-factor-pack-v1"
QKVO_FACTOR_MANIFEST_SCHEMA = "neural-forge-qkvo-factor-manifest-v1"
QKVO_ROLE_SCALE_PROVENANCE_SCHEMA = "neural-forge-qkvo-role-scale-provenance-v1"
QKVO_SOURCE_PROVENANCE_SCHEMA = "neural-forge-qkvo-factor-source-provenance-v1"
MAX_QKVO_FACTOR_FILE_BYTES = 256 * 1024
MAX_QKVO_MANIFEST_FILE_BYTES = 128 * 1024
QKVO_FACTOR_KEYS = (
    "q_left",
    "q_right",
    "k_left",
    "k_right",
    "v_left",
    "v_right",
    "o_left",
    "o_right",
)
QKVO_REGISTRY_BUFFER_KEYS = (
    "site_indices",
    "_registry_schema_version",
    "_registry_layout",
)
QKVO_REGISTRY_STATE_KEYS = (*QKVO_FACTOR_KEYS, *QKVO_REGISTRY_BUFFER_KEYS)


class QKVOFactorPackError(RuntimeError):
    """A factor-only artifact violates its content or provenance contract."""


def _exact_int(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise QKVOFactorPackError(f"{name} must be an exact integer.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise QKVOFactorPackError(f"{name} must be an exact integer.") from error
    if result < (1 if positive else 0):
        raise QKVOFactorPackError(f"{name} is outside its valid range.")
    return result


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_provenance(
    role_scale_provenance: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        not isinstance(role_scale_provenance, Mapping)
        or role_scale_provenance.get("schema") != QKVO_ROLE_SCALE_PROVENANCE_SCHEMA
        or role_scale_provenance.get("role_scales_trainable") is not False
        or not _is_sha256(role_scale_provenance.get("table_sha256"))
    ):
        raise QKVOFactorPackError("Role-scale provenance schema is invalid.")
    for role in ("q", "k", "v", "o"):
        value = role_scale_provenance.get(role)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise QKVOFactorPackError("Role-scale values must be finite numbers.")
        number = float(value)
        if not torch.isfinite(torch.tensor(number)):
            raise QKVOFactorPackError("Role-scale values must be finite numbers.")
    required_source_hashes = (
        "source_report_file_sha256",
        "source_report_sha256",
        "fixture_provenance_sha256",
        "calibration_evaluation_sha256",
        "heldout_evaluation_sha256",
    )
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("schema") != QKVO_SOURCE_PROVENANCE_SCHEMA
        or any(not _is_sha256(provenance.get(name)) for name in required_source_hashes)
    ):
        raise QKVOFactorPackError("Factor source provenance schema is invalid.")
    # Canonical serialization here rejects NaN, infinity, or non-JSON objects
    # before any output reservation is made.
    role_payload = dict(role_scale_provenance)
    source_payload = dict(provenance)
    _canonical_json(role_payload)
    _canonical_json(source_payload)
    return role_payload, source_payload


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise QKVOFactorPackError(
            "Factor-pack provenance is not canonical JSON."
        ) from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def adapter_state_sha256(state: Mapping[str, Tensor]) -> str:
    return canonical_sha256(
        {name: tensor_sha256(value) for name, value in sorted(state.items())}
    )


def factor_tensors_from_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    if not isinstance(state, Mapping) or set(state) != set(QKVO_REGISTRY_STATE_KEYS):
        raise QKVOFactorPackError(
            "Adapter state must contain exactly eight factors and the three "
            "canonical immutable registry buffers."
        )
    factors = {
        key: state[key].detach().cpu().contiguous().clone() for key in QKVO_FACTOR_KEYS
    }
    if any(
        not value.dtype.is_floating_point
        or not bool(torch.isfinite(value).all())
        or value.ndim != 3
        for value in factors.values()
    ):
        raise QKVOFactorPackError(
            "Every factor must be one finite rank-three floating tensor."
        )
    return factors


def _canonical_registry_contract(
    state: Mapping[str, Tensor],
    factors: Mapping[str, Tensor],
    *,
    site: Mapping[str, Any],
    rank: int,
    registry_geometry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and serialize every omitted immutable registry value."""

    if set(factors) != set(QKVO_FACTOR_KEYS) or any(
        not isinstance(value, Tensor)
        or value.ndim != 3
        or not value.dtype.is_floating_point
        for value in factors.values()
    ):
        raise QKVOFactorPackError(
            "Every Q/K/V/O factor must be a rank-three floating tensor."
        )

    required_geometry = {"logical_layers", "physical_groups"}
    if not isinstance(registry_geometry, Mapping) or not required_geometry.issubset(
        registry_geometry
    ):
        raise QKVOFactorPackError(
            "registry_geometry requires logical_layers and physical_groups."
        )
    try:
        logical_layers = _exact_int(
            registry_geometry["logical_layers"], "logical_layers", positive=True
        )
        physical_groups = _exact_int(
            registry_geometry["physical_groups"],
            "physical_groups",
            positive=True,
        )
        rank_value = _exact_int(rank, "rank", positive=True)
        layer = _exact_int(site["layer_index"], "layer_index")
        group = _exact_int(site["physical_group"], "physical_group")
    except KeyError as error:
        raise QKVOFactorPackError(
            "Site, rank, and registry geometry must be canonical integers."
        ) from error
    if (
        logical_layers <= 0
        or physical_groups <= 0
        or rank_value <= 0
        or not 0 <= layer < logical_layers
        or not 0 <= group < physical_groups
    ):
        raise QKVOFactorPackError("Site/rank/registry geometry is out of range.")

    site_indices = state["site_indices"].detach().cpu().contiguous()
    schema_version = state["_registry_schema_version"].detach().cpu().contiguous()
    layout = state["_registry_layout"].detach().cpu().contiguous()
    if (
        site_indices.dtype != torch.int32
        or tuple(site_indices.shape) != (1, 2)
        or site_indices.tolist() != [[layer, group]]
        or schema_version.dtype != torch.int32
        or schema_version.ndim != 0
        or int(schema_version.item()) != 1
        or layout.dtype != torch.int32
        or layout.ndim != 0
        or int(layout.item()) != 1
    ):
        raise QKVOFactorPackError(
            "Immutable registry buffers do not match the one-site v1 layout."
        )

    dtypes = {value.dtype for value in factors.values()}
    if len(dtypes) != 1:
        raise QKVOFactorPackError("All eight factors must have one exact dtype.")
    if any(not bool(torch.isfinite(value).all()) for value in factors.values()):
        raise QKVOFactorPackError("All eight factors must contain only finite values.")
    factor_dtype = next(iter(dtypes))
    q_left = factors["q_left"]
    q_right = factors["q_right"]
    k_left = factors["k_left"]
    query_group_width = int(q_left.shape[1])
    hidden_size = int(q_right.shape[2])
    head_dim = int(k_left.shape[1])
    expected_shapes = {
        "q_left": (1, query_group_width, rank_value),
        "q_right": (1, rank_value, hidden_size),
        "k_left": (1, head_dim, rank_value),
        "k_right": (1, rank_value, hidden_size),
        "v_left": (1, head_dim, rank_value),
        "v_right": (1, rank_value, hidden_size),
        "o_left": (1, hidden_size, rank_value),
        "o_right": (1, rank_value, query_group_width),
    }
    if any(
        tuple(factors[name].shape) != shape for name, shape in expected_shapes.items()
    ):
        raise QKVOFactorPackError(
            "Q/K/V/O factor shapes do not form one coherent registry geometry."
        )
    derived_geometry = {
        "logical_layers": logical_layers,
        "physical_groups": physical_groups,
        "site_count": 1,
        "hidden_size": hidden_size,
        "query_group_width": query_group_width,
        "head_dim": head_dim,
        "rank": rank_value,
        "factor_dtype": str(factor_dtype),
    }
    for name, expected in derived_geometry.items():
        if name in registry_geometry and registry_geometry[name] != expected:
            raise QKVOFactorPackError(
                f"Declared registry geometry {name!r} does not match factors."
            )
    immutable = {
        "site_indices": {
            "dtype": str(site_indices.dtype),
            "shape": list(site_indices.shape),
            "values": site_indices.tolist(),
            "sha256": tensor_sha256(site_indices),
        },
        "_registry_schema_version": {
            "dtype": str(schema_version.dtype),
            "shape": [],
            "value": int(schema_version.item()),
            "sha256": tensor_sha256(schema_version),
        },
        "_registry_layout": {
            "dtype": str(layout.dtype),
            "shape": [],
            "value": int(layout.item()),
            "sha256": tensor_sha256(layout),
        },
    }
    return derived_geometry, immutable


def _state_from_pack_contract(
    factors: Mapping[str, Tensor],
    immutable: Mapping[str, Any],
) -> dict[str, Tensor]:
    try:
        site_spec = immutable["site_indices"]
        schema_spec = immutable["_registry_schema_version"]
        layout_spec = immutable["_registry_layout"]
        site_indices = torch.tensor(site_spec["values"], dtype=torch.int32)
        schema_version = torch.tensor(schema_spec["value"], dtype=torch.int32)
        layout = torch.tensor(layout_spec["value"], dtype=torch.int32)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise QKVOFactorPackError("Immutable registry state is malformed.") from error
    buffers = {
        "site_indices": site_indices,
        "_registry_schema_version": schema_version,
        "_registry_layout": layout,
    }
    for name, value in buffers.items():
        spec = immutable.get(name)
        if not isinstance(spec, Mapping) or (
            spec.get("dtype") != str(value.dtype)
            or spec.get("shape") != list(value.shape)
            or spec.get("sha256") != tensor_sha256(value)
        ):
            raise QKVOFactorPackError(
                f"Immutable registry tensor {name!r} failed verification."
            )
    return {
        **{name: factors[name] for name in QKVO_FACTOR_KEYS},
        **buffers,
    }


def _factor_inventory(factors: Mapping[str, Tensor]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.numel()),
            "tensor_bytes": int(value.numel() * value.element_size()),
            "sha256": tensor_sha256(value),
        }
        for name, value in sorted(factors.items())
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                json.dumps(
                    payload,
                    allow_nan=False,
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                ).encode("utf-8")
            )
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_save_safetensors(
    path: Path,
    factors: Mapping[str, Tensor],
    metadata: Mapping[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".safetensors.tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        save_file(dict(factors), temporary, metadata=dict(metadata))
        # Windows requires a writable descriptor for ``fsync`` even though the
        # bytes were already produced by safetensors.
        with open(temporary, "r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _reserve_output_pair(factor_path: Path, manifest_path: Path) -> None:
    """Atomically reserve both final names against concurrent writers."""

    if factor_path.parent != manifest_path.parent:
        raise QKVOFactorPackError(
            "Factor and manifest must share one parent for atomic reservation."
        )
    factor_path.parent.mkdir(parents=True, exist_ok=True)
    reserved: list[Path] = []
    try:
        for path in (factor_path, manifest_path):
            descriptor = os.open(
                path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            os.close(descriptor)
            reserved.append(path)
    except FileExistsError as error:
        for path in reversed(reserved):
            path.unlink(missing_ok=True)
        raise QKVOFactorPackError(
            "Factor pack outputs already exist; overwrite is prohibited."
        ) from error
    except Exception:
        for path in reversed(reserved):
            path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class LoadedQKVOFactorPack:
    factor_path: Path
    manifest_path: Path
    factors: Mapping[str, Tensor]
    manifest: Mapping[str, Any]


def install_qkvo_factor_pack(
    adapter_registry: nn.Module,
    loaded: LoadedQKVOFactorPack,
) -> str:
    """Install eight loaded factors transactionally into a fresh registry.

    Non-factor topology/schema buffers must already come from the same fresh
    adapter construction.  The complete resulting state, including those
    buffers, must reproduce the selected adapter state hash in the manifest.
    """

    if not isinstance(adapter_registry, nn.Module) or not isinstance(
        loaded, LoadedQKVOFactorPack
    ):
        raise QKVOFactorPackError(
            "A registry module and verified loaded factor pack are required."
        )
    state = adapter_registry.state_dict()
    if set(state) != set(QKVO_REGISTRY_STATE_KEYS):
        raise QKVOFactorPackError(
            "Fresh adapter registry state keys do not match the exact pack schema."
        )
    geometry = loaded.manifest.get("registry_geometry")
    immutable = loaded.manifest.get("immutable_registry_state")
    site = loaded.manifest.get("site")
    if (
        not isinstance(geometry, Mapping)
        or not isinstance(immutable, Mapping)
        or not isinstance(site, Mapping)
    ):
        raise QKVOFactorPackError("Verified pack lacks registry geometry.")
    if set(loaded.factors) != set(QKVO_FACTOR_KEYS):
        raise QKVOFactorPackError("Loaded pack does not contain exactly eight factors.")
    expected_state = _state_from_pack_contract(loaded.factors, immutable)
    derived_geometry, derived_immutable = _canonical_registry_contract(
        expected_state,
        loaded.factors,
        site=site,
        rank=loaded.manifest.get("rank"),
        registry_geometry=geometry,
    )
    if derived_geometry != geometry or derived_immutable != immutable:
        raise QKVOFactorPackError("Loaded pack registry contract is not canonical.")
    if adapter_state_sha256(expected_state) != loaded.manifest.get(
        "selected_adapter_state_sha256"
    ):
        raise QKVOFactorPackError("Loaded pack selected state hash is invalid.")
    for attribute in (
        "logical_layers",
        "physical_groups",
        "hidden_size",
        "query_group_width",
        "head_dim",
        "rank",
        "site_count",
    ):
        if getattr(adapter_registry, attribute, None) != geometry.get(attribute):
            raise QKVOFactorPackError(
                f"Fresh adapter registry geometry {attribute!r} differs from pack."
            )
    for name in QKVO_REGISTRY_STATE_KEYS:
        destination = state[name]
        source = expected_state[name]
        if (
            tuple(destination.shape) != tuple(source.shape)
            or destination.dtype != source.dtype
        ):
            raise QKVOFactorPackError(
                f"Fresh adapter tensor {name!r} has incompatible shape/dtype."
            )
    for name in QKVO_REGISTRY_BUFFER_KEYS:
        if not torch.equal(state[name].detach().cpu(), expected_state[name]):
            raise QKVOFactorPackError(
                f"Fresh adapter immutable buffer {name!r} differs from pack."
            )
    original = {name: value.detach().clone() for name, value in state.items()}
    try:
        with torch.no_grad():
            for name in QKVO_FACTOR_KEYS:
                destination = state[name]
                source = loaded.factors[name]
                destination.copy_(source.to(device=destination.device))
        installed = {
            name: value.detach().cpu().clone()
            for name, value in sorted(adapter_registry.state_dict().items())
        }
        installed_sha = adapter_state_sha256(installed)
        expected_sha = loaded.manifest.get("selected_adapter_state_sha256")
        if installed_sha != expected_sha:
            raise QKVOFactorPackError(
                "Installed factors do not reconstruct the complete selected "
                "adapter state; fresh topology/schema provenance differs."
            )
        return installed_sha
    except Exception:
        with torch.no_grad():
            current = adapter_registry.state_dict()
            for name, value in original.items():
                current[name].copy_(value.to(device=current[name].device))
        raise


def write_qkvo_factor_pack(
    factor_path: Path,
    manifest_path: Path,
    adapter_state: Mapping[str, Tensor],
    *,
    selected_adapter_state_sha256: str,
    site: Mapping[str, Any],
    rank: int,
    registry_geometry: Mapping[str, Any],
    role_scale_provenance: Mapping[str, Any],
    provenance: Mapping[str, Any],
    expected_factor_parameter_count: int = 92_160,
) -> dict[str, Any]:
    """Atomically write the eight-factor payload and its canonical manifest."""

    factor_path = factor_path.resolve()
    manifest_path = manifest_path.resolve()
    if factor_path == manifest_path or factor_path.suffix != ".safetensors":
        raise QKVOFactorPackError(
            "Factor and manifest paths must differ; factor suffix must be .safetensors."
        )
    factors = factor_tensors_from_state(adapter_state)
    geometry, immutable_registry_state = _canonical_registry_contract(
        adapter_state,
        factors,
        site=site,
        rank=rank,
        registry_geometry=registry_geometry,
    )
    actual_state_hash = adapter_state_sha256(adapter_state)
    if not _is_sha256(selected_adapter_state_sha256) or actual_state_hash != (
        selected_adapter_state_sha256
    ):
        raise QKVOFactorPackError(
            "Adapter state does not match the selected state SHA-256."
        )
    inventory = _factor_inventory(factors)
    parameter_count = sum(value["numel"] for value in inventory.values())
    tensor_bytes = sum(value["tensor_bytes"] for value in inventory.values())
    if parameter_count != _exact_int(
        expected_factor_parameter_count,
        "expected_factor_parameter_count",
        positive=True,
    ):
        raise QKVOFactorPackError(
            "Factor parameter count differs from the exact selected adapter."
        )
    payload_sha = canonical_sha256(inventory)
    role_scale_payload, source_payload = _validated_provenance(
        role_scale_provenance, provenance
    )
    _reserve_output_pair(factor_path, manifest_path)
    try:
        _atomic_save_safetensors(
            factor_path,
            factors,
            {
                "schema": QKVO_FACTOR_PACK_SCHEMA,
                "selected_adapter_state_sha256": actual_state_hash,
                "factor_inventory_sha256": payload_sha,
                "rank": str(int(rank)),
            },
        )
        if factor_path.stat().st_size > MAX_QKVO_FACTOR_FILE_BYTES:
            raise QKVOFactorPackError("Factor file exceeds the hard byte cap.")
        pack_file_sha = file_sha256(factor_path)
        manifest: dict[str, Any] = {
            "schema": QKVO_FACTOR_MANIFEST_SCHEMA,
            "factor_file": factor_path.name,
            "factor_file_sha256": pack_file_sha,
            "factor_file_bytes": factor_path.stat().st_size,
            "factor_inventory": inventory,
            "factor_inventory_sha256": payload_sha,
            "factor_parameter_count": parameter_count,
            "factor_tensor_bytes": tensor_bytes,
            "selected_adapter_state_sha256": actual_state_hash,
            "site": dict(site),
            "rank": int(rank),
            "registry_geometry": geometry,
            "immutable_registry_state": immutable_registry_state,
            "immutable_registry_state_sha256": canonical_sha256(
                immutable_registry_state
            ),
            "role_scale_provenance": role_scale_payload,
            "provenance": source_payload,
            "factor_keys": list(QKVO_FACTOR_KEYS),
            "factor_tensor_count": len(QKVO_FACTOR_KEYS),
            "optimizer_tensor_count": 0,
            "parent_tensor_count": 0,
            "base_model_tensor_count": 0,
            "full_model_checkpoint": False,
            "factor_pack_only": True,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        _atomic_write_json(manifest_path, manifest)
        loaded = load_qkvo_factor_pack(factor_path, manifest_path)
        if loaded.manifest["manifest_sha256"] != manifest["manifest_sha256"]:
            raise QKVOFactorPackError("Written manifest failed exact reload.")
        return manifest
    except Exception:
        # Both paths were proven absent above, so cleanup cannot remove user
        # artifacts. Never leave an orphan factor when manifest creation or
        # post-write verification fails.
        for output in (manifest_path, factor_path):
            try:
                output.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def load_qkvo_factor_pack(
    factor_path: Path,
    manifest_path: Path,
) -> LoadedQKVOFactorPack:
    """Load and fail-closed verify a factor-only QKVO artifact."""

    factor_path = factor_path.resolve()
    manifest_path = manifest_path.resolve()
    if not factor_path.is_file() or not manifest_path.is_file():
        raise QKVOFactorPackError("Factor pack or manifest does not exist.")
    try:
        factor_bytes = factor_path.stat().st_size
        manifest_bytes = manifest_path.stat().st_size
    except OSError as error:
        raise QKVOFactorPackError("Factor pack size could not be read.") from error
    if (
        factor_bytes <= 0
        or factor_bytes > MAX_QKVO_FACTOR_FILE_BYTES
        or manifest_bytes <= 0
        or manifest_bytes > MAX_QKVO_MANIFEST_FILE_BYTES
    ):
        raise QKVOFactorPackError("Factor pack exceeds its hard byte contract.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QKVOFactorPackError("Manifest is not valid UTF-8 JSON.") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != (
        QKVO_FACTOR_MANIFEST_SCHEMA
    ):
        raise QKVOFactorPackError("Unknown QKVO factor manifest schema.")
    manifest_without_hash = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if manifest.get("manifest_sha256") != canonical_sha256(manifest_without_hash):
        raise QKVOFactorPackError("Manifest SHA-256 is invalid.")
    if (
        manifest.get("factor_file") != factor_path.name
        or manifest.get("factor_file_sha256") != file_sha256(factor_path)
        or manifest.get("factor_file_bytes") != factor_bytes
        or manifest.get("factor_keys") != list(QKVO_FACTOR_KEYS)
        or manifest.get("factor_tensor_count") != len(QKVO_FACTOR_KEYS)
        or manifest.get("optimizer_tensor_count") != 0
        or manifest.get("parent_tensor_count") != 0
        or manifest.get("base_model_tensor_count") != 0
        or manifest.get("full_model_checkpoint") is not False
        or manifest.get("factor_pack_only") is not True
    ):
        raise QKVOFactorPackError("Factor manifest content contract failed.")
    _validated_provenance(
        manifest.get("role_scale_provenance"),
        manifest.get("provenance"),
    )
    try:
        with safe_open(str(factor_path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
    except Exception as error:
        raise QKVOFactorPackError("Safetensors header could not be read.") from error
    if metadata != {
        "schema": QKVO_FACTOR_PACK_SCHEMA,
        "selected_adapter_state_sha256": manifest.get("selected_adapter_state_sha256"),
        "factor_inventory_sha256": manifest.get("factor_inventory_sha256"),
        "rank": str(manifest.get("rank")),
    }:
        raise QKVOFactorPackError("Safetensors metadata disagrees with manifest.")
    factors = load_file(str(factor_path), device="cpu")
    if set(factors) != set(QKVO_FACTOR_KEYS):
        raise QKVOFactorPackError("Safetensors payload is not exactly eight factors.")
    factors = {
        key: factors[key].detach().cpu().contiguous() for key in QKVO_FACTOR_KEYS
    }
    inventory = _factor_inventory(factors)
    if (
        manifest.get("factor_inventory") != inventory
        or manifest.get("factor_inventory_sha256") != canonical_sha256(inventory)
        or manifest.get("factor_parameter_count")
        != sum(value["numel"] for value in inventory.values())
        or manifest.get("factor_tensor_bytes")
        != sum(value["tensor_bytes"] for value in inventory.values())
    ):
        raise QKVOFactorPackError("Factor tensor inventory verification failed.")
    immutable = manifest.get("immutable_registry_state")
    if not isinstance(immutable, Mapping) or manifest.get(
        "immutable_registry_state_sha256"
    ) != canonical_sha256(immutable):
        raise QKVOFactorPackError("Immutable registry state hash is invalid.")
    reconstructed_state = _state_from_pack_contract(factors, immutable)
    geometry = manifest.get("registry_geometry")
    site = manifest.get("site")
    if not isinstance(geometry, Mapping) or not isinstance(site, Mapping):
        raise QKVOFactorPackError("Registry geometry/site manifest is missing.")
    derived_geometry, derived_immutable = _canonical_registry_contract(
        reconstructed_state,
        factors,
        site=site,
        rank=manifest.get("rank"),
        registry_geometry=geometry,
    )
    if geometry != derived_geometry or immutable != derived_immutable:
        raise QKVOFactorPackError(
            "Registry geometry or immutable buffers are not canonical."
        )
    if adapter_state_sha256(reconstructed_state) != manifest.get(
        "selected_adapter_state_sha256"
    ):
        raise QKVOFactorPackError(
            "Factor pack cannot reconstruct the selected full registry state hash."
        )
    return LoadedQKVOFactorPack(
        factor_path=factor_path,
        manifest_path=manifest_path,
        factors=factors,
        manifest=manifest,
    )


__all__ = [
    "MAX_QKVO_FACTOR_FILE_BYTES",
    "MAX_QKVO_MANIFEST_FILE_BYTES",
    "QKVO_FACTOR_KEYS",
    "QKVO_FACTOR_MANIFEST_SCHEMA",
    "QKVO_FACTOR_PACK_SCHEMA",
    "QKVO_ROLE_SCALE_PROVENANCE_SCHEMA",
    "QKVO_SOURCE_PROVENANCE_SCHEMA",
    "QKVO_REGISTRY_BUFFER_KEYS",
    "QKVO_REGISTRY_STATE_KEYS",
    "LoadedQKVOFactorPack",
    "QKVOFactorPackError",
    "adapter_state_sha256",
    "canonical_sha256",
    "factor_tensors_from_state",
    "file_sha256",
    "install_qkvo_factor_pack",
    "load_qkvo_factor_pack",
    "tensor_sha256",
    "write_qkvo_factor_pack",
]
