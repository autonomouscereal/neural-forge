#!/usr/bin/env python3
"""Manifest-driven R12 launch for the child-only exact-4x V5 runtime.

This is intentionally a launch adapter, not another folding algorithm.  It
authenticates the already-published R12 child, V3 evidence, balanced basis,
and strict tokenizer bridge; derives the authoritative basis split; then
hands all nine phases to the existing V5 live runtime.  It has no parent-model
loader and no model-hub or network path.

The HMAC secret is supplied separately from the canonical launch manifest.
Only runtime authority objects see those bytes; neither the manifest, logs,
nor the final receipt serialize them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import torch

from experiments import run_lfm2_exact4x_moving_gauge_v5 as v5
from experiments import run_lfm2_exact4x_moving_gauge_v5_live as v5_live
from neural_forge.dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from neural_forge.dynamic_fold_phase_cache import DynamicFoldPhaseCache
from neural_forge.hf_lfm2_balanced_basis_phase_cache import (
    PhaseAnchorAuthority,
    load_existing_lfm2_circuit_balanced_basis_phase,
)
from neural_forge.hf_lfm2_exact4x_causal_diagnostic import (
    authenticate_v3_first_divergence_ledger,
)
from neural_forge.hf_lfm2_exact4x_first_divergence import (
    authoritative_split_by_batch,
    authenticate_lfm2_exact4x_first_divergence_request,
)
from neural_forge.hf_lfm2_direct_local import (
    BRIDGE_REFERENCE_FILENAME,
    SCHEMA as DIRECT_LOCAL_BRIDGE_REFERENCE_SCHEMA,
)
from neural_forge.hf_lfm2_circuit_trace_cache import open_lfm2_circuit_trace_cache
from neural_forge.hf_lfm2_moving_gauge_boundary_phase_cache import (
    Lfm2MovingGaugeBoundaryPhaseAuthority,
)
from neural_forge.hf_lfm2_moving_gauge_conditioned_recapture_phase_cache import (
    Lfm2ConditionedRecapturePhaseAuthority,
)
from neural_forge.hf_lfm2_moving_gauge_feature_phase_cache import (
    Lfm2MovingGaugeFeaturePhaseAuthority,
)
from neural_forge.hf_lfm2_tokenizer_bridge_trace_adapter import (
    Lfm2TargetTokenizedRecord,
    build_lfm2_strict_tokenizer_bridge_trace_adapter,
)
from neural_forge.hf_tokenizer_bridge import (
    load_strict_tokenizer_bridge_artifact,
    strict_tokenizer_bridge_target_package_fingerprint,
)
from neural_forge.lfm2_carrier_child_package import (
    load_authenticated_lfm2_carrier_child_package,
)


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

LAUNCH_SCHEMA = "neural-forge-lfm2-exact4x-moving-gauge-v5-r12-launch-v1"
RECEIPT_SCHEMA = "neural-forge-lfm2-exact4x-moving-gauge-v5-r12-receipt-v1"
TARGET_CALIBRATION_SCHEMA = "neural-forge-lfm2-target-tokenized-trajectory-v1"
CATALOG_BRIDGE_REFERENCE_SCHEMA = "neural-forge-catalog-tokenizer-bridge-reference-v1"
AUTHORITY_ROLES = (
    "basis_phase",
    "boundary_phase",
    "feature_phase",
    "conditioned_recapture",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*\Z")


class Lfm2Exact4xMovingGaugeV5R12LaunchError(RuntimeError):
    """The canonical R12 launch or one of its local authorities is invalid."""


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            f"{name} must be one lowercase SHA-256 digest."
        )
    return value


def _identifier(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 160
        or _IDENTIFIER.fullmatch(value) is None
    ):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            f"{name} must be one bounded identifier."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            f"{name} must be a positive integer."
        )
    return value


def _absolute_local_path(value: Any, name: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or "://" in str(value):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            f"{name} must be an ordinary local path."
        )
    path = Path(value).expanduser().absolute()
    if not path.is_absolute():  # pragma: no cover - ``absolute`` is defensive.
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(f"{name} must be absolute.")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_canonical_json(path: Path, name: str) -> dict[str, Any] | list[Any]:
    source = _absolute_local_path(path, name)
    try:
        metadata = source.lstat()
        raw = source.read_bytes()
        after = source.lstat()
    except OSError as exc:
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            f"{name} is missing or unreadable."
        ) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or int(metadata.st_nlink) != 1
        or (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            f"{name} must be one stable ordinary local file."
        )
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            f"{name} must be strict ASCII JSON."
        ) from exc
    if (
        not isinstance(value, (dict, list))
        or raw != canonical_json_bytes(value) + b"\n"
    ):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            f"{name} must use canonical JSON encoding."
        )
    return value


def _read_stable_json_object(
    path: Path,
    name: str,
) -> tuple[dict[str, Any], str, int]:
    """Read one inventory-bound JSON object without imposing its formatting."""

    source = _absolute_local_path(path, name)
    try:
        before = source.lstat()
        raw = source.read_bytes()
        after = source.lstat()
    except OSError as exc:
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            f"{name} is missing or unreadable."
        ) from exc
    identity = lambda value: (  # noqa: E731 - compact TOCTOU identity tuple.
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
        or identity(before) != identity(after)
    ):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            f"{name} must be one stable ordinary local file."
        )
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            f"{name} must be strict ASCII JSON."
        ) from exc
    if not isinstance(value, dict):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(f"{name} must be a JSON object.")
    return value, hashlib.sha256(raw).hexdigest(), len(raw)


def _read_authority_key(path: Path) -> bytes:
    """Read one private ordinary key without exposing its path or bytes."""

    source = _absolute_local_path(path, "authority key")
    try:
        before = source.lstat()
    except OSError as exc:
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Authority key is unavailable."
        ) from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
        or not 32 <= int(before.st_size) <= 4096
        or (os.name != "nt" and bool(stat.S_IMODE(before.st_mode) & 0o077))
    ):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Authority key must be one private ordinary 32-4096 byte file."
        )
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor: int | None = None
    try:
        descriptor = os.open(source, flags)
        opened = os.fstat(descriptor)
        secret = os.read(descriptor, 4097)
        after = os.fstat(descriptor)
        path_after = source.lstat()
    except OSError as exc:
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Authority key could not be read safely."
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    identity = lambda value: (  # noqa: E731 - compact TOCTOU identity tuple.
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )
    if (
        identity(before) != identity(opened)
        or identity(opened) != identity(after)
        or identity(before) != identity(path_after)
        or not 32 <= len(secret) <= 4096
    ):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Authority key changed while it was read."
        )
    return secret


def seal_r12_v5_launch_manifest(body: Mapping[str, Any]) -> dict[str, Any]:
    """Seal a complete launch body; the HMAC key path is intentionally absent."""

    row = json.loads(canonical_json_bytes(dict(body)))
    if not isinstance(row, dict) or "launch_manifest_sha256" in row:
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Launch body must be a JSON mapping without a preexisting seal."
        )
    return {**row, "launch_manifest_sha256": canonical_sha256(row)}


def _authenticate_launch_manifest(value: Mapping[str, Any]) -> Mapping[str, Any]:
    row = json.loads(canonical_json_bytes(dict(value)))
    required = {
        "schema",
        "local_paths",
        "tokenizer_bridge_root",
        "target_calibration",
        "lineage_sha256s",
        "provenance_sha256s",
        "authority_key_ids",
        "replay",
        "storage_dtypes",
        "receipt_path",
        "launch_manifest_sha256",
    }
    if set(row) != required or row.get("schema") != LAUNCH_SCHEMA:
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "R12 launch manifest fields or schema changed."
        )
    body = dict(row)
    claimed = body.pop("launch_manifest_sha256", None)
    if _digest(claimed, "launch_manifest_sha256") != canonical_sha256(body):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "R12 launch manifest seal is invalid."
        )
    paths = dict(v5.normalize_v5_local_paths(row["local_paths"]))
    bridge_root_supplied = _absolute_local_path(
        row["tokenizer_bridge_root"], "tokenizer bridge root"
    )
    try:
        bridge_root = bridge_root_supplied.resolve(strict=True)
    except OSError as exc:
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Tokenizer bridge root is missing or unreadable."
        ) from exc
    if bridge_root_supplied.is_symlink() or not bridge_root.is_dir():
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Tokenizer bridge root must be one real local directory."
        )
    for name in v5.MUTABLE_OUTPUT_PATH_KEYS:
        mutable = Path(paths[name]).resolve(strict=False)
        if mutable == bridge_root or bridge_root in mutable.parents:
            raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
                f"Mutable root {name!r} would alter the tokenizer bridge input."
            )
    lineage = dict(row["lineage_sha256s"])
    provenance = dict(row["provenance_sha256s"])
    if set(lineage) != v5.REQUIRED_LINEAGE_KEYS:
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Launch lineage inventory is incomplete or unknown."
        )
    if set(provenance) != v5_live.REQUIRED_BOUNDARY_PROVENANCE_KEYS:
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Launch boundary provenance inventory is incomplete or unknown."
        )
    lineage = {name: _digest(value, name) for name, value in lineage.items()}
    provenance = {name: _digest(value, name) for name, value in provenance.items()}
    authority_ids = dict(row["authority_key_ids"])
    if set(authority_ids) != set(AUTHORITY_ROLES):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Authority key-ID inventory is incomplete or unknown."
        )
    authority_ids = {
        name: _identifier(value, f"{name} key ID")
        for name, value in authority_ids.items()
    }
    target = dict(row["target_calibration"])
    if set(target) != {"path", "file_sha256"}:
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Target calibration binding is malformed."
        )
    target_path = _absolute_local_path(target["path"], "target calibration")
    target_sha256 = _digest(target["file_sha256"], "target calibration SHA-256")
    replay = dict(row["replay"])
    if set(replay) != {
        "devices",
        "runtime_device",
        "runtime_dtype",
        "recapture_checkpoint_batch_count",
        "boundary_shard_batch_count",
        "feature_shard_batch_count",
    }:
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError("Replay binding is malformed.")
    devices = tuple(replay["devices"])
    if (
        not devices
        or len(set(devices)) != len(devices)
        or any(not isinstance(value, str) or not value for value in devices)
    ):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Replay devices must be a nonempty unique string sequence."
        )
    runtime_device = replay["runtime_device"]
    runtime_dtype = replay["runtime_dtype"]
    if (
        not isinstance(runtime_device, str)
        or not runtime_device
        or runtime_dtype not in {"bfloat16", "float32"}
    ):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Runtime device/dtype binding is invalid."
        )
    storage = dict(row["storage_dtypes"])
    if set(storage) != {"parent", "child", "child_features"}:
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Storage dtype binding is malformed."
        )
    child_features = dict(storage["child_features"])
    if (
        storage["parent"] != "bfloat16"
        or storage["child"] != "float32"
        or child_features
        != {"short_conv.context": "float32", "mlp.post_swiglu": "float32"}
    ):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "R12 V5 requires BF16 parent cache evidence and FP32 child evidence."
        )
    receipt_path = _absolute_local_path(row["receipt_path"], "receipt path")
    output_root = Path(paths["output_root"])
    if receipt_path.parent != output_root:
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "The immutable launch receipt must be a direct child of output_root."
        )
    normalized = {
        **body,
        "local_paths": paths,
        "tokenizer_bridge_root": str(bridge_root),
        "target_calibration": {
            "path": str(target_path),
            "file_sha256": target_sha256,
        },
        "lineage_sha256s": lineage,
        "provenance_sha256s": provenance,
        "authority_key_ids": authority_ids,
        "replay": {
            **replay,
            "devices": list(devices),
            "recapture_checkpoint_batch_count": _positive_int(
                replay["recapture_checkpoint_batch_count"],
                "recapture checkpoint batch count",
            ),
            "boundary_shard_batch_count": _positive_int(
                replay["boundary_shard_batch_count"],
                "boundary shard batch count",
            ),
            "feature_shard_batch_count": _positive_int(
                replay["feature_shard_batch_count"],
                "feature shard batch count",
            ),
        },
        "storage_dtypes": {**storage, "child_features": child_features},
        "receipt_path": str(receipt_path),
    }
    if canonical_sha256(normalized) != claimed:
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Launch path or scalar normalization changed its sealed identity."
        )
    return MappingProxyType({**normalized, "launch_manifest_sha256": claimed})


def read_r12_v5_launch_manifest(path: Path) -> Mapping[str, Any]:
    value = _read_canonical_json(path, "R12 V5 launch manifest")
    if not isinstance(value, dict):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "R12 V5 launch manifest must be an object."
        )
    return _authenticate_launch_manifest(value)


def _target_records(
    path: Path,
    *,
    expected_file_sha256: str,
) -> tuple[Lfm2TargetTokenizedRecord, ...]:
    value = _read_canonical_json(path, "target-tokenized calibration")
    if not isinstance(value, list) or len(value) != v5.EXPECTED_BATCH_COUNT:
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Target calibration must contain exactly 256 canonical rows."
        )
    if _file_sha256(path) != expected_file_sha256:
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Target calibration differs from its launch-manifest file seal."
        )
    result: list[Lfm2TargetTokenizedRecord] = []
    required = {
        "schema",
        "id",
        "split",
        "target_tokenization_mode",
        "target_forced_decode_tokenization_mode",
        "target_input_texts",
        "target_forced_decode_texts",
        "input_ids",
        "attention_mask",
        "forced_decode_ids",
    }
    for ordinal, raw in enumerate(value):
        if not isinstance(raw, dict) or set(raw) != required:
            raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
                f"Target calibration row {ordinal} has a stale field inventory."
            )
        if (
            raw.get("schema") != TARGET_CALIBRATION_SCHEMA
            or raw.get("target_tokenization_mode")
            != "tokenizer-call-add-special-tokens"
            or raw.get("target_forced_decode_tokenization_mode")
            != "tokenizer-call-no-special-tokens"
            or raw.get("split") not in {"selection", "fit", "validation"}
        ):
            raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
                f"Target calibration row {ordinal} violates its sealed mode."
            )
        try:
            input_ids = torch.tensor(raw["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(raw["attention_mask"], dtype=torch.long)
            forced_decode_ids = torch.tensor(raw["forced_decode_ids"], dtype=torch.long)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
                f"Target calibration row {ordinal} has invalid integer tensors."
            ) from exc
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if forced_decode_ids.ndim == 1:
            forced_decode_ids = forced_decode_ids.unsqueeze(0)
        try:
            result.append(
                Lfm2TargetTokenizedRecord(
                    record_id=f"batch-{ordinal:06d}",
                    split=str(raw["split"]),
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    forced_decode_ids=forced_decode_ids,
                )
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
                f"Target calibration row {ordinal} cannot form a bridge record."
            ) from exc
    return tuple(result)


def _authenticated_parent_input_sha256s(
    parent_cache: Any,
    *,
    provenance_sha256s: Mapping[str, str],
    expected_cache_key: str,
) -> tuple[str, ...]:
    manifest = getattr(parent_cache, "manifest", None)
    completion = getattr(parent_cache, "_completion", None)
    corpus = manifest.get("corpus") if isinstance(manifest, Mapping) else None
    parent = manifest.get("parent") if isinstance(manifest, Mapping) else None
    topology = manifest.get("layer_topology") if isinstance(manifest, Mapping) else None
    hashes = (
        corpus.get("ordered_batch_input_sha256")
        if isinstance(corpus, Mapping)
        else None
    )
    if (
        getattr(parent_cache, "cache_key", None) != expected_cache_key
        or not isinstance(manifest, Mapping)
        or manifest.get("manifest_sha256") != provenance_sha256s["parent_manifest"]
        or not isinstance(completion, Mapping)
        or completion.get("completion_sha256")
        != provenance_sha256s["parent_completion"]
        or not isinstance(parent, Mapping)
        or parent.get("identity_sha256") != provenance_sha256s["parent_identity"]
        or not isinstance(topology, Mapping)
        or topology.get("layer_topology_sha256")
        != provenance_sha256s["parent_topology"]
        or not isinstance(corpus, Mapping)
        or corpus.get("batch_count") != v5.EXPECTED_BATCH_COUNT
        or not isinstance(hashes, list)
        or len(hashes) != v5.EXPECTED_BATCH_COUNT
    ):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Metadata-only parent trace cache differs from launch provenance."
        )
    return tuple(
        _digest(value, f"parent input SHA-256 {index}")
        for index, value in enumerate(hashes)
    )


def derive_authoritative_basis_split(
    basis_phase: Any,
    parent_cache: Any,
    *,
    provenance_sha256s: Mapping[str, str],
    expected_cache_key: str,
    split_deriver: Callable[[Mapping[str, Any], Sequence[Any]], Mapping[int, str]] = (
        authoritative_split_by_batch
    ),
) -> Mapping[int, str]:
    """Bind basis horizons to exact parent input hashes, never adapter IDs."""

    input_sha256s = _authenticated_parent_input_sha256s(
        parent_cache,
        provenance_sha256s=provenance_sha256s,
        expected_cache_key=expected_cache_key,
    )
    report = getattr(basis_phase, "report", None)
    if not isinstance(report, Mapping):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Authenticated basis phase exposes no canonical report."
        )
    references = tuple(SimpleNamespace(input_sha256=value) for value in input_sha256s)
    try:
        split = dict(split_deriver(report, references))
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Basis split does not bind the exact parent trace input identities."
        ) from exc
    values = tuple(split.values())
    if set(split) != set(range(v5.EXPECTED_BATCH_COUNT)) or (
        values.count("fit") != v5.FIT_BATCH_COUNT
        or values.count("heldout") != v5.HELDOUT_BATCH_COUNT
    ):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Authenticated basis split is not the exact 192 FIT / 64 heldout split."
        )
    return MappingProxyType(split)


def authenticate_packaged_tokenizer_bridge_reference(
    package: Any,
    bridge: Any,
) -> Mapping[str, Any]:
    """Cross-bind the strict bridge through the child's sealed file inventory.

    ``source_bridge_artifact_sha256`` in the carrier-package receipt identifies
    the continuous carrier-route source artifact.  It is intentionally not the
    strict tokenizer bridge.  The strict bridge is instead named by the
    packaged reference file, whose byte seal is already part of the outer child
    inventory.  Keeping those two identities separate avoids conflating token
    transport with carrier-route transport.
    """

    root = getattr(package, "root", None)
    receipt = getattr(package, "receipt", None)
    if not isinstance(root, (str, os.PathLike, Path)) or not isinstance(
        receipt, Mapping
    ):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Authenticated child package lacks its root or sealed receipt."
        )
    reference, file_sha256, size_bytes = _read_stable_json_object(
        Path(root) / BRIDGE_REFERENCE_FILENAME,
        "packaged tokenizer bridge reference",
    )
    common_fields = {
        "bridge_artifact_sha256",
        "bridge_profile_sha256",
        "schema",
        "strict_bridge_manifest_is_external_provenance",
        "target_tokenizer_package_fingerprint",
        "target_tokenizer_semantics_sha256",
    }
    direct_local_fields = common_fields | {"resource_fingerprint"}
    base_records = receipt.get("base_root_file_records")
    inventory_rows = (
        []
        if not isinstance(base_records, list)
        else [
            row
            for row in base_records
            if isinstance(row, Mapping)
            and row.get("relative_path") == BRIDGE_REFERENCE_FILENAME
        ]
    )
    try:
        target_package_fingerprint = strict_tokenizer_bridge_target_package_fingerprint(
            bridge
        )
        bridge_artifact_sha256 = bridge.artifact_sha256
        bridge_profile_sha256 = bridge.profile_fingerprint
        target_semantics_sha256 = bridge.target_tokenizer_semantics_sha256
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Strict tokenizer bridge exposes no complete target identity."
        ) from exc
    reference_schema = reference.get("schema")
    if (
        (
            reference_schema == CATALOG_BRIDGE_REFERENCE_SCHEMA
            and set(reference) != common_fields
        )
        or (
            reference_schema == DIRECT_LOCAL_BRIDGE_REFERENCE_SCHEMA
            and set(reference) != direct_local_fields
        )
        or reference_schema
        not in {
            CATALOG_BRIDGE_REFERENCE_SCHEMA,
            DIRECT_LOCAL_BRIDGE_REFERENCE_SCHEMA,
        }
        or reference.get("strict_bridge_manifest_is_external_provenance") is not True
        or len(inventory_rows) != 1
        or dict(inventory_rows[0])
        != {
            "relative_path": BRIDGE_REFERENCE_FILENAME,
            "size_bytes": size_bytes,
            "sha256": file_sha256,
        }
        or reference.get("bridge_artifact_sha256") != bridge_artifact_sha256
        or reference.get("bridge_profile_sha256") != bridge_profile_sha256
        or reference.get("target_tokenizer_semantics_sha256") != target_semantics_sha256
        or reference.get("target_tokenizer_package_fingerprint")
        != target_package_fingerprint
    ):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Packaged tokenizer bridge reference differs from the authenticated "
            "child inventory or external strict bridge."
        )
    if reference_schema == DIRECT_LOCAL_BRIDGE_REFERENCE_SCHEMA:
        _digest(
            reference.get("resource_fingerprint"),
            "packaged bridge resource fingerprint",
        )
    return MappingProxyType(reference)


@dataclass(frozen=True, slots=True)
class R12V5LaunchDependencies:
    """Test seam; production defaults are all existing local authenticated APIs."""

    verify_runtime_surface: Callable[[], Mapping[str, Any]] = (
        v5.verify_v5_runtime_surface
    )
    authenticate_local_preflight: Callable[[Any], Mapping[str, Any]] = (
        v5.authenticate_v5_local_preflight
    )
    load_package: Callable[..., Any] = load_authenticated_lfm2_carrier_child_package
    open_trace_cache: Callable[..., Any] = open_lfm2_circuit_trace_cache
    reconstruct_counted_plan: Callable[..., Any] = (
        v5.reconstruct_exact4x_counted_plan_from_package
    )
    authenticate_v3_request: Callable[[Mapping[str, Any]], Mapping[str, Any]] = (
        authenticate_lfm2_exact4x_first_divergence_request
    )
    authenticate_v3_ledger: Callable[..., Mapping[str, Any]] = (
        authenticate_v3_first_divergence_ledger
    )
    load_basis_phase: Callable[..., Any] = (
        load_existing_lfm2_circuit_balanced_basis_phase
    )
    derive_basis_split: Callable[..., Mapping[int, str]] = (
        derive_authoritative_basis_split
    )
    load_bridge: Callable[..., Any] = load_strict_tokenizer_bridge_artifact
    authenticate_packaged_bridge_reference: Callable[..., Mapping[str, Any]] = (
        authenticate_packaged_tokenizer_bridge_reference
    )
    build_trace_adapter: Callable[..., Any] = (
        build_lfm2_strict_tokenizer_bridge_trace_adapter
    )
    basis_authority_factory: Callable[..., Any] = PhaseAnchorAuthority
    boundary_authority_factory: Callable[..., Any] = (
        Lfm2MovingGaugeBoundaryPhaseAuthority
    )
    feature_authority_factory: Callable[..., Any] = Lfm2MovingGaugeFeaturePhaseAuthority
    conditioned_authority_factory: Callable[..., Any] = (
        Lfm2ConditionedRecapturePhaseAuthority
    )
    build_live_handlers: Callable[..., Mapping[str, Any]] = (
        v5_live.build_lfm2_exact4x_moving_gauge_v5_live_handlers
    )
    run_phase_machine: Callable[..., Any] = v5.run_v5_phase_machine


def _assert_lineage_cross_bindings(
    *,
    lineage: Mapping[str, str],
    provenance: Mapping[str, str],
    package: Any,
    counted_plan: Any,
    v3_request: Mapping[str, Any],
    v3_ledger: Mapping[str, Any],
    basis_phase: Any,
    bridge: Any,
    packaged_bridge_reference: Mapping[str, Any],
    adapter: Any,
) -> None:
    expected_pairs = {
        "r12_child_package": provenance["child_package"],
        "r12_child_identity": provenance["child_identity"],
        "r12_state_dict": provenance["child_state_dict"],
        "v3_request": provenance["v3_request"],
        "v3_ledger": provenance["v3_ledger"],
        "parent_trace_cache_completion": provenance["parent_completion"],
        "child_trace_cache_completion": provenance["child_completion"],
        "basis_phase_key": provenance["basis_phase_key"],
        "basis_phase_completion": provenance["basis_completion"],
        "tokenizer_bridge": provenance["bridge_artifact"],
    }
    if any(lineage[name] != value for name, value in expected_pairs.items()):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "V5 lineage and full boundary provenance describe different artifacts."
        )
    try:
        source_anchor = basis_phase.source_anchor["hmac_sha256"]
        basis_key = basis_phase.phase_key.phase_key_sha256
        basis_completion = basis_phase.completion_sha256
        basis_map = basis_phase.map_sha256
    except (AttributeError, KeyError, TypeError) as exc:
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Authenticated basis phase lacks required identity fields."
        ) from exc
    if (
        package.package_sha256 != lineage["r12_child_package"]
        or counted_plan.fingerprint != lineage["counted_child_state_plan"]
        or v3_request.get("request_sha256") != lineage["v3_request"]
        or v3_ledger.get("ledger_sha256") != lineage["v3_ledger"]
        or v3_ledger.get("batch_count") != v5.EXPECTED_BATCH_COUNT
        or v3_request.get("child_package_sha256") != package.package_sha256
        or v3_request.get("child_state_sha256") != provenance["child_identity"]
        or v3_request.get("route_package_sha256") != provenance["route_package"]
        or v3_request.get("route_runtime_sha256") != provenance["route_runtime"]
        or v3_request.get("portable_runtime_receipt_sha256")
        != provenance["portable_runtime_receipt"]
        or v3_request.get("basis_phase_key_sha256") != basis_key
        or v3_request.get("basis_completion_sha256") != basis_completion
        or v3_request.get("basis_map_sha256") != basis_map
        or basis_key != lineage["basis_phase_key"]
        or basis_completion != lineage["basis_phase_completion"]
        or basis_map != provenance["basis_map"]
        or source_anchor != lineage["basis_source_anchor"]
        or bridge.artifact_sha256 != provenance["bridge_artifact"]
        or bridge.profile_fingerprint != provenance["bridge_profile"]
        or packaged_bridge_reference.get("bridge_artifact_sha256")
        != bridge.artifact_sha256
        or packaged_bridge_reference.get("bridge_profile_sha256")
        != bridge.profile_fingerprint
        or adapter.evidence_sha256 != provenance["bridge_trace_adapter"]
        or adapter.bridge_artifact_sha256 != bridge.artifact_sha256
        or adapter.transport_profile_fingerprint != bridge.profile_fingerprint
    ):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Package, V3, basis, or tokenizer-bridge identities do not cross-bind."
        )
    records = tuple(getattr(adapter, "records", ()))
    if tuple(getattr(value, "record_id", None) for value in records) != tuple(
        f"batch-{index:06d}" for index in range(v5.EXPECTED_BATCH_COUNT)
    ):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Tokenizer bridge adapter does not preserve exact canonical batch IDs."
        )


def _write_receipt_noreplace(path: Path, value: Mapping[str, Any]) -> None:
    destination = _absolute_local_path(path, "receipt path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Receipt output root is link-like or not a directory."
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0))
    try:
        descriptor = os.open(destination, flags, 0o600)
    except FileExistsError as exc:
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Refusing to replace an existing R12 V5 launch receipt."
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(dict(value)) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def execute_r12_v5_launch(
    launch_manifest: Mapping[str, Any],
    *,
    authority_key_path: Path,
    dependencies: R12V5LaunchDependencies | None = None,
    logger: Callable[..., None] | None = None,
) -> Mapping[str, Any]:
    """Authenticate local R12 inputs and execute the existing nine V5 phases."""

    manifest = _authenticate_launch_manifest(launch_manifest)
    receipt_destination = Path(manifest["receipt_path"])
    if receipt_destination.exists() or receipt_destination.is_symlink():
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Refusing to execute over an existing R12 V5 launch receipt."
        )
    deps = dependencies or R12V5LaunchDependencies()
    # Resolve the complete API surface before loading any model or evidence.
    runtime_surface = deps.verify_runtime_surface()
    if not isinstance(runtime_surface, Mapping):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "V5 runtime-surface verification returned invalid evidence."
        )
    secret = _read_authority_key(authority_key_path)
    authority_ids = manifest["authority_key_ids"]
    basis_authority = deps.basis_authority_factory(
        key_id=authority_ids["basis_phase"], secret=secret
    )
    boundary_authority = deps.boundary_authority_factory(
        key_id=authority_ids["boundary_phase"], secret=secret
    )
    feature_authority = deps.feature_authority_factory(
        key_id=authority_ids["feature_phase"], secret=secret
    )
    conditioned_authority = deps.conditioned_authority_factory(
        key_id=authority_ids["conditioned_recapture"], secret=secret
    )
    # No serialized value below contains ``secret`` or the key path.

    local_paths = manifest["local_paths"]
    lineage = manifest["lineage_sha256s"]
    provenance = manifest["provenance_sha256s"]
    package = deps.load_package(
        Path(local_paths["published_child"]),
        expected_package_sha256=lineage["r12_child_package"],
    )
    counted_plan = deps.reconstruct_counted_plan(
        package,
        expected_package_sha256=lineage["r12_child_package"],
    )
    v3_root = Path(local_paths["v3_evidence"])
    raw_v3_request = _read_canonical_json(
        v3_root / "request.json", "V3 first-divergence request"
    )
    raw_v3_ledger = _read_canonical_json(
        v3_root / "first-divergence-ledger.json", "V3 first-divergence ledger"
    )
    if not isinstance(raw_v3_request, dict) or not isinstance(raw_v3_ledger, dict):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "V3 request and ledger must be canonical JSON objects."
        )
    v3_request = deps.authenticate_v3_request(raw_v3_request)
    v3_ledger = deps.authenticate_v3_ledger(
        raw_v3_ledger,
        expected_request=v3_request,
    )
    if (
        v3_request.get("request_sha256") != lineage["v3_request"]
        or v3_ledger.get("ledger_sha256") != lineage["v3_ledger"]
    ):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Authenticated V3 evidence differs from the launch lineage."
        )
    parent_cache = deps.open_trace_cache(
        local_paths["parent_trace_cache"], verify=False
    )
    basis_phase = deps.load_basis_phase(
        DynamicFoldPhaseCache(local_paths["basis_phase_cache"]),
        lineage["basis_phase_key"],
        anchor_authority=basis_authority,
        expected_parent_identity_sha256=provenance["parent_identity"],
        expected_trace_cache_key=v3_request["trace_cache_key"],
        expected_trace_selective_session_sha256=provenance["parent_selective_session"],
        expected_source_width=v5.SOURCE_WIDTH,
        expected_target_width=v5.TARGET_WIDTH,
        expected_map_sha256=provenance["basis_map"],
    )
    basis_split = deps.derive_basis_split(
        basis_phase,
        parent_cache,
        provenance_sha256s=provenance,
        expected_cache_key=v3_request["trace_cache_key"],
    )
    parent_input_sha256s = _authenticated_parent_input_sha256s(
        parent_cache,
        provenance_sha256s=provenance,
        expected_cache_key=v3_request["trace_cache_key"],
    )
    target = manifest["target_calibration"]
    target_records = _target_records(
        Path(target["path"]), expected_file_sha256=target["file_sha256"]
    )
    bridge = deps.load_bridge(
        Path(manifest["tokenizer_bridge_root"]),
        expected_artifact_sha256=provenance["bridge_artifact"],
        packaged=False,
    )
    packaged_bridge_reference = deps.authenticate_packaged_bridge_reference(
        package,
        bridge,
    )
    adapter = deps.build_trace_adapter(
        bridge,
        target_records,
        expected_artifact_sha256=provenance["bridge_artifact"],
    )
    verify_adapter = getattr(adapter, "verify", None)
    if not callable(verify_adapter) or verify_adapter() is not adapter:
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Rebuilt tokenizer bridge adapter did not authenticate."
        )
    _assert_lineage_cross_bindings(
        lineage=lineage,
        provenance=provenance,
        package=package,
        counted_plan=counted_plan,
        v3_request=v3_request,
        v3_ledger=v3_ledger,
        basis_phase=basis_phase,
        bridge=bridge,
        packaged_bridge_reference=packaged_bridge_reference,
        adapter=adapter,
    )
    replay = manifest["replay"]
    if (
        v3_request.get("runtime_device") != replay["runtime_device"]
        or v3_request.get("runtime_dtype") != replay["runtime_dtype"]
    ):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Launch runtime device/dtype differs from authenticated V3 evidence."
        )
    request = v5.Lfm2Exact4xMovingGaugeV5Request(
        local_paths=local_paths,
        lineage_sha256s=lineage,
        counted_child_state_plan=counted_plan,
        replay_devices=tuple(replay["devices"]),
        recapture_checkpoint_batch_count=replay["recapture_checkpoint_batch_count"],
    )
    local_preflight = deps.authenticate_local_preflight(request)
    handlers = deps.build_live_handlers(
        request,
        trace_adapter=adapter,
        basis_phase_authority=basis_authority,
        boundary_phase_authority=boundary_authority,
        feature_phase_authority=feature_authority,
        conditioned_recapture_authority=conditioned_authority,
        provenance_sha256s=provenance,
        v3_first_divergence_request=v3_request,
        v3_first_divergence_ledger=v3_ledger,
        basis_split_by_batch=basis_split,
        parent_storage_dtype=manifest["storage_dtypes"]["parent"],
        child_storage_dtype=manifest["storage_dtypes"]["child"],
        child_feature_storage_dtypes=manifest["storage_dtypes"]["child_features"],
        runtime_device=replay["runtime_device"],
        runtime_dtype=replay["runtime_dtype"],
        boundary_shard_batch_count=replay["boundary_shard_batch_count"],
        feature_shard_batch_count=replay["feature_shard_batch_count"],
    )
    if set(handlers) != set(v5.PHASE_ORDER):
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "Live runtime did not supply exactly all nine V5 handlers."
        )
    run_kwargs: dict[str, Any] = {}
    if logger is not None:
        run_kwargs["logger"] = logger
    outcomes, run_receipt = deps.run_phase_machine(request, handlers, **run_kwargs)
    if tuple(outcomes) != v5.PHASE_ORDER:
        raise Lfm2Exact4xMovingGaugeV5R12LaunchError(
            "V5 phase machine did not complete all nine phases in order."
        )
    split_rows = [
        {
            "batch_index": index,
            "source_input_sha256": parent_input_sha256s[index],
            "split": basis_split[index],
        }
        for index in range(v5.EXPECTED_BATCH_COUNT)
    ]
    receipt_body = {
        "schema": RECEIPT_SCHEMA,
        "launch_manifest_sha256": manifest["launch_manifest_sha256"],
        "v5_request_sha256": request.request_sha256,
        "runtime_surface_result_sha256": runtime_surface.get("result_sha256"),
        "local_preflight_result_sha256": local_preflight.get("result_sha256"),
        "phase_machine_result_sha256": run_receipt["result_sha256"],
        "phase_order": list(v5.PHASE_ORDER),
        "phase_artifact_sha256s": {
            name: outcomes[name].artifact_sha256 for name in v5.PHASE_ORDER
        },
        "r12_child_package_sha256": package.package_sha256,
        "counted_child_state_plan_sha256": counted_plan.fingerprint,
        "v3_request_sha256": v3_request["request_sha256"],
        "v3_ledger_sha256": v3_ledger["ledger_sha256"],
        "basis_phase_key_sha256": basis_phase.phase_key.phase_key_sha256,
        "basis_completion_sha256": basis_phase.completion_sha256,
        "basis_map_sha256": basis_phase.map_sha256,
        "basis_split_sha256": canonical_sha256(split_rows),
        "basis_split_counts": {
            "fit": v5.FIT_BATCH_COUNT,
            "heldout": v5.HELDOUT_BATCH_COUNT,
        },
        "trace_adapter_sha256": adapter.evidence_sha256,
        "trace_record_ids_sha256": canonical_sha256(
            [f"batch-{index:06d}" for index in range(v5.EXPECTED_BATCH_COUNT)]
        ),
        "target_calibration_file_sha256": target["file_sha256"],
        "all_nine_handlers_executed_via_v5_phase_machine": True,
        "parent_model_loaded": False,
        "parent_forward_calls": 0,
        "hugging_face_loader_used": False,
        "network_access": False,
        "authority_objects_serialized": False,
        "authority_key_path_serialized": False,
        "authority_key_bytes_serialized": False,
    }
    for field in ("runtime_surface_result_sha256", "local_preflight_result_sha256"):
        _digest(receipt_body[field], field)
    receipt = MappingProxyType(
        {**receipt_body, "receipt_sha256": canonical_sha256(receipt_body)}
    )
    _write_receipt_noreplace(Path(manifest["receipt_path"]), receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-manifest-json", type=Path, required=True)
    parser.add_argument("--authority-key-file", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = read_r12_v5_launch_manifest(args.launch_manifest_json)
    receipt = execute_r12_v5_launch(
        manifest,
        authority_key_path=args.authority_key_file,
    )
    sys.stdout.write(
        canonical_json_bytes(
            {
                "schema": RECEIPT_SCHEMA,
                "event": "launch_complete",
                "receipt_sha256": receipt["receipt_sha256"],
                "receipt_path": manifest["receipt_path"],
            }
        ).decode("ascii")
        + "\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through ``main``.
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_ROLES",
    "LAUNCH_SCHEMA",
    "RECEIPT_SCHEMA",
    "Lfm2Exact4xMovingGaugeV5R12LaunchError",
    "R12V5LaunchDependencies",
    "derive_authoritative_basis_split",
    "execute_r12_v5_launch",
    "read_r12_v5_launch_manifest",
    "seal_r12_v5_launch_manifest",
]
