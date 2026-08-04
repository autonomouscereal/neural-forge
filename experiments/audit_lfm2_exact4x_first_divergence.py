#!/usr/bin/env python3
"""Audit exact-4x LFM2 component divergence from sealed cached evidence only."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import hmac
import importlib.metadata
import json
import os
import platform
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import neural_forge.hf_lfm2_balanced_basis_phase_cache as basis_phase_module
import neural_forge.hf_lfm2_canonical as canonical_module
import neural_forge.hf_lfm2_circuit_trace_cache as trace_module
import neural_forge.hf_lfm2_direct_local as direct_local_module
import neural_forge.hf_lfm2_dynamic_blank_materializer as dynamic_blank_module
import neural_forge.hf_lfm2_exact4x_first_divergence as diagnostic_module
import neural_forge.hf_lfm2_semantic_step_token_fold as routing_module
import neural_forge.hf_lfm2_teacher_forced_trajectory_audit as teacher_audit_module
import neural_forge.lfm2_carrier_bridge_publication as carrier_module
import neural_forge.lfm2_carrier_child_package as child_package_module
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_phase_cache import DynamicFoldPhaseCache
from neural_forge.hf_lfm2_balanced_basis_phase_cache import (
    PhaseAnchorAuthority,
    load_existing_lfm2_circuit_balanced_basis_phase,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    capture_lfm2_child_embedding_replay,
    fingerprint_lfm2_parent,
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_direct_local import load_lfm2_direct_local_model
from neural_forge.hf_lfm2_exact4x_first_divergence import (
    LFM2_EXACT4X_FIRST_DIVERGENCE_IMPLEMENTATION_SCHEMA,
    Lfm2Exact4xFirstDivergenceError,
    aggregate_lfm2_exact4x_first_divergence,
    audit_lfm2_exact4x_first_divergence_batch,
    authenticate_lfm2_exact4x_first_divergence_batch,
    authenticate_lfm2_exact4x_first_divergence_request,
    authoritative_split_by_batch,
    build_lfm2_exact4x_first_divergence_request,
    build_lfm2_exact4x_parent_logit_domain_contract,
    parent_selection_logical_bytes,
    parent_tensor_names_for_first_divergence,
    phase_names,
    verify_complete_exact_budget_child,
)
from neural_forge.lfm2_carrier_bridge_publication import (
    fingerprint_lfm2_carrier_runtime_owned_tensor_state,
    relocate_and_bind_packaged_lfm2_carrier_routes,
)
from neural_forge.lfm2_carrier_child_package import (
    load_authenticated_lfm2_carrier_child_package,
)
from transformers.models.lfm2 import modeling_lfm2 as transformers_lfm2_module


RUNNER_SCHEMA = "neural-forge-lfm2-exact4x-first-divergence-runner-v3"
BATCH_CHECKPOINT_SCHEMA = (
    "neural-forge-lfm2-exact4x-first-divergence-batch-checkpoint-v3"
)
COMPLETION_SCHEMA = "neural-forge-lfm2-exact4x-first-divergence-completion-v4"
PROGRESS_SCHEMA = "neural-forge-lfm2-exact4x-first-divergence-progress-v3"
CHECKPOINT_HMAC_ALGORITHM = "hmac-sha256"
_CHECKPOINT_HMAC_DOMAIN = (
    b"neural-forge-lfm2-exact4x-first-divergence-batch-checkpoint-v3\0"
)


class Lfm2Exact4xFirstDivergenceRunnerError(RuntimeError):
    """The local diagnostic request or resumable artifact is invalid."""


class FirstDivergenceCheckpointAuthority:
    """Runtime-only HMAC authority for immutable diagnostic checkpoints."""

    __slots__ = ("__secret", "key_id")

    def __init__(self, *, key_id: str, secret: bytes) -> None:
        if not isinstance(key_id, str) or not key_id or len(key_id) > 256:
            raise Lfm2Exact4xFirstDivergenceRunnerError(
                "Checkpoint HMAC key ID must be a nonempty bounded string."
            )
        if not isinstance(secret, bytes) or len(secret) < 32:
            raise Lfm2Exact4xFirstDivergenceRunnerError(
                "Checkpoint HMAC secret must contain at least 32 runtime bytes."
            )
        self.key_id = key_id
        self.__secret = secret

    def sign(self, body: Mapping[str, Any]) -> str:
        return hmac.new(
            self.__secret,
            _CHECKPOINT_HMAC_DOMAIN + _canonical_bytes(dict(body)),
            hashlib.sha256,
        ).hexdigest()

    def verify(self, body: Mapping[str, Any], claimed: Any) -> bool:
        return (
            isinstance(claimed, str)
            and len(claimed) == 64
            and hmac.compare_digest(self.sign(body), claimed)
        )


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Runner evidence must be finite canonical JSON."
        ) from exc


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ordinary_directory(value: str | Path, name: str) -> Path:
    if "://" in str(value):
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            f"{name} must be an ordinary local path."
        )
    supplied = Path(value).expanduser().absolute()
    if supplied.is_symlink():
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            f"{name} must be a real local directory."
        )
    path = supplied.resolve(strict=True)
    if not path.is_dir():
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            f"{name} must be a real local directory."
        )
    return path


def _read_json_with_sha256(
    path: Path,
    name: str,
) -> tuple[dict[str, Any], str, int]:
    if (
        not path.is_file()
        or path.is_symlink()
        or getattr(path.stat(), "st_nlink", 1) != 1
    ):
        raise Lfm2Exact4xFirstDivergenceRunnerError(f"{name} is missing or link-like.")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            f"{name} is not strict ASCII JSON."
        ) from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value) + b"\n":
        raise Lfm2Exact4xFirstDivergenceRunnerError(f"{name} is not canonical JSON.")
    return value, hashlib.sha256(raw).hexdigest(), len(raw)


def _read_json(path: Path, name: str) -> dict[str, Any]:
    value, _sha256, _size_bytes = _read_json_with_sha256(path, name)
    return value


def _authenticated_parent_logit_descriptor_inventory(
    references: Sequence[Any],
) -> tuple[int, int, str]:
    """Bind one uniform physical logit capacity to sealed trace descriptors."""

    if not references:
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Parent logit-domain contract requires trace references."
        )
    physical_capacities: set[int] = set()
    inventory_rows: list[dict[str, Any]] = []
    phase_count = 0
    element_sizes = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}
    for expected_index, reference in enumerate(references):
        record = getattr(reference, "record", None)
        if getattr(reference, "index", None) != expected_index or not isinstance(
            record, Mapping
        ):
            raise Lfm2Exact4xFirstDivergenceRunnerError(
                "Trace logit references are not canonically ordered."
            )
        capture = record.get("capture")
        tensors = record.get("tensors")
        if not isinstance(capture, Mapping) or not isinstance(tensors, Mapping):
            raise Lfm2Exact4xFirstDivergenceRunnerError(
                "Trace logit reference metadata is incomplete."
            )
        decode_steps = capture.get("decode_steps")
        if (
            isinstance(decode_steps, bool)
            or not isinstance(decode_steps, int)
            or decode_steps < 0
        ):
            raise Lfm2Exact4xFirstDivergenceRunnerError(
                "Trace logit decode horizon is invalid."
            )
        phase_descriptors = []
        for phase in phase_names(decode_steps):
            name = f"{phase}.logits"
            descriptor = tensors.get(name)
            if not isinstance(descriptor, Mapping) or set(descriptor) != {
                "dtype",
                "numel",
                "sha256",
                "shape",
                "size_bytes",
            }:
                raise Lfm2Exact4xFirstDivergenceRunnerError(
                    f"Trace logit descriptor {name!r} is noncanonical."
                )
            dtype = descriptor.get("dtype")
            shape = descriptor.get("shape")
            if (
                dtype not in element_sizes
                or not isinstance(shape, list)
                or len(shape) != 3
                or any(
                    isinstance(size, bool) or not isinstance(size, int) or size <= 0
                    for size in shape
                )
                or shape[0] != 1
                or shape[2] <= 1
            ):
                raise Lfm2Exact4xFirstDivergenceRunnerError(
                    f"Trace logit descriptor {name!r} has invalid geometry."
                )
            numel = shape[0] * shape[1] * shape[2]
            if (
                descriptor.get("numel") != numel
                or descriptor.get("size_bytes") != numel * element_sizes[dtype]
            ):
                raise Lfm2Exact4xFirstDivergenceRunnerError(
                    f"Trace logit descriptor {name!r} has inconsistent storage."
                )
            _digest(descriptor.get("sha256"), f"trace logit {name} SHA-256")
            physical_capacities.add(shape[2])
            phase_descriptors.append({"phase": phase, "descriptor": dict(descriptor)})
            phase_count += 1
        inventory_rows.append(
            {
                "batch_index": expected_index,
                "input_sha256": _digest(
                    getattr(reference, "input_sha256", None),
                    "trace reference input SHA-256",
                ),
                "record_sha256": _digest(
                    getattr(reference, "record_sha256", None),
                    "trace reference record SHA-256",
                ),
                "phase_logit_descriptors": phase_descriptors,
            }
        )
    if len(physical_capacities) != 1:
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Trace phase logits do not share one physical row capacity."
        )
    inventory = {
        "schema": "neural-forge-lfm2-trace-logit-descriptor-inventory-v1",
        "rows": inventory_rows,
    }
    return next(iter(physical_capacities)), phase_count, canonical_sha256(inventory)


def _build_parent_logit_domain_contract(
    *,
    package: Any,
    routing: Any,
    references: Sequence[Any],
    trace_cache_key: str,
    trace_selective_session_sha256: str,
) -> dict[str, Any]:
    lexicon = getattr(package, "semantic_source_lexicon", None)
    tokenizer = getattr(lexicon, "tokenizer", None)
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Child package has no authenticated semantic-source tokenizer."
        )
    vocabulary = get_vocab()
    if not isinstance(vocabulary, Mapping) or not vocabulary:
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Semantic-source tokenizer vocabulary is invalid."
        )
    token_ids = list(vocabulary.values())
    defined_count = len(vocabulary)
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) for value in token_ids
        )
        or len(set(token_ids)) != defined_count
        or set(token_ids) != set(range(defined_count))
        or len(tokenizer) != defined_count
        or int(getattr(routing, "donor_row_count", 0)) != defined_count
    ):
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Semantic-source tokenizer IDs and donor routes are not one exact prefix."
        )
    bridge_path = Path(package.root) / direct_local_module.BRIDGE_REFERENCE_FILENAME
    bridge, bridge_file_sha256, bridge_file_size = _read_json_with_sha256(
        bridge_path,
        "packaged tokenizer bridge reference",
    )
    common_bridge_fields = {
        "bridge_artifact_sha256",
        "bridge_profile_sha256",
        "schema",
        "strict_bridge_manifest_is_external_provenance",
        "target_tokenizer_package_fingerprint",
        "target_tokenizer_semantics_sha256",
    }
    catalog_schema = "neural-forge-catalog-tokenizer-bridge-reference-v1"
    direct_local_fields = common_bridge_fields | {"resource_fingerprint"}
    bridge_schema = bridge.get("schema")
    if (
        (bridge_schema == catalog_schema and set(bridge) != common_bridge_fields)
        or (
            bridge_schema == direct_local_module.SCHEMA
            and set(bridge) != direct_local_fields
        )
        or bridge_schema not in {catalog_schema, direct_local_module.SCHEMA}
        or bridge.get("strict_bridge_manifest_is_external_provenance") is not True
    ):
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Packaged tokenizer bridge reference is noncanonical."
        )
    if bridge_schema == direct_local_module.SCHEMA:
        _digest(
            bridge["resource_fingerprint"],
            "packaged bridge resource fingerprint",
        )
    receipt = package.receipt
    base_records = receipt.get("base_root_file_records")
    bridge_records = (
        []
        if not isinstance(base_records, list)
        else [
            row
            for row in base_records
            if isinstance(row, Mapping)
            and row.get("relative_path")
            == direct_local_module.BRIDGE_REFERENCE_FILENAME
        ]
    )
    if len(bridge_records) != 1 or bridge_records[0] != {
        "relative_path": direct_local_module.BRIDGE_REFERENCE_FILENAME,
        "size_bytes": bridge_file_size,
        "sha256": bridge_file_sha256,
    }:
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Packaged tokenizer bridge reference differs from the authenticated "
            "child inventory."
        )
    physical_capacity, phase_count, descriptor_sha256 = (
        _authenticated_parent_logit_descriptor_inventory(references)
    )
    semantic_tokenizer_sha256 = _digest(
        receipt.get("semantic_source_tokenizer_sha256"),
        "semantic-source tokenizer SHA-256",
    )
    semantic_lexicon_package_sha256 = _digest(
        getattr(lexicon, "package_sha256", None),
        "semantic-source lexicon package SHA-256",
    )
    lexicon_receipt = getattr(lexicon, "receipt", None)
    if (
        receipt.get("semantic_source_lexicon_package_sha256")
        != semantic_lexicon_package_sha256
        or not isinstance(lexicon_receipt, Mapping)
        or lexicon_receipt.get("tokenizer_semantics_sha256")
        != semantic_tokenizer_sha256
        or lexicon_receipt.get("package_sha256") != semantic_lexicon_package_sha256
        or receipt.get("carrier_route_package_sha256") != package.routes.package_sha256
    ):
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Child package receipt, semantic lexicon, or route identities differ."
        )
    return build_lfm2_exact4x_parent_logit_domain_contract(
        child_package_sha256=package.package_sha256,
        route_package_sha256=package.routes.package_sha256,
        semantic_source_tokenizer_sha256=semantic_tokenizer_sha256,
        semantic_source_lexicon_package_sha256=(semantic_lexicon_package_sha256),
        packaged_bridge_reference_sha256=bridge_file_sha256,
        bridge_reference_schema=bridge["schema"],
        bridge_artifact_sha256=_digest(
            bridge.get("bridge_artifact_sha256"), "bridge artifact SHA-256"
        ),
        bridge_profile_sha256=_digest(
            bridge.get("bridge_profile_sha256"), "bridge profile SHA-256"
        ),
        bridge_target_tokenizer_semantics_sha256=_digest(
            bridge.get("target_tokenizer_semantics_sha256"),
            "bridge target-tokenizer semantics SHA-256",
        ),
        bridge_target_tokenizer_package_fingerprint=_digest(
            bridge.get("target_tokenizer_package_fingerprint"),
            "bridge target-tokenizer package fingerprint",
        ),
        trace_cache_key=trace_cache_key,
        trace_selective_session_sha256=trace_selective_session_sha256,
        trace_logit_descriptor_inventory_sha256=descriptor_sha256,
        trace_batch_count=len(references),
        trace_phase_count=phase_count,
        defined_semantic_row_count=defined_count,
        maximum_defined_token_id=defined_count - 1,
        route_donor_row_count=int(routing.donor_row_count),
        parent_physical_row_capacity=physical_capacity,
    )


def _atomic_write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            f"Refusing to overwrite immutable diagnostic artifact {path}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(dict(value)) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            raise Lfm2Exact4xFirstDivergenceRunnerError(
                f"Diagnostic artifact appeared during publication: {path}."
            )
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise Lfm2Exact4xFirstDivergenceRunnerError(
                f"Diagnostic artifact appeared during publication: {path}."
            ) from exc
        except OSError as exc:
            raise Lfm2Exact4xFirstDivergenceRunnerError(
                f"Could not publish immutable diagnostic artifact {path}."
            ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()


class _ProgressJournal:
    def __init__(self, path: Path, request_sha256: str) -> None:
        self.path = path
        self.request_sha256 = _digest(request_sha256, "request_sha256")
        self.sequence = 0
        self.previous = "0" * 64
        if path.exists():
            if (
                path.is_symlink()
                or not path.is_file()
                or getattr(path.stat(), "st_nlink", 1) != 1
            ):
                raise Lfm2Exact4xFirstDivergenceRunnerError(
                    "Progress journal is link-like or not a file."
                )
            payload = path.read_bytes()
            if payload and not payload.endswith(b"\n"):
                raise Lfm2Exact4xFirstDivergenceRunnerError(
                    "Progress journal has an unterminated final event."
                )
            for index, raw in enumerate(payload.splitlines()):
                try:
                    value = json.loads(raw.decode("ascii"))
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise Lfm2Exact4xFirstDivergenceRunnerError(
                        "Progress journal is malformed."
                    ) from exc
                body = dict(value)
                claimed = body.pop("event_sha256", None)
                if (
                    value.get("schema") != PROGRESS_SCHEMA
                    or value.get("request_sha256") != self.request_sha256
                    or value.get("sequence") != index
                    or value.get("previous_event_sha256") != self.previous
                    or claimed != canonical_sha256(body)
                    or raw != _canonical_bytes(value)
                ):
                    raise Lfm2Exact4xFirstDivergenceRunnerError(
                        "Progress journal hash chain does not verify."
                    )
                self.sequence = index + 1
                self.previous = str(claimed)

    def emit(self, event: str, **details: Any) -> None:
        body = {
            "schema": PROGRESS_SCHEMA,
            "request_sha256": self.request_sha256,
            "sequence": self.sequence,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z"),
            "event": event,
            "previous_event_sha256": self.previous,
            "details": json.loads(_canonical_bytes(details).decode("ascii")),
        }
        row = {**body, "event_sha256": canonical_sha256(body)}
        payload = _canonical_bytes(row) + b"\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        sys.stdout.write(payload.decode("ascii"))
        sys.stdout.flush()
        self.sequence += 1
        self.previous = row["event_sha256"]


def _is_reparse_point(metadata: os.stat_result) -> bool:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    marker = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return bool(marker and attributes & marker)


def _key_file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _read_raw_hmac_key(path: Path) -> bytes:
    source = Path(path).expanduser().absolute()
    try:
        for ancestor in reversed(source.parents):
            metadata = ancestor.lstat()
            if stat.S_ISLNK(metadata.st_mode) or _is_reparse_point(metadata):
                raise Lfm2Exact4xFirstDivergenceRunnerError(
                    "Balanced-basis HMAC key ancestors cannot be link-like."
                )
        before = source.lstat()
    except OSError as exc:
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Balanced-basis HMAC key is unavailable."
        ) from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or _is_reparse_point(before)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
        or int(before.st_size) < 32
        or int(before.st_size) > 4096
        or (os.name != "nt" and bool(stat.S_IMODE(before.st_mode) & 0o077))
    ):
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Balanced-basis HMAC key must be one private ordinary 32-4096 byte file."
        )
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor: int | None = None
    try:
        descriptor = os.open(source, flags)
        opened = os.fstat(descriptor)
        if (
            _key_file_identity(opened) != _key_file_identity(before)
            or not stat.S_ISREG(opened.st_mode)
            or int(opened.st_nlink) != 1
            or (os.name != "nt" and bool(stat.S_IMODE(opened.st_mode) & 0o077))
        ):
            raise Lfm2Exact4xFirstDivergenceRunnerError(
                "Balanced-basis HMAC key changed while being opened."
            )
        chunks = []
        remaining = 4097
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        secret = b"".join(chunks)
        after = os.fstat(descriptor)
        after_path = source.lstat()
    except Lfm2Exact4xFirstDivergenceRunnerError:
        raise
    except OSError as exc:
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Balanced-basis HMAC key could not be read safely."
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        _key_file_identity(before) != _key_file_identity(opened)
        or _key_file_identity(opened) != _key_file_identity(after)
        or _key_file_identity(before) != _key_file_identity(after_path)
    ):
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Balanced-basis HMAC key changed while being read."
        )
    if len(secret) < 32 or len(secret) > 4096:
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Balanced-basis HMAC key length changed while being read."
        )
    return secret


def _runtime_dtype(value: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Runtime dtype must be float16, bfloat16, float32, or float64."
        ) from exc


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:  # pragma: no cover
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            f"Required runtime package {name!r} is unavailable."
        ) from exc


def _numeric_backend_value(owner: Any, name: str) -> bool | str | None:
    try:
        value = getattr(owner, name, None)
    except (AttributeError, RuntimeError):
        return None
    if isinstance(value, bool):
        return value
    return None if value is None else str(value)


def _implementation_identity(
    *,
    runtime_device: torch.device,
    runtime_dtype_name: str,
) -> dict[str, Any]:
    files = {
        "runner": Path(__file__).resolve(),
        "core": Path(diagnostic_module.__file__).resolve(),
        "trace_cache": Path(trace_module.__file__).resolve(),
        "balanced_basis_phase": Path(basis_phase_module.__file__).resolve(),
        "routing_runtime": Path(routing_module.__file__).resolve(),
        "teacher_forced_readout": Path(teacher_audit_module.__file__).resolve(),
        "carrier_relocation_package": Path(carrier_module.__file__).resolve(),
        "carrier_child_package": Path(child_package_module.__file__).resolve(),
        "direct_local_loader": Path(direct_local_module.__file__).resolve(),
        "canonical_lfm2": Path(canonical_module.__file__).resolve(),
        "dynamic_blank_execution": Path(dynamic_blank_module.__file__).resolve(),
        "transformers_lfm2_runtime": Path(transformers_lfm2_module.__file__).resolve(),
    }
    body = {
        "schema": LFM2_EXACT4X_FIRST_DIVERGENCE_IMPLEMENTATION_SCHEMA,
        "files": {
            name: {"path": str(path), "sha256": _file_sha256(path)}
            for name, path in sorted(files.items())
        },
        "packages": {
            "python": platform.python_version(),
            "safetensors": _package_version("safetensors"),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
        },
        "numeric_contract": {
            "runtime_device": str(runtime_device),
            "runtime_device_type": runtime_device.type,
            "runtime_device_index": runtime_device.index,
            "runtime_dtype": runtime_dtype_name,
            "torch_default_dtype": str(torch.get_default_dtype()).replace("torch.", ""),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "torch_backends_fp32_precision": _numeric_backend_value(
                torch.backends, "fp32_precision"
            ),
            "cuda_matmul_fp32_precision": _numeric_backend_value(
                torch.backends.cuda.matmul, "fp32_precision"
            ),
            "cuda_matmul_allow_tf32": _numeric_backend_value(
                torch.backends.cuda.matmul, "allow_tf32"
            ),
            "cuda_matmul_allow_fp16_reduced_precision_reduction": (
                _numeric_backend_value(
                    torch.backends.cuda.matmul,
                    "allow_fp16_reduced_precision_reduction",
                )
            ),
            "cuda_matmul_allow_bf16_reduced_precision_reduction": (
                _numeric_backend_value(
                    torch.backends.cuda.matmul,
                    "allow_bf16_reduced_precision_reduction",
                )
            ),
            "cudnn_allow_tf32": _numeric_backend_value(
                torch.backends.cudnn, "allow_tf32"
            ),
            "deterministic_algorithms_enabled": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "device_name": (
                None
                if runtime_device.type != "cuda"
                else torch.cuda.get_device_name(runtime_device)
            ),
            "device_compute_capability": (
                None
                if runtime_device.type != "cuda"
                else list(torch.cuda.get_device_capability(runtime_device))
            ),
        },
        "execution_contract": {
            "network_allowed": False,
            "parent_model_loaded": False,
            "parent_forward_calls": 0,
            "training_steps": 0,
        },
    }
    return {**body, "implementation_sha256": canonical_sha256(body)}


def _seal_batch_checkpoint(
    request: Mapping[str, Any],
    reference: Any,
    split: str,
    result: Mapping[str, Any],
    *,
    authority: FirstDivergenceCheckpointAuthority,
) -> dict[str, Any]:
    authenticated_request = authenticate_lfm2_exact4x_first_divergence_request(request)
    request_sha256 = authenticated_request["request_sha256"]
    authenticated = authenticate_lfm2_exact4x_first_divergence_batch(
        result,
        expected_request=authenticated_request,
    )
    if (
        authenticated.get("batch_index") != int(reference.index)
        or authenticated.get("trace_input_sha256") != str(reference.input_sha256)
        or authenticated.get("trace_record_sha256") != str(reference.record_sha256)
        or authenticated.get("split") != split
    ):
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Batch result differs from its trace reference or split."
        )
    body = {
        "schema": BATCH_CHECKPOINT_SCHEMA,
        "request_sha256": request_sha256,
        "batch_index": int(reference.index),
        "trace_input_sha256": str(reference.input_sha256),
        "trace_record_sha256": str(reference.record_sha256),
        "split": split,
        "batch_audit_sha256": authenticated["batch_audit_sha256"],
        "checkpoint_hmac_algorithm": CHECKPOINT_HMAC_ALGORITHM,
        "checkpoint_hmac_key_id": authority.key_id,
        "result": authenticated,
    }
    sealed = {**body, "checkpoint_sha256": canonical_sha256(body)}
    return {
        **sealed,
        "checkpoint_hmac_sha256": authority.sign(sealed),
    }


def _authenticate_batch_checkpoint(
    value: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    reference: Any,
    split: str,
    authority: FirstDivergenceCheckpointAuthority,
) -> dict[str, Any]:
    authenticated_request = authenticate_lfm2_exact4x_first_divergence_request(request)
    request_sha256 = authenticated_request["request_sha256"]
    row = json.loads(_canonical_bytes(value).decode("ascii"))
    claimed_hmac = row.pop("checkpoint_hmac_sha256", None)
    claimed = row.pop("checkpoint_sha256", None)
    result = row.get("result")
    if (
        set(row)
        != {
            "batch_audit_sha256",
            "batch_index",
            "checkpoint_hmac_algorithm",
            "checkpoint_hmac_key_id",
            "request_sha256",
            "result",
            "schema",
            "split",
            "trace_input_sha256",
            "trace_record_sha256",
        }
        or row.get("schema") != BATCH_CHECKPOINT_SCHEMA
        or row.get("request_sha256") != request_sha256
        or row.get("batch_index") != int(reference.index)
        or row.get("trace_input_sha256") != str(reference.input_sha256)
        or row.get("trace_record_sha256") != str(reference.record_sha256)
        or row.get("split") != split
        or row.get("checkpoint_hmac_algorithm") != CHECKPOINT_HMAC_ALGORITHM
        or row.get("checkpoint_hmac_key_id") != authority.key_id
        or claimed != canonical_sha256(row)
        or not isinstance(result, Mapping)
    ):
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Batch checkpoint differs from its request or trace reference."
        )
    sealed = {**row, "checkpoint_sha256": claimed}
    if not authority.verify(sealed, claimed_hmac):
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Batch checkpoint HMAC does not verify."
        )
    authenticated = authenticate_lfm2_exact4x_first_divergence_batch(
        result,
        expected_request=authenticated_request,
    )
    if (
        row.get("batch_audit_sha256") != authenticated["batch_audit_sha256"]
        or authenticated.get("batch_index") != int(reference.index)
        or authenticated.get("trace_input_sha256") != str(reference.input_sha256)
        or authenticated.get("trace_record_sha256") != str(reference.record_sha256)
        or authenticated.get("split") != split
    ):
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Batch checkpoint result identity does not verify."
        )
    return {
        **sealed,
        "checkpoint_hmac_sha256": claimed_hmac,
    }


def _batch_child_forward_calls(value: Mapping[str, Any]) -> int:
    operation_audit = value.get("operation_audit")
    calls = (
        operation_audit.get("child_forward_calls")
        if isinstance(operation_audit, Mapping)
        else None
    )
    if isinstance(calls, bool) or not isinstance(calls, int) or calls != 1:
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Every persisted first-divergence batch must evidence one child forward."
        )
    return calls


@dataclass(frozen=True, slots=True)
class FirstDivergenceBatchResume:
    """Batch evidence plus invocation-relative forward provenance."""

    results: tuple[dict[str, Any], ...]
    checkpoint_hit_count: int
    computed_batch_count: int
    historical_checkpoint_child_forward_calls: int
    current_run_child_forward_calls: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_hit_count": self.checkpoint_hit_count,
            "computed_batch_count": self.computed_batch_count,
            "historical_checkpoint_child_forward_calls": (
                self.historical_checkpoint_child_forward_calls
            ),
            "current_run_child_forward_calls": self.current_run_child_forward_calls,
            "persisted_batch_evidence_child_forward_calls": (
                self.historical_checkpoint_child_forward_calls
                + self.current_run_child_forward_calls
            ),
        }


def resume_first_divergence_batches(
    *,
    request: Mapping[str, Any],
    references: Sequence[Any],
    split_by_batch: Mapping[int, str],
    checkpoint_directory: Path,
    processor: Callable[[Any, str], Mapping[str, Any]],
    checkpoint_authority: FirstDivergenceCheckpointAuthority,
    progress: _ProgressJournal | None = None,
) -> FirstDivergenceBatchResume:
    """Authenticate existing rows and compute only missing canonical batches."""

    authenticated_request = authenticate_lfm2_exact4x_first_divergence_request(request)
    if not isinstance(checkpoint_authority, FirstDivergenceCheckpointAuthority):
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "checkpoint_authority must be a typed runtime HMAC authority."
        )
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    if checkpoint_directory.is_symlink():
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Batch checkpoint directory cannot be a symlink."
        )
    results = []
    checkpoint_hit_count = 0
    computed_batch_count = 0
    historical_child_forwards = 0
    current_child_forwards = 0
    for expected_index, reference in enumerate(references):
        if int(reference.index) != expected_index:
            raise Lfm2Exact4xFirstDivergenceRunnerError(
                "Trace references are not canonically ordered."
            )
        split = split_by_batch.get(expected_index)
        if split not in {"fit", "heldout"}:
            raise Lfm2Exact4xFirstDivergenceRunnerError(
                "Trace reference lacks an authenticated FIT/heldout split."
            )
        path = checkpoint_directory / f"batch-{expected_index:06d}.json"
        if path.exists() or path.is_symlink():
            checkpoint = _authenticate_batch_checkpoint(
                _read_json(path, f"batch {expected_index} checkpoint"),
                request=authenticated_request,
                reference=reference,
                split=split,
                authority=checkpoint_authority,
            )
            checkpoint_hit_count += 1
            historical_child_forwards += _batch_child_forward_calls(
                checkpoint["result"]
            )
            if progress is not None:
                progress.emit(
                    "batch-checkpoint-hit",
                    batch_index=expected_index,
                    split=split,
                    checkpoint_sha256=checkpoint["checkpoint_sha256"],
                    historical_checkpoint_child_forward_calls=1,
                    current_run_child_forward_calls=0,
                )
        else:
            if progress is not None:
                progress.emit(
                    "batch-start",
                    batch_index=expected_index,
                    split=split,
                )
            result = processor(reference, split)
            checkpoint = _seal_batch_checkpoint(
                authenticated_request,
                reference,
                split,
                result,
                authority=checkpoint_authority,
            )
            computed_batch_count += 1
            current_child_forwards += _batch_child_forward_calls(checkpoint["result"])
            _atomic_write_new_json(path, checkpoint)
            if progress is not None:
                progress.emit(
                    "batch-complete",
                    batch_index=expected_index,
                    split=split,
                    checkpoint_sha256=checkpoint["checkpoint_sha256"],
                    batch_audit_sha256=checkpoint["batch_audit_sha256"],
                    historical_checkpoint_child_forward_calls=0,
                    current_run_child_forward_calls=1,
                )
        results.append(dict(checkpoint["result"]))
    return FirstDivergenceBatchResume(
        results=tuple(results),
        checkpoint_hit_count=checkpoint_hit_count,
        computed_batch_count=computed_batch_count,
        historical_checkpoint_child_forward_calls=historical_child_forwards,
        current_run_child_forward_calls=current_child_forwards,
    )


def _compact_selected_parent_tensors(
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    compact = {}
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            raise Lfm2Exact4xFirstDivergenceRunnerError(
                f"Selected parent tensor {name!r} is invalid."
            )
        retained = value if name.startswith("input.") else value[:, -1:, :]
        compact[name] = retained.detach().to(device="cpu").contiguous().clone()
    return compact


def _trajectory_position_ids(
    parent_tensors: Mapping[str, torch.Tensor],
    forced_count: int,
) -> torch.Tensor | None:
    prompt = parent_tensors.get("input.position_ids")
    if prompt is None:
        return None
    if prompt.ndim != 2 or int(prompt.shape[0]) != 1 or int(prompt.shape[1]) <= 0:
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Cached parent position IDs are invalid."
        )
    continuation = prompt[:, -1:] + torch.arange(
        1,
        forced_count + 1,
        dtype=torch.long,
    ).unsqueeze(0)
    return torch.cat((prompt.to(torch.long), continuation), dim=1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-directory", type=Path, required=True)
    parser.add_argument("--child-package-sha256", required=True)
    parser.add_argument("--trace-cache", type=Path, required=True)
    parser.add_argument("--basis-phase-cache-root", type=Path, required=True)
    parser.add_argument("--basis-phase-key-sha256", required=True)
    parser.add_argument("--basis-phase-hmac-key-file", type=Path, required=True)
    parser.add_argument("--basis-phase-hmac-key-id", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--runtime-device", default="cuda:0")
    parser.add_argument("--runtime-dtype", default="float32")
    parser.add_argument("--source-width", type=int)
    parser.add_argument("--target-hidden-size", type=int)
    parser.add_argument("--focus-layer-indices")
    parser.add_argument(
        "--maximum-selected-parent-bytes",
        type=int,
        default=512 * 1024 * 1024,
    )
    parser.add_argument(
        "--maximum-donor-rows-per-score-chunk",
        type=int,
        default=4096,
    )
    return parser


def _parse_layer_indices(value: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as exc:
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "focus layer indices must be comma-separated integers."
        ) from exc
    if not values or values[0] < 0:
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "focus layer indices must be nonnegative."
        )
    return values


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    child_directory = _ordinary_directory(args.child_directory, "child directory")
    trace_directory = _ordinary_directory(args.trace_cache, "trace cache")
    phase_root = _ordinary_directory(
        args.basis_phase_cache_root,
        "balanced-basis phase cache root",
    )
    output = Path(args.output_directory).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Output directory cannot be a symlink."
        )
    expected_child_package = _digest(
        args.child_package_sha256,
        "child_package_sha256",
    )
    runtime_dtype = _runtime_dtype(args.runtime_dtype)
    runtime_device = torch.device(args.runtime_device)
    implementation_identity = _implementation_identity(
        runtime_device=runtime_device,
        runtime_dtype_name=args.runtime_dtype,
    )
    implementation_sha256 = implementation_identity["implementation_sha256"]
    requested_focus_layers = (
        None
        if args.focus_layer_indices is None
        else _parse_layer_indices(args.focus_layer_indices)
    )

    package = load_authenticated_lfm2_carrier_child_package(
        child_directory,
        expected_package_sha256=expected_child_package,
    )
    target_hidden_size = int(package.receipt["hidden_size"])
    if (
        args.target_hidden_size is not None
        and args.target_hidden_size != target_hidden_size
    ):
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Requested target hidden size differs from the child package."
        )
    canonical_child = load_lfm2_direct_local_model(
        child_directory,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    package = load_authenticated_lfm2_carrier_child_package(
        child_directory,
        child_model=canonical_child,
        expected_package_sha256=package.package_sha256,
    )
    child, routing, relocation = relocate_and_bind_packaged_lfm2_carrier_routes(
        canonical_child,
        package.routes,
        provenance_resource_fingerprint=implementation_sha256,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        expected_target_rows=int(package.receipt["target_rows"]),
        expected_hidden_size=target_hidden_size,
    )
    del canonical_child
    gc.collect()
    child.eval()
    child_before = fingerprint_lfm2_parent(child)
    route_before = fingerprint_lfm2_carrier_runtime_owned_tensor_state(routing)
    trace_cache = open_lfm2_circuit_trace_cache(trace_directory, verify=False)
    manifest = trace_cache.manifest
    topology = manifest.get("layer_topology")
    topology_layers = topology.get("layers") if isinstance(topology, Mapping) else None
    if not isinstance(topology_layers, list) or not topology_layers:
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Trace cache has no complete layer topology."
        )
    layer_count = len(topology_layers)
    try:
        child_layers = tuple(child.model.layers)
    except (AttributeError, TypeError) as exc:
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Complete child has no iterable LFM2 layer stack."
        ) from exc
    if len(child_layers) != layer_count:
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Parent trace and complete child do not share a layer boundary alignment."
        )
    source_width_values = [
        layer["contract"].get("hidden_size")
        for layer in topology_layers
        if isinstance(layer, Mapping) and isinstance(layer.get("contract"), Mapping)
    ]
    source_widths = set(source_width_values)
    discovered_source_width = next(iter(source_widths), None)
    if (
        len(source_width_values) != layer_count
        or len(source_widths) != 1
        or isinstance(discovered_source_width, bool)
        or not isinstance(discovered_source_width, int)
        or discovered_source_width <= 0
    ):
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Trace topology has no single authenticated residual source width."
        )
    source_width = discovered_source_width
    if args.source_width is not None and args.source_width != source_width:
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Requested source width differs from the trace topology."
        )
    all_layers = tuple(range(layer_count))
    focus_layers = (
        all_layers if requested_focus_layers is None else requested_focus_layers
    )
    if focus_layers[-1] >= layer_count:
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Focus layer indices leave the complete child topology."
        )
    secret = _read_raw_hmac_key(args.basis_phase_hmac_key_file)
    try:
        authority = PhaseAnchorAuthority(
            key_id=args.basis_phase_hmac_key_id,
            secret=secret,
        )
        checkpoint_authority = FirstDivergenceCheckpointAuthority(
            key_id=args.basis_phase_hmac_key_id,
            secret=secret,
        )
    finally:
        del secret

    with trace_cache.selective_authenticated_session() as trace_session:
        basis_phase = load_existing_lfm2_circuit_balanced_basis_phase(
            DynamicFoldPhaseCache(phase_root),
            args.basis_phase_key_sha256,
            anchor_authority=authority,
            expected_parent_identity_sha256=trace_cache.parent_fingerprint,
            expected_trace_cache_key=trace_cache.cache_key,
            expected_trace_selective_session_sha256=trace_session.evidence_sha256,
            expected_source_width=source_width,
            expected_target_width=target_hidden_size,
        )
        references = trace_session.batch_references
        parent_logit_domain_contract = _build_parent_logit_domain_contract(
            package=package,
            routing=routing,
            references=references,
            trace_cache_key=trace_cache.cache_key,
            trace_selective_session_sha256=trace_session.evidence_sha256,
        )
        split_by_batch = authoritative_split_by_batch(
            basis_phase.report,
            references,
        )
        budget = verify_complete_exact_budget_child(
            child,
            basis_phase=basis_phase,
            route_package=package.routes,
            portable_runtime_receipt=relocation,
        )
        route_runtime_receipt = routing.as_dict()
        portable_runtime_receipt = relocation.as_dict()
        request = build_lfm2_exact4x_first_divergence_request(
            child_package_sha256=package.package_sha256,
            child_state_sha256=child_before.identity_sha256,
            route_package_sha256=package.routes.package_sha256,
            route_runtime_sha256=route_runtime_receipt["runtime_sha256"],
            portable_runtime_receipt_sha256=portable_runtime_receipt["receipt_sha256"],
            trace_cache_key=trace_cache.cache_key,
            trace_selective_session_sha256=trace_session.evidence_sha256,
            basis_phase_key_sha256=basis_phase.phase_key.phase_key_sha256,
            basis_completion_sha256=basis_phase.completion_sha256,
            basis_map_sha256=basis_phase.map_sha256,
            budget_observation=budget,
            layer_count=layer_count,
            focus_layer_indices=focus_layers,
            runtime_device=str(runtime_device),
            runtime_dtype=args.runtime_dtype,
            maximum_selected_parent_bytes=args.maximum_selected_parent_bytes,
            implementation_identity=implementation_identity,
            parent_logit_domain_contract=parent_logit_domain_contract,
        )
        request_path = output / "request.json"
        if request_path.exists() or request_path.is_symlink():
            existing_request = authenticate_lfm2_exact4x_first_divergence_request(
                _read_json(request_path, "diagnostic request")
            )
            if existing_request != request:
                raise Lfm2Exact4xFirstDivergenceRunnerError(
                    "Existing diagnostic request differs from current identities."
                )
        else:
            _atomic_write_new_json(request_path, request)
        progress = _ProgressJournal(
            output / "progress.jsonl",
            request["request_sha256"],
        )
        progress.emit(
            "preflight-complete",
            batch_count=len(references),
            child_package_sha256=package.package_sha256,
            route_package_sha256=package.routes.package_sha256,
            trace_cache_key=trace_cache.cache_key,
            basis_map_sha256=basis_phase.map_sha256,
            exact_total_parameter_elements=budget["counted_total_parameter_elements"],
            authenticated_parent_parameter_elements=budget["parent_parameter_elements"],
            caller_parameter_count_assertion_used=False,
            parent_logit_domain_contract_sha256=parent_logit_domain_contract[
                "contract_sha256"
            ],
            defined_semantic_row_count=parent_logit_domain_contract[
                "defined_semantic_row_count"
            ],
            parent_physical_row_capacity=parent_logit_domain_contract[
                "parent_physical_row_capacity"
            ],
            undefined_capacity_row_count=parent_logit_domain_contract[
                "undefined_capacity_row_count"
            ],
            parent_forward_calls=0,
        )
        input_table = child.get_input_embeddings().weight
        output_table = child.get_output_embeddings().weight
        if (
            input_table is output_table
            or input_table.untyped_storage().data_ptr()
            == output_table.untyped_storage().data_ptr()
            or tuple(input_table.shape)
            != (int(package.receipt["target_rows"]), target_hidden_size)
            or tuple(output_table.shape) != tuple(input_table.shape)
        ):
            raise Lfm2Exact4xFirstDivergenceRunnerError(
                "Exact-4x carrier child must expose independent complete C_in/C_out tables."
            )
        paired_sessions = routing.authenticated_live_table_sessions(
            input_table,
            output_table,
        )
        with paired_sessions as (input_session, output_session):

            def process(reference: Any, split: str) -> Mapping[str, Any]:
                names = parent_tensor_names_for_first_divergence(
                    reference,
                    layer_count=layer_count,
                    focus_layer_indices=focus_layers,
                )
                logical_bytes = parent_selection_logical_bytes(reference, names)
                if logical_bytes > args.maximum_selected_parent_bytes:
                    raise Lfm2Exact4xFirstDivergenceRunnerError(
                        "Selected parent payload exceeds the configured bounded read ceiling."
                    )
                payload = trace_session.load_selected_tensors(reference.index, names)
                try:
                    selected_sha256 = payload.selection_sha256
                    parent_tensors = _compact_selected_parent_tensors(payload.tensors)
                    prompt_ids = parent_tensors["input.input_ids"].to(torch.long)
                    forced_ids = parent_tensors["input.forced_decode_ids"].to(
                        torch.long
                    )
                    trajectory_ids = torch.cat((prompt_ids[0], forced_ids[0]), dim=0)
                    reconstructed = routing.reconstruct_selected_donor_embeddings(
                        input_session,
                        trajectory_ids,
                    ).unsqueeze(0)
                    trajectory_mask = torch.ones(
                        (1, int(trajectory_ids.numel())),
                        dtype=torch.long,
                        device=reconstructed.device,
                    )
                    position_ids = _trajectory_position_ids(
                        parent_tensors,
                        int(forced_ids.shape[1]),
                    )
                    replay = capture_lfm2_child_embedding_replay(
                        child,
                        reconstructed,
                        attention_mask=trajectory_mask,
                        position_ids=position_ids,
                        selected_layer_indices=all_layers,
                        sample_last_tokens=int(forced_ids.shape[1]) + 1,
                        max_capture_bytes=args.maximum_selected_parent_bytes,
                        input_device=runtime_device,
                    )
                    result = audit_lfm2_exact4x_first_divergence_batch(
                        request=request,
                        batch_index=int(reference.index),
                        trace_input_sha256=str(reference.input_sha256),
                        trace_record_sha256=str(reference.record_sha256),
                        split=split,
                        parent_tensors=parent_tensors,
                        selected_parent_payload=payload,
                        trace_selective_session=trace_session,
                        reconstructed_trajectory_embeddings=reconstructed,
                        child_replay=replay,
                        coordinate_map=basis_phase,
                        route_package=package.routes,
                        portable_runtime_receipt=relocation,
                        routing=routing,
                        input_routing_session=input_session,
                        output_routing_session=output_session,
                        maximum_donor_rows_per_score_chunk=(
                            args.maximum_donor_rows_per_score_chunk
                        ),
                    )
                finally:
                    trace_session.release_selected_tensors(payload)
                progress.emit(
                    "parent-selected-payload-released",
                    batch_index=int(reference.index),
                    selected_logical_bytes=logical_bytes,
                    compact_tensor_count=len(parent_tensors),
                    selected_payload_sha256=selected_sha256,
                )
                del parent_tensors, reconstructed, replay
                return result

            batch_resume = resume_first_divergence_batches(
                request=request,
                references=references,
                split_by_batch=split_by_batch,
                checkpoint_directory=output / "batches",
                processor=process,
                checkpoint_authority=checkpoint_authority,
                progress=progress,
            )
            batch_results = batch_resume.results
        routing_sessions_receipt = paired_sessions.as_dict()
        trace_session.verify()

    child_after = fingerprint_lfm2_parent(child)
    route_after = fingerprint_lfm2_carrier_runtime_owned_tensor_state(routing)
    if child_after != child_before or route_after != route_before:
        raise Lfm2Exact4xFirstDivergenceRunnerError(
            "Child or carrier route state changed during diagnostic observation."
        )
    ledger = aggregate_lfm2_exact4x_first_divergence(request, batch_results)
    ledger_path = output / "first-divergence-ledger.json"
    if ledger_path.exists() or ledger_path.is_symlink():
        if _read_json(ledger_path, "first-divergence ledger") != ledger:
            raise Lfm2Exact4xFirstDivergenceRunnerError(
                "Existing first-divergence ledger differs from resumed results."
            )
    else:
        _atomic_write_new_json(ledger_path, ledger)
    completion_body = {
        "schema": COMPLETION_SCHEMA,
        "status": "complete-observation",
        "request_sha256": request["request_sha256"],
        "ledger_sha256": ledger["ledger_sha256"],
        "child_package_sha256": package.package_sha256,
        "route_package_sha256": package.routes.package_sha256,
        "child_state_sha256": child_after.identity_sha256,
        "route_state_sha256": route_after["state_sha256"],
        "implementation_identity": implementation_identity,
        "checkpoint_hmac_algorithm": CHECKPOINT_HMAC_ALGORITHM,
        "checkpoint_hmac_key_id": checkpoint_authority.key_id,
        "portable_runtime_receipt": relocation.as_dict(),
        "batch_count": len(batch_results),
        "parent_forward_calls": 0,
        "persisted_batch_evidence_child_forward_calls": ledger["operation_audit"][
            "child_forward_calls"
        ],
        "completion_scope": "stable-persisted-evidence-across-invocations",
        "invocation_relative_resume_accounting_is_in_progress_journal": True,
        "training_steps": 0,
        "model_mutations": 0,
        "diagnostic_model_mutations": 0,
        "runtime_relocation_before_observation_is_separately_receipted": True,
        "numeric_decision": None,
        "safety_observations_are_separate_not_failures": True,
        "generation_cap_saturation_is_censored_not_failure": True,
    }
    completion = {
        **completion_body,
        "completion_sha256": canonical_sha256(completion_body),
    }
    completion_path = output / "complete.json"
    if completion_path.exists() or completion_path.is_symlink():
        if _read_json(completion_path, "diagnostic completion") != completion:
            raise Lfm2Exact4xFirstDivergenceRunnerError(
                "Existing diagnostic completion differs from current results."
            )
    else:
        _atomic_write_new_json(completion_path, completion)
    progress.emit(
        "run-complete",
        ledger_sha256=ledger["ledger_sha256"],
        completion_sha256=completion["completion_sha256"],
        batch_count=len(batch_results),
        batch_resume_audit=batch_resume.as_dict(),
        routing_sessions_receipt=routing_sessions_receipt,
        parent_forward_calls=0,
        current_run_child_forward_calls=(
            1 + batch_resume.current_run_child_forward_calls
        ),
        historical_checkpoint_child_forward_calls=(
            batch_resume.historical_checkpoint_child_forward_calls
        ),
        numeric_decision=None,
    )
    return completion


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        completion = run(args)
    except (
        Lfm2Exact4xFirstDivergenceError,
        Lfm2Exact4xFirstDivergenceRunnerError,
        RuntimeError,
        ValueError,
    ) as exc:
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1
    sys.stdout.write(_canonical_bytes(completion).decode("ascii") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BATCH_CHECKPOINT_SCHEMA",
    "CHECKPOINT_HMAC_ALGORITHM",
    "COMPLETION_SCHEMA",
    "FirstDivergenceCheckpointAuthority",
    "FirstDivergenceBatchResume",
    "Lfm2Exact4xFirstDivergenceRunnerError",
    "PROGRESS_SCHEMA",
    "RUNNER_SCHEMA",
    "main",
    "resume_first_divergence_batches",
    "run",
]
