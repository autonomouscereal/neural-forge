"""Persistent complete-child replay phases for canonical LFM2 capture.

This codec is intentionally outside the replay's existing sealed metadata.
The replay continues to truthfully report that its capture was child-only and
memory-only; an immutable dynamic-fold phase may persist that already sealed
value afterward and restore it only for the exact same live child state and
capture contract.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from . import dynamic_child_replay_phase_cache as _generic_cache_module
from . import hf_lfm2_circuit_trace_cache as _trace
from .dynamic_child_replay_phase_cache import (
    DYNAMIC_CHILD_REPLAY_PHASE_IDENTITY_SCHEMA,
    AuthenticatedDynamicChildReplayPhase,
    DynamicChildReplayPhaseCacheContext,
    DynamicChildReplayPhaseIdentity,
    load_or_build_dynamic_child_replay_phase,
)
from .dynamic_fold_phase_cache import (
    FoldPhaseCacheIntegrityError,
    FoldPhaseKey,
    VerifiedFoldPhaseCacheEntry,
)
from .hf_lfm2_circuit_trace_cache import (
    CHILD_REPLAY_BATCH_SCHEMA,
    CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
    CHILD_REPLAY_SCHEMA,
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2ChildEvidenceProfile,
    Lfm2ChildReplay,
    Lfm2ChildReplayBatch,
    Lfm2CircuitTraceBatch,
    Lfm2CircuitTraceCacheError,
    Lfm2CircuitTraceCaptureConfig,
    capture_lfm2_child_replay,
    compile_lfm2_child_replay_numeric_policy,
    fingerprint_lfm2_parent,
    implementation_identity,
)
from .hf_lfm2_replay_sharding import Lfm2ReplayShardingConfig

LFM2_CHILD_REPLAY_PHASE = "lfm2-child-replay-v1"
LFM2_CHILD_REPLAY_PHASE_CODEC_ID = "lfm2-child-replay-codec-v1"
LFM2_CHILD_REPLAY_PHASE_REQUEST_SCHEMA = (
    "neural-forge-lfm2-child-replay-phase-request-v1"
)
LFM2_CHILD_REPLAY_PHASE_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-child-replay-phase-receipt-v1"
)
LFM2_CHILD_REPLAY_SAFETENSORS_SCHEMA = (
    "neural-forge-lfm2-child-replay-phase-tensors-v1"
)

_REQUEST_FILENAME = "request.json"
_REPLAY_FILENAME = "replay.json"
_RECEIPT_FILENAME = "receipt.json"
_BATCH_DIRECTORY = "batches"
_BATCH_RECORD_FILENAME = "record.json"
_BATCH_TENSOR_FILENAME = "tensors.safetensors"
_SHA256_LENGTH = 64


class Lfm2ChildReplayPhaseCacheError(RuntimeError):
    """The LFM2 complete-replay phase request is unsupported."""


class Lfm2ChildReplayPhaseAuthenticationError(FoldPhaseCacheIntegrityError):
    """An LFM2 replay or its persisted codec payload failed authentication."""


def _canonical_json_bytes(value: Any, name: str) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2ChildReplayPhaseCacheError(
            f"{name} must be finite canonical JSON."
        ) from exc


def _canonical_value(value: Any, name: str) -> Any:
    return json.loads(_canonical_json_bytes(value, name).decode("ascii"))


def _canonical_mapping(value: Any, name: str) -> dict[str, Any]:
    result = _canonical_value(value, name)
    if not isinstance(result, dict):
        raise Lfm2ChildReplayPhaseCacheError(f"{name} must be a mapping.")
    return result


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2ChildReplayPhaseAuthenticationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _json_sha256(value: Any, name: str) -> str:
    return hashlib.sha256(_canonical_json_bytes(value, name)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise Lfm2ChildReplayPhaseAuthenticationError(
            f"Could not hash child replay payload file {path.name!r}."
        ) from exc
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = _canonical_json_bytes(dict(value), path.name) + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise Lfm2ChildReplayPhaseCacheError(
            f"Could not write immutable child replay payload {path.name!r}."
        ) from exc


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise Lfm2ChildReplayPhaseAuthenticationError(
            f"Child replay payload {path.name!r} is not a regular file."
        )
    try:
        raw = path.read_bytes()
        result = json.loads(raw, object_pairs_hook=_strict_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise Lfm2ChildReplayPhaseAuthenticationError(
            f"Could not parse child replay payload {path.name!r}."
        ) from exc
    if (
        not isinstance(result, dict)
        or raw != _canonical_json_bytes(result, path.name) + b"\n"
    ):
        raise Lfm2ChildReplayPhaseAuthenticationError(
            f"Child replay payload {path.name!r} is not canonical."
        )
    return result


def _verify_seal(
    value: Any,
    seal_name: str,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2ChildReplayPhaseAuthenticationError(
            f"{name} must be a mapping."
        )
    result = _canonical_mapping(value, name)
    claimed = result.pop(seal_name, None)
    if not isinstance(claimed, str) or _json_sha256(result, name) != claimed:
        raise Lfm2ChildReplayPhaseAuthenticationError(
            f"{name} self-seal is invalid."
        )
    return {**result, seal_name: claimed}


def _source_sha256(path: Path, domain: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(path.resolve(strict=True).read_bytes())
    return digest.hexdigest()


def lfm2_child_replay_phase_codec_implementation_sha256() -> str:
    """Fingerprint the exact family codec and generic publication engine."""

    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-child-replay-phase-codec-v1\0")
    digest.update(Path(__file__).resolve(strict=True).read_bytes())
    digest.update(b"\0generic-cache\0")
    digest.update(
        Path(_generic_cache_module.__file__).resolve(strict=True).read_bytes()
    )
    return digest.hexdigest()


def _selected_config(
    config: Lfm2CircuitTraceCaptureConfig | None,
) -> Lfm2CircuitTraceCaptureConfig:
    result = (
        Lfm2CircuitTraceCaptureConfig(
            decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE
        )
        if config is None
        else config
    )
    if not isinstance(result, Lfm2CircuitTraceCaptureConfig):
        raise TypeError("config must be Lfm2CircuitTraceCaptureConfig or None.")
    if result.decode_trace_mode != FULL_PREFIX_CANONICAL_TRACE_MODE:
        raise Lfm2ChildReplayPhaseCacheError(
            "Persistent child replay requires canonical full-prefix capture."
        )
    return result


def _input_descriptors(torch: Any, batch: Any) -> dict[str, Any]:
    values = {
        "input.input_ids": batch.input_ids,
        "input.attention_mask": batch.attention_mask,
        "input.forced_decode_ids": batch.forced_decode_ids,
    }
    if batch.position_ids is not None:
        values["input.position_ids"] = batch.position_ids
    return {
        name: _trace._tensor_descriptor(torch, tensor)
        for name, tensor in sorted(values.items())
    }


@dataclass(frozen=True, slots=True)
class Lfm2ChildReplayPhaseRequest:
    """Portable content address for one complete canonical LFM2 replay."""

    identity: DynamicChildReplayPhaseIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.identity, DynamicChildReplayPhaseIdentity):
            raise TypeError("identity must be DynamicChildReplayPhaseIdentity.")
        if self.identity.codec_id != LFM2_CHILD_REPLAY_PHASE_CODEC_ID:
            raise Lfm2ChildReplayPhaseCacheError(
                "LFM2 child replay request has the wrong codec identity."
            )
        family = self.family_request
        if family.get("schema") != LFM2_CHILD_REPLAY_PHASE_REQUEST_SCHEMA:
            raise Lfm2ChildReplayPhaseCacheError(
                "LFM2 child replay request schema is unsupported."
            )
        _verify_seal(
            family,
            "family_request_sha256",
            "LFM2 child replay family request",
        )

    @property
    def phase_key(self) -> FoldPhaseKey:
        return self.identity.phase_key

    @property
    def request_sha256(self) -> str:
        return self.identity.request_sha256

    @property
    def family_request(self) -> dict[str, Any]:
        request = self.identity.as_dict()
        request.pop("request_sha256")
        family = request.get("family_request")
        if not isinstance(family, Mapping):
            raise Lfm2ChildReplayPhaseCacheError(
                "Dynamic child replay request has no LFM2 family request."
            )
        return _canonical_mapping(family, "LFM2 child replay family request")

    def as_dict(self) -> dict[str, Any]:
        return self.identity.as_dict()


def describe_lfm2_child_replay_phase(
    child_model: Any,
    replay_batches: Iterable[
        Mapping[str, Any] | Lfm2CircuitTraceBatch | Lfm2ChildReplayBatch
    ],
    *,
    selected_layer_indices: Sequence[int],
    config: Lfm2CircuitTraceCaptureConfig | None = None,
    input_device: Any | None = None,
    replay_capture_mode: str = CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
    child_evidence_profile: Lfm2ChildEvidenceProfile | None = None,
) -> Lfm2ChildReplayPhaseRequest:
    """Describe a reusable canonical replay without executing the child."""

    if replay_capture_mode != CHILD_REPLAY_PREFIX_BY_PREFIX_MODE:
        raise Lfm2ChildReplayPhaseCacheError(
            "P0 persistent child replay supports only "
            "'prefix_by_prefix_canonical'."
        )
    selected_capture_config = _selected_config(config)
    if child_evidence_profile is not None and not isinstance(
        child_evidence_profile,
        Lfm2ChildEvidenceProfile,
    ):
        raise TypeError(
            "child_evidence_profile must be Lfm2ChildEvidenceProfile or None."
        )
    torch = _trace._torch()
    batches = tuple(replay_batches)
    prepared = _trace._prepare_batches(
        torch,
        _trace._exact_replay_batch_mappings(batches),
        selected_capture_config,
    )
    layers, _contracts, topology = _trace._layer_topology(child_model)
    selected_indices = _trace._selected_layer_indices(
        len(layers),
        selected_layer_indices,
    )
    if (
        child_evidence_profile is not None
        and child_evidence_profile.selected_layer_indices != selected_indices
    ):
        raise Lfm2ChildReplayPhaseCacheError(
            "Child evidence profile differs from selected_layer_indices."
        )
    device = _trace._resolve_input_device(torch, child_model, input_device)
    child = fingerprint_lfm2_parent(child_model).as_dict()
    corpus = _trace._corpus_identity(prepared)
    execution = _trace._execution_identity(child_model, device)
    implementation = implementation_identity()
    numeric_policy = compile_lfm2_child_replay_numeric_policy(
        child_model,
        input_device=device,
    )
    selected_topology_body = {
        "schema": "neural-forge-lfm2-child-selected-topology-v1",
        "model_layer_count": len(layers),
        "selected_layer_indices": list(selected_indices),
        "layers": [topology["layers"][index] for index in selected_indices],
    }
    selected_topology = {
        **selected_topology_body,
        "selected_topology_sha256": _json_sha256(
            selected_topology_body,
            "selected child topology",
        ),
    }
    capture_config = selected_capture_config.as_dict()
    selection_body = {
        "schema": "neural-forge-lfm2-child-replay-selection-v2",
        "selected_layer_indices": list(selected_indices),
        "capture": capture_config,
        "replay_capture_mode": CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
        "selected_topology": selected_topology,
        "child_evidence_profile": (
            None
            if child_evidence_profile is None
            else child_evidence_profile.as_dict()
        ),
        "child_evidence_profile_sha256": (
            None
            if child_evidence_profile is None
            else child_evidence_profile.profile_sha256
        ),
    }
    selection = {
        **selection_body,
        "selection_sha256": _json_sha256(selection_body, "replay selection"),
    }
    batch_contracts = []
    canonical_prefix_forward_count = 0
    for batch in prepared:
        phase_rows = _trace._child_replay_phase_rows(
            batch,
            selected_capture_config,
        )
        canonical_prefix_forward_count += len(phase_rows)
        batch_contracts.append(
            {
                "batch_index": batch.index,
                "input_sha256": batch.input_sha256,
                "phase_order": [row["phase"] for row in phase_rows],
                "phase_rows": list(phase_rows),
                "input_tensor_descriptors": _input_descriptors(torch, batch),
            }
        )
    codec_implementation_sha256 = (
        lfm2_child_replay_phase_codec_implementation_sha256()
    )
    family_body = {
        "schema": LFM2_CHILD_REPLAY_PHASE_REQUEST_SCHEMA,
        "child": child,
        "corpus": corpus,
        "implementation": implementation,
        "layer_topology": topology,
        "selection": selection,
        "execution": execution,
        "numeric_execution_policy": numeric_policy,
        "batch_count": len(batch_contracts),
        "ordered_batch_contracts": batch_contracts,
        "canonical_prefix_forward_count": canonical_prefix_forward_count,
        "codec_id": LFM2_CHILD_REPLAY_PHASE_CODEC_ID,
        "codec_implementation_sha256": codec_implementation_sha256,
        "parent_model_bound": False,
        "tokenizer_bound": False,
        "training_used": False,
    }
    family_request = {
        **family_body,
        "family_request_sha256": _json_sha256(
            family_body,
            "LFM2 child replay family request",
        ),
    }
    generic_source_sha256 = _source_sha256(
        Path(_generic_cache_module.__file__),
        b"neural-forge-dynamic-child-replay-phase-cache-source-v1\0",
    )
    phase_key = FoldPhaseKey(
        phase_name=LFM2_CHILD_REPLAY_PHASE,
        dependency_sha256s={
            "child-identity": child["identity_sha256"],
            "child-state": child["state_dict_sha256"],
            "child-runtime-buffers": child["runtime_buffer_sha256"],
            "corpus": corpus["corpus_sha256"],
            "implementation": implementation["implementation_sha256"],
            "layer-topology": topology["layer_topology_sha256"],
            "selection": selection["selection_sha256"],
            "execution": execution["execution_sha256"],
            "numeric-policy": numeric_policy[
                "replay_numeric_policy_sha256"
            ],
            "family-request": family_request["family_request_sha256"],
            "codec-source": codec_implementation_sha256,
            "generic-cache-source": generic_source_sha256,
        },
        numeric_contract={
            "schema": "neural-forge-lfm2-child-replay-phase-numeric-v1",
            "capture": capture_config,
            "execution": execution,
            "numeric_execution_policy": numeric_policy,
            "replay_capture_mode": CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
        },
        abi={
            "schema": "neural-forge-lfm2-child-replay-phase-abi-v1",
            "codec_id": LFM2_CHILD_REPLAY_PHASE_CODEC_ID,
            "codec_implementation_sha256": codec_implementation_sha256,
            "payload": "canonical-json-plus-per-batch-safetensors",
            "replay_schema": CHILD_REPLAY_SCHEMA,
            "replay_batch_schema": CHILD_REPLAY_BATCH_SCHEMA,
        },
    )
    generic_request = {
        "schema": DYNAMIC_CHILD_REPLAY_PHASE_IDENTITY_SCHEMA,
        "phase_key_sha256": phase_key.phase_key_sha256,
        "codec_id": LFM2_CHILD_REPLAY_PHASE_CODEC_ID,
        "family": "lfm2",
        "family_request": family_request,
    }
    return Lfm2ChildReplayPhaseRequest(
        DynamicChildReplayPhaseIdentity(
            phase_key=phase_key,
            request=generic_request,
            codec_id=LFM2_CHILD_REPLAY_PHASE_CODEC_ID,
        )
    )


def _family_request(
    identity: DynamicChildReplayPhaseIdentity,
) -> dict[str, Any]:
    if identity.codec_id != LFM2_CHILD_REPLAY_PHASE_CODEC_ID:
        raise Lfm2ChildReplayPhaseAuthenticationError(
            "LFM2 codec received a foreign phase identity."
        )
    request = identity.as_dict()
    request.pop("request_sha256")
    family = request.get("family_request")
    if not isinstance(family, Mapping):
        raise Lfm2ChildReplayPhaseAuthenticationError(
            "LFM2 phase identity has no family request."
        )
    result = _verify_seal(
        family,
        "family_request_sha256",
        "LFM2 child replay family request",
    )
    if (
        result.get("schema") != LFM2_CHILD_REPLAY_PHASE_REQUEST_SCHEMA
        or result.get("codec_id") != LFM2_CHILD_REPLAY_PHASE_CODEC_ID
    ):
        raise Lfm2ChildReplayPhaseAuthenticationError(
            "LFM2 child replay family request contract changed."
        )
    return result


def _authenticate_replay(
    value: Any,
    identity: DynamicChildReplayPhaseIdentity,
) -> str:
    if not isinstance(value, Lfm2ChildReplay):
        raise TypeError("LFM2 child replay codec requires Lfm2ChildReplay.")
    family = _family_request(identity)
    try:
        metadata = value.metadata
        _trace._verify_seal(metadata, "replay_sha256", "child replay metadata")
    except Lfm2CircuitTraceCacheError as exc:
        raise Lfm2ChildReplayPhaseAuthenticationError(
            "Child replay metadata self-seal is invalid."
        ) from exc
    expected_metadata_fields = {
        "schema",
        "child",
        "corpus",
        "implementation",
        "layer_topology",
        "selection",
        "execution",
        "numeric_execution_policy",
        "batch_count",
        "ordered_batches",
        "forward_counts",
        "replay_benchmark",
        "causal_equivalence",
        "runtime",
        "replay_sha256",
    }
    if set(metadata) != expected_metadata_fields or metadata.get("schema") != (
        CHILD_REPLAY_SCHEMA
    ):
        raise Lfm2ChildReplayPhaseAuthenticationError(
            "Child replay metadata fields or schema changed."
        )
    for field in (
        "child",
        "corpus",
        "implementation",
        "layer_topology",
        "selection",
        "execution",
        "numeric_execution_policy",
    ):
        if metadata.get(field) != family.get(field):
            raise Lfm2ChildReplayPhaseAuthenticationError(
                f"Child replay {field} differs from the phase request."
            )
    batch_contracts = family.get("ordered_batch_contracts")
    if (
        not isinstance(batch_contracts, list)
        or metadata.get("batch_count") != family.get("batch_count")
        or len(value) != family.get("batch_count")
        or len(batch_contracts) != len(value)
    ):
        raise Lfm2ChildReplayPhaseAuthenticationError(
            "Child replay batch count differs from the phase request."
        )
    forward_count = family.get("canonical_prefix_forward_count")
    if (
        isinstance(forward_count, bool)
        or not isinstance(forward_count, int)
        or forward_count <= 0
        or metadata.get("forward_counts")
        != {"child": forward_count, "parent": 0}
        or metadata.get("replay_benchmark")
        != {
            "batch_count": len(value),
            "phase_count": forward_count,
            "actual_child_forwards": forward_count,
            "canonical_prefix_child_forwards": forward_count,
            "saved_child_forwards": 0,
        }
        or metadata.get("causal_equivalence")
        != {"reference_supplied": False, "tolerance_equivalent": None}
        or metadata.get("runtime")
        != {
            "tokenizer_accessed": False,
            "parent_model_supplied": False,
            "parent_forward_count": 0,
            "disk_artifacts_written": 0,
            "intermediate_cache": "memory_only",
            "child_state_preserved": True,
        }
    ):
        raise Lfm2ChildReplayPhaseAuthenticationError(
            "Child replay runtime or forward accounting changed."
        )
    ordered = metadata.get("ordered_batches")
    if not isinstance(ordered, list) or len(ordered) != len(value):
        raise Lfm2ChildReplayPhaseAuthenticationError(
            "Child replay ordered-batch chain is missing."
        )
    selection = family["selection"]
    selected_indices = tuple(selection["selected_layer_indices"])
    profile_value = selection.get("child_evidence_profile")
    try:
        profile = (
            None
            if profile_value is None
            else Lfm2ChildEvidenceProfile.from_dict(profile_value)
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2ChildReplayPhaseAuthenticationError(
            "Child replay evidence profile is invalid."
        ) from exc
    torch = _trace._torch()
    for index, (batch, contract, summary) in enumerate(
        zip(value.batches, batch_contracts, ordered, strict=True)
    ):
        if not isinstance(contract, Mapping) or not isinstance(summary, Mapping):
            raise Lfm2ChildReplayPhaseAuthenticationError(
                "Child replay batch identity chain is malformed."
            )
        record = batch.record
        try:
            _trace._verify_seal(
                record,
                "replay_batch_sha256",
                f"child replay batch {index}",
            )
        except Lfm2CircuitTraceCacheError as exc:
            raise Lfm2ChildReplayPhaseAuthenticationError(
                f"Child replay batch {index} self-seal is invalid."
            ) from exc
        if (
            record.get("schema") != CHILD_REPLAY_BATCH_SCHEMA
            or batch.index != index
            or record.get("batch_index") != index
            or batch.input_sha256 != contract.get("input_sha256")
            or record.get("input_sha256") != contract.get("input_sha256")
            or record.get("selected_layer_indices") != list(selected_indices)
            or record.get("phase_order") != contract.get("phase_order")
            or record.get("phase_rows") != contract.get("phase_rows")
            or record.get("child_evidence_profile") != profile_value
            or record.get("child_evidence_profile_sha256")
            != selection.get("child_evidence_profile_sha256")
        ):
            raise Lfm2ChildReplayPhaseAuthenticationError(
                f"Child replay batch {index} differs from its request."
            )
        tensors = batch.tensors
        inventory = record.get("tensors")
        names = tuple(sorted(tensors))
        if (
            not isinstance(inventory, Mapping)
            or record.get("tensor_count") != len(names)
            or tuple(sorted(inventory)) != names
        ):
            raise Lfm2ChildReplayPhaseAuthenticationError(
                f"Child replay batch {index} tensor inventory changed."
            )
        for name in names:
            tensor = tensors[name]
            if (
                not torch.is_tensor(tensor)
                or tensor.layout != torch.strided
                or tensor.device.type != "cpu"
                or tensor.requires_grad
                or tensor.grad_fn is not None
                or _trace._tensor_descriptor(torch, tensor)
                != dict(inventory[name])
                or (
                    (tensor.is_floating_point() or tensor.is_complex())
                    and not bool(torch.isfinite(tensor).all().item())
                )
            ):
                raise Lfm2ChildReplayPhaseAuthenticationError(
                    f"Child replay tensor {name!r} differs from its descriptor."
                )
        input_descriptors = contract.get("input_tensor_descriptors")
        if not isinstance(input_descriptors, Mapping) or any(
            name not in tensors
            or dict(inventory[name]) != dict(descriptor)
            for name, descriptor in input_descriptors.items()
        ):
            raise Lfm2ChildReplayPhaseAuthenticationError(
                f"Child replay batch {index} input tensors changed."
            )
        capture = dict(record.get("capture", {}))
        diagnostics = capture.get("phase_diagnostics")
        if isinstance(diagnostics, Mapping):
            # Canonical JSON sorts object keys, whereas the original in-memory
            # validator also checks the causal phase insertion order. Restore
            # that authenticated order from the sealed phase_order list before
            # applying the same validator to a persisted record.
            capture["phase_diagnostics"] = {
                phase: diagnostics[phase]
                for phase in record["phase_order"]
                if phase in diagnostics
            }
        try:
            _trace._validate_child_replay_capture(
                tensors,
                capture,
                selected_indices,
                tuple(record["phase_order"]),
                CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
                profile,
            )
        except Lfm2CircuitTraceCacheError as exc:
            raise Lfm2ChildReplayPhaseAuthenticationError(
                f"Child replay batch {index} capture contract changed."
            ) from exc
        expected_summary = {
            "batch_index": index,
            "input_sha256": batch.input_sha256,
            "replay_batch_sha256": record["replay_batch_sha256"],
            "phase_order": record["phase_order"],
            "child_evidence_profile_sha256": record.get(
                "child_evidence_profile_sha256"
            ),
        }
        if dict(summary) != expected_summary:
            raise Lfm2ChildReplayPhaseAuthenticationError(
                f"Child replay batch {index} metadata chain changed."
            )
    return _sha256(metadata.get("replay_sha256"), "replay_sha256")


def _batch_directory(root: Path, index: int) -> Path:
    return root / _BATCH_DIRECTORY / f"batch-{index:06d}"


class Lfm2ChildReplayPhaseCodec:
    """Canonical JSON plus per-batch SafeTensors codec for LFM2 replay."""

    codec_id = LFM2_CHILD_REPLAY_PHASE_CODEC_ID

    def authenticate(
        self,
        value: Lfm2ChildReplay,
        identity: DynamicChildReplayPhaseIdentity,
    ) -> str:
        return _authenticate_replay(value, identity)

    def write(
        self,
        value: Lfm2ChildReplay,
        payload_path: Path,
        identity: DynamicChildReplayPhaseIdentity,
    ) -> Mapping[str, Any]:
        replay_sha256 = self.authenticate(value, identity)
        try:
            from safetensors.torch import save_file
        except ImportError as exc:  # pragma: no cover - optional HF dependency
            raise RuntimeError(
                "Persistent LFM2 child replay requires safetensors."
            ) from exc
        _write_json(payload_path / _REQUEST_FILENAME, identity.as_dict())
        _write_json(payload_path / _REPLAY_FILENAME, value.metadata)
        batch_root = payload_path / _BATCH_DIRECTORY
        batch_root.mkdir()
        ordered_batches = []
        for batch in value.batches:
            directory = _batch_directory(payload_path, batch.index)
            directory.mkdir()
            record_path = directory / _BATCH_RECORD_FILENAME
            tensor_path = directory / _BATCH_TENSOR_FILENAME
            _write_json(record_path, batch.record)
            tensors = {
                name: tensor.detach().to(device="cpu").contiguous().clone()
                for name, tensor in sorted(batch.tensors.items())
            }
            metadata = {
                "schema": LFM2_CHILD_REPLAY_SAFETENSORS_SCHEMA,
                "codec_id": self.codec_id,
                "request_sha256": identity.request_sha256,
                "replay_sha256": replay_sha256,
                "batch_index": str(batch.index),
                "replay_batch_sha256": batch.record["replay_batch_sha256"],
                "tensor_count": str(len(tensors)),
            }
            try:
                save_file(tensors, tensor_path, metadata=metadata)
            except Exception as exc:
                raise Lfm2ChildReplayPhaseCacheError(
                    f"Could not write replay batch {batch.index} tensors."
                ) from exc
            ordered_batches.append(
                {
                    "batch_index": batch.index,
                    "record_sha256": _file_sha256(record_path),
                    "replay_batch_sha256": batch.record[
                        "replay_batch_sha256"
                    ],
                    "tensor_count": len(tensors),
                    "tensor_file_sha256": _file_sha256(tensor_path),
                }
            )
        receipt_body = {
            "schema": LFM2_CHILD_REPLAY_PHASE_RECEIPT_SCHEMA,
            "codec_id": self.codec_id,
            "phase_key_sha256": identity.phase_key.phase_key_sha256,
            "request_sha256": identity.request_sha256,
            "replay_sha256": replay_sha256,
            "batch_count": len(value),
            "ordered_batches": ordered_batches,
            "capture_timing_serialized": False,
        }
        receipt = {
            **receipt_body,
            "receipt_sha256": _json_sha256(
                receipt_body,
                "LFM2 child replay phase receipt",
            ),
        }
        _write_json(payload_path / _RECEIPT_FILENAME, receipt)
        return {
            "schema": LFM2_CHILD_REPLAY_PHASE_RECEIPT_SCHEMA,
            "receipt_sha256": receipt["receipt_sha256"],
            "replay_sha256": replay_sha256,
            "batch_count": len(value),
        }

    def load(
        self,
        entry: VerifiedFoldPhaseCacheEntry,
        identity: DynamicChildReplayPhaseIdentity,
    ) -> Lfm2ChildReplay:
        payload = entry.payload_path
        expected_root = {
            _REQUEST_FILENAME,
            _REPLAY_FILENAME,
            _RECEIPT_FILENAME,
            _BATCH_DIRECTORY,
        }
        if {path.name for path in payload.iterdir()} != expected_root:
            raise Lfm2ChildReplayPhaseAuthenticationError(
                "LFM2 child replay payload root entries changed."
            )
        if _read_json(payload / _REQUEST_FILENAME) != identity.as_dict():
            raise Lfm2ChildReplayPhaseAuthenticationError(
                "Persisted LFM2 child replay request differs from the lookup."
            )
        metadata = _read_json(payload / _REPLAY_FILENAME)
        receipt = _verify_seal(
            _read_json(payload / _RECEIPT_FILENAME),
            "receipt_sha256",
            "LFM2 child replay phase receipt",
        )
        family = _family_request(identity)
        batch_count = family.get("batch_count")
        if (
            receipt.get("schema") != LFM2_CHILD_REPLAY_PHASE_RECEIPT_SCHEMA
            or receipt.get("codec_id") != self.codec_id
            or receipt.get("phase_key_sha256")
            != identity.phase_key.phase_key_sha256
            or receipt.get("request_sha256") != identity.request_sha256
            or receipt.get("replay_sha256") != metadata.get("replay_sha256")
            or receipt.get("batch_count") != batch_count
            or receipt.get("capture_timing_serialized") is not False
        ):
            raise Lfm2ChildReplayPhaseAuthenticationError(
                "LFM2 child replay phase receipt differs from its request."
            )
        completion_codec_result = entry.result.get("codec_result")
        expected_codec_result = {
            "schema": LFM2_CHILD_REPLAY_PHASE_RECEIPT_SCHEMA,
            "receipt_sha256": receipt["receipt_sha256"],
            "replay_sha256": receipt["replay_sha256"],
            "batch_count": batch_count,
        }
        if completion_codec_result != expected_codec_result:
            raise Lfm2ChildReplayPhaseAuthenticationError(
                "LFM2 codec receipt differs from the phase completion."
            )
        batch_root = payload / _BATCH_DIRECTORY
        if not batch_root.is_dir() or batch_root.is_symlink():
            raise Lfm2ChildReplayPhaseAuthenticationError(
                "LFM2 child replay batch root is invalid."
            )
        expected_directories = {
            f"batch-{index:06d}" for index in range(batch_count)
        }
        if {path.name for path in batch_root.iterdir()} != expected_directories:
            raise Lfm2ChildReplayPhaseAuthenticationError(
                "LFM2 child replay batch directory inventory changed."
            )
        ordered_receipt = receipt.get("ordered_batches")
        if not isinstance(ordered_receipt, list) or len(ordered_receipt) != batch_count:
            raise Lfm2ChildReplayPhaseAuthenticationError(
                "LFM2 child replay receipt batch chain is malformed."
            )
        batches = []
        for index in range(batch_count):
            directory = _batch_directory(payload, index)
            if (
                not directory.is_dir()
                or directory.is_symlink()
                or {path.name for path in directory.iterdir()}
                != {_BATCH_RECORD_FILENAME, _BATCH_TENSOR_FILENAME}
            ):
                raise Lfm2ChildReplayPhaseAuthenticationError(
                    f"LFM2 child replay batch {index} payload layout changed."
                )
            record_path = directory / _BATCH_RECORD_FILENAME
            tensor_path = directory / _BATCH_TENSOR_FILENAME
            record = _read_json(record_path)
            expected_tensor_metadata = {
                "schema": LFM2_CHILD_REPLAY_SAFETENSORS_SCHEMA,
                "codec_id": self.codec_id,
                "request_sha256": identity.request_sha256,
                "replay_sha256": receipt["replay_sha256"],
                "batch_index": str(index),
                "replay_batch_sha256": record.get("replay_batch_sha256"),
                "tensor_count": str(record.get("tensor_count")),
            }
            try:
                from safetensors import safe_open

                with safe_open(
                    tensor_path,
                    framework="pt",
                    device="cpu",
                ) as handle:
                    if handle.metadata() != expected_tensor_metadata:
                        raise Lfm2ChildReplayPhaseAuthenticationError(
                            f"Replay batch {index} SafeTensors metadata changed."
                        )
                    names = tuple(sorted(handle.keys()))
                    inventory = record.get("tensors")
                    if (
                        not isinstance(inventory, Mapping)
                        or tuple(sorted(inventory)) != names
                    ):
                        raise Lfm2ChildReplayPhaseAuthenticationError(
                            f"Replay batch {index} SafeTensors keys changed."
                        )
                    tensors = {
                        name: handle.get_tensor(name)
                        .detach()
                        .to(device="cpu")
                        .contiguous()
                        .clone()
                        for name in names
                    }
            except Lfm2ChildReplayPhaseAuthenticationError:
                raise
            except Exception as exc:
                raise Lfm2ChildReplayPhaseAuthenticationError(
                    f"Could not load replay batch {index} SafeTensors."
                ) from exc
            row = ordered_receipt[index]
            expected_row = {
                "batch_index": index,
                "record_sha256": _file_sha256(record_path),
                "replay_batch_sha256": record.get("replay_batch_sha256"),
                "tensor_count": len(tensors),
                "tensor_file_sha256": _file_sha256(tensor_path),
            }
            if row != expected_row:
                raise Lfm2ChildReplayPhaseAuthenticationError(
                    f"LFM2 child replay receipt batch {index} changed."
                )
            batches.append(
                Lfm2ChildReplayBatch(
                    index=index,
                    input_sha256=str(record.get("input_sha256")),
                    _tensors=MappingProxyType(tensors),
                    _record=MappingProxyType(record),
                )
            )
        replay = Lfm2ChildReplay(
            _batches=tuple(batches),
            _metadata=MappingProxyType(metadata),
            _capture_timing=MappingProxyType({}),
        )
        self.authenticate(replay, identity)
        return replay


def load_or_capture_lfm2_child_replay_phase(
    context: DynamicChildReplayPhaseCacheContext,
    child_model: Any,
    replay_batches: Iterable[
        Mapping[str, Any] | Lfm2CircuitTraceBatch | Lfm2ChildReplayBatch
    ],
    *,
    selected_layer_indices: Sequence[int],
    config: Lfm2CircuitTraceCaptureConfig | None = None,
    input_device: Any | None = None,
    replay_capture_mode: str = CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
    child_evidence_profile: Lfm2ChildEvidenceProfile | None = None,
    replay_devices: Sequence[Any] | Lfm2ReplayShardingConfig | None = None,
    capture_provider: Callable[[], Lfm2ChildReplay] | None = None,
) -> AuthenticatedDynamicChildReplayPhase[Lfm2ChildReplay]:
    """Load or capture one exact canonical LFM2 child replay phase.

    The public surface deliberately accepts no parent model and no tokenizer.
    ``replay_devices`` controls cold execution only and never enters the key.
    """

    if not isinstance(context, DynamicChildReplayPhaseCacheContext):
        raise TypeError("context must be DynamicChildReplayPhaseCacheContext.")
    batches = tuple(replay_batches)
    request = describe_lfm2_child_replay_phase(
        child_model,
        batches,
        selected_layer_indices=selected_layer_indices,
        config=config,
        input_device=input_device,
        replay_capture_mode=replay_capture_mode,
        child_evidence_profile=child_evidence_profile,
    )
    if capture_provider is not None and not callable(capture_provider):
        raise TypeError("capture_provider must be callable or None.")

    def capture() -> Lfm2ChildReplay:
        if capture_provider is not None:
            return capture_provider()
        return capture_lfm2_child_replay(
            child_model,
            batches,
            selected_layer_indices=selected_layer_indices,
            config=config,
            input_device=input_device,
            replay_capture_mode=replay_capture_mode,
            child_evidence_profile=child_evidence_profile,
            replay_devices=replay_devices,
        )

    result = load_or_build_dynamic_child_replay_phase(
        context,
        request.identity,
        codec=Lfm2ChildReplayPhaseCodec(),
        producer=capture,
    )
    final_child = fingerprint_lfm2_parent(child_model).as_dict()
    if final_child != request.family_request["child"]:
        raise Lfm2ChildReplayPhaseAuthenticationError(
            "Live child state changed while its replay phase was resolved."
        )
    return result


__all__ = [
    "LFM2_CHILD_REPLAY_PHASE",
    "LFM2_CHILD_REPLAY_PHASE_CODEC_ID",
    "LFM2_CHILD_REPLAY_PHASE_RECEIPT_SCHEMA",
    "LFM2_CHILD_REPLAY_PHASE_REQUEST_SCHEMA",
    "LFM2_CHILD_REPLAY_SAFETENSORS_SCHEMA",
    "Lfm2ChildReplayPhaseAuthenticationError",
    "Lfm2ChildReplayPhaseCacheError",
    "Lfm2ChildReplayPhaseCodec",
    "Lfm2ChildReplayPhaseRequest",
    "describe_lfm2_child_replay_phase",
    "lfm2_child_replay_phase_codec_implementation_sha256",
    "load_or_capture_lfm2_child_replay_phase",
]
