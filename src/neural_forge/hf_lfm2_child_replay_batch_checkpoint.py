"""Authenticated per-batch resume points for canonical LFM2 child replay.

The complete replay remains owned by :mod:`hf_lfm2_child_replay_phase_cache`.
This module adds subordinate immutable phase entries, one for each completely
sealed batch.  A subordinate entry is never a complete replay: it has its own
phase name, codec, request schema, and receipt, and it binds the exact complete
phase request that will eventually consume it.

Keeping this implementation outside the complete replay codec is deliberate.
The already established complete-phase content key and codec source identity
remain byte-for-byte unchanged while interrupted captures gain resumability.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from . import dynamic_child_replay_phase_cache as _generic_cache
from . import dynamic_fold_phase_cache as _fold_phase
from . import hf_lfm2_child_replay_phase_cache as _complete_phase
from . import hf_lfm2_circuit_trace_cache as _trace
from .dynamic_child_replay_phase_cache import (
    DYNAMIC_CHILD_REPLAY_PHASE_IDENTITY_SCHEMA,
    AuthenticatedDynamicChildReplayPhase,
    DynamicChildReplayPhaseCacheContext,
    DynamicChildReplayPhaseIdentity,
)
from .dynamic_fold_phase_cache import (
    FoldPhaseCacheAlreadyCompleteError,
    FoldPhaseCacheIntegrityError,
    FoldPhaseCachePublicationError,
    FoldPhaseKey,
    VerifiedFoldPhaseCacheEntry,
)
from .hf_lfm2_child_replay_phase_cache import (
    LFM2_CHILD_REPLAY_PHASE,
    Lfm2ChildReplayPhaseAuthenticationError,
    Lfm2ChildReplayPhaseCodec,
    Lfm2ChildReplayPhaseRequest,
    describe_lfm2_child_replay_phase,
)
from .hf_lfm2_circuit_trace_cache import (
    CHILD_REPLAY_BATCH_SCHEMA,
    CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
    CHILD_REPLAY_SCHEMA,
    Lfm2ChildEvidenceProfile,
    Lfm2ChildReplay,
    Lfm2ChildReplayBatch,
    Lfm2CircuitTraceBatch,
    Lfm2CircuitTraceCacheError,
    Lfm2CircuitTraceCaptureConfig,
    fingerprint_lfm2_parent,
)
from .hf_lfm2_replay_sharding import (
    Lfm2ReplayShardingConfig,
    Lfm2ReplayShardingError,
    plan_replay_shards,
)

LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_PHASE = "lfm2-child-replay-batch-checkpoint-v1"
LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_CODEC_ID = (
    "lfm2-child-replay-batch-checkpoint-codec-v1"
)
LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_REQUEST_SCHEMA = (
    "neural-forge-lfm2-child-replay-batch-checkpoint-request-v1"
)
LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-child-replay-batch-checkpoint-receipt-v1"
)
LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_SAFETENSORS_SCHEMA = (
    "neural-forge-lfm2-child-replay-batch-checkpoint-tensors-v1"
)
LFM2_CHILD_REPLAY_BATCH_RESUME_TIMING_SCHEMA = (
    "neural-forge-lfm2-child-replay-batch-resume-timing-v1"
)

LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_LOOKUP_STAGE = (
    "lfm2_child_replay_batch_checkpoint_lookup"
)
LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_HIT_STAGE = "lfm2_child_replay_batch_checkpoint_hit"
LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_MISS_STAGE = (
    "lfm2_child_replay_batch_checkpoint_miss"
)
LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_PUBLISHED_STAGE = (
    "lfm2_child_replay_batch_checkpoint_published"
)

_REQUEST_FILENAME = "request.json"
_RECORD_FILENAME = "record.json"
_RECEIPT_FILENAME = "receipt.json"
_TENSOR_FILENAME = "tensors.safetensors"


class Lfm2ChildReplayBatchCheckpointError(RuntimeError):
    """A canonical LFM2 batch checkpoint request is unsupported."""


class Lfm2ChildReplayBatchCheckpointAuthenticationError(FoldPhaseCacheIntegrityError):
    """A batch checkpoint failed request, record, or tensor authentication."""


def _progress(
    context: DynamicChildReplayPhaseCacheContext,
    stage: str,
    **details: Any,
) -> None:
    if context.progress is not None:
        context.progress(stage, MappingProxyType(dict(details)))


def _checkpoint_implementation_sha256() -> str:
    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-child-replay-batch-checkpoint-v1\0")
    digest.update(Path(__file__).resolve(strict=True).read_bytes())
    return digest.hexdigest()


def _batch_checkpoint_family(
    identity: DynamicChildReplayPhaseIdentity,
) -> dict[str, Any]:
    if (
        identity.codec_id != LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_CODEC_ID
        or identity.phase_key.phase_name != LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_PHASE
    ):
        raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
            "LFM2 batch checkpoint codec received a foreign identity."
        )
    request = identity.as_dict()
    request.pop("request_sha256")
    family = request.get("family_request")
    if not isinstance(family, Mapping):
        raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
            "LFM2 batch checkpoint identity has no family request."
        )
    result = _complete_phase._verify_seal(
        family,
        "family_request_sha256",
        "LFM2 child replay batch checkpoint family request",
    )
    if (
        result.get("schema") != LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_REQUEST_SCHEMA
        or result.get("codec_id") != LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_CODEC_ID
        or result.get("complete_phase_name") != LFM2_CHILD_REPLAY_PHASE
    ):
        raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
            "LFM2 batch checkpoint family request contract changed."
        )
    return result


@dataclass(frozen=True, slots=True)
class Lfm2ChildReplayBatchCheckpointRequest:
    """Content address for one sealed batch subordinate to a full replay."""

    identity: DynamicChildReplayPhaseIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.identity, DynamicChildReplayPhaseIdentity):
            raise TypeError("identity must be DynamicChildReplayPhaseIdentity.")
        _batch_checkpoint_family(self.identity)

    @property
    def phase_key(self) -> FoldPhaseKey:
        return self.identity.phase_key

    @property
    def request_sha256(self) -> str:
        return self.identity.request_sha256

    @property
    def family_request(self) -> dict[str, Any]:
        return _batch_checkpoint_family(self.identity)

    @property
    def batch_index(self) -> int:
        return int(self.family_request["batch_index"])

    def as_dict(self) -> dict[str, Any]:
        return self.identity.as_dict()


def describe_lfm2_child_replay_batch_checkpoint(
    complete_request: Lfm2ChildReplayPhaseRequest,
    batch_index: int,
) -> Lfm2ChildReplayBatchCheckpointRequest:
    """Derive one batch address from an exact complete replay request."""

    if not isinstance(complete_request, Lfm2ChildReplayPhaseRequest):
        raise TypeError("complete_request must be Lfm2ChildReplayPhaseRequest.")
    if isinstance(batch_index, bool) or not isinstance(batch_index, int):
        raise TypeError("batch_index must be an integer.")
    complete_family = complete_request.family_request
    contracts = complete_family.get("ordered_batch_contracts")
    if not isinstance(contracts, list) or not 0 <= batch_index < len(contracts):
        raise Lfm2ChildReplayBatchCheckpointError(
            "batch_index is outside the complete replay request."
        )
    batch_contract = contracts[batch_index]
    if (
        not isinstance(batch_contract, Mapping)
        or batch_contract.get("batch_index") != batch_index
    ):
        raise Lfm2ChildReplayBatchCheckpointError(
            "The complete replay batch order is malformed."
        )
    batch_contract = _complete_phase._canonical_mapping(
        batch_contract,
        "LFM2 child replay batch contract",
    )
    ordered_contracts_sha256 = _complete_phase._json_sha256(
        contracts,
        "ordered LFM2 child replay batch contracts",
    )
    batch_contract_sha256 = _complete_phase._json_sha256(
        batch_contract,
        "LFM2 child replay batch contract",
    )
    implementation_sha256 = _checkpoint_implementation_sha256()
    child = complete_family["child"]
    selection = complete_family["selection"]
    execution = complete_family["execution"]
    numeric_policy = complete_family["numeric_execution_policy"]
    family_body = {
        "schema": LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_REQUEST_SCHEMA,
        "codec_id": LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_CODEC_ID,
        "checkpoint_implementation_sha256": implementation_sha256,
        "complete_phase_name": complete_request.phase_key.phase_name,
        "complete_phase_key_sha256": (complete_request.phase_key.phase_key_sha256),
        "complete_request_sha256": complete_request.request_sha256,
        "complete_family_request_sha256": complete_family["family_request_sha256"],
        "complete_codec_id": complete_family["codec_id"],
        "complete_codec_implementation_sha256": complete_family[
            "codec_implementation_sha256"
        ],
        "child": child,
        "corpus_sha256": complete_family["corpus"]["corpus_sha256"],
        "collector_implementation": complete_family["implementation"],
        "layer_topology_sha256": complete_family["layer_topology"][
            "layer_topology_sha256"
        ],
        "selection": selection,
        "execution": execution,
        "numeric_execution_policy": numeric_policy,
        "batch_count": complete_family["batch_count"],
        "batch_index": batch_index,
        "batch_contract": batch_contract,
        "batch_contract_sha256": batch_contract_sha256,
        "ordered_batch_contracts_sha256": ordered_contracts_sha256,
        "parent_model_bound": False,
        "tokenizer_bound": False,
        "training_used": False,
    }
    family_request = {
        **family_body,
        "family_request_sha256": _complete_phase._json_sha256(
            family_body,
            "LFM2 child replay batch checkpoint family request",
        ),
    }
    phase_key = FoldPhaseKey(
        phase_name=LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_PHASE,
        dependency_sha256s={
            "complete-phase": complete_request.phase_key.phase_key_sha256,
            "complete-request": complete_request.request_sha256,
            "complete-family-request": complete_family["family_request_sha256"],
            "child-identity": child["identity_sha256"],
            "child-state": child["state_dict_sha256"],
            "child-runtime-buffers": child["runtime_buffer_sha256"],
            "corpus": complete_family["corpus"]["corpus_sha256"],
            "collector-implementation": complete_family["implementation"][
                "implementation_sha256"
            ],
            "layer-topology": complete_family["layer_topology"][
                "layer_topology_sha256"
            ],
            "selection": selection["selection_sha256"],
            "execution": execution["execution_sha256"],
            "numeric-policy": numeric_policy["replay_numeric_policy_sha256"],
            "input": batch_contract["input_sha256"],
            "batch-contract": batch_contract_sha256,
            "batch-order": ordered_contracts_sha256,
            "checkpoint-source": implementation_sha256,
            "complete-codec-source": complete_family["codec_implementation_sha256"],
        },
        numeric_contract={
            "schema": ("neural-forge-lfm2-child-replay-batch-checkpoint-numeric-v1"),
            "batch_index": batch_index,
            "capture": selection["capture"],
            "execution": execution,
            "numeric_execution_policy": numeric_policy,
            "replay_capture_mode": selection["replay_capture_mode"],
        },
        abi={
            "schema": ("neural-forge-lfm2-child-replay-batch-checkpoint-abi-v1"),
            "codec_id": LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_CODEC_ID,
            "checkpoint_implementation_sha256": implementation_sha256,
            "payload": "canonical-json-plus-safetensors",
            "replay_batch_schema": CHILD_REPLAY_BATCH_SCHEMA,
        },
    )
    generic_request = {
        "schema": DYNAMIC_CHILD_REPLAY_PHASE_IDENTITY_SCHEMA,
        "phase_key_sha256": phase_key.phase_key_sha256,
        "codec_id": LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_CODEC_ID,
        "family": "lfm2",
        "family_request": family_request,
    }
    return Lfm2ChildReplayBatchCheckpointRequest(
        DynamicChildReplayPhaseIdentity(
            phase_key=phase_key,
            request=generic_request,
            codec_id=LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_CODEC_ID,
        )
    )


def _authenticate_batch(
    value: Any,
    identity: DynamicChildReplayPhaseIdentity,
) -> str:
    if not isinstance(value, Lfm2ChildReplayBatch):
        raise TypeError("LFM2 batch checkpoint codec requires Lfm2ChildReplayBatch.")
    family = _batch_checkpoint_family(identity)
    contract = family.get("batch_contract")
    if not isinstance(contract, Mapping):
        raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
            "LFM2 batch checkpoint has no batch contract."
        )
    if _complete_phase._json_sha256(
        contract,
        "LFM2 child replay batch contract",
    ) != family.get("batch_contract_sha256"):
        raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
            "LFM2 batch checkpoint contract seal changed."
        )
    record = value.record
    try:
        _trace._verify_seal(
            record,
            "replay_batch_sha256",
            f"child replay batch {family['batch_index']}",
        )
    except Lfm2CircuitTraceCacheError as exc:
        raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
            "LFM2 batch checkpoint record self-seal is invalid."
        ) from exc
    expected_fields = {
        "schema",
        "batch_index",
        "input_sha256",
        "selected_layer_indices",
        "phase_order",
        "phase_rows",
        "child_evidence_profile",
        "child_evidence_profile_sha256",
        "capture",
        "tensor_count",
        "tensors",
        "replay_batch_sha256",
    }
    selection = family["selection"]
    selected_indices = tuple(selection["selected_layer_indices"])
    profile_value = selection.get("child_evidence_profile")
    batch_index = int(family["batch_index"])
    if (
        set(record) != expected_fields
        or record.get("schema") != CHILD_REPLAY_BATCH_SCHEMA
        or value.index != batch_index
        or record.get("batch_index") != batch_index
        or value.input_sha256 != contract.get("input_sha256")
        or record.get("input_sha256") != contract.get("input_sha256")
        or record.get("selected_layer_indices") != list(selected_indices)
        or record.get("phase_order") != contract.get("phase_order")
        or record.get("phase_rows") != contract.get("phase_rows")
        or record.get("child_evidence_profile") != profile_value
        or record.get("child_evidence_profile_sha256")
        != selection.get("child_evidence_profile_sha256")
    ):
        raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
            "LFM2 batch checkpoint record differs from its exact request."
        )
    tensors = value.tensors
    inventory = record.get("tensors")
    names = tuple(sorted(tensors))
    if (
        not isinstance(inventory, Mapping)
        or record.get("tensor_count") != len(names)
        or tuple(sorted(inventory)) != names
    ):
        raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
            "LFM2 batch checkpoint tensor inventory changed."
        )
    torch = _trace._torch()
    for name in names:
        tensor = tensors[name]
        descriptor = inventory[name]
        if (
            not torch.is_tensor(tensor)
            or not isinstance(descriptor, Mapping)
            or tensor.layout != torch.strided
            or tensor.device.type != "cpu"
            or tensor.requires_grad
            or tensor.grad_fn is not None
            or _trace._tensor_descriptor(torch, tensor) != dict(descriptor)
            or (
                (tensor.is_floating_point() or tensor.is_complex())
                and not bool(torch.isfinite(tensor).all().item())
            )
        ):
            raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
                f"LFM2 batch checkpoint tensor {name!r} changed."
            )
    input_descriptors = contract.get("input_tensor_descriptors")
    if not isinstance(input_descriptors, Mapping) or any(
        name not in tensors or dict(inventory[name]) != dict(descriptor)
        for name, descriptor in input_descriptors.items()
    ):
        raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
            "LFM2 batch checkpoint input tensors changed."
        )
    try:
        profile = (
            None
            if profile_value is None
            else Lfm2ChildEvidenceProfile.from_dict(profile_value)
        )
        capture = dict(record["capture"])
        diagnostics = capture.get("phase_diagnostics")
        if isinstance(diagnostics, Mapping):
            capture["phase_diagnostics"] = {
                phase: diagnostics[phase]
                for phase in record["phase_order"]
                if phase in diagnostics
            }
        _trace._validate_child_replay_capture(
            tensors,
            capture,
            selected_indices,
            tuple(record["phase_order"]),
            CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
            profile,
        )
    except (Lfm2CircuitTraceCacheError, TypeError, ValueError) as exc:
        raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
            "LFM2 batch checkpoint capture contract changed."
        ) from exc
    return _complete_phase._sha256(
        record.get("replay_batch_sha256"),
        "replay_batch_sha256",
    )


class Lfm2ChildReplayBatchCheckpointCodec:
    """Canonical JSON and SafeTensors codec for one replay batch."""

    codec_id = LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_CODEC_ID

    def authenticate(
        self,
        value: Lfm2ChildReplayBatch,
        identity: DynamicChildReplayPhaseIdentity,
    ) -> str:
        return _authenticate_batch(value, identity)

    def write(
        self,
        value: Lfm2ChildReplayBatch,
        payload_path: Path,
        identity: DynamicChildReplayPhaseIdentity,
    ) -> Mapping[str, Any]:
        replay_batch_sha256 = self.authenticate(value, identity)
        family = _batch_checkpoint_family(identity)
        try:
            from safetensors.torch import save_file
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Persistent LFM2 batch checkpoints require safetensors."
            ) from exc
        _complete_phase._write_json(
            payload_path / _REQUEST_FILENAME,
            identity.as_dict(),
        )
        record_path = payload_path / _RECORD_FILENAME
        tensor_path = payload_path / _TENSOR_FILENAME
        _complete_phase._write_json(record_path, value.record)
        tensors = {
            name: tensor.detach().to(device="cpu").contiguous().clone()
            for name, tensor in sorted(value.tensors.items())
        }
        tensor_metadata = {
            "schema": LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_SAFETENSORS_SCHEMA,
            "codec_id": self.codec_id,
            "request_sha256": identity.request_sha256,
            "complete_phase_key_sha256": family["complete_phase_key_sha256"],
            "batch_index": str(value.index),
            "replay_batch_sha256": replay_batch_sha256,
            "tensor_count": str(len(tensors)),
        }
        try:
            save_file(tensors, tensor_path, metadata=tensor_metadata)
        except Exception as exc:
            raise Lfm2ChildReplayBatchCheckpointError(
                f"Could not write replay batch {value.index} tensors."
            ) from exc
        receipt_body = {
            "schema": LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_RECEIPT_SCHEMA,
            "codec_id": self.codec_id,
            "phase_key_sha256": identity.phase_key.phase_key_sha256,
            "request_sha256": identity.request_sha256,
            "complete_phase_key_sha256": family["complete_phase_key_sha256"],
            "batch_index": value.index,
            "replay_batch_sha256": replay_batch_sha256,
            "tensor_count": len(tensors),
            "record_file_sha256": _complete_phase._file_sha256(record_path),
            "tensor_file_sha256": _complete_phase._file_sha256(tensor_path),
        }
        receipt = {
            **receipt_body,
            "receipt_sha256": _complete_phase._json_sha256(
                receipt_body,
                "LFM2 child replay batch checkpoint receipt",
            ),
        }
        _complete_phase._write_json(
            payload_path / _RECEIPT_FILENAME,
            receipt,
        )
        return {
            "schema": LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_RECEIPT_SCHEMA,
            "receipt_sha256": receipt["receipt_sha256"],
            "replay_batch_sha256": replay_batch_sha256,
            "batch_index": value.index,
            "tensor_count": len(tensors),
        }

    def load(
        self,
        entry: VerifiedFoldPhaseCacheEntry,
        identity: DynamicChildReplayPhaseIdentity,
    ) -> Lfm2ChildReplayBatch:
        payload = entry.payload_path
        expected_files = {
            _REQUEST_FILENAME,
            _RECORD_FILENAME,
            _RECEIPT_FILENAME,
            _TENSOR_FILENAME,
        }
        if {path.name for path in payload.iterdir()} != expected_files:
            raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
                "LFM2 batch checkpoint payload layout changed."
            )
        if (
            _complete_phase._read_json(payload / _REQUEST_FILENAME)
            != identity.as_dict()
        ):
            raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
                "Persisted LFM2 batch checkpoint request differs from lookup."
            )
        family = _batch_checkpoint_family(identity)
        record_path = payload / _RECORD_FILENAME
        tensor_path = payload / _TENSOR_FILENAME
        record = _complete_phase._read_json(record_path)
        receipt = _complete_phase._verify_seal(
            _complete_phase._read_json(payload / _RECEIPT_FILENAME),
            "receipt_sha256",
            "LFM2 child replay batch checkpoint receipt",
        )
        batch_index = int(family["batch_index"])
        expected_receipt_fields = {
            "schema",
            "codec_id",
            "phase_key_sha256",
            "request_sha256",
            "complete_phase_key_sha256",
            "batch_index",
            "replay_batch_sha256",
            "tensor_count",
            "record_file_sha256",
            "tensor_file_sha256",
            "receipt_sha256",
        }
        if (
            set(receipt) != expected_receipt_fields
            or receipt.get("schema")
            != LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_RECEIPT_SCHEMA
            or receipt.get("codec_id") != self.codec_id
            or receipt.get("phase_key_sha256") != identity.phase_key.phase_key_sha256
            or receipt.get("request_sha256") != identity.request_sha256
            or receipt.get("complete_phase_key_sha256")
            != family["complete_phase_key_sha256"]
            or receipt.get("batch_index") != batch_index
            or receipt.get("replay_batch_sha256") != record.get("replay_batch_sha256")
            or receipt.get("tensor_count") != record.get("tensor_count")
            or receipt.get("record_file_sha256")
            != _complete_phase._file_sha256(record_path)
            or receipt.get("tensor_file_sha256")
            != _complete_phase._file_sha256(tensor_path)
        ):
            raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
                "LFM2 batch checkpoint receipt differs from its request."
            )
        expected_codec_result = {
            "schema": LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_RECEIPT_SCHEMA,
            "receipt_sha256": receipt["receipt_sha256"],
            "replay_batch_sha256": receipt["replay_batch_sha256"],
            "batch_index": batch_index,
            "tensor_count": receipt["tensor_count"],
        }
        if entry.result.get("codec_result") != expected_codec_result:
            raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
                "LFM2 batch checkpoint codec receipt differs from completion."
            )
        expected_tensor_metadata = {
            "schema": LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_SAFETENSORS_SCHEMA,
            "codec_id": self.codec_id,
            "request_sha256": identity.request_sha256,
            "complete_phase_key_sha256": family["complete_phase_key_sha256"],
            "batch_index": str(batch_index),
            "replay_batch_sha256": receipt["replay_batch_sha256"],
            "tensor_count": str(receipt["tensor_count"]),
        }
        try:
            from safetensors import safe_open

            with safe_open(tensor_path, framework="pt", device="cpu") as handle:
                if handle.metadata() != expected_tensor_metadata:
                    raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
                        "LFM2 batch checkpoint SafeTensors metadata changed."
                    )
                names = tuple(sorted(handle.keys()))
                inventory = record.get("tensors")
                if (
                    not isinstance(inventory, Mapping)
                    or tuple(sorted(inventory)) != names
                ):
                    raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
                        "LFM2 batch checkpoint SafeTensors keys changed."
                    )
                tensors = {
                    name: handle.get_tensor(name)
                    .detach()
                    .to(device="cpu")
                    .contiguous()
                    .clone()
                    for name in names
                }
        except Lfm2ChildReplayBatchCheckpointAuthenticationError:
            raise
        except Exception as exc:
            raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
                "Could not load LFM2 batch checkpoint SafeTensors."
            ) from exc
        batch = Lfm2ChildReplayBatch(
            index=batch_index,
            input_sha256=str(record.get("input_sha256")),
            _tensors=MappingProxyType(tensors),
            _record=MappingProxyType(record),
        )
        replay_batch_sha256 = self.authenticate(batch, identity)
        if replay_batch_sha256 != receipt["replay_batch_sha256"]:
            raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
                "Loaded LFM2 batch differs from its receipt."
            )
        return batch


def _load_or_build_lockless_phase(
    context: DynamicChildReplayPhaseCacheContext,
    identity: DynamicChildReplayPhaseIdentity,
    *,
    codec: Any,
    producer: Callable[[], Any],
) -> AuthenticatedDynamicChildReplayPhase[Any]:
    """Resolve an immutable phase without a crash-persistent key lock.

    Every writer owns a unique staging directory.  A complete, fully fsynced
    staging tree is promoted with the cache's atomic no-replace primitive;
    losers authenticate the winner and discard only their own staging tree.
    Hard process death can therefore leave an inert ``*.staging`` directory,
    but never a lock that prevents a later writer from completing the phase.

    ``FoldPhaseEventLog`` uses its own persistent lock, so this crash-safe path
    deliberately reports through the lock-free progress callback rather than
    appending phase events.  Production child-replay contexts use that callback.
    """

    if not isinstance(context, DynamicChildReplayPhaseCacheContext):
        raise TypeError("context must be DynamicChildReplayPhaseCacheContext.")
    if not isinstance(identity, DynamicChildReplayPhaseIdentity):
        raise TypeError("identity must be DynamicChildReplayPhaseIdentity.")
    if not callable(producer):
        raise TypeError("producer must be callable.")
    if context.events is not None:
        raise Lfm2ChildReplayBatchCheckpointError(
            "Crash-safe lockless child-replay publication does not support "
            "FoldPhaseEventLog: its persistent lock can be orphaned by hard "
            "process death. Use the lock-free progress callback instead."
        )
    _generic_cache._validate_codec(codec, identity)
    key = identity.phase_key
    _progress(
        context,
        "dynamic_child_replay_phase_cache_lookup",
        phase_key_sha256=key.phase_key_sha256,
        codec_id=identity.codec_id,
        publication="lockless_atomic_noreplace",
    )
    entry = context.cache.lookup(key)
    if entry is not None:
        replay, replay_sha256 = _generic_cache._load_entry(
            context,
            identity,
            codec,
            entry,
        )
        _progress(
            context,
            "dynamic_child_replay_phase_cache_hit",
            phase_key_sha256=key.phase_key_sha256,
            completion_sha256=entry.completion_sha256,
            producer_invoked=False,
            publication="lockless_atomic_noreplace",
        )
        return AuthenticatedDynamicChildReplayPhase(
            replay=replay,
            entry=entry,
            cache_hit=True,
            producer_invoked=False,
            publication_race_lost=False,
            replay_sha256=replay_sha256,
        )

    _progress(
        context,
        "dynamic_child_replay_phase_cache_miss",
        phase_key_sha256=key.phase_key_sha256,
        publication="lockless_atomic_noreplace",
    )
    _progress(
        context,
        "dynamic_child_replay_phase_producer_started",
        phase_key_sha256=key.phase_key_sha256,
    )
    cold = producer()
    cold_sha256 = _generic_cache._digest(
        codec.authenticate(cold, identity),
        "cold replay SHA-256",
    )
    _progress(
        context,
        "dynamic_child_replay_phase_producer_complete",
        phase_key_sha256=key.phase_key_sha256,
        replay_sha256=cold_sha256,
    )
    _progress(
        context,
        "dynamic_child_replay_phase_publish_started",
        phase_key_sha256=key.phase_key_sha256,
        publication="lockless_atomic_noreplace",
    )

    destination = context.cache.entry_path(key)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    _fold_phase._assert_no_reparse_ancestors(
        parent,
        FoldPhaseCachePublicationError,
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{key.phase_key_sha256}.",
            suffix=".staging",
            dir=parent,
        )
    )
    payload = staging / _fold_phase.PAYLOAD_DIRECTORY_NAME
    payload.mkdir()
    try:
        codec_result = codec.write(cold, payload, identity)
        completion_result = _generic_cache._completion_result(
            identity,
            cold_sha256,
            codec_result,
        )
        inventory = _fold_phase._payload_inventory(payload)
        completion_body = {
            "schema": _fold_phase.PHASE_COMPLETION_SCHEMA,
            "phase_key": key.as_dict(),
            "phase_key_sha256": key.phase_key_sha256,
            "payload_inventory": inventory,
            "result": completion_result,
            "immutable_by_contract": True,
        }
        completion_sha256 = _fold_phase._canonical_sha256(
            completion_body,
            "phase completion",
        )
        completion = {
            **completion_body,
            "completion_sha256": completion_sha256,
        }
        _fold_phase._atomic_create_bytes(
            staging / _fold_phase.COMPLETE_FILENAME,
            _fold_phase._canonical_json_bytes(
                completion,
                "phase completion",
            )
            + b"\n",
            FoldPhaseCachePublicationError,
        )
        _fold_phase._fsync_tree(staging)
        staged = _fold_phase._verify_phase_entry(staging, expected_key=key)
        if staged.completion_sha256 != completion_sha256:
            raise FoldPhaseCachePublicationError(
                "Lockless staged phase changed before publication."
            )
        try:
            _fold_phase._rename_directory_noreplace(staging, destination)
        except FoldPhaseCacheAlreadyCompleteError:
            raced = context.cache.lookup(key)
            if raced is None:  # pragma: no cover - defensive concurrent deletion
                raise FoldPhaseCacheIntegrityError(
                    "Concurrent lockless phase publication disappeared."
                )
            _raced_replay, replay_sha256 = _generic_cache._load_entry(
                context,
                identity,
                codec,
                raced,
            )
            if replay_sha256 != cold_sha256:
                raise FoldPhaseCacheIntegrityError(
                    "Concurrent lockless phase differs from the local producer."
                )
            if (
                _generic_cache._digest(
                    codec.authenticate(cold, identity),
                    "post-race local replay SHA-256",
                )
                != cold_sha256
            ):
                raise FoldPhaseCacheIntegrityError(
                    "Local replay changed while the lockless winner was authenticated."
                )
            _progress(
                context,
                "dynamic_child_replay_phase_publish_race_resumed",
                phase_key_sha256=key.phase_key_sha256,
                completion_sha256=raced.completion_sha256,
                publication="lockless_atomic_noreplace",
            )
            return AuthenticatedDynamicChildReplayPhase(
                replay=cold,
                entry=raced,
                cache_hit=True,
                producer_invoked=True,
                publication_race_lost=True,
                replay_sha256=replay_sha256,
            )
        _fold_phase._fsync_directory(parent)
        entry = _fold_phase._verify_phase_entry(destination, expected_key=key)
    finally:
        # A normal loser or Python exception removes only this invocation's
        # unique staging path.  A hard-death orphan is inert and cannot block.
        if _fold_phase._lexists(staging):
            shutil.rmtree(staging)

    replay, replay_sha256 = _generic_cache._load_entry(
        context,
        identity,
        codec,
        entry,
    )
    if replay_sha256 != cold_sha256:
        raise FoldPhaseCacheIntegrityError(
            "Cold and reloaded lockless phase digests differ."
        )
    if (
        _generic_cache._digest(
            codec.authenticate(cold, identity),
            "post-publication cold replay SHA-256",
        )
        != cold_sha256
    ):
        raise FoldPhaseCacheIntegrityError(
            "Cold replay changed while its lockless payload was published."
        )
    _progress(
        context,
        "dynamic_child_replay_phase_published",
        phase_key_sha256=key.phase_key_sha256,
        completion_sha256=entry.completion_sha256,
        replay_sha256=replay_sha256,
        payload_bytes=entry.payload_inventory["total_bytes"],
        cold_reload_exact=True,
        publication="lockless_atomic_noreplace",
    )
    return AuthenticatedDynamicChildReplayPhase(
        replay=cold,
        entry=entry,
        cache_hit=False,
        producer_invoked=True,
        publication_race_lost=False,
        replay_sha256=replay_sha256,
    )


def _load_known_batch_checkpoint(
    context: DynamicChildReplayPhaseCacheContext,
    request: Lfm2ChildReplayBatchCheckpointRequest,
) -> Lfm2ChildReplayBatch | None:
    key = request.phase_key
    _progress(
        context,
        LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_LOOKUP_STAGE,
        complete_phase_key_sha256=request.family_request["complete_phase_key_sha256"],
        batch_index=request.batch_index,
        phase_key_sha256=key.phase_key_sha256,
    )
    if context.cache.lookup(key) is None:
        _progress(
            context,
            LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_MISS_STAGE,
            batch_index=request.batch_index,
            phase_key_sha256=key.phase_key_sha256,
        )
        return None

    def forbidden() -> Lfm2ChildReplayBatch:
        raise FoldPhaseCacheIntegrityError(
            "An observed batch checkpoint disappeared before authentication."
        )

    resolved = _load_or_build_lockless_phase(
        context,
        request.identity,
        codec=Lfm2ChildReplayBatchCheckpointCodec(),
        producer=forbidden,
    )
    _progress(
        context,
        LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_HIT_STAGE,
        batch_index=request.batch_index,
        phase_key_sha256=key.phase_key_sha256,
        completion_sha256=resolved.entry.completion_sha256,
        replay_batch_sha256=resolved.replay_sha256,
    )
    return resolved.replay


def _publish_batch_checkpoint(
    context: DynamicChildReplayPhaseCacheContext,
    request: Lfm2ChildReplayBatchCheckpointRequest,
    batch: Lfm2ChildReplayBatch,
) -> Lfm2ChildReplayBatch:
    codec = Lfm2ChildReplayBatchCheckpointCodec()
    local_sha256 = codec.authenticate(batch, request.identity)
    resolved = _load_or_build_lockless_phase(
        context,
        request.identity,
        codec=codec,
        producer=lambda: batch,
    )
    if resolved.replay_sha256 != local_sha256:
        raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
            "Concurrent LFM2 batch checkpoint differs from the local capture."
        )
    if codec.authenticate(batch, request.identity) != local_sha256:
        raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
            "Local LFM2 batch changed during checkpoint publication."
        )
    if resolved.producer_invoked:
        _progress(
            context,
            LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_PUBLISHED_STAGE,
            complete_phase_key_sha256=request.family_request[
                "complete_phase_key_sha256"
            ],
            batch_index=request.batch_index,
            phase_key_sha256=request.phase_key.phase_key_sha256,
            completion_sha256=resolved.entry.completion_sha256,
            replay_batch_sha256=resolved.replay_sha256,
            payload_bytes=resolved.entry.payload_inventory["total_bytes"],
        )
    return resolved.replay


class _CheckpointingLfm2ChildReplaySealPipeline(_trace._Lfm2ChildReplaySealPipeline):
    """Seal exact legacy bytes, then atomically publish each finished batch."""

    __slots__ = ("_checkpoint_context", "_checkpoint_requests")

    def __init__(
        self,
        torch: Any,
        *,
        telemetry: Any,
        enabled: bool,
        checkpoint_context: DynamicChildReplayPhaseCacheContext,
        checkpoint_requests: Mapping[int, Lfm2ChildReplayBatchCheckpointRequest],
    ) -> None:
        super().__init__(torch, telemetry=telemetry, enabled=enabled)
        self._checkpoint_context = checkpoint_context
        self._checkpoint_requests = dict(checkpoint_requests)

    def _seal(
        self,
        body: dict[str, Any],
        tensors: Mapping[str, Any],
    ) -> Lfm2ChildReplayBatch:
        batch = super()._seal(body, tensors)
        try:
            request = self._checkpoint_requests[batch.index]
        except KeyError as exc:  # pragma: no cover - internal invariant
            raise Lfm2ChildReplayBatchCheckpointError(
                f"No checkpoint request exists for batch {batch.index}."
            ) from exc
        return _publish_batch_checkpoint(
            self._checkpoint_context,
            request,
            batch,
        )

    def submit(self, body: Mapping[str, Any], tensors: Mapping[str, Any]) -> None:
        super().submit(body, tensors)
        if self._executor is None:
            # The explicitly serial mode is used by deterministic diagnostics
            # and tests.  Surface a publication/progress failure at the batch
            # boundary instead of replaying later batches first.
            with self._lock:
                future = self._futures[-1][1]
            future.result()


def _validate_capture_request(
    complete_request: Lfm2ChildReplayPhaseRequest,
    *,
    child_before: Any,
    corpus: Mapping[str, Any],
    topology: Mapping[str, Any],
    execution: Mapping[str, Any],
    implementation: Mapping[str, Any],
    numeric_policy: Mapping[str, Any],
) -> dict[str, Any]:
    family = complete_request.family_request
    observed = {
        "child": child_before.as_dict(),
        "corpus": dict(corpus),
        "layer_topology": dict(topology),
        "execution": dict(execution),
        "implementation": dict(implementation),
        "numeric_execution_policy": dict(numeric_policy),
    }
    for field, value in observed.items():
        if value != family.get(field):
            raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
                f"Live LFM2 replay {field} differs from its complete request."
            )
    return family


def _bounded_sharding_config(
    replay_devices: Sequence[Any] | Lfm2ReplayShardingConfig,
    missing_batch_count: int,
) -> Lfm2ReplayShardingConfig:
    selected = Lfm2ReplayShardingConfig.coerce(replay_devices)
    if len(selected.devices) <= missing_batch_count:
        return selected
    return Lfm2ReplayShardingConfig(
        devices=selected.devices[:missing_batch_count],
        assignment=selected.assignment,
    )


def capture_lfm2_child_replay_resumable(
    context: DynamicChildReplayPhaseCacheContext,
    complete_request: Lfm2ChildReplayPhaseRequest,
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
) -> Lfm2ChildReplay:
    """Resume an exact canonical capture from authenticated sealed batches."""

    if not isinstance(context, DynamicChildReplayPhaseCacheContext):
        raise TypeError("context must be DynamicChildReplayPhaseCacheContext.")
    if not isinstance(complete_request, Lfm2ChildReplayPhaseRequest):
        raise TypeError("complete_request must be Lfm2ChildReplayPhaseRequest.")
    if replay_capture_mode != CHILD_REPLAY_PREFIX_BY_PREFIX_MODE:
        raise Lfm2ChildReplayBatchCheckpointError(
            "Batch checkpoints support only prefix_by_prefix_canonical replay."
        )
    torch = _trace._torch()
    selected_config = _complete_phase._selected_config(config)
    if child_evidence_profile is not None and not isinstance(
        child_evidence_profile,
        Lfm2ChildEvidenceProfile,
    ):
        raise TypeError(
            "child_evidence_profile must be Lfm2ChildEvidenceProfile or None."
        )
    if (
        child_evidence_profile is not None
        and selected_config.capture_causal_circuit_evidence
    ):
        raise Lfm2ChildReplayBatchCheckpointError(
            "Point-specific child evidence profiles do not authenticate "
            "causal-circuit output-boundary globals."
        )
    prepared = _trace._prepare_batches(
        torch,
        _trace._exact_replay_batch_mappings(tuple(replay_batches)),
        selected_config,
    )
    layers, contracts, topology = _trace._layer_topology(child_model)
    selected_indices = _trace._selected_layer_indices(
        len(layers),
        selected_layer_indices,
    )
    if (
        child_evidence_profile is not None
        and child_evidence_profile.selected_layer_indices != selected_indices
    ):
        raise Lfm2ChildReplayBatchCheckpointError(
            "Child evidence profile differs from selected_layer_indices."
        )
    device = _trace._resolve_input_device(torch, child_model, input_device)
    telemetry = _trace._CaptureTelemetry()
    fingerprint_started = time.perf_counter()
    child_before = fingerprint_lfm2_parent(child_model)
    telemetry.fingerprint_seconds += time.perf_counter() - fingerprint_started
    corpus = _trace._corpus_identity(prepared)
    execution = _trace._execution_identity(child_model, device)
    implementation = _trace.implementation_identity()
    numeric_policy = _trace.compile_lfm2_child_replay_numeric_policy(
        child_model,
        input_device=device,
    )
    family = _validate_capture_request(
        complete_request,
        child_before=child_before,
        corpus=corpus,
        topology=topology,
        execution=execution,
        implementation=implementation,
        numeric_policy=numeric_policy,
    )
    selection = family["selection"]
    if (
        tuple(selection["selected_layer_indices"]) != selected_indices
        or selection["capture"] != selected_config.as_dict()
        or selection["replay_capture_mode"] != CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
        or selection.get("child_evidence_profile")
        != (
            None if child_evidence_profile is None else child_evidence_profile.as_dict()
        )
    ):
        raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
            "Live LFM2 replay selection differs from its complete request."
        )
    if len(prepared) != family["batch_count"]:
        raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
            "Live LFM2 replay batch count differs from its complete request."
        )
    checkpoint_requests = {
        batch.index: describe_lfm2_child_replay_batch_checkpoint(
            complete_request,
            batch.index,
        )
        for batch in prepared
    }
    lookup_started = time.perf_counter()
    resumed: dict[int, Lfm2ChildReplayBatch] = {}
    for batch in prepared:
        loaded = _load_known_batch_checkpoint(
            context,
            checkpoint_requests[batch.index],
        )
        if loaded is not None:
            resumed[batch.index] = loaded
    checkpoint_lookup_seconds = time.perf_counter() - lookup_started
    missing = tuple(batch for batch in prepared if batch.index not in resumed)

    current_invocation_child_forwards = 0
    seal_pipeline = _CheckpointingLfm2ChildReplaySealPipeline(
        torch,
        telemetry=telemetry,
        enabled=_trace._SEAL_PIPELINE_ENABLED,
        checkpoint_context=context,
        checkpoint_requests=checkpoint_requests,
    )
    try:
        if missing and replay_devices is not None:
            try:
                sharding_config = _bounded_sharding_config(
                    replay_devices,
                    len(missing),
                )
                shard_plan = plan_replay_shards(
                    torch,
                    sharding_config,
                    primary_device=device,
                    batch_costs=[
                        _trace._child_replay_batch_cost(
                            batch,
                            CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
                        )
                        for batch in missing
                    ],
                )
                current_invocation_child_forwards, _canonical_count = (
                    _trace._sharded_child_replay(
                        torch,
                        child_model,
                        layers,
                        contracts,
                        topology,
                        missing,
                        shard_plan,
                        config=selected_config,
                        device=device,
                        selected_indices=selected_indices,
                        replay_mode=CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
                        child_evidence_profile=child_evidence_profile,
                        execution=execution,
                        numeric_policy=numeric_policy,
                        causal_contract=None,
                        telemetry=telemetry,
                        seal_pipeline=seal_pipeline,
                    )
                )
                if telemetry.sharding is None:  # pragma: no cover - invariant
                    raise Lfm2ChildReplayBatchCheckpointError(
                        "Sharded replay produced no runtime telemetry."
                    )
                planned_batch_indices = [batch.index for batch in missing]
                telemetry.sharding = {
                    **telemetry.sharding,
                    "batch_assignment_index_space": "planned_batch_indices",
                    "planned_batch_indices": planned_batch_indices,
                    "original_batch_assignment": [
                        {
                            "batch_index": planned_batch_indices[local_index],
                            "worker_index": worker_index,
                            "batch_cost": shard_plan.batch_costs[local_index],
                        }
                        for local_index, worker_index in enumerate(
                            shard_plan.batch_assignment
                        )
                    ],
                }
            except Lfm2ReplayShardingError as exc:
                raise Lfm2ChildReplayBatchCheckpointError(str(exc)) from exc
        else:
            for batch in missing:
                result = _trace._replay_child_batch(
                    torch,
                    child_model,
                    layers,
                    contracts,
                    batch,
                    config=selected_config,
                    device=device,
                    selected_indices=selected_indices,
                    replay_mode=CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
                    child_evidence_profile=child_evidence_profile,
                    causal_contract=None,
                    telemetry=telemetry,
                )
                current_invocation_child_forwards += result.child_forward_count
                seal_pipeline.submit(
                    _trace._child_replay_seal_body(
                        result,
                        batch,
                        selected_indices,
                        child_evidence_profile,
                    ),
                    result.tensors,
                )
        captured_batches = seal_pipeline.results()
    finally:
        seal_pipeline.close()

    merged = dict(resumed)
    for batch in captured_batches:
        if batch.index in merged:
            raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
                f"LFM2 replay produced batch {batch.index} more than once."
            )
        merged[batch.index] = batch
    expected_indices = list(range(len(prepared)))
    if sorted(merged) != expected_indices:
        raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
            "LFM2 resumed replay did not resolve every batch exactly once."
        )
    memory_batches = [merged[index] for index in expected_indices]

    fingerprint_started = time.perf_counter()
    child_after = fingerprint_lfm2_parent(child_model)
    telemetry.fingerprint_seconds += time.perf_counter() - fingerprint_started
    if child_after != child_before:
        raise Lfm2ChildReplayBatchCheckpointAuthenticationError(
            "Child state or configuration changed during resumable replay."
        )

    canonical_forward_count = int(family["canonical_prefix_forward_count"])
    metadata = _trace._sealed_document(
        {
            "schema": CHILD_REPLAY_SCHEMA,
            "child": family["child"],
            "corpus": family["corpus"],
            "implementation": family["implementation"],
            "layer_topology": family["layer_topology"],
            "selection": family["selection"],
            "execution": family["execution"],
            "numeric_execution_policy": family["numeric_execution_policy"],
            "batch_count": len(memory_batches),
            "ordered_batches": [
                {
                    "batch_index": batch.index,
                    "input_sha256": batch.input_sha256,
                    "replay_batch_sha256": batch._record["replay_batch_sha256"],
                    "phase_order": list(batch.phase_order),
                    "child_evidence_profile_sha256": batch._record.get(
                        "child_evidence_profile_sha256"
                    ),
                }
                for batch in memory_batches
            ],
            # These are logical capture provenance, not the work repeated in
            # this invocation.  Completed checkpoints carry the exact same
            # child forwards, so preserving the complete count keeps the
            # final replay byte-identical to uninterrupted canonical capture.
            "forward_counts": {
                "child": canonical_forward_count,
                "parent": 0,
            },
            "replay_benchmark": {
                "batch_count": len(memory_batches),
                "phase_count": canonical_forward_count,
                "actual_child_forwards": canonical_forward_count,
                "canonical_prefix_child_forwards": canonical_forward_count,
                "saved_child_forwards": 0,
            },
            "causal_equivalence": {
                "reference_supplied": False,
                "tolerance_equivalent": None,
            },
            "runtime": {
                "tokenizer_accessed": False,
                "parent_model_supplied": False,
                "parent_forward_count": 0,
                "disk_artifacts_written": 0,
                "intermediate_cache": "memory_only",
                "child_state_preserved": True,
            },
        },
        "replay_sha256",
    )
    timing = telemetry.as_dict()
    timing["batch_checkpoint_resume"] = {
        "schema": LFM2_CHILD_REPLAY_BATCH_RESUME_TIMING_SCHEMA,
        "sealed_in_metadata": False,
        "complete_phase_key_sha256": complete_request.phase_key.phase_key_sha256,
        "batch_count": len(prepared),
        "resumed_batch_count": len(resumed),
        "resumed_batch_indices": sorted(resumed),
        "captured_batch_count": len(captured_batches),
        "captured_batch_indices": sorted(batch.index for batch in captured_batches),
        "current_invocation_child_forwards": (current_invocation_child_forwards),
        "persisted_capture_child_forwards": canonical_forward_count,
        "checkpoint_lookup_seconds": round(checkpoint_lookup_seconds, 6),
    }
    return Lfm2ChildReplay(
        _batches=tuple(memory_batches),
        _metadata=MappingProxyType(metadata),
        _capture_timing=MappingProxyType(timing),
    )


def load_or_capture_lfm2_child_replay_phase_resumable(
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
    """Load a complete replay or resume its missing canonical batches."""

    if not isinstance(context, DynamicChildReplayPhaseCacheContext):
        raise TypeError("context must be DynamicChildReplayPhaseCacheContext.")
    batches = tuple(replay_batches)
    complete_request = describe_lfm2_child_replay_phase(
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
        return capture_lfm2_child_replay_resumable(
            context,
            complete_request,
            child_model,
            batches,
            selected_layer_indices=selected_layer_indices,
            config=config,
            input_device=input_device,
            replay_capture_mode=replay_capture_mode,
            child_evidence_profile=child_evidence_profile,
            replay_devices=replay_devices,
        )

    result = _load_or_build_lockless_phase(
        context,
        complete_request.identity,
        codec=Lfm2ChildReplayPhaseCodec(),
        producer=capture,
    )
    if (
        fingerprint_lfm2_parent(child_model).as_dict()
        != (complete_request.family_request["child"])
    ):
        raise Lfm2ChildReplayPhaseAuthenticationError(
            "Live child state changed while its replay phase was resolved."
        )
    return result


__all__ = [
    "LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_CODEC_ID",
    "LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_HIT_STAGE",
    "LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_LOOKUP_STAGE",
    "LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_MISS_STAGE",
    "LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_PHASE",
    "LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_PUBLISHED_STAGE",
    "LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_RECEIPT_SCHEMA",
    "LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_REQUEST_SCHEMA",
    "LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_SAFETENSORS_SCHEMA",
    "LFM2_CHILD_REPLAY_BATCH_RESUME_TIMING_SCHEMA",
    "Lfm2ChildReplayBatchCheckpointAuthenticationError",
    "Lfm2ChildReplayBatchCheckpointCodec",
    "Lfm2ChildReplayBatchCheckpointError",
    "Lfm2ChildReplayBatchCheckpointRequest",
    "capture_lfm2_child_replay_resumable",
    "describe_lfm2_child_replay_batch_checkpoint",
    "load_or_capture_lfm2_child_replay_phase_resumable",
]
