"""Authenticated conditioned-child recapture for the LFM2 moving-gauge fold.

This phase is deliberately child-only.  A deferred range provider supplies
``Lfm2ChildReplay`` objects after the accepted layer-zero operator has been
installed.  Every replay document, batch record, tensor descriptor, and tensor
byte is authenticated before a checkpoint is admitted.  Checkpoints form a
domain-separated HMAC chain and store the retained tensors in safetensors.

The exact authenticated basis assignment supplies 192 solve-visible FIT
batches and 64 interleaved HELDOUT batches.  HELDOUT rows cannot be read
without a phase-specific frozen-candidate barrier capability.  Completed cache
hits happen before the provider or work directory is touched.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from .dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from .dynamic_fold_phase_cache import (
    DynamicFoldPhaseCache,
    FoldPhaseCacheAlreadyCompleteError,
    FoldPhaseEventLog,
    FoldPhaseKey,
    VerifiedFoldPhaseCacheEntry,
)
from .hf_lfm2_circuit_trace_cache import (
    CHILD_REPLAY_BATCH_SCHEMA,
    CHILD_REPLAY_SCHEMA,
    Lfm2ChildEvidenceProfile,
    Lfm2ChildReplay,
)
from .hf_lfm2_exact4x_causal_diagnostic import equal_record_mass_weights
from .hf_lfm2_moving_gauge_residual_reclosure import (
    fingerprint_lfm2_moving_gauge_tensor,
)


LFM2_CONDITIONED_RECAPTURE_PHASE = "lfm2-moving-gauge-conditioned-recapture-v1"
LFM2_CONDITIONED_RECAPTURE_REQUEST_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-conditioned-recapture-request-v1"
)
LFM2_CONDITIONED_RECAPTURE_CHECKPOINT_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-conditioned-recapture-checkpoint-v1"
)
LFM2_CONDITIONED_RECAPTURE_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-conditioned-recapture-receipt-v1"
)
LFM2_CONDITIONED_RECAPTURE_ROW_INDEX_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-conditioned-recapture-row-index-v1"
)
LFM2_CONDITIONED_RECAPTURE_ANCHOR_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-conditioned-recapture-anchor-v1"
)
LFM2_CONDITIONED_RECAPTURE_SHARD_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-conditioned-recapture-shard-v1"
)
LFM2_CONDITIONED_RECAPTURE_FIELD_CONTRACT_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-conditioned-recapture-field-v1"
)
LFM2_CONDITIONED_RECAPTURE_SPLIT_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-conditioned-recapture-split-v1"
)
LFM2_CONDITIONED_RECAPTURE_HELDOUT_CAPABILITY_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-conditioned-recapture-heldout-capability-v1"
)
LFM2_CONDITIONED_CHILD_STATE_RECEIPT_BINDING_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-conditioned-child-state-receipt-binding-v1"
)

LFM2_CONDITIONED_RECAPTURE_REQUIRED_FIELDS = (
    "short_conv.context",
    "operator.output",
    "residual.pre",
    "residual.post_operator",
    "residual.post_layer",
    "mlp.post_swiglu",
    "mlp.down",
)
LFM2_CONDITIONED_RECAPTURE_PROFILE_ROLE = "moving_gauge.conditioned_recapture"

_CHECKPOINT_DOMAIN = b"neural-forge-lfm2-conditioned-recapture-checkpoint-v1\0"
_ANCHOR_DOMAIN = b"neural-forge-lfm2-conditioned-recapture-anchor-v1\0"
_CAPABILITY_DOMAIN = b"neural-forge-lfm2-conditioned-recapture-heldout-v1\0"
_TRACE_TENSOR_DOMAIN = b"neural-forge-lfm2-circuit-tensor-v1\0"
_ZERO_SHA256 = "0" * 64
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*\Z")
_PHASE = re.compile(r"(?:prefill|decode\.(\d{3}))\Z")
_TENSOR_DESCRIPTOR_FIELDS = frozenset(
    {"dtype", "numel", "sha256", "shape", "size_bytes"}
)
_FLOAT_DTYPES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
}
_ZERO_OPERATION_AUDIT = {
    "parent_model_objects_constructed": 0,
    "parent_model_forward_calls": 0,
    "tokenizer_accesses": 0,
    "network_accesses": 0,
    "model_mutations": 0,
    "backward_calls": 0,
    "autograd_grad_calls": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "training_steps": 0,
}


class Lfm2ConditionedRecapturePhaseError(RuntimeError):
    """Base error for conditioned-child recapture."""


class Lfm2ConditionedRecaptureAuthenticationError(
    Lfm2ConditionedRecapturePhaseError
):
    """Raised when authenticated recapture evidence changes."""


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2ConditionedRecapturePhaseError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise Lfm2ConditionedRecapturePhaseError(f"{name} is not a safe identifier.")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2ConditionedRecapturePhaseError(f"{name} must be positive.")
    return value


def _canonical(value: Any, name: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value).decode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise Lfm2ConditionedRecaptureAuthenticationError(
            f"{name} is not canonical JSON."
        ) from exc


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_deep_thaw(item) for item in value]
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(dict(value)) + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Lfm2ConditionedRecaptureAuthenticationError(
            f"Could not authenticate JSON document {path.name!r}."
        ) from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) + b"\n" != path.read_bytes():
        raise Lfm2ConditionedRecaptureAuthenticationError(
            f"JSON document {path.name!r} is not canonical."
        )
    return value


def _verify_seal(value: Any, seal_name: str, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            f"{name} must be a mapping."
        )
    result = _canonical(dict(value), name)
    claimed = result.pop(seal_name, None)
    if not isinstance(claimed, str) or canonical_sha256(result) != claimed:
        raise Lfm2ConditionedRecaptureAuthenticationError(
            f"{name} self-seal is invalid."
        )
    return {**result, seal_name: claimed}


def _torch_dtype_name(value: torch.Tensor) -> str:
    return str(value.dtype).removeprefix("torch.")


def _trace_tensor_descriptor(value: torch.Tensor) -> dict[str, Any]:
    if not torch.is_tensor(value) or value.layout != torch.strided:
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Replay tensors must be dense strided tensors."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    dtype = _torch_dtype_name(tensor)
    shape = [int(item) for item in tensor.shape]
    digest = hashlib.sha256()
    digest.update(_TRACE_TENSOR_DOMAIN)
    digest.update(dtype.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json_bytes(shape))
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return {
        "dtype": dtype,
        "shape": shape,
        "numel": int(tensor.numel()),
        "size_bytes": int(tensor.numel() * tensor.element_size()),
        "sha256": digest.hexdigest(),
    }


def _stored_tensor_descriptor(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().to(device="cpu").contiguous()
    return {
        "dtype": _torch_dtype_name(tensor),
        "shape": [int(item) for item in tensor.shape],
        "numel": int(tensor.numel()),
        "size_bytes": int(tensor.numel() * tensor.element_size()),
        "sha256": fingerprint_lfm2_moving_gauge_tensor(tensor),
    }


def _validate_descriptor(value: Any, name: str, *, floating: bool) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TENSOR_DESCRIPTOR_FIELDS:
        raise Lfm2ConditionedRecapturePhaseError(
            f"{name} must have the exact tensor descriptor fields."
        )
    result = _canonical(dict(value), name)
    dtype = result.get("dtype")
    shape = result.get("shape")
    if (
        not isinstance(dtype, str)
        or (floating and dtype not in _FLOAT_DTYPES)
        or not isinstance(shape, list)
        or not shape
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)
    ):
        raise Lfm2ConditionedRecapturePhaseError(f"{name} is malformed.")
    numel = math.prod(shape)
    if result.get("numel") != numel or not isinstance(result.get("size_bytes"), int):
        raise Lfm2ConditionedRecapturePhaseError(f"{name} geometry is invalid.")
    if dtype in _FLOAT_DTYPES and result["size_bytes"] != numel * _FLOAT_DTYPES[dtype]:
        raise Lfm2ConditionedRecapturePhaseError(f"{name} byte size is invalid.")
    _digest(result.get("sha256"), f"{name}.sha256")
    return result


def lfm2_conditioned_recapture_spec_sha256() -> str:
    """Return the immutable semantic contract digest for this phase."""

    return canonical_sha256(
        {
            "phase": LFM2_CONDITIONED_RECAPTURE_PHASE,
            "request_schema": LFM2_CONDITIONED_RECAPTURE_REQUEST_SCHEMA,
            "checkpoint_schema": LFM2_CONDITIONED_RECAPTURE_CHECKPOINT_SCHEMA,
            "required_fields": list(LFM2_CONDITIONED_RECAPTURE_REQUIRED_FIELDS),
            "batch_partition": {
                "assignment_source": "request.basis_split_by_batch",
                "assignment_order": "global-source-batch-index",
                "fit_batch_count": 192,
                "heldout_batch_count": 64,
                "contiguity_assumed": False,
            },
            "storage": "canonical-json-plus-safetensors",
            "checkpoint_authentication": "domain-separated-hmac-chain",
            "heldout_policy": "frozen-candidate-barrier-capability-required",
            "provider_policy": "child-only-inference-mode-range-provider",
        }
    )


def lfm2_conditioned_recapture_implementation_sha256() -> str:
    """Fingerprint the exact executing conditioned-recapture implementation."""

    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-conditioned-recapture-source-v1\0")
    digest.update(Path(__file__).resolve().read_bytes())
    return digest.hexdigest()


def derive_lfm2_conditioned_recapture_split_identity(
    *,
    source_input_session_sha256: str,
    source_input_sha256s: Sequence[str],
    source_record_ids: Sequence[str],
    basis_split_by_batch: Sequence[str],
) -> str:
    """Bind the authenticated interleaved 192/64 split to source identities."""

    session = _digest(source_input_session_sha256, "source_input_session_sha256")
    inputs = tuple(source_input_sha256s)
    records = tuple(source_record_ids)
    splits = tuple(basis_split_by_batch)
    if len(inputs) != 256 or len(records) != 256 or len(splits) != 256:
        raise Lfm2ConditionedRecapturePhaseError(
            "Conditioned recapture requires exactly 256 ordered source batches."
        )
    if (
        any(value not in {"fit", "heldout"} for value in splits)
        or splits.count("fit") != 192
        or splits.count("heldout") != 64
    ):
        raise Lfm2ConditionedRecapturePhaseError(
            "Conditioned recapture requires the exact 192 FIT / 64 HELDOUT "
            "basis assignment."
        )
    rows: list[dict[str, Any]] = []
    for index, (input_sha256, record_id, split) in enumerate(
        zip(inputs, records, splits)
    ):
        rows.append(
            {
                "batch_index": index,
                "input_sha256": _digest(input_sha256, "source input SHA-256"),
                "record_id": _identifier(record_id, "source record ID"),
                "split": split,
            }
        )
    return canonical_sha256(
        {
            "schema": LFM2_CONDITIONED_RECAPTURE_SPLIT_SCHEMA,
            "source_input_session_sha256": session,
            "batch_count": 256,
            "fit_batch_count": 192,
            "heldout_batch_count": 64,
            "ordered_batches": rows,
        }
    )


def derive_lfm2_frozen_conditioned_child_state_receipt_sha256(
    *,
    base_child_identity_sha256: str,
    base_child_state_dict_sha256: str,
    endpoint_chain_receipt_sha256: str,
    operator_replacement_tensor_descriptor: Mapping[str, Any],
    operator_solution_sha256: str,
    conditioned_child_identity_sha256: str,
    conditioned_child_state_dict_sha256: str,
) -> str:
    """Bind the accepted operator transition to its exact frozen post-state."""

    descriptor = _validate_descriptor(
        operator_replacement_tensor_descriptor,
        "operator replacement descriptor",
        floating=True,
    )
    return canonical_sha256(
        {
            "schema": LFM2_CONDITIONED_CHILD_STATE_RECEIPT_BINDING_SCHEMA,
            "base_child_identity_sha256": _digest(
                base_child_identity_sha256, "base_child_identity_sha256"
            ),
            "base_child_state_dict_sha256": _digest(
                base_child_state_dict_sha256, "base_child_state_dict_sha256"
            ),
            "endpoint_chain_receipt_sha256": _digest(
                endpoint_chain_receipt_sha256, "endpoint_chain_receipt_sha256"
            ),
            "operator_replacement_tensor_descriptor": descriptor,
            "operator_solution_sha256": _digest(
                operator_solution_sha256, "operator_solution_sha256"
            ),
            "conditioned_child_identity_sha256": _digest(
                conditioned_child_identity_sha256,
                "conditioned_child_identity_sha256",
            ),
            "conditioned_child_state_dict_sha256": _digest(
                conditioned_child_state_dict_sha256,
                "conditioned_child_state_dict_sha256",
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class Lfm2ConditionedRecaptureFieldContract:
    """One request-pinned retained tensor width and storage dtype."""

    field_name: str
    source_dtype: str
    feature_width: int
    contract_sha256: str = ""

    def __post_init__(self) -> None:
        if self.field_name not in LFM2_CONDITIONED_RECAPTURE_REQUIRED_FIELDS:
            raise Lfm2ConditionedRecapturePhaseError(
                f"Unsupported conditioned-recapture field {self.field_name!r}."
            )
        if self.source_dtype not in _FLOAT_DTYPES:
            raise Lfm2ConditionedRecapturePhaseError(
                "Retained field dtype must be an explicit floating dtype."
            )
        object.__setattr__(
            self, "feature_width", _positive_int(self.feature_width, "feature_width")
        )
        expected = canonical_sha256(self._body())
        if self.contract_sha256 not in {"", expected}:
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "Retained field contract seal is invalid."
            )
        object.__setattr__(self, "contract_sha256", expected)

    def _body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CONDITIONED_RECAPTURE_FIELD_CONTRACT_SCHEMA,
            "field_name": self.field_name,
            "source_dtype": self.source_dtype,
            "feature_width": self.feature_width,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "contract_sha256": self.contract_sha256}


@dataclass(frozen=True, slots=True)
class Lfm2ConditionedRecapturePhaseRequest:
    """Complete content address for one layer-zero conditioned recapture."""

    v5_request_sha256: str
    v5_lineage_sha256: str
    base_child_identity_sha256: str
    base_child_state_dict_sha256: str
    endpoint_chain_receipt_sha256: str
    operator_replacement_tensor_descriptor: Mapping[str, Any]
    operator_solution_sha256: str
    frozen_conditioned_child_state_receipt_sha256: str
    conditioned_child_identity_sha256: str
    conditioned_child_state_dict_sha256: str
    child_layer_topology_sha256: str
    chart_mode: str
    child_replay_config_sha256: str
    child_replay_implementation_sha256: str
    child_replay_execution_sha256: str
    child_replay_numeric_policy_sha256: str
    child_replay_capture_mode: str
    source_input_session_sha256: str
    source_input_sha256s: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    basis_split_by_batch: tuple[str, ...]
    split_identity_sha256: str
    field_contracts: tuple[Lfm2ConditionedRecaptureFieldContract, ...]
    runtime_device_type: str
    runtime_floating_dtypes: tuple[str, ...]
    conditioned_recapture_spec_sha256: str
    implementation_sha256: str
    checkpoint_hmac_key_id: str
    checkpoint_batch_count: int = 64
    expected_batch_count: int = 256
    fit_batch_count: int = 192
    logical_layer_index: int = 0
    child_layer_index: int = 0
    request_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "v5_request_sha256",
            "v5_lineage_sha256",
            "base_child_identity_sha256",
            "base_child_state_dict_sha256",
            "endpoint_chain_receipt_sha256",
            "operator_solution_sha256",
            "frozen_conditioned_child_state_receipt_sha256",
            "conditioned_child_identity_sha256",
            "conditioned_child_state_dict_sha256",
            "child_layer_topology_sha256",
            "child_replay_config_sha256",
            "child_replay_implementation_sha256",
            "child_replay_execution_sha256",
            "child_replay_numeric_policy_sha256",
            "source_input_session_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.chart_mode not in {"E0", "E0R", "E0A"}:
            raise Lfm2ConditionedRecapturePhaseError(
                "chart_mode must be E0, E0R, or E0A."
            )
        if not isinstance(self.child_replay_capture_mode, str) or not self.child_replay_capture_mode:
            raise Lfm2ConditionedRecapturePhaseError(
                "child_replay_capture_mode must be nonempty."
            )
        object.__setattr__(
            self,
            "operator_replacement_tensor_descriptor",
            _deep_freeze(
                _validate_descriptor(
                    self.operator_replacement_tensor_descriptor,
                    "operator replacement descriptor",
                    floating=True,
                )
            ),
        )
        expected_state_receipt = (
            derive_lfm2_frozen_conditioned_child_state_receipt_sha256(
                base_child_identity_sha256=self.base_child_identity_sha256,
                base_child_state_dict_sha256=self.base_child_state_dict_sha256,
                endpoint_chain_receipt_sha256=self.endpoint_chain_receipt_sha256,
                operator_replacement_tensor_descriptor=(
                    self.operator_replacement_tensor_descriptor
                ),
                operator_solution_sha256=self.operator_solution_sha256,
                conditioned_child_identity_sha256=(
                    self.conditioned_child_identity_sha256
                ),
                conditioned_child_state_dict_sha256=(
                    self.conditioned_child_state_dict_sha256
                ),
            )
        )
        if self.frozen_conditioned_child_state_receipt_sha256 != (
            expected_state_receipt
        ):
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "Frozen conditioned-child state receipt does not bind the exact "
                "base state, endpoint chain, operator, solution, and post-state."
            )
        if (self.expected_batch_count, self.fit_batch_count) != (256, 192):
            raise Lfm2ConditionedRecapturePhaseError(
                "The authenticated split must contain 192 FIT and 64 HELDOUT batches."
            )
        object.__setattr__(
            self,
            "checkpoint_batch_count",
            _positive_int(self.checkpoint_batch_count, "checkpoint_batch_count"),
        )
        if 256 % self.checkpoint_batch_count != 0:
            raise Lfm2ConditionedRecapturePhaseError(
                "checkpoint_batch_count must divide 256 exactly."
            )
        if self.logical_layer_index != 0 or self.child_layer_index != 0:
            raise Lfm2ConditionedRecapturePhaseError(
                "Conditioned recapture is intentionally restricted to layer zero."
            )
        inputs = tuple(self.source_input_sha256s)
        records = tuple(self.source_record_ids)
        splits = tuple(self.basis_split_by_batch)
        for value in inputs:
            _digest(value, "source input SHA-256")
        for value in records:
            _identifier(value, "source record ID")
        if len(set(records)) != len(records):
            raise Lfm2ConditionedRecapturePhaseError(
                "source_record_ids must be unique."
            )
        if (
            len(splits) != self.expected_batch_count
            or any(value not in {"fit", "heldout"} for value in splits)
            or splits.count("fit") != self.fit_batch_count
            or splits.count("heldout")
            != self.expected_batch_count - self.fit_batch_count
        ):
            raise Lfm2ConditionedRecapturePhaseError(
                "basis_split_by_batch must be the exact 256-entry 192/64 assignment."
            )
        object.__setattr__(self, "source_input_sha256s", inputs)
        object.__setattr__(self, "source_record_ids", records)
        object.__setattr__(self, "basis_split_by_batch", splits)
        expected_split = derive_lfm2_conditioned_recapture_split_identity(
            source_input_session_sha256=self.source_input_session_sha256,
            source_input_sha256s=inputs,
            source_record_ids=records,
            basis_split_by_batch=splits,
        )
        if self.split_identity_sha256 != expected_split:
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "Conditioned recapture split identity is invalid."
            )
        contracts = tuple(self.field_contracts)
        if (
            tuple(value.field_name for value in contracts)
            != LFM2_CONDITIONED_RECAPTURE_REQUIRED_FIELDS
            or any(
                not isinstance(value, Lfm2ConditionedRecaptureFieldContract)
                for value in contracts
            )
        ):
            raise Lfm2ConditionedRecapturePhaseError(
                "field_contracts must be the exact ordered required inventory."
            )
        object.__setattr__(self, "field_contracts", contracts)
        if not isinstance(self.runtime_device_type, str) or not self.runtime_device_type:
            raise Lfm2ConditionedRecapturePhaseError(
                "runtime_device_type must be nonempty."
            )
        dtypes = tuple(self.runtime_floating_dtypes)
        if dtypes != tuple(sorted(set(dtypes))) or any(
            value not in _FLOAT_DTYPES for value in dtypes
        ):
            raise Lfm2ConditionedRecapturePhaseError(
                "runtime_floating_dtypes must be unique, sorted floating dtypes."
            )
        if set(value.source_dtype for value in contracts) - set(dtypes):
            raise Lfm2ConditionedRecapturePhaseError(
                "Every retained field dtype must appear in runtime_floating_dtypes."
            )
        object.__setattr__(self, "runtime_floating_dtypes", dtypes)
        if (
            _digest(
                self.conditioned_recapture_spec_sha256,
                "conditioned_recapture_spec_sha256",
            )
            != lfm2_conditioned_recapture_spec_sha256()
        ):
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "Conditioned recapture spec differs from the executing contract."
            )
        if (
            _digest(self.implementation_sha256, "implementation_sha256")
            != lfm2_conditioned_recapture_implementation_sha256()
        ):
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "Conditioned recapture implementation differs from executing source."
            )
        object.__setattr__(
            self,
            "checkpoint_hmac_key_id",
            _identifier(self.checkpoint_hmac_key_id, "checkpoint_hmac_key_id"),
        )
        expected = canonical_sha256(self._body())
        if self.request_sha256 not in {"", expected}:
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "Conditioned recapture request seal is invalid."
            )
        object.__setattr__(self, "request_sha256", expected)

    def _body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CONDITIONED_RECAPTURE_REQUEST_SCHEMA,
            "phase_name": LFM2_CONDITIONED_RECAPTURE_PHASE,
            "v5_request_sha256": self.v5_request_sha256,
            "v5_lineage_sha256": self.v5_lineage_sha256,
            "base_child_identity_sha256": self.base_child_identity_sha256,
            "base_child_state_dict_sha256": self.base_child_state_dict_sha256,
            "endpoint_chain_receipt_sha256": self.endpoint_chain_receipt_sha256,
            "operator_replacement_tensor_descriptor": _deep_thaw(
                self.operator_replacement_tensor_descriptor
            ),
            "operator_solution_sha256": self.operator_solution_sha256,
            "frozen_conditioned_child_state_receipt_sha256": (
                self.frozen_conditioned_child_state_receipt_sha256
            ),
            "conditioned_child_identity_sha256": self.conditioned_child_identity_sha256,
            "conditioned_child_state_dict_sha256": (
                self.conditioned_child_state_dict_sha256
            ),
            "child_layer_topology_sha256": self.child_layer_topology_sha256,
            "chart_mode": self.chart_mode,
            "child_replay_config_sha256": self.child_replay_config_sha256,
            "child_replay_implementation_sha256": (
                self.child_replay_implementation_sha256
            ),
            "child_replay_execution_sha256": self.child_replay_execution_sha256,
            "child_replay_numeric_policy_sha256": (
                self.child_replay_numeric_policy_sha256
            ),
            "child_replay_capture_mode": self.child_replay_capture_mode,
            "source_input_session_sha256": self.source_input_session_sha256,
            "source_input_sha256s": list(self.source_input_sha256s),
            "source_record_ids": list(self.source_record_ids),
            "basis_split_by_batch": list(self.basis_split_by_batch),
            "split_identity_sha256": self.split_identity_sha256,
            "field_contracts": [value.as_dict() for value in self.field_contracts],
            "runtime_device_type": self.runtime_device_type,
            "runtime_floating_dtypes": list(self.runtime_floating_dtypes),
            "conditioned_recapture_spec_sha256": (
                self.conditioned_recapture_spec_sha256
            ),
            "implementation_sha256": self.implementation_sha256,
            "checkpoint_hmac_key_id": self.checkpoint_hmac_key_id,
            "checkpoint_batch_count": self.checkpoint_batch_count,
            "expected_batch_count": self.expected_batch_count,
            "fit_batch_count": self.fit_batch_count,
            "logical_layer_index": self.logical_layer_index,
            "child_layer_index": self.child_layer_index,
            "no_parent_tokenizer_network_or_training": True,
            "operation_contract": dict(_ZERO_OPERATION_AUDIT),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "request_sha256": self.request_sha256}

    def verify_authentication(self) -> bool:
        try:
            return bool(
                self.request_sha256 == canonical_sha256(self._body())
                and self.implementation_sha256
                == lfm2_conditioned_recapture_implementation_sha256()
                and self.conditioned_recapture_spec_sha256
                == lfm2_conditioned_recapture_spec_sha256()
            )
        except (OSError, TypeError, ValueError):
            return False

    @property
    def phase_key(self) -> FoldPhaseKey:
        dependencies = {
            "request": self.request_sha256,
            "v5_request": self.v5_request_sha256,
            "v5_lineage": self.v5_lineage_sha256,
            "base_child_identity": self.base_child_identity_sha256,
            "base_child_state_dict": self.base_child_state_dict_sha256,
            "endpoint_chain_receipt": self.endpoint_chain_receipt_sha256,
            "operator_replacement": str(
                self.operator_replacement_tensor_descriptor["sha256"]
            ),
            "operator_solution": self.operator_solution_sha256,
            "frozen_conditioned_child_state_receipt": (
                self.frozen_conditioned_child_state_receipt_sha256
            ),
            "conditioned_child_identity": self.conditioned_child_identity_sha256,
            "conditioned_child_state_dict": self.conditioned_child_state_dict_sha256,
            "child_layer_topology": self.child_layer_topology_sha256,
            "child_replay_config": self.child_replay_config_sha256,
            "child_replay_implementation": self.child_replay_implementation_sha256,
            "child_replay_execution": self.child_replay_execution_sha256,
            "child_replay_numeric_policy": self.child_replay_numeric_policy_sha256,
            "source_input_session": self.source_input_session_sha256,
            "split_identity": self.split_identity_sha256,
            **{
                f"field.{value.field_name}": value.contract_sha256
                for value in self.field_contracts
            },
        }
        numeric = {
            "chart_mode": self.chart_mode,
            "capture_mode": self.child_replay_capture_mode,
            "runtime_device_type": self.runtime_device_type,
            "runtime_floating_dtypes": list(self.runtime_floating_dtypes),
            "checkpoint_batch_count": self.checkpoint_batch_count,
            "basis_split_by_batch": list(self.basis_split_by_batch),
            "fit_batch_count": self.fit_batch_count,
            "heldout_batch_count": (
                self.expected_batch_count - self.fit_batch_count
            ),
            "split_assignment_is_noncontiguous_capable": True,
            "selected_layer_indices": [0],
            "field_widths": {
                value.field_name: value.feature_width for value in self.field_contracts
            },
        }
        abi = {
            "request_schema": LFM2_CONDITIONED_RECAPTURE_REQUEST_SCHEMA,
            "checkpoint_schema": LFM2_CONDITIONED_RECAPTURE_CHECKPOINT_SCHEMA,
            "receipt_schema": LFM2_CONDITIONED_RECAPTURE_RECEIPT_SCHEMA,
            "spec_sha256": self.conditioned_recapture_spec_sha256,
            "implementation_sha256": self.implementation_sha256,
        }
        return FoldPhaseKey(
            phase_name=LFM2_CONDITIONED_RECAPTURE_PHASE,
            dependency_sha256s=dependencies,
            numeric_contract=numeric,
            abi=abi,
        )


class Lfm2ConditionedRecapturePhaseAuthority:
    """Runtime-only domain-separated HMAC authority."""

    __slots__ = ("__secret", "key_id")

    def __init__(self, *, key_id: str, secret: bytes) -> None:
        self.key_id = _identifier(key_id, "conditioned recapture HMAC key_id")
        if not isinstance(secret, bytes) or len(secret) < 32:
            raise Lfm2ConditionedRecapturePhaseError(
                "Conditioned recapture HMAC secret must contain at least 32 bytes."
            )
        self.__secret = bytes(secret)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(key_id={self.key_id!r})"

    def __reduce_ex__(self, _protocol: int) -> Any:
        raise TypeError("Conditioned recapture HMAC authority cannot be serialized.")

    def _sign(self, domain: bytes, body: Mapping[str, Any]) -> str:
        return hmac.new(
            self.__secret,
            domain + canonical_json_bytes(dict(body)),
            hashlib.sha256,
        ).hexdigest()

    def sign_checkpoint(self, body: Mapping[str, Any]) -> str:
        return self._sign(_CHECKPOINT_DOMAIN, body)

    def verify_checkpoint(self, body: Mapping[str, Any], claimed: Any) -> bool:
        return bool(
            isinstance(claimed, str)
            and _SHA256.fullmatch(claimed) is not None
            and hmac.compare_digest(self.sign_checkpoint(body), claimed)
        )

    def sign_anchor(self, body: Mapping[str, Any]) -> str:
        return self._sign(_ANCHOR_DOMAIN, body)

    def verify_anchor(self, body: Mapping[str, Any], claimed: Any) -> bool:
        return bool(
            isinstance(claimed, str)
            and _SHA256.fullmatch(claimed) is not None
            and hmac.compare_digest(self.sign_anchor(body), claimed)
        )

    def sign_heldout_capability(self, body: Mapping[str, Any]) -> str:
        return self._sign(_CAPABILITY_DOMAIN, body)

    def verify_heldout_capability(
        self, body: Mapping[str, Any], claimed: Any
    ) -> bool:
        return bool(
            isinstance(claimed, str)
            and _SHA256.fullmatch(claimed) is not None
            and hmac.compare_digest(self.sign_heldout_capability(body), claimed)
        )


@dataclass(frozen=True, slots=True)
class Lfm2ConditionedRecaptureHeldoutAccessCapability:
    """Minimal authenticated proof that candidate state was frozen."""

    phase_key_sha256: str
    completion_sha256: str
    frozen_candidate_barrier_sha256: str
    checkpoint_hmac_key_id: str
    capability_hmac_sha256: str

    def _body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CONDITIONED_RECAPTURE_HELDOUT_CAPABILITY_SCHEMA,
            "phase_key_sha256": self.phase_key_sha256,
            "completion_sha256": self.completion_sha256,
            "frozen_candidate_barrier_sha256": (
                self.frozen_candidate_barrier_sha256
            ),
            "checkpoint_hmac_key_id": self.checkpoint_hmac_key_id,
            "scope": "heldout-read-only",
        }


@dataclass(frozen=True, slots=True)
class Lfm2ConditionedRecaptureProgress:
    phase_key_sha256: str
    total_checkpoints: int
    completed_checkpoints: int
    checkpoint_chain_sha256: str


@dataclass(frozen=True, slots=True)
class Lfm2ConditionedRecaptureFieldRows:
    field_name: str
    split: str
    rows: torch.Tensor = field(repr=False, compare=False)
    record_ids: tuple[str, ...]
    global_row_indices: tuple[int, ...]
    global_batch_indices: tuple[int, ...]
    weights: torch.Tensor = field(repr=False, compare=False)
    source_dtype: str
    feature_width: int
    row_index_sha256: str
    split_identity_sha256: str


def _authenticate_child_fingerprint(
    value: Any, request: Lfm2ConditionedRecapturePhaseRequest
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Replay child fingerprint is missing."
        )
    child = _canonical(dict(value), "replay child fingerprint")
    required = {
        "model_class",
        "config_sha256",
        "state_dict_sha256",
        "state_tensor_count",
        "runtime_buffer_sha256",
        "runtime_buffer_count",
        "identity_sha256",
    }
    if set(child) != required:
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Replay child fingerprint fields changed."
        )
    identity = child.pop("identity_sha256")
    expected = canonical_sha256(
        {"schema": "neural-forge-lfm2-parent-fingerprint-v2", **child}
    )
    if (
        identity != expected
        or identity != request.conditioned_child_identity_sha256
        or child.get("state_dict_sha256")
        != request.conditioned_child_state_dict_sha256
    ):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Replay belongs to a different conditioned child identity or state."
        )
    return {**child, "identity_sha256": identity}


def _authenticate_replay_metadata(
    value: Any,
    request: Lfm2ConditionedRecapturePhaseRequest,
    *,
    batch_start: int,
    batch_stop: int,
) -> dict[str, Any]:
    metadata = _verify_seal(value, "replay_sha256", "child replay metadata")
    if metadata.get("schema") != CHILD_REPLAY_SCHEMA:
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Child replay metadata schema changed."
        )
    _authenticate_child_fingerprint(metadata.get("child"), request)
    count = batch_stop - batch_start
    if metadata.get("batch_count") != count:
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Range replay batch count changed."
        )

    corpus = _verify_seal(metadata.get("corpus"), "corpus_sha256", "replay corpus")
    expected_inputs = list(request.source_input_sha256s[batch_start:batch_stop])
    if (
        corpus.get("schema") != "neural-forge-lfm2-circuit-corpus-v1"
        or corpus.get("batch_count") != count
        or corpus.get("ordered_batch_input_sha256") != expected_inputs
    ):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Range replay corpus differs from the requested source inputs."
        )

    implementation = _verify_seal(
        metadata.get("implementation"),
        "implementation_sha256",
        "child replay implementation",
    )
    if (
        implementation["implementation_sha256"]
        != request.child_replay_implementation_sha256
    ):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Child replay implementation differs from the request."
        )

    topology = _verify_seal(
        metadata.get("layer_topology"),
        "layer_topology_sha256",
        "child layer topology",
    )
    if topology["layer_topology_sha256"] != request.child_layer_topology_sha256:
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Child replay topology differs from the request."
        )

    selection = _verify_seal(
        metadata.get("selection"), "selection_sha256", "child replay selection"
    )
    capture = _verify_seal(
        selection.get("capture"), "selection_sha256", "replay capture config"
    )
    selected_topology = _verify_seal(
        selection.get("selected_topology"),
        "selected_topology_sha256",
        "replay selected topology",
    )
    if (
        selection.get("schema")
        != "neural-forge-lfm2-child-replay-selection-v2"
        or selection.get("selected_layer_indices") != [0]
        or capture["selection_sha256"] != request.child_replay_config_sha256
        or selection.get("replay_capture_mode")
        != request.child_replay_capture_mode
        or selected_topology.get("selected_layer_indices") != [0]
        or selected_topology.get("model_layer_count") != topology.get("layer_count")
        or selected_topology.get("layers") != [topology["layers"][0]]
    ):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Child replay selection differs from the layer-zero request."
        )
    for name, expected in (
        ("inference_mode", True),
        ("model_eval_mode", True),
        ("gradients_used", False),
        ("training_steps", 0),
    ):
        if capture.get(name) != expected:
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "Child replay was not captured under the no-training contract."
            )

    profile_value = selection.get("child_evidence_profile")
    try:
        profile = Lfm2ChildEvidenceProfile.from_dict(profile_value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Child evidence profile is invalid."
        ) from exc
    expected_suffixes = tuple(sorted(LFM2_CONDITIONED_RECAPTURE_REQUIRED_FIELDS))
    if (
        profile.profile_role != LFM2_CONDITIONED_RECAPTURE_PROFILE_ROLE
        or profile.layer_suffixes != ((0, expected_suffixes),)
        or profile.retain_logits
        or profile.retain_lm_head_input
        or profile.retain_phase_forced_token_ids
        or selection.get("child_evidence_profile_sha256") != profile.profile_sha256
    ):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Child replay retained-field inventory differs from the request."
        )

    execution = _verify_seal(
        metadata.get("execution"), "execution_sha256", "child replay execution"
    )
    if (
        execution["execution_sha256"] != request.child_replay_execution_sha256
        or execution.get("input_device_type") != request.runtime_device_type
    ):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Child replay execution policy differs from the request."
        )

    numeric = _verify_seal(
        metadata.get("numeric_execution_policy"),
        "replay_numeric_policy_sha256",
        "child replay numeric policy",
    )
    envelope = _verify_seal(
        numeric.get("non_logit_envelope_policy"),
        "envelope_sha256",
        "child replay non-logit envelope",
    )
    if (
        numeric["replay_numeric_policy_sha256"]
        != request.child_replay_numeric_policy_sha256
        or numeric.get("non_logit_envelope_sha256") != envelope["envelope_sha256"]
    ):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Child replay numeric policy differs from the request."
        )
    policy = numeric.get("numeric_execution_policy")
    if not isinstance(policy, Mapping):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Child replay target dtype inventory is missing."
        )
    policy_body = _verify_seal(policy, "policy_sha256", "numeric execution policy")
    if policy_body["policy_sha256"] != numeric.get("numeric_policy_sha256"):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Child replay numeric execution policy binding changed."
        )
    target_inventory = policy.get("target_inventory")
    floating_counts = (
        target_inventory.get("floating_dtype_counts")
        if isinstance(target_inventory, Mapping)
        else None
    )
    if not isinstance(floating_counts, Mapping) or tuple(sorted(floating_counts)) != (
        request.runtime_floating_dtypes
    ):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Child replay runtime floating dtype inventory changed."
        )

    runtime = metadata.get("runtime")
    if not isinstance(runtime, Mapping) or any(
        runtime.get(name) != expected
        for name, expected in (
            ("tokenizer_accessed", False),
            ("parent_model_supplied", False),
            ("parent_forward_count", 0),
            ("disk_artifacts_written", 0),
            ("intermediate_cache", "memory_only"),
            ("child_state_preserved", True),
        )
    ):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Child replay runtime violates the child-only in-memory contract."
        )
    forward_counts = metadata.get("forward_counts")
    if (
        not isinstance(forward_counts, Mapping)
        or forward_counts.get("parent") != 0
        or isinstance(forward_counts.get("child"), bool)
        or not isinstance(forward_counts.get("child"), int)
        or forward_counts["child"] <= 0
    ):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Child replay forward-count evidence is invalid."
        )
    ordered = metadata.get("ordered_batches")
    if not isinstance(ordered, list) or len(ordered) != count:
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Child replay ordered-batch inventory changed."
        )
    return metadata


def _canonical_phase_order(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or value[0] != "prefill":
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Replay phase order must begin with prefill."
        )
    phases = tuple(value)
    if any(not isinstance(phase, str) or _PHASE.fullmatch(phase) is None for phase in phases):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Replay phase order is malformed."
        )
    expected = ("prefill",) + tuple(
        f"decode.{index:03d}" for index in range(len(phases) - 1)
    )
    if phases != expected:
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Replay decode phases are not contiguous."
        )
    return phases


def _authenticate_batch_record(
    value: Any,
    *,
    local_index: int,
    expected_input_sha256: str,
    expected_capture: Mapping[str, Any],
    expected_profile_sha256: str,
) -> dict[str, Any]:
    record = _verify_seal(value, "replay_batch_sha256", "child replay batch")
    phases = _canonical_phase_order(record.get("phase_order"))
    if (
        record.get("schema") != CHILD_REPLAY_BATCH_SCHEMA
        or record.get("batch_index") != local_index
        or record.get("input_sha256") != expected_input_sha256
        or record.get("selected_layer_indices") != [0]
        or record.get("capture") != dict(expected_capture)
        or record.get("child_evidence_profile_sha256")
        != expected_profile_sha256
    ):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Child replay batch identity or capture contract changed."
        )
    profile = record.get("child_evidence_profile")
    try:
        authenticated_profile = Lfm2ChildEvidenceProfile.from_dict(profile)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Child replay batch evidence profile is invalid."
        ) from exc
    if authenticated_profile.profile_sha256 != expected_profile_sha256:
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Child replay batch evidence profile changed."
        )
    phase_rows = record.get("phase_rows")
    required = {
        "forced_decode_step",
        "phase",
        "prefix_length",
        "sample_count",
        "sample_start",
    }
    if not isinstance(phase_rows, list) or len(phase_rows) != len(phases):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Child replay phase-row inventory changed."
        )
    for phase, row in zip(phases, phase_rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != required
            or row.get("phase") != phase
            or isinstance(row.get("sample_count"), bool)
            or not isinstance(row.get("sample_count"), int)
            or row["sample_count"] <= 0
            or isinstance(row.get("sample_start"), bool)
            or not isinstance(row.get("sample_start"), int)
            or row["sample_start"] < 0
        ):
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "Child replay phase-row geometry is invalid."
            )
    return record


def _authenticate_source_replay(
    replay: Any,
    request: Lfm2ConditionedRecapturePhaseRequest,
    *,
    batch_start: int,
    batch_stop: int,
) -> tuple[dict[str, Any], tuple[tuple[dict[str, Any], Mapping[str, torch.Tensor]], ...]]:
    if not isinstance(replay, Lfm2ChildReplay):
        raise TypeError("range_provider must return Lfm2ChildReplay.")
    metadata = _authenticate_replay_metadata(
        replay.metadata,
        request,
        batch_start=batch_start,
        batch_stop=batch_stop,
    )
    capture = metadata["selection"]["capture"]
    profile_sha = metadata["selection"]["child_evidence_profile_sha256"]
    ordered = metadata["ordered_batches"]
    expected_layer_fields = {
        f"{{phase}}.layer.000.{name}"
        for name in LFM2_CONDITIONED_RECAPTURE_REQUIRED_FIELDS
    }
    authenticated: list[tuple[dict[str, Any], Mapping[str, torch.Tensor]]] = []
    batches = replay.batches
    if len(batches) != batch_stop - batch_start:
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Range replay object and metadata batch counts differ."
        )
    for local_index, batch in enumerate(batches):
        expected_input = request.source_input_sha256s[batch_start + local_index]
        if batch.index != local_index or batch.input_sha256 != expected_input:
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "Range provider did not use canonical local replay indices."
            )
        record = _authenticate_batch_record(
            batch.record,
            local_index=local_index,
            expected_input_sha256=expected_input,
            expected_capture=capture,
            expected_profile_sha256=profile_sha,
        )
        summary = ordered[local_index]
        if (
            not isinstance(summary, Mapping)
            or summary.get("batch_index") != local_index
            or summary.get("input_sha256") != expected_input
            or summary.get("replay_batch_sha256")
            != record["replay_batch_sha256"]
            or summary.get("phase_order") != record["phase_order"]
            or summary.get("child_evidence_profile_sha256") != profile_sha
        ):
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "Replay metadata-to-batch chain changed."
            )
        raw_tensors = batch.tensors
        inventory = record.get("tensors")
        if not isinstance(raw_tensors, Mapping) or not isinstance(inventory, Mapping):
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "Replay batch tensor inventory is missing."
            )
        names = tuple(sorted(raw_tensors))
        if record.get("tensor_count") != len(names) or tuple(sorted(inventory)) != names:
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "Replay batch tensor names differ from its record."
            )
        phases = tuple(record["phase_order"])
        allowed = {
            "input.input_ids",
            "input.attention_mask",
            "input.forced_decode_ids",
            "input.position_ids",
        }
        allowed.update(
            template.format(phase=phase)
            for phase in phases
            for template in expected_layer_fields
        )
        required = allowed - {"input.position_ids"}
        if not required.issubset(names) or set(names) - allowed:
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "Replay retained tensor inventory is not exact."
            )
        clones: dict[str, torch.Tensor] = {}
        for name in names:
            value = raw_tensors[name]
            if not torch.is_tensor(value) or value.requires_grad:
                raise Lfm2ConditionedRecaptureAuthenticationError(
                    "Replay tensors must be detached no-grad evidence."
                )
            clone = value.detach().to(device="cpu").contiguous().clone()
            if _trace_tensor_descriptor(clone) != dict(inventory[name]):
                raise Lfm2ConditionedRecaptureAuthenticationError(
                    f"Replay tensor {name!r} differs from its descriptor."
                )
            if torch.is_floating_point(clone) and not bool(torch.isfinite(clone).all()):
                raise Lfm2ConditionedRecaptureAuthenticationError(
                    f"Replay tensor {name!r} contains non-finite values."
                )
            clones[name] = clone
        for phase, phase_row in zip(phases, record["phase_rows"]):
            count = phase_row["sample_count"]
            for contract in request.field_contracts:
                name = f"{phase}.layer.000.{contract.field_name}"
                tensor = clones[name]
                if (
                    tensor.ndim != 3
                    or tuple(tensor.shape) != (1, count, contract.feature_width)
                    or _torch_dtype_name(tensor) != contract.source_dtype
                ):
                    raise Lfm2ConditionedRecaptureAuthenticationError(
                        f"Replay tensor {name!r} violates its width/dtype contract."
                    )
        authenticated.append((record, MappingProxyType(clones)))
    return metadata, tuple(authenticated)


def _checkpoint_count(request: Lfm2ConditionedRecapturePhaseRequest) -> int:
    return request.expected_batch_count // request.checkpoint_batch_count


def _checkpoint_path(root: Path, ordinal: int) -> Path:
    return root / "checkpoints" / f"checkpoint-{ordinal:04d}.json"


def _shard_path(root: Path, ordinal: int) -> Path:
    return root / "shards" / f"conditioned-recapture-{ordinal:04d}.safetensors"


def _checkpoint_chain_sha256(checkpoints: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256(
        [value["checkpoint_hmac_sha256"] for value in checkpoints]
    )


def _checkpoint_rows_and_tensors(
    request: Lfm2ConditionedRecapturePhaseRequest,
    authenticated_batches: Sequence[
        tuple[Mapping[str, Any], Mapping[str, torch.Tensor]]
    ],
    *,
    batch_start: int,
    row_start: int,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    rows: list[dict[str, Any]] = []
    pieces: dict[str, list[torch.Tensor]] = {
        value.field_name: [] for value in request.field_contracts
    }
    global_row_index = row_start
    for local_index, (record, tensors) in enumerate(authenticated_batches):
        global_batch_index = batch_start + local_index
        record_id = request.source_record_ids[global_batch_index]
        split = request.basis_split_by_batch[global_batch_index]
        for phase, phase_row in zip(record["phase_order"], record["phase_rows"]):
            sample_start = int(phase_row["sample_start"])
            sample_count = int(phase_row["sample_count"])
            for offset in range(sample_count):
                rows.append(
                    {
                        "global_row_index": global_row_index,
                        "global_batch_index": global_batch_index,
                        "local_replay_batch_index": local_index,
                        "record_id": record_id,
                        "input_sha256": request.source_input_sha256s[
                            global_batch_index
                        ],
                        "basis_split": split,
                        "phase": phase,
                        "phase_row_index": offset,
                        "source_sample_index": sample_start + offset,
                    }
                )
                global_row_index += 1
            for contract in request.field_contracts:
                name = f"{phase}.layer.000.{contract.field_name}"
                pieces[contract.field_name].append(
                    tensors[name][0].detach().to(device="cpu").contiguous().clone()
                )
    stored = {
        name: torch.cat(values, dim=0).contiguous()
        for name, values in pieces.items()
    }
    for contract in request.field_contracts:
        tensor = stored[contract.field_name]
        if tuple(tensor.shape) != (len(rows), contract.feature_width):
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "Checkpoint tensor rows disagree with replay phase geometry."
            )
    return rows, stored


def _write_checkpoint(
    root: Path,
    request: Lfm2ConditionedRecapturePhaseRequest,
    authority: Lfm2ConditionedRecapturePhaseAuthority,
    *,
    ordinal: int,
    batch_start: int,
    batch_stop: int,
    row_start: int,
    previous_checkpoint_hmac_sha256: str,
    metadata: Mapping[str, Any],
    authenticated_batches: Sequence[
        tuple[Mapping[str, Any], Mapping[str, torch.Tensor]]
    ],
) -> dict[str, Any]:
    rows, tensors = _checkpoint_rows_and_tensors(
        request,
        authenticated_batches,
        batch_start=batch_start,
        row_start=row_start,
    )
    shard_path = _shard_path(root, ordinal)
    checkpoint_path = _checkpoint_path(root, ordinal)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_shard = shard_path.with_name(
        f".{shard_path.name}.{os.getpid()}.tmp"
    )
    try:
        from safetensors.torch import save_file

        save_file(
            tensors,
            str(temporary_shard),
            metadata={
                "schema": LFM2_CONDITIONED_RECAPTURE_SHARD_SCHEMA,
                "request_sha256": request.request_sha256,
                "checkpoint_ordinal": str(ordinal),
            },
        )
        os.replace(temporary_shard, shard_path)
        descriptors = {
            name: _stored_tensor_descriptor(tensor)
            for name, tensor in sorted(tensors.items())
        }
        shard = {
            "filename": shard_path.name,
            "size_bytes": shard_path.stat().st_size,
            "file_sha256": _file_sha256(shard_path),
            "tensor_descriptors": descriptors,
            "tensor_inventory_sha256": canonical_sha256(descriptors),
        }
        source_batches = [
            {
                "global_batch_index": batch_start + local_index,
                "local_replay_batch_index": local_index,
                "input_sha256": request.source_input_sha256s[
                    batch_start + local_index
                ],
                "record_id": request.source_record_ids[batch_start + local_index],
                "record": dict(record),
            }
            for local_index, (record, _tensors) in enumerate(authenticated_batches)
        ]
        execution = {
            **_ZERO_OPERATION_AUDIT,
            "range_provider_calls": 1,
            "child_model_forward_calls": metadata["forward_counts"]["child"],
            "child_replay_objects_authenticated": 1,
            "child_replay_batches_authenticated": batch_stop - batch_start,
            "all_source_tensor_bytes_authenticated": True,
            "provider_called_under_inference_mode": True,
            "requires_grad_outputs_rejected": True,
        }
        body = {
            "schema": LFM2_CONDITIONED_RECAPTURE_CHECKPOINT_SCHEMA,
            "phase_key_sha256": request.phase_key.phase_key_sha256,
            "request_sha256": request.request_sha256,
            "checkpoint_hmac_key_id": authority.key_id,
            "checkpoint_ordinal": ordinal,
            "batch_start": batch_start,
            "batch_stop": batch_stop,
            "row_start": row_start,
            "row_stop": row_start + len(rows),
            "previous_checkpoint_hmac_sha256": (
                previous_checkpoint_hmac_sha256
            ),
            "source_replay_metadata": dict(metadata),
            "source_batches": source_batches,
            "rows": rows,
            "row_slice_sha256": canonical_sha256(rows),
            "shard": shard,
            "execution": execution,
        }
        checkpoint_sha256 = canonical_sha256(body)
        signed = {**body, "checkpoint_sha256": checkpoint_sha256}
        document = {
            **signed,
            "checkpoint_hmac_sha256": authority.sign_checkpoint(signed),
        }
        _write_json(checkpoint_path, document)
        authenticated = _authenticate_checkpoint(
            checkpoint_path,
            request=request,
            authority=authority,
            expected_ordinal=ordinal,
            expected_row_start=row_start,
            expected_previous_hmac_sha256=previous_checkpoint_hmac_sha256,
            shard_root=shard_path.parent,
        )
        if authenticated != document:
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "Persisted checkpoint changed before admission."
            )
        return authenticated
    except BaseException:
        checkpoint_path.unlink(missing_ok=True)
        shard_path.unlink(missing_ok=True)
        raise
    finally:
        temporary_shard.unlink(missing_ok=True)


def _authenticate_checkpoint(
    path: Path,
    *,
    request: Lfm2ConditionedRecapturePhaseRequest,
    authority: Lfm2ConditionedRecapturePhaseAuthority,
    expected_ordinal: int,
    expected_row_start: int,
    expected_previous_hmac_sha256: str,
    shard_root: Path,
) -> dict[str, Any]:
    document = _read_json(path)
    signed = dict(document)
    claimed_hmac = signed.pop("checkpoint_hmac_sha256", None)
    if not authority.verify_checkpoint(signed, claimed_hmac):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Conditioned-recapture checkpoint HMAC is invalid."
        )
    body = dict(signed)
    claimed_sha = body.pop("checkpoint_sha256", None)
    if not isinstance(claimed_sha, str) or canonical_sha256(body) != claimed_sha:
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Conditioned-recapture checkpoint self-seal is invalid."
        )
    expected_start = expected_ordinal * request.checkpoint_batch_count
    expected_stop = expected_start + request.checkpoint_batch_count
    if (
        body.get("schema") != LFM2_CONDITIONED_RECAPTURE_CHECKPOINT_SCHEMA
        or body.get("phase_key_sha256") != request.phase_key.phase_key_sha256
        or body.get("request_sha256") != request.request_sha256
        or body.get("checkpoint_hmac_key_id") != authority.key_id
        or body.get("checkpoint_ordinal") != expected_ordinal
        or body.get("batch_start") != expected_start
        or body.get("batch_stop") != expected_stop
        or body.get("row_start") != expected_row_start
        or body.get("previous_checkpoint_hmac_sha256")
        != expected_previous_hmac_sha256
    ):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Conditioned-recapture checkpoint chain coordinates changed."
        )
    metadata = _authenticate_replay_metadata(
        body.get("source_replay_metadata"),
        request,
        batch_start=expected_start,
        batch_stop=expected_stop,
    )
    source_batches = body.get("source_batches")
    if not isinstance(source_batches, list) or len(source_batches) != (
        request.checkpoint_batch_count
    ):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Checkpoint source-batch inventory changed."
        )
    expected_rows: list[dict[str, Any]] = []
    next_row = expected_row_start
    capture = metadata["selection"]["capture"]
    profile_sha = metadata["selection"]["child_evidence_profile_sha256"]
    for local_index, source in enumerate(source_batches):
        global_index = expected_start + local_index
        if (
            not isinstance(source, Mapping)
            or source.get("global_batch_index") != global_index
            or source.get("local_replay_batch_index") != local_index
            or source.get("input_sha256")
            != request.source_input_sha256s[global_index]
            or source.get("record_id") != request.source_record_ids[global_index]
        ):
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "Checkpoint local-to-global replay remap changed."
            )
        record = _authenticate_batch_record(
            source.get("record"),
            local_index=local_index,
            expected_input_sha256=request.source_input_sha256s[global_index],
            expected_capture=capture,
            expected_profile_sha256=profile_sha,
        )
        summary = metadata["ordered_batches"][local_index]
        if (
            summary.get("replay_batch_sha256")
            != record["replay_batch_sha256"]
            or summary.get("input_sha256") != record["input_sha256"]
            or summary.get("phase_order") != record["phase_order"]
        ):
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "Checkpoint replay metadata-to-record binding changed."
            )
        inventory = record.get("tensors")
        if not isinstance(inventory, Mapping):
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "Checkpoint source tensor descriptors are missing."
            )
        split = request.basis_split_by_batch[global_index]
        for phase, phase_row in zip(record["phase_order"], record["phase_rows"]):
            sample_start = phase_row["sample_start"]
            sample_count = phase_row["sample_count"]
            for contract in request.field_contracts:
                source_name = f"{phase}.layer.000.{contract.field_name}"
                descriptor = inventory.get(source_name)
                if not isinstance(descriptor, Mapping):
                    raise Lfm2ConditionedRecaptureAuthenticationError(
                        f"Source descriptor {source_name!r} is missing."
                    )
                validated = _validate_descriptor(
                    descriptor, f"source descriptor {source_name}", floating=True
                )
                if (
                    validated["shape"]
                    != [1, sample_count, contract.feature_width]
                    or validated["dtype"] != contract.source_dtype
                ):
                    raise Lfm2ConditionedRecaptureAuthenticationError(
                        "Source tensor descriptor violates field contract."
                    )
            for offset in range(sample_count):
                expected_rows.append(
                    {
                        "global_row_index": next_row,
                        "global_batch_index": global_index,
                        "local_replay_batch_index": local_index,
                        "record_id": request.source_record_ids[global_index],
                        "input_sha256": request.source_input_sha256s[global_index],
                        "basis_split": split,
                        "phase": phase,
                        "phase_row_index": offset,
                        "source_sample_index": sample_start + offset,
                    }
                )
                next_row += 1
    if (
        body.get("rows") != expected_rows
        or body.get("row_slice_sha256") != canonical_sha256(expected_rows)
        or body.get("row_stop") != next_row
    ):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Checkpoint row mapping changed."
        )

    shard = body.get("shard")
    if not isinstance(shard, Mapping):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Checkpoint shard descriptor is missing."
        )
    filename = shard.get("filename")
    expected_filename = _shard_path(Path("."), expected_ordinal).name
    if filename != expected_filename:
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Checkpoint shard filename changed."
        )
    shard_path = shard_root / filename
    if (
        not shard_path.is_file()
        or shard_path.stat().st_size != shard.get("size_bytes")
        or _file_sha256(shard_path) != shard.get("file_sha256")
    ):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Checkpoint safetensors shard bytes changed."
        )
    descriptors = shard.get("tensor_descriptors")
    expected_names = tuple(sorted(LFM2_CONDITIONED_RECAPTURE_REQUIRED_FIELDS))
    if (
        not isinstance(descriptors, Mapping)
        or tuple(sorted(descriptors)) != expected_names
        or shard.get("tensor_inventory_sha256") != canonical_sha256(descriptors)
    ):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Checkpoint shard tensor inventory changed."
        )
    from safetensors import safe_open

    with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
        if (handle.metadata() or {}) != {
            "schema": LFM2_CONDITIONED_RECAPTURE_SHARD_SCHEMA,
            "request_sha256": request.request_sha256,
            "checkpoint_ordinal": str(expected_ordinal),
        } or tuple(sorted(handle.keys())) != expected_names:
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "Checkpoint safetensors metadata or keys changed."
            )
        for contract in request.field_contracts:
            tensor = handle.get_tensor(contract.field_name).contiguous()
            if (
                _stored_tensor_descriptor(tensor)
                != _canonical(descriptors[contract.field_name], "shard descriptor")
                or tuple(tensor.shape) != (len(expected_rows), contract.feature_width)
                or _torch_dtype_name(tensor) != contract.source_dtype
                or not bool(torch.isfinite(tensor).all())
            ):
                raise Lfm2ConditionedRecaptureAuthenticationError(
                    f"Checkpoint field {contract.field_name!r} bytes changed."
                )
    execution = body.get("execution")
    if (
        not isinstance(execution, Mapping)
        or execution.get("range_provider_calls") != 1
        or execution.get("child_model_forward_calls")
        != metadata["forward_counts"]["child"]
        or any(execution.get(name) != 0 for name in _ZERO_OPERATION_AUDIT)
        or execution.get("all_source_tensor_bytes_authenticated") is not True
        or execution.get("provider_called_under_inference_mode") is not True
        or execution.get("requires_grad_outputs_rejected") is not True
    ):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Checkpoint operation audit changed."
        )
    return document


def _load_checkpoint_prefix(
    root: Path,
    *,
    request: Lfm2ConditionedRecapturePhaseRequest,
    authority: Lfm2ConditionedRecapturePhaseAuthority,
    shard_root: Path | None = None,
) -> tuple[dict[str, Any], ...]:
    checkpoints: list[dict[str, Any]] = []
    previous_hmac = _ZERO_SHA256
    row_start = 0
    total = _checkpoint_count(request)
    selected_shard_root = root / "shards" if shard_root is None else shard_root
    missing_seen = False
    for ordinal in range(total):
        path = _checkpoint_path(root, ordinal)
        if not path.exists():
            missing_seen = True
            continue
        if missing_seen:
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "Checkpoint prefix contains a gap."
            )
        checkpoint = _authenticate_checkpoint(
            path,
            request=request,
            authority=authority,
            expected_ordinal=ordinal,
            expected_row_start=row_start,
            expected_previous_hmac_sha256=previous_hmac,
            shard_root=selected_shard_root,
        )
        checkpoints.append(checkpoint)
        previous_hmac = checkpoint["checkpoint_hmac_sha256"]
        row_start = checkpoint["row_stop"]
    checkpoint_dir = root / "checkpoints"
    if checkpoint_dir.exists():
        expected_files = {
            _checkpoint_path(root, ordinal).name for ordinal in range(len(checkpoints))
        }
        actual_files = {value.name for value in checkpoint_dir.iterdir()}
        if actual_files != expected_files:
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "Checkpoint directory contains unauthenticated files."
            )
    shard_dir = selected_shard_root
    if shard_dir.exists():
        expected_shards = {
            _shard_path(root, ordinal).name for ordinal in range(len(checkpoints))
        }
        actual_shards = {value.name for value in shard_dir.iterdir()}
        if actual_shards != expected_shards:
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "Shard directory contains unauthenticated files."
            )
    return tuple(checkpoints)


def _capture_missing_prefix(
    request: Lfm2ConditionedRecapturePhaseRequest,
    authority: Lfm2ConditionedRecapturePhaseAuthority,
    root: Path,
    checkpoints: Sequence[Mapping[str, Any]],
    range_provider: Callable[[int, int], Lfm2ChildReplay],
    *,
    maximum_new_checkpoints: int | None,
) -> tuple[dict[str, Any], ...]:
    completed = [dict(value) for value in checkpoints]
    total = _checkpoint_count(request)
    stop_ordinal = total
    if maximum_new_checkpoints is not None:
        stop_ordinal = min(total, len(completed) + maximum_new_checkpoints)
    for ordinal in range(len(completed), stop_ordinal):
        batch_start = ordinal * request.checkpoint_batch_count
        batch_stop = batch_start + request.checkpoint_batch_count
        with torch.inference_mode():
            if torch.is_grad_enabled():  # pragma: no cover - PyTorch invariant
                raise Lfm2ConditionedRecaptureAuthenticationError(
                    "Provider inference guard did not disable gradients."
                )
            replay = range_provider(batch_start, batch_stop)
        metadata, authenticated_batches = _authenticate_source_replay(
            replay,
            request,
            batch_start=batch_start,
            batch_stop=batch_stop,
        )
        row_start = 0 if not completed else int(completed[-1]["row_stop"])
        previous = (
            _ZERO_SHA256
            if not completed
            else str(completed[-1]["checkpoint_hmac_sha256"])
        )
        completed.append(
            _write_checkpoint(
                root,
                request,
                authority,
                ordinal=ordinal,
                batch_start=batch_start,
                batch_stop=batch_stop,
                row_start=row_start,
                previous_checkpoint_hmac_sha256=previous,
                metadata=metadata,
                authenticated_batches=authenticated_batches,
            )
        )
    return tuple(completed)


def _complete_phase_documents(
    request: Lfm2ConditionedRecapturePhaseRequest,
    authority: Lfm2ConditionedRecapturePhaseAuthority,
    checkpoints: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if len(checkpoints) != _checkpoint_count(request):
        raise Lfm2ConditionedRecapturePhaseError(
            "Conditioned recapture is not complete."
        )
    rows = [dict(row) for checkpoint in checkpoints for row in checkpoint["rows"]]
    if tuple(row["global_row_index"] for row in rows) != tuple(range(len(rows))):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Completed conditioned-recapture rows are not contiguous."
        )
    fit_batches = {
        int(row["global_batch_index"])
        for row in rows
        if row["basis_split"] == "fit"
    }
    heldout_batches = {
        int(row["global_batch_index"])
        for row in rows
        if row["basis_split"] == "heldout"
    }
    expected_fit_batches = {
        index
        for index, split in enumerate(request.basis_split_by_batch)
        if split == "fit"
    }
    expected_heldout_batches = {
        index
        for index, split in enumerate(request.basis_split_by_batch)
        if split == "heldout"
    }
    if (
        len(expected_fit_batches) != 192
        or len(expected_heldout_batches) != 64
        or fit_batches != expected_fit_batches
        or heldout_batches != expected_heldout_batches
        or fit_batches & heldout_batches
        or fit_batches | heldout_batches != set(range(256))
    ):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Completed conditioned-recapture split coverage changed."
        )
    row_body = {
        "schema": LFM2_CONDITIONED_RECAPTURE_ROW_INDEX_SCHEMA,
        "phase_key_sha256": request.phase_key.phase_key_sha256,
        "request_sha256": request.request_sha256,
        "split_identity_sha256": request.split_identity_sha256,
        "basis_split_by_batch": list(request.basis_split_by_batch),
        "fit_batch_count": len(expected_fit_batches),
        "heldout_batch_count": len(expected_heldout_batches),
        "row_count": len(rows),
        "fit_row_count": sum(row["basis_split"] == "fit" for row in rows),
        "heldout_row_count": sum(
            row["basis_split"] == "heldout" for row in rows
        ),
        "rows": rows,
    }
    row_index = {**row_body, "row_index_sha256": canonical_sha256(row_body)}
    summaries = [
        {
            "checkpoint_ordinal": checkpoint["checkpoint_ordinal"],
            "batch_start": checkpoint["batch_start"],
            "batch_stop": checkpoint["batch_stop"],
            "row_start": checkpoint["row_start"],
            "row_stop": checkpoint["row_stop"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "checkpoint_hmac_sha256": checkpoint["checkpoint_hmac_sha256"],
            "source_replay_sha256": checkpoint["source_replay_metadata"][
                "replay_sha256"
            ],
            "shard": dict(checkpoint["shard"]),
        }
        for checkpoint in checkpoints
    ]
    historical_audit = {
        **_ZERO_OPERATION_AUDIT,
        "range_provider_calls": len(checkpoints),
        "child_model_forward_calls": sum(
            int(value["execution"]["child_model_forward_calls"])
            for value in checkpoints
        ),
        "child_replay_objects_authenticated": len(checkpoints),
        "child_replay_batches_authenticated": request.expected_batch_count,
    }
    receipt_body = {
        "schema": LFM2_CONDITIONED_RECAPTURE_RECEIPT_SCHEMA,
        "phase_key_sha256": request.phase_key.phase_key_sha256,
        "request_sha256": request.request_sha256,
        "v5_request_sha256": request.v5_request_sha256,
        "v5_lineage_sha256": request.v5_lineage_sha256,
        "endpoint_chain_receipt_sha256": request.endpoint_chain_receipt_sha256,
        "operator_replacement_tensor_descriptor": _deep_thaw(
            request.operator_replacement_tensor_descriptor
        ),
        "operator_solution_sha256": request.operator_solution_sha256,
        "frozen_conditioned_child_state_receipt_sha256": (
            request.frozen_conditioned_child_state_receipt_sha256
        ),
        "conditioned_child_identity_sha256": (
            request.conditioned_child_identity_sha256
        ),
        "conditioned_child_state_dict_sha256": (
            request.conditioned_child_state_dict_sha256
        ),
        "split_identity_sha256": request.split_identity_sha256,
        "basis_split_by_batch": list(request.basis_split_by_batch),
        "fit_batch_indices": sorted(expected_fit_batches),
        "heldout_batch_indices": sorted(expected_heldout_batches),
        "row_index_sha256": row_index["row_index_sha256"],
        "checkpoint_count": len(checkpoints),
        "checkpoint_chain_sha256": _checkpoint_chain_sha256(checkpoints),
        "ordered_checkpoints": summaries,
        "historical_operation_audit": historical_audit,
        "fit_read_requires_barrier": False,
        "heldout_read_requires_frozen_candidate_barrier": True,
    }
    receipt = {
        **receipt_body,
        "receipt_sha256": canonical_sha256(receipt_body),
    }
    anchor_body = {
        "schema": LFM2_CONDITIONED_RECAPTURE_ANCHOR_SCHEMA,
        "phase_key_sha256": request.phase_key.phase_key_sha256,
        "request_sha256": request.request_sha256,
        "receipt_sha256": receipt["receipt_sha256"],
        "row_index_sha256": row_index["row_index_sha256"],
        "checkpoint_chain_sha256": receipt["checkpoint_chain_sha256"],
        "final_checkpoint_hmac_sha256": checkpoints[-1][
            "checkpoint_hmac_sha256"
        ],
        "checkpoint_hmac_key_id": authority.key_id,
    }
    anchor_sha256 = canonical_sha256(anchor_body)
    signed_anchor = {**anchor_body, "anchor_sha256": anchor_sha256}
    anchor = {
        **signed_anchor,
        "anchor_hmac_sha256": authority.sign_anchor(signed_anchor),
    }
    result = {
        "schema": LFM2_CONDITIONED_RECAPTURE_RECEIPT_SCHEMA,
        "phase_key_sha256": request.phase_key.phase_key_sha256,
        "request_sha256": request.request_sha256,
        "receipt_sha256": receipt["receipt_sha256"],
        "row_index_sha256": row_index["row_index_sha256"],
        "anchor_sha256": anchor_sha256,
        "checkpoint_chain_sha256": receipt["checkpoint_chain_sha256"],
        "conditioned_child_identity_sha256": (
            request.conditioned_child_identity_sha256
        ),
        "conditioned_child_state_dict_sha256": (
            request.conditioned_child_state_dict_sha256
        ),
        "split_identity_sha256": request.split_identity_sha256,
        "fit_batch_indices": sorted(expected_fit_batches),
        "heldout_batch_indices": sorted(expected_heldout_batches),
        "fit_batch_count": len(expected_fit_batches),
        "heldout_batch_count": len(expected_heldout_batches),
        "row_count": len(rows),
    }
    return row_index, receipt, anchor, result


def _verify_completed_documents(
    request: Lfm2ConditionedRecapturePhaseRequest,
    authority: Lfm2ConditionedRecapturePhaseAuthority,
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    row_index: Mapping[str, Any],
    receipt: Mapping[str, Any],
    anchor: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    expected = _complete_phase_documents(request, authority, checkpoints)
    if tuple(map(dict, (row_index, receipt, anchor, result))) != tuple(expected):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Completed conditioned-recapture documents changed."
        )
    signed_anchor = dict(anchor)
    claimed_hmac = signed_anchor.pop("anchor_hmac_sha256", None)
    if not authority.verify_anchor(signed_anchor, claimed_hmac):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Conditioned-recapture source anchor HMAC is invalid."
        )
    anchor_body = dict(signed_anchor)
    claimed_sha = anchor_body.pop("anchor_sha256", None)
    if claimed_sha != canonical_sha256(anchor_body):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Conditioned-recapture source anchor self-seal is invalid."
        )


def _publish_complete_phase(
    cache: DynamicFoldPhaseCache,
    request: Lfm2ConditionedRecapturePhaseRequest,
    authority: Lfm2ConditionedRecapturePhaseAuthority,
    work_root: Path,
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    events: FoldPhaseEventLog | None,
) -> VerifiedFoldPhaseCacheEntry:
    row_index, receipt, anchor, result = _complete_phase_documents(
        request, authority, checkpoints
    )
    try:
        transaction = cache.begin(request.phase_key, events=events)
    except FoldPhaseCacheAlreadyCompleteError:
        entry = cache.lookup(request.phase_key, events=events)
        if entry is None:  # pragma: no cover - guarded by locked begin
            raise Lfm2ConditionedRecapturePhaseError(
                "Concurrent conditioned-recapture publication disappeared."
            )
        return entry
    with transaction:
        payload = transaction.payload_path
        (payload / "checkpoints").mkdir()
        (payload / "shards").mkdir()
        _write_json(payload / "request.json", request.as_dict())
        _write_json(payload / "row-index.json", row_index)
        _write_json(payload / "receipt.json", receipt)
        _write_json(payload / "source-anchor.json", anchor)
        for ordinal in range(len(checkpoints)):
            shutil.copyfile(
                _checkpoint_path(work_root, ordinal),
                _checkpoint_path(payload, ordinal),
            )
            shutil.copyfile(
                _shard_path(work_root, ordinal),
                _shard_path(payload, ordinal),
            )
        staged_checkpoints = _load_checkpoint_prefix(
            payload,
            request=request,
            authority=authority,
            shard_root=payload / "shards",
        )
        staged_row = _read_json(payload / "row-index.json")
        staged_receipt = _read_json(payload / "receipt.json")
        staged_anchor = _read_json(payload / "source-anchor.json")
        if _read_json(payload / "request.json") != request.as_dict():
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "Staged conditioned-recapture request changed."
            )
        _verify_completed_documents(
            request,
            authority,
            staged_checkpoints,
            row_index=staged_row,
            receipt=staged_receipt,
            anchor=staged_anchor,
            result=result,
        )
        if staged_checkpoints != tuple(checkpoints):
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "Staged checkpoint chain changed before publication."
            )
        return transaction.complete(result=result)


@dataclass(frozen=True, slots=True)
class AuthenticatedLfm2ConditionedRecapturePhase:
    """Completed recapture with authenticated lazy split/field access."""

    entry: VerifiedFoldPhaseCacheEntry = field(repr=False, compare=False)
    request: Lfm2ConditionedRecapturePhaseRequest
    receipt: Mapping[str, Any]
    row_index: Mapping[str, Any]
    source_anchor: Mapping[str, Any]
    authority: Lfm2ConditionedRecapturePhaseAuthority = field(
        repr=False, compare=False
    )

    @property
    def receipt_sha256(self) -> str:
        return str(self.receipt["receipt_sha256"])

    @property
    def row_index_sha256(self) -> str:
        return str(self.row_index["row_index_sha256"])

    @property
    def invocation_audit(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                **_ZERO_OPERATION_AUDIT,
                "range_provider_calls": 0,
                "child_model_forward_calls": 0,
                "child_replay_objects_authenticated": 0,
                "child_replay_batches_authenticated": 0,
            }
        )

    def verify_authentication(self) -> bool:
        try:
            loaded = load_existing_lfm2_conditioned_recapture_phase(
                DynamicFoldPhaseCache(self.entry.path.parents[1]),
                self.request,
                authority=self.authority,
            )
            return bool(
                loaded.entry.path == self.entry.path
                and loaded.entry.completion_sha256 == self.entry.completion_sha256
                and dict(loaded.receipt) == dict(self.receipt)
                and dict(loaded.row_index) == dict(self.row_index)
                and dict(loaded.source_anchor) == dict(self.source_anchor)
            )
        except (OSError, TypeError, ValueError, RuntimeError):
            return False

    def issue_heldout_access_capability(
        self, *, frozen_candidate_barrier_sha256: str
    ) -> Lfm2ConditionedRecaptureHeldoutAccessCapability:
        barrier = _digest(
            frozen_candidate_barrier_sha256, "frozen_candidate_barrier_sha256"
        )
        if barrier == _ZERO_SHA256:
            raise Lfm2ConditionedRecapturePhaseError(
                "Frozen-candidate barrier digest must be nonzero."
            )
        body = {
            "schema": LFM2_CONDITIONED_RECAPTURE_HELDOUT_CAPABILITY_SCHEMA,
            "phase_key_sha256": self.request.phase_key.phase_key_sha256,
            "completion_sha256": self.entry.completion_sha256,
            "frozen_candidate_barrier_sha256": barrier,
            "checkpoint_hmac_key_id": self.authority.key_id,
            "scope": "heldout-read-only",
        }
        return Lfm2ConditionedRecaptureHeldoutAccessCapability(
            phase_key_sha256=body["phase_key_sha256"],
            completion_sha256=body["completion_sha256"],
            frozen_candidate_barrier_sha256=barrier,
            checkpoint_hmac_key_id=self.authority.key_id,
            capability_hmac_sha256=self.authority.sign_heldout_capability(body),
        )

    def _authenticate_heldout_capability(
        self,
        capability: Lfm2ConditionedRecaptureHeldoutAccessCapability | None,
    ) -> None:
        if not isinstance(
            capability, Lfm2ConditionedRecaptureHeldoutAccessCapability
        ):
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "HELDOUT access requires a frozen-candidate barrier capability."
            )
        body = capability._body()
        if (
            capability.phase_key_sha256
            != self.request.phase_key.phase_key_sha256
            or capability.completion_sha256 != self.entry.completion_sha256
            or capability.checkpoint_hmac_key_id != self.authority.key_id
            or capability.frozen_candidate_barrier_sha256 == _ZERO_SHA256
            or not self.authority.verify_heldout_capability(
                body, capability.capability_hmac_sha256
            )
        ):
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "HELDOUT frozen-candidate capability is invalid."
            )

    def _load_field(
        self,
        contract: Lfm2ConditionedRecaptureFieldContract,
        selected_rows: Sequence[Mapping[str, Any]],
    ) -> torch.Tensor:
        selected = {int(row["global_row_index"]) for row in selected_rows}
        pieces: list[torch.Tensor] = []
        from safetensors import safe_open

        for summary in self.receipt["ordered_checkpoints"]:
            start = int(summary["row_start"])
            stop = int(summary["row_stop"])
            local = [index - start for index in range(start, stop) if index in selected]
            if not local:
                continue
            descriptor = summary["shard"]["tensor_descriptors"].get(
                contract.field_name
            )
            if not isinstance(descriptor, Mapping):
                raise Lfm2ConditionedRecaptureAuthenticationError(
                    f"Field {contract.field_name!r} is absent from a shard."
                )
            path = self.entry.payload_path / "shards" / summary["shard"]["filename"]
            if (
                path.stat().st_size != summary["shard"]["size_bytes"]
                or _file_sha256(path) != summary["shard"]["file_sha256"]
            ):
                raise Lfm2ConditionedRecaptureAuthenticationError(
                    "Conditioned-recapture shard bytes changed during field access."
                )
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                if (handle.metadata() or {}) != {
                    "schema": LFM2_CONDITIONED_RECAPTURE_SHARD_SCHEMA,
                    "request_sha256": self.request.request_sha256,
                    "checkpoint_ordinal": str(summary["checkpoint_ordinal"]),
                }:
                    raise Lfm2ConditionedRecaptureAuthenticationError(
                        "Conditioned-recapture shard metadata changed."
                    )
                tensor = handle.get_tensor(contract.field_name).contiguous()
            if _stored_tensor_descriptor(tensor) != _deep_thaw(descriptor):
                raise Lfm2ConditionedRecaptureAuthenticationError(
                    f"Conditioned-recapture field {contract.field_name!r} changed."
                )
            pieces.append(
                tensor.index_select(0, torch.tensor(local, dtype=torch.long))
            )
        if not pieces:
            raise Lfm2ConditionedRecapturePhaseError(
                f"Conditioned-recapture selection for {contract.field_name!r} is empty."
            )
        result = torch.cat(pieces, dim=0).contiguous()
        if tuple(result.shape) != (len(selected_rows), contract.feature_width):
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "Lazy field selection changed its authenticated geometry."
            )
        return result

    def field_rows(
        self,
        field_name: str,
        *,
        split: str,
        heldout_capability: (
            Lfm2ConditionedRecaptureHeldoutAccessCapability | None
        ) = None,
    ) -> Lfm2ConditionedRecaptureFieldRows:
        if split not in {"fit", "heldout"}:
            raise Lfm2ConditionedRecapturePhaseError(
                "Conditioned-recapture split must be fit or heldout."
            )
        if split == "heldout":
            self._authenticate_heldout_capability(heldout_capability)
        elif heldout_capability is not None:
            raise Lfm2ConditionedRecapturePhaseError(
                "FIT access does not accept a HELDOUT capability."
            )
        try:
            contract = next(
                value
                for value in self.request.field_contracts
                if value.field_name == field_name
            )
        except StopIteration as exc:
            raise Lfm2ConditionedRecapturePhaseError(
                f"Field {field_name!r} was not authenticated."
            ) from exc
        selected_rows = tuple(
            row for row in self.row_index["rows"] if row["basis_split"] == split
        )
        record_ids = tuple(str(row["record_id"]) for row in selected_rows)
        return Lfm2ConditionedRecaptureFieldRows(
            field_name=field_name,
            split=split,
            rows=self._load_field(contract, selected_rows),
            record_ids=record_ids,
            global_row_indices=tuple(
                int(row["global_row_index"]) for row in selected_rows
            ),
            global_batch_indices=tuple(
                int(row["global_batch_index"]) for row in selected_rows
            ),
            weights=equal_record_mass_weights(record_ids),
            source_dtype=contract.source_dtype,
            feature_width=contract.feature_width,
            row_index_sha256=self.row_index_sha256,
            split_identity_sha256=self.request.split_identity_sha256,
        )


def _authenticate_complete_entry(
    entry: VerifiedFoldPhaseCacheEntry,
    request: Lfm2ConditionedRecapturePhaseRequest,
    authority: Lfm2ConditionedRecapturePhaseAuthority,
) -> AuthenticatedLfm2ConditionedRecapturePhase:
    if entry.phase_key.as_dict() != request.phase_key.as_dict():
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Completed phase key differs from the request."
        )
    payload = entry.payload_path
    if _read_json(payload / "request.json") != request.as_dict():
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Completed conditioned-recapture request changed."
        )
    checkpoints = _load_checkpoint_prefix(
        payload,
        request=request,
        authority=authority,
        shard_root=payload / "shards",
    )
    row_index = _read_json(payload / "row-index.json")
    receipt = _read_json(payload / "receipt.json")
    anchor = _read_json(payload / "source-anchor.json")
    result = _canonical(dict(entry.result), "conditioned-recapture result")
    _verify_completed_documents(
        request,
        authority,
        checkpoints,
        row_index=row_index,
        receipt=receipt,
        anchor=anchor,
        result=result,
    )
    return AuthenticatedLfm2ConditionedRecapturePhase(
        entry=entry,
        request=request,
        receipt=_deep_freeze(receipt),
        row_index=_deep_freeze(row_index),
        source_anchor=_deep_freeze(anchor),
        authority=authority,
    )


def _validate_call(
    cache: Any,
    request: Any,
    authority: Any,
) -> tuple[
    DynamicFoldPhaseCache,
    Lfm2ConditionedRecapturePhaseRequest,
    Lfm2ConditionedRecapturePhaseAuthority,
]:
    if not isinstance(cache, DynamicFoldPhaseCache):
        raise TypeError("cache must be DynamicFoldPhaseCache.")
    if not isinstance(request, Lfm2ConditionedRecapturePhaseRequest):
        raise TypeError("request must be Lfm2ConditionedRecapturePhaseRequest.")
    if not request.verify_authentication():
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Conditioned recapture request authentication failed."
        )
    if (
        not isinstance(authority, Lfm2ConditionedRecapturePhaseAuthority)
        or authority.key_id != request.checkpoint_hmac_key_id
    ):
        raise Lfm2ConditionedRecaptureAuthenticationError(
            "Conditioned recapture authority differs from the request."
        )
    return cache, request, authority


def load_existing_lfm2_conditioned_recapture_phase(
    cache: DynamicFoldPhaseCache,
    request: Lfm2ConditionedRecapturePhaseRequest,
    *,
    authority: Lfm2ConditionedRecapturePhaseAuthority,
    events: FoldPhaseEventLog | None = None,
) -> AuthenticatedLfm2ConditionedRecapturePhase:
    """Load one complete HMAC-authenticated conditioned-child recapture."""

    cache, request, authority = _validate_call(cache, request, authority)
    entry = cache.lookup(request.phase_key, events=events)
    if entry is None:
        raise Lfm2ConditionedRecapturePhaseError(
            "Requested conditioned-child recapture is not complete."
        )
    return _authenticate_complete_entry(entry, request, authority)


def load_or_capture_lfm2_conditioned_recapture_phase(
    cache: DynamicFoldPhaseCache,
    request: Lfm2ConditionedRecapturePhaseRequest,
    *,
    authority: Lfm2ConditionedRecapturePhaseAuthority,
    checkpoint_root: str | os.PathLike[str],
    range_provider: Callable[[int, int], Lfm2ChildReplay] | None = None,
    maximum_new_checkpoints: int | None = None,
    events: FoldPhaseEventLog | None = None,
) -> (
    AuthenticatedLfm2ConditionedRecapturePhase | Lfm2ConditionedRecaptureProgress
):
    """Load a hit or capture the next authenticated child-only range prefix."""

    cache, request, authority = _validate_call(cache, request, authority)
    if maximum_new_checkpoints is not None:
        _positive_int(maximum_new_checkpoints, "maximum_new_checkpoints")

    # Lookup intentionally precedes provider invocation and work-root access.
    hit = cache.lookup(request.phase_key, events=events)
    if hit is not None:
        return _authenticate_complete_entry(hit, request, authority)

    root = Path(checkpoint_root).expanduser().absolute() / (
        request.phase_key.phase_key_sha256
    )
    root.mkdir(parents=True, exist_ok=True)
    request_path = root / "request.json"
    if request_path.exists():
        if _read_json(request_path) != request.as_dict():
            raise Lfm2ConditionedRecaptureAuthenticationError(
                "Checkpoint root belongs to another conditioned-recapture request."
            )
    else:
        _write_json(request_path, request.as_dict())
    checkpoints = _load_checkpoint_prefix(
        root, request=request, authority=authority
    )
    total = _checkpoint_count(request)
    if len(checkpoints) < total:
        if range_provider is None:
            raise Lfm2ConditionedRecapturePhaseError(
                "Incomplete conditioned recapture requires a range provider."
            )
        checkpoints = _capture_missing_prefix(
            request,
            authority,
            root,
            checkpoints,
            range_provider,
            maximum_new_checkpoints=maximum_new_checkpoints,
        )
    if len(checkpoints) < total:
        return Lfm2ConditionedRecaptureProgress(
            phase_key_sha256=request.phase_key.phase_key_sha256,
            total_checkpoints=total,
            completed_checkpoints=len(checkpoints),
            checkpoint_chain_sha256=_checkpoint_chain_sha256(checkpoints),
        )
    entry = _publish_complete_phase(
        cache,
        request,
        authority,
        root,
        checkpoints,
        events=events,
    )
    return _authenticate_complete_entry(entry, request, authority)


__all__ = [
    "AuthenticatedLfm2ConditionedRecapturePhase",
    "LFM2_CONDITIONED_RECAPTURE_CHECKPOINT_SCHEMA",
    "LFM2_CONDITIONED_RECAPTURE_PHASE",
    "LFM2_CONDITIONED_RECAPTURE_PROFILE_ROLE",
    "LFM2_CONDITIONED_RECAPTURE_REQUIRED_FIELDS",
    "LFM2_CONDITIONED_RECAPTURE_REQUEST_SCHEMA",
    "Lfm2ConditionedRecaptureAuthenticationError",
    "Lfm2ConditionedRecaptureFieldContract",
    "Lfm2ConditionedRecaptureFieldRows",
    "Lfm2ConditionedRecaptureHeldoutAccessCapability",
    "Lfm2ConditionedRecapturePhaseAuthority",
    "Lfm2ConditionedRecapturePhaseError",
    "Lfm2ConditionedRecapturePhaseRequest",
    "Lfm2ConditionedRecaptureProgress",
    "derive_lfm2_conditioned_recapture_split_identity",
    "derive_lfm2_frozen_conditioned_child_state_receipt_sha256",
    "lfm2_conditioned_recapture_implementation_sha256",
    "lfm2_conditioned_recapture_spec_sha256",
    "load_existing_lfm2_conditioned_recapture_phase",
    "load_or_capture_lfm2_conditioned_recapture_phase",
]
