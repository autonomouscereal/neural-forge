"""Authenticated sealed-cache adapter for streaming vocabulary margins.

This is the opt-in integration seam between the immutable LFM2 circuit-trace
cache, :mod:`hf_lfm2_streaming_vocabulary_margin`, and the family-neutral
balanced basis builder.  It authenticates the complete sealed cache and its
complete tensor inventory through one selective session, but it makes the
narrower semantic claim that only final-layer residuals and logits are read by
the vocabulary-margin objective.  Every row of every declared fit phase is
fed to the v2 streaming arithmetic; no cap, sampling, or row selection is
performed.

Held-out evidence remains excluded from basis selection.  Projection and
candidate diagnostics reopen one authenticated phase at a time and release it
before the next phase.  Consequently the retained numeric evidence is one
``H x H`` Gram (plus the v2 ``H`` mean while statistics are being built) and
bounded phase-local state rather than all margin directions or residual rows.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from .dynamic_balanced_circuit_basis import (
    EvidenceSplit,
    PrecomputedDownstreamObservability,
)
from .dynamic_circuit_ir import CircuitKind, canonical_sha256
from .hf_lfm2_circuit_trace_cache import (
    SELECTIVE_AUTHENTICATED_SESSION_SCHEMA,
    Lfm2CircuitTraceSelectiveAuthenticatedSession,
)
from .hf_lfm2_streaming_vocabulary_margin import (
    Lfm2StreamingVocabularyMarginConfig,
    Lfm2StreamingVocabularyLogitProjection,
    Lfm2StreamingVocabularyMarginManifest,
    Lfm2StreamingVocabularyMarginPhaseBatch,
    Lfm2StreamingVocabularyMarginPhaseDescriptor,
    build_streaming_vocabulary_margin_statistics_v2,
    deterministic_top_token_table_v2,
)
from .hf_lfm2_tokenizer_bridge_trace_adapter import (
    Lfm2StrictTokenizerBridgeTraceAdapter,
    Lfm2StrictTokenizerBridgeTraceError,
)
from . import hf_lfm2_streaming_vocabulary_margin as _streaming


LFM2_STREAMING_VOCABULARY_MARGIN_ADAPTER_SCHEMA = (
    "neural-forge-lfm2-streaming-vocabulary-margin-cache-adapter-v2"
)
LFM2_STREAMING_VOCABULARY_MARGIN_GATE_SCHEMA = (
    "neural-forge-lfm2-streaming-vocabulary-margin-gate-v2"
)
LFM2_STREAMING_VOCABULARY_MARGIN_INVENTORY_SCHEMA = (
    "neural-forge-lfm2-streaming-vocabulary-margin-cache-inventory-v2"
)
LFM2_TARGET_TOKENIZER_VOCABULARY_MARGIN_BINDING_SCHEMA = (
    "neural-forge-lfm2-target-tokenizer-vocabulary-margin-binding-v1"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DECODE_PHASE = re.compile(r"decode\.(?P<index>[0-9]{3})\Z")
_RESIDUAL_NAME = re.compile(
    r"^(?P<phase>.+)\.layer\.(?P<layer>[0-9]{3})\."
    r"residual\.(?P<endpoint>pre|post_operator|post_layer)\Z"
)


class Lfm2StreamingVocabularyMarginAdapterError(ValueError):
    """The cache/session/split cannot authorize streaming margin evidence."""


@dataclass(frozen=True, slots=True)
class Lfm2TargetTokenizerVocabularyMarginBinding:
    """Fully authenticated target-domain view of one strict token bridge."""

    trace_adapter: Lfm2StrictTokenizerBridgeTraceAdapter
    runtime_receipt: Mapping[str, Any]
    binding_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(
            self.trace_adapter,
            Lfm2StrictTokenizerBridgeTraceAdapter,
        ):
            raise Lfm2StreamingVocabularyMarginAdapterError(
                "Target vocabulary-margin binding requires a strict trace "
                "adapter."
            )
        try:
            self.trace_adapter.verify()
        except Lfm2StrictTokenizerBridgeTraceError as exc:
            raise Lfm2StreamingVocabularyMarginAdapterError(
                "Target vocabulary-margin trace adapter failed verification."
            ) from exc
        if not isinstance(self.runtime_receipt, Mapping):
            raise Lfm2StreamingVocabularyMarginAdapterError(
                "Target vocabulary-margin runtime receipt is missing."
            )
        receipt = _authenticate_seal(
            self.runtime_receipt,
            "runtime_receipt_sha256",
            "target tokenizer bridge runtime",
        )
        adapter = self.trace_adapter
        mapping_sha256 = canonical_sha256(list(adapter.mapping))
        required_digests = (
            "artifact_sha256",
            "transport_profile_sha256",
            "parent_tokenizer_semantics_sha256",
            "target_tokenizer_semantics_sha256",
            "target_tokenizer_artifact_fingerprint",
            "target_calibration_sha256",
            "trace_adapter_evidence_sha256",
            "canonical_target_tokenization_evidence_sha256",
            "paired_input_evidence_sha256",
            "target_to_donor_row_indices_sha256",
        )
        for name in required_digests:
            _digest(receipt.get(name), name)
        if (
            receipt.get("schema")
            != "neural-forge-production-lfm2-tokenizer-bridge-runtime-v1"
            or receipt.get("enabled") is not True
            or receipt.get("artifact_sha256")
            != adapter.bridge_artifact_sha256
            or receipt.get("transport_profile_sha256")
            != adapter.transport_profile_fingerprint
            or receipt.get("parent_tokenizer_semantics_sha256")
            != adapter.donor_tokenizer_semantics_sha256
            or receipt.get("target_tokenizer_semantics_sha256")
            != adapter.target_tokenizer_semantics_sha256
            or receipt.get("trace_adapter_evidence_sha256")
            != adapter.evidence_sha256
            or receipt.get("target_row_capacity")
            != adapter.target_row_capacity
            or receipt.get("donor_row_capacity")
            != adapter.donor_row_capacity
            or receipt.get("target_to_donor_row_indices_sha256")
            != mapping_sha256
            or receipt.get("paired_record_count") != adapter.record_count
            or receipt.get("all_physical_target_rows_mapped") is not True
            or receipt.get("mapping_injective_required") is not False
            or receipt.get(
                "repeated_donor_rows_preserve_distinct_target_rows"
            )
            is not True
            or receipt.get("numeric_token_id_equality_required") is not False
            or receipt.get("donor_canonical_sequence_equivalence_claimed")
            is not False
            or receipt.get("parent_cache_input_mode")
            != "target-tokenizer-canonical-then-one-row-mapped-to-donor"
            or receipt.get("parent_model_forwards") != 0
            or receipt.get("training_steps") != 0
        ):
            raise Lfm2StreamingVocabularyMarginAdapterError(
                "Target tokenizer bridge runtime receipt differs from its "
                "artifact, row map, calibration, or trace adapter."
            )
        body = {
            "schema": LFM2_TARGET_TOKENIZER_VOCABULARY_MARGIN_BINDING_SCHEMA,
            "runtime_receipt_sha256": receipt["runtime_receipt_sha256"],
            "trace_adapter_evidence_sha256": adapter.evidence_sha256,
            "bridge_artifact_sha256": adapter.bridge_artifact_sha256,
            "transport_profile_sha256": (
                adapter.transport_profile_fingerprint
            ),
            "target_to_donor_row_indices_sha256": mapping_sha256,
            "donor_row_capacity": adapter.donor_row_capacity,
            "target_row_capacity": adapter.target_row_capacity,
            "target_calibration_sha256": receipt[
                "target_calibration_sha256"
            ],
            "canonical_target_tokenization_evidence_sha256": receipt[
                "canonical_target_tokenization_evidence_sha256"
            ],
            "paired_input_evidence_sha256": receipt[
                "paired_input_evidence_sha256"
            ],
            "all_target_rows_mapped": len(adapter.mapping)
            == adapter.target_row_capacity,
            "mapping_injective_required": False,
            "repeated_donor_rows_preserve_distinct_target_rows": True,
            "parent_model_forwards": 0,
            "training_steps": 0,
        }
        claimed = self.binding_sha256
        observed = canonical_sha256(body)
        if claimed and claimed != observed:
            raise Lfm2StreamingVocabularyMarginAdapterError(
                "Target vocabulary-margin binding seal is invalid."
            )
        object.__setattr__(self, "runtime_receipt", _freeze(receipt))
        object.__setattr__(self, "binding_sha256", observed)

    def as_dict(self) -> dict[str, Any]:
        self.verify()
        receipt = _thaw(self.runtime_receipt)
        return {
            "schema": LFM2_TARGET_TOKENIZER_VOCABULARY_MARGIN_BINDING_SCHEMA,
            "runtime_receipt_sha256": receipt["runtime_receipt_sha256"],
            "trace_adapter_evidence_sha256": (
                self.trace_adapter.evidence_sha256
            ),
            "bridge_artifact_sha256": (
                self.trace_adapter.bridge_artifact_sha256
            ),
            "transport_profile_sha256": (
                self.trace_adapter.transport_profile_fingerprint
            ),
            "target_to_donor_row_indices_sha256": canonical_sha256(
                list(self.trace_adapter.mapping)
            ),
            "donor_row_capacity": self.trace_adapter.donor_row_capacity,
            "target_row_capacity": self.trace_adapter.target_row_capacity,
            "target_calibration_sha256": receipt[
                "target_calibration_sha256"
            ],
            "canonical_target_tokenization_evidence_sha256": receipt[
                "canonical_target_tokenization_evidence_sha256"
            ],
            "paired_input_evidence_sha256": receipt[
                "paired_input_evidence_sha256"
            ],
            "all_target_rows_mapped": True,
            "mapping_injective_required": False,
            "repeated_donor_rows_preserve_distinct_target_rows": True,
            "parent_model_forwards": 0,
            "training_steps": 0,
            "binding_sha256": self.binding_sha256,
        }

    def verify(self) -> Lfm2TargetTokenizerVocabularyMarginBinding:
        observed = type(self)(
            trace_adapter=self.trace_adapter,
            runtime_receipt=self.runtime_receipt,
            binding_sha256=self.binding_sha256,
        )
        if observed.binding_sha256 != self.binding_sha256:
            raise Lfm2StreamingVocabularyMarginAdapterError(
                "Target vocabulary-margin binding changed."
            )
        return self


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2StreamingVocabularyMarginAdapterError(
            f"{name} must be a positive integer."
        )
    return value


def _fraction(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2StreamingVocabularyMarginAdapterError(
            f"{name} must be a finite fraction within (0, 1]."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2StreamingVocabularyMarginAdapterError(
            f"{name} must be a finite fraction within (0, 1]."
        ) from exc
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise Lfm2StreamingVocabularyMarginAdapterError(
            f"{name} must be a finite fraction within (0, 1]."
        )
    return result


def _threshold(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2StreamingVocabularyMarginAdapterError(
            f"{name} must be a finite fraction within [0, 1]."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2StreamingVocabularyMarginAdapterError(
            f"{name} must be a finite fraction within [0, 1]."
        ) from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise Lfm2StreamingVocabularyMarginAdapterError(
            f"{name} must be a finite fraction within [0, 1]."
        )
    return result


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2StreamingVocabularyMarginAdapterError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _phase_key(value: str) -> tuple[int, int, str]:
    if value == "prefill":
        return (0, 0, value)
    match = _DECODE_PHASE.fullmatch(value)
    if match is not None:
        return (1, int(match.group("index")), value)
    return (2, 0, value)


def _canonical_copy(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as exc:
        raise Lfm2StreamingVocabularyMarginAdapterError(
            "Adapter evidence must be canonical JSON."
        ) from exc


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _sealed(body: Mapping[str, Any], seal_name: str) -> dict[str, Any]:
    if seal_name in body:
        raise Lfm2StreamingVocabularyMarginAdapterError(
            f"Evidence body already contains {seal_name!r}."
        )
    result = _canonical_copy(dict(body))
    result[seal_name] = canonical_sha256(result)
    return result


def _authenticate_seal(
    value: Mapping[str, Any],
    seal_name: str,
    label: str,
) -> dict[str, Any]:
    result = _canonical_copy(_thaw(value))
    claimed = result.pop(seal_name, None)
    if claimed != canonical_sha256(result):
        raise Lfm2StreamingVocabularyMarginAdapterError(
            f"{label} seal is invalid."
        )
    result[seal_name] = claimed
    return result


def _validate_tensor_descriptor(
    value: Any,
    *,
    name: str,
    last_width: int,
) -> tuple[int, str]:
    if not isinstance(value, Mapping):
        raise Lfm2StreamingVocabularyMarginAdapterError(
            f"Selected tensor {name!r} has no sealed descriptor."
        )
    shape = value.get("shape")
    dtype = value.get("dtype")
    numel = value.get("numel")
    size_bytes = value.get("size_bytes")
    digest = value.get("sha256")
    if (
        not isinstance(shape, list)
        or len(shape) < 2
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in shape
        )
        or shape[-1] != last_width
        or not isinstance(dtype, str)
        or not dtype
        or isinstance(numel, bool)
        or not isinstance(numel, int)
        or numel != math.prod(shape)
        or isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes <= 0
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
    ):
        raise Lfm2StreamingVocabularyMarginAdapterError(
            f"Selected tensor descriptor {name!r} is invalid."
        )
    return math.prod(shape[:-1]), digest


def _authenticate_selective_session(
    trace_cache: Any,
    selective_session: Any,
    *,
    expected_session_sha256: str | None = None,
) -> dict[str, Any]:
    verify = getattr(selective_session, "verify", None)
    as_dict = getattr(selective_session, "as_dict", None)
    if not callable(verify) or not callable(as_dict):
        raise Lfm2StreamingVocabularyMarginAdapterError(
            "selective_session must expose verify() and as_dict()."
        )
    verified = verify()
    if verified is not selective_session:
        raise Lfm2StreamingVocabularyMarginAdapterError(
            "selective_session.verify() returned another object."
        )
    manifest = getattr(trace_cache, "manifest", None)
    if not isinstance(manifest, Mapping):
        raise Lfm2StreamingVocabularyMarginAdapterError(
            "trace_cache must expose its sealed manifest."
        )
    manifest = _canonical_copy(dict(manifest))
    manifest_sha256 = _digest(
        manifest.get("manifest_sha256"),
        "trace cache manifest_sha256",
    )
    cache_key = _digest(
        getattr(trace_cache, "cache_key", None),
        "trace cache key",
    )
    batch_count = _positive_int(
        getattr(trace_cache, "batch_count", None),
        "trace cache batch_count",
    )
    if (
        getattr(selective_session, "cache_key", None) != cache_key
        or getattr(selective_session, "batch_count", None) != batch_count
        or getattr(selective_session, "manifest", None) != manifest
        or getattr(selective_session, "path", None)
        != getattr(trace_cache, "path", None)
    ):
        raise Lfm2StreamingVocabularyMarginAdapterError(
            "Selective session is not exactly bound to trace_cache."
        )
    receipt = _authenticate_seal(
        as_dict(),
        "session_sha256",
        "selective authenticated session",
    )
    session_sha256 = _digest(
        receipt.get("session_sha256"),
        "selective session SHA-256",
    )
    if (
        receipt.get("schema") != SELECTIVE_AUTHENTICATED_SESSION_SCHEMA
        or receipt.get("cache_key") != cache_key
        or receipt.get("manifest_sha256") != manifest_sha256
        or receipt.get("batch_count") != batch_count
        or (
            expected_session_sha256 is not None
            and session_sha256 != expected_session_sha256
        )
    ):
        raise Lfm2StreamingVocabularyMarginAdapterError(
            "Selective session receipt is not bound to this exact cache."
        )
    _digest(receipt.get("completion_sha256"), "cache completion SHA-256")
    batches = receipt.get("ordered_authenticated_batches")
    if (
        not isinstance(batches, list)
        or len(batches) != batch_count
        or [value.get("batch_index") for value in batches]
        != list(range(batch_count))
    ):
        raise Lfm2StreamingVocabularyMarginAdapterError(
            "Selective session receipt omits authenticated cache batches."
        )
    contract = receipt.get("authentication_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("manifest_completion_and_layout_verified") is not True
        or contract.get("trace_file_sha256_passes_per_session") != 1
        or contract.get("record_file_sha256_passes_per_session") != 1
        or contract.get("record_self_hash_verified") is not True
        or contract.get("safetensors_metadata_verified") is not True
        or contract.get("complete_tensor_key_inventory_verified") is not True
        or contract.get("selected_tensor_descriptors_verified_per_load")
        is not True
        or contract.get("unselected_tensor_payloads_loaded") is not False
        or contract.get("parent_forward_calls") != 0
        or contract.get("training_steps") != 0
    ):
        raise Lfm2StreamingVocabularyMarginAdapterError(
            "Selective session receipt does not certify complete sealed-cache "
            "integrity and selective payload loading."
        )
    return receipt


@dataclass(frozen=True, slots=True)
class _PhaseBinding:
    descriptor: Lfm2StreamingVocabularyMarginPhaseDescriptor
    residual_name: str
    logits_name: str


@dataclass(frozen=True, slots=True)
class _PhaseInventory:
    fit_manifest: Lfm2StreamingVocabularyMarginManifest
    heldout_manifest: Lfm2StreamingVocabularyMarginManifest
    bindings: Mapping[tuple[int, str], _PhaseBinding]
    report: Mapping[str, Any]


def _build_phase_inventory(
    selective_session: Any,
    *,
    fit_record_ids: frozenset[str],
    heldout_record_ids: frozenset[str],
    layer_count: int,
    source_width: int,
    vocabulary_size: int,
    cache_receipt: Mapping[str, Any],
) -> _PhaseInventory:
    references = getattr(selective_session, "batch_references", None)
    if not isinstance(references, tuple) or not references:
        raise Lfm2StreamingVocabularyMarginAdapterError(
            "Selective session has no authenticated batch references."
        )
    if len(references) != cache_receipt["batch_count"]:
        raise Lfm2StreamingVocabularyMarginAdapterError(
            "Selective reference count differs from sealed cache completion."
        )
    all_record_ids = fit_record_ids | heldout_record_ids
    if (
        not fit_record_ids
        or not heldout_record_ids
        or fit_record_ids & heldout_record_ids
    ):
        raise Lfm2StreamingVocabularyMarginAdapterError(
            "Fit and held-out record IDs must be nonempty and disjoint."
        )
    bindings: dict[tuple[int, str], _PhaseBinding] = {}
    fit_descriptors: list[Lfm2StreamingVocabularyMarginPhaseDescriptor] = []
    heldout_descriptors: list[
        Lfm2StreamingVocabularyMarginPhaseDescriptor
    ] = []
    inventory_rows: list[dict[str, Any]] = []
    observed_record_ids: set[str] = set()
    for expected_index, reference in enumerate(references):
        index = getattr(reference, "index", None)
        record_id = _digest(
            getattr(reference, "input_sha256", None),
            "selective reference input_sha256",
        )
        ordered_names = getattr(reference, "ordered_tensor_names", None)
        record = getattr(reference, "record", None)
        if (
            index != expected_index
            or not isinstance(ordered_names, tuple)
            or ordered_names != tuple(sorted(set(ordered_names)))
            or not isinstance(record, Mapping)
        ):
            raise Lfm2StreamingVocabularyMarginAdapterError(
                "Selective batch references are not complete and canonical."
            )
        receipt_batch = cache_receipt["ordered_authenticated_batches"][index]
        if (
            receipt_batch.get("input_sha256") != record_id
            or receipt_batch.get("record_sha256")
            != getattr(reference, "record_sha256", None)
            or receipt_batch.get("tensor_count")
            != getattr(reference, "tensor_count", None)
            or receipt_batch.get("logical_bytes")
            != getattr(reference, "logical_bytes", None)
        ):
            raise Lfm2StreamingVocabularyMarginAdapterError(
                f"Selective batch reference {index} differs from its "
                "authenticated session receipt."
            )
        inventory = record.get("tensors")
        if (
            not isinstance(inventory, Mapping)
            or tuple(sorted(inventory)) != ordered_names
        ):
            raise Lfm2StreamingVocabularyMarginAdapterError(
                f"Cache batch {index} tensor inventory is incomplete."
            )
        if record_id not in all_record_ids or record_id in observed_record_ids:
            raise Lfm2StreamingVocabularyMarginAdapterError(
                "Cache record split does not exactly partition batch references."
            )
        observed_record_ids.add(record_id)
        residual_phases: set[str] = set()
        for name in ordered_names:
            match = _RESIDUAL_NAME.fullmatch(name)
            if match is not None and not name.startswith("cached_prefill."):
                residual_phases.add(match.group("phase"))
        capture = record.get("capture")
        phase_diagnostics = (
            None if not isinstance(capture, Mapping)
            else capture.get("phase_diagnostics")
        )
        declared_phases = (
            set() if not isinstance(phase_diagnostics, Mapping)
            else set(phase_diagnostics)
        )
        logits_phases = {
            name[:-len(".logits")]
            for name in ordered_names
            if name.endswith(".logits")
            and not name.startswith("cached_prefill.")
        }
        if (
            not declared_phases
            or residual_phases != declared_phases
            or logits_phases != declared_phases
        ):
            raise Lfm2StreamingVocabularyMarginAdapterError(
                f"Cache batch {index} sealed capture, residual, and logits "
                "phase inventories disagree."
            )
        phases = tuple(sorted(declared_phases, key=_phase_key))
        decode_indices = [
            int(match.group("index"))
            for phase in phases
            if (match := _DECODE_PHASE.fullmatch(phase)) is not None
        ]
        if (
            phases.count("prefill") != 1
            or not decode_indices
            or decode_indices != list(range(len(decode_indices)))
            or any(
                phase != "prefill" and _DECODE_PHASE.fullmatch(phase) is None
                for phase in phases
            )
        ):
            raise Lfm2StreamingVocabularyMarginAdapterError(
                f"Cache batch {index} has an invalid complete phase horizon."
            )
        for phase in phases:
            residual_name = (
                f"{phase}.layer.{layer_count - 1:03d}.residual.post_layer"
            )
            logits_name = f"{phase}.logits"
            if residual_name not in inventory or logits_name not in inventory:
                raise Lfm2StreamingVocabularyMarginAdapterError(
                    f"Cache batch {index} phase {phase!r} omits final residual "
                    "or full logits."
                )
            residual_rows, residual_sha256 = _validate_tensor_descriptor(
                inventory[residual_name],
                name=residual_name,
                last_width=source_width,
            )
            logit_rows, logits_sha256 = _validate_tensor_descriptor(
                inventory[logits_name],
                name=logits_name,
                last_width=vocabulary_size,
            )
            if residual_rows != logit_rows:
                raise Lfm2StreamingVocabularyMarginAdapterError(
                    f"Cache batch {index} phase {phase!r} residual/logit "
                    "descriptors are row-misaligned."
                )
            descriptor = Lfm2StreamingVocabularyMarginPhaseDescriptor(
                record_index=index,
                record_id=record_id,
                phase=phase,
                row_count=residual_rows,
                residual_tensor_sha256=residual_sha256,
                logits_tensor_sha256=logits_sha256,
            )
            binding = _PhaseBinding(
                descriptor=descriptor,
                residual_name=residual_name,
                logits_name=logits_name,
            )
            bindings[(index, phase)] = binding
            target = (
                fit_descriptors
                if record_id in fit_record_ids
                else heldout_descriptors
            )
            target.append(descriptor)
            inventory_rows.append(
                {
                    **descriptor.as_dict(),
                    "split": (
                        EvidenceSplit.FIT.value
                        if record_id in fit_record_ids
                        else EvidenceSplit.HELDOUT.value
                    ),
                    "residual_name": residual_name,
                    "logits_name": logits_name,
                }
            )
    if observed_record_ids != all_record_ids:
        raise Lfm2StreamingVocabularyMarginAdapterError(
            "Record split omits or adds sealed cache record IDs."
        )
    fit_manifest = Lfm2StreamingVocabularyMarginManifest(
        split=EvidenceSplit.FIT,
        phases=tuple(fit_descriptors),
    )
    heldout_manifest = Lfm2StreamingVocabularyMarginManifest(
        split=EvidenceSplit.HELDOUT,
        phases=tuple(heldout_descriptors),
    )
    report = _sealed(
        {
            "schema": LFM2_STREAMING_VOCABULARY_MARGIN_INVENTORY_SCHEMA,
            "cache_key": cache_receipt["cache_key"],
            "manifest_sha256": cache_receipt["manifest_sha256"],
            "completion_sha256": cache_receipt["completion_sha256"],
            "selective_session_sha256": cache_receipt["session_sha256"],
            "sealed_batch_count": cache_receipt["batch_count"],
            "record_count": len(observed_record_ids),
            "phase_count": len(inventory_rows),
            "declared_row_count": sum(
                value["row_count"] for value in inventory_rows
            ),
            "fit_manifest_sha256": fit_manifest.manifest_sha256,
            "heldout_manifest_sha256": heldout_manifest.manifest_sha256,
            "fit_record_ids_sha256": canonical_sha256(sorted(fit_record_ids)),
            "heldout_record_ids_sha256": canonical_sha256(
                sorted(heldout_record_ids)
            ),
            "records_form_exact_cache_partition": True,
            "complete_record_phase_inventory_authenticated": True,
            "every_phase_has_aligned_final_residual_and_logits": True,
            "phases": inventory_rows,
            "coverage_claims": {
                "full_sealed_cache_integrity_scope": (
                    "manifest-completion-all-trace-file-bytes-all-record-file-"
                    "bytes-and-complete-tensor-key-inventory"
                ),
                "full_sealed_cache_integrity_authenticated": True,
                "vocabulary_margin_semantic_consumption_scope": (
                    "selected-final-layer-post-layer-residual-and-logits-tensors-only"
                ),
                "all_cache_tensor_semantics_consumed": False,
                "selected_subset_is_not_claimed_as_all_cache_tensors": True,
            },
        },
        "inventory_sha256",
    )
    return _PhaseInventory(
        fit_manifest=fit_manifest,
        heldout_manifest=heldout_manifest,
        bindings=MappingProxyType(bindings),
        report=_freeze(report),
    )


class _SelectedPhaseStream:
    """One-pass exact phase loader that releases the previous payload."""

    def __init__(
        self,
        selective_session: Any,
        manifest: Lfm2StreamingVocabularyMarginManifest,
        bindings: Mapping[tuple[int, str], _PhaseBinding],
    ) -> None:
        self._session = selective_session
        self._manifest = manifest
        self._bindings = bindings
        self._index = 0
        self._payload: Any | None = None
        self._closed = False
        self.selection_receipts: list[dict[str, Any]] = []

    def __iter__(self) -> _SelectedPhaseStream:
        return self

    def _release(self) -> None:
        if self._payload is not None:
            self._session.release_selected_tensors(self._payload)
            if getattr(self._payload, "released", None) is not True:
                raise Lfm2StreamingVocabularyMarginAdapterError(
                    "Selective phase payload did not release deterministically."
                )
            self._payload = None

    def __next__(self) -> Lfm2StreamingVocabularyMarginPhaseBatch:
        if self._closed:
            raise StopIteration
        self._release()
        if self._index >= len(self._manifest.phases):
            self._closed = True
            raise StopIteration
        descriptor = self._manifest.phases[self._index]
        binding = self._bindings[(descriptor.record_index, descriptor.phase)]
        names = tuple(sorted((binding.residual_name, binding.logits_name)))
        payload = self._session.load_selected_tensors(
            descriptor.record_index,
            names,
        )
        tensors = getattr(payload, "tensors", None)
        if (
            getattr(payload, "batch_index", None) != descriptor.record_index
            or getattr(payload, "input_sha256", None) != descriptor.record_id
            or getattr(payload, "ordered_tensor_names", None) != names
            or getattr(payload, "session_sha256", None)
            != getattr(self._session, "evidence_sha256", None)
            or not isinstance(tensors, Mapping)
            or tuple(sorted(tensors)) != names
        ):
            self._session.release_selected_tensors(payload)
            raise Lfm2StreamingVocabularyMarginAdapterError(
                "Selected phase payload is not bound to its manifest."
            )
        selection_sha256 = _digest(
            getattr(payload, "selection_sha256", None),
            "selected phase receipt",
        )
        self._payload = payload
        self._index += 1
        self.selection_receipts.append(
            {
                "record_index": descriptor.record_index,
                "record_id": descriptor.record_id,
                "phase": descriptor.phase,
                "row_count": descriptor.row_count,
                "ordered_tensor_names": list(names),
                "logical_bytes": getattr(payload, "logical_bytes", None),
                "selection_sha256": selection_sha256,
                "all_selected_tensor_rows_consumed": True,
            }
        )
        return Lfm2StreamingVocabularyMarginPhaseBatch(
            descriptor=descriptor,
            final_residual=tensors[binding.residual_name],
            logits=tensors[binding.logits_name],
        )

    def close(self) -> None:
        self._release()
        self._closed = True


def _streaming_statistics(
    selective_session: Any,
    manifest: Lfm2StreamingVocabularyMarginManifest,
    bindings: Mapping[tuple[int, str], _PhaseBinding],
    *,
    lexical_head: torch.Tensor,
    final_norm: torch.Tensor,
    config: Lfm2StreamingVocabularyMarginConfig,
    logit_projection: Lfm2StreamingVocabularyLogitProjection | None = None,
) -> tuple[Any, dict[str, Any]]:
    stream = _SelectedPhaseStream(selective_session, manifest, bindings)
    try:
        statistics = build_streaming_vocabulary_margin_statistics_v2(
            manifest,
            stream,
            lexical_head=lexical_head,
            final_norm=final_norm,
            config=config,
            logit_projection=logit_projection,
        )
    finally:
        stream.close()
    if len(stream.selection_receipts) != len(manifest.phases):
        raise Lfm2StreamingVocabularyMarginAdapterError(
            "Streaming arithmetic did not consume every declared phase."
        )
    selection = _sealed(
        {
            "schema": LFM2_STREAMING_VOCABULARY_MARGIN_ADAPTER_SCHEMA,
            "kind": "complete-manifest-phase-selection-pass",
            "split": manifest.split.value,
            "manifest_sha256": manifest.manifest_sha256,
            "phase_count": len(manifest.phases),
            "declared_row_count": sum(
                value.row_count for value in manifest.phases
            ),
            "selected_phase_count": len(stream.selection_receipts),
            "selected_rows_equal_declared_rows": True,
            "row_cap_or_subsample_applied": False,
            "selections": stream.selection_receipts,
        },
        "selection_pass_sha256",
    )
    return statistics, selection


def _target_domain_logits(
    source_logits: torch.Tensor,
    projection: Lfm2StreamingVocabularyLogitProjection | None,
) -> torch.Tensor:
    if projection is None:
        return source_logits
    if int(source_logits.shape[-1]) != projection.source_vocabulary_size:
        raise Lfm2StreamingVocabularyMarginAdapterError(
            "Cached source logits differ from target projection capacity."
        )
    indices = torch.tensor(
        projection.target_to_source_row_indices,
        device=source_logits.device,
        dtype=torch.long,
    )
    return source_logits.index_select(-1, indices).contiguous()


def _phase_projection_metrics(
    batch: Lfm2StreamingVocabularyMarginPhaseBatch,
    *,
    lexical_head: torch.Tensor,
    final_norm: torch.Tensor,
    basis: torch.Tensor,
    top_k: int,
    config: Lfm2StreamingVocabularyMarginConfig,
    logit_projection: Lfm2StreamingVocabularyLogitProjection | None = None,
) -> dict[str, Any]:
    source_width = int(lexical_head.shape[1])
    vocabulary_size = int(lexical_head.shape[0])
    residual, logits = _streaming._phase_tensors(
        batch,
        source_width=source_width,
        vocabulary_size=vocabulary_size,
        maximum_phase_rows=config.maximum_phase_rows,
        logit_projection=logit_projection,
    )
    table = deterministic_top_token_table_v2(logits, top_k=top_k)
    top_ids = table.token_ids[:, :1].expand(-1, top_k).reshape(-1)
    competitor_ids = table.token_ids[:, 1:].reshape(-1)
    head_device = lexical_head.device
    top_rows = lexical_head.detach().index_select(
        0,
        top_ids.to(device=head_device),
    ).to(device="cpu", dtype=torch.float64)
    competitor_rows = lexical_head.detach().index_select(
        0,
        competitor_ids.to(device=head_device),
    ).to(device="cpu", dtype=torch.float64)
    norm = final_norm.detach().to(device="cpu", dtype=torch.float64)
    effective = (
        (top_rows - competitor_rows) * norm.unsqueeze(0)
    ).contiguous()
    norms = torch.linalg.vector_norm(effective, dim=1)
    valid = norms > (
        torch.finfo(torch.float64).eps * max(1, source_width) * 16.0
    )
    if int(valid.sum().item()) <= 0:
        raise Lfm2StreamingVocabularyMarginAdapterError(
            f"All {batch.descriptor.phase!r} held-out directions are zero."
        )
    effective = effective[valid]
    repeated_residual = residual.repeat_interleave(top_k, dim=0)[valid]
    cached_margins = table.margins.reshape(-1)[valid]
    ranks = torch.arange(1, top_k + 1, dtype=torch.long).repeat(
        table.row_count
    )[valid]
    groups = torch.arange(table.row_count, dtype=torch.long).repeat_interleave(
        top_k
    )[valid]
    original = (repeated_residual * effective).sum(dim=1)
    projected = ((repeated_residual @ basis) * (effective @ basis)).sum(dim=1)
    scale = torch.maximum(
        torch.ones_like(original),
        torch.maximum(original.abs(), projected.abs()),
    )
    tolerance = torch.finfo(torch.float64).eps * source_width * 64.0 * scale

    def signs(values: torch.Tensor) -> torch.Tensor:
        return torch.where(
            values > tolerance,
            torch.ones_like(values, dtype=torch.int8),
            torch.where(
                values < -tolerance,
                -torch.ones_like(values, dtype=torch.int8),
                torch.zeros_like(values, dtype=torch.int8),
            ),
        )

    cached_signs = torch.where(
        cached_margins > tolerance,
        torch.ones_like(cached_margins, dtype=torch.int8),
        torch.zeros_like(cached_margins, dtype=torch.int8),
    )
    nonnegative = projected >= -tolerance
    unique_groups = torch.unique(groups, sorted=True)
    return {
        "pair_count": int(cached_margins.numel()),
        "nonzero_row_group_count": int(unique_groups.numel()),
        "cached_nonnegative_count": int((cached_margins >= 0.0).sum().item()),
        "original_sign_agreement_count": int(
            (signs(original) == cached_signs).sum().item()
        ),
        "projected_sign_agreement_count": int(
            (signs(projected) == cached_signs).sum().item()
        ),
        "rank_one_count": int((ranks == 1).sum().item()),
        "rank_one_nonnegative_count": int(
            nonnegative[ranks == 1].sum().item()
        ),
        "all_top_k_nonnegative_count": sum(
            bool(nonnegative[groups == group].all().item())
            for group in unique_groups
        ),
        "minimum": float(projected.min().item()),
        "phase_median": float(torch.median(projected).item()),
        "maximum": float(projected.max().item()),
        "decision_metrics_are_legacy_equivalent": (
            logit_projection is None
        ),
        "decision_metrics_are_materialized_target_domain_equivalent": (
            logit_projection is not None
        ),
    }


@dataclass(frozen=True, slots=True)
class Lfm2StreamingVocabularyMarginGateV2:
    """Held-out projection/candidate gate with lazy authenticated rescans."""

    vocabulary_size: int
    top_k: int
    manifest: Lfm2StreamingVocabularyMarginManifest
    bindings: Mapping[tuple[int, str], _PhaseBinding]
    selective_session: Any
    lexical_head: torch.Tensor
    final_norm: torch.Tensor
    config: Lfm2StreamingVocabularyMarginConfig
    logit_projection: Lfm2StreamingVocabularyLogitProjection | None
    target_tokenizer_binding: (
        Lfm2TargetTokenizerVocabularyMarginBinding | None
    )
    expected_session_sha256: str
    cache_key: str
    evidence_sha256: str
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            not isinstance(
                self.manifest, Lfm2StreamingVocabularyMarginManifest
            )
            or self.manifest.split is not EvidenceSplit.HELDOUT
            or not isinstance(self.config, Lfm2StreamingVocabularyMarginConfig)
            or self.config.top_k != self.top_k
            or not isinstance(self.bindings, Mapping)
            or not isinstance(self.report, Mapping)
            or (
                self.logit_projection is not None
                and not isinstance(
                    self.logit_projection,
                    Lfm2StreamingVocabularyLogitProjection,
                )
            )
            or (
                self.target_tokenizer_binding is not None
                and not isinstance(
                    self.target_tokenizer_binding,
                    Lfm2TargetTokenizerVocabularyMarginBinding,
                )
            )
            or ((self.logit_projection is None) != (
                self.target_tokenizer_binding is None
            ))
        ):
            raise Lfm2StreamingVocabularyMarginAdapterError(
                "Streaming vocabulary-margin gate invariants are invalid."
            )
        _digest(self.expected_session_sha256, "expected session SHA-256")
        _digest(self.cache_key, "cache key")
        _digest(self.evidence_sha256, "gate evidence SHA-256")
        report = _authenticate_seal(
            self.report,
            "gate_evidence_sha256",
            "streaming vocabulary-margin gate",
        )
        if report["gate_evidence_sha256"] != self.evidence_sha256:
            raise Lfm2StreamingVocabularyMarginAdapterError(
                "Streaming vocabulary-margin gate evidence changed."
            )
        object.__setattr__(
            self, "bindings", MappingProxyType(dict(self.bindings))
        )
        object.__setattr__(self, "report", _freeze(report))

    @property
    def heldout_record_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted({value.record_id for value in self.manifest.phases})
        )

    def _authenticate(self) -> None:
        if self.target_tokenizer_binding is not None:
            self.target_tokenizer_binding.verify()
        receipt = _authenticate_selective_session(
            _GateCacheView(
                cache_key=self.cache_key,
                batch_count=getattr(
                    self.selective_session, "batch_count", None
                ),
                manifest=getattr(self.selective_session, "manifest", None),
                path=getattr(self.selective_session, "path", None),
            ),
            self.selective_session,
            expected_session_sha256=self.expected_session_sha256,
        )
        if receipt["session_sha256"] != self.expected_session_sha256:
            raise Lfm2StreamingVocabularyMarginAdapterError(
                "Selective session changed after gate construction."
            )
        report = _authenticate_seal(
            self.report,
            "gate_evidence_sha256",
            "streaming vocabulary-margin gate",
        )
        if report["gate_evidence_sha256"] != self.evidence_sha256:
            raise Lfm2StreamingVocabularyMarginAdapterError(
                "Streaming vocabulary-margin gate report changed."
            )
        if (
            _streaming._tensor_sha256(self.lexical_head)
            != report["lexical_head_sha256"]
            or _streaming._tensor_sha256(self.final_norm)
            != report["final_norm_sha256"]
            or (
                self.logit_projection is not None
                and self.logit_projection.projection_sha256
                != report.get("target_logit_projection_sha256")
            )
        ):
            raise Lfm2StreamingVocabularyMarginAdapterError(
                "Vocabulary-margin head or final norm changed after sealing."
            )

    def as_dict(self) -> dict[str, Any]:
        self._authenticate()
        return _thaw(self.report)

    def evaluate_projection(
        self,
        basis: torch.Tensor,
        *,
        minimum_retained_energy_fraction: float,
        minimum_sign_agreement_fraction: float,
        minimum_top_competitor_nonnegative_fraction: float,
        minimum_all_top_k_nonnegative_fraction: float,
    ) -> dict[str, Any]:
        self._authenticate()
        source_width = int(self.lexical_head.shape[1])
        if (
            not isinstance(basis, torch.Tensor)
            or basis.layout != torch.strided
            or basis.device.type == "meta"
            or basis.ndim != 2
            or int(basis.shape[0]) != source_width
            or int(basis.shape[1]) <= 0
            or not torch.is_floating_point(basis)
            or torch.is_complex(basis)
            or not bool(torch.isfinite(basis).all().item())
        ):
            raise Lfm2StreamingVocabularyMarginAdapterError(
                "projection basis must be a finite source-by-target matrix."
            )
        basis64 = (
            basis.detach().to(device="cpu", dtype=torch.float64).contiguous()
        )
        statistics, statistics_selection = _streaming_statistics(
            self.selective_session,
            self.manifest,
            self.bindings,
            lexical_head=self.lexical_head,
            final_norm=self.final_norm,
            config=self.config,
            logit_projection=self.logit_projection,
        )
        stream = _SelectedPhaseStream(
            self.selective_session,
            self.manifest,
            self.bindings,
        )
        aggregates = {
            "pair_count": 0,
            "nonzero_row_group_count": 0,
            "cached_nonnegative_count": 0,
            "original_sign_agreement_count": 0,
            "projected_sign_agreement_count": 0,
            "rank_one_count": 0,
            "rank_one_nonnegative_count": 0,
            "all_top_k_nonnegative_count": 0,
        }
        minima: list[float] = []
        maxima: list[float] = []
        phase_medians: list[float] = []
        phase_reports: list[dict[str, Any]] = []
        try:
            for batch in stream:
                metrics = _phase_projection_metrics(
                    batch,
                    lexical_head=self.lexical_head,
                    final_norm=self.final_norm,
                    basis=basis64,
                    top_k=self.top_k,
                    config=self.config,
                    logit_projection=self.logit_projection,
                )
                for name in aggregates:
                    aggregates[name] += int(metrics[name])
                minima.append(metrics["minimum"])
                maxima.append(metrics["maximum"])
                phase_medians.append(metrics["phase_median"])
                phase_reports.append(
                    {
                        "record_index": batch.descriptor.record_index,
                        "record_id": batch.descriptor.record_id,
                        "phase": batch.descriptor.phase,
                        "row_count": batch.descriptor.row_count,
                        **metrics,
                    }
                )
        finally:
            stream.close()
        pair_count = aggregates["pair_count"]
        group_count = aggregates["nonzero_row_group_count"]
        rank_one_count = aggregates["rank_one_count"]
        if pair_count <= 0 or group_count <= 0 or rank_one_count <= 0:
            raise Lfm2StreamingVocabularyMarginAdapterError(
                "Held-out projection diagnostics have no decision rows."
            )
        metrics = {
            "retained_top_k_margin_direction_energy_fraction": (
                statistics.retained_energy_fraction(basis64)
            ),
            "pair_count": pair_count,
            "logit_row_count": group_count,
            "cached_top1_margin_nonnegative_fraction": (
                aggregates["cached_nonnegative_count"] / pair_count
            ),
            "closed_form_unprojected_vs_cached_sign_agreement_fraction": (
                aggregates["original_sign_agreement_count"] / pair_count
            ),
            "projected_vs_cached_sign_agreement_fraction": (
                aggregates["projected_sign_agreement_count"] / pair_count
            ),
            "projected_top1_vs_top_competitor_nonnegative_fraction": (
                aggregates["rank_one_nonnegative_count"] / rank_one_count
            ),
            "projected_top1_vs_all_top_k_nonnegative_fraction": (
                aggregates["all_top_k_nonnegative_count"] / group_count
            ),
            "projected_margin_minimum": min(minima),
            "projected_margin_phase_median_median": float(
                torch.median(
                    torch.tensor(phase_medians, dtype=torch.float64)
                ).item()
            ),
            "projected_margin_maximum": max(maxima),
            "global_margin_median_materialized": False,
            "decision_metrics_are_legacy_equivalent": (
                self.logit_projection is None
            ),
            "decision_metrics_are_materialized_target_domain_equivalent": (
                self.logit_projection is not None
            ),
        }
        thresholds = {
            "retained_top_k_margin_direction_energy_fraction": _threshold(
                minimum_retained_energy_fraction,
                "minimum_retained_energy_fraction",
            ),
            "projected_vs_cached_sign_agreement_fraction": _threshold(
                minimum_sign_agreement_fraction,
                "minimum_sign_agreement_fraction",
            ),
            "projected_top1_vs_top_competitor_nonnegative_fraction": _threshold(
                minimum_top_competitor_nonnegative_fraction,
                "minimum_top_competitor_nonnegative_fraction",
            ),
            "projected_top1_vs_all_top_k_nonnegative_fraction": _threshold(
                minimum_all_top_k_nonnegative_fraction,
                "minimum_all_top_k_nonnegative_fraction",
            ),
        }
        checks = {
            name: float(metrics[name]) + 1e-15 >= minimum
            for name, minimum in thresholds.items()
        }
        self._authenticate()
        result = {
            "schema": LFM2_STREAMING_VOCABULARY_MARGIN_GATE_SCHEMA,
            "gate_kind": "lazy-heldout-projection-top1-vs-top-k-v2",
            "metrics": metrics,
            "minimums": thresholds,
            "checks": checks,
            "gate_passed": all(checks.values()),
            "heldout_used_for_selection": False,
            "heldout_statistics": statistics.as_dict(),
            "statistics_selection_pass": statistics_selection,
            "projection_diagnostic_selection_pass": _sealed(
                {
                    "manifest_sha256": self.manifest.manifest_sha256,
                    "phase_count": len(stream.selection_receipts),
                    "all_declared_rows_rescanned": True,
                    "row_cap_or_subsample_applied": False,
                    "selections": stream.selection_receipts,
                },
                "selection_pass_sha256",
            ),
            "phases": phase_reports,
            "retained_parent_phase_tensors_after_evaluation": 0,
            "parent_forward_calls": 0,
            "candidate_forward_calls": 0,
            "trace_regeneration_calls": 0,
            "training_steps": 0,
        }
        result["gate_sha256"] = canonical_sha256(result)
        return result

    def evaluate_candidate_replay(
        self,
        candidate_replay: Any,
        *,
        minimum_sign_agreement_fraction: float,
        minimum_top_competitor_nonnegative_fraction: float,
        minimum_all_top_k_nonnegative_fraction: float,
        minimum_exact_top1_agreement_fraction: float,
    ) -> dict[str, Any]:
        self._authenticate()
        if self.target_tokenizer_binding is not None:
            raise Lfm2StreamingVocabularyMarginAdapterError(
                "Cross-tokenizer candidate replay gating remains disabled; "
                "target-domain streaming-v2 evidence is diagnostic and manual "
                "whole-response review remains the verdict."
            )
        iterator = getattr(candidate_replay, "iter_batches", None)
        if not callable(iterator):
            raise Lfm2StreamingVocabularyMarginAdapterError(
                "candidate_replay must expose iter_batches()."
            )
        candidates: dict[str, Any] = {}
        for batch in iterator():
            record_id = _digest(
                getattr(batch, "input_sha256", None),
                "candidate replay input_sha256",
            )
            if record_id in candidates:
                raise Lfm2StreamingVocabularyMarginAdapterError(
                    "Candidate replay repeats a held-out record ID."
                )
            candidates[record_id] = batch
        missing = sorted(set(self.heldout_record_ids).difference(candidates))
        if missing:
            raise Lfm2StreamingVocabularyMarginAdapterError(
                "Candidate replay omits held-out records: "
                + ", ".join(missing)
                + "."
            )
        descriptors = tuple(
            sorted(
                self.manifest.phases,
                key=lambda value: (value.record_id, _phase_key(value.phase)),
            )
        )
        total_rows = sum(value.row_count for value in descriptors)
        selected_digest = hashlib.sha256()
        selected_digest.update(str(torch.float64).encode("ascii"))
        selected_digest.update(b"\0")
        selected_digest.update(
            canonical_sha256([total_rows, self.top_k + 1]).encode("ascii")
        )
        selected_digest.update(b"\0")
        sign_agreement = 0
        top_nonnegative = 0
        all_nonnegative = 0
        exact_top1 = 0
        pair_count = 0
        row_count = 0
        minima: list[float] = []
        maxima: list[float] = []
        phase_medians: list[float] = []
        selections: list[dict[str, Any]] = []
        phase_reports: list[dict[str, Any]] = []
        for descriptor in descriptors:
            binding = self.bindings[
                (descriptor.record_index, descriptor.phase)
            ]
            payload = self.selective_session.load_selected_tensors(
                descriptor.record_index,
                (binding.logits_name,),
            )
            try:
                tensors = getattr(payload, "tensors", None)
                if (
                    getattr(payload, "batch_index", None)
                    != descriptor.record_index
                    or getattr(payload, "input_sha256", None)
                    != descriptor.record_id
                    or getattr(payload, "ordered_tensor_names", None)
                    != (binding.logits_name,)
                    or getattr(payload, "session_sha256", None)
                    != self.expected_session_sha256
                    or not isinstance(tensors, Mapping)
                ):
                    raise Lfm2StreamingVocabularyMarginAdapterError(
                        "Candidate parent-logit selection is not authenticated."
                    )
                parent_logits = tensors[binding.logits_name]
                if (
                    _streaming._authenticated_trace_tensor_sha256(
                        parent_logits
                    )
                    != descriptor.logits_tensor_sha256
                ):
                    raise Lfm2StreamingVocabularyMarginAdapterError(
                        "Candidate parent-logit tensor differs from its manifest."
                    )
                parent_logits = _target_domain_logits(
                    parent_logits,
                    self.logit_projection,
                )
                parent_table = deterministic_top_token_table_v2(
                    parent_logits,
                    top_k=self.top_k,
                )
                candidate_tensors = getattr(
                    candidates[descriptor.record_id],
                    "tensors",
                    None,
                )
                if (
                    not isinstance(candidate_tensors, Mapping)
                    or binding.logits_name not in candidate_tensors
                ):
                    raise Lfm2StreamingVocabularyMarginAdapterError(
                        f"Candidate replay omits {binding.logits_name!r}."
                    )
                candidate_logits = candidate_tensors[binding.logits_name]
                if (
                    not isinstance(candidate_logits, torch.Tensor)
                    or candidate_logits.layout != torch.strided
                    or candidate_logits.device.type == "meta"
                    or candidate_logits.ndim < 2
                    or int(candidate_logits.shape[-1]) != self.vocabulary_size
                    or not torch.is_floating_point(candidate_logits)
                    or torch.is_complex(candidate_logits)
                ):
                    raise Lfm2StreamingVocabularyMarginAdapterError(
                        "Candidate logits have an incompatible tensor shape."
                    )
                rows = (
                    candidate_logits.detach()
                    .to(device="cpu", dtype=torch.float64)
                    .reshape(-1, self.vocabulary_size)
                    .contiguous()
                )
                if int(rows.shape[0]) != descriptor.row_count or not bool(
                    torch.isfinite(rows).all().item()
                ):
                    raise Lfm2StreamingVocabularyMarginAdapterError(
                        "Candidate logits do not align with held-out parent rows."
                    )
                selected = rows.gather(1, parent_table.token_ids).contiguous()
                candidate_margins = selected[:, :1] - selected[:, 1:]
                cached_margins = parent_table.margins
                candidate_top1 = deterministic_top_token_table_v2(
                    candidate_logits,
                    top_k=1,
                ).token_ids[:, 0]
                exact = candidate_top1 == parent_table.token_ids[:, 0]
                scale = torch.maximum(
                    torch.ones_like(candidate_margins),
                    torch.maximum(
                        candidate_margins.abs(), cached_margins.abs()
                    ),
                )
                tolerance = (
                    torch.finfo(torch.float64).eps
                    * self.vocabulary_size
                    * 16.0
                    * scale
                )
                candidate_signs = torch.where(
                    candidate_margins > tolerance,
                    torch.ones_like(candidate_margins, dtype=torch.int8),
                    torch.where(
                        candidate_margins < -tolerance,
                        -torch.ones_like(candidate_margins, dtype=torch.int8),
                        torch.zeros_like(candidate_margins, dtype=torch.int8),
                    ),
                )
                cached_signs = torch.where(
                    cached_margins > tolerance,
                    torch.ones_like(cached_margins, dtype=torch.int8),
                    torch.zeros_like(cached_margins, dtype=torch.int8),
                )
                nonnegative = candidate_margins >= -tolerance
                phase_pairs = int(candidate_margins.numel())
                pair_count += phase_pairs
                row_count += descriptor.row_count
                sign_agreement += int(
                    (candidate_signs == cached_signs).sum().item()
                )
                top_nonnegative += int(nonnegative[:, 0].sum().item())
                all_nonnegative += int(nonnegative.all(dim=1).sum().item())
                exact_top1 += int(exact.sum().item())
                minima.append(float(candidate_margins.min().item()))
                maxima.append(float(candidate_margins.max().item()))
                phase_medians.append(
                    float(torch.median(candidate_margins).item())
                )
                selected_digest.update(
                    memoryview(
                        selected.reshape(-1).view(torch.uint8).numpy()
                    ).cast("B")
                )
                phase_reports.append(
                    {
                        "record_id": descriptor.record_id,
                        "phase": descriptor.phase,
                        "row_count": descriptor.row_count,
                        "candidate_top1_exact_count": int(exact.sum().item()),
                        "candidate_top1_boundary_count": int(
                            nonnegative[:, 0].sum().item()
                        ),
                    }
                )
                selections.append(
                    {
                        "record_index": descriptor.record_index,
                        "record_id": descriptor.record_id,
                        "phase": descriptor.phase,
                        "row_count": descriptor.row_count,
                        "selection_sha256": _digest(
                            getattr(payload, "selection_sha256", None),
                            "candidate parent-logit selection SHA-256",
                        ),
                        "all_parent_logit_rows_consumed": True,
                    }
                )
            finally:
                self.selective_session.release_selected_tensors(payload)
        if pair_count != row_count * self.top_k or row_count != total_rows:
            raise Lfm2StreamingVocabularyMarginAdapterError(
                "Candidate diagnostics did not consume the complete manifest."
            )
        metrics = {
            "candidate_vs_cached_sign_agreement_fraction": (
                sign_agreement / pair_count
            ),
            "candidate_top1_vs_top_competitor_nonnegative_fraction": (
                top_nonnegative / row_count
            ),
            "candidate_top1_vs_all_top_k_nonnegative_fraction": (
                all_nonnegative / row_count
            ),
            "candidate_exact_parent_top1_agreement_fraction": (
                exact_top1 / row_count
            ),
            "candidate_margin_minimum": min(minima),
            "candidate_margin_phase_median_median": float(
                torch.median(
                    torch.tensor(phase_medians, dtype=torch.float64)
                ).item()
            ),
            "candidate_margin_maximum": max(maxima),
            "global_margin_median_materialized": False,
            "decision_metrics_are_legacy_equivalent": True,
        }
        thresholds = {
            "candidate_vs_cached_sign_agreement_fraction": _threshold(
                minimum_sign_agreement_fraction,
                "minimum_sign_agreement_fraction",
            ),
            "candidate_top1_vs_top_competitor_nonnegative_fraction": _threshold(
                minimum_top_competitor_nonnegative_fraction,
                "minimum_top_competitor_nonnegative_fraction",
            ),
            "candidate_top1_vs_all_top_k_nonnegative_fraction": _threshold(
                minimum_all_top_k_nonnegative_fraction,
                "minimum_all_top_k_nonnegative_fraction",
            ),
            "candidate_exact_parent_top1_agreement_fraction": _threshold(
                minimum_exact_top1_agreement_fraction,
                "minimum_exact_top1_agreement_fraction",
            ),
        }
        checks = {
            name: float(metrics[name]) + 1e-15 >= minimum
            for name, minimum in thresholds.items()
        }
        self._authenticate()
        result = {
            "schema": LFM2_STREAMING_VOCABULARY_MARGIN_GATE_SCHEMA,
            "gate_kind": "lazy-heldout-candidate-replay-top1-vs-top-k-v2",
            "heldout_record_count": len(self.heldout_record_ids),
            "phase_count": len(phase_reports),
            "top_k": self.top_k,
            "metrics": metrics,
            "minimums": thresholds,
            "checks": checks,
            "gate_passed": all(checks.values()),
            "phases": phase_reports,
            "candidate_selected_logits_sha256": selected_digest.hexdigest(),
            "parent_logit_selection_pass": _sealed(
                {
                    "manifest_sha256": self.manifest.manifest_sha256,
                    "phase_count": len(selections),
                    "declared_row_count": total_rows,
                    "all_declared_parent_logit_rows_consumed": True,
                    "row_cap_or_subsample_applied": False,
                    "selections": selections,
                },
                "selection_pass_sha256",
            ),
            "heldout_used_for_selection": False,
            "retained_parent_phase_tensors_after_evaluation": 0,
            "parent_forward_calls": 0,
            "candidate_forward_calls": 0,
            "trace_capture_calls": 0,
            "training_steps": 0,
        }
        result["gate_sha256"] = canonical_sha256(result)
        return result


@dataclass(frozen=True, slots=True)
class _GateCacheView:
    cache_key: str
    batch_count: int
    manifest: Mapping[str, Any]
    path: Any


@dataclass(frozen=True, slots=True)
class Lfm2StreamingVocabularyMarginAdapterResult:
    """Fit precomputed Gram, lazy held-out gate, and sealed audit report."""

    precomputed_observability: PrecomputedDownstreamObservability
    gate: Lfm2StreamingVocabularyMarginGateV2
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            not isinstance(
                self.precomputed_observability,
                PrecomputedDownstreamObservability,
            )
            or not isinstance(self.gate, Lfm2StreamingVocabularyMarginGateV2)
            or not isinstance(self.report, Mapping)
        ):
            raise Lfm2StreamingVocabularyMarginAdapterError(
                "Streaming vocabulary-margin adapter result is invalid."
            )
        report = _authenticate_seal(
            self.report,
            "adapter_sha256",
            "streaming vocabulary-margin adapter",
        )
        object.__setattr__(self, "report", _freeze(report))

    def as_dict(self) -> dict[str, Any]:
        if not self.precomputed_observability.verify():
            raise Lfm2StreamingVocabularyMarginAdapterError(
                "Precomputed fit observability changed after construction."
            )
        return _thaw(self.report)


def build_lfm2_streaming_vocabulary_margin_adapter_v2(
    trace_cache: Any,
    selective_session: Lfm2CircuitTraceSelectiveAuthenticatedSession,
    *,
    fit_record_ids: Sequence[str],
    heldout_record_ids: Sequence[str],
    layer_count: int,
    source_width: int,
    lexical_head: torch.Tensor,
    final_norm: torch.Tensor,
    top_k: int,
    margin_floor_fraction: float,
    pair_chunk_size: int = 256,
    target_tokenizer_binding: (
        Lfm2TargetTokenizerVocabularyMarginBinding | None
    ) = None,
) -> Lfm2StreamingVocabularyMarginAdapterResult:
    """Bind every declared fit row to one precomputed observability Gram."""

    layer_count = _positive_int(layer_count, "layer_count")
    source_width = _positive_int(source_width, "source_width")
    top_k = _positive_int(top_k, "top_k")
    pair_chunk_size = _positive_int(pair_chunk_size, "pair_chunk_size")
    margin_floor_fraction = _fraction(
        margin_floor_fraction,
        "margin_floor_fraction",
    )
    if target_tokenizer_binding is not None:
        if not isinstance(
            target_tokenizer_binding,
            Lfm2TargetTokenizerVocabularyMarginBinding,
        ):
            raise Lfm2StreamingVocabularyMarginAdapterError(
                "target_tokenizer_binding has the wrong type."
            )
        target_tokenizer_binding.verify()
    if (
        not isinstance(lexical_head, torch.Tensor)
        or lexical_head.ndim != 2
        or int(lexical_head.shape[1]) != source_width
        or not torch.is_floating_point(lexical_head)
        or torch.is_complex(lexical_head)
        or not isinstance(final_norm, torch.Tensor)
        or tuple(final_norm.shape) != (source_width,)
        or not torch.is_floating_point(final_norm)
        or torch.is_complex(final_norm)
    ):
        raise Lfm2StreamingVocabularyMarginAdapterError(
            "lexical_head/final_norm do not match the source width."
        )
    # Same-tokenizer runs retain only authenticated parent-weight references.
    # A strict target bridge materializes exactly one full target-row head view,
    # using the same sealed row selection as child lexical-head compilation.
    source_head = lexical_head.detach()
    norm = final_norm.detach()
    source_vocabulary_size = int(source_head.shape[0])
    logit_projection = None
    if target_tokenizer_binding is None:
        head = source_head
    else:
        adapter = target_tokenizer_binding.trace_adapter
        if source_vocabulary_size != adapter.donor_row_capacity:
            raise Lfm2StreamingVocabularyMarginAdapterError(
                "Parent lexical-head rows differ from the authenticated donor "
                "row capacity."
            )
        references = getattr(selective_session, "batch_references", None)
        if (
            not isinstance(references, tuple)
            or len(references) != adapter.record_count
            or any(
                record.ordinal != index
                or getattr(references[index], "input_sha256", None)
                != record.donor_cache_input_sha256
                for index, record in enumerate(adapter.records)
            )
        ):
            raise Lfm2StreamingVocabularyMarginAdapterError(
                "Target tokenizer bridge records differ from the authenticated "
                "sealed-cache ordering."
            )
        indices = torch.tensor(
            adapter.mapping,
            device=source_head.device,
            dtype=torch.long,
        )
        head = source_head.index_select(0, indices).contiguous()
        logit_projection = Lfm2StreamingVocabularyLogitProjection(
            source_vocabulary_size=source_vocabulary_size,
            target_to_source_row_indices=adapter.mapping,
            provenance=target_tokenizer_binding.as_dict(),
        )
    vocabulary_size = int(head.shape[0])
    if top_k >= vocabulary_size:
        raise Lfm2StreamingVocabularyMarginAdapterError(
            "top_k must be smaller than the vocabulary size."
        )
    fit_ids = frozenset(
        _digest(value, "fit record ID") for value in fit_record_ids
    )
    heldout_ids = frozenset(
        _digest(value, "held-out record ID") for value in heldout_record_ids
    )
    cache_receipt = _authenticate_selective_session(
        trace_cache,
        selective_session,
    )
    inventory = _build_phase_inventory(
        selective_session,
        fit_record_ids=fit_ids,
        heldout_record_ids=heldout_ids,
        layer_count=layer_count,
        source_width=source_width,
        vocabulary_size=source_vocabulary_size,
        cache_receipt=cache_receipt,
    )
    maximum_phase_rows = max(
        value.row_count
        for value in (
            *inventory.fit_manifest.phases,
            *inventory.heldout_manifest.phases,
        )
    )
    config = Lfm2StreamingVocabularyMarginConfig(
        top_k=top_k,
        margin_floor_fraction=margin_floor_fraction,
        pair_chunk_size=pair_chunk_size,
        maximum_phase_rows=maximum_phase_rows,
    )
    fit_statistics, fit_selection = _streaming_statistics(
        selective_session,
        inventory.fit_manifest,
        inventory.bindings,
        lexical_head=head,
        final_norm=norm,
        config=config,
        logit_projection=logit_projection,
    )
    fit_report = fit_statistics.as_dict()
    if (
        fit_report["logit_row_count"]
        != sum(value.row_count for value in inventory.fit_manifest.phases)
        or fit_report["all_manifest_declared_rows_consumed"] is not True
        or fit_report["row_cap_or_subsample_applied"] is not False
    ):
        raise Lfm2StreamingVocabularyMarginAdapterError(
            "Fit streaming statistics do not cover every declared row."
        )
    _authenticate_selective_session(
        trace_cache,
        selective_session,
        expected_session_sha256=cache_receipt["session_sha256"],
    )
    provenance = _sealed(
        {
            "schema": LFM2_STREAMING_VOCABULARY_MARGIN_ADAPTER_SCHEMA,
            "kind": "fit-precomputed-observability-gram",
            "cache_key": cache_receipt["cache_key"],
            "manifest_sha256": cache_receipt["manifest_sha256"],
            "completion_sha256": cache_receipt["completion_sha256"],
            "selective_session_sha256": cache_receipt["session_sha256"],
            "inventory_sha256": inventory.report["inventory_sha256"],
            "fit_manifest_sha256": inventory.fit_manifest.manifest_sha256,
            "fit_statistics_result_sha256": fit_report["result_sha256"],
            "fit_selection_pass_sha256": fit_selection[
                "selection_pass_sha256"
            ],
            "all_fit_manifest_rows_consumed": True,
            "row_cap_or_subsample_applied": False,
            "parent_forward_calls": 0,
            "trace_regeneration_calls": 0,
            "training_steps": 0,
            "target_tokenizer_binding_sha256": (
                None
                if target_tokenizer_binding is None
                else target_tokenizer_binding.binding_sha256
            ),
            "target_logit_projection_sha256": (
                None
                if logit_projection is None
                else logit_projection.projection_sha256
            ),
        },
        "provenance_sha256",
    )
    precomputed = PrecomputedDownstreamObservability(
        evidence_id="interface.text.logits.vocabulary_margin.streaming_v2",
        node_id="output.lm_head",
        circuit_kind=CircuitKind.OUTPUT_HEAD,
        role="vocabulary_margin_geometry",
        gram=fit_statistics.second_moment_gram,
        provenance=provenance,
        interface_ids=("text.logits.output",),
    )
    gate_body = {
        "schema": LFM2_STREAMING_VOCABULARY_MARGIN_GATE_SCHEMA,
        "kind": "lazy-authenticated-heldout-rescan",
        "cache_key": cache_receipt["cache_key"],
        "manifest_sha256": cache_receipt["manifest_sha256"],
        "completion_sha256": cache_receipt["completion_sha256"],
        "selective_session_sha256": cache_receipt["session_sha256"],
        "inventory_sha256": inventory.report["inventory_sha256"],
        "heldout_manifest": inventory.heldout_manifest.as_dict(),
        "vocabulary_size": vocabulary_size,
        "top_k": top_k,
        "lexical_head_sha256": _streaming._tensor_sha256(head),
        "source_lexical_head_sha256": _streaming._tensor_sha256(source_head),
        "final_norm_sha256": _streaming._tensor_sha256(norm),
        "target_logit_projection_sha256": (
            None
            if logit_projection is None
            else logit_projection.projection_sha256
        ),
        "heldout_rows_retained": 0,
        "heldout_top_token_tables_retained": 0,
        "heldout_directions_retained": 0,
        "projection_and_candidate_evidence_loaded_one_phase_at_a_time": True,
        "heldout_excluded_from_basis_selection": True,
        "parent_forward_calls": 0,
        "trace_regeneration_calls": 0,
        "training_steps": 0,
    }
    gate_report = _sealed(gate_body, "gate_evidence_sha256")
    gate = Lfm2StreamingVocabularyMarginGateV2(
        vocabulary_size=vocabulary_size,
        top_k=top_k,
        manifest=inventory.heldout_manifest,
        bindings=inventory.bindings,
        selective_session=selective_session,
        lexical_head=head,
        final_norm=norm,
        config=config,
        logit_projection=logit_projection,
        target_tokenizer_binding=target_tokenizer_binding,
        expected_session_sha256=cache_receipt["session_sha256"],
        cache_key=cache_receipt["cache_key"],
        evidence_sha256=gate_report["gate_evidence_sha256"],
        report=gate_report,
    )
    report = _sealed(
        {
            "schema": LFM2_STREAMING_VOCABULARY_MARGIN_ADAPTER_SCHEMA,
            "enabled": True,
            "opt_in_only": True,
            "cache_authentication": {
                "cache_key": cache_receipt["cache_key"],
                "manifest_sha256": cache_receipt["manifest_sha256"],
                "completion_sha256": cache_receipt["completion_sha256"],
                "selective_session_sha256": cache_receipt["session_sha256"],
                "full_sealed_cache_file_and_inventory_integrity": True,
            },
            "inventory": _thaw(inventory.report),
            "vocabulary_domain": {
                "mode": (
                    "same-tokenizer-source-domain"
                    if target_tokenizer_binding is None
                    else "strict-bridge-target-physical-vocabulary"
                ),
                "source_vocabulary_size": source_vocabulary_size,
                "target_vocabulary_size": vocabulary_size,
                "target_tokenizer_binding": (
                    None
                    if target_tokenizer_binding is None
                    else target_tokenizer_binding.as_dict()
                ),
                "target_logit_projection": (
                    None
                    if logit_projection is None
                    else logit_projection.as_dict()
                ),
                "lexical_head_construction": (
                    "existing-parent-head"
                    if target_tokenizer_binding is None
                    else "parent-head-index-select-by-sealed-target-row-map"
                ),
                "matches_child_lexical_transport_row_selection": True,
                "candidate_numerical_gate_supported": (
                    target_tokenizer_binding is None
                ),
                "manual_whole_response_review_remains_verdict": (
                    target_tokenizer_binding is not None
                ),
            },
            "fit": {
                "statistics": fit_report,
                "selection_pass": fit_selection,
                "precomputed_observability": precomputed.as_dict(),
                "used_for_basis_selection": True,
                "all_declared_rows_consumed": True,
            },
            "heldout": {
                "manifest": inventory.heldout_manifest.as_dict(),
                "gate": gate.as_dict(),
                "used_for_basis_selection": False,
                "phase_payloads_retained_at_construction": 0,
            },
            "retention": {
                "fit_precomputed_gram_numel": source_width * source_width,
                "streaming_weighted_mean_numel_during_build": source_width,
                "retained_extra_lexical_head_copy_numel": (
                    0
                    if target_tokenizer_binding is None
                    else int(head.numel())
                ),
                "existing_parent_weight_references_are_not_copies": (
                    target_tokenizer_binding is None
                ),
                "target_head_copy_is_full_target_row_domain": (
                    target_tokenizer_binding is not None
                ),
                "retained_full_direction_rows": 0,
                "retained_full_residual_rows": 0,
                "retained_top_token_tables": 0,
                "numeric_evidence_bound": "H*H+H-plus-one-phase",
            },
            "coverage_claims": {
                "full_sealed_cache_integrity_authenticated": True,
                "complete_record_phase_inventory_authenticated": True,
                "vocabulary_margin_selected_tensor_subset_only": True,
                "selected_subset_is_not_claimed_as_all_cache_tensor_semantics": True,
            },
            "model_forward_calls": 0,
            "parent_trace_regeneration_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "autograd_used": False,
        },
        "adapter_sha256",
    )
    return Lfm2StreamingVocabularyMarginAdapterResult(
        precomputed_observability=precomputed,
        gate=gate,
        report=report,
    )


__all__ = [
    "LFM2_STREAMING_VOCABULARY_MARGIN_ADAPTER_SCHEMA",
    "LFM2_STREAMING_VOCABULARY_MARGIN_GATE_SCHEMA",
    "LFM2_STREAMING_VOCABULARY_MARGIN_INVENTORY_SCHEMA",
    "LFM2_TARGET_TOKENIZER_VOCABULARY_MARGIN_BINDING_SCHEMA",
    "Lfm2StreamingVocabularyMarginAdapterError",
    "Lfm2StreamingVocabularyMarginAdapterResult",
    "Lfm2StreamingVocabularyMarginGateV2",
    "Lfm2TargetTokenizerVocabularyMarginBinding",
    "build_lfm2_streaming_vocabulary_margin_adapter_v2",
]
