"""Portable immutable phase cache for one complete LFM2 body reclosure.

This v1 module owns content identity, tensor serialization, immutable cache
publication, and authentication.  It deliberately does *not* know how an
adapter installs the tensors into a child.  Acceptance thresholds and manual
review policy are run-local decisions and never participate in the reusable
math identity.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn

from .dynamic_fold_phase_cache import (
    DynamicFoldPhaseCache,
    FoldPhaseCacheAlreadyCompleteError,
    FoldPhaseCacheConcurrencyError,
    FoldPhaseCacheIntegrityError,
    FoldPhaseCacheTransaction,
    FoldPhaseEventLog,
    FoldPhaseKey,
    VerifiedFoldPhaseCacheEntry,
)


LFM2_BODY_RECLOSURE_PHASE = "lfm2-body-reclosure-v1"
LFM2_BODY_RECLOSURE_REQUEST_SCHEMA = "neural-forge-lfm2-body-reclosure-phase-request-v1"
LFM2_BODY_RECLOSURE_ABI_SCHEMA = "neural-forge-lfm2-body-reclosure-phase-abi-v1"
LFM2_BODY_RECLOSURE_ARTIFACT_SCHEMA = "neural-forge-lfm2-body-reclosure-artifact-v1"
LFM2_BODY_RECLOSURE_ARTIFACT_METADATA_SCHEMA = (
    "neural-forge-lfm2-body-reclosure-artifact-metadata-v1"
)
LFM2_BODY_RECLOSURE_RECEIPT_SCHEMA = "neural-forge-lfm2-body-reclosure-phase-receipt-v1"
LFM2_BODY_RECLOSURE_RESULT_SCHEMA = "neural-forge-lfm2-body-reclosure-phase-result-v1"
LFM2_BODY_RECLOSURE_SAFETENSORS_SCHEMA = (
    "neural-forge-lfm2-body-reclosure-safetensors-v1"
)

_REQUEST_FILENAME = "request.json"
_RECEIPT_FILENAME = "receipt.json"
_TENSOR_FILENAME = "tensors.safetensors"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")

# These fields are observations or acceptance decisions.  They must never be
# smuggled into math/evidence/ABI identity.  A caller may pass them through the
# separate ``acceptance_policy`` argument; that argument is validated only and
# intentionally discarded by the request builder.
ACCEPTANCE_ONLY_POLICY_FIELDS = frozenset(
    {
        "accepted",
        "acceptance",
        "acceptance_policy",
        "acceptance_threshold",
        "acceptance_tolerance",
        "validation_accepted",
        "validation_decision",
        "validation_tolerance",
        "terminal_composition_regression_tolerance",
        "performance_metrics_diagnostic_only",
        "final_performance_metrics_diagnostic_only",
        "all_layer_finite_numeric_checks_diagnostic_only",
        "dense_closed_loop_terminal_regression_tolerance",
        "dense_closed_loop_all_layer_finite_numeric_checks_diagnostic_only",
        "dense_closed_loop_intermediate_numeric_checks_diagnostic_only",
        "manual_whole_output_review_required",
        "manual_intelligence_review_required",
        "quality_gate",
        "coherence_gate",
        "gate_passed",
    }
)
_NORMALIZED_POLICY_FIELD = re.compile(r"[^a-z0-9]+")

_ZERO_EXECUTION_AUDIT = {
    "parent_forward_calls": 0,
    "parent_trace_regeneration_calls": 0,
    "training_steps": 0,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "optimizer_constructions": 0,
    "autograd_grad_calls": 0,
}

BodyReclosureProgressCallback = Callable[[str, Mapping[str, Any]], None]
BodyReclosureProducer = Callable[[], "Lfm2BodyReclosureArtifact"]


class Lfm2BodyReclosurePhaseCacheError(RuntimeError):
    """The body-reclosure phase request or bundle is malformed."""


class Lfm2BodyReclosurePhaseAuthenticationError(FoldPhaseCacheIntegrityError):
    """A completed body-reclosure phase failed authentication."""


class Lfm2BodyReclosurePhaseInProgressError(Lfm2BodyReclosurePhaseCacheError):
    """An exclusive cache reservation prevents ownership of this phase key.

    The disposition is deliberately nonblocking.  A body reclosure can run for
    hours, so callers must retry the ordinary authenticated lookup later rather
    than duplicate the producer.  Reservation presence does not authenticate
    owner liveness, and this path never infers that an old lock is safe to steal.
    """

    status = "in_progress"
    retryable = True
    producer_invoked = False
    ownership_stolen = False
    owner_liveness_authenticated = False

    def __init__(self, phase_key_sha256: str) -> None:
        self.phase_key_sha256 = _digest(
            phase_key_sha256,
            "phase_key_sha256",
        )
        super().__init__(
            "Exact body-reclosure phase is already in progress; retry the "
            "authenticated lookup later."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "neural-forge-lfm2-body-reclosure-in-progress-v1",
            "status": self.status,
            "phase_key_sha256": self.phase_key_sha256,
            "retryable": self.retryable,
            "producer_invoked": self.producer_invoked,
            "ownership_stolen": self.ownership_stolen,
            "owner_liveness_authenticated": self.owner_liveness_authenticated,
        }


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
        raise Lfm2BodyReclosurePhaseCacheError(
            f"{name} must be finite canonical JSON."
        ) from exc


def _canonical_value(value: Any, name: str) -> Any:
    return json.loads(_canonical_json_bytes(value, name).decode("ascii"))


def _canonical_mapping(value: Any, name: str) -> dict[str, Any]:
    normalized = _canonical_value(value, name)
    if not isinstance(normalized, dict):
        raise Lfm2BodyReclosurePhaseCacheError(f"{name} must be a mapping.")
    return normalized


def _canonical_sha256(value: Any, name: str) -> str:
    return hashlib.sha256(_canonical_json_bytes(value, name)).hexdigest()


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


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2BodyReclosurePhaseCacheError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise Lfm2BodyReclosurePhaseCacheError(
            f"{name} must be a stable identifier without path separators."
        )
    return value


def _normalized_policy_field(value: str) -> str:
    return _NORMALIZED_POLICY_FIELD.sub("_", value.casefold()).strip("_")


def _is_acceptance_only_policy_field(value: str) -> bool:
    field_name = _normalized_policy_field(value)
    if field_name in ACCEPTANCE_ONLY_POLICY_FIELDS:
        return True
    if field_name.endswith(
        (
            "_performance_metrics_diagnostic_only",
            "_all_layer_finite_numeric_checks_diagnostic_only",
            "_intermediate_numeric_checks_diagnostic_only",
        )
    ):
        return True
    # Terminal/final regression tolerance is a post-fold decision.  Boundary
    # tolerances (for example moving_gauge_boundary_regression_tolerance) can
    # change writer selection and intentionally remain math identity.
    return (
        "terminal" in field_name.split("_") or "final" in field_name.split("_")
    ) and field_name.endswith("_regression_tolerance")


def _assert_policy_neutral(value: Any, name: str, path: str = "") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            location = f"{path}.{key}" if path else key
            if _is_acceptance_only_policy_field(key):
                raise Lfm2BodyReclosurePhaseCacheError(
                    f"{name} contains acceptance-only policy field {location!r}."
                )
            _assert_policy_neutral(item, name, location)
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _assert_policy_neutral(item, name, f"{path}[{index}]")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(_canonical_json_bytes(dict(value), path.name) + b"\n")
    except OSError as exc:
        raise Lfm2BodyReclosurePhaseCacheError(
            f"Could not write staged {path.name}."
        ) from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Lfm2BodyReclosurePhaseAuthenticationError(
            f"Body-reclosure {path.name} is unreadable."
        ) from exc
    if (
        not isinstance(value, dict)
        or raw != _canonical_json_bytes(value, path.name) + b"\n"
    ):
        raise Lfm2BodyReclosurePhaseAuthenticationError(
            f"Body-reclosure {path.name} is not strict canonical JSON."
        )
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise Lfm2BodyReclosurePhaseAuthenticationError(
            f"Could not hash body-reclosure payload {path.name}."
        ) from exc
    return digest.hexdigest()


_REQUEST_DOCUMENT_FIELDS = frozenset(
    {
        "schema",
        "phase_key",
        "phase_key_sha256",
        "portable_pre_child_state_sha256",
        "portable_pre_child_topology_sha256",
        "pre_capture_sha256",
        "math_contract",
        "math_contract_sha256",
        "evidence_contract",
        "evidence_contract_sha256",
        "abi_contract",
        "abi_contract_sha256",
        "acceptance_policy_serialized",
        "adapter_restore_wiring_serialized",
    }
)


def _request_phase_key(
    *,
    pre_state: str,
    pre_topology: str,
    pre_capture: str,
    math: Mapping[str, Any],
    evidence: Mapping[str, Any],
    abi_contract: Mapping[str, Any],
) -> FoldPhaseKey:
    math_sha = _canonical_sha256(math, "math_contract")
    evidence_sha = _canonical_sha256(evidence, "evidence_contract")
    abi_sha = _canonical_sha256(abi_contract, "abi_contract")
    return FoldPhaseKey(
        phase_name=LFM2_BODY_RECLOSURE_PHASE,
        dependency_sha256s={
            "pre-child-state": pre_state,
            "pre-child-topology": pre_topology,
            "pre-capture": pre_capture,
            "math-contract": math_sha,
            "evidence-contract": evidence_sha,
            "adapter-abi": abi_sha,
        },
        numeric_contract={
            "schema": "neural-forge-lfm2-body-reclosure-math-contract-v1",
            "math_contract": dict(math),
        },
        abi={
            "schema": LFM2_BODY_RECLOSURE_ABI_SCHEMA,
            "storage": "strict-canonical-json-plus-safetensors",
            "pickle_allowed": False,
            "adapter_restore_wiring_serialized": False,
            "request_schema": LFM2_BODY_RECLOSURE_REQUEST_SCHEMA,
            "receipt_schema": LFM2_BODY_RECLOSURE_RECEIPT_SCHEMA,
            "tensor_schema": LFM2_BODY_RECLOSURE_SAFETENSORS_SCHEMA,
            "abi_contract": dict(abi_contract),
        },
    )


def _request_document(
    phase_key: FoldPhaseKey,
    *,
    pre_state: str,
    pre_topology: str,
    pre_capture: str,
    math: Mapping[str, Any],
    evidence: Mapping[str, Any],
    abi_contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": LFM2_BODY_RECLOSURE_REQUEST_SCHEMA,
        "phase_key": phase_key.as_dict(),
        "phase_key_sha256": phase_key.phase_key_sha256,
        "portable_pre_child_state_sha256": pre_state,
        "portable_pre_child_topology_sha256": pre_topology,
        "pre_capture_sha256": pre_capture,
        "math_contract": dict(math),
        "math_contract_sha256": _canonical_sha256(math, "math_contract"),
        "evidence_contract": dict(evidence),
        "evidence_contract_sha256": _canonical_sha256(
            evidence,
            "evidence_contract",
        ),
        "abi_contract": dict(abi_contract),
        "abi_contract_sha256": _canonical_sha256(
            abi_contract,
            "abi_contract",
        ),
        "acceptance_policy_serialized": False,
        "adapter_restore_wiring_serialized": False,
    }


@dataclass(frozen=True, slots=True)
class Lfm2BodyReclosurePhaseRequest:
    """Policy-neutral portable identity for one exact reclosure bundle."""

    phase_key: FoldPhaseKey
    document: Mapping[str, Any]
    request_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.phase_key, FoldPhaseKey):
            raise TypeError("phase_key must be FoldPhaseKey.")
        document = _canonical_mapping(self.document, "body-reclosure request")
        if set(document) != _REQUEST_DOCUMENT_FIELDS:
            raise Lfm2BodyReclosurePhaseCacheError(
                "Body-reclosure request does not contain exactly the v1 fields."
            )
        pre_state = _digest(
            document.get("portable_pre_child_state_sha256"),
            "portable_pre_child_state_sha256",
        )
        pre_topology = _digest(
            document.get("portable_pre_child_topology_sha256"),
            "portable_pre_child_topology_sha256",
        )
        pre_capture = _digest(
            document.get("pre_capture_sha256"),
            "pre_capture_sha256",
        )
        math = _canonical_mapping(document.get("math_contract"), "math_contract")
        evidence = _canonical_mapping(
            document.get("evidence_contract"),
            "evidence_contract",
        )
        abi_contract = _canonical_mapping(
            document.get("abi_contract"),
            "abi_contract",
        )
        for value, name in (
            (math, "math_contract"),
            (evidence, "evidence_contract"),
            (abi_contract, "abi_contract"),
        ):
            if not value:
                raise Lfm2BodyReclosurePhaseCacheError(f"{name} cannot be empty.")
            _assert_policy_neutral(value, name)
        expected_key = _request_phase_key(
            pre_state=pre_state,
            pre_topology=pre_topology,
            pre_capture=pre_capture,
            math=math,
            evidence=evidence,
            abi_contract=abi_contract,
        )
        expected_document = _request_document(
            expected_key,
            pre_state=pre_state,
            pre_topology=pre_topology,
            pre_capture=pre_capture,
            math=math,
            evidence=evidence,
            abi_contract=abi_contract,
        )
        expected = _canonical_sha256(document, "body-reclosure request")
        if (
            document != expected_document
            or self.phase_key.as_dict() != expected_key.as_dict()
            or self.request_sha256 not in {"", expected}
        ):
            raise Lfm2BodyReclosurePhaseCacheError(
                "Body-reclosure request differs from its phase key or seal."
            )
        object.__setattr__(self, "document", _deep_freeze(document))
        object.__setattr__(self, "request_sha256", expected)

    def as_dict(self) -> dict[str, Any]:
        return {
            **_deep_thaw(self.document),
            "request_sha256": self.request_sha256,
        }


def build_lfm2_body_reclosure_phase_request(
    *,
    portable_pre_child_state_sha256: str,
    portable_pre_child_topology_sha256: str,
    pre_capture_sha256: str,
    math_contract: Mapping[str, Any],
    evidence_contract: Mapping[str, Any],
    abi_contract: Mapping[str, Any],
    acceptance_policy: Mapping[str, Any] | None = None,
) -> Lfm2BodyReclosurePhaseRequest:
    """Build a reusable key while excluding run-local acceptance policy.

    ``acceptance_policy`` is canonicalized to reject malformed input, but its
    values are not retained or hashed.  Acceptance-only fields placed in any
    identity-bearing contract are rejected instead of silently contaminating
    the reusable math key.
    """

    pre_state = _digest(
        portable_pre_child_state_sha256,
        "portable_pre_child_state_sha256",
    )
    pre_topology = _digest(
        portable_pre_child_topology_sha256,
        "portable_pre_child_topology_sha256",
    )
    pre_capture = _digest(pre_capture_sha256, "pre_capture_sha256")
    math = _canonical_mapping(math_contract, "math_contract")
    evidence = _canonical_mapping(evidence_contract, "evidence_contract")
    abi_value = _canonical_mapping(abi_contract, "abi_contract")
    for value, name in (
        (math, "math_contract"),
        (evidence, "evidence_contract"),
        (abi_value, "abi_contract"),
    ):
        if not value:
            raise Lfm2BodyReclosurePhaseCacheError(f"{name} cannot be empty.")
        _assert_policy_neutral(value, name)
    if acceptance_policy is not None:
        _canonical_mapping(acceptance_policy, "acceptance_policy")

    key = _request_phase_key(
        pre_state=pre_state,
        pre_topology=pre_topology,
        pre_capture=pre_capture,
        math=math,
        evidence=evidence,
        abi_contract=abi_value,
    )
    document = _request_document(
        key,
        pre_state=pre_state,
        pre_topology=pre_topology,
        pre_capture=pre_capture,
        math=math,
        evidence=evidence,
        abi_contract=abi_value,
    )
    return Lfm2BodyReclosurePhaseRequest(phase_key=key, document=document)


describe_lfm2_body_reclosure_phase = build_lfm2_body_reclosure_phase_request


def _tensor_descriptor(name: str, tensor: Any) -> dict[str, Any]:
    import torch

    raw = (
        tensor.detach()
        .to(device="cpu")
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    return {
        "name": name,
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "numel": int(tensor.numel()),
        "size_bytes": int(len(raw)),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


_ARTIFACT_METADATA_INPUT_FIELDS = frozenset(
    {
        "math_artifact_sha256",
        "portable_post_child_state_sha256",
        "portable_post_child_topology_sha256",
        "post_capture_sha256",
        "cold_math_report_sha256",
        "restore_manifest",
        "validation_manifest",
    }
)
_ARTIFACT_METADATA_FIELDS = _ARTIFACT_METADATA_INPUT_FIELDS | {
    "schema",
    "restore_manifest_sha256",
    "validation_manifest_sha256",
}


def _artifact_metadata(value: Any) -> dict[str, Any]:
    metadata = _canonical_mapping(value, "artifact_metadata")
    metadata_fields = frozenset(metadata)
    if metadata_fields not in (
        _ARTIFACT_METADATA_INPUT_FIELDS,
        _ARTIFACT_METADATA_FIELDS,
    ):
        raise Lfm2BodyReclosurePhaseCacheError(
            "artifact_metadata must contain exactly the portable v1 output fields."
        )
    for name in (
        "math_artifact_sha256",
        "portable_post_child_state_sha256",
        "portable_post_child_topology_sha256",
        "post_capture_sha256",
        "cold_math_report_sha256",
    ):
        metadata[name] = _digest(metadata.get(name), f"artifact_metadata.{name}")
    for name in ("restore_manifest", "validation_manifest"):
        manifest = _canonical_mapping(metadata.get(name), f"artifact_metadata.{name}")
        if not manifest:
            raise Lfm2BodyReclosurePhaseCacheError(
                f"artifact_metadata.{name} cannot be empty."
            )
        _assert_policy_neutral(manifest, f"artifact_metadata.{name}")
        metadata[name] = manifest
    normalized = {
        "schema": LFM2_BODY_RECLOSURE_ARTIFACT_METADATA_SCHEMA,
        **{name: metadata[name] for name in sorted(_ARTIFACT_METADATA_INPUT_FIELDS)},
        "restore_manifest_sha256": _canonical_sha256(
            metadata["restore_manifest"],
            "restore_manifest",
        ),
        "validation_manifest_sha256": _canonical_sha256(
            metadata["validation_manifest"],
            "validation_manifest",
        ),
    }
    if metadata_fields == _ARTIFACT_METADATA_FIELDS and metadata != normalized:
        raise Lfm2BodyReclosurePhaseCacheError(
            "artifact_metadata schema or derived manifest digest is invalid."
        )
    return normalized


@dataclass(frozen=True, slots=True)
class Lfm2BodyReclosureArtifact:
    """Portable tensors plus authenticated output metadata and zero-use audit."""

    tensors: Mapping[str, Any] = field(repr=False)
    artifact_metadata: Mapping[str, Any]
    aliases: Mapping[str, str] = field(default_factory=dict)
    execution_audit: Mapping[str, int] = field(
        default_factory=lambda: dict(_ZERO_EXECUTION_AUDIT)
    )
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional HF boundary
            raise Lfm2BodyReclosurePhaseCacheError(
                "torch is required for body-reclosure artifacts."
            ) from exc
        if not isinstance(self.tensors, Mapping) or not self.tensors:
            raise Lfm2BodyReclosurePhaseCacheError(
                "Body-reclosure artifact tensors cannot be empty."
            )
        tensors: dict[str, Any] = {}
        for raw_name, raw_tensor in sorted(self.tensors.items()):
            name = _identifier(raw_name, "tensor name")
            if not isinstance(raw_tensor, torch.Tensor):
                raise Lfm2BodyReclosurePhaseCacheError(
                    f"Body-reclosure tensor {name!r} is not a torch.Tensor."
                )
            if raw_tensor.layout != torch.strided:
                raise Lfm2BodyReclosurePhaseCacheError(
                    f"Body-reclosure tensor {name!r} must be dense and strided."
                )
            tensors[name] = raw_tensor.detach().to(device="cpu").contiguous().clone()
        aliases: dict[str, str] = {}
        if not isinstance(self.aliases, Mapping):
            raise Lfm2BodyReclosurePhaseCacheError("aliases must be a mapping.")
        for raw_alias, raw_target in sorted(self.aliases.items()):
            alias = _identifier(raw_alias, "tensor alias")
            target = _identifier(raw_target, f"tensor alias {alias!r} target")
            if alias in tensors:
                raise Lfm2BodyReclosurePhaseCacheError(
                    f"Tensor alias {alias!r} collides with a stored tensor."
                )
            if target not in tensors:
                raise Lfm2BodyReclosurePhaseCacheError(
                    f"Tensor alias {alias!r} has an unknown target {target!r}."
                )
            aliases[alias] = target
        audit = _canonical_mapping(self.execution_audit, "execution_audit")
        if audit != _ZERO_EXECUTION_AUDIT:
            raise Lfm2BodyReclosurePhaseCacheError(
                "Body-reclosure phase audit must contain the exact zero-use v1 counters."
            )
        descriptors = [
            _tensor_descriptor(name, tensor) for name, tensor in tensors.items()
        ]
        metadata = _artifact_metadata(self.artifact_metadata)
        body = {
            "schema": LFM2_BODY_RECLOSURE_ARTIFACT_SCHEMA,
            "artifact_metadata": metadata,
            "tensor_descriptors": descriptors,
            "tensor_aliases": aliases,
            "execution_audit": audit,
        }
        expected = _canonical_sha256(body, "body-reclosure artifact")
        if self.artifact_sha256 not in {"", expected}:
            raise Lfm2BodyReclosurePhaseCacheError(
                "Body-reclosure artifact SHA-256 is invalid."
            )
        object.__setattr__(self, "tensors", MappingProxyType(tensors))
        object.__setattr__(self, "artifact_metadata", _deep_freeze(metadata))
        object.__setattr__(self, "aliases", MappingProxyType(aliases))
        object.__setattr__(self, "execution_audit", MappingProxyType(audit))
        object.__setattr__(self, "artifact_sha256", expected)

    @property
    def tensor_descriptors(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            MappingProxyType(_tensor_descriptor(name, tensor))
            for name, tensor in self.tensors.items()
        )

    def reauthenticate(self) -> str:
        """Recompute the complete tensor-bound artifact seal in live memory."""

        body = {
            "schema": LFM2_BODY_RECLOSURE_ARTIFACT_SCHEMA,
            "artifact_metadata": _deep_thaw(self.artifact_metadata),
            "tensor_descriptors": [dict(value) for value in self.tensor_descriptors],
            "tensor_aliases": dict(self.aliases),
            "execution_audit": dict(self.execution_audit),
        }
        actual = _canonical_sha256(body, "body-reclosure artifact")
        if actual != self.artifact_sha256:
            raise Lfm2BodyReclosurePhaseAuthenticationError(
                "Body-reclosure artifact tensors changed after authentication."
            )
        return actual

    def authenticated_tensor_copies(self) -> Mapping[str, Any]:
        """Return detached copies bracketed by live artifact reauthentication.

        Adapter integration should consume only these copies.  The method does
        not perform any model-family restore or parameter wiring.
        """

        before = self.reauthenticate()
        copies = {
            name: tensor.detach().to(device="cpu").contiguous().clone()
            for name, tensor in self.tensors.items()
        }
        after = self.reauthenticate()
        if before != after or [
            _tensor_descriptor(name, tensor) for name, tensor in copies.items()
        ] != [dict(value) for value in self.tensor_descriptors]:
            raise Lfm2BodyReclosurePhaseAuthenticationError(
                "Body-reclosure artifact changed while tensor copies were made."
            )
        return MappingProxyType(copies)


@dataclass(frozen=True, slots=True)
class Lfm2ExistingBodyReclosurePhaseReuseRequest:
    """Exact hit-only request; a miss can never invoke the producer."""

    phase_key_sha256: str
    expected_completion_sha256: str | None = None
    expected_artifact_sha256: str | None = None
    expected_math_artifact_sha256: str | None = None
    expected_post_child_state_sha256: str | None = None
    expected_post_child_topology_sha256: str | None = None
    expected_post_capture_sha256: str | None = None
    expected_cold_math_report_sha256: str | None = None
    expected_restore_manifest_sha256: str | None = None
    expected_validation_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "phase_key_sha256",
            _digest(self.phase_key_sha256, "phase_key_sha256"),
        )
        for name in (
            "expected_completion_sha256",
            "expected_artifact_sha256",
            "expected_math_artifact_sha256",
            "expected_post_child_state_sha256",
            "expected_post_child_topology_sha256",
            "expected_post_capture_sha256",
            "expected_cold_math_report_sha256",
            "expected_restore_manifest_sha256",
            "expected_validation_manifest_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _digest(value, name))

    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema": "neural-forge-lfm2-existing-body-reclosure-reuse-request-v1",
            "phase_key_sha256": self.phase_key_sha256,
            "expected_completion_sha256": self.expected_completion_sha256,
            "expected_artifact_sha256": self.expected_artifact_sha256,
            "expected_math_artifact_sha256": self.expected_math_artifact_sha256,
            "expected_post_child_state_sha256": (self.expected_post_child_state_sha256),
            "expected_post_child_topology_sha256": (
                self.expected_post_child_topology_sha256
            ),
            "expected_post_capture_sha256": self.expected_post_capture_sha256,
            "expected_cold_math_report_sha256": (self.expected_cold_math_report_sha256),
            "expected_restore_manifest_sha256": (self.expected_restore_manifest_sha256),
            "expected_validation_manifest_sha256": (
                self.expected_validation_manifest_sha256
            ),
            "cache_miss_build_allowed": False,
        }
        return {
            **body,
            "request_sha256": _canonical_sha256(body, "reuse request"),
        }


@dataclass(frozen=True, slots=True)
class Lfm2BodyReclosurePhaseCacheContext:
    """Immutable cache, optional event/progress, and optional hit-only mode."""

    cache: DynamicFoldPhaseCache
    events: FoldPhaseEventLog | None = None
    progress: BodyReclosureProgressCallback | None = None
    existing_phase_reuse_request: Lfm2ExistingBodyReclosurePhaseReuseRequest | None = (
        None
    )

    def __post_init__(self) -> None:
        if not isinstance(self.cache, DynamicFoldPhaseCache):
            raise TypeError("cache must be DynamicFoldPhaseCache.")
        if self.events is not None and not isinstance(self.events, FoldPhaseEventLog):
            raise TypeError("events must be FoldPhaseEventLog or None.")
        if self.progress is not None and not callable(self.progress):
            raise TypeError("progress must be callable or None.")
        if self.existing_phase_reuse_request is not None and not isinstance(
            self.existing_phase_reuse_request,
            Lfm2ExistingBodyReclosurePhaseReuseRequest,
        ):
            raise TypeError(
                "existing_phase_reuse_request must be "
                "Lfm2ExistingBodyReclosurePhaseReuseRequest or None."
            )


@dataclass(frozen=True, slots=True)
class AuthenticatedLfm2BodyReclosurePhase:
    """One fully authenticated portable bundle; no install wiring is applied."""

    request: Lfm2BodyReclosurePhaseRequest
    artifact: Lfm2BodyReclosureArtifact
    entry: VerifiedFoldPhaseCacheEntry
    cache_hit: bool
    producer_invoked: bool
    publication_race_lost: bool

    @property
    def completion_sha256(self) -> str:
        return self.entry.completion_sha256


def _progress(
    context: Lfm2BodyReclosurePhaseCacheContext,
    stage: str,
    **details: Any,
) -> None:
    if context.progress is not None:
        context.progress(stage, MappingProxyType(dict(details)))


def _tensor_metadata(
    request: Lfm2BodyReclosurePhaseRequest,
    artifact: Lfm2BodyReclosureArtifact,
) -> dict[str, str]:
    descriptors = [dict(value) for value in artifact.tensor_descriptors]
    artifact_metadata = _deep_thaw(artifact.artifact_metadata)
    return {
        "schema": LFM2_BODY_RECLOSURE_SAFETENSORS_SCHEMA,
        "request_sha256": request.request_sha256,
        "phase_key_sha256": request.phase_key.phase_key_sha256,
        "math_artifact_sha256": artifact_metadata["math_artifact_sha256"],
        "artifact_metadata_sha256": _canonical_sha256(
            artifact_metadata,
            "artifact metadata",
        ),
        "artifact_sha256": artifact.artifact_sha256,
        "tensor_descriptors_sha256": _canonical_sha256(
            descriptors,
            "tensor descriptors",
        ),
        "tensor_aliases_sha256": _canonical_sha256(
            dict(artifact.aliases),
            "tensor aliases",
        ),
        "execution_audit_sha256": _canonical_sha256(
            dict(artifact.execution_audit),
            "execution audit",
        ),
        "tensor_count": str(len(artifact.tensors)),
        "pickle_allowed": "false",
    }


def _write_bundle(
    payload_path: Path,
    request: Lfm2BodyReclosurePhaseRequest,
    artifact: Lfm2BodyReclosureArtifact,
) -> dict[str, Any]:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - optional HF boundary
        raise Lfm2BodyReclosurePhaseCacheError(
            "safetensors is required for body-reclosure phase caching."
        ) from exc
    tensor_snapshot = artifact.authenticated_tensor_copies()
    _write_json(payload_path / _REQUEST_FILENAME, request.as_dict())
    tensor_path = payload_path / _TENSOR_FILENAME
    tensor_metadata = _tensor_metadata(request, artifact)
    try:
        save_file(
            dict(tensor_snapshot),
            tensor_path,
            metadata=tensor_metadata,
        )
    except Exception as exc:
        raise Lfm2BodyReclosurePhaseCacheError(
            "Could not write body-reclosure SafeTensors payload."
        ) from exc
    artifact.reauthenticate()
    tensor_file = {
        "filename": _TENSOR_FILENAME,
        "format": "safetensors",
        "pickle_allowed": False,
        "metadata": tensor_metadata,
        "size_bytes": tensor_path.stat().st_size,
        "sha256": _file_sha256(tensor_path),
    }
    body = {
        "schema": LFM2_BODY_RECLOSURE_RECEIPT_SCHEMA,
        "phase_key_sha256": request.phase_key.phase_key_sha256,
        "request_sha256": request.request_sha256,
        "portable_pre_child_state_sha256": request.document[
            "portable_pre_child_state_sha256"
        ],
        "portable_pre_child_topology_sha256": request.document[
            "portable_pre_child_topology_sha256"
        ],
        "pre_capture_sha256": request.document["pre_capture_sha256"],
        "artifact_sha256": artifact.artifact_sha256,
        "artifact_metadata": _deep_thaw(artifact.artifact_metadata),
        "artifact_metadata_sha256": _canonical_sha256(
            _deep_thaw(artifact.artifact_metadata),
            "artifact metadata",
        ),
        "tensor_file": tensor_file,
        "tensor_descriptors": [dict(value) for value in artifact.tensor_descriptors],
        "tensor_aliases": dict(artifact.aliases),
        "execution_audit": dict(artifact.execution_audit),
        "adapter_restore_wiring_serialized": False,
    }
    receipt = {
        **body,
        "receipt_sha256": _canonical_sha256(body, "body-reclosure receipt"),
    }
    _write_json(payload_path / _RECEIPT_FILENAME, receipt)
    return receipt


def _completion_result(receipt: Mapping[str, Any]) -> dict[str, Any]:
    tensor_file = receipt["tensor_file"]
    metadata = receipt["artifact_metadata"]
    return {
        "schema": LFM2_BODY_RECLOSURE_RESULT_SCHEMA,
        "receipt_sha256": receipt["receipt_sha256"],
        "artifact_sha256": receipt["artifact_sha256"],
        "artifact_metadata_sha256": receipt["artifact_metadata_sha256"],
        "math_artifact_sha256": metadata["math_artifact_sha256"],
        "portable_post_child_state_sha256": metadata[
            "portable_post_child_state_sha256"
        ],
        "portable_post_child_topology_sha256": metadata[
            "portable_post_child_topology_sha256"
        ],
        "post_capture_sha256": metadata["post_capture_sha256"],
        "cold_math_report_sha256": metadata["cold_math_report_sha256"],
        "restore_manifest_sha256": metadata["restore_manifest_sha256"],
        "validation_manifest_sha256": metadata["validation_manifest_sha256"],
        "tensor_file_sha256": tensor_file["sha256"],
        "tensor_descriptors_sha256": _canonical_sha256(
            receipt["tensor_descriptors"],
            "tensor descriptors",
        ),
        "tensor_aliases_sha256": _canonical_sha256(
            receipt["tensor_aliases"],
            "tensor aliases",
        ),
        "zero_execution_audit_sha256": _canonical_sha256(
            receipt["execution_audit"],
            "execution audit",
        ),
    }


def _authenticate_receipt(
    receipt: Mapping[str, Any],
    request: Lfm2BodyReclosurePhaseRequest,
) -> dict[str, Any]:
    value = _canonical_mapping(receipt, "body-reclosure receipt")
    body = dict(value)
    claimed = body.pop("receipt_sha256", None)
    if claimed != _canonical_sha256(body, "body-reclosure receipt"):
        raise Lfm2BodyReclosurePhaseAuthenticationError(
            "Body-reclosure receipt seal is invalid."
        )
    expected_request = request.document
    try:
        metadata = _artifact_metadata(value.get("artifact_metadata"))
    except Lfm2BodyReclosurePhaseCacheError as exc:
        raise Lfm2BodyReclosurePhaseAuthenticationError(
            "Body-reclosure receipt output metadata is invalid."
        ) from exc
    if (
        value.get("schema") != LFM2_BODY_RECLOSURE_RECEIPT_SCHEMA
        or value.get("phase_key_sha256") != request.phase_key.phase_key_sha256
        or value.get("request_sha256") != request.request_sha256
        or value.get("portable_pre_child_state_sha256")
        != expected_request["portable_pre_child_state_sha256"]
        or value.get("portable_pre_child_topology_sha256")
        != expected_request["portable_pre_child_topology_sha256"]
        or value.get("pre_capture_sha256") != expected_request["pre_capture_sha256"]
        or value.get("artifact_metadata") != metadata
        or value.get("artifact_metadata_sha256")
        != _canonical_sha256(metadata, "artifact metadata")
        or value.get("execution_audit") != _ZERO_EXECUTION_AUDIT
        or value.get("adapter_restore_wiring_serialized") is not False
    ):
        raise Lfm2BodyReclosurePhaseAuthenticationError(
            "Body-reclosure receipt differs from its exact request or zero-use audit."
        )
    return value


def _load_entry(
    context: Lfm2BodyReclosurePhaseCacheContext,
    request: Lfm2BodyReclosurePhaseRequest,
    entry: VerifiedFoldPhaseCacheEntry,
) -> Lfm2BodyReclosureArtifact:
    if entry.phase_key.as_dict() != request.phase_key.as_dict():
        raise Lfm2BodyReclosurePhaseAuthenticationError(
            "Body-reclosure cache entry differs from the requested key."
        )
    files = entry.payload_inventory.get("files")
    if not isinstance(files, list) or {
        value.get("path") for value in files if isinstance(value, Mapping)
    } != {_REQUEST_FILENAME, _RECEIPT_FILENAME, _TENSOR_FILENAME}:
        raise Lfm2BodyReclosurePhaseAuthenticationError(
            "Body-reclosure payload inventory is missing or foreign."
        )
    if _read_json(entry.payload_path / _REQUEST_FILENAME) != request.as_dict():
        raise Lfm2BodyReclosurePhaseAuthenticationError(
            "Persisted body-reclosure request differs from the lookup."
        )
    receipt = _authenticate_receipt(
        _read_json(entry.payload_path / _RECEIPT_FILENAME),
        request,
    )
    tensor_file = receipt.get("tensor_file")
    tensor_path = entry.payload_path / _TENSOR_FILENAME
    if (
        not isinstance(tensor_file, dict)
        or tensor_file.get("filename") != _TENSOR_FILENAME
        or tensor_file.get("format") != "safetensors"
        or tensor_file.get("pickle_allowed") is not False
        or tensor_file.get("size_bytes") != tensor_path.stat().st_size
        or tensor_file.get("sha256") != _file_sha256(tensor_path)
    ):
        raise Lfm2BodyReclosurePhaseAuthenticationError(
            "Body-reclosure SafeTensors file seal is invalid."
        )
    try:
        from safetensors import safe_open

        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            if handle.metadata() != tensor_file.get("metadata"):
                raise Lfm2BodyReclosurePhaseAuthenticationError(
                    "Body-reclosure SafeTensors metadata changed."
                )
            names = tuple(sorted(handle.keys()))
            descriptor_names = tuple(
                value.get("name") for value in receipt["tensor_descriptors"]
            )
            if names != descriptor_names:
                raise Lfm2BodyReclosurePhaseAuthenticationError(
                    "Body-reclosure SafeTensors keys changed."
                )
            tensors = {
                name: handle.get_tensor(name)
                .detach()
                .to(device="cpu")
                .contiguous()
                .clone()
                for name in names
            }
    except Lfm2BodyReclosurePhaseAuthenticationError:
        raise
    except Exception as exc:
        raise Lfm2BodyReclosurePhaseAuthenticationError(
            "Could not load body-reclosure SafeTensors payload."
        ) from exc
    artifact = Lfm2BodyReclosureArtifact(
        tensors=tensors,
        artifact_metadata=receipt["artifact_metadata"],
        aliases=receipt["tensor_aliases"],
        execution_audit=receipt["execution_audit"],
    )
    if (
        artifact.artifact_sha256 != receipt.get("artifact_sha256")
        or [dict(value) for value in artifact.tensor_descriptors]
        != receipt.get("tensor_descriptors")
        or entry.result != _completion_result(receipt)
    ):
        raise Lfm2BodyReclosurePhaseAuthenticationError(
            "Loaded body-reclosure tensors differ from their receipt."
        )
    # A second full inventory lookup after tensor materialization closes the
    # read/deserialize window and detects replacement or mutation during load.
    reverified = context.cache.lookup(request.phase_key)
    if (
        reverified is None
        or reverified.completion_sha256 != entry.completion_sha256
        or dict(reverified.payload_inventory) != dict(entry.payload_inventory)
        or dict(reverified.result) != dict(entry.result)
    ):
        raise Lfm2BodyReclosurePhaseAuthenticationError(
            "Body-reclosure phase changed while tensors were loaded."
        )
    return artifact


def _verify_hit_only_request(
    request: Lfm2BodyReclosurePhaseRequest,
    reuse: Lfm2ExistingBodyReclosurePhaseReuseRequest,
) -> None:
    if reuse.phase_key_sha256 != request.phase_key.phase_key_sha256:
        raise Lfm2BodyReclosurePhaseCacheError(
            "Exact body-reclosure reuse request differs from the live phase key."
        )


def _verify_hit_only_artifact(
    reuse: Lfm2ExistingBodyReclosurePhaseReuseRequest,
    artifact: Lfm2BodyReclosureArtifact,
) -> None:
    metadata = artifact.artifact_metadata
    expected = {
        "expected_artifact_sha256": artifact.artifact_sha256,
        "expected_math_artifact_sha256": metadata["math_artifact_sha256"],
        "expected_post_child_state_sha256": metadata[
            "portable_post_child_state_sha256"
        ],
        "expected_post_child_topology_sha256": metadata[
            "portable_post_child_topology_sha256"
        ],
        "expected_post_capture_sha256": metadata["post_capture_sha256"],
        "expected_cold_math_report_sha256": metadata["cold_math_report_sha256"],
        "expected_restore_manifest_sha256": metadata["restore_manifest_sha256"],
        "expected_validation_manifest_sha256": metadata["validation_manifest_sha256"],
    }
    for name, actual in expected.items():
        pinned = getattr(reuse, name)
        if pinned is not None and pinned != actual:
            raise Lfm2BodyReclosurePhaseAuthenticationError(
                f"Exact body-reclosure output differs from {name}."
            )


def _validate_context_and_request(
    context: Lfm2BodyReclosurePhaseCacheContext,
    request: Lfm2BodyReclosurePhaseRequest,
) -> Lfm2ExistingBodyReclosurePhaseReuseRequest | None:
    if not isinstance(context, Lfm2BodyReclosurePhaseCacheContext):
        raise TypeError("context must be Lfm2BodyReclosurePhaseCacheContext.")
    if not isinstance(request, Lfm2BodyReclosurePhaseRequest):
        raise TypeError("request must be Lfm2BodyReclosurePhaseRequest.")
    reuse = context.existing_phase_reuse_request
    if reuse is not None:
        _verify_hit_only_request(request, reuse)
    return reuse


def _authenticated_result(
    context: Lfm2BodyReclosurePhaseCacheContext,
    request: Lfm2BodyReclosurePhaseRequest,
    entry: VerifiedFoldPhaseCacheEntry,
    *,
    cache_hit: bool,
    producer_invoked: bool,
    publication_race_lost: bool,
    candidate: Lfm2BodyReclosureArtifact | None = None,
) -> AuthenticatedLfm2BodyReclosurePhase:
    reuse = context.existing_phase_reuse_request
    if (
        reuse is not None
        and reuse.expected_completion_sha256 is not None
        and reuse.expected_completion_sha256 != entry.completion_sha256
    ):
        raise Lfm2BodyReclosurePhaseAuthenticationError(
            "Exact body-reclosure completion differs from the reuse request."
        )
    candidate_sha256 = None if candidate is None else candidate.reauthenticate()
    artifact = _load_entry(context, request, entry)
    if reuse is not None:
        _verify_hit_only_artifact(reuse, artifact)
    if candidate is not None:
        _verify_candidate_matches_publication(
            candidate,
            artifact,
            candidate_sha256=candidate_sha256,
        )
    return AuthenticatedLfm2BodyReclosurePhase(
        request=request,
        artifact=artifact,
        entry=entry,
        cache_hit=cache_hit,
        producer_invoked=producer_invoked,
        publication_race_lost=publication_race_lost,
    )


def _verify_candidate_matches_publication(
    candidate: Lfm2BodyReclosureArtifact,
    published: Lfm2BodyReclosureArtifact,
    *,
    candidate_sha256: str | None = None,
) -> None:
    expected = (
        candidate.reauthenticate()
        if candidate_sha256 is None
        else _digest(candidate_sha256, "candidate artifact SHA-256")
    )
    if published.artifact_sha256 != expected:
        raise Lfm2BodyReclosurePhaseAuthenticationError(
            "Existing body-reclosure publication differs from the caller's "
            "authenticated artifact."
        )
    if candidate.reauthenticate() != expected:
        raise Lfm2BodyReclosurePhaseAuthenticationError(
            "Caller body-reclosure artifact changed while the existing "
            "publication was authenticated."
        )


def _raise_in_progress(
    context: Lfm2BodyReclosurePhaseCacheContext,
    request: Lfm2BodyReclosurePhaseRequest,
    error: FoldPhaseCacheConcurrencyError,
) -> NoReturn:
    key = request.phase_key
    _progress(
        context,
        "body_reclosure_phase_in_progress",
        phase_key_sha256=key.phase_key_sha256,
        status="in_progress",
        retryable=True,
        producer_invoked=False,
        ownership_stolen=False,
        owner_liveness_authenticated=False,
    )
    raise Lfm2BodyReclosurePhaseInProgressError(key.phase_key_sha256) from error


def lookup_lfm2_body_reclosure_phase(
    context: Lfm2BodyReclosurePhaseCacheContext,
    request: Lfm2BodyReclosurePhaseRequest,
) -> AuthenticatedLfm2BodyReclosurePhase | None:
    """Return an exact authenticated hit or ``None`` without producing.

    ``None`` is the explicit ordinary-miss representation.  Publication
    reservation contention is reported separately by the publication API as
    :class:`Lfm2BodyReclosurePhaseInProgressError`; lookup itself never takes a
    reservation, waits for a writer, or invokes model work.
    """

    reuse = _validate_context_and_request(context, request)
    key = request.phase_key
    _progress(
        context,
        "body_reclosure_phase_cache_lookup",
        phase_key_sha256=key.phase_key_sha256,
        exact_hit_only=reuse is not None,
    )
    entry = context.cache.lookup(key, events=context.events)
    if entry is not None:
        authenticated = _authenticated_result(
            context,
            request,
            entry,
            cache_hit=True,
            producer_invoked=False,
            publication_race_lost=False,
        )
        _progress(
            context,
            "body_reclosure_phase_cache_hit",
            phase_key_sha256=key.phase_key_sha256,
            completion_sha256=entry.completion_sha256,
            producer_invoked=False,
            exact_hit_only=reuse is not None,
        )
        return authenticated
    _progress(
        context,
        "body_reclosure_phase_cache_miss",
        phase_key_sha256=key.phase_key_sha256,
    )
    if reuse is not None:
        raise Lfm2BodyReclosurePhaseCacheError(
            "Requested body-reclosure phase is absent; exact hit-only reuse "
            "forbids building or repairing it."
        )
    return None


def _resume_completed_publication(
    context: Lfm2BodyReclosurePhaseCacheContext,
    request: Lfm2BodyReclosurePhaseRequest,
    *,
    candidate: Lfm2BodyReclosureArtifact | None,
    producer_invoked: bool,
) -> AuthenticatedLfm2BodyReclosurePhase:
    key = request.phase_key
    raced = context.cache.lookup(key, events=context.events)
    if raced is None:
        raise Lfm2BodyReclosurePhaseAuthenticationError(
            "Completed body-reclosure phase disappeared during lookup."
        )
    authenticated = _authenticated_result(
        context,
        request,
        raced,
        cache_hit=True,
        producer_invoked=producer_invoked,
        publication_race_lost=True,
        candidate=candidate,
    )
    _progress(
        context,
        "body_reclosure_phase_publish_race_resumed",
        phase_key_sha256=key.phase_key_sha256,
        completion_sha256=raced.completion_sha256,
        producer_invoked=producer_invoked,
        byte_identical_candidate=candidate is not None,
    )
    return authenticated


def _complete_reserved_publication(
    context: Lfm2BodyReclosurePhaseCacheContext,
    request: Lfm2BodyReclosurePhaseRequest,
    artifact: Lfm2BodyReclosureArtifact,
    transaction: FoldPhaseCacheTransaction,
    *,
    producer_invoked: bool,
) -> AuthenticatedLfm2BodyReclosurePhase:
    key = request.phase_key
    artifact_sha256 = artifact.reauthenticate()
    receipt = _write_bundle(transaction.payload_path, request, artifact)
    entry = transaction.complete(result=_completion_result(receipt))
    authenticated = _authenticated_result(
        context,
        request,
        entry,
        cache_hit=False,
        producer_invoked=producer_invoked,
        publication_race_lost=False,
        candidate=artifact,
    )
    if authenticated.artifact.artifact_sha256 != artifact_sha256:
        raise Lfm2BodyReclosurePhaseAuthenticationError(
            "Cold and reloaded body-reclosure artifacts differ."
        )
    _progress(
        context,
        "body_reclosure_phase_published",
        phase_key_sha256=key.phase_key_sha256,
        completion_sha256=entry.completion_sha256,
        artifact_sha256=authenticated.artifact.artifact_sha256,
        producer_invoked=producer_invoked,
        cold_reload_exact=True,
    )
    return authenticated


def publish_lfm2_body_reclosure_phase(
    context: Lfm2BodyReclosurePhaseCacheContext,
    request: Lfm2BodyReclosurePhaseRequest,
    artifact: Lfm2BodyReclosureArtifact,
) -> AuthenticatedLfm2BodyReclosurePhase:
    """Atomically publish one already-validated artifact and reload it.

    The caller owns all compilation and structural validation.  This function
    performs no producer work: it authenticates the supplied artifact, wins an
    immutable reservation or returns a byte-identical completed winner, and
    returns only the fully reloaded authenticated bundle.
    """

    _validate_context_and_request(context, request)
    if not isinstance(artifact, Lfm2BodyReclosureArtifact):
        raise TypeError("artifact must be Lfm2BodyReclosureArtifact.")
    artifact.reauthenticate()
    hit = lookup_lfm2_body_reclosure_phase(context, request)
    if hit is not None:
        _verify_candidate_matches_publication(artifact, hit.artifact)
        return hit

    key = request.phase_key
    _progress(
        context,
        "body_reclosure_phase_publish_started",
        phase_key_sha256=key.phase_key_sha256,
        caller_supplied_artifact=True,
        producer_invoked=False,
    )
    try:
        transaction = context.cache.begin(key, events=context.events)
    except FoldPhaseCacheAlreadyCompleteError:
        return _resume_completed_publication(
            context,
            request,
            candidate=artifact,
            producer_invoked=False,
        )
    except FoldPhaseCacheConcurrencyError as exc:
        _raise_in_progress(context, request, exc)
    with transaction:
        return _complete_reserved_publication(
            context,
            request,
            artifact,
            transaction,
            producer_invoked=False,
        )


def load_or_build_lfm2_body_reclosure_phase(
    context: Lfm2BodyReclosurePhaseCacheContext,
    request: Lfm2BodyReclosurePhaseRequest,
    *,
    producer: BodyReclosureProducer,
) -> AuthenticatedLfm2BodyReclosurePhase:
    """Compose lookup/publication internals with exactly-once producer ownership.

    Reservation precedes producer invocation, so the compatibility surface
    retains its historical exactly-once guarantee while sharing the split
    API's authentication and atomic-publication implementation.
    """

    _validate_context_and_request(context, request)
    if not callable(producer):
        raise TypeError("producer must be callable.")
    hit = lookup_lfm2_body_reclosure_phase(context, request)
    if hit is not None:
        return hit

    key = request.phase_key
    try:
        transaction = context.cache.begin(key, events=context.events)
    except FoldPhaseCacheAlreadyCompleteError:
        return _resume_completed_publication(
            context,
            request,
            candidate=None,
            producer_invoked=False,
        )
    except FoldPhaseCacheConcurrencyError as exc:
        _raise_in_progress(context, request, exc)
    with transaction:
        _progress(
            context,
            "body_reclosure_phase_producer_started",
            phase_key_sha256=key.phase_key_sha256,
            phase_transaction_reserved=True,
        )
        cold = producer()
        if not isinstance(cold, Lfm2BodyReclosureArtifact):
            raise Lfm2BodyReclosurePhaseCacheError(
                "Body-reclosure producer must return Lfm2BodyReclosureArtifact."
            )
        cold.reauthenticate()
        _progress(
            context,
            "body_reclosure_phase_producer_complete",
            phase_key_sha256=key.phase_key_sha256,
            artifact_sha256=cold.artifact_sha256,
        )
        return _complete_reserved_publication(
            context,
            request,
            cold,
            transaction,
            producer_invoked=True,
        )


__all__ = [
    "ACCEPTANCE_ONLY_POLICY_FIELDS",
    "AuthenticatedLfm2BodyReclosurePhase",
    "LFM2_BODY_RECLOSURE_ABI_SCHEMA",
    "LFM2_BODY_RECLOSURE_PHASE",
    "LFM2_BODY_RECLOSURE_RECEIPT_SCHEMA",
    "LFM2_BODY_RECLOSURE_REQUEST_SCHEMA",
    "LFM2_BODY_RECLOSURE_SAFETENSORS_SCHEMA",
    "Lfm2BodyReclosureArtifact",
    "Lfm2BodyReclosurePhaseAuthenticationError",
    "Lfm2BodyReclosurePhaseCacheContext",
    "Lfm2BodyReclosurePhaseCacheError",
    "Lfm2BodyReclosurePhaseInProgressError",
    "Lfm2BodyReclosurePhaseRequest",
    "Lfm2ExistingBodyReclosurePhaseReuseRequest",
    "build_lfm2_body_reclosure_phase_request",
    "describe_lfm2_body_reclosure_phase",
    "load_or_build_lfm2_body_reclosure_phase",
    "lookup_lfm2_body_reclosure_phase",
    "publish_lfm2_body_reclosure_phase",
]
