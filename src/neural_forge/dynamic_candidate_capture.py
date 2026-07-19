"""Immutable, family-neutral capture for an unreviewed fold candidate.

This module deliberately implements a quarantine lifecycle, not compilation
publication.  A captured candidate is a standalone artifact that can be
inspected and freshly reloaded, but its receipt always denies promotion.  A
separate reviewed rerun must reproduce the candidate before any production
publisher can consider it.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import CodeType, ModuleType
from typing import Any

from .dynamic_behavior_execution import observe_model_execution
from .dynamic_behavior_gate import identify_behavior_callable
from .dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from .dynamic_fold_artifact import (
    _artifact_tree as _standalone_artifact_tree,
    _assert_disjoint_reload,
)
from .dynamic_fold_executor import (
    _FoldOnlyTorchGuard,
    fingerprint_model_runtime_state,
    fingerprint_model_state,
)
from .dynamic_target_synthesis import serialized_inference_state_inventory


QUARANTINE_CAPTURE_SCHEMA = "neural-forge-dynamic-candidate-quarantine-v1"
QUARANTINE_REQUEST_SCHEMA = "neural-forge-dynamic-candidate-capture-request-v1"
QUARANTINE_AUDIT_SCHEMA = "neural-forge-dynamic-candidate-zero-operation-audit-v1"
QUARANTINE_INVENTORY_SCHEMA = (
    "neural-forge-dynamic-candidate-artifact-inventory-v1"
)
QUARANTINE_RELOAD_SCHEMA = (
    "neural-forge-dynamic-candidate-standalone-reload-v1"
)
QUARANTINE_VERIFICATION_SCHEMA = (
    "neural-forge-dynamic-candidate-load-verification-v1"
)

CANDIDATE_DIRECTORY_NAME = "candidate"
QUARANTINE_RECEIPT_FILENAME = "quarantine_receipt.json"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_JSON_BYTES = 64 * 1024 * 1024
_FORBIDDEN_ARTIFACT_NAME_PARTS = (
    "cache",
    "checkpoint",
    "coordinator",
    "donor",
    "optimizer",
    "parent",
    "pending",
    "resume",
    "sidecar",
    "teacher",
    "trainer",
    "training_args",
)
_REQUEST_DIGEST_FIELDS = (
    "plan_sha256",
    "parent_sha256",
    "cache_sha256",
    "bridge_sha256",
    "corpus_sha256",
    "compiler_sha256",
    "source_sha256",
    "saver_sha256",
    "loader_sha256",
    "evaluator_sha256",
)
_CALLABLE_ROLES = frozenset({"saver", "loader", "evaluator"})
_COMPILER_BINDING_SEMANTICS = (
    "authenticated-compiler-run-receipt-bound-to-source"
)
_OPTIONAL_REQUEST_DIGEST_FIELDS = (
    "expected_model_state_sha256",
    "expected_output_sha256",
    "pre_reviewed_candidate_fingerprint_sha256",
)
_ZERO_COUNTER_FIELDS = (
    "optimizer_constructions",
    "optimizer_steps",
    "backward_calls",
    "autograd_grad_calls",
    "training_steps",
    "pruning_operations",
    "quantization_operations",
    "parent_forward_calls",
    "parent_inference_calls",
)
_RECEIPT_BODY_FIELDS = frozenset(
    {
        "schema",
        "status",
        "lifecycle",
        "capture_request",
        "bindings",
        "callable_identities",
        "candidate_artifact",
        "model_state",
        "evaluations",
        "standalone_reload_receipt",
        "audit",
        "filesystem_identity",
        "complete_artifact_inventory",
        "standalone_candidate",
        "published_compilation",
        "accepted_compilation",
        "behavior_capability_proven",
        "promotable",
        "publication_authorized",
        "requires_reviewed_rerun",
        "pre_reviewed_candidate_fingerprint_supplied",
        "no_parent_sidecar",
        "no_cache_sidecar",
        "no_coordinator_sidecar",
        "no_resume_sidecar",
        "receipt_outside_candidate_artifact",
        "root_entry_allowlist",
    }
)
_RECEIPT_FIELDS = _RECEIPT_BODY_FIELDS | {"receipt_sha256"}


class DynamicCandidateCaptureError(RuntimeError):
    """Base error for immutable candidate quarantine."""


class DynamicCandidateCapturePublicationError(DynamicCandidateCaptureError):
    """A quarantine could not be atomically captured."""


class DynamicCandidateCaptureIntegrityError(DynamicCandidateCaptureError):
    """A quarantine, model, binding, or receipt failed authentication."""


@dataclass(frozen=True, slots=True)
class QuarantineZeroOperationAudit:
    """Caller-rooted audit counters for work completed before capture.

    The capture runtime independently observes its own training API and parent
    execution counters.  This object binds the already-completed fold phase
    without pretending the capture process can retrospectively observe it.
    """

    optimizer_constructions: int = 0
    optimizer_steps: int = 0
    backward_calls: int = 0
    autograd_grad_calls: int = 0
    training_steps: int = 0
    pruning_operations: int = 0
    quantization_operations: int = 0
    parent_forward_calls: int = 0
    parent_inference_calls: int = 0

    def __post_init__(self) -> None:
        for name in _ZERO_COUNTER_FIELDS:
            _exact_zero(getattr(self, name), name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": QUARANTINE_AUDIT_SCHEMA,
            **{name: getattr(self, name) for name in _ZERO_COUNTER_FIELDS},
            "training_used": False,
            "pruning_used": False,
            "quantization_used": False,
            "live_parent_calls_used": False,
            "clean": True,
        }


@dataclass(frozen=True, slots=True)
class QuarantineCaptureRequest:
    """Externally rooted identities for one family-neutral candidate capture."""

    plan_sha256: str
    parent_sha256: str
    cache_sha256: str
    bridge_sha256: str
    corpus_sha256: str
    compiler_sha256: str
    source_sha256: str
    saver_sha256: str
    loader_sha256: str
    evaluator_sha256: str
    expected_model_state_sha256: str | None = None
    expected_output_sha256: str | None = None
    pre_reviewed_candidate_fingerprint_sha256: str | None = None
    prior_audit: QuarantineZeroOperationAudit = field(
        default_factory=QuarantineZeroOperationAudit
    )

    def __post_init__(self) -> None:
        for name in _REQUEST_DIGEST_FIELDS:
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        for name in _OPTIONAL_REQUEST_DIGEST_FIELDS:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _digest(value, name))
        if type(self.prior_audit) is not QuarantineZeroOperationAudit:
            raise DynamicCandidateCaptureIntegrityError(
                "prior_audit must be an exact QuarantineZeroOperationAudit."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": QUARANTINE_REQUEST_SCHEMA,
            **{name: getattr(self, name) for name in _REQUEST_DIGEST_FIELDS},
            "compiler_binding_semantics": _COMPILER_BINDING_SEMANTICS,
            **{
                name: getattr(self, name)
                for name in _OPTIONAL_REQUEST_DIGEST_FIELDS
            },
            "prior_audit": self.prior_audit.as_dict(),
        }


@dataclass(frozen=True, slots=True, init=False)
class QuarantineEvaluation:
    """Canonical whole-output and evaluation evidence from one model run."""

    _output_json: bytes
    _evaluation_json: bytes
    output_sha256: str
    evaluation_sha256: str

    def __init__(self, *, output: Any, evaluation: Any) -> None:
        output_json = _canonical_value_bytes(output, "candidate output")
        evaluation_json = _canonical_value_bytes(
            evaluation,
            "candidate evaluation",
        )
        object.__setattr__(self, "_output_json", output_json)
        object.__setattr__(self, "_evaluation_json", evaluation_json)
        object.__setattr__(
            self,
            "output_sha256",
            hashlib.sha256(output_json).hexdigest(),
        )
        object.__setattr__(
            self,
            "evaluation_sha256",
            hashlib.sha256(evaluation_json).hexdigest(),
        )

    @property
    def output(self) -> Any:
        return json.loads(self._output_json.decode("ascii"))

    @property
    def evaluation(self) -> Any:
        return json.loads(self._evaluation_json.decode("ascii"))

    def hashes(self) -> dict[str, str]:
        return {
            "output_sha256": self.output_sha256,
            "evaluation_sha256": self.evaluation_sha256,
        }


@dataclass(frozen=True, slots=True, init=False)
class QuarantineCaptureReceipt:
    """Canonical self-hashed receipt that permanently denies promotion."""

    _body_json: bytes
    receipt_sha256: str

    def __init__(
        self,
        body: Mapping[str, Any],
        *,
        receipt_sha256: str | None = None,
    ) -> None:
        normalized = _json_mapping(body, "quarantine receipt body")
        if set(normalized) != _RECEIPT_BODY_FIELDS:
            raise DynamicCandidateCaptureIntegrityError(
                "Quarantine receipt does not contain exactly the v1 body fields."
            )
        _validate_receipt_body(normalized)
        body_json = canonical_json_bytes(normalized)
        actual = hashlib.sha256(body_json).hexdigest()
        if receipt_sha256 is not None and (
            _digest(receipt_sha256, "receipt_sha256") != actual
        ):
            raise DynamicCandidateCaptureIntegrityError(
                "Quarantine receipt self-hash does not verify."
            )
        object.__setattr__(self, "_body_json", body_json)
        object.__setattr__(self, "receipt_sha256", actual)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> QuarantineCaptureReceipt:
        payload = _json_mapping(value, "quarantine receipt")
        if set(payload) != _RECEIPT_FIELDS:
            raise DynamicCandidateCaptureIntegrityError(
                "Quarantine receipt does not contain exactly the v1 fields."
            )
        claimed = payload.pop("receipt_sha256")
        return cls(payload, receipt_sha256=claimed)

    def as_dict(self) -> dict[str, Any]:
        body = json.loads(self._body_json.decode("ascii"))
        body["receipt_sha256"] = self.receipt_sha256
        return body


@dataclass(frozen=True, slots=True)
class QuarantinedDynamicFoldCandidate:
    """One authenticated candidate that remains permanently non-promotable."""

    path: Path
    artifact_path: Path
    model: Any
    receipt: QuarantineCaptureReceipt
    verification_receipt: Mapping[str, Any]
    promotable: bool = field(default=False, init=False)
    publication_authorized: bool = field(default=False, init=False)
    requires_reviewed_rerun: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if type(self.receipt) is not QuarantineCaptureReceipt:
            raise DynamicCandidateCaptureIntegrityError(
                "receipt must be an exact QuarantineCaptureReceipt."
            )
        verification = _json_mapping(
            self.verification_receipt,
            "load verification receipt",
        )
        _validate_verification_receipt(
            verification,
            expected_capture_sha256=self.receipt.receipt_sha256,
        )
        object.__setattr__(self, "verification_receipt", verification)


def capture_quarantined_candidate(
    destination: str | os.PathLike[str],
    *,
    candidate: Any,
    request: QuarantineCaptureRequest,
    saver: Callable[[Any, Path], None],
    loader: Callable[[Path], Any],
    evaluator: Callable[[Any], QuarantineEvaluation],
    forbidden_parent_models: Sequence[Any] = (),
) -> QuarantinedDynamicFoldCandidate:
    """Capture, atomically quarantine, and reload one unreviewed candidate.

    The destination is immutable and is never a compiled-child publication.
    The first fresh reload occurs from the staged standalone bytes.  After the
    no-clobber directory rename, this function calls
    :func:`load_quarantined_candidate`, which performs another fresh reload
    from the retained final destination before returning.
    """

    _require_request(request)
    _require_callbacks(saver=saver, loader=loader, evaluator=evaluator)
    callable_identities = _authenticate_capture_callables(
        request,
        saver=saver,
        loader=loader,
        evaluator=evaluator,
    )
    parents = _parent_models(forbidden_parent_models, candidate=candidate)
    final_path = _new_destination(destination)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path = _new_destination(final_path)
    parent_identity = _directory_identity(final_path.parent, "destination parent")
    staging: Path | None = None
    loaded: Any | None = None
    try:
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{final_path.name}.",
                suffix=".quarantine-staging",
                dir=final_path.parent,
            )
        )
        _assert_ordinary_directory(staging, "quarantine staging directory")
        if _directory_identity(staging.parent, "staging parent") != parent_identity:
            raise DynamicCandidateCapturePublicationError(
                "Quarantine staging parent changed before capture."
            )
        candidate_path = staging / CANDIDATE_DIRECTORY_NAME
        candidate_path.mkdir()

        live_state_sha = _model_state_sha(candidate)
        live_inventory = _model_inventory(candidate)
        _assert_dense_unpruned_model(candidate)
        _match_optional_digest(
            request.expected_model_state_sha256,
            live_state_sha,
            "expected model state",
        )
        live_runtime_before = fingerprint_model_runtime_state(candidate)
        parent_states = _protected_model_snapshots(parents)

        frozen = _freeze_models((candidate, *parents))
        live_runtime_guarded = fingerprint_model_runtime_state(candidate)
        guard = _FoldOnlyTorchGuard()
        parent_counts: list[Mapping[str, int]] = []
        try:
            with guard:
                with ExitStack() as stack:
                    for parent in parents:
                        parent_counts.append(
                            stack.enter_context(observe_model_execution(parent))
                        )
                    in_memory_evidence, in_memory_execution = _evaluate_model(
                        candidate,
                        evaluator=evaluator,
                        label="in-memory candidate",
                    )
                    try:
                        saver(candidate, candidate_path)
                    except DynamicCandidateCaptureError:
                        raise
                    except BaseException as error:
                        raise DynamicCandidateCapturePublicationError(
                            "Candidate saver failed."
                        ) from error
                    if (
                        _model_state_sha(candidate) != live_state_sha
                        or fingerprint_model_runtime_state(candidate)
                        != live_runtime_guarded
                    ):
                        raise DynamicCandidateCapturePublicationError(
                            "Candidate saver or evaluator mutated the live candidate."
                        )

                    inventory_before = _candidate_inventory(candidate_path)
                    identities_before = _tree_identity_snapshot(candidate_path)
                    challenge_exception = _challenge_loader(
                        loader,
                        staging=staging,
                    )
                    try:
                        loaded = loader(candidate_path)
                    except DynamicCandidateCaptureError:
                        raise
                    except BaseException as error:
                        raise DynamicCandidateCapturePublicationError(
                            "Fresh candidate loader failed."
                        ) from error
                    if loaded is candidate:
                        raise DynamicCandidateCapturePublicationError(
                            "Fresh loader returned the in-memory candidate object."
                        )
                    try:
                        _assert_disjoint_reload(
                            candidate,
                            loaded,
                            original_label="quarantined in-memory candidate",
                        )
                    except BaseException as error:
                        raise DynamicCandidateCapturePublicationError(
                            "Fresh loader shares modules or tensor storage with "
                            "the in-memory candidate."
                        ) from error
                    fresh_state_sha = _model_state_sha(loaded)
                    if fresh_state_sha != live_state_sha:
                        raise DynamicCandidateCapturePublicationError(
                            "Freshly loaded candidate state differs from memory."
                        )
                    fresh_inventory = _model_inventory(loaded)
                    _assert_inventory_compatible(
                        live_inventory,
                        fresh_inventory,
                    )
                    _assert_dense_unpruned_model(loaded)
                    loaded_runtime_before = fingerprint_model_runtime_state(loaded)
                    loaded_frozen = _freeze_models((loaded,))
                    try:
                        reload_evidence, reload_execution = _evaluate_model(
                            loaded,
                            evaluator=evaluator,
                            label="freshly reloaded candidate",
                        )
                    finally:
                        _restore_frozen_models(loaded_frozen)
                    if (
                        _model_state_sha(loaded) != fresh_state_sha
                        or fingerprint_model_runtime_state(loaded)
                        != loaded_runtime_before
                    ):
                        raise DynamicCandidateCapturePublicationError(
                            "Fresh-reload evaluation mutated the candidate."
                        )
                    _require_reproduced_evaluation(
                        in_memory_evidence,
                        reload_evidence,
                    )
                    _match_optional_digest(
                        request.expected_output_sha256,
                        in_memory_evidence.output_sha256,
                        "expected candidate output",
                    )
                    if (
                        _candidate_inventory(candidate_path) != inventory_before
                        or _tree_identity_snapshot(candidate_path)
                        != identities_before
                    ):
                        raise DynamicCandidateCapturePublicationError(
                            "Saver, loader, or evaluator replaced candidate bytes."
                        )
        finally:
            _restore_frozen_models(frozen)

        observed_audit = _observed_audit(
            guard=guard,
            parent_counts=parent_counts,
        )
        if (
            _model_state_sha(candidate) != live_state_sha
            or fingerprint_model_runtime_state(candidate) != live_runtime_before
        ):
            raise DynamicCandidateCapturePublicationError(
                "Quarantine capture did not restore the live candidate runtime."
            )
        _assert_protected_models_unchanged(parents, parent_states)
        artifact_inventory = _candidate_inventory(candidate_path)
        artifact_identities = _tree_identity_snapshot(candidate_path)

        receipt_path = staging / QUARANTINE_RECEIPT_FILENAME
        receipt_descriptor = _open_exclusive(receipt_path)
        try:
            receipt_stat = os.fstat(receipt_descriptor)
            root_identity = _path_identity(staging, "quarantine root")
            candidate_identity = _path_identity(
                candidate_path,
                "candidate artifact directory",
            )
            receipt_identity = _stat_identity(
                receipt_stat,
                include_size=False,
            )
            reload_receipt = _sealed_reload_receipt(
                loader_sha256=request.loader_sha256,
                artifact_inventory_sha256=artifact_inventory[
                    "inventory_sha256"
                ],
                in_memory_state_sha256=live_state_sha,
                fresh_reload_state_sha256=fresh_state_sha,
                in_memory_evidence=in_memory_evidence,
                reload_evidence=reload_evidence,
                in_memory_execution=in_memory_execution,
                reload_execution=reload_execution,
                challenge_exception=challenge_exception,
            )
            receipt_body = {
                "schema": QUARANTINE_CAPTURE_SCHEMA,
                "status": "quarantined-unreviewed-candidate",
                "lifecycle": "immutable-quarantine",
                "capture_request": request.as_dict(),
                "bindings": {
                    "plan_sha256": request.plan_sha256,
                    "parent_sha256": request.parent_sha256,
                    "cache_sha256": request.cache_sha256,
                    "bridge_sha256": request.bridge_sha256,
                    "corpus_sha256": request.corpus_sha256,
                    "compiler_sha256": request.compiler_sha256,
                    "compiler_binding_semantics": (
                        _COMPILER_BINDING_SEMANTICS
                    ),
                    "source_sha256": request.source_sha256,
                    "model_state_sha256": live_state_sha,
                    "output_sha256": in_memory_evidence.output_sha256,
                },
                "callable_identities": callable_identities,
                "candidate_artifact": {
                    "path": CANDIDATE_DIRECTORY_NAME,
                    "inventory": artifact_inventory,
                    "inventory_sha256": artifact_inventory["inventory_sha256"],
                    "standalone": True,
                    "strict_sidecar_scan_passed": True,
                    "foreign_entries_present": False,
                },
                "model_state": {
                    "in_memory_sha256": live_state_sha,
                    "fresh_reload_sha256": fresh_state_sha,
                    "byte_identical_after_reload": True,
                    "in_memory_inventory": live_inventory,
                    "fresh_reload_inventory": fresh_inventory,
                    "dense_materialized": True,
                    "quantized": False,
                    "pruned": False,
                },
                "evaluations": {
                    "in_memory_output_sha256": (
                        in_memory_evidence.output_sha256
                    ),
                    "fresh_reload_output_sha256": (
                        reload_evidence.output_sha256
                    ),
                    "in_memory_evaluation_sha256": (
                        in_memory_evidence.evaluation_sha256
                    ),
                    "fresh_reload_evaluation_sha256": (
                        reload_evidence.evaluation_sha256
                    ),
                    "output_reproduced_after_reload": True,
                    "evaluation_reproduced_after_reload": True,
                    "corpus_sha256": request.corpus_sha256,
                    "evaluator_sha256": request.evaluator_sha256,
                },
                "standalone_reload_receipt": reload_receipt,
                "audit": {
                    "prior_fold": request.prior_audit.as_dict(),
                    "capture_observed": observed_audit,
                    "combined_clean": True,
                    "historical_fold_scope_is_caller_rooted": True,
                    "capture_scope_is_runtime_observed": True,
                },
                "filesystem_identity": {
                    "destination_path_sha256": canonical_sha256(
                        os.fspath(final_path)
                    ),
                    "destination_parent": parent_identity,
                    "quarantine_root": root_identity,
                    "candidate_directory": candidate_identity,
                    "receipt_file": receipt_identity,
                    "candidate_tree_sha256": canonical_sha256(
                        artifact_identities
                    ),
                },
                "complete_artifact_inventory": True,
                "standalone_candidate": True,
                "published_compilation": False,
                "accepted_compilation": False,
                "behavior_capability_proven": False,
                "promotable": False,
                "publication_authorized": False,
                "requires_reviewed_rerun": True,
                "pre_reviewed_candidate_fingerprint_supplied": (
                    request.pre_reviewed_candidate_fingerprint_sha256 is not None
                ),
                "no_parent_sidecar": True,
                "no_cache_sidecar": True,
                "no_coordinator_sidecar": True,
                "no_resume_sidecar": True,
                "receipt_outside_candidate_artifact": True,
                "root_entry_allowlist": [
                    CANDIDATE_DIRECTORY_NAME,
                    QUARANTINE_RECEIPT_FILENAME,
                ],
            }
            receipt = QuarantineCaptureReceipt(receipt_body)
            _write_descriptor(
                receipt_descriptor,
                canonical_json_bytes(receipt.as_dict()) + b"\n",
            )
        finally:
            os.close(receipt_descriptor)

        _fsync_tree(staging)
        _verify_quarantine_metadata(
            staging,
            request=request,
            expected_receipt_sha256=receipt.receipt_sha256,
            expected_destination_path=final_path,
            allow_staging_root=True,
        )
        if _directory_identity(final_path.parent, "destination parent") != (
            parent_identity
        ):
            raise DynamicCandidateCapturePublicationError(
                "Destination parent was replaced before atomic publication."
            )
        _rename_noreplace(staging, final_path)
        staging = None
        _fsync_directory(final_path.parent)
    except DynamicCandidateCaptureError:
        raise
    except BaseException as error:
        raise DynamicCandidateCapturePublicationError(
            "Quarantine capture failed before atomic completion."
        ) from error
    finally:
        if staging is not None and _lexists(staging):
            shutil.rmtree(staging)

    return load_quarantined_candidate(
        final_path,
        request=request,
        expected_receipt_sha256=receipt.receipt_sha256,
        loader=loader,
        evaluator=evaluator,
        forbidden_parent_models=parents,
    )


def load_quarantined_candidate(
    source: str | os.PathLike[str],
    *,
    request: QuarantineCaptureRequest,
    expected_receipt_sha256: str,
    loader: Callable[[Path], Any],
    evaluator: Callable[[Any], QuarantineEvaluation],
    forbidden_parent_models: Sequence[Any] = (),
) -> QuarantinedDynamicFoldCandidate:
    """Authenticate every byte, then freshly reload a non-promotable candidate."""

    _require_request(request)
    _require_callbacks(saver=None, loader=loader, evaluator=evaluator)
    callable_identities = _authenticate_load_callables(
        request,
        loader=loader,
        evaluator=evaluator,
    )
    expected_receipt_sha256 = _digest(
        expected_receipt_sha256,
        "expected_receipt_sha256",
    )
    parents = _parent_models(forbidden_parent_models, candidate=None)
    root = _existing_destination(source)
    receipt, artifact_path = _verify_quarantine_metadata(
        root,
        request=request,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_destination_path=root,
        allow_staging_root=False,
    )
    payload = receipt.as_dict()
    if any(
        payload["callable_identities"][role] != callable_identities[role]
        for role in ("loader", "evaluator")
    ):
        raise DynamicCandidateCaptureIntegrityError(
            "Loader or evaluator implementation/resources differ from capture."
        )
    artifact_before = _candidate_inventory(artifact_path)
    identities_before = _tree_identity_snapshot(artifact_path)
    root_identity_before = _path_identity(root, "quarantine root")
    receipt_identity_before = _path_identity(
        root / QUARANTINE_RECEIPT_FILENAME,
        "quarantine receipt",
    )
    parent_states = _protected_model_snapshots(parents)

    guard = _FoldOnlyTorchGuard()
    parent_counts: list[Mapping[str, int]] = []
    model: Any | None = None
    challenge_exception = ""
    try:
        with guard:
            with ExitStack() as stack:
                for parent in parents:
                    parent_counts.append(
                        stack.enter_context(observe_model_execution(parent))
                    )
                challenge_exception = _challenge_loader(loader, staging=root)
                try:
                    model = loader(artifact_path)
                except DynamicCandidateCaptureError:
                    raise
                except BaseException as error:
                    raise DynamicCandidateCaptureIntegrityError(
                        "Fresh quarantine loader failed."
                    ) from error
                state_sha = _model_state_sha(model)
                expected_state = payload["bindings"]["model_state_sha256"]
                if state_sha != expected_state:
                    raise DynamicCandidateCaptureIntegrityError(
                        "Freshly loaded quarantine state differs from its binding."
                    )
                model_inventory = _model_inventory(model)
                _assert_inventory_compatible(
                    payload["model_state"]["fresh_reload_inventory"],
                    model_inventory,
                )
                _assert_dense_unpruned_model(model)
                runtime_before = fingerprint_model_runtime_state(model)
                frozen = _freeze_models((model, *parents))
                try:
                    evidence, execution = _evaluate_model(
                        model,
                        evaluator=evaluator,
                        label="verified quarantine reload",
                    )
                finally:
                    _restore_frozen_models(frozen)
                if (
                    _model_state_sha(model) != state_sha
                    or fingerprint_model_runtime_state(model) != runtime_before
                ):
                    raise DynamicCandidateCaptureIntegrityError(
                        "Quarantine verification evaluator mutated the model."
                    )
    finally:
        pass

    observed_audit = _observed_audit(
        guard=guard,
        parent_counts=parent_counts,
    )
    _assert_protected_models_unchanged(parents, parent_states)
    evaluations = payload["evaluations"]
    if (
        evidence.output_sha256
        != evaluations["fresh_reload_output_sha256"]
        or evidence.evaluation_sha256
        != evaluations["fresh_reload_evaluation_sha256"]
    ):
        raise DynamicCandidateCaptureIntegrityError(
            "Fresh quarantine evaluation differs from the captured reload."
        )
    if challenge_exception != payload["standalone_reload_receipt"][
        "loader_path_challenge_exception_type"
    ]:
        raise DynamicCandidateCaptureIntegrityError(
            "Loader path-challenge behavior differs from capture."
        )
    if (
        _candidate_inventory(artifact_path) != artifact_before
        or _tree_identity_snapshot(artifact_path) != identities_before
        or _path_identity(root, "quarantine root") != root_identity_before
        or _path_identity(
            root / QUARANTINE_RECEIPT_FILENAME,
            "quarantine receipt",
        )
        != receipt_identity_before
    ):
        raise DynamicCandidateCaptureIntegrityError(
            "Fresh loader or evaluator replaced quarantine paths."
        )

    verification = _sealed_verification_receipt(
        capture_receipt_sha256=receipt.receipt_sha256,
        destination_path_sha256=canonical_sha256(os.fspath(root)),
        state_sha256=state_sha,
        output_sha256=evidence.output_sha256,
        evaluation_sha256=evidence.evaluation_sha256,
        artifact_inventory_sha256=artifact_before["inventory_sha256"],
        loader_sha256=request.loader_sha256,
        evaluator_sha256=request.evaluator_sha256,
        execution=execution,
        observed_audit=observed_audit,
        challenge_exception=challenge_exception,
    )
    return QuarantinedDynamicFoldCandidate(
        path=root,
        artifact_path=artifact_path,
        model=model,
        receipt=receipt,
        verification_receipt=verification,
    )


def fingerprint_quarantine_callable(
    value: Callable[..., Any],
    *,
    role: str,
) -> str:
    """Return the exact implementation/resource binding required by a request."""

    return _callable_identity(value, role=role)["binding_sha256"]


def _authenticate_capture_callables(
    request: QuarantineCaptureRequest,
    *,
    saver: Callable[[Any, Path], None],
    loader: Callable[[Path], Any],
    evaluator: Callable[[Any], QuarantineEvaluation],
) -> dict[str, Any]:
    identities = {
        "saver": _callable_identity(saver, role="saver"),
        "loader": _callable_identity(loader, role="loader"),
        "evaluator": _callable_identity(evaluator, role="evaluator"),
    }
    for role, field_name in (
        ("saver", "saver_sha256"),
        ("loader", "loader_sha256"),
        ("evaluator", "evaluator_sha256"),
    ):
        if identities[role]["binding_sha256"] != getattr(request, field_name):
            raise DynamicCandidateCaptureIntegrityError(
                f"Captured {role} differs from its request implementation or "
                "reachable resources."
            )
    return identities


def _authenticate_load_callables(
    request: QuarantineCaptureRequest,
    *,
    loader: Callable[[Path], Any],
    evaluator: Callable[[Any], QuarantineEvaluation],
) -> dict[str, Any]:
    identities = {
        "loader": _callable_identity(loader, role="loader"),
        "evaluator": _callable_identity(evaluator, role="evaluator"),
    }
    if identities["loader"]["binding_sha256"] != request.loader_sha256:
        raise DynamicCandidateCaptureIntegrityError(
            "Loader differs from the captured implementation or resources."
        )
    if identities["evaluator"]["binding_sha256"] != request.evaluator_sha256:
        raise DynamicCandidateCaptureIntegrityError(
            "Evaluator differs from the captured implementation or resources."
        )
    return identities


def _callable_identity(value: Callable[..., Any], *, role: str) -> dict[str, Any]:
    first = _callable_identity_once(value, role=role)
    second = _callable_identity_once(value, role=role)
    if first == second:
        return second
    third = _callable_identity_once(value, role=role)
    if second != third:
        raise DynamicCandidateCaptureIntegrityError(
            f"Quarantine {role} identity is unstable across repeated observation."
        )
    return third


def _callable_identity_once(
    value: Callable[..., Any],
    *,
    role: str,
) -> dict[str, Any]:
    if role not in _CALLABLE_ROLES:
        raise DynamicCandidateCaptureIntegrityError(
            "Quarantine callable role is invalid."
        )
    if not callable(value):
        raise DynamicCandidateCaptureIntegrityError(
            f"Quarantine {role} must be callable."
        )
    try:
        base_identity = identify_behavior_callable(value)
        reachable = _reachable_callable_manifest(
            value,
            name=f"quarantine {role}",
            active=frozenset(),
            depth=0,
        )
    except DynamicCandidateCaptureError:
        raise
    except BaseException as error:
        raise DynamicCandidateCaptureIntegrityError(
            f"Could not authenticate quarantine {role} implementation."
        ) from error
    body = {
        "schema": "neural-forge-dynamic-candidate-callable-binding-v1",
        "role": role,
        "implementation_sha256": _digest(
            base_identity.code_fingerprint,
            f"{role} implementation SHA",
        ),
        "resource_sha256": _digest(
            base_identity.resource_fingerprint,
            f"{role} resource SHA",
        ),
        "resource_identity_explicitly_declared": bool(
            _declared_callable_resource(value)
        ),
        "reachable_manifest_sha256": canonical_sha256(reachable),
        "reachable_manifest": reachable,
    }
    return {**body, "binding_sha256": canonical_sha256(body)}


def _declared_callable_resource(value: Any) -> bool:
    owners = (value, getattr(value, "__self__", None))
    for owner in owners:
        if owner is None:
            continue
        for name in ("resource_fingerprint", "behavior_resource_fingerprint"):
            candidate = getattr(owner, name, None)
            if isinstance(candidate, str) and _SHA256.fullmatch(candidate):
                return True
    return False


def _stable_code_sha256(code: CodeType) -> str:
    return canonical_sha256(
        {
            "bytecode_sha256": hashlib.sha256(code.co_code).hexdigest(),
            "constants": [
                _stable_code_constant(value) for value in code.co_consts
            ],
            "names": list(code.co_names),
            "variable_names": list(code.co_varnames),
            "free_variables": list(code.co_freevars),
            "cell_variables": list(code.co_cellvars),
            "positional_argument_count": int(code.co_argcount),
            "positional_only_argument_count": int(code.co_posonlyargcount),
            "keyword_only_argument_count": int(code.co_kwonlyargcount),
            "flags": int(code.co_flags),
        }
    )


def _stable_code_constant(value: Any) -> Any:
    if isinstance(value, CodeType):
        return {
            "kind": "code",
            "sha256": _stable_code_sha256(value),
        }
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return (
            value
            if math.isfinite(value)
            else {"kind": "nonfinite-float", "value": str(value)}
        )
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "size_bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_stable_code_constant(item) for item in value],
        }
    return {
        "kind": "constant",
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "text_sha256": hashlib.sha256(str(value).encode("utf-8")).hexdigest(),
    }


def _reachable_callable_manifest(
    value: Any,
    *,
    name: str,
    active: frozenset[int],
    depth: int,
) -> dict[str, Any]:
    function = getattr(value, "__func__", value)
    owner: Any | None = getattr(value, "__self__", None)
    if not callable(function) or not isinstance(
        getattr(function, "__code__", None),
        CodeType,
    ):
        owner = value
        function = getattr(type(value), "__call__", None)
    code = getattr(function, "__code__", None)
    if not isinstance(code, CodeType):
        return {
            "kind": "native-callable",
            "module": getattr(value, "__module__", type(value).__module__),
            "qualname": getattr(
                value,
                "__qualname__",
                type(value).__qualname__,
            ),
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
        }
    identity = id(function)
    module_name = str(getattr(function, "__module__", ""))
    qualname = str(getattr(function, "__qualname__", ""))
    module = sys.modules.get(module_name)
    source_path = getattr(module, "__file__", None)
    base = {
        "kind": "python-callable",
        "module": module_name,
        "qualname": qualname,
        "code_sha256": _stable_code_sha256(code),
        "source_sha256": (
            None
            if not isinstance(source_path, str)
            or not Path(source_path).is_file()
            else _plain_file_sha256(Path(source_path))
        ),
    }
    if identity in active:
        return {**base, "recursive_reference": True}
    if depth >= 4:
        return {**base, "depth_limited": True}
    nested_active = active | {identity}
    closure = getattr(function, "__closure__", None) or ()
    freevars = tuple(code.co_freevars)
    if len(closure) != len(freevars):
        raise DynamicCandidateCaptureIntegrityError(
            f"{name} closure cannot be authenticated."
        )
    namespace = getattr(function, "__globals__", {})
    referenced_names = tuple(sorted(set(str(item) for item in code.co_names)))
    return {
        **base,
        "defaults": _reachable_resource_manifest(
            getattr(function, "__defaults__", None),
            name=f"{name}.__defaults__",
            active=nested_active,
            depth=depth + 1,
            referenced_names=frozenset(referenced_names),
        ),
        "keyword_defaults": _reachable_resource_manifest(
            getattr(function, "__kwdefaults__", None),
            name=f"{name}.__kwdefaults__",
            active=nested_active,
            depth=depth + 1,
            referenced_names=frozenset(referenced_names),
        ),
        "closure": {
            freevar: _reachable_resource_manifest(
                cell.cell_contents,
                name=f"{name}.__closure__.{freevar}",
                active=nested_active,
                depth=depth + 1,
                referenced_names=frozenset(referenced_names),
            )
            for freevar, cell in zip(freevars, closure, strict=True)
        },
        "referenced_globals": {
            global_name: _reachable_resource_manifest(
                namespace[global_name],
                name=f"{name}.__globals__.{global_name}",
                active=nested_active,
                depth=depth + 1,
                referenced_names=frozenset(referenced_names),
            )
            for global_name in referenced_names
            if global_name in namespace
        },
        "bound_owner": (
            None
            if owner is None or isinstance(owner, type)
            else _reachable_resource_manifest(
                owner,
                name=f"{name}.__self__",
                active=nested_active,
                depth=depth + 1,
                referenced_names=frozenset(referenced_names),
            )
        ),
    }


def _reachable_resource_manifest(
    value: Any,
    *,
    name: str,
    active: frozenset[int],
    depth: int,
    referenced_names: frozenset[str],
) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"kind": "nonfinite-float", "value": str(value)}
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "size_bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, Enum):
        return {
            "kind": "enum",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "name": value.name,
        }
    if isinstance(value, os.PathLike):
        return {"kind": "path", "value": os.fspath(value)}
    if depth >= 5:
        return {
            "kind": "depth-limited",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
        }
    identity = id(value)
    if identity in active:
        return {
            "kind": "recursive-resource",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
        }
    nested_active = active | {identity}
    if isinstance(value, ModuleType):
        source = getattr(value, "__file__", None)
        attributes = {}
        if depth < 3:
            for attribute in sorted(referenced_names):
                if hasattr(value, attribute):
                    attributes[attribute] = _reachable_resource_manifest(
                        getattr(value, attribute),
                        name=f"{name}.{attribute}",
                        active=nested_active,
                        depth=depth + 1,
                        referenced_names=referenced_names,
                    )
        return {
            "kind": "module",
            "name": getattr(value, "__name__", None),
            "source_sha256": (
                None
                if not isinstance(source, str) or not Path(source).is_file()
                else _plain_file_sha256(Path(source))
            ),
            "referenced_attributes": attributes,
        }
    if isinstance(value, type):
        module = sys.modules.get(getattr(value, "__module__", ""))
        source = getattr(module, "__file__", None)
        methods = {}
        if depth < 3:
            for member_name, raw_member in sorted(vars(value).items()):
                member = (
                    raw_member.__func__
                    if isinstance(raw_member, (staticmethod, classmethod))
                    else raw_member
                )
                if isinstance(getattr(member, "__code__", None), CodeType):
                    methods[member_name] = _reachable_callable_manifest(
                        member,
                        name=f"{name}.{member_name}",
                        active=nested_active,
                        depth=depth + 1,
                    )
        return {
            "kind": "class",
            "module": getattr(value, "__module__", None),
            "qualname": getattr(value, "__qualname__", None),
            "source_sha256": (
                None
                if not isinstance(source, str) or not Path(source).is_file()
                else _plain_file_sha256(Path(source))
            ),
            "python_methods": methods,
        }
    if callable(value):
        return _reachable_callable_manifest(
            value,
            name=name,
            active=active,
            depth=depth + 1,
        )
    try:
        import torch
    except ImportError:  # pragma: no cover - dependency boundary
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        return {
            "kind": "tensor",
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(
                tensor.view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }
    if isinstance(value, Mapping):
        records = []
        for key, item in sorted(value.items(), key=lambda row: str(row[0])):
            records.append(
                {
                    "key": _reachable_resource_manifest(
                        key,
                        name=f"{name}.key",
                        active=nested_active,
                        depth=depth + 1,
                        referenced_names=referenced_names,
                    ),
                    "value": _reachable_resource_manifest(
                        item,
                        name=f"{name}[{key!s}]",
                        active=nested_active,
                        depth=depth + 1,
                        referenced_names=referenced_names,
                    ),
                }
            )
        return {
            "kind": "mapping",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "items": records,
        }
    if isinstance(value, (list, tuple)):
        return {
            "kind": "sequence",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "items": [
                _reachable_resource_manifest(
                    item,
                    name=f"{name}[{index}]",
                    active=nested_active,
                    depth=depth + 1,
                    referenced_names=referenced_names,
                )
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, (set, frozenset)):
        children = [
            _reachable_resource_manifest(
                item,
                name=f"{name}[]",
                active=nested_active,
                depth=depth + 1,
                referenced_names=referenced_names,
            )
            for item in value
        ]
        return {
            "kind": "set",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "items": sorted(children, key=canonical_json_bytes),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "kind": "dataclass",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                descriptor.name: _reachable_resource_manifest(
                    getattr(value, descriptor.name),
                    name=f"{name}.{descriptor.name}",
                    active=nested_active,
                    depth=depth + 1,
                    referenced_names=referenced_names,
                )
                for descriptor in fields(value)
            },
        }
    for attribute in (
        "resource_fingerprint",
        "fingerprint",
        "source_sha256",
        "cache_key",
    ):
        declared = getattr(value, attribute, None)
        if isinstance(declared, str) and _SHA256.fullmatch(declared):
            return {
                "kind": "identified-object",
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "identity_field": attribute,
                "identity": declared,
            }
    try:
        object_state = vars(value)
    except TypeError:
        object_state = None
    if isinstance(object_state, Mapping):
        return {
            "kind": "object",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "state": {
                str(key): _reachable_resource_manifest(
                    item,
                    name=f"{name}.{key}",
                    active=nested_active,
                    depth=depth + 1,
                    referenced_names=referenced_names,
                )
                for key, item in sorted(
                    object_state.items(),
                    key=lambda row: str(row[0]),
                )
            },
        }
    return {
        "kind": "opaque-object",
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
    }


def _plain_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_request(value: Any) -> None:
    if type(value) is not QuarantineCaptureRequest:
        raise DynamicCandidateCaptureIntegrityError(
            "request must be an exact QuarantineCaptureRequest."
        )


def _require_callbacks(
    *,
    saver: Callable[[Any, Path], None] | None,
    loader: Callable[[Path], Any],
    evaluator: Callable[[Any], QuarantineEvaluation],
) -> None:
    if saver is not None and not callable(saver):
        raise DynamicCandidateCaptureIntegrityError("saver must be callable.")
    if not callable(loader) or not callable(evaluator):
        raise DynamicCandidateCaptureIntegrityError(
            "loader and evaluator must be callable."
        )


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DynamicCandidateCaptureIntegrityError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _exact_zero(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != 0:
        raise DynamicCandidateCaptureIntegrityError(f"{name} must be exactly zero.")
    return value


def _canonical_value_bytes(value: Any, name: str) -> bytes:
    try:
        encoded = canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise DynamicCandidateCaptureIntegrityError(
            f"{name} must be finite canonical JSON."
        ) from error
    return encoded


def _json_mapping(value: Any, name: str) -> dict[str, Any]:
    try:
        encoded = canonical_json_bytes(value)
        normalized = json.loads(encoded.decode("ascii"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise DynamicCandidateCaptureIntegrityError(
            f"{name} must be finite canonical JSON."
        ) from error
    if not isinstance(normalized, dict):
        raise DynamicCandidateCaptureIntegrityError(f"{name} must be an object.")
    return normalized


def _match_optional_digest(expected: str | None, actual: str, name: str) -> None:
    if expected is not None and expected != actual:
        raise DynamicCandidateCaptureIntegrityError(
            f"{name} differs from the captured candidate."
        )


def _parent_models(
    values: Sequence[Any],
    *,
    candidate: Any | None,
) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise DynamicCandidateCaptureIntegrityError(
            "forbidden_parent_models must be a sequence."
        )
    result: list[Any] = []
    seen: set[int] = set()
    for value in values:
        if candidate is not None and value is candidate:
            raise DynamicCandidateCaptureIntegrityError(
                "The candidate cannot also be a forbidden parent."
            )
        if id(value) in seen:
            continue
        seen.add(id(value))
        if not all(
            callable(getattr(value, name, None))
            for name in ("named_modules", "named_parameters", "state_dict")
        ):
            raise DynamicCandidateCaptureIntegrityError(
                "Each forbidden parent must expose a model execution surface."
            )
        result.append(value)
    return tuple(result)


def _model_state_sha(model: Any) -> str:
    try:
        return _digest(
            fingerprint_model_state(model),
            "model state fingerprint",
        )
    except DynamicCandidateCaptureError:
        raise
    except BaseException as error:
        raise DynamicCandidateCaptureIntegrityError(
            "Could not fingerprint candidate model state."
        ) from error


def _model_inventory(model: Any) -> dict[str, Any]:
    try:
        return _json_mapping(
            serialized_inference_state_inventory(model),
            "model inference-state inventory",
        )
    except DynamicCandidateCaptureError:
        raise
    except BaseException as error:
        raise DynamicCandidateCaptureIntegrityError(
            "Candidate is not a dense materialized inference model."
        ) from error


def _assert_inventory_compatible(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> None:
    stable_fields = (
        "serialized_state_elements",
        "serialized_state_bytes",
        "persistent_buffer_elements",
        "persistent_buffer_bytes",
        "runtime_state_elements",
        "runtime_state_bytes",
        "runtime_buffer_elements",
        "runtime_buffer_bytes",
        "nonpersistent_buffer_elements",
        "nonpersistent_buffer_bytes",
        "runtime_named_buffer_count",
        "nonpersistent_named_buffer_count",
        "runtime_unique_buffer_view_count",
    )
    if any(expected.get(name) != actual.get(name) for name in stable_fields):
        raise DynamicCandidateCaptureIntegrityError(
            "Fresh model inference-state inventory differs from memory."
        )


def _assert_dense_unpruned_model(model: Any) -> None:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - dependency boundary
        raise DynamicCandidateCaptureIntegrityError(
            "Candidate validation requires PyTorch."
        ) from error
    named_parameters = getattr(model, "named_parameters", None)
    named_buffers = getattr(model, "named_buffers", None)
    modules = getattr(model, "modules", None)
    if not all(callable(value) for value in (named_parameters, named_buffers, modules)):
        raise DynamicCandidateCaptureIntegrityError(
            "Candidate lacks a complete dense model surface."
        )
    try:
        parameters = tuple(named_parameters(remove_duplicate=False))
        buffers = tuple(named_buffers(remove_duplicate=False))
    except TypeError:
        parameters = tuple(named_parameters())
        buffers = tuple(named_buffers())
    if not parameters:
        raise DynamicCandidateCaptureIntegrityError(
            "Candidate has no learned parameters."
        )
    for name, value in (*parameters, *buffers):
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.device.type == "meta"
            or bool(getattr(value, "is_quantized", False))
        ):
            raise DynamicCandidateCaptureIntegrityError(
                "Candidate state is sparse, quantized, meta, or malformed."
            )
    for name, value in parameters:
        if not (value.is_floating_point() or value.is_complex()):
            raise DynamicCandidateCaptureIntegrityError(
                f"Learned parameter {name!r} is not dense floating-point state."
            )
    for module in modules():
        module_name = f"{type(module).__module__}.{type(module).__qualname__}".casefold()
        if "quantized" in module_name or ".ao.quant" in module_name:
            raise DynamicCandidateCaptureIntegrityError(
                "Quantized module wrappers are forbidden in quarantine."
            )
        hooks = getattr(module, "_forward_pre_hooks", {})
        if isinstance(hooks, Mapping):
            for hook in hooks.values():
                hook_name = (
                    f"{type(hook).__module__}.{type(hook).__qualname__}".casefold()
                )
                if "torch.nn.utils.prune" in hook_name:
                    raise DynamicCandidateCaptureIntegrityError(
                        "Pruning reparameterizations are forbidden in quarantine."
                    )


def _protected_model_snapshots(
    models: Sequence[Any],
) -> tuple[tuple[Any, str, str], ...]:
    return tuple(
        (
            model,
            _model_state_sha(model),
            fingerprint_model_runtime_state(model),
        )
        for model in models
    )


def _assert_protected_models_unchanged(
    models: Sequence[Any],
    snapshots: Sequence[tuple[Any, str, str]],
) -> None:
    if len(models) != len(snapshots):
        raise DynamicCandidateCaptureIntegrityError(
            "Protected parent snapshot cardinality changed."
        )
    for model, state_sha, runtime_sha in snapshots:
        if (
            _model_state_sha(model) != state_sha
            or fingerprint_model_runtime_state(model) != runtime_sha
        ):
            raise DynamicCandidateCaptureIntegrityError(
                "Candidate capture mutated a forbidden parent."
            )


def _freeze_models(
    models: Sequence[Any],
) -> tuple[tuple[Any, bool], ...]:
    result: list[tuple[Any, bool]] = []
    seen: set[int] = set()
    try:
        for model in models:
            parameters = getattr(model, "parameters", None)
            if not callable(parameters):
                raise DynamicCandidateCaptureIntegrityError(
                    "Protected model has no parameters()."
                )
            for parameter in parameters():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                if parameter.grad is not None:
                    raise DynamicCandidateCaptureIntegrityError(
                        "Quarantine requires zero-gradient models."
                    )
                result.append((parameter, bool(parameter.requires_grad)))
                parameter.requires_grad_(False)
    except BaseException:
        _restore_frozen_models(result)
        raise
    return tuple(result)


def _restore_frozen_models(values: Sequence[tuple[Any, bool]]) -> None:
    for parameter, requires_grad in values:
        parameter.grad = None
        parameter.requires_grad_(requires_grad)


def _evaluate_model(
    model: Any,
    *,
    evaluator: Callable[[Any], QuarantineEvaluation],
    label: str,
) -> tuple[QuarantineEvaluation, dict[str, int]]:
    try:
        with observe_model_execution(model) as counts:
            result = evaluator(model)
    except DynamicCandidateCaptureError:
        raise
    except BaseException as error:
        raise DynamicCandidateCaptureIntegrityError(
            f"{label} evaluation failed."
        ) from error
    if type(result) is not QuarantineEvaluation:
        raise DynamicCandidateCaptureIntegrityError(
            "Evaluator must return an exact QuarantineEvaluation."
        )
    normalized = {
        name: int(counts.get(name, 0))
        for name in (
            "module_forward_calls",
            "root_forward_calls",
            "inference_entrypoint_calls",
        )
    }
    if (
        normalized["module_forward_calls"] <= 0
        or normalized["root_forward_calls"] <= 0
        or normalized["inference_entrypoint_calls"] <= 0
    ):
        raise DynamicCandidateCaptureIntegrityError(
            f"{label} did not execute the candidate model."
        )
    return result, normalized


def _require_reproduced_evaluation(
    in_memory: QuarantineEvaluation,
    reloaded: QuarantineEvaluation,
) -> None:
    if in_memory.output_sha256 != reloaded.output_sha256:
        raise DynamicCandidateCaptureIntegrityError(
            "Fresh reload did not reproduce the in-memory whole output."
        )
    if in_memory.evaluation_sha256 != reloaded.evaluation_sha256:
        raise DynamicCandidateCaptureIntegrityError(
            "Fresh reload did not reproduce the in-memory evaluation."
        )


def _challenge_loader(
    loader: Callable[[Path], Any],
    *,
    staging: Path,
) -> str:
    challenge = staging / (
        f".unavailable-loader-challenge-{os.getpid()}-"
        f"{next(tempfile._get_candidate_names())}"
    )
    if _lexists(challenge):
        raise DynamicCandidateCaptureIntegrityError(
            "Loader challenge path unexpectedly exists."
        )
    try:
        loader(challenge)
    except Exception as error:
        exception_type = f"{type(error).__module__}.{type(error).__qualname__}"
    else:
        raise DynamicCandidateCaptureIntegrityError(
            "Loader accepted an absent path and is not standalone-path-bound."
        )
    if _lexists(challenge):
        raise DynamicCandidateCaptureIntegrityError(
            "Loader created or replaced its absent-path challenge."
        )
    return exception_type


def _observed_audit(
    *,
    guard: _FoldOnlyTorchGuard,
    parent_counts: Sequence[Mapping[str, int]],
) -> dict[str, Any]:
    values = {
        "optimizer_constructions": int(guard.optimizer_constructions),
        "optimizer_steps": int(guard.optimizer_steps),
        "backward_calls": int(guard.backward_calls),
        "autograd_grad_calls": int(guard.autograd_grad_calls),
        "training_steps": 0,
        "pruning_operations": int(guard.pruning_operations),
        "quantization_operations": int(guard.quantization_operations),
        "parent_forward_calls": sum(
            int(value.get("module_forward_calls", 0)) for value in parent_counts
        ),
        "parent_inference_calls": sum(
            int(value.get("inference_entrypoint_calls", 0))
            for value in parent_counts
        ),
    }
    for name in _ZERO_COUNTER_FIELDS:
        _exact_zero(values[name], f"capture observed {name}")
    return {
        "schema": QUARANTINE_AUDIT_SCHEMA,
        **values,
        "protected_parent_count": len(parent_counts),
        "training_steps_evidence_scope": (
            "derived-from-zero-guarded-training-api-attempts"
        ),
        "pruning_operations_runtime_observed": True,
        "pruning_observation_scope": (
            "process-local public PyTorch pruning module functions and "
            "exported BasePruningMethod class apply/prune/apply_mask/remove "
            "bindings installed during the guard"
        ),
        "quantization_operations_runtime_observed": True,
        "quantization_observation_scope": (
            "process-local common public PyTorch/torchao quantization and "
            "low-level torch quantize/fake-quant bindings installed during "
            "the guard"
        ),
        "prebound_aliases_runtime_observed": False,
        "prebound_alias_operation_history_proven": False,
        "dense_unpruned_unquantized_retained_state_checked_separately": True,
        "training_used": False,
        "pruning_used": False,
        "quantization_used": False,
        "live_parent_calls_used": False,
        "clean": True,
    }


def _sealed_reload_receipt(
    *,
    loader_sha256: str,
    artifact_inventory_sha256: str,
    in_memory_state_sha256: str,
    fresh_reload_state_sha256: str,
    in_memory_evidence: QuarantineEvaluation,
    reload_evidence: QuarantineEvaluation,
    in_memory_execution: Mapping[str, int],
    reload_execution: Mapping[str, int],
    challenge_exception: str,
) -> dict[str, Any]:
    body = {
        "schema": QUARANTINE_RELOAD_SCHEMA,
        "source": "same-filesystem-staged-immutable-bytes",
        "final_destination_reload_required": True,
        "artifact_inventory_sha256": artifact_inventory_sha256,
        "loader_sha256": loader_sha256,
        "loader_path_challenge_performed": True,
        "loader_path_challenge_exception_type": challenge_exception,
        "loader_returned_disjoint_model": True,
        "in_memory_model_state_sha256": in_memory_state_sha256,
        "fresh_reload_model_state_sha256": fresh_reload_state_sha256,
        "state_reproduced_after_reload": True,
        "in_memory_output_sha256": in_memory_evidence.output_sha256,
        "fresh_reload_output_sha256": reload_evidence.output_sha256,
        "output_reproduced_after_reload": True,
        "in_memory_evaluation_sha256": in_memory_evidence.evaluation_sha256,
        "fresh_reload_evaluation_sha256": reload_evidence.evaluation_sha256,
        "evaluation_reproduced_after_reload": True,
        "in_memory_execution": dict(in_memory_execution),
        "fresh_reload_execution": dict(reload_execution),
        "standalone": True,
        "local_path_only": True,
        "promotable": False,
        "publication_authorized": False,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _sealed_verification_receipt(
    *,
    capture_receipt_sha256: str,
    destination_path_sha256: str,
    state_sha256: str,
    output_sha256: str,
    evaluation_sha256: str,
    artifact_inventory_sha256: str,
    loader_sha256: str,
    evaluator_sha256: str,
    execution: Mapping[str, int],
    observed_audit: Mapping[str, Any],
    challenge_exception: str,
) -> dict[str, Any]:
    body = {
        "schema": QUARANTINE_VERIFICATION_SCHEMA,
        "capture_receipt_sha256": capture_receipt_sha256,
        "destination_path_sha256": destination_path_sha256,
        "model_state_sha256": state_sha256,
        "output_sha256": output_sha256,
        "evaluation_sha256": evaluation_sha256,
        "artifact_inventory_sha256": artifact_inventory_sha256,
        "loader_sha256": loader_sha256,
        "evaluator_sha256": evaluator_sha256,
        "loader_path_challenge_performed": True,
        "loader_path_challenge_exception_type": challenge_exception,
        "fresh_final_destination_reload_performed": True,
        "fresh_reload_execution": dict(execution),
        "capture_observed_audit": dict(observed_audit),
        "promotable": False,
        "publication_authorized": False,
        "requires_reviewed_rerun": True,
    }
    return {**body, "verification_sha256": canonical_sha256(body)}


def _validate_verification_receipt(
    value: Mapping[str, Any],
    *,
    expected_capture_sha256: str,
) -> None:
    body = dict(value)
    claimed = body.pop("verification_sha256", None)
    if (
        body.get("schema") != QUARANTINE_VERIFICATION_SCHEMA
        or body.get("capture_receipt_sha256") != expected_capture_sha256
        or _digest(claimed, "verification_sha256") != canonical_sha256(body)
        or body.get("fresh_final_destination_reload_performed") is not True
        or body.get("promotable") is not False
        or body.get("publication_authorized") is not False
        or body.get("requires_reviewed_rerun") is not True
    ):
        raise DynamicCandidateCaptureIntegrityError(
            "Load verification receipt is invalid or overclaims promotion."
        )


def _validate_receipt_body(body: Mapping[str, Any]) -> None:
    if (
        body.get("schema") != QUARANTINE_CAPTURE_SCHEMA
        or body.get("status") != "quarantined-unreviewed-candidate"
        or body.get("lifecycle") != "immutable-quarantine"
    ):
        raise DynamicCandidateCaptureIntegrityError(
            "Quarantine receipt schema or lifecycle is invalid."
        )
    false_claims = (
        "published_compilation",
        "accepted_compilation",
        "behavior_capability_proven",
        "promotable",
        "publication_authorized",
    )
    true_claims = (
        "complete_artifact_inventory",
        "standalone_candidate",
        "requires_reviewed_rerun",
        "no_parent_sidecar",
        "no_cache_sidecar",
        "no_coordinator_sidecar",
        "no_resume_sidecar",
        "receipt_outside_candidate_artifact",
    )
    if any(body.get(name) is not False for name in false_claims) or any(
        body.get(name) is not True for name in true_claims
    ):
        raise DynamicCandidateCaptureIntegrityError(
            "Quarantine receipt promotion or isolation claims are invalid."
        )
    if body.get("root_entry_allowlist") != [
        CANDIDATE_DIRECTORY_NAME,
        QUARANTINE_RECEIPT_FILENAME,
    ]:
        raise DynamicCandidateCaptureIntegrityError(
            "Quarantine root allowlist is invalid."
        )
    request = body.get("capture_request")
    expected_request_fields = {
        "schema",
        *_REQUEST_DIGEST_FIELDS,
        *_OPTIONAL_REQUEST_DIGEST_FIELDS,
        "compiler_binding_semantics",
        "prior_audit",
    }
    if (
        not isinstance(request, Mapping)
        or set(request) != expected_request_fields
        or request.get("schema") != QUARANTINE_REQUEST_SCHEMA
    ):
        raise DynamicCandidateCaptureIntegrityError(
            "Quarantine capture request is absent or invalid."
        )
    if request.get("compiler_binding_semantics") != (
        _COMPILER_BINDING_SEMANTICS
    ):
        raise DynamicCandidateCaptureIntegrityError(
            "Compiler digest is not labeled as an authenticated compiler-run "
            "receipt/source binding."
        )
    for name in _REQUEST_DIGEST_FIELDS:
        _digest(request.get(name), f"capture request {name}")
    for name in _OPTIONAL_REQUEST_DIGEST_FIELDS:
        value = request.get(name)
        if value is not None:
            _digest(value, f"capture request {name}")
    _validate_zero_audit(request.get("prior_audit"), "prior fold audit")
    bindings = body.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "plan_sha256",
        "parent_sha256",
        "cache_sha256",
        "bridge_sha256",
        "corpus_sha256",
        "compiler_sha256",
        "compiler_binding_semantics",
        "source_sha256",
        "model_state_sha256",
        "output_sha256",
    }:
        raise DynamicCandidateCaptureIntegrityError(
            "Quarantine exact bindings are incomplete."
        )
    for name, value in bindings.items():
        if name == "compiler_binding_semantics":
            if value != _COMPILER_BINDING_SEMANTICS:
                raise DynamicCandidateCaptureIntegrityError(
                    "Compiler binding semantics are invalid."
                )
            continue
        _digest(value, f"binding {name}")
    for name in (
        "plan_sha256",
        "parent_sha256",
        "cache_sha256",
        "bridge_sha256",
        "corpus_sha256",
        "compiler_sha256",
        "source_sha256",
    ):
        if bindings[name] != request[name]:
            raise DynamicCandidateCaptureIntegrityError(
                f"Quarantine {name} binding differs from its request."
            )
    if (
        request["expected_model_state_sha256"] is not None
        and request["expected_model_state_sha256"]
        != bindings["model_state_sha256"]
    ):
        raise DynamicCandidateCaptureIntegrityError(
            "Requested model-state binding was not captured."
        )
    if (
        request["expected_output_sha256"] is not None
        and request["expected_output_sha256"] != bindings["output_sha256"]
    ):
        raise DynamicCandidateCaptureIntegrityError(
            "Requested output binding was not captured."
        )
    supplied = request["pre_reviewed_candidate_fingerprint_sha256"] is not None
    if body.get("pre_reviewed_candidate_fingerprint_supplied") is not supplied:
        raise DynamicCandidateCaptureIntegrityError(
            "Pre-reviewed candidate fingerprint presence claim is invalid."
        )
    callables = body.get("callable_identities")
    if (
        not isinstance(callables, Mapping)
        or set(callables) != _CALLABLE_ROLES
    ):
        raise DynamicCandidateCaptureIntegrityError(
            "Quarantine callable identities are incomplete."
        )
    for role, field_name in (
        ("saver", "saver_sha256"),
        ("loader", "loader_sha256"),
        ("evaluator", "evaluator_sha256"),
    ):
        _validate_callable_identity(
            callables[role],
            role=role,
            expected_binding_sha256=request[field_name],
        )
    _validate_artifact_record(body.get("candidate_artifact"))
    _validate_model_state_record(
        body.get("model_state"),
        expected_state_sha256=bindings["model_state_sha256"],
    )
    _validate_evaluations_record(
        body.get("evaluations"),
        expected_output_sha256=bindings["output_sha256"],
        expected_corpus_sha256=bindings["corpus_sha256"],
        expected_evaluator_sha256=request["evaluator_sha256"],
    )
    _validate_reload_receipt(
        body.get("standalone_reload_receipt"),
        expected_state_sha256=bindings["model_state_sha256"],
        expected_output_sha256=bindings["output_sha256"],
        expected_loader_sha256=request["loader_sha256"],
        expected_inventory_sha256=body["candidate_artifact"][
            "inventory_sha256"
        ],
    )
    audit = body.get("audit")
    if (
        not isinstance(audit, Mapping)
        or audit.get("combined_clean") is not True
        or audit.get("historical_fold_scope_is_caller_rooted") is not True
        or audit.get("capture_scope_is_runtime_observed") is not True
    ):
        raise DynamicCandidateCaptureIntegrityError(
            "Quarantine audit scopes are absent or invalid."
        )
    _validate_zero_audit(audit.get("prior_fold"), "receipt prior fold audit")
    _validate_zero_audit(
        audit.get("capture_observed"),
        "receipt capture observed audit",
        allow_parent_count=True,
    )
    identity = body.get("filesystem_identity")
    if not isinstance(identity, Mapping):
        raise DynamicCandidateCaptureIntegrityError(
            "Quarantine filesystem identity is absent."
        )
    _digest(
        identity.get("destination_path_sha256"),
        "destination path SHA",
    )
    _digest(identity.get("candidate_tree_sha256"), "candidate tree SHA")
    for name in (
        "destination_parent",
        "quarantine_root",
        "candidate_directory",
        "receipt_file",
    ):
        _validate_path_identity(identity.get(name), name)


def _validate_zero_audit(
    value: Any,
    name: str,
    *,
    allow_parent_count: bool = False,
) -> None:
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != QUARANTINE_AUDIT_SCHEMA
        or value.get("training_used") is not False
        or value.get("pruning_used") is not False
        or value.get("quantization_used") is not False
        or value.get("live_parent_calls_used") is not False
        or value.get("clean") is not True
    ):
        raise DynamicCandidateCaptureIntegrityError(f"{name} is invalid.")
    for field_name in _ZERO_COUNTER_FIELDS:
        _exact_zero(value.get(field_name), f"{name} {field_name}")
    if allow_parent_count:
        count = value.get("protected_parent_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise DynamicCandidateCaptureIntegrityError(
                f"{name} has an invalid protected-parent count."
            )


def _validate_callable_identity(
    value: Any,
    *,
    role: str,
    expected_binding_sha256: str,
) -> None:
    expected_fields = {
        "schema",
        "role",
        "implementation_sha256",
        "resource_sha256",
        "resource_identity_explicitly_declared",
        "reachable_manifest_sha256",
        "reachable_manifest",
        "binding_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_fields
        or value.get("schema")
        != "neural-forge-dynamic-candidate-callable-binding-v1"
        or value.get("role") != role
        or not isinstance(
            value.get("resource_identity_explicitly_declared"),
            bool,
        )
    ):
        raise DynamicCandidateCaptureIntegrityError(
            f"Quarantine {role} callable identity is malformed."
        )
    for name in (
        "implementation_sha256",
        "resource_sha256",
        "reachable_manifest_sha256",
        "binding_sha256",
    ):
        _digest(value.get(name), f"{role} {name}")
    if value["reachable_manifest_sha256"] != canonical_sha256(
        value["reachable_manifest"]
    ):
        raise DynamicCandidateCaptureIntegrityError(
            f"Quarantine {role} reachable manifest seal is invalid."
        )
    body = dict(value)
    claimed = body.pop("binding_sha256")
    if claimed != canonical_sha256(body) or claimed != expected_binding_sha256:
        raise DynamicCandidateCaptureIntegrityError(
            f"Quarantine {role} callable binding differs from its request."
        )


def _validate_artifact_record(value: Any) -> None:
    if (
        not isinstance(value, Mapping)
        or value.get("path") != CANDIDATE_DIRECTORY_NAME
        or value.get("standalone") is not True
        or value.get("strict_sidecar_scan_passed") is not True
        or value.get("foreign_entries_present") is not False
    ):
        raise DynamicCandidateCaptureIntegrityError(
            "Quarantine candidate artifact record is invalid."
        )
    inventory = value.get("inventory")
    _validate_inventory(inventory)
    if value.get("inventory_sha256") != inventory.get("inventory_sha256"):
        raise DynamicCandidateCaptureIntegrityError(
            "Candidate inventory roots disagree."
        )


def _validate_model_state_record(
    value: Any,
    *,
    expected_state_sha256: str,
) -> None:
    if (
        not isinstance(value, Mapping)
        or value.get("in_memory_sha256") != expected_state_sha256
        or value.get("fresh_reload_sha256") != expected_state_sha256
        or value.get("byte_identical_after_reload") is not True
        or value.get("dense_materialized") is not True
        or value.get("quantized") is not False
        or value.get("pruned") is not False
        or not isinstance(value.get("in_memory_inventory"), Mapping)
        or not isinstance(value.get("fresh_reload_inventory"), Mapping)
    ):
        raise DynamicCandidateCaptureIntegrityError(
            "Quarantine model-state record is invalid."
        )


def _validate_evaluations_record(
    value: Any,
    *,
    expected_output_sha256: str,
    expected_corpus_sha256: str,
    expected_evaluator_sha256: str,
) -> None:
    if not isinstance(value, Mapping):
        raise DynamicCandidateCaptureIntegrityError(
            "Quarantine evaluation record is absent."
        )
    for name in (
        "in_memory_output_sha256",
        "fresh_reload_output_sha256",
        "in_memory_evaluation_sha256",
        "fresh_reload_evaluation_sha256",
    ):
        _digest(value.get(name), f"evaluation {name}")
    if (
        value.get("in_memory_output_sha256") != expected_output_sha256
        or value.get("fresh_reload_output_sha256") != expected_output_sha256
        or value.get("in_memory_evaluation_sha256")
        != value.get("fresh_reload_evaluation_sha256")
        or value.get("output_reproduced_after_reload") is not True
        or value.get("evaluation_reproduced_after_reload") is not True
        or value.get("corpus_sha256") != expected_corpus_sha256
        or value.get("evaluator_sha256") != expected_evaluator_sha256
    ):
        raise DynamicCandidateCaptureIntegrityError(
            "In-memory and fresh-reload evaluations are not exactly bound."
        )


def _validate_reload_receipt(
    value: Any,
    *,
    expected_state_sha256: str,
    expected_output_sha256: str,
    expected_loader_sha256: str,
    expected_inventory_sha256: str,
) -> None:
    if not isinstance(value, Mapping):
        raise DynamicCandidateCaptureIntegrityError(
            "Standalone reload receipt is absent."
        )
    body = dict(value)
    claimed = body.pop("receipt_sha256", None)
    if (
        body.get("schema") != QUARANTINE_RELOAD_SCHEMA
        or _digest(claimed, "standalone reload receipt SHA")
        != canonical_sha256(body)
        or body.get("artifact_inventory_sha256")
        != expected_inventory_sha256
        or body.get("loader_sha256") != expected_loader_sha256
        or body.get("in_memory_model_state_sha256")
        != expected_state_sha256
        or body.get("fresh_reload_model_state_sha256")
        != expected_state_sha256
        or body.get("in_memory_output_sha256") != expected_output_sha256
        or body.get("fresh_reload_output_sha256") != expected_output_sha256
        or body.get("state_reproduced_after_reload") is not True
        or body.get("output_reproduced_after_reload") is not True
        or body.get("evaluation_reproduced_after_reload") is not True
        or body.get("loader_path_challenge_performed") is not True
        or not isinstance(
            body.get("loader_path_challenge_exception_type"),
            str,
        )
        or not body.get("loader_path_challenge_exception_type")
        or body.get("loader_returned_disjoint_model") is not True
        or body.get("standalone") is not True
        or body.get("local_path_only") is not True
        or body.get("promotable") is not False
        or body.get("publication_authorized") is not False
    ):
        raise DynamicCandidateCaptureIntegrityError(
            "Standalone reload receipt is invalid."
        )
    for name in ("in_memory_execution", "fresh_reload_execution"):
        execution = body.get(name)
        if (
            not isinstance(execution, Mapping)
            or any(
                isinstance(execution.get(field_name), bool)
                or not isinstance(execution.get(field_name), int)
                or execution.get(field_name) <= 0
                for field_name in (
                    "module_forward_calls",
                    "root_forward_calls",
                    "inference_entrypoint_calls",
                )
            )
        ):
            raise DynamicCandidateCaptureIntegrityError(
                "Standalone reload receipt lacks observed model execution."
            )


def _validate_inventory(value: Any) -> None:
    if not isinstance(value, Mapping) or (
        value.get("schema") != QUARANTINE_INVENTORY_SCHEMA
    ):
        raise DynamicCandidateCaptureIntegrityError(
            "Candidate artifact inventory schema is invalid."
        )
    files = value.get("files")
    directories = value.get("directories")
    directory_identities = value.get("directory_identities")
    root_identity = value.get("root_identity")
    if (
        not isinstance(files, list)
        or not files
        or not isinstance(directories, list)
        or not isinstance(directory_identities, list)
        or any(not isinstance(path, str) for path in directories)
        or len(directory_identities) != len(directories)
    ):
        raise DynamicCandidateCaptureIntegrityError(
            "Candidate artifact inventory is incomplete."
        )
    _validate_path_identity(root_identity, "candidate inventory root")
    for index, record in enumerate(directory_identities):
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "device", "inode", "mode"}
            or record.get("path") != directories[index]
        ):
            raise DynamicCandidateCaptureIntegrityError(
                "Candidate directory identity inventory is malformed."
            )
        for name in ("device", "inode", "mode"):
            number = record.get(name)
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise DynamicCandidateCaptureIntegrityError(
                    f"Candidate directory identity {name} is invalid."
                )
    expected_file_fields = {
        "path",
        "size_bytes",
        "sha256",
        "device",
        "inode",
        "link_count",
        "mode",
    }
    for record in files:
        if not isinstance(record, Mapping) or set(record) != expected_file_fields:
            raise DynamicCandidateCaptureIntegrityError(
                "Candidate artifact file inventory is malformed."
            )
        _digest(record.get("sha256"), "candidate artifact file SHA")
        for name in ("size_bytes", "device", "inode", "link_count", "mode"):
            number = record.get(name)
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise DynamicCandidateCaptureIntegrityError(
                    f"Candidate artifact file {name} is invalid."
                )
        if record.get("link_count") != 1:
            raise DynamicCandidateCaptureIntegrityError(
                "Candidate artifact files must not be hard-linked."
            )
    content_records = [
        {
            "path": record["path"],
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
        }
        for record in files
    ]
    content_body = {
        "directories": directories,
        "files": content_records,
    }
    if (
        value.get("file_count") != len(files)
        or value.get("directory_count") != len(directories)
        or value.get("total_bytes")
        != sum(record["size_bytes"] for record in files)
        or value.get("content_tree_sha256") != canonical_sha256(content_body)
        or value.get("filesystem_identity_sha256")
        != canonical_sha256(
            {
                "root": root_identity,
                "directories": directory_identities,
                "files": [
                    {
                        "path": record["path"],
                        "device": record["device"],
                        "inode": record["inode"],
                        "link_count": record["link_count"],
                        "mode": record["mode"],
                    }
                    for record in files
                ],
            }
        )
    ):
        raise DynamicCandidateCaptureIntegrityError(
            "Candidate artifact inventory seals do not verify."
        )
    body = dict(value)
    claimed = body.pop("inventory_sha256", None)
    if _digest(claimed, "inventory_sha256") != canonical_sha256(body):
        raise DynamicCandidateCaptureIntegrityError(
            "Candidate artifact inventory self-hash does not verify."
        )


def _candidate_inventory(root: Path) -> dict[str, Any]:
    identities = _tree_identity_snapshot(root)
    files = []
    for record in identities["files"]:
        path = root / Path(record["path"])
        files.append(
            {
                "path": record["path"],
                "size_bytes": record["size_bytes"],
                "sha256": _file_sha256_stable(path, expected=record),
                "device": record["device"],
                "inode": record["inode"],
                "link_count": record["link_count"],
                "mode": record["mode"],
            }
        )
    if not files:
        raise DynamicCandidateCaptureIntegrityError(
            "Candidate saver produced an empty artifact."
        )
    directories = [record["path"] for record in identities["directories"]]
    content_records = [
        {
            "path": record["path"],
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
        }
        for record in files
    ]
    content_body = {
        "directories": directories,
        "files": content_records,
    }
    directory_identities = identities["directories"]
    filesystem_identity = {
        "root": identities["root"],
        "directories": directory_identities,
        "files": [
            {
                "path": record["path"],
                "device": record["device"],
                "inode": record["inode"],
                "link_count": record["link_count"],
                "mode": record["mode"],
            }
            for record in files
        ],
    }
    body = {
        "schema": QUARANTINE_INVENTORY_SCHEMA,
        "file_count": len(files),
        "directory_count": len(directories),
        "total_bytes": sum(record["size_bytes"] for record in files),
        "content_tree_sha256": canonical_sha256(content_body),
        "filesystem_identity_sha256": canonical_sha256(filesystem_identity),
        "directories": directories,
        "root_identity": identities["root"],
        "directory_identities": directory_identities,
        "files": files,
    }
    inventory = {**body, "inventory_sha256": canonical_sha256(body)}
    _validate_inventory(inventory)
    _cross_check_standalone_artifact_tree(root, inventory)
    return inventory


def _cross_check_standalone_artifact_tree(
    root: Path,
    inventory: Mapping[str, Any],
) -> None:
    try:
        checked = _standalone_artifact_tree(root)
    except BaseException as error:
        raise DynamicCandidateCaptureIntegrityError(
            "Candidate artifact failed the canonical standalone sidecar scan."
        ) from error
    expected = tuple(
        {
            "path": record["path"],
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
        }
        for record in inventory["files"]
    )
    if checked != expected:
        raise DynamicCandidateCaptureIntegrityError(
            "Canonical artifact scan differs from complete inventory."
        )
    for record in inventory["files"]:
        lowered = record["path"].casefold()
        if any(part in lowered for part in _FORBIDDEN_ARTIFACT_NAME_PARTS):
            raise DynamicCandidateCaptureIntegrityError(
                "Candidate contains a forbidden sidecar-like artifact name."
            )


def _tree_identity_snapshot(root: Path) -> dict[str, Any]:
    _assert_ordinary_directory(root, "candidate artifact")
    root_stat = root.lstat()
    root_device = int(root_stat.st_dev)
    directories: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise DynamicCandidateCaptureIntegrityError(
                "Cannot enumerate candidate artifact."
            ) from error
        for entry in entries:
            path = Path(entry.path)
            if _is_reparse_point(path):
                raise DynamicCandidateCaptureIntegrityError(
                    "Candidate artifact contains a symlink or reparse point."
                )
            try:
                metadata = path.lstat()
            except OSError as error:
                raise DynamicCandidateCaptureIntegrityError(
                    "Cannot inspect candidate artifact entry."
                ) from error
            if int(metadata.st_dev) != root_device:
                raise DynamicCandidateCaptureIntegrityError(
                    "Candidate artifact crosses a filesystem device."
                )
            relative = path.relative_to(root).as_posix()
            mode = int(stat.S_IFMT(metadata.st_mode))
            if entry.is_dir(follow_symlinks=False):
                directories.append(
                    {
                        "path": relative,
                        "device": int(metadata.st_dev),
                        "inode": int(metadata.st_ino),
                        "mode": mode,
                    }
                )
                visit(path)
            elif entry.is_file(follow_symlinks=False):
                if int(metadata.st_nlink) != 1:
                    raise DynamicCandidateCaptureIntegrityError(
                        "Candidate artifact contains a hard-linked file."
                    )
                files.append(
                    {
                        "path": relative,
                        "size_bytes": int(metadata.st_size),
                        "device": int(metadata.st_dev),
                        "inode": int(metadata.st_ino),
                        "link_count": int(metadata.st_nlink),
                        "mode": mode,
                    }
                )
            else:
                raise DynamicCandidateCaptureIntegrityError(
                    "Candidate artifact contains a non-file entry."
                )

    visit(root)
    return {
        "root": _stat_identity(root_stat, include_size=False),
        "directories": directories,
        "files": files,
    }


def _file_sha256_stable(path: Path, *, expected: Mapping[str, Any]) -> str:
    _assert_ordinary_file(path, f"candidate artifact {expected['path']}")
    before = path.stat()
    if not _stat_matches_record(before, expected):
        raise DynamicCandidateCaptureIntegrityError(
            "Candidate artifact path changed before hashing."
        )
    digest = hashlib.sha256()
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            opened = os.fstat(descriptor)
            if not _stat_matches_record(opened, expected):
                raise DynamicCandidateCaptureIntegrityError(
                    "Candidate artifact was replaced while opening."
                )
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            after = os.fstat(descriptor)
            if not _stat_matches_record(after, expected):
                raise DynamicCandidateCaptureIntegrityError(
                    "Candidate artifact changed while hashing."
                )
        finally:
            os.close(descriptor)
    except DynamicCandidateCaptureError:
        raise
    except OSError as error:
        raise DynamicCandidateCaptureIntegrityError(
            "Cannot hash candidate artifact."
        ) from error
    final = path.stat()
    if not _stat_matches_record(final, expected):
        raise DynamicCandidateCaptureIntegrityError(
            "Candidate artifact path changed after hashing."
        )
    return digest.hexdigest()


def _stat_matches_record(value: os.stat_result, expected: Mapping[str, Any]) -> bool:
    return (
        int(value.st_dev) == expected["device"]
        and int(value.st_ino) == expected["inode"]
        and int(value.st_nlink) == expected["link_count"]
        and int(value.st_size) == expected["size_bytes"]
        and int(stat.S_IFMT(value.st_mode)) == expected["mode"]
    )


def _validate_path_identity(value: Any, name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "device",
        "inode",
        "link_count",
        "mode",
    }:
        raise DynamicCandidateCaptureIntegrityError(
            f"{name} filesystem identity is malformed."
        )
    for field_name in ("device", "inode", "link_count", "mode"):
        number = value.get(field_name)
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise DynamicCandidateCaptureIntegrityError(
                f"{name} {field_name} is invalid."
            )
    if (
        stat.S_ISREG(value.get("mode"))
        and value.get("link_count") != 1
    ):
        raise DynamicCandidateCaptureIntegrityError(
            f"{name} must not be hard-linked."
        )


def _path_identity(path: Path, name: str) -> dict[str, int]:
    try:
        value = path.lstat()
    except OSError as error:
        raise DynamicCandidateCaptureIntegrityError(
            f"Cannot inspect {name}."
        ) from error
    if _is_reparse_point(path):
        raise DynamicCandidateCaptureIntegrityError(
            f"{name} cannot be a symlink or reparse point."
        )
    return _stat_identity(value, include_size=False)


def _directory_identity(path: Path, name: str) -> dict[str, int]:
    _assert_ordinary_directory(path, name)
    return _path_identity(path, name)


def _stat_identity(
    value: os.stat_result,
    *,
    include_size: bool,
) -> dict[str, int]:
    result = {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "link_count": int(value.st_nlink),
        "mode": int(stat.S_IFMT(value.st_mode)),
    }
    if include_size:
        result["size_bytes"] = int(value.st_size)
    return result


def _new_destination(value: str | os.PathLike[str] | Path) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise DynamicCandidateCapturePublicationError(
            "Quarantine destination must be a local filesystem path."
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = Path(os.path.abspath(path))
    if path.name in {"", ".", ".."}:
        raise DynamicCandidateCapturePublicationError(
            "Quarantine destination must name a child directory."
        )
    _reject_reparse_ancestors(path.parent)
    if _lexists(path):
        raise DynamicCandidateCapturePublicationError(
            f"Immutable quarantine destination already exists: {path}"
        )
    return path


def _existing_destination(value: str | os.PathLike[str]) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise DynamicCandidateCaptureIntegrityError(
            "Quarantine source must be a local filesystem path."
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = Path(os.path.abspath(path))
    _reject_reparse_ancestors(path)
    _assert_ordinary_directory(path, "quarantine root")
    return path


def _reject_reparse_ancestors(path: Path) -> None:
    chain = []
    current = path
    while True:
        chain.append(current)
        if current.parent == current:
            break
        current = current.parent
    for candidate in reversed(chain):
        if not _lexists(candidate):
            continue
        if _is_reparse_point(candidate):
            raise DynamicCandidateCaptureIntegrityError(
                "Quarantine paths cannot traverse symlinks or reparse points."
            )
        if candidate != path and not candidate.is_dir():
            raise DynamicCandidateCaptureIntegrityError(
                "Quarantine ancestor is not an ordinary directory."
            )


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _is_reparse_point(path: Path) -> bool:
    try:
        value = path.lstat()
    except OSError as error:
        raise DynamicCandidateCaptureIntegrityError(
            f"Cannot inspect path {path}."
        ) from error
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(value, "st_file_attributes", 0)
    return bool(path.is_symlink() or (flag and attributes & flag))


def _assert_ordinary_directory(path: Path, name: str) -> None:
    if not _lexists(path) or _is_reparse_point(path) or not path.is_dir():
        raise DynamicCandidateCaptureIntegrityError(
            f"{name} must be an ordinary directory."
        )


def _assert_ordinary_file(path: Path, name: str) -> None:
    if not _lexists(path) or _is_reparse_point(path) or not path.is_file():
        raise DynamicCandidateCaptureIntegrityError(
            f"{name} must be an ordinary file."
        )
    try:
        links = int(path.stat().st_nlink)
    except OSError as error:
        raise DynamicCandidateCaptureIntegrityError(
            f"Cannot inspect {name}."
        ) from error
    if links != 1:
        raise DynamicCandidateCaptureIntegrityError(
            f"{name} must not be hard-linked."
        )


def _open_exclusive(path: Path) -> int:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    flags |= getattr(os, "O_BINARY", 0)
    try:
        return os.open(path, flags, 0o600)
    except OSError as error:
        raise DynamicCandidateCapturePublicationError(
            "Cannot exclusively create quarantine receipt."
        ) from error


def _write_descriptor(descriptor: int, payload: bytes) -> None:
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise DynamicCandidateCapturePublicationError(
            "Cannot durably write quarantine receipt."
        ) from error


def _strict_json_file(path: Path, name: str) -> dict[str, Any]:
    _assert_ordinary_file(path, name)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise DynamicCandidateCaptureIntegrityError(
            f"Cannot read {name}."
        ) from error
    if not payload or len(payload) > _MAX_JSON_BYTES:
        raise DynamicCandidateCaptureIntegrityError(
            f"{name} is empty or exceeds its size limit."
        )

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise DynamicCandidateCaptureIntegrityError(
                    f"{name} contains duplicate key {key!r}."
                )
            result[key] = value
        return result

    def invalid_constant(value: str) -> Any:
        raise DynamicCandidateCaptureIntegrityError(
            f"{name} contains non-finite value {value!r}."
        )

    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
    except DynamicCandidateCaptureError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DynamicCandidateCaptureIntegrityError(
            f"{name} is not canonical ASCII JSON."
        ) from error
    if not isinstance(value, dict):
        raise DynamicCandidateCaptureIntegrityError(f"{name} must be an object.")
    if payload != canonical_json_bytes(value) + b"\n":
        raise DynamicCandidateCaptureIntegrityError(
            f"{name} is not in its unique canonical encoding."
        )
    return value


def _verify_quarantine_metadata(
    root: Path,
    *,
    request: QuarantineCaptureRequest,
    expected_receipt_sha256: str,
    expected_destination_path: Path,
    allow_staging_root: bool,
) -> tuple[QuarantineCaptureReceipt, Path]:
    _assert_ordinary_directory(root, "quarantine root")
    try:
        entries = {entry.name for entry in os.scandir(root)}
    except OSError as error:
        raise DynamicCandidateCaptureIntegrityError(
            "Cannot enumerate quarantine root."
        ) from error
    expected_entries = {
        CANDIDATE_DIRECTORY_NAME,
        QUARANTINE_RECEIPT_FILENAME,
    }
    if entries != expected_entries:
        raise DynamicCandidateCaptureIntegrityError(
            "Quarantine root contains missing or foreign entries."
        )
    artifact_path = root / CANDIDATE_DIRECTORY_NAME
    receipt_path = root / QUARANTINE_RECEIPT_FILENAME
    _assert_ordinary_directory(artifact_path, "candidate artifact")
    _assert_ordinary_file(receipt_path, "quarantine receipt")
    receipt = QuarantineCaptureReceipt.from_dict(
        _strict_json_file(receipt_path, "quarantine receipt")
    )
    if receipt.receipt_sha256 != _digest(
        expected_receipt_sha256,
        "expected_receipt_sha256",
    ):
        raise DynamicCandidateCaptureIntegrityError(
            "Quarantine receipt differs from its external content root."
        )
    payload = receipt.as_dict()
    if payload["capture_request"] != request.as_dict():
        raise DynamicCandidateCaptureIntegrityError(
            "Quarantine capture request differs from external bindings."
        )
    identity = payload["filesystem_identity"]
    if identity["destination_path_sha256"] != canonical_sha256(
        os.fspath(expected_destination_path)
    ):
        raise DynamicCandidateCaptureIntegrityError(
            "Quarantine destination path binding is invalid."
        )
    if identity["destination_parent"] != _directory_identity(
        expected_destination_path.parent,
        "destination parent",
    ):
        raise DynamicCandidateCaptureIntegrityError(
            "Quarantine destination parent was replaced."
        )
    if not allow_staging_root and (
        identity["quarantine_root"]
        != _path_identity(root, "quarantine root")
    ):
        raise DynamicCandidateCaptureIntegrityError(
            "Quarantine root path was replaced."
        )
    if identity["candidate_directory"] != _path_identity(
        artifact_path,
        "candidate artifact",
    ):
        raise DynamicCandidateCaptureIntegrityError(
            "Candidate artifact directory was replaced."
        )
    if identity["receipt_file"] != _path_identity(
        receipt_path,
        "quarantine receipt",
    ):
        raise DynamicCandidateCaptureIntegrityError(
            "Quarantine receipt path was replaced."
        )
    actual_inventory = _candidate_inventory(artifact_path)
    expected_inventory = payload["candidate_artifact"]["inventory"]
    if actual_inventory != expected_inventory:
        raise DynamicCandidateCaptureIntegrityError(
            "Candidate artifact inventory differs from its receipt."
        )
    identities = _tree_identity_snapshot(artifact_path)
    if identity["candidate_tree_sha256"] != canonical_sha256(identities):
        raise DynamicCandidateCaptureIntegrityError(
            "Candidate artifact tree identities were replaced."
        )
    return receipt, artifact_path


def _rename_noreplace(source: Path, destination: Path) -> None:
    if _lexists(destination):
        raise DynamicCandidateCapturePublicationError(
            f"Immutable quarantine destination already exists: {destination}"
        )
    if os.name == "nt":
        try:
            os.rename(source, destination)
        except FileExistsError as error:
            raise DynamicCandidateCapturePublicationError(
                "Concurrent writer won the quarantine destination."
            ) from error
        except OSError as error:
            raise DynamicCandidateCapturePublicationError(
                "Atomic quarantine publication failed."
            ) from error
        return
    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise DynamicCandidateCapturePublicationError(
                "This Linux runtime lacks atomic rename-without-replacement."
            )
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
        if result != 0:
            code = ctypes.get_errno()
            if code in {errno.EEXIST, errno.ENOTEMPTY}:
                raise DynamicCandidateCapturePublicationError(
                    "Concurrent writer won the quarantine destination."
                )
            raise DynamicCandidateCapturePublicationError(
                f"Atomic quarantine publication failed with errno {code}."
            )
        return
    raise DynamicCandidateCapturePublicationError(
        "Atomic no-clobber directory publication is unsupported on this platform."
    )


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if _is_reparse_point(path):
            raise DynamicCandidateCaptureIntegrityError(
                "Cannot fsync a quarantine containing reparse points."
            )
        if path.is_file():
            try:
                descriptor = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except OSError:
                if os.name != "nt":
                    raise
    for directory in sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        _fsync_directory(directory)
    _fsync_directory(root)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "CANDIDATE_DIRECTORY_NAME",
    "QUARANTINE_AUDIT_SCHEMA",
    "QUARANTINE_CAPTURE_SCHEMA",
    "QUARANTINE_INVENTORY_SCHEMA",
    "QUARANTINE_RECEIPT_FILENAME",
    "QUARANTINE_RELOAD_SCHEMA",
    "QUARANTINE_REQUEST_SCHEMA",
    "QUARANTINE_VERIFICATION_SCHEMA",
    "DynamicCandidateCaptureError",
    "DynamicCandidateCaptureIntegrityError",
    "DynamicCandidateCapturePublicationError",
    "QuarantineCaptureReceipt",
    "QuarantineCaptureRequest",
    "QuarantineEvaluation",
    "QuarantineZeroOperationAudit",
    "QuarantinedDynamicFoldCandidate",
    "capture_quarantined_candidate",
    "fingerprint_quarantine_callable",
    "load_quarantined_candidate",
]
