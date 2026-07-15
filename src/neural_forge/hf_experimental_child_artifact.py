"""Atomic, neutral evidence bundle for an experimental folded child.

The publisher deliberately does not reuse the accepted compiled-child cache:
an experimental recovery initialization is not a compiler acceptance event.
Instead, one atomically promoted publication directory contains a clean
``child`` Hugging Face artifact and two sibling evidence files.  The caller's
bridge materialization context is entered only while the model and tokenizer
are serialized.  It is exited before the staged child is freshly loaded.

The receipt is intentionally outside the child directory, so the child stays
standalone and contains no factor, optimizer, parent, or evidence sidecars.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural_forge.hf_compiled_child_cache import model_state_fingerprint
from neural_forge.hf_exact_candidate_frontier import artifact_tree


EXPERIMENTAL_CHILD_ARTIFACT_SCHEMA = (
    "neural-forge-experimental-folded-child-artifact-v1"
)
EXACT_TARGET_EXPERIMENTAL_RECOVERY_MATERIALIZATION_SCHEMA = (
    "neural-forge-exact-target-experimental-recovery-materialization-v1"
)
ARTIFACT_DIRECTORY_NAME = "child"
BRIDGE_REPORT_FILENAME = "bridge_materialization_report.json"
ARTIFACT_RECEIPT_FILENAME = "experimental_child_artifact_receipt.json"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFETENSORS_SHARD = re.compile(r"model-\d{5}-of-\d{5}\.safetensors")
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MODEL_JSON_FILES = frozenset(
    {
        "config.json",
        "generation_config.json",
    }
)
_TOKENIZER_JSON_FILES = frozenset(
    {
        "added_tokens.json",
        "chat_template.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
)
_TOKENIZER_OTHER_FILES = frozenset(
    {
        "chat_template.jinja",
        "merges.txt",
        "sentencepiece.bpe.model",
        "tokenizer.model",
    }
)
_TOKENIZER_PAYLOAD_FILES = frozenset(
    {
        "merges.txt",
        "sentencepiece.bpe.model",
        "tokenizer.json",
        "tokenizer.model",
        "vocab.json",
    }
)
_FORBIDDEN_NAME_PARTS = (
    "adapter",
    "factor",
    "optimizer",
    "parent",
    "scheduler",
    "sidecar",
    "trainer",
    "training_args",
)
_AUDIT_COUNT_FIELDS = frozenset(
    {
        "adapter_or_registry_module_count",
        "optimizer_parameter_count",
        "source_sized_sidecar_parameter_count",
        "parent_runtime_reference_count",
    }
)
_BRIDGE_REPORT_RECORD_KEYS = frozenset(
    {
        "path",
        "file_sha256",
        "size_bytes",
        "materialization_transaction_sha256",
        "native_child_state_sha256",
        "materialized_bridge_state_sha256",
        "canonical_bytes_verified",
    }
)
_ARTIFACT_RECORD_KEYS = frozenset(
    {
        "path",
        "tree",
        "config_payload_sha256",
        "strict_model_tokenizer_allowlist_verified",
        "receipt_stored_outside_artifact_tree",
    }
)
_MODEL_STATE_FINGERPRINT_KEYS = frozenset(
    {
        "schema",
        "model_class",
        "state_dict_sha256",
        "compiler_state_sha256",
        "state_key_sha256",
        "state_manifest_sha256",
        "state_key_count",
        "logical_state_elements",
        "unique_parameter_count",
        "logical_parameter_count",
        "parameter_topology_sha256",
        "buffer_topology_sha256",
        "all_state_tensors_finite",
    }
)
_RECEIPT_BODY_KEYS = frozenset(
    {
        "schema",
        "status",
        "arm_id",
        "preregistration_sha256",
        "shared_execution_contract_sha256",
        "bridge_report",
        "artifact",
        "fresh_reload_model_state_fingerprint",
        "fresh_reload_native_certificate",
        "fresh_reload_runtime_audit",
        "fresh_reload_bridge_state_sha256",
        "fresh_reload_state_dict_sha256",
        "exact_unique_parameters",
        "exact_logical_parameters",
        "exact_budget_match",
        "fresh_reload_performed",
        "artifact_tree_unchanged_by_fresh_reload",
        "local_files_only",
        "trust_remote_code",
        "adapter_or_registry_module_count",
        "optimizer_parameter_count",
        "source_sized_sidecar_parameter_count",
        "parent_runtime_reference_count",
        "optimizer_steps",
        "sealed_data_accessed",
        "publication_authorized",
        "accepted_artifact",
        "behavior_capability_proven",
    }
)
_RECEIPT_KEYS = _RECEIPT_BODY_KEYS | {"receipt_sha256"}


class ExperimentalChildArtifactError(RuntimeError):
    """Base failure for experimental child artifact evidence."""


class ExperimentalChildArtifactPublicationError(ExperimentalChildArtifactError):
    """The experimental child could not be atomically published."""


class ExperimentalChildArtifactIntegrityError(ExperimentalChildArtifactError):
    """Artifact bytes, model state, or evidence failed authentication."""


@dataclass(frozen=True, slots=True)
class ExperimentalChildArtifactExpectations:
    """Externally rooted bindings for one experimental child publication."""

    preregistration_sha256: str
    shared_execution_contract_sha256: str
    config_payload_sha256: str
    state_key_sha256: str
    parameter_topology_sha256: str
    buffer_topology_sha256: str
    exact_unique_parameters: int
    exact_logical_parameters: int
    arm_id: str = "folded"

    def __post_init__(self) -> None:
        for name in (
            "preregistration_sha256",
            "shared_execution_contract_sha256",
            "config_payload_sha256",
            "state_key_sha256",
            "parameter_topology_sha256",
            "buffer_topology_sha256",
        ):
            _sha256(getattr(self, name), name)
        for name in ("exact_unique_parameters", "exact_logical_parameters"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ExperimentalChildArtifactIntegrityError(
                    f"{name} must be a positive integer."
                )
        if not isinstance(self.arm_id, str) or not self.arm_id.strip():
            raise ExperimentalChildArtifactIntegrityError(
                "arm_id must be a nonempty string."
            )


@dataclass(frozen=True, slots=True, init=False)
class ExperimentalChildArtifactReceipt:
    """Canonical self-hashed receipt for a neutral experimental child."""

    _body_json: bytes
    receipt_sha256: str

    def __init__(
        self,
        body: Mapping[str, Any],
        *,
        receipt_sha256: str | None = None,
    ) -> None:
        normalized = _json_value(body, "artifact receipt")
        if not isinstance(normalized, dict) or set(normalized) != _RECEIPT_BODY_KEYS:
            raise ExperimentalChildArtifactIntegrityError(
                "Artifact receipt does not contain exactly the v1 body fields."
            )
        body_json = _canonical_json_bytes(normalized)
        actual = hashlib.sha256(body_json).hexdigest()
        if (
            receipt_sha256 is not None
            and _sha256(receipt_sha256, "receipt_sha256") != actual
        ):
            raise ExperimentalChildArtifactIntegrityError(
                "Artifact receipt self-hash does not verify."
            )
        object.__setattr__(self, "_body_json", body_json)
        object.__setattr__(self, "receipt_sha256", actual)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExperimentalChildArtifactReceipt:
        value = _json_value(payload, "artifact receipt")
        if not isinstance(value, dict) or set(value) != _RECEIPT_KEYS:
            raise ExperimentalChildArtifactIntegrityError(
                "Artifact receipt does not contain exactly the v1 fields."
            )
        claimed = value.pop("receipt_sha256")
        return cls(value, receipt_sha256=claimed)

    def as_dict(self) -> dict[str, Any]:
        result = json.loads(self._body_json.decode("ascii"))
        result["receipt_sha256"] = self.receipt_sha256
        return result


@dataclass(frozen=True, slots=True)
class VerifiedExperimentalChildArtifact:
    """Fully reloaded and authenticated experimental child metadata."""

    publication_path: Path
    artifact_path: Path
    bridge_report: Mapping[str, Any]
    receipt: ExperimentalChildArtifactReceipt


def publish_experimental_child_artifact(
    publication_dir: str | os.PathLike[str],
    *,
    model: Any,
    tokenizer: Any,
    materialization_context: AbstractContextManager[bytes],
    expectations: ExperimentalChildArtifactExpectations,
    model_loader: Callable[[Path], Any],
    native_certifier: Callable[[Any], Mapping[str, Any] | Any],
    runtime_auditor: Callable[[Any, Path], Mapping[str, Any]],
    model_saver: Callable[[Any, Path], None] | None = None,
    tokenizer_saver: Callable[[Any, Path], None] | None = None,
    dense_state_hasher: Callable[[Any], str] | None = None,
) -> ExperimentalChildArtifactReceipt:
    """Serialize in a bridge context, exit it, then reload and publish atomically.

    ``materialization_context`` must yield the exact canonical bytes produced by
    ``ExactTargetCompilerTransaction.experimental_recovery_materialization``.
    The model and tokenizer savers run inside that context.  The loader and all
    independent certificate callbacks run only after the context has exited.
    """

    _require_expectations(expectations)
    if not callable(model_loader):
        raise ExperimentalChildArtifactPublicationError(
            "model_loader must be callable."
        )
    if not callable(native_certifier) or not callable(runtime_auditor):
        raise ExperimentalChildArtifactPublicationError(
            "native_certifier and runtime_auditor must be callable."
        )
    saver = _default_model_saver if model_saver is None else model_saver
    tokenizer_writer = (
        _default_tokenizer_saver if tokenizer_saver is None else tokenizer_saver
    )
    hasher = (
        _default_dense_state_hasher
        if dense_state_hasher is None
        else dense_state_hasher
    )
    if not callable(saver) or not callable(tokenizer_writer) or not callable(hasher):
        raise ExperimentalChildArtifactPublicationError(
            "model/tokenizer savers and dense_state_hasher must be callable."
        )

    destination = _publication_destination(publication_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ExperimentalChildArtifactPublicationError(
            f"Immutable experimental child publication already exists: {destination}"
        )
    lock_path = destination.parent / f".{destination.name}.artifact.lock"
    lock_fd: int | None = None
    lock_identity: tuple[int, int] | None = None
    staging: Path | None = None
    try:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise ExperimentalChildArtifactPublicationError(
                f"Experimental child publication lock already exists: {lock_path}"
            ) from error
        lock_stat = os.fstat(lock_fd)
        lock_identity = (int(lock_stat.st_dev), int(lock_stat.st_ino))
        os.write(lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(lock_fd)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.",
                suffix=".staging",
                dir=destination.parent,
            )
        )
        artifact_path = staging / ARTIFACT_DIRECTORY_NAME
        artifact_path.mkdir()

        report_bytes: bytes
        report: dict[str, Any]
        live_bridge_sha: str
        try:
            with materialization_context as yielded_report_bytes:
                report_bytes, report = _authenticate_bridge_report_bytes(
                    yielded_report_bytes,
                    expectations=expectations,
                )
                live_bridge_sha = _sha256(
                    hasher(model), "live materialized bridge state SHA"
                )
                if live_bridge_sha != report["materialized_native_state_sha256"]:
                    raise ExperimentalChildArtifactPublicationError(
                        "Live materialized model does not match the canonical bridge report."
                    )
                saver(model, artifact_path)
                tokenizer_writer(tokenizer, artifact_path)
                if (
                    _sha256(
                        hasher(model),
                        "post-serialization materialized bridge state SHA",
                    )
                    != live_bridge_sha
                ):
                    raise ExperimentalChildArtifactPublicationError(
                        "Serialization mutated the live materialized model."
                    )
        except ExperimentalChildArtifactError:
            raise
        except BaseException as error:
            raise ExperimentalChildArtifactPublicationError(
                "Bridge-context serialization failed."
            ) from error

        expected_state_keys = tuple(sorted(str(key) for key in model.state_dict()))
        config_payload_sha = _validate_hf_artifact_files(
            artifact_path,
            expected_state_keys=expected_state_keys,
            expected_state_key_sha256=None,
        )
        if config_payload_sha != expectations.config_payload_sha256:
            raise ExperimentalChildArtifactPublicationError(
                "Serialized config does not match the expected target config."
            )
        tree_before_reload = artifact_tree(artifact_path)
        reloaded_evidence = _authenticate_fresh_reload(
            artifact_path,
            report=report,
            expectations=expectations,
            model_loader=model_loader,
            native_certifier=native_certifier,
            runtime_auditor=runtime_auditor,
            dense_state_hasher=hasher,
        )
        tree_after_reload = artifact_tree(artifact_path)
        if tree_after_reload != tree_before_reload:
            raise ExperimentalChildArtifactPublicationError(
                "Fresh reload or an auditor mutated the staged artifact tree."
            )

        (staging / BRIDGE_REPORT_FILENAME).write_bytes(report_bytes)
        fingerprint = reloaded_evidence["model_state_fingerprint"]
        audit = reloaded_evidence["runtime_audit"]
        receipt_body = {
            "schema": EXPERIMENTAL_CHILD_ARTIFACT_SCHEMA,
            "status": "experimental-unaccepted",
            "arm_id": expectations.arm_id,
            "preregistration_sha256": expectations.preregistration_sha256,
            "shared_execution_contract_sha256": (
                expectations.shared_execution_contract_sha256
            ),
            "bridge_report": {
                "path": BRIDGE_REPORT_FILENAME,
                "file_sha256": hashlib.sha256(report_bytes).hexdigest(),
                "size_bytes": len(report_bytes),
                "materialization_transaction_sha256": report[
                    "materialization_transaction_sha256"
                ],
                "native_child_state_sha256": report["native_child_state_sha256"],
                "materialized_bridge_state_sha256": report[
                    "materialized_native_state_sha256"
                ],
                "canonical_bytes_verified": True,
            },
            "artifact": {
                "path": ARTIFACT_DIRECTORY_NAME,
                "tree": tree_after_reload,
                "config_payload_sha256": config_payload_sha,
                "strict_model_tokenizer_allowlist_verified": True,
                "receipt_stored_outside_artifact_tree": True,
            },
            "fresh_reload_model_state_fingerprint": fingerprint,
            "fresh_reload_native_certificate": reloaded_evidence["native_certificate"],
            "fresh_reload_runtime_audit": audit,
            "fresh_reload_bridge_state_sha256": reloaded_evidence[
                "bridge_state_sha256"
            ],
            "fresh_reload_state_dict_sha256": fingerprint["state_dict_sha256"],
            "exact_unique_parameters": expectations.exact_unique_parameters,
            "exact_logical_parameters": expectations.exact_logical_parameters,
            "exact_budget_match": True,
            "fresh_reload_performed": True,
            "artifact_tree_unchanged_by_fresh_reload": True,
            "local_files_only": True,
            "trust_remote_code": False,
            **{name: audit[name] for name in sorted(_AUDIT_COUNT_FIELDS)},
            "optimizer_steps": 0,
            "sealed_data_accessed": False,
            "publication_authorized": False,
            "accepted_artifact": False,
            "behavior_capability_proven": False,
        }
        receipt = ExperimentalChildArtifactReceipt(receipt_body)
        _write_exclusive(
            staging / ARTIFACT_RECEIPT_FILENAME,
            _canonical_json_bytes(receipt.as_dict()) + b"\n",
        )
        _fsync_tree(staging)
        _verify_metadata(
            staging,
            expectations=expectations,
            expected_receipt_sha256=receipt.receipt_sha256,
        )
        if destination.exists():
            raise ExperimentalChildArtifactPublicationError(
                f"Immutable experimental child publication already exists: {destination}"
            )
        os.rename(staging, destination)
        staging = None
        _fsync_directory(destination.parent)
        return receipt
    except ExperimentalChildArtifactError:
        raise
    except BaseException as error:
        raise ExperimentalChildArtifactPublicationError(
            "Experimental child publication failed before atomic promotion."
        ) from error
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        if lock_fd is not None:
            os.close(lock_fd)
        if lock_identity is not None:
            try:
                current = lock_path.stat()
            except FileNotFoundError:
                current = None
            if (
                current is not None
                and (
                    int(current.st_dev),
                    int(current.st_ino),
                )
                == lock_identity
            ):
                lock_path.unlink()


def verify_experimental_child_artifact(
    publication_dir: str | os.PathLike[str],
    *,
    expectations: ExperimentalChildArtifactExpectations,
    expected_receipt_sha256: str,
    model_loader: Callable[[Path], Any],
    native_certifier: Callable[[Any], Mapping[str, Any] | Any],
    runtime_auditor: Callable[[Any, Path], Mapping[str, Any]],
    dense_state_hasher: Callable[[Any], str] | None = None,
) -> VerifiedExperimentalChildArtifact:
    """Verify all bytes first, then independently reload and recertify the child."""

    _require_expectations(expectations)
    if (
        not callable(model_loader)
        or not callable(native_certifier)
        or not callable(runtime_auditor)
    ):
        raise ExperimentalChildArtifactIntegrityError(
            "model_loader, native_certifier, and runtime_auditor must be callable."
        )
    hasher = (
        _default_dense_state_hasher
        if dense_state_hasher is None
        else dense_state_hasher
    )
    if not callable(hasher):
        raise ExperimentalChildArtifactIntegrityError(
            "dense_state_hasher must be callable."
        )
    verified = _verify_metadata(
        publication_dir,
        expectations=expectations,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    receipt_payload = verified.receipt.as_dict()
    tree_before_reload = artifact_tree(verified.artifact_path)
    evidence = _authenticate_fresh_reload(
        verified.artifact_path,
        report=dict(verified.bridge_report),
        expectations=expectations,
        model_loader=model_loader,
        native_certifier=native_certifier,
        runtime_auditor=runtime_auditor,
        dense_state_hasher=hasher,
    )
    tree_after_reload = artifact_tree(verified.artifact_path)
    if tree_after_reload != tree_before_reload:
        raise ExperimentalChildArtifactIntegrityError(
            "Fresh verification reload mutated the artifact tree."
        )
    if (
        evidence["model_state_fingerprint"]
        != receipt_payload["fresh_reload_model_state_fingerprint"]
        or evidence["native_certificate"]
        != receipt_payload["fresh_reload_native_certificate"]
        or evidence["runtime_audit"] != receipt_payload["fresh_reload_runtime_audit"]
        or evidence["bridge_state_sha256"]
        != receipt_payload["fresh_reload_bridge_state_sha256"]
    ):
        raise ExperimentalChildArtifactIntegrityError(
            "Independent fresh reload differs from the published evidence."
        )
    return verified


def _verify_metadata(
    publication_dir: str | os.PathLike[str],
    *,
    expectations: ExperimentalChildArtifactExpectations,
    expected_receipt_sha256: str,
) -> VerifiedExperimentalChildArtifact:
    root = _existing_publication_root(publication_dir)
    artifact_path = root / ARTIFACT_DIRECTORY_NAME
    report_path = root / BRIDGE_REPORT_FILENAME
    receipt_path = root / ARTIFACT_RECEIPT_FILENAME
    expected_entries = {
        ARTIFACT_DIRECTORY_NAME,
        BRIDGE_REPORT_FILENAME,
        ARTIFACT_RECEIPT_FILENAME,
    }
    try:
        entries = {path.name for path in root.iterdir()}
    except OSError as error:
        raise ExperimentalChildArtifactIntegrityError(
            "Cannot enumerate experimental child publication."
        ) from error
    if entries != expected_entries:
        raise ExperimentalChildArtifactIntegrityError(
            "Publication root contains missing or unapproved entries."
        )
    _assert_ordinary_directory(artifact_path, "child artifact")
    _assert_ordinary_file(report_path, "bridge report")
    _assert_ordinary_file(receipt_path, "artifact receipt")

    receipt_value = _read_json_object(receipt_path, "artifact receipt")
    receipt = ExperimentalChildArtifactReceipt.from_dict(receipt_value)
    if receipt_path.read_bytes() != _canonical_json_bytes(receipt.as_dict()) + b"\n":
        raise ExperimentalChildArtifactIntegrityError(
            "Artifact receipt file is not in its unique canonical encoding."
        )
    if receipt.receipt_sha256 != _sha256(
        expected_receipt_sha256, "expected_receipt_sha256"
    ):
        raise ExperimentalChildArtifactIntegrityError(
            "Artifact receipt hash does not match the external root."
        )
    payload = receipt.as_dict()
    _validate_receipt_claims(payload, expectations=expectations)

    report_bytes = report_path.read_bytes()
    authenticated_bytes, report = _authenticate_bridge_report_bytes(
        report_bytes,
        expectations=expectations,
    )
    report_record = payload["bridge_report"]
    if (
        report_record["path"] != BRIDGE_REPORT_FILENAME
        or report_record["file_sha256"]
        != hashlib.sha256(authenticated_bytes).hexdigest()
        or report_record["size_bytes"] != len(authenticated_bytes)
        or report_record["materialization_transaction_sha256"]
        != report["materialization_transaction_sha256"]
        or report_record["native_child_state_sha256"]
        != report["native_child_state_sha256"]
        or report_record["materialized_bridge_state_sha256"]
        != report["materialized_native_state_sha256"]
        or report_record["canonical_bytes_verified"] is not True
    ):
        raise ExperimentalChildArtifactIntegrityError(
            "Bridge report record does not match the canonical report bytes."
        )

    artifact_record = payload["artifact"]
    if (
        artifact_record.get("path") != ARTIFACT_DIRECTORY_NAME
        or artifact_record.get("strict_model_tokenizer_allowlist_verified") is not True
        or artifact_record.get("receipt_stored_outside_artifact_tree") is not True
        or artifact_record.get("config_payload_sha256")
        != expectations.config_payload_sha256
    ):
        raise ExperimentalChildArtifactIntegrityError(
            "Artifact receipt does not bind a clean standalone child."
        )
    fingerprint = payload["fresh_reload_model_state_fingerprint"]
    config_sha = _validate_hf_artifact_files(
        artifact_path,
        expected_state_keys=None,
        expected_state_key_sha256=fingerprint["state_key_sha256"],
    )
    actual_tree = artifact_tree(artifact_path)
    if (
        config_sha != expectations.config_payload_sha256
        or artifact_record.get("tree") != actual_tree
    ):
        raise ExperimentalChildArtifactIntegrityError(
            "Child artifact tree or config differs from the receipt."
        )
    return VerifiedExperimentalChildArtifact(
        publication_path=root,
        artifact_path=artifact_path,
        bridge_report=report,
        receipt=receipt,
    )


def _authenticate_fresh_reload(
    artifact_path: Path,
    *,
    report: Mapping[str, Any],
    expectations: ExperimentalChildArtifactExpectations,
    model_loader: Callable[[Path], Any],
    native_certifier: Callable[[Any], Mapping[str, Any] | Any],
    runtime_auditor: Callable[[Any, Path], Mapping[str, Any]],
    dense_state_hasher: Callable[[Any], str],
) -> dict[str, Any]:
    try:
        reloaded = model_loader(artifact_path)
    except BaseException as error:
        raise ExperimentalChildArtifactIntegrityError(
            "Fresh model reload failed."
        ) from error
    try:
        try:
            bridge_sha = _sha256(
                dense_state_hasher(reloaded), "fresh reload bridge state SHA"
            )
            if bridge_sha != report["materialized_native_state_sha256"]:
                raise ExperimentalChildArtifactIntegrityError(
                    "Freshly reloaded model does not match materialized bridge state."
                )
            fingerprint = _json_mapping(
                model_state_fingerprint(reloaded), "fresh reload model fingerprint"
            )
            _validate_fingerprint(fingerprint, expectations=expectations)
            certificate = _certificate_mapping(native_certifier(reloaded))
            report_certificate = _json_mapping(
                report["native_materialized"],
                "bridge native materialized certificate",
            )
            if certificate != report_certificate:
                raise ExperimentalChildArtifactIntegrityError(
                    "Fresh native certificate differs from the bridge materialization."
                )
            if _built_in_adapter_or_registry_count(reloaded) != 0:
                raise ExperimentalChildArtifactIntegrityError(
                    "Freshly reloaded child contains an adapter or residual registry."
                )
            audit = _json_mapping(
                runtime_auditor(reloaded, artifact_path), "fresh reload runtime audit"
            )
            if set(audit) != _AUDIT_COUNT_FIELDS:
                raise ExperimentalChildArtifactIntegrityError(
                    "Runtime audit does not contain exactly the required zero-count fields."
                )
            for name in _AUDIT_COUNT_FIELDS:
                value = audit[name]
                if isinstance(value, bool) or not isinstance(value, int) or value != 0:
                    raise ExperimentalChildArtifactIntegrityError(
                        f"Runtime audit must prove {name}=0."
                    )
            post_callback_bridge_sha = _sha256(
                dense_state_hasher(reloaded),
                "post-certificate fresh reload bridge state SHA",
            )
            post_callback_fingerprint = _json_mapping(
                model_state_fingerprint(reloaded),
                "post-certificate fresh reload model fingerprint",
            )
            if (
                post_callback_bridge_sha != bridge_sha
                or post_callback_fingerprint != fingerprint
                or _built_in_adapter_or_registry_count(reloaded) != 0
            ):
                raise ExperimentalChildArtifactIntegrityError(
                    "A fresh-reload certifier or auditor mutated the loaded child."
                )
            return {
                "bridge_state_sha256": bridge_sha,
                "model_state_fingerprint": fingerprint,
                "native_certificate": certificate,
                "runtime_audit": audit,
            }
        except ExperimentalChildArtifactError:
            raise
        except BaseException as error:
            raise ExperimentalChildArtifactIntegrityError(
                "Fresh model certification or runtime audit failed."
            ) from error
    finally:
        del reloaded


def _authenticate_bridge_report_bytes(
    value: Any,
    *,
    expectations: ExperimentalChildArtifactExpectations,
) -> tuple[bytes, dict[str, Any]]:
    if not isinstance(value, bytes) or not value or len(value) > _MAX_JSON_BYTES:
        raise ExperimentalChildArtifactIntegrityError(
            "Bridge materialization report must be nonempty canonical bytes."
        )
    try:
        report = _loads_json_object(value, "bridge materialization report")
        canonical = _canonical_json_bytes(report)
    except UnicodeError as error:
        raise ExperimentalChildArtifactIntegrityError(
            "Bridge materialization report is not canonical ASCII JSON."
        ) from error
    if value != canonical:
        raise ExperimentalChildArtifactIntegrityError(
            "Bridge materialization report bytes are not uniquely canonical."
        )
    body = dict(report)
    transaction_sha = _sha256(
        body.pop("materialization_transaction_sha256", None),
        "materialization_transaction_sha256",
    )
    if _canonical_sha256(body) != transaction_sha:
        raise ExperimentalChildArtifactIntegrityError(
            "Bridge materialization transaction self-hash does not verify."
        )
    contract = report.get("experimental_recovery_contract")
    if not isinstance(contract, Mapping):
        raise ExperimentalChildArtifactIntegrityError(
            "Bridge report omits its experimental recovery contract."
        )
    embedded_receipt = report.get("experimental_recovery_receipt")
    _verify_embedded_self_hash(embedded_receipt, "experimental recovery receipt")
    required_false = (
        "publication_authorized",
        "accepted_artifact",
        "sealed_data_accessed",
        "behavior_capability_proven",
    )
    if (
        report.get("schema")
        != EXACT_TARGET_EXPERIMENTAL_RECOVERY_MATERIALIZATION_SCHEMA
        or report.get("status") != "experimental-unaccepted"
        or report.get("exact_unique_parameters") != expectations.exact_unique_parameters
        or report.get("exact_budget_match") is not True
        or report.get("standalone_structure_materialized") is not True
        or report.get("serialization_only_context") is not True
        or report.get("serialized_state_must_match_materialized_state_sha256")
        is not True
        or report.get("fresh_reload_and_recovery_required") is not True
        or any(report.get(name) is not False for name in required_false)
        or not _is_exact_zero(
            report.get("materialized_registered_adapter_parameter_count")
        )
        or not _is_exact_zero(report.get("materialized_optimizer_parameter_count"))
        or not _is_exact_zero(
            report.get("materialized_registered_source_sized_sidecar_parameter_count")
        )
        or contract.get("recovery_ab_preregistration_sha256")
        != expectations.preregistration_sha256
        or contract.get("shared_execution_contract_sha256")
        != expectations.shared_execution_contract_sha256
        or contract.get("exact_unique_parameters")
        != expectations.exact_unique_parameters
        or contract.get("publication_authorized") is not False
        or contract.get("sealed_data_accessed") is not False
        or not isinstance(report.get("native_materialized"), Mapping)
    ):
        raise ExperimentalChildArtifactIntegrityError(
            "Bridge report violates the experimental materialization contract."
        )
    _sha256(report.get("native_child_state_sha256"), "native_child_state_sha256")
    _sha256(
        report.get("materialized_native_state_sha256"),
        "materialized_native_state_sha256",
    )
    return value, report


def _validate_receipt_claims(
    payload: Mapping[str, Any],
    *,
    expectations: ExperimentalChildArtifactExpectations,
) -> None:
    fingerprint = payload.get("fresh_reload_model_state_fingerprint")
    _validate_fingerprint(fingerprint, expectations=expectations)
    audit = payload.get("fresh_reload_runtime_audit")
    if not isinstance(audit, Mapping) or set(audit) != _AUDIT_COUNT_FIELDS:
        raise ExperimentalChildArtifactIntegrityError(
            "Receipt runtime audit is malformed."
        )
    bridge_record = payload.get("bridge_report")
    artifact_record = payload.get("artifact")
    if (
        not isinstance(bridge_record, Mapping)
        or set(bridge_record) != _BRIDGE_REPORT_RECORD_KEYS
        or not isinstance(artifact_record, Mapping)
        or set(artifact_record) != _ARTIFACT_RECORD_KEYS
    ):
        raise ExperimentalChildArtifactIntegrityError(
            "Receipt bridge or artifact record is malformed."
        )
    if (
        payload.get("schema") != EXPERIMENTAL_CHILD_ARTIFACT_SCHEMA
        or payload.get("status") != "experimental-unaccepted"
        or payload.get("arm_id") != expectations.arm_id
        or payload.get("preregistration_sha256") != expectations.preregistration_sha256
        or payload.get("shared_execution_contract_sha256")
        != expectations.shared_execution_contract_sha256
        or payload.get("fresh_reload_state_dict_sha256")
        != fingerprint["state_dict_sha256"]
        or payload.get("fresh_reload_bridge_state_sha256")
        != bridge_record.get("materialized_bridge_state_sha256")
        or payload.get("exact_unique_parameters")
        != expectations.exact_unique_parameters
        or payload.get("exact_logical_parameters")
        != expectations.exact_logical_parameters
        or payload.get("exact_budget_match") is not True
        or payload.get("fresh_reload_performed") is not True
        or payload.get("artifact_tree_unchanged_by_fresh_reload") is not True
        or payload.get("local_files_only") is not True
        or payload.get("trust_remote_code") is not False
        or not _is_exact_zero(payload.get("optimizer_steps"))
        or payload.get("sealed_data_accessed") is not False
        or payload.get("publication_authorized") is not False
        or payload.get("accepted_artifact") is not False
        or payload.get("behavior_capability_proven") is not False
        or any(not _is_exact_zero(payload.get(name)) for name in _AUDIT_COUNT_FIELDS)
        or any(not _is_exact_zero(audit.get(name)) for name in _AUDIT_COUNT_FIELDS)
    ):
        raise ExperimentalChildArtifactIntegrityError(
            "Artifact receipt claims do not match the external experiment contract."
        )
    _json_mapping(
        payload.get("fresh_reload_native_certificate"),
        "fresh reload native certificate",
    )


def _validate_fingerprint(
    value: Any,
    *,
    expectations: ExperimentalChildArtifactExpectations,
) -> None:
    if not isinstance(value, Mapping) or set(value) != _MODEL_STATE_FINGERPRINT_KEYS:
        raise ExperimentalChildArtifactIntegrityError(
            "Fresh reload model fingerprint is malformed."
        )
    if (
        value.get("schema") != "neural-forge-model-state-fingerprint-v1"
        or not isinstance(value.get("model_class"), str)
        or not value.get("model_class")
        or value.get("state_key_sha256") != expectations.state_key_sha256
        or value.get("parameter_topology_sha256")
        != expectations.parameter_topology_sha256
        or value.get("buffer_topology_sha256") != expectations.buffer_topology_sha256
        or value.get("unique_parameter_count") != expectations.exact_unique_parameters
        or value.get("logical_parameter_count") != expectations.exact_logical_parameters
        or isinstance(value.get("state_key_count"), bool)
        or not isinstance(value.get("state_key_count"), int)
        or value.get("state_key_count") <= 0
        or isinstance(value.get("logical_state_elements"), bool)
        or not isinstance(value.get("logical_state_elements"), int)
        or value.get("logical_state_elements") <= 0
        or value.get("all_state_tensors_finite") is not True
    ):
        raise ExperimentalChildArtifactIntegrityError(
            "Fresh reload state/count/topology differs from the expected blank target."
        )
    for name in (
        "state_dict_sha256",
        "compiler_state_sha256",
        "state_manifest_sha256",
        "state_key_sha256",
        "parameter_topology_sha256",
        "buffer_topology_sha256",
    ):
        _sha256(value.get(name), name)


def _validate_hf_artifact_files(
    root: Path,
    *,
    expected_state_keys: Sequence[str] | None,
    expected_state_key_sha256: str | None,
) -> str:
    if (expected_state_keys is None) == (expected_state_key_sha256 is None):
        raise ExperimentalChildArtifactIntegrityError(
            "Exactly one expected state-key binding is required."
        )
    _assert_ordinary_directory(root, "child artifact")
    files: dict[str, Path] = {}
    try:
        entries = list(root.iterdir())
    except OSError as error:
        raise ExperimentalChildArtifactIntegrityError(
            "Cannot enumerate child artifact."
        ) from error
    for path in entries:
        _assert_ordinary_file(path, f"child artifact {path.name!r}")
        if path.parent != root or path.name in files:
            raise ExperimentalChildArtifactIntegrityError(
                "Child artifact contains an unsafe or duplicate entry."
            )
        name = path.name
        if any(fragment in name.lower() for fragment in _FORBIDDEN_NAME_PARTS):
            raise ExperimentalChildArtifactIntegrityError(
                f"Child artifact contains forbidden sidecar-like file {name!r}."
            )
        if (
            name not in _MODEL_JSON_FILES
            and name not in _TOKENIZER_JSON_FILES
            and name not in _TOKENIZER_OTHER_FILES
            and name not in {"model.safetensors", "model.safetensors.index.json"}
            and _SAFETENSORS_SHARD.fullmatch(name) is None
        ):
            raise ExperimentalChildArtifactIntegrityError(
                f"Child artifact contains unapproved file {name!r}."
            )
        files[name] = path
    if "config.json" not in files:
        raise ExperimentalChildArtifactIntegrityError(
            "Child artifact omits config.json."
        )
    if "tokenizer_config.json" not in files or not (
        set(files) & _TOKENIZER_PAYLOAD_FILES
    ):
        raise ExperimentalChildArtifactIntegrityError(
            "Child artifact omits a tokenizer config or tokenizer payload."
        )
    for name in (_MODEL_JSON_FILES | _TOKENIZER_JSON_FILES) & set(files):
        _read_json_value(files[name], name)
    config = _read_json_object(files["config.json"], "config.json")
    _validate_safetensors_keys(
        files,
        expected_state_keys=expected_state_keys,
        expected_state_key_sha256=expected_state_key_sha256,
    )
    return _canonical_sha256(config)


def _validate_safetensors_keys(
    files: Mapping[str, Path],
    *,
    expected_state_keys: Sequence[str] | None,
    expected_state_key_sha256: str | None,
) -> None:
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - production dependency
        raise ExperimentalChildArtifactIntegrityError(
            "safetensors is required to authenticate child weights."
        ) from error
    index_path = files.get("model.safetensors.index.json")
    unsharded = files.get("model.safetensors")
    shard_names = {
        name for name in files if _SAFETENSORS_SHARD.fullmatch(name) is not None
    }
    actual_keys: list[str] = []
    try:
        if index_path is None:
            if unsharded is None or shard_names:
                raise ExperimentalChildArtifactIntegrityError(
                    "Unsharded child must contain exactly model.safetensors."
                )
            with safe_open(unsharded, framework="pt", device="cpu") as handle:
                actual_keys.extend(str(value) for value in handle.keys())
        else:
            if unsharded is not None:
                raise ExperimentalChildArtifactIntegrityError(
                    "Sharded child cannot also contain model.safetensors."
                )
            index = _read_json_object(index_path, "safetensors index")
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ExperimentalChildArtifactIntegrityError(
                    "Safetensors index has no weight_map."
                )
            if any(
                not isinstance(key, str)
                or not isinstance(value, str)
                or _SAFETENSORS_SHARD.fullmatch(value) is None
                for key, value in weight_map.items()
            ):
                raise ExperimentalChildArtifactIntegrityError(
                    "Safetensors index contains an invalid key or shard name."
                )
            referenced = set(weight_map.values())
            if referenced != shard_names:
                raise ExperimentalChildArtifactIntegrityError(
                    "Safetensors shards are missing or unreferenced."
                )
            for shard_name in sorted(referenced):
                with safe_open(
                    files[shard_name], framework="pt", device="cpu"
                ) as handle:
                    shard_keys = {str(value) for value in handle.keys()}
                mapped = {
                    str(key) for key, value in weight_map.items() if value == shard_name
                }
                if shard_keys != mapped:
                    raise ExperimentalChildArtifactIntegrityError(
                        f"Safetensors shard {shard_name!r} disagrees with its index."
                    )
                actual_keys.extend(sorted(shard_keys))
    except ExperimentalChildArtifactError:
        raise
    except Exception as error:
        raise ExperimentalChildArtifactIntegrityError(
            "Cannot authenticate child safetensors."
        ) from error
    if len(actual_keys) != len(set(actual_keys)):
        raise ExperimentalChildArtifactIntegrityError(
            "Child safetensors contain duplicate state keys."
        )
    ordered = tuple(sorted(actual_keys))
    if expected_state_keys is not None and ordered != tuple(
        sorted(str(value) for value in expected_state_keys)
    ):
        raise ExperimentalChildArtifactIntegrityError(
            "Child safetensors do not exactly cover the live model state keys."
        )
    if expected_state_key_sha256 is not None and _canonical_sha256(
        list(ordered)
    ) != _sha256(expected_state_key_sha256, "expected_state_key_sha256"):
        raise ExperimentalChildArtifactIntegrityError(
            "Child safetensors state keys differ from the receipt."
        )


def _verify_embedded_self_hash(value: Any, name: str) -> None:
    if not isinstance(value, Mapping) or "receipt_sha256" not in value:
        raise ExperimentalChildArtifactIntegrityError(
            f"Bridge {name} is missing its self-hash."
        )
    body = dict(value)
    claimed = _sha256(body.pop("receipt_sha256"), f"{name} SHA")
    if _canonical_sha256(body) != claimed:
        raise ExperimentalChildArtifactIntegrityError(
            f"Bridge {name} self-hash does not verify."
        )


def _certificate_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        method = getattr(value, "as_dict", None)
        if not callable(method):
            raise ExperimentalChildArtifactIntegrityError(
                "native_certifier must return a mapping or expose as_dict()."
            )
        value = method()
    return _json_mapping(value, "fresh reload native certificate")


def _built_in_adapter_or_registry_count(model: Any) -> int:
    modules = getattr(model, "modules", None)
    if not callable(modules):
        raise ExperimentalChildArtifactIntegrityError(
            "Freshly reloaded model does not expose modules()."
        )
    fragments = ("adapter", "residualprojection", "residualregistry")
    return sum(
        1
        for module in modules()
        if any(fragment in type(module).__name__.casefold() for fragment in fragments)
    )


def _default_model_saver(model: Any, destination: Path) -> None:
    method = getattr(model, "save_pretrained", None)
    if not callable(method):
        raise ExperimentalChildArtifactPublicationError(
            "model has no save_pretrained(); supply model_saver."
        )
    method(destination, safe_serialization=True)


def _default_tokenizer_saver(tokenizer: Any, destination: Path) -> None:
    method = getattr(tokenizer, "save_pretrained", None)
    if not callable(method):
        raise ExperimentalChildArtifactPublicationError(
            "tokenizer has no save_pretrained(); supply tokenizer_saver."
        )
    method(destination)


def _default_dense_state_hasher(model: Any) -> str:
    from neural_forge.hf_progressive_target_bridge import _dense_model_state_sha256

    return _dense_model_state_sha256(model)


def _publication_destination(value: str | os.PathLike[str]) -> Path:
    raw = Path(value).expanduser()
    if raw.exists() and _is_reparse_point(raw):
        raise ExperimentalChildArtifactPublicationError(
            "Publication destination cannot be a symlink or reparse point."
        )
    return raw.resolve(strict=False)


def _existing_publication_root(value: str | os.PathLike[str]) -> Path:
    source = Path(value).expanduser()
    try:
        root = source.resolve(strict=True)
    except FileNotFoundError as error:
        raise ExperimentalChildArtifactIntegrityError(
            "Experimental child publication is absent."
        ) from error
    _assert_ordinary_directory(root, "publication root")
    return root


def _is_reparse_point(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ExperimentalChildArtifactIntegrityError(
            f"Cannot inspect path {path}."
        ) from error
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(metadata, "st_file_attributes", 0)
    return bool(path.is_symlink() or (flag and attributes & flag))


def _assert_ordinary_directory(path: Path, name: str) -> None:
    if not path.is_dir() or _is_reparse_point(path):
        raise ExperimentalChildArtifactIntegrityError(
            f"{name} must be an ordinary directory."
        )


def _assert_ordinary_file(path: Path, name: str) -> None:
    if not path.is_file() or _is_reparse_point(path):
        raise ExperimentalChildArtifactIntegrityError(
            f"{name} must be an ordinary file."
        )
    try:
        links = int(path.stat().st_nlink)
    except OSError as error:
        raise ExperimentalChildArtifactIntegrityError(
            f"Cannot inspect {name}."
        ) from error
    if links != 1:
        raise ExperimentalChildArtifactIntegrityError(
            f"{name} must not be hard-linked."
        )


def _read_json_value(path: Path, name: str) -> Any:
    _assert_ordinary_file(path, name)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ExperimentalChildArtifactIntegrityError(f"Cannot read {name}.") from error
    return _loads_json(payload, name)


def _read_json_object(path: Path, name: str) -> dict[str, Any]:
    value = _read_json_value(path, name)
    if not isinstance(value, dict):
        raise ExperimentalChildArtifactIntegrityError(f"{name} must be a JSON object.")
    return value


def _loads_json_object(payload: bytes, name: str) -> dict[str, Any]:
    value = _loads_json(payload, name)
    if not isinstance(value, dict):
        raise ExperimentalChildArtifactIntegrityError(f"{name} must be a JSON object.")
    return value


def _loads_json(payload: bytes, name: str) -> Any:
    if not payload or len(payload) > _MAX_JSON_BYTES:
        raise ExperimentalChildArtifactIntegrityError(
            f"{name} is empty or exceeds the JSON size limit."
        )

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ExperimentalChildArtifactIntegrityError(
                    f"{name} contains duplicate JSON key {key!r}."
                )
            result[key] = value
        return result

    def invalid_constant(value: str) -> Any:
        raise ExperimentalChildArtifactIntegrityError(
            f"{name} contains non-finite JSON number {value}."
        )

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
    except ExperimentalChildArtifactError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExperimentalChildArtifactIntegrityError(
            f"{name} is not valid UTF-8 JSON."
        ) from error


def _json_mapping(value: Any, name: str) -> dict[str, Any]:
    normalized = _json_value(value, name)
    if not isinstance(normalized, dict):
        raise ExperimentalChildArtifactIntegrityError(f"{name} must be an object.")
    return normalized


def _json_value(value: Any, name: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        normalized = json.loads(encoded)
    except (TypeError, ValueError, UnicodeError) as error:
        raise ExperimentalChildArtifactIntegrityError(
            f"{name} must be finite canonical JSON."
        ) from error
    return normalized


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ExperimentalChildArtifactIntegrityError(
            "Value is not finite canonical JSON."
        ) from error


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ExperimentalChildArtifactIntegrityError(
            f"{name} must be a lowercase SHA-256 hex digest."
        )
    return value


def _is_exact_zero(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value == 0


def _require_expectations(value: Any) -> None:
    if not isinstance(value, ExperimentalChildArtifactExpectations):
        raise ExperimentalChildArtifactIntegrityError(
            "expectations must be ExperimentalChildArtifactExpectations."
        )


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
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
    for path in sorted(
        (value for value in root.rglob("*") if value.is_dir()),
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        _fsync_directory(path)
    _fsync_directory(root)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "ARTIFACT_DIRECTORY_NAME",
    "ARTIFACT_RECEIPT_FILENAME",
    "BRIDGE_REPORT_FILENAME",
    "EXACT_TARGET_EXPERIMENTAL_RECOVERY_MATERIALIZATION_SCHEMA",
    "EXPERIMENTAL_CHILD_ARTIFACT_SCHEMA",
    "ExperimentalChildArtifactError",
    "ExperimentalChildArtifactExpectations",
    "ExperimentalChildArtifactIntegrityError",
    "ExperimentalChildArtifactPublicationError",
    "ExperimentalChildArtifactReceipt",
    "VerifiedExperimentalChildArtifact",
    "publish_experimental_child_artifact",
    "verify_experimental_child_artifact",
]
