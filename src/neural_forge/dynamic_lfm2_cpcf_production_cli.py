"""Local-only production entrypoint for one reviewed exact LFM2 CPCF fold.

This module does not plan a new topology or create new parent evidence.  It
accepts every trusted resource-factory pin, reconstructs the reviewed target
plan, and submits that exact request to a one-entry production catalog.  The
catalog owns transactional blank folding, behavioral acceptance, publication,
and standalone reload.  This wrapper only seals the accepted result and its
artifact certificate into a durable, exclusive receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from .dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from .dynamic_fold_artifact import (
    DynamicFoldArtifactError,
    load_dynamic_fold_artifact_certificate,
    remove_authenticated_dynamic_fold_artifact,
)
from .dynamic_lfm2_cpcf_resources import (
    Lfm2CpcfProductionWiring,
    build_lfm2_cpcf_production_resources,
)
from .dynamic_lfm2_cpcf_reviewed_gate import (
    CanonicalJsonArtifact,
    Lfm2CpcfReviewedPromotionArtifacts,
)
from .dynamic_production_catalog import ProductionDynamicFoldCatalog
from .dynamic_search import DynamicFoldSearchObjective, DynamicFoldSearchResult


SCHEMA = "neural-forge-lfm2-cpcf-production-cli-receipt-v1"
SUMMARY_SCHEMA = "neural-forge-lfm2-cpcf-production-cli-summary-v1"
CANDIDATE_ID = "bridged-cpcf"

_REVIEWED_ARTIFACT_NAMES = (
    "manifest",
    "parent_envelope",
    "cpcf_run_receipt",
    "candidate_outputs",
    "capture_receipt",
    "automatic_report",
    "manual_packet",
    "parent_relative_manual_packet",
    "parent_relative_ratings",
    "parent_relative_report",
    "composite_promotion_report",
)

_FACTORY_DIGEST_PIN_NAMES = (
    "expected_full_reviewed_evidence_fingerprint",
    "expected_cpcf_run_receipt_sha256",
    "expected_cpcf_run_receipt_document_sha256",
    "expected_bridge_artifact_sha256",
    "expected_cache_key",
    "expected_cache_manifest_sha256",
    "expected_trajectory_corpus_sha256",
    "expected_parent_graph_fingerprint",
)


class Lfm2CpcfProductionCliError(RuntimeError):
    """The reviewed local production invocation failed closed."""


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CpcfProductionCliError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CpcfProductionCliError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2CpcfProductionCliError(
            f"{name} must be a nonnegative integer."
        )
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2CpcfProductionCliError(f"{name} must be nonempty text.")
    return value.strip()


def _raw_local_path(value: Any, name: str) -> str:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise Lfm2CpcfProductionCliError(f"{name} must be a local path.") from exc
    if not raw or "://" in raw:
        raise Lfm2CpcfProductionCliError(f"{name} must be a local path.")
    return raw


def _input_directory(value: Any, name: str) -> Path:
    raw = _raw_local_path(value, name)
    try:
        path = Path(raw).expanduser().resolve(strict=True)
    except OSError as exc:
        raise Lfm2CpcfProductionCliError(f"{name} does not exist.") from exc
    if not path.is_dir() or path.is_symlink():
        raise Lfm2CpcfProductionCliError(
            f"{name} must be a nonsymlink local directory."
        )
    return path


def _input_file(value: Any, name: str) -> Path:
    raw = _raw_local_path(value, name)
    try:
        path = Path(raw).expanduser().resolve(strict=True)
    except OSError as exc:
        raise Lfm2CpcfProductionCliError(f"{name} does not exist.") from exc
    if not path.is_file() or path.is_symlink():
        raise Lfm2CpcfProductionCliError(
            f"{name} must be a nonsymlink local file."
        )
    return path


def _output_path(value: Any, name: str) -> Path:
    raw = _raw_local_path(value, name)
    path = Path(raw).expanduser().resolve(strict=False)
    if path.exists() and path.is_symlink():
        raise Lfm2CpcfProductionCliError(f"{name} cannot be a symlink.")
    return path


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return whether either resolved path contains the other."""

    if _is_within(left, right) or _is_within(right, left):
        return True
    if _path_exists(left) and _path_exists(right):
        try:
            return os.path.samefile(left, right)
        except OSError:
            return False
    return False


def _path_exists(path: Path) -> bool:
    """Observe a path without treating a broken symlink as absent."""

    return path.exists() or path.is_symlink()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ReviewedArtifactPin:
    """One local canonical JSON document and its out-of-band document root."""

    path: Path
    document_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "path",
            _input_file(self.path, "reviewed artifact"),
        )
        object.__setattr__(
            self,
            "document_sha256",
            _digest(self.document_sha256, "reviewed artifact document_sha256"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "document_sha256": self.document_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfProductionCliConfig:
    """Complete trusted input and durable-output contract for one exact run."""

    parent_model_directory: Path
    donor_tokenizer_directory: Path
    target_tokenizer_directory: Path
    bridge_directory: Path
    trace_cache_directory: Path
    trajectory_corpus: Path
    reviewed_artifact_pins: Mapping[str, ReviewedArtifactPin]
    expected_full_reviewed_evidence_fingerprint: str
    expected_cpcf_run_receipt_sha256: str
    expected_cpcf_run_receipt_document_sha256: str
    expected_bridge_artifact_sha256: str
    expected_cache_key: str
    expected_cache_manifest_sha256: str
    expected_trajectory_corpus_sha256: str
    expected_parent_graph_fingerprint: str
    expected_exact_target_unique_parameters: int
    destination_root: Path
    ledger_path: Path
    receipt_path: Path
    device: str
    dtype: str
    parent_name: str = "lfm2-parent"
    child_name: str = "lfm2-cpcf-folded-child"
    blank_seed: int = 0
    minimum_hidden_size: int = 32

    def __post_init__(self) -> None:
        for name in (
            "parent_model_directory",
            "donor_tokenizer_directory",
            "target_tokenizer_directory",
            "bridge_directory",
            "trace_cache_directory",
        ):
            object.__setattr__(
                self,
                name,
                _input_directory(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "trajectory_corpus",
            _input_file(self.trajectory_corpus, "trajectory_corpus"),
        )
        pins = dict(self.reviewed_artifact_pins)
        if set(pins) != set(_REVIEWED_ARTIFACT_NAMES) or any(
            not isinstance(value, ReviewedArtifactPin)
            for value in pins.values()
        ):
            raise Lfm2CpcfProductionCliError(
                "reviewed_artifact_pins must contain every exact reviewed "
                "artifact and no unknown entries."
            )
        object.__setattr__(
            self,
            "reviewed_artifact_pins",
            MappingProxyType(
                {name: pins[name] for name in _REVIEWED_ARTIFACT_NAMES}
            ),
        )
        for name in _FACTORY_DIGEST_PIN_NAMES:
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "expected_exact_target_unique_parameters",
            _positive_int(
                self.expected_exact_target_unique_parameters,
                "expected_exact_target_unique_parameters",
            ),
        )
        for name in ("destination_root", "ledger_path", "receipt_path"):
            object.__setattr__(
                self,
                name,
                _output_path(getattr(self, name), name),
            )
        if self.destination_root.exists() and (
            not self.destination_root.is_dir()
            or self.destination_root.is_symlink()
        ):
            raise Lfm2CpcfProductionCliError(
                "destination_root must be a nonsymlink directory or be absent."
            )
        if self.ledger_path.exists() and (
            not self.ledger_path.is_file() or self.ledger_path.is_symlink()
        ):
            raise Lfm2CpcfProductionCliError(
                "ledger_path must be a nonsymlink regular file or be absent."
            )
        if self.receipt_path.exists():
            raise Lfm2CpcfProductionCliError(
                "receipt_path already exists; production receipts are immutable."
            )
        if _paths_overlap(self.ledger_path, self.receipt_path):
            raise Lfm2CpcfProductionCliError(
                "ledger_path and receipt_path must be disjoint."
            )
        candidate_destination = (
            self.destination_root / CANDIDATE_ID
        ).resolve(strict=False)
        for name in ("ledger_path", "receipt_path"):
            if _is_within(getattr(self, name), candidate_destination):
                raise Lfm2CpcfProductionCliError(
                    f"{name} cannot be inside the certified child artifact."
                )
        input_paths = (
            self.parent_model_directory,
            self.donor_tokenizer_directory,
            self.target_tokenizer_directory,
            self.bridge_directory,
            self.trace_cache_directory,
            self.trajectory_corpus,
            *(pin.path for pin in self.reviewed_artifact_pins.values()),
        )
        if any(
            _is_within(self.destination_root, path)
            or _is_within(path, self.destination_root)
            for path in input_paths
        ):
            raise Lfm2CpcfProductionCliError(
                "destination_root must be disjoint from every immutable input."
            )
        for output_name in ("ledger_path", "receipt_path"):
            output_path = getattr(self, output_name)
            for input_path in input_paths:
                if _paths_overlap(output_path, input_path):
                    raise Lfm2CpcfProductionCliError(
                        f"{output_name} must be disjoint from every "
                        "authenticated input file and directory."
                    )
            if _paths_overlap(output_path, candidate_destination):
                raise Lfm2CpcfProductionCliError(
                    f"{output_name} must be disjoint from the certified child "
                    "artifact."
                )
        for name in ("device", "dtype", "parent_name", "child_name"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise Lfm2CpcfProductionCliError(
                "dtype must be bfloat16, float16, or float32."
            )
        object.__setattr__(
            self,
            "blank_seed",
            _nonnegative_int(self.blank_seed, "blank_seed"),
        )
        object.__setattr__(
            self,
            "minimum_hidden_size",
            _positive_int(self.minimum_hidden_size, "minimum_hidden_size"),
        )

    def trusted_pins(self) -> dict[str, Any]:
        return {
            **{
                name: getattr(self, name)
                for name in _FACTORY_DIGEST_PIN_NAMES
            },
            "expected_exact_target_unique_parameters": (
                self.expected_exact_target_unique_parameters
            ),
            "reviewed_artifacts": {
                name: pin.as_dict()
                for name, pin in self.reviewed_artifact_pins.items()
            },
        }

    def identity(self) -> dict[str, Any]:
        return {
            "parent_model_directory": str(self.parent_model_directory),
            "donor_tokenizer_directory": str(
                self.donor_tokenizer_directory
            ),
            "target_tokenizer_directory": str(
                self.target_tokenizer_directory
            ),
            "bridge_directory": str(self.bridge_directory),
            "trace_cache_directory": str(self.trace_cache_directory),
            "trajectory_corpus": str(self.trajectory_corpus),
            "trusted_pins": self.trusted_pins(),
            "destination_root": str(self.destination_root),
            "ledger_path": str(self.ledger_path),
            "receipt_path": str(self.receipt_path),
            "device": self.device,
            "dtype": self.dtype,
            "parent_name": self.parent_name,
            "child_name": self.child_name,
            "blank_seed": self.blank_seed,
            "minimum_hidden_size": self.minimum_hidden_size,
            "maximum_attempts": 1,
        }


def _reviewed_artifacts(
    config: Lfm2CpcfProductionCliConfig,
) -> Lfm2CpcfReviewedPromotionArtifacts:
    return Lfm2CpcfReviewedPromotionArtifacts(
        **{
            name: CanonicalJsonArtifact(
                pin.path,
                pin.document_sha256,
            )
            for name, pin in config.reviewed_artifact_pins.items()
        }
    )


def _torch_device_and_dtype(config: Lfm2CpcfProductionCliConfig) -> tuple[Any, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise Lfm2CpcfProductionCliError(
            "PyTorch is required for the CPCF production entrypoint."
        ) from exc
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[config.dtype]
    try:
        device = torch.device(config.device)
    except (RuntimeError, TypeError) as exc:
        raise Lfm2CpcfProductionCliError("device is invalid.") from exc
    return device, dtype


def _exclusive_atomic_json(
    path: Path,
    document: Mapping[str, Any],
) -> tuple[str, str]:
    """Publish one canonical JSON file atomically without replacing a receipt."""

    destination = path.expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise Lfm2CpcfProductionCliError(
            "Refusing to replace an existing immutable production receipt."
        )
    payload = canonical_json_bytes(dict(document)) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise Lfm2CpcfProductionCliError(
                "A concurrent writer already published this receipt."
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    if destination.read_bytes() != payload:
        raise Lfm2CpcfProductionCliError(
            "Published receipt bytes failed immediate verification."
        )
    return canonical_sha256(dict(document)), _sha256_bytes(payload)


def _path_outside_artifact(
    path: Path,
    artifact_directory: Path,
    name: str,
) -> None:
    if _paths_overlap(path, artifact_directory):
        raise Lfm2CpcfProductionCliError(
            f"{name} must be disjoint from the certified child artifact."
        )


def _retained_candidate_observation(path: Path) -> dict[str, Any] | None:
    """Describe a residual candidate without following symlinks."""

    if not _path_exists(path):
        return None
    observation: dict[str, Any] = {
        "path": str(path),
        "is_symlink": path.is_symlink(),
        "is_directory": path.is_dir() and not path.is_symlink(),
        "is_regular_file": path.is_file() and not path.is_symlink(),
    }
    if observation["is_symlink"]:
        try:
            observation["symlink_target"] = os.readlink(path)
        except OSError:
            observation["symlink_target"] = None
    elif observation["is_regular_file"]:
        try:
            observation["file_bytes"] = path.stat().st_size
            observation["file_sha256"] = _file_sha256(path)
        except OSError:
            observation["file_bytes"] = None
            observation["file_sha256"] = None
    elif observation["is_directory"]:
        certificate_path = path / "certificate.json"
        certificate_is_regular = (
            certificate_path.is_file() and not certificate_path.is_symlink()
        )
        observation["certificate_is_regular_file"] = certificate_is_regular
        if certificate_is_regular:
            try:
                observation["certificate_file_bytes"] = (
                    certificate_path.stat().st_size
                )
                observation["certificate_file_sha256"] = _file_sha256(
                    certificate_path
                )
            except OSError:
                observation["certificate_file_bytes"] = None
                observation["certificate_file_sha256"] = None
    observation["observation_sha256"] = canonical_sha256(observation)
    return observation


def _returned_candidate_ownership_proof(
    result: Any,
    *,
    candidate_destination: Path,
    objective: DynamicFoldSearchObjective,
) -> dict[str, Any] | None:
    """Bind one canonical on-disk candidate to the facade's returned result."""

    if not isinstance(result, DynamicFoldSearchResult):
        return None
    try:
        if (
            result.selected_candidate_id != CANDIDATE_ID
            or result.selected_destination is None
            or result.selected_result is None
            or result.objective.fingerprint != objective.fingerprint
            or len(result.attempts) != 1
            or result.attempts[0].status != "accepted"
            or result.attempts[0].artifact_retained is not True
        ):
            return None
        selected_destination = Path(
            result.selected_destination
        ).expanduser().resolve(strict=True)
        attempt_destination = Path(
            result.attempts[0].destination
        ).expanduser().resolve(strict=True)
        selected = result.selected_result
        selected_document = selected.as_dict()
        selected_fingerprint = canonical_sha256(selected_document)
        artifact = selected.published.artifact
        artifact_document = artifact.as_dict()
        artifact_destination = Path(
            artifact.path
        ).expanduser().resolve(strict=True)
        certificate_path = candidate_destination / "certificate.json"
        if (
            selected_destination != candidate_destination
            or attempt_destination != candidate_destination
            or artifact_destination != candidate_destination
            or not candidate_destination.is_dir()
            or candidate_destination.is_symlink()
            or not certificate_path.is_file()
            or certificate_path.is_symlink()
            or result.attempts[0].result_fingerprint
            != selected_fingerprint
            or artifact.certificate_sha256
            != canonical_sha256(artifact.body())
        ):
            return None
        certificate_payload = certificate_path.read_bytes()
        disk_certificate = json.loads(certificate_payload.decode("ascii"))
        if (
            disk_certificate != artifact_document
            or certificate_payload
            != canonical_json_bytes(artifact_document) + b"\n"
        ):
            return None
        result_document = result.as_dict()
        result_fingerprint = canonical_sha256(result_document)
        if result.fingerprint != result_fingerprint:
            return None
    except (
        AttributeError,
        KeyError,
        OSError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return None
    proof = {
        "schema": (
            "neural-forge-lfm2-cpcf-returned-candidate-ownership-proof-v1"
        ),
        "candidate_destination": str(candidate_destination),
        "objective_fingerprint": objective.fingerprint,
        "search_result_fingerprint": result_fingerprint,
        "selected_compilation_fingerprint": selected_fingerprint,
        "artifact_certificate_sha256": artifact.certificate_sha256,
        "artifact_certificate_file_sha256": _sha256_bytes(
            certificate_payload
        ),
    }
    return {
        **proof,
        "ownership_proof_sha256": canonical_sha256(proof),
    }


def _load_candidate_bound_to_ownership_proof(
    candidate_destination: Path,
    ownership_proof: Mapping[str, Any],
) -> Any | None:
    """Return a complete-tree-authenticated certificate bound to this invocation."""

    proof = dict(ownership_proof)
    try:
        ownership_proof_sha256 = proof.pop("ownership_proof_sha256")
        certificate_path = candidate_destination / "certificate.json"
        if (
            ownership_proof_sha256 != canonical_sha256(proof)
            or proof.get("candidate_destination")
            != str(candidate_destination)
            or not candidate_destination.is_dir()
            or candidate_destination.is_symlink()
            or not certificate_path.is_file()
            or certificate_path.is_symlink()
            or _file_sha256(certificate_path)
            != proof.get("artifact_certificate_file_sha256")
        ):
            return None
        certificate = load_dynamic_fold_artifact_certificate(
            candidate_destination
        )
        certificate_destination = Path(certificate.path).expanduser().resolve(
            strict=True
        )
        if (
            certificate_destination != candidate_destination
            or certificate.certificate_sha256
            != proof.get("artifact_certificate_sha256")
            or certificate.certificate_sha256
            != canonical_sha256(certificate.body())
        ):
            return None
        return certificate
    except (
        DynamicFoldArtifactError,
        KeyError,
        OSError,
        TypeError,
    ):
        return None


def _candidate_matches_ownership_proof(
    candidate_destination: Path,
    ownership_proof: Mapping[str, Any],
) -> bool:
    """Re-authenticate the complete artifact tree against returned ownership."""

    return (
        _load_candidate_bound_to_ownership_proof(
            candidate_destination,
            ownership_proof,
        )
        is not None
    )


def _recover_owned_candidate_after_failure(
    candidate_destination: Path,
    ownership_proof: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Remove only a candidate authenticated by the facade's returned result."""

    observed = _path_exists(candidate_destination)
    owned_certificate = (
        None
        if ownership_proof is None
        else _load_candidate_bound_to_ownership_proof(
            candidate_destination,
            ownership_proof,
        )
    )
    ownership_authenticated = owned_certificate is not None
    cleanup_attempted = False
    cleanup_error: dict[str, str] | None = None
    cleanup_not_attempted_reason: str | None = None
    if observed and ownership_authenticated:
        cleanup_attempted = True
        try:
            removed = remove_authenticated_dynamic_fold_artifact(
                candidate_destination,
                expected_parent=candidate_destination.parent,
                expected_name=candidate_destination.name,
            )
            if (
                removed.certificate_sha256
                != owned_certificate.certificate_sha256
                or removed.as_dict() != owned_certificate.as_dict()
            ):
                raise Lfm2CpcfProductionCliError(
                    "Removed artifact certificate changed after ownership "
                    "authentication."
                )
        except BaseException as exc:
            cleanup_error = {
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
    elif observed:
        cleanup_not_attempted_reason = (
            "The current candidate path is not cryptographically bound to "
            "this invocation's returned result."
        )
    retained_observation = _retained_candidate_observation(
        candidate_destination
    )
    retained = retained_observation is not None
    recovery = {
        "candidate_destination": str(candidate_destination),
        "publication_observed": observed,
        "ownership_authenticated": ownership_authenticated,
        "ownership_proof": (
            None if ownership_proof is None else dict(ownership_proof)
        ),
        "cleanup_attempted": cleanup_attempted,
        "cleanup_succeeded": cleanup_attempted and not retained,
        "cleanup_not_attempted_reason": cleanup_not_attempted_reason,
        "artifact_retained": retained,
        "recovery_required": retained,
        "cleanup_error": cleanup_error,
        "retained_artifact_observation": retained_observation,
    }
    return {
        **recovery,
        "recovery_sha256": canonical_sha256(recovery),
    }


def _accepted_receipt_reports_candidate(
    receipt_path: Path,
    candidate_destination: Path,
) -> bool:
    """Authenticate an already-committed accepted receipt after an exception."""

    if (
        not receipt_path.is_file()
        or receipt_path.is_symlink()
    ):
        return False
    try:
        payload = receipt_path.read_bytes()
        document = json.loads(payload.decode("ascii"))
        if not isinstance(document, dict):
            return False
        body = dict(document)
        receipt_sha256 = body.pop("receipt_sha256")
        result = body.get("result")
        if not isinstance(result, dict):
            return False
        reported_destination = Path(
            _text(result.get("selected_destination"), "selected_destination")
        ).expanduser().resolve(strict=False)
    except (
        KeyError,
        OSError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        Lfm2CpcfProductionCliError,
    ):
        return False
    return (
        document.get("schema") == SCHEMA
        and document.get("status") == "accepted"
        and receipt_sha256 == canonical_sha256(body)
        and payload == canonical_json_bytes(document) + b"\n"
        and reported_destination == candidate_destination
    )


def _accepted_result_payload(
    result: DynamicFoldSearchResult,
    *,
    wiring: Lfm2CpcfProductionWiring,
    config: Lfm2CpcfProductionCliConfig,
) -> dict[str, Any]:
    if not isinstance(result, DynamicFoldSearchResult):
        raise Lfm2CpcfProductionCliError(
            "Production facade returned an unknown result type."
        )
    if (
        not result.complete
        or result.selected_result is None
        or result.selected_candidate_id is None
        or result.selected_destination is None
        or len(result.attempts) != 1
        or result.attempts[0].status != "accepted"
        or result.attempts[0].artifact_retained is not True
    ):
        raise Lfm2CpcfProductionCliError(
            "Production catalog did not retain exactly one accepted result."
        )
    selected = result.selected_result
    if not bool(getattr(selected, "complete", False)) or not callable(
        getattr(selected, "as_dict", None)
    ):
        raise Lfm2CpcfProductionCliError(
            "Selected result is not a complete published compilation."
        )
    try:
        artifact = selected.published.artifact
        artifact_document = artifact.as_dict()
        selected_target = selected.target
    except (AttributeError, TypeError) as exc:
        raise Lfm2CpcfProductionCliError(
            "Selected result lacks its target or artifact certificate."
        ) from exc
    try:
        destination = Path(result.selected_destination).expanduser().resolve(
            strict=True
        )
        artifact_directory = Path(artifact.path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise Lfm2CpcfProductionCliError(
            "Selected artifact destination is not a retained local directory."
        ) from exc
    expected_destination = (
        config.destination_root / CANDIDATE_ID
    ).resolve(strict=False)
    if (
        destination != artifact_directory
        or destination != expected_destination
        or not destination.is_dir()
        or destination.is_symlink()
        or artifact.certificate_sha256
        != canonical_sha256(artifact.body())
        or artifact.exact_unique_parameters
        != config.expected_exact_target_unique_parameters
        or selected_target.request.fingerprint
        != result.attempts[0].candidate.target.fingerprint
        or selected_target.resource_profile.unique_parameters
        != config.expected_exact_target_unique_parameters
        or result.objective.target.fingerprint
        != wiring.target_plan.target_request.fingerprint
        or dict(result.attempts[0].candidate.metadata).get(
            "target_plan_fingerprint"
        )
        != wiring.target_plan.fingerprint
    ):
        raise Lfm2CpcfProductionCliError(
            "Selected artifact does not bind the exact reviewed target plan, "
            "parameter count, destination, and certificate."
        )
    certificate_path = destination / "certificate.json"
    if not certificate_path.is_file() or certificate_path.is_symlink():
        raise Lfm2CpcfProductionCliError(
            "Selected artifact has no regular on-disk certificate."
        )
    try:
        disk_certificate = json.loads(
            certificate_path.read_bytes().decode("ascii")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CpcfProductionCliError(
            "Selected artifact certificate is unreadable."
        ) from exc
    if disk_certificate != artifact_document:
        raise Lfm2CpcfProductionCliError(
            "In-memory and on-disk artifact certificates differ."
        )
    try:
        authenticated_artifact = load_dynamic_fold_artifact_certificate(
            destination
        )
    except DynamicFoldArtifactError as exc:
        raise Lfm2CpcfProductionCliError(
            "Selected artifact failed complete-tree authentication."
        ) from exc
    if authenticated_artifact.as_dict() != artifact_document:
        raise Lfm2CpcfProductionCliError(
            "Freshly authenticated and returned artifact certificates differ."
        )
    _path_outside_artifact(config.ledger_path, destination, "ledger_path")
    _path_outside_artifact(config.receipt_path, destination, "receipt_path")
    if (
        not config.ledger_path.is_file()
        or config.ledger_path.is_symlink()
    ):
        raise Lfm2CpcfProductionCliError(
            "Accepted result has no retained durable ledger."
        )
    selected_document = selected.as_dict()
    selected_fingerprint = canonical_sha256(selected_document)
    if result.attempts[0].result_fingerprint != selected_fingerprint:
        raise Lfm2CpcfProductionCliError(
            "Accepted search attempt and selected compilation fingerprints differ."
        )
    return {
        "search_result": result.as_dict(),
        "search_result_fingerprint": result.fingerprint,
        "selected_compilation": selected_document,
        "selected_compilation_fingerprint": selected_fingerprint,
        "selected_artifact_certificate": artifact_document,
        "selected_artifact_certificate_sha256": (
            artifact.certificate_sha256
        ),
        "selected_artifact_certificate_file_sha256": (
            _file_sha256(certificate_path)
        ),
        "ledger": {
            "path": str(config.ledger_path),
            "bytes": config.ledger_path.stat().st_size,
            "sha256": _file_sha256(config.ledger_path),
        },
        "selected_destination": str(destination),
    }


def _receipt_body(
    *,
    config: Lfm2CpcfProductionCliConfig,
    status: str,
    result: Mapping[str, Any] | None = None,
    wiring: Lfm2CpcfProductionWiring | None = None,
    facade_fingerprint: str | None = None,
    objective: DynamicFoldSearchObjective | None = None,
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "configuration": config.identity(),
        "configuration_sha256": canonical_sha256(config.identity()),
        "trusted_pins": config.trusted_pins(),
        "wiring": None if wiring is None else wiring.as_dict(),
        "facade_fingerprint": facade_fingerprint,
        "objective": None if objective is None else objective.as_dict(),
        "target_plan": (
            None if wiring is None else wiring.target_plan.as_dict()
        ),
        "result": None if result is None else dict(result),
        "failure": None if failure is None else dict(failure),
        "local_resources_only": True,
        "cached_parent_evidence_only": True,
        "exact_reviewed_target_plan_only": True,
        "blank_target_materialized_by_compiler": True,
        "transactional_publication_and_reload_required": True,
        "compaction_dimension": "unique_parameter_count",
        "folding_only": True,
        "training_used": False,
        "pruning_used": False,
        "quantization_used": False,
        "network_allowed": False,
        "maximum_attempts": 1,
        "entrypoint_implementation_sha256": _file_sha256(
            Path(__file__).resolve(strict=True)
        ),
    }
    return body


def run_lfm2_cpcf_production_cli(
    config: Lfm2CpcfProductionCliConfig,
) -> dict[str, Any]:
    """Run the exact reviewed fold and write one immutable sealed receipt."""

    if not isinstance(config, Lfm2CpcfProductionCliConfig):
        raise Lfm2CpcfProductionCliError(
            "config must be Lfm2CpcfProductionCliConfig."
        )
    wiring: Lfm2CpcfProductionWiring | None = None
    objective: DynamicFoldSearchObjective | None = None
    facade_fingerprint: str | None = None
    candidate_destination = (
        config.destination_root / CANDIDATE_ID
    ).resolve(strict=False)
    ownership_proof: Mapping[str, Any] | None = None
    try:
        if _path_exists(candidate_destination):
            raise Lfm2CpcfProductionCliError(
                "Candidate destination already exists; this invocation will "
                "not adopt or delete an unowned artifact."
            )
        device, dtype = _torch_device_and_dtype(config)
        wiring = build_lfm2_cpcf_production_resources(
            parent_model_directory=config.parent_model_directory,
            donor_tokenizer_directory=config.donor_tokenizer_directory,
            target_tokenizer_directory=config.target_tokenizer_directory,
            bridge_directory=config.bridge_directory,
            trace_cache_directory=config.trace_cache_directory,
            trajectory_corpus=config.trajectory_corpus,
            reviewed_artifacts=_reviewed_artifacts(config),
            expected_full_reviewed_evidence_fingerprint=(
                config.expected_full_reviewed_evidence_fingerprint
            ),
            expected_cpcf_run_receipt_sha256=(
                config.expected_cpcf_run_receipt_sha256
            ),
            expected_cpcf_run_receipt_document_sha256=(
                config.expected_cpcf_run_receipt_document_sha256
            ),
            expected_bridge_artifact_sha256=(
                config.expected_bridge_artifact_sha256
            ),
            expected_cache_key=config.expected_cache_key,
            expected_cache_manifest_sha256=(
                config.expected_cache_manifest_sha256
            ),
            expected_trajectory_corpus_sha256=(
                config.expected_trajectory_corpus_sha256
            ),
            expected_parent_graph_fingerprint=(
                config.expected_parent_graph_fingerprint
            ),
            expected_exact_target_unique_parameters=(
                config.expected_exact_target_unique_parameters
            ),
            device=device,
            dtype=dtype,
            parent_name=config.parent_name,
            child_name=config.child_name,
            blank_seed=config.blank_seed,
            minimum_hidden_size=config.minimum_hidden_size,
        )
        if (
            wiring.target_plan.target_request.exact_unique_parameters
            != config.expected_exact_target_unique_parameters
        ):
            raise Lfm2CpcfProductionCliError(
                "Resource factory reconstructed a different target size."
            )
        candidate_specs = wiring.production_bundle.candidate_specs
        if (
            len(candidate_specs) != 1
            or candidate_specs[0].candidate_id != CANDIDATE_ID
        ):
            raise Lfm2CpcfProductionCliError(
                "Production bundle does not contain the one pinned CPCF "
                "candidate."
            )
        _path_outside_artifact(
            config.ledger_path,
            candidate_destination,
            "ledger_path",
        )
        _path_outside_artifact(
            config.receipt_path,
            candidate_destination,
            "receipt_path",
        )
        catalog = ProductionDynamicFoldCatalog(
            (wiring.production_bundle,),
            ledger_path=config.ledger_path,
            require_positive_compaction=True,
        )
        facade = catalog.assemble()
        facade_fingerprint = facade.fingerprint
        objective = DynamicFoldSearchObjective(
            target=wiring.target_plan.target_request,
            maximum_attempts=1,
            portfolio_id=wiring.production_bundle.portfolio_id,
            stop_after_first_accept=True,
        )
        if _path_exists(candidate_destination):
            raise Lfm2CpcfProductionCliError(
                "Candidate destination appeared before production execution; "
                "refusing to treat it as invocation-owned."
        )
        result = facade.run(
            parent_models={config.parent_name: wiring.parent_model},
            objective=objective,
            destination_root=config.destination_root,
            parent_tokenizer_fingerprints={
                config.parent_name: (
                    wiring.tokenizer_bridge.donor_tokenizer_semantics_sha256
                )
            },
        )
        ownership_proof = _returned_candidate_ownership_proof(
            result,
            candidate_destination=candidate_destination,
            objective=objective,
        )
        if ownership_proof is None:
            raise Lfm2CpcfProductionCliError(
                "Facade result did not cryptographically establish ownership "
                "of the candidate artifact."
            )
        accepted = _accepted_result_payload(
            result,
            wiring=wiring,
            config=config,
        )
        body = _receipt_body(
            config=config,
            status="accepted",
            result=accepted,
            wiring=wiring,
            facade_fingerprint=facade_fingerprint,
            objective=objective,
        )
        if not _candidate_matches_ownership_proof(
            candidate_destination,
            ownership_proof,
        ):
            raise Lfm2CpcfProductionCliError(
                "Candidate artifact tree changed before accepted receipt "
                "publication."
            )
        receipt = {**body, "receipt_sha256": canonical_sha256(body)}
        document_sha256, file_sha256 = _exclusive_atomic_json(
            config.receipt_path,
            receipt,
        )
        if document_sha256 != canonical_sha256(receipt):
            raise Lfm2CpcfProductionCliError(
                "Receipt document root changed during publication."
            )
        return {
            "schema": SUMMARY_SCHEMA,
            "status": "accepted",
            "receipt_path": str(config.receipt_path),
            "receipt_sha256": receipt["receipt_sha256"],
            "receipt_document_sha256": document_sha256,
            "receipt_file_sha256": file_sha256,
            "selected_destination": accepted["selected_destination"],
            "selected_artifact_certificate_sha256": (
                accepted["selected_artifact_certificate_sha256"]
            ),
            "exact_target_unique_parameters": (
                config.expected_exact_target_unique_parameters
            ),
            "parent_unique_parameters": (
                wiring.parent_graph.unique_parameter_count
            ),
            "parameter_compression_ratio": (
                wiring.parent_graph.unique_parameter_count
                / config.expected_exact_target_unique_parameters
            ),
            "compaction_dimension": "unique_parameter_count",
            "training_used": False,
            "pruning_used": False,
            "quantization_used": False,
            "network_allowed": False,
        }
    except BaseException as exc:
        accepted_receipt_committed = _accepted_receipt_reports_candidate(
            config.receipt_path,
            candidate_destination,
        )
        recovery: Mapping[str, Any] | None = None
        if (
            not accepted_receipt_committed
            and _path_exists(candidate_destination)
        ):
            recovery = _recover_owned_candidate_after_failure(
                candidate_destination,
                ownership_proof,
            )
        if not config.receipt_path.exists():
            failure_body = _receipt_body(
                config=config,
                status="failed_closed",
                wiring=wiring,
                facade_fingerprint=facade_fingerprint,
                objective=objective,
                failure={
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "accepted_artifact_claimed": False,
                    "artifact_recovery": recovery,
                },
            )
            failure_receipt = {
                **failure_body,
                "receipt_sha256": canonical_sha256(failure_body),
            }
            _exclusive_atomic_json(config.receipt_path, failure_receipt)
        if isinstance(exc, Lfm2CpcfProductionCliError):
            raise
        if not isinstance(exc, Exception):
            raise
        raise Lfm2CpcfProductionCliError(
            f"Reviewed production fold failed closed: {type(exc).__name__}: "
            f"{exc}"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m neural_forge.dynamic_lfm2_cpcf_production_cli",
        description=(
            "Run one independently pinned, reviewed, local-only LFM2 CPCF "
            "production fold."
        ),
    )
    for name in (
        "parent-model-directory",
        "donor-tokenizer-directory",
        "target-tokenizer-directory",
        "bridge-directory",
        "trace-cache-directory",
        "trajectory-corpus",
        "destination-root",
        "ledger-path",
        "receipt-path",
    ):
        parser.add_argument(f"--{name}", required=True)
    for name in _REVIEWED_ARTIFACT_NAMES:
        flag = name.replace("_", "-")
        parser.add_argument(f"--reviewed-{flag}", required=True)
        parser.add_argument(
            f"--reviewed-{flag}-document-sha256",
            required=True,
        )
    for name in _FACTORY_DIGEST_PIN_NAMES:
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument(
        "--expected-exact-target-unique-parameters",
        type=int,
        required=True,
    )
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        required=True,
    )
    parser.add_argument("--parent-name", default="lfm2-parent")
    parser.add_argument("--child-name", default="lfm2-cpcf-folded-child")
    parser.add_argument("--blank-seed", type=int, default=0)
    parser.add_argument("--minimum-hidden-size", type=int, default=32)
    return parser


def config_from_namespace(
    args: argparse.Namespace,
) -> Lfm2CpcfProductionCliConfig:
    artifact_pins = {
        name: ReviewedArtifactPin(
            path=Path(getattr(args, f"reviewed_{name}")),
            document_sha256=getattr(
                args,
                f"reviewed_{name}_document_sha256",
            ),
        )
        for name in _REVIEWED_ARTIFACT_NAMES
    }
    values = {
        name: getattr(args, name) for name in _FACTORY_DIGEST_PIN_NAMES
    }
    return Lfm2CpcfProductionCliConfig(
        parent_model_directory=Path(args.parent_model_directory),
        donor_tokenizer_directory=Path(args.donor_tokenizer_directory),
        target_tokenizer_directory=Path(args.target_tokenizer_directory),
        bridge_directory=Path(args.bridge_directory),
        trace_cache_directory=Path(args.trace_cache_directory),
        trajectory_corpus=Path(args.trajectory_corpus),
        reviewed_artifact_pins=artifact_pins,
        **values,
        expected_exact_target_unique_parameters=(
            args.expected_exact_target_unique_parameters
        ),
        destination_root=Path(args.destination_root),
        ledger_path=Path(args.ledger_path),
        receipt_path=Path(args.receipt_path),
        device=args.device,
        dtype=args.dtype,
        parent_name=args.parent_name,
        child_name=args.child_name,
        blank_seed=args.blank_seed,
        minimum_hidden_size=args.minimum_hidden_size,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        config = config_from_namespace(parser.parse_args(argv))
        summary = run_lfm2_cpcf_production_cli(config)
    except Lfm2CpcfProductionCliError as exc:
        error = {
            "schema": SUMMARY_SCHEMA,
            "status": "failed_closed",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        sys.stderr.buffer.write(canonical_json_bytes(error) + b"\n")
        return 1
    sys.stdout.buffer.write(canonical_json_bytes(summary) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_ID",
    "Lfm2CpcfProductionCliConfig",
    "Lfm2CpcfProductionCliError",
    "ReviewedArtifactPin",
    "SCHEMA",
    "SUMMARY_SCHEMA",
    "build_parser",
    "config_from_namespace",
    "main",
    "run_lfm2_cpcf_production_cli",
]
