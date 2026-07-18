"""Bounded automatic candidate search above the dynamic folding compiler.

Family adapters register a deterministic portfolio once.  A search request then
tries exact-budget candidates without caller-side surgery, continues only after
candidate-specific rejection, stops on infrastructure failure, and returns the
first or highest-scoring freshly certified child.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from .dynamic_circuit_ir import (
    JsonScalar,
    ModelCircuitGraph,
    canonical_sha256,
)
from .dynamic_fold_artifact import (
    DynamicFoldArtifactCertificate,
    DynamicFoldArtifactError,
    authenticate_dynamic_fold_artifact_tree,
    load_dynamic_fold_artifact_certificate,
    remove_authenticated_dynamic_fold_artifact,
)
from .dynamic_fold_compiler import (
    DynamicFoldCompilation,
    DynamicFoldCompilerError,
    PublishedDynamicFoldCompilation,
    SynthesizedPublishedDynamicFoldCompilation,
    validate_disjoint_parent_models,
)
from .dynamic_fold_executor import (
    FoldDecision,
    fingerprint_model_state,
    run_fold_only_callback,
)
from .dynamic_fold_registry import ExperimentLedger, RecordStatus
from .dynamic_target_synthesis import (
    DynamicTargetRequest,
    DynamicTargetSynthesisError,
    DynamicTargetSynthesisResult,
    serialized_inference_state_inventory,
    validate_registered_inference_state,
    validate_structural_runtime_buffers,
    verify_compiler_authenticated_target,
)


SCHEMA = "neural-forge-dynamic-fold-search-v1"
_CANDIDATE_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")


class DynamicFoldSearchError(RuntimeError):
    """A portfolio, candidate, failure disposition, or result is invalid."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DynamicFoldSearchError(f"{name} must be a nonempty string.")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DynamicFoldSearchError(f"{name} must be a positive integer.")
    return value


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DynamicFoldSearchError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _nonzero_digest(value: Any, name: str) -> str:
    digest = _digest(value, name)
    if digest == "0" * 64:
        raise DynamicFoldSearchError(f"{name} must not be unset.")
    return digest


def _validate_evidence_derivation(
    *,
    scope: str,
    action: Any,
    child_state_fingerprint: str,
    evidence_fingerprint: str,
    evidence_root_fingerprint: str,
    payload_fingerprint: str | None,
    payload_integrity_mode: str | None,
    evidence_channels: Sequence[str],
    binding_fingerprint: str | None,
    derivation_manifest: Mapping[str, Any] | None,
    derivation_fingerprint: str | None,
    payload_integrity_reverified: bool,
    expected_root_fingerprint: str,
) -> None:
    payload = _nonzero_digest(
        payload_fingerprint,
        f"{scope} evidence payload fingerprint",
    )
    binding = _nonzero_digest(
        binding_fingerprint,
        f"{scope} evidence binding fingerprint",
    )
    derivation = _nonzero_digest(
        derivation_fingerprint,
        f"{scope} evidence derivation fingerprint",
    )
    if (
        payload_integrity_reverified is not True
        or not isinstance(payload_integrity_mode, str)
        or not payload_integrity_mode.strip()
        or payload_integrity_mode == "unrecorded"
        or evidence_root_fingerprint != expected_root_fingerprint
        or tuple(evidence_channels) != tuple(action.evidence_requirements)
        or not isinstance(derivation_manifest, Mapping)
    ):
        raise DynamicFoldSearchError(
            f"{scope} lacks compiler-owned, reverified evidence payload provenance."
        )
    manifest = dict(derivation_manifest)
    expected_manifest = {
        "schema": "neural-forge-fold-evidence-derivation-v1",
        "action_id": action.action_id,
        "action_sha256": action.action_sha256,
        "method_id": action.method_id,
        "child_state_fingerprint": child_state_fingerprint,
        "evidence_fingerprint": evidence_fingerprint,
        "evidence_root_fingerprint": expected_root_fingerprint,
        "evidence_payload_fingerprint": payload,
        "evidence_payload_integrity_mode": payload_integrity_mode,
        "evidence_channels": list(evidence_channels),
        "evidence_binding_fingerprint": binding,
    }
    if manifest != expected_manifest or canonical_sha256(manifest) != derivation:
        raise DynamicFoldSearchError(
            f"{scope} evidence derivation is not rooted in its current "
            "action, state, payload, and candidate evidence."
        )


def _scalar(value: Any, name: str) -> JsonScalar:
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise DynamicFoldSearchError(f"{name} must be a JSON scalar.")
    if isinstance(value, float) and not math.isfinite(value):
        raise DynamicFoldSearchError(f"{name} must be finite.")
    return value


def _metadata(
    value: Mapping[str, JsonScalar] | Sequence[tuple[str, JsonScalar]] | None,
) -> tuple[tuple[str, JsonScalar], ...]:
    if value is None:
        return ()
    items = value.items() if isinstance(value, Mapping) else value
    result = tuple(
        sorted(
            (
                _text(str(key), "metadata key"),
                _scalar(item, f"metadata {key}"),
            )
            for key, item in items
        )
    )
    if len({key for key, _item in result}) != len(result):
        raise DynamicFoldSearchError("Metadata keys must be unique.")
    return result


def _names(values: Any, name: str) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise DynamicFoldSearchError(f"{name} must be a tuple or list.")
    result = tuple(sorted({_text(value, name).lower() for value in values}))
    if not result:
        raise DynamicFoldSearchError(f"{name} must not be empty.")
    return result


class FailureDisposition(str, Enum):
    CANDIDATE_REJECTED = "candidate_rejected"
    INFRASTRUCTURE = "infrastructure"
    FATAL = "fatal"


@dataclass(frozen=True, slots=True)
class DynamicFoldSearchObjective:
    """One exact target plus bounded automatic-search policy."""

    target: DynamicTargetRequest
    maximum_attempts: int
    portfolio_id: str | None = None
    stop_after_first_accept: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.target, DynamicTargetRequest):
            raise DynamicFoldSearchError(
                "Search objective target must be DynamicTargetRequest."
            )
        object.__setattr__(
            self,
            "maximum_attempts",
            _positive_int(self.maximum_attempts, "maximum_attempts"),
        )
        if self.portfolio_id is not None:
            object.__setattr__(
                self,
                "portfolio_id",
                _text(self.portfolio_id, "portfolio_id"),
            )
        if not isinstance(self.stop_after_first_accept, bool):
            raise DynamicFoldSearchError("stop_after_first_accept must be boolean.")

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "objective",
            "target": self.target.as_dict(),
            "maximum_attempts": self.maximum_attempts,
            "portfolio_id": self.portfolio_id,
            "stop_after_first_accept": self.stop_after_first_accept,
        }


@dataclass(frozen=True, slots=True)
class DynamicFoldSearchCandidate:
    """One exact-budget configuration generated by a registered portfolio."""

    candidate_id: str
    target: DynamicTargetRequest
    evidence_fingerprint: str
    priority: int = 0
    metadata: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        candidate_id = _text(self.candidate_id, "candidate_id").lower()
        if _CANDIDATE_ID.fullmatch(candidate_id) is None or ".." in candidate_id:
            raise DynamicFoldSearchError(
                "candidate_id must be a safe lowercase path component."
            )
        object.__setattr__(self, "candidate_id", candidate_id)
        if not isinstance(self.target, DynamicTargetRequest):
            raise DynamicFoldSearchError(
                "Search candidate target must be DynamicTargetRequest."
            )
        object.__setattr__(
            self,
            "evidence_fingerprint",
            _digest(self.evidence_fingerprint, "evidence_fingerprint"),
        )
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise DynamicFoldSearchError("Candidate priority must be an integer.")
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "candidate",
            "candidate_id": self.candidate_id,
            "target": self.target.as_dict(),
            "evidence_fingerprint": self.evidence_fingerprint,
            "priority": self.priority,
            "metadata": dict(self.metadata),
        }


class DynamicFoldPortfolioAdapter(Protocol):
    """Family-owned deterministic search strategy registered once."""

    portfolio_id: str
    version: int
    priority: int
    target_families: tuple[str, ...]
    target_modalities: tuple[str, ...]

    def supports(self, objective: DynamicFoldSearchObjective) -> bool: ...

    def candidates(
        self,
        objective: DynamicFoldSearchObjective,
    ) -> Sequence[DynamicFoldSearchCandidate]: ...

    def classify_failure(
        self,
        candidate: DynamicFoldSearchCandidate,
        error: Exception,
    ) -> FailureDisposition | str: ...

    def score(
        self,
        candidate: DynamicFoldSearchCandidate,
        result: Any,
    ) -> float: ...


class DynamicFoldPortfolioRegistry:
    """Select one deterministic repair/tuning portfolio for an objective."""

    def __init__(
        self,
        adapters: Sequence[DynamicFoldPortfolioAdapter] = (),
    ) -> None:
        self._adapters: dict[str, DynamicFoldPortfolioAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: DynamicFoldPortfolioAdapter) -> None:
        portfolio_id = _text(
            getattr(adapter, "portfolio_id", None),
            "portfolio_id",
        )
        version = getattr(adapter, "version", None)
        priority = getattr(adapter, "priority", None)
        if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
            raise DynamicFoldSearchError(
                "Portfolio version must be a positive integer."
            )
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise DynamicFoldSearchError("Portfolio priority must be an integer.")
        _names(
            getattr(adapter, "target_families", None),
            "portfolio target_families",
        )
        _names(
            getattr(adapter, "target_modalities", None),
            "portfolio target_modalities",
        )
        for name in ("supports", "candidates", "classify_failure", "score"):
            if not callable(getattr(adapter, name, None)):
                raise DynamicFoldSearchError(
                    f"Portfolio adapter must implement {name}()."
                )
        if portfolio_id in self._adapters:
            raise DynamicFoldSearchError(
                f"Portfolio {portfolio_id!r} is already registered."
            )
        self._adapters[portfolio_id] = adapter

    @property
    def portfolio_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def select(
        self,
        objective: DynamicFoldSearchObjective,
    ) -> DynamicFoldPortfolioAdapter:
        if objective.portfolio_id is not None:
            try:
                adapter = self._adapters[objective.portfolio_id]
            except KeyError as exc:
                raise DynamicFoldSearchError(
                    f"Unknown requested portfolio {objective.portfolio_id!r}."
                ) from exc
            if not adapter.supports(objective):
                raise DynamicFoldSearchError(
                    f"Portfolio {objective.portfolio_id!r} does not support "
                    "the exact objective."
                )
            return adapter
        requested_family = objective.target.target_family
        requested_modalities = set(objective.target.target_modalities)
        candidates = []
        for adapter in self._adapters.values():
            families = set(_names(adapter.target_families, "portfolio target_families"))
            modalities = set(
                _names(adapter.target_modalities, "portfolio target_modalities")
            )
            if requested_family not in families and not (
                requested_family in {"any", "auto"} or "auto" in families
            ):
                continue
            if not requested_modalities.issubset(modalities):
                continue
            if adapter.supports(objective):
                candidates.append(adapter)
        if not candidates:
            raise DynamicFoldSearchError(
                "No registered portfolio supports the exact objective."
            )
        candidates.sort(
            key=lambda value: (-int(value.priority), str(value.portfolio_id))
        )
        return candidates[0]


@dataclass(frozen=True, slots=True)
class DynamicFoldSearchAttempt:
    candidate: DynamicFoldSearchCandidate
    destination: str
    status: str
    score: float | None = None
    result_fingerprint: str | None = None
    reason: str | None = None
    error_type: str | None = None
    artifact_retained: bool | None = None

    def __post_init__(self) -> None:
        if self.status not in {
            "accepted",
            FailureDisposition.CANDIDATE_REJECTED.value,
            FailureDisposition.INFRASTRUCTURE.value,
            FailureDisposition.FATAL.value,
        }:
            raise DynamicFoldSearchError("Search attempt status is invalid.")
        if self.score is not None and not math.isfinite(float(self.score)):
            raise DynamicFoldSearchError("Search attempt score must be finite.")
        if self.result_fingerprint is not None:
            _digest(self.result_fingerprint, "result_fingerprint")
        if self.status == "accepted":
            if not isinstance(self.artifact_retained, bool):
                raise DynamicFoldSearchError(
                    "Accepted search attempts must report artifact retention."
                )
        elif self.artifact_retained is not None:
            raise DynamicFoldSearchError(
                "Rejected search attempts cannot retain a certified artifact."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.as_dict(),
            "destination": self.destination,
            "status": self.status,
            "score": self.score,
            "result_fingerprint": self.result_fingerprint,
            "reason": self.reason,
            "error_type": self.error_type,
            "artifact_retained": self.artifact_retained,
        }


@dataclass(frozen=True, slots=True)
class DynamicFoldSearchResult:
    objective: DynamicFoldSearchObjective
    portfolio_id: str
    portfolio_version: int
    attempts: tuple[DynamicFoldSearchAttempt, ...]
    selected_candidate_id: str | None
    selected_destination: str | None
    selected_result: Any = None

    @property
    def complete(self) -> bool:
        return self.selected_candidate_id is not None

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "result",
            "objective": self.objective.as_dict(),
            "portfolio_id": self.portfolio_id,
            "portfolio_version": self.portfolio_version,
            "attempts": [attempt.as_dict() for attempt in self.attempts],
            "selected_candidate_id": self.selected_candidate_id,
            "selected_destination": self.selected_destination,
            "complete": self.complete,
        }


class DynamicFoldingRunner(Protocol):
    def run(
        self,
        *,
        parent_models: Mapping[str, Any],
        target_request: DynamicTargetRequest,
        evidence_fingerprint: str,
        destination: str | Path,
        parent_tokenizer_fingerprints: Mapping[str, str | None] | None = None,
    ) -> Any: ...


@dataclass(slots=True)
class AutomaticDynamicFoldSearchSystem:
    """Execute a registered bounded portfolio without per-candidate oversight."""

    folding_system: DynamicFoldingRunner
    portfolios: DynamicFoldPortfolioRegistry

    def __post_init__(self) -> None:
        if not callable(getattr(self.folding_system, "run", None)):
            raise DynamicFoldSearchError("folding_system must implement run().")
        if not isinstance(self.portfolios, DynamicFoldPortfolioRegistry):
            raise DynamicFoldSearchError(
                "portfolios must be DynamicFoldPortfolioRegistry."
            )

    def _ledger(self) -> ExperimentLedger | None:
        ledger = getattr(self.folding_system, "ledger", None)
        return ledger if isinstance(ledger, ExperimentLedger) else None

    @staticmethod
    def _accepted_record_hashes(result: Any) -> tuple[str, ...]:
        try:
            attempts = result.published.compilation.run.attempts
        except AttributeError:
            return ()
        return tuple(
            attempt.ledger_record_sha256 for attempt in attempts if attempt.accepted
        )

    def _retire_owned_artifact(
        self,
        *,
        root: Path,
        candidate: DynamicFoldSearchCandidate,
        destination: Path,
        artifact: DynamicFoldArtifactCertificate,
        owned_destinations: set[Path],
        record_sha256s: Sequence[str],
        reason: str,
    ) -> None:
        """Block ledger records before deletion, then finalize the rollback."""

        ledger = self._ledger()
        pending = ()
        if ledger is not None and record_sha256s:
            try:
                pending = ledger.prepare_supersession(
                    record_sha256s,
                    reason_class="search_artifact_rollback_pending",
                    reason=("Authenticated artifact removal is pending. " + reason),
                )
            except Exception as exc:
                raise DynamicFoldSearchError(
                    "Could not durably mark provisional records rollback-pending; "
                    "the certified artifact was retained."
                ) from exc
        self._prune_owned_superseded_artifact(
            root=root,
            candidate=candidate,
            destination=destination,
            artifact=artifact,
            owned_destinations=owned_destinations,
        )
        if not pending:
            return
        try:
            ledger.supersede_records(
                tuple(record.record_sha256 for record in pending),
                reason_class="search_artifact_rollback",
                reason=reason,
            )
        except Exception as exc:
            raise DynamicFoldSearchError(
                "Artifact removal completed, but ledger finalization failed; "
                "the provisional records remain safely rollback-pending."
            ) from exc

    @staticmethod
    def _validate_candidate_target(
        objective: DynamicFoldSearchObjective,
        candidate: DynamicFoldSearchCandidate,
    ) -> None:
        expected = objective.target
        observed = candidate.target
        for name in (
            "target_family",
            "target_modalities",
            "exact_unique_parameters",
            "maximum_parameter_bytes",
            "tokenizer_fingerprint",
        ):
            if getattr(observed, name) != getattr(expected, name):
                raise DynamicFoldSearchError(
                    f"Candidate {candidate.candidate_id!r} changes objective {name}."
                )
        if (
            expected.adapter_id is not None
            and observed.adapter_id != expected.adapter_id
        ):
            raise DynamicFoldSearchError(
                f"Candidate {candidate.candidate_id!r} changes objective adapter_id."
            )
        if (
            expected.fold_method_id is not None
            and observed.fold_method_id != expected.fold_method_id
        ):
            raise DynamicFoldSearchError(
                f"Candidate {candidate.candidate_id!r} changes objective "
                "fold_method_id."
            )
        declared_method = dict(candidate.metadata).get("method_id")
        if declared_method is not None and observed.fold_method_id != declared_method:
            raise DynamicFoldSearchError(
                f"Candidate {candidate.candidate_id!r} metadata and target "
                "fold methods differ."
            )

    @staticmethod
    def _read_matching_disk_certificate(
        destination: Path,
        artifact: DynamicFoldArtifactCertificate,
    ) -> None:
        certificate_path = destination / "certificate.json"
        if not certificate_path.is_file() or certificate_path.is_symlink():
            raise DynamicFoldSearchError(
                "Accepted result has no regular on-disk publication certificate."
            )
        try:
            disk_certificate = json.loads(certificate_path.read_bytes().decode("ascii"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DynamicFoldSearchError(
                "Accepted result publication certificate is unreadable."
            ) from exc
        if disk_certificate != artifact.as_dict():
            raise DynamicFoldSearchError(
                "In-memory artifact certificate differs from certificate.json."
            )

    @classmethod
    def _resolve_returned_artifact_ownership(
        cls,
        result: Any,
        destination: Path,
    ) -> tuple[DynamicFoldArtifactCertificate, Path] | None:
        """Resolve ownership only from this invocation's returned artifact."""

        if (
            not isinstance(
                result,
                SynthesizedPublishedDynamicFoldCompilation,
            )
            or not isinstance(
                result.published,
                PublishedDynamicFoldCompilation,
            )
            or not isinstance(
                result.published.artifact,
                DynamicFoldArtifactCertificate,
            )
        ):
            return None
        artifact = result.published.artifact
        try:
            if (
                not destination.is_dir()
                or destination.is_symlink()
                or artifact.certificate_sha256
                != canonical_sha256(artifact.body())
            ):
                return None
            resolved_destination = destination.resolve(strict=True)
            resolved_artifact_path = (
                Path(artifact.path).expanduser().resolve(strict=True)
            )
            if resolved_artifact_path != resolved_destination:
                return None
            cls._read_matching_disk_certificate(destination, artifact)
            loaded_artifact = load_dynamic_fold_artifact_certificate(
                destination
            )
            if loaded_artifact != artifact:
                return None
            authenticate_dynamic_fold_artifact_tree(artifact)
        except (DynamicFoldArtifactError, DynamicFoldSearchError, OSError):
            return None
        return artifact, resolved_destination

    @classmethod
    def _validate_published_result(
        cls,
        *,
        candidate: DynamicFoldSearchCandidate,
        result: Any,
        destination: Path,
        expected_parent_names: tuple[str, ...],
        expected_source_unique_parameters: int | None,
        expected_source_serialized_state_bytes: int | None,
        expected_source_runtime_state_bytes: int | None,
    ) -> tuple[DynamicFoldArtifactCertificate, Path]:
        if not isinstance(
            result,
            SynthesizedPublishedDynamicFoldCompilation,
        ):
            raise DynamicFoldSearchError(
                "Accepted result must be a synthesized published compilation."
            )
        if not isinstance(result.target, DynamicTargetSynthesisResult):
            raise DynamicFoldSearchError(
                "Accepted result has no authenticated target synthesis result."
            )
        if not isinstance(result.published, PublishedDynamicFoldCompilation):
            raise DynamicFoldSearchError(
                "Accepted result has no published compilation."
            )
        compilation = result.published.compilation
        artifact = result.published.artifact
        if not isinstance(compilation, DynamicFoldCompilation):
            raise DynamicFoldSearchError(
                "Accepted result has no dynamic fold compilation."
            )
        if not isinstance(artifact, DynamicFoldArtifactCertificate):
            raise DynamicFoldSearchError(
                "Accepted result has no real artifact certificate."
            )
        if not result.complete or not compilation.complete:
            raise DynamicFoldSearchError(
                "Accepted published compilation is incomplete."
            )

        target = result.target
        if not isinstance(target.graph, ModelCircuitGraph):
            raise DynamicFoldSearchError(
                "Published target graph is not a ModelCircuitGraph."
            )
        expected_request_fingerprint = candidate.target.fingerprint
        if target.request.fingerprint != expected_request_fingerprint:
            raise DynamicFoldSearchError(
                "Published target request differs from the exact candidate."
            )
        if (
            candidate.target.adapter_id is not None
            and target.adapter_id != candidate.target.adapter_id
        ):
            raise DynamicFoldSearchError(
                "Published target did not use the candidate's pinned adapter."
            )
        exact_parameters = candidate.target.exact_unique_parameters
        graph_modalities = tuple(
            sorted({value.modality for value in target.graph.interfaces})
        )
        if (
            target.graph.model_name != candidate.target.child_name
            or target.resource_profile.family != target.graph.family
            or target.resource_profile.modalities != graph_modalities
            or target.resource_profile.unique_parameters
            != target.graph.unique_parameter_count
            or target.resource_profile.parameter_bytes
            != target.graph.unique_parameter_bytes
            or target.resource_profile.layer_count != len(target.graph.layer_indices)
            or target.graph.tokenizer_fingerprint
            != candidate.target.tokenizer_fingerprint
        ):
            raise DynamicFoldSearchError(
                "Published target graph and resource profile are internally "
                "inconsistent."
            )
        if target.resource_profile.unique_parameters != exact_parameters:
            raise DynamicFoldSearchError(
                "Published target resource profile misses the exact budget."
            )
        if (
            target.resource_profile.serialized_state_elements
            != target.resource_profile.unique_parameters
            + target.resource_profile.persistent_buffer_elements
            or target.resource_profile.serialized_state_bytes
            != target.resource_profile.parameter_bytes
            + target.resource_profile.persistent_buffer_bytes
        ):
            raise DynamicFoldSearchError(
                "Published target serialized inference-state accounting is "
                "inconsistent with its parameters and persistent buffers."
            )
        expected_target_inventory = target.resource_profile.inference_state_inventory()
        policy = target.structural_buffer_policy
        expected_structural_audit = target.structural_buffer_audit
        if policy is None or expected_structural_audit is None:
            raise DynamicFoldSearchError(
                "Published target lacks compiler-authenticated structural-buffer "
                "policy evidence."
            )
        try:
            validate_registered_inference_state(target.model)
            observed_target_inventory = serialized_inference_state_inventory(
                target.model
            )
            observed_structural_audit = validate_structural_runtime_buffers(
                target.model,
                policy,
                expected_adapter_id=target.adapter_id,
                expected_adapter_version=target.adapter_version,
            )
        except Exception as exc:
            raise DynamicFoldSearchError(
                "Published target complete inference state could not be independently "
                "reauthenticated."
            ) from exc
        if observed_target_inventory != expected_target_inventory:
            raise DynamicFoldSearchError(
                "Published target resource profile differs from its live complete "
                "runtime tensor-state inventory."
            )
        if observed_structural_audit != expected_structural_audit:
            raise DynamicFoldSearchError(
                "Published target structural-buffer audit differs from its live "
                "complete inference state."
            )
        if (
            artifact.fresh_reload_state_inventory is None
            or dict(artifact.fresh_reload_state_inventory) != expected_target_inventory
        ):
            raise DynamicFoldSearchError(
                "Fresh-reload certificate state inventory differs from the "
                "authenticated target, including its nonpersistent buffers."
            )
        artifact_body = artifact.body()
        if (
            artifact_body.get(
                "fresh_reload_registered_inference_state_revalidated"
            )
            is not True
            or artifact_body.get(
                "fresh_reload_structural_buffer_policy_revalidated"
            )
            is not True
        ):
            raise DynamicFoldSearchError(
                "Fresh-reload certificate lacks exact target inference-state and "
                "structural-buffer revalidation."
            )
        if (
            expected_source_serialized_state_bytes is not None
            and target.resource_profile.serialized_state_bytes
            >= expected_source_serialized_state_bytes
        ):
            raise DynamicFoldSearchError(
                "Certified result does not prove positive serialized "
                "inference-state compaction against the live parent set."
            )
        if (
            expected_source_runtime_state_bytes is not None
            and target.resource_profile.runtime_state_bytes
            >= expected_source_runtime_state_bytes
        ):
            raise DynamicFoldSearchError(
                "Certified result does not prove positive complete runtime "
                "tensor-state compaction against the live parent set; "
                "nonpersistent buffers are included."
            )
        blank_provenance = target.blank_provenance
        if (
            blank_provenance is None
            or not blank_provenance.compiler_created_blank
            or not blank_provenance.parent_child_alias_free
            or not blank_provenance.zeroed_under_no_grad
            or not blank_provenance.all_parameters_zero
            or not blank_provenance.parameter_gradients_absent
            or blank_provenance.unique_parameter_element_count != exact_parameters
            or blank_provenance.unique_parameter_bytes
            != target.resource_profile.parameter_bytes
        ):
            raise DynamicFoldSearchError(
                "Published target lacks compiler-owned zero-blank provenance "
                "for the exact learned-parameter budget."
            )
        try:
            verify_compiler_authenticated_target(target)
        except DynamicTargetSynthesisError as exc:
            raise DynamicFoldSearchError(
                "Published target topology fingerprint was not derived by the "
                "compiler from independent graph/state evidence."
            ) from exc
        if (
            artifact.dynamic_target_provenance_present is not True
            or artifact.provenance_target_fingerprint != target.fingerprint
            or artifact.provenance_target_sha256 == "0" * 64
        ):
            raise DynamicFoldSearchError(
                "Artifact lacks the compiler-authenticated target synthesis "
                "provenance for this exact result."
            )
        if (
            candidate.target.target_family not in {"any", "auto"}
            and target.resource_profile.family != candidate.target.target_family
        ):
            raise DynamicFoldSearchError(
                "Published target resource profile changes the candidate family."
            )
        if set(target.resource_profile.modalities) != set(
            candidate.target.target_modalities
        ):
            raise DynamicFoldSearchError(
                "Published target resource profile changes candidate modalities."
            )
        if (
            candidate.target.maximum_parameter_bytes is not None
            and target.resource_profile.parameter_bytes
            > candidate.target.maximum_parameter_bytes
        ):
            raise DynamicFoldSearchError(
                "Published target exceeds the candidate byte budget."
            )
        if target.graph.unique_parameter_count != exact_parameters:
            raise DynamicFoldSearchError(
                "Published target graph misses the exact budget."
            )
        if compilation.plan.exact_target_unique_parameters != exact_parameters:
            raise DynamicFoldSearchError(
                "Published fold plan differs from the exact candidate budget."
            )
        if artifact.exact_unique_parameters != exact_parameters:
            raise DynamicFoldSearchError(
                "Fresh-reload artifact misses the exact candidate budget."
            )
        parent_names = tuple(graph.model_name for graph in compilation.parent_graphs)
        parent_fingerprints = tuple(
            graph.fingerprint for graph in compilation.parent_graphs
        )
        if (
            not compilation.parent_graphs
            or len(set(parent_names)) != len(parent_names)
            or len(set(parent_fingerprints)) != len(parent_fingerprints)
            or tuple(sorted(parent_names)) != expected_parent_names
        ):
            raise DynamicFoldSearchError(
                "Certified result has duplicate, missing, or unexpected "
                "parent graph identities."
            )
        declared_source_unique_parameters = sum(
            graph.unique_parameter_count for graph in compilation.parent_graphs
        )
        if (
            expected_source_unique_parameters is not None
            and declared_source_unique_parameters != expected_source_unique_parameters
        ):
            raise DynamicFoldSearchError(
                "Certified parent graph count differs from the live distinct "
                "parent parameter count."
            )
        if exact_parameters >= declared_source_unique_parameters:
            raise DynamicFoldSearchError(
                "Certified result does not prove positive learned-parameter "
                "compaction against its declared parent set."
            )
        compilation_parent_graphs = {
            graph.model_name: graph.fingerprint for graph in compilation.parent_graphs
        }
        if dict(target.parent_graph_fingerprints) != compilation_parent_graphs or tuple(
            compilation.plan.parent_graph_fingerprints
        ) != tuple(sorted(compilation_parent_graphs.values())):
            raise DynamicFoldSearchError(
                "Certified result parent graph bindings are inconsistent."
            )
        if (
            tuple(
                sorted(
                    name for name, _value in compilation.plan.parent_state_fingerprints
                )
            )
            != expected_parent_names
        ):
            raise DynamicFoldSearchError(
                "Fold plan parent-state names differ from the live parent set."
            )

        target_graph_fingerprint = target.graph.fingerprint
        if compilation.child_graph.fingerprint != target_graph_fingerprint:
            raise DynamicFoldSearchError(
                "Published compilation child identity differs from synthesis."
            )
        if (
            compilation.plan.child_graph_fingerprint
            != compilation.child_graph.fingerprint
            or compilation.run.plan_fingerprint != compilation.plan.fingerprint
        ):
            raise DynamicFoldSearchError(
                "Certified plan, run, and child graph bindings are inconsistent."
            )
        if (
            compilation.plan.evidence_fingerprint != candidate.evidence_fingerprint
            or artifact.evidence_fingerprint != candidate.evidence_fingerprint
        ):
            raise DynamicFoldSearchError(
                "Certified fold evidence differs from the search candidate."
            )
        if (
            target.blank_state_fingerprint != compilation.plan.child_state_fingerprint
            or compilation.run.initial_state_fingerprint
            != compilation.plan.child_state_fingerprint
            or compilation.run.final_state_fingerprint
            == compilation.run.initial_state_fingerprint
        ):
            raise DynamicFoldSearchError(
                "Certified result does not prove a changed pre-fold blank state."
            )
        if (
            compilation.run.bootstrap_report.get("changed_state_paths") != []
            or compilation.run.bootstrap_report.get("non_mutating_initializer_verified")
            is not True
        ):
            raise DynamicFoldSearchError(
                "Certified result does not prove a non-mutating initializer."
            )
        if (
            not compilation.plan.ready
            or not compilation.plan.actions
            or compilation.plan.gaps
        ):
            raise DynamicFoldSearchError(
                "Certified result lacks a nonempty ready fold plan."
            )
        planned_action_ids = tuple(
            action.action_id for action in compilation.plan.actions
        )
        attempted_action_ids = tuple(
            attempt.action_id for attempt in compilation.run.attempts
        )
        if (
            compilation.run.accepted_action_ids != planned_action_ids
            or attempted_action_ids != planned_action_ids
            or any(
                not attempt.decision.accepted for attempt in compilation.run.attempts
            )
            or not isinstance(compilation.run.final_decision, FoldDecision)
            or not compilation.run.final_decision.accepted
            or set(compilation.run.completed_target_parameter_paths)
            != set(compilation.child_graph.parameter_paths)
        ):
            raise DynamicFoldSearchError(
                "Certified run does not contain one accepted execution of "
                "every planned fold action and final behavior gate."
            )
        if (
            any(
                attempt.evidence_root_fingerprint != candidate.evidence_fingerprint
                for attempt in compilation.run.attempts
            )
            or compilation.run.final_evidence_root_fingerprint
            != candidate.evidence_fingerprint
            or compilation.run.final_evidence_fingerprint is None
        ):
            raise DynamicFoldSearchError(
                "Certified action/final evidence is not rooted in the "
                "candidate's sealed evidence."
            )
        action_by_id = {
            action.action_id: action for action in compilation.plan.actions
        }
        for attempt in compilation.run.attempts:
            action = action_by_id.get(attempt.action_id)
            if action is None:
                raise DynamicFoldSearchError(
                    "Certified attempt has no matching fold-plan action."
                )
            _validate_evidence_derivation(
                scope=f"Fold attempt {attempt.action_id!r}",
                action=action,
                child_state_fingerprint=attempt.incumbent_state_fingerprint,
                evidence_fingerprint=attempt.evidence_fingerprint,
                evidence_root_fingerprint=attempt.evidence_root_fingerprint,
                payload_fingerprint=attempt.evidence_payload_fingerprint,
                payload_integrity_mode=attempt.evidence_payload_integrity_mode,
                evidence_channels=attempt.evidence_channels,
                binding_fingerprint=attempt.evidence_binding_fingerprint,
                derivation_manifest=attempt.evidence_derivation_manifest,
                derivation_fingerprint=attempt.evidence_derivation_fingerprint,
                payload_integrity_reverified=(
                    attempt.evidence_payload_integrity_reverified
                ),
                expected_root_fingerprint=candidate.evidence_fingerprint,
            )
        _validate_evidence_derivation(
            scope="Final behavior gate",
            action=compilation.plan.actions[-1],
            child_state_fingerprint=compilation.run.final_state_fingerprint,
            evidence_fingerprint=compilation.run.final_evidence_fingerprint,
            evidence_root_fingerprint=(
                compilation.run.final_evidence_root_fingerprint
            ),
            payload_fingerprint=(
                compilation.run.final_evidence_payload_fingerprint
            ),
            payload_integrity_mode=(
                compilation.run.final_evidence_payload_integrity_mode
            ),
            evidence_channels=compilation.run.final_evidence_channels,
            binding_fingerprint=(
                compilation.run.final_evidence_binding_fingerprint
            ),
            derivation_manifest=(
                compilation.run.final_evidence_derivation_manifest
            ),
            derivation_fingerprint=(
                compilation.run.final_evidence_derivation_fingerprint
            ),
            payload_integrity_reverified=(
                compilation.run.final_evidence_payload_integrity_reverified
            ),
            expected_root_fingerprint=candidate.evidence_fingerprint,
        )
        expected_method_id = dict(candidate.metadata).get("method_id")
        executed_method_ids = {action.method_id for action in compilation.plan.actions}
        if expected_method_id is not None and (
            expected_method_id not in executed_method_ids
            or executed_method_ids - {expected_method_id, "copy_exact"}
        ):
            raise DynamicFoldSearchError(
                "Certified fold method differs from the catalog candidate."
            )
        if expected_method_id is not None and not any(
            attempt.method_id == expected_method_id
            and attempt.decision.accepted
            and attempt.candidate_state_fingerprint
            != attempt.incumbent_state_fingerprint
            and bool(attempt.transaction_report.get("changed_state_paths"))
            for attempt in compilation.run.attempts
        ):
            raise DynamicFoldSearchError(
                "The catalog-selected fold method did not cause an accepted "
                "child-state transition."
            )
        if artifact.child_graph_fingerprint != target_graph_fingerprint:
            raise DynamicFoldSearchError(
                "Fresh-reload child identity differs from synthesis."
            )
        if artifact.plan_fingerprint != compilation.plan.fingerprint:
            raise DynamicFoldSearchError(
                "Artifact certificate is bound to a different fold plan."
            )
        if artifact.run_fingerprint != compilation.run.fingerprint:
            raise DynamicFoldSearchError(
                "Artifact certificate is bound to a different fold run."
            )
        if artifact.child_state_fingerprint != compilation.run.final_state_fingerprint:
            raise DynamicFoldSearchError(
                "Fresh-reload child state identity differs from the accepted run."
            )
        if (
            not isinstance(artifact.fresh_reload_decision, FoldDecision)
            or not artifact.fresh_reload_decision.accepted
        ):
            raise DynamicFoldSearchError(
                "Artifact lacks an accepted fresh-reload behavior decision."
            )
        if (
            compilation.run.observed_parent_module_forward_calls != 0
            or compilation.run.observed_parent_root_forward_calls != 0
            or compilation.run.observed_final_child_module_forward_calls <= 0
            or compilation.run.observed_final_child_root_forward_calls <= 0
            or compilation.run.observed_final_child_inference_entrypoint_calls
            <= 0
            or compilation.run.observed_final_child_root_forward_calls
            > compilation.run.observed_final_child_module_forward_calls
            or artifact.fresh_reload_module_forward_calls <= 0
            or artifact.fresh_reload_root_forward_calls <= 0
            or artifact.fresh_reload_inference_entrypoint_calls <= 0
            or artifact.preparation_parent_module_forward_calls != 0
            or artifact.preparation_parent_root_forward_calls != 0
            or artifact.publication_parent_module_forward_calls != 0
            or artifact.publication_parent_root_forward_calls != 0
        ):
            raise DynamicFoldSearchError(
                "Production acceptance requires zero observed parent forwards "
                "and observed final/fresh-reload child execution."
            )
        if any(
            value != 0
            for value in (
                compilation.run.optimizer_constructions,
                compilation.run.optimizer_steps,
                compilation.run.backward_calls,
                compilation.run.autograd_grad_calls,
                artifact.compilation_optimizer_constructions,
                artifact.compilation_optimizer_steps,
                artifact.compilation_backward_calls,
                artifact.compilation_autograd_grad_calls,
                artifact.optimizer_constructions,
                artifact.optimizer_steps,
                artifact.backward_calls,
                artifact.autograd_grad_calls,
                artifact.preparation_optimizer_constructions,
                artifact.preparation_optimizer_steps,
                artifact.preparation_backward_calls,
                artifact.preparation_autograd_grad_calls,
                artifact.publication_optimizer_constructions,
                artifact.publication_optimizer_steps,
                artifact.publication_backward_calls,
                artifact.publication_autograd_grad_calls,
            )
        ):
            raise DynamicFoldSearchError(
                "Artifact certificate reports a training operation."
            )
        if (
            compilation.run.operator_implementation_fingerprint == "0" * 64
            or not compilation.run.operator_static_zero_training_audit_passed
        ):
            raise DynamicFoldSearchError(
                "Certified run lacks a source-bound static zero-training audit "
                "for its registered fold operators."
            )
        if (
            artifact.compilation_optimizer_constructions
            != compilation.run.optimizer_constructions
            or artifact.compilation_optimizer_steps != compilation.run.optimizer_steps
            or artifact.compilation_backward_calls != compilation.run.backward_calls
            or artifact.compilation_autograd_grad_calls
            != compilation.run.autograd_grad_calls
        ):
            raise DynamicFoldSearchError(
                "Artifact compilation guard counters differ from the fold run."
            )
        if (
            not artifact.preparation_guard_applied
            or not artifact.publication_guard_covers_save_load
            or not artifact.behavior_gate_id
            or artifact.behavior_gate_id == "unregistered-behavior-gate"
            or artifact.behavior_gate_version <= 0
            or artifact.behavior_baseline_fingerprint == "0" * 64
            or artifact.behavior_gate_manifest_fingerprint == "0" * 64
            or not artifact.behavior_gate_manifest
        ):
            raise DynamicFoldSearchError(
                "Artifact lacks a versioned, sealed behavior and full guard contract."
            )
        if (
            all(
                callable(getattr(target.model, name, None))
                for name in ("state_dict", "parameters", "modules")
            )
            and fingerprint_model_state(target.model)
            != compilation.run.final_state_fingerprint
        ):
            raise DynamicFoldSearchError(
                "Returned in-memory child differs from the certified fold state."
            )
        if artifact.certificate_sha256 != canonical_sha256(artifact.body()):
            raise DynamicFoldSearchError(
                "Artifact certificate SHA-256 does not authenticate its body."
            )

        if (
            not destination.is_dir()
            or destination.is_symlink()
            or not (destination / "child").is_dir()
            or (destination / "child").is_symlink()
        ):
            raise DynamicFoldSearchError(
                "Certified publication destination is not a standalone child."
            )
        resolved_destination = destination.resolve(strict=True)
        try:
            resolved_artifact_path = (
                Path(artifact.path).expanduser().resolve(strict=True)
            )
        except OSError as exc:
            raise DynamicFoldSearchError(
                "Artifact certificate path does not exist."
            ) from exc
        if resolved_artifact_path != resolved_destination:
            raise DynamicFoldSearchError(
                "Artifact certificate is bound to a different destination."
            )
        cls._read_matching_disk_certificate(destination, artifact)
        try:
            loaded_artifact = load_dynamic_fold_artifact_certificate(destination)
            if loaded_artifact != artifact:
                raise DynamicFoldArtifactError(
                    "Loaded artifact certificate differs from the accepted result."
                )
            authenticate_dynamic_fold_artifact_tree(artifact)
        except DynamicFoldArtifactError as exc:
            raise DynamicFoldSearchError(
                "Certified standalone child tree failed reauthentication."
            ) from exc
        return artifact, resolved_destination

    @classmethod
    def _prune_owned_superseded_artifact(
        cls,
        *,
        root: Path,
        candidate: DynamicFoldSearchCandidate,
        destination: Path,
        artifact: DynamicFoldArtifactCertificate,
        owned_destinations: set[Path],
    ) -> None:
        if destination.is_symlink() or not destination.is_dir():
            raise DynamicFoldSearchError(
                "Refusing to prune a non-directory or symbolic-link artifact."
            )
        resolved_root = root.resolve(strict=True)
        resolved_destination = destination.resolve(strict=True)
        if (
            resolved_destination not in owned_destinations
            or resolved_destination.parent != resolved_root
            or resolved_destination.name != candidate.candidate_id
        ):
            raise DynamicFoldSearchError(
                "Refusing to prune an artifact not owned by this search run."
            )
        cls._read_matching_disk_certificate(destination, artifact)
        try:
            removed = remove_authenticated_dynamic_fold_artifact(
                resolved_destination,
                expected_parent=resolved_root,
                expected_name=candidate.candidate_id,
            )
        except DynamicFoldArtifactError as exc:
            raise DynamicFoldSearchError(
                "Could not prune a superseded certified artifact."
            ) from exc
        if removed != artifact:
            raise DynamicFoldSearchError(
                "Removed artifact certificate differed from the retained result."
            )
        owned_destinations.remove(resolved_destination)

    def run(
        self,
        *,
        parent_models: Mapping[str, Any],
        objective: DynamicFoldSearchObjective,
        destination_root: str | Path,
        parent_tokenizer_fingerprints: Mapping[str, str | None] | None = None,
    ) -> DynamicFoldSearchResult:
        parents = dict(parent_models)
        if not parents:
            raise DynamicFoldSearchError(
                "Automatic search requires at least one parent model."
            )
        try:
            source_unique_parameters = validate_disjoint_parent_models(parents)
        except DynamicFoldCompilerError as exc:
            raise DynamicFoldSearchError(
                "Automatic search parent models alias or share learned storage."
            ) from exc
        source_serialized_state_bytes: int | None = 0
        source_runtime_state_bytes: int | None = 0
        try:
            for model in parents.values():
                if not callable(getattr(model, "state_dict", None)):
                    source_serialized_state_bytes = None
                    source_runtime_state_bytes = None
                    break
                inventory = serialized_inference_state_inventory(model)
                source_serialized_state_bytes += int(
                    inventory["serialized_state_bytes"]
                )
                source_runtime_state_bytes += int(inventory["runtime_state_bytes"])
        except Exception as exc:
            raise DynamicFoldSearchError(
                "Automatic search could not authenticate parent serialized "
                "inference-state size."
            ) from exc
        expected_parent_names = tuple(sorted(parents))
        portfolio = self.portfolios.select(objective)
        generated = tuple(portfolio.candidates(objective))
        if not generated:
            raise DynamicFoldSearchError("Selected portfolio returned no candidates.")
        if any(
            not isinstance(value, DynamicFoldSearchCandidate) for value in generated
        ):
            raise DynamicFoldSearchError("Portfolio returned a non-candidate value.")
        if len({value.candidate_id for value in generated}) != len(generated):
            raise DynamicFoldSearchError("Portfolio candidate IDs repeat.")
        if len({value.fingerprint for value in generated}) != len(generated):
            raise DynamicFoldSearchError("Portfolio candidates repeat exactly.")
        for candidate in generated:
            self._validate_candidate_target(objective, candidate)
        ordered = sorted(
            generated,
            key=lambda value: (-value.priority, value.candidate_id),
        )[: objective.maximum_attempts]
        root = Path(destination_root).expanduser().absolute()
        if root.exists() and (not root.is_dir() or root.is_symlink()):
            raise DynamicFoldSearchError(
                "destination_root must be a regular local directory."
            )
        root.mkdir(parents=True, exist_ok=True)
        root = root.resolve(strict=True)

        attempts: list[DynamicFoldSearchAttempt] = []
        best_accepted: (
            tuple[
                float,
                DynamicFoldSearchCandidate,
                Any,
                Path,
                DynamicFoldArtifactCertificate,
                tuple[str, ...],
            ]
            | None
        ) = None
        best_attempt_index: int | None = None
        owned_destinations: set[Path] = set()
        terminal_failure = False
        for candidate in ordered:
            destination = root / candidate.candidate_id
            current_artifact: DynamicFoldArtifactCertificate | None = None
            current_owned_destination: Path | None = None
            current_record_hashes: tuple[str, ...] = ()
            runner_returned_result = False
            ledger = self._ledger()
            ledger_record_count_before = 0 if ledger is None else len(ledger.records)
            if destination.exists():
                raise DynamicFoldSearchError(
                    f"Candidate destination already exists: {destination}."
                )
            try:
                result = self.folding_system.run(
                    parent_models=parents,
                    target_request=candidate.target,
                    evidence_fingerprint=candidate.evidence_fingerprint,
                    destination=destination,
                    parent_tokenizer_fingerprints=(parent_tokenizer_fingerprints),
                )
                runner_returned_result = True
                returned_ownership = self._resolve_returned_artifact_ownership(
                    result,
                    destination,
                )
                if returned_ownership is not None:
                    current_artifact, current_owned_destination = (
                        returned_ownership
                    )
                    owned_destinations.add(current_owned_destination)
                artifact, resolved_destination = self._validate_published_result(
                    candidate=candidate,
                    result=result,
                    destination=destination,
                    expected_parent_names=expected_parent_names,
                    expected_source_unique_parameters=(source_unique_parameters),
                    expected_source_serialized_state_bytes=(
                        source_serialized_state_bytes
                    ),
                    expected_source_runtime_state_bytes=(source_runtime_state_bytes),
                )
                owned_destinations.add(resolved_destination)
                current_artifact = artifact
                current_owned_destination = resolved_destination
                current_record_hashes = self._accepted_record_hashes(result)
                as_dict = getattr(result, "as_dict", None)
                if not callable(as_dict):
                    raise DynamicFoldSearchError(
                        "Accepted folding result must implement as_dict()."
                    )
                score_value, _score_guard = run_fold_only_callback(
                    lambda: portfolio.score(candidate, result),
                    phase="dynamic fold portfolio scoring",
                    protected_models=(
                        *parents.values(),
                        result.target.model,
                    ),
                    freeze_parameters=False,
                )
                score = float(score_value)
                if not math.isfinite(score):
                    raise DynamicFoldSearchError("Portfolio score must be finite.")
                result_fingerprint = canonical_sha256(as_dict())
                current = (
                    score,
                    candidate,
                    result,
                    destination,
                    artifact,
                    current_record_hashes,
                )
                if objective.stop_after_first_accept:
                    attempts.append(
                        DynamicFoldSearchAttempt(
                            candidate=candidate,
                            destination=str(destination),
                            status="accepted",
                            score=score,
                            result_fingerprint=result_fingerprint,
                            artifact_retained=True,
                        )
                    )
                    best_accepted = current
                    best_attempt_index = len(attempts) - 1
                    break
                if best_accepted is None:
                    attempts.append(
                        DynamicFoldSearchAttempt(
                            candidate=candidate,
                            destination=str(destination),
                            status="accepted",
                            score=score,
                            result_fingerprint=result_fingerprint,
                            artifact_retained=True,
                        )
                    )
                    best_accepted = current
                    best_attempt_index = len(attempts) - 1
                    continue

                (
                    best_score,
                    best_candidate,
                    _best_result,
                    best_path,
                    best_artifact,
                    best_record_hashes,
                ) = best_accepted
                current_is_better = score > best_score or (
                    score == best_score
                    and candidate.candidate_id < best_candidate.candidate_id
                )
                if current_is_better:
                    try:
                        self._retire_owned_artifact(
                            root=root,
                            candidate=best_candidate,
                            destination=best_path,
                            artifact=best_artifact,
                            owned_destinations=owned_destinations,
                            record_sha256s=best_record_hashes,
                            reason=(
                                "A higher-scoring certified candidate replaced "
                                "this provisional artifact."
                            ),
                        )
                    except DynamicFoldSearchError as prune_error:
                        if best_path.resolve(strict=False) not in owned_destinations:
                            if best_attempt_index is not None:
                                attempts[best_attempt_index] = replace(
                                    attempts[best_attempt_index],
                                    artifact_retained=False,
                                    reason=(
                                        "Artifact removal completed, but its "
                                        "ledger finalization remained pending."
                                    ),
                                )
                            best_accepted = None
                        try:
                            self._retire_owned_artifact(
                                root=root,
                                candidate=candidate,
                                destination=destination,
                                artifact=artifact,
                                owned_destinations=owned_destinations,
                                record_sha256s=current_record_hashes,
                                reason=(
                                    "The current candidate was rolled back after "
                                    "retiring the prior candidate failed."
                                ),
                            )
                            current_artifact = None
                            current_owned_destination = None
                        except DynamicFoldSearchError as cleanup_error:
                            raise DynamicFoldSearchError(
                                "Could not restore bounded artifact retention "
                                "after superseded-artifact pruning failed."
                            ) from cleanup_error
                        raise prune_error
                    if best_attempt_index is None:
                        raise DynamicFoldSearchError(
                            "Search lost its retained-attempt identity."
                        )
                    previous_attempt = attempts[best_attempt_index]
                    attempts[best_attempt_index] = replace(
                        previous_attempt,
                        artifact_retained=False,
                        reason=(
                            "Certified artifact pruned after a higher-scoring "
                            f"candidate was accepted: {candidate.candidate_id}."
                        ),
                    )
                    attempts.append(
                        DynamicFoldSearchAttempt(
                            candidate=candidate,
                            destination=str(destination),
                            status="accepted",
                            score=score,
                            result_fingerprint=result_fingerprint,
                            artifact_retained=True,
                        )
                    )
                    best_accepted = current
                    best_attempt_index = len(attempts) - 1
                else:
                    self._retire_owned_artifact(
                        root=root,
                        candidate=candidate,
                        destination=destination,
                        artifact=artifact,
                        owned_destinations=owned_destinations,
                        record_sha256s=current_record_hashes,
                        reason=(
                            "The bounded search pruned this lower-scoring "
                            "provisional artifact."
                        ),
                    )
                    current_artifact = None
                    current_owned_destination = None
                    attempts.append(
                        DynamicFoldSearchAttempt(
                            candidate=candidate,
                            destination=str(destination),
                            status="accepted",
                            score=score,
                            result_fingerprint=result_fingerprint,
                            reason=(
                                "Certified artifact pruned because the retained "
                                f"candidate scored higher: {best_candidate.candidate_id}."
                            ),
                            artifact_retained=False,
                        )
                    )
            except Exception as caught:
                error: Exception = caught
                rollback_hashes = current_record_hashes
                if not rollback_hashes and ledger is not None:
                    rollback_hashes = tuple(
                        record.record_sha256
                        for record in ledger.records[ledger_record_count_before:]
                        if record.status == RecordStatus.ACCEPTED.value
                    )
                if (
                    current_artifact is not None
                    and current_owned_destination is not None
                    and current_owned_destination in owned_destinations
                ):
                    try:
                        self._retire_owned_artifact(
                            root=root,
                            candidate=candidate,
                            destination=destination,
                            artifact=current_artifact,
                            owned_destinations=owned_destinations,
                            record_sha256s=rollback_hashes,
                            reason=(
                                "A post-publication search failure invalidated "
                                "the candidate artifact."
                            ),
                        )
                    except DynamicFoldSearchError as cleanup_error:
                        error = cleanup_error
                elif runner_returned_result and destination.exists():
                    unowned_error = DynamicFoldSearchError(
                        "A runner returned after creating a candidate "
                        "destination that could not be authenticated and "
                        "safely removed as this invocation's artifact."
                    )
                    unowned_error.__cause__ = error
                    error = unowned_error
                if isinstance(error, DynamicFoldSearchError):
                    disposition = FailureDisposition.FATAL
                else:
                    try:
                        disposition = FailureDisposition(
                            portfolio.classify_failure(candidate, error)
                        )
                    except (TypeError, ValueError) as classification_error:
                        raise DynamicFoldSearchError(
                            "Portfolio returned an invalid failure disposition."
                        ) from classification_error
                attempts.append(
                    DynamicFoldSearchAttempt(
                        candidate=candidate,
                        destination=str(destination),
                        status=disposition.value,
                        reason=str(error),
                        error_type=type(error).__name__,
                    )
                )
                if disposition is not FailureDisposition.CANDIDATE_REJECTED:
                    terminal_failure = True
                    break

        if terminal_failure and best_accepted is not None:
            (
                _best_score,
                best_candidate,
                _best_result,
                best_path,
                best_artifact,
                best_record_hashes,
            ) = best_accepted
            self._retire_owned_artifact(
                root=root,
                candidate=best_candidate,
                destination=best_path,
                artifact=best_artifact,
                owned_destinations=owned_destinations,
                record_sha256s=best_record_hashes,
                reason=(
                    "A later infrastructure or fatal failure invalidated the "
                    "provisional exhaustive-search result."
                ),
            )
            if best_attempt_index is None:
                raise DynamicFoldSearchError(
                    "Search lost its provisional accepted-attempt identity."
                )
            attempts[best_attempt_index] = replace(
                attempts[best_attempt_index],
                artifact_retained=False,
                reason=(
                    "Provisional artifact pruned because a later "
                    "infrastructure or fatal failure prevented exhaustive "
                    "completion."
                ),
            )
            best_accepted = None

        if best_accepted is None:
            return DynamicFoldSearchResult(
                objective=objective,
                portfolio_id=str(portfolio.portfolio_id),
                portfolio_version=int(portfolio.version),
                attempts=tuple(attempts),
                selected_candidate_id=None,
                selected_destination=None,
            )
        (
            _score,
            selected,
            selected_result,
            selected_destination,
            _artifact,
            selected_record_hashes,
        ) = best_accepted
        try:
            final_artifact, resolved_selected_destination = (
                self._validate_published_result(
                    candidate=selected,
                    result=selected_result,
                    destination=selected_destination,
                    expected_parent_names=expected_parent_names,
                    expected_source_unique_parameters=source_unique_parameters,
                    expected_source_serialized_state_bytes=(
                        source_serialized_state_bytes
                    ),
                    expected_source_runtime_state_bytes=(source_runtime_state_bytes),
                )
            )
            if (
                final_artifact != _artifact
                or owned_destinations != {resolved_selected_destination}
                or sum(attempt.artifact_retained is True for attempt in attempts) != 1
            ):
                raise DynamicFoldSearchError(
                    "Search did not finish with exactly one authenticated "
                    "retained artifact."
                )
        except BaseException:
            if selected_destination.resolve(strict=False) in owned_destinations:
                self._retire_owned_artifact(
                    root=root,
                    candidate=selected,
                    destination=selected_destination,
                    artifact=_artifact,
                    owned_destinations=owned_destinations,
                    record_sha256s=selected_record_hashes,
                    reason=(
                        "Final search-result authentication failed after "
                        "provisional publication."
                    ),
                )
            raise
        return DynamicFoldSearchResult(
            objective=objective,
            portfolio_id=str(portfolio.portfolio_id),
            portfolio_version=int(portfolio.version),
            attempts=tuple(attempts),
            selected_candidate_id=selected.candidate_id,
            selected_destination=str(selected_destination),
            selected_result=selected_result,
        )


__all__ = [
    "AutomaticDynamicFoldSearchSystem",
    "DynamicFoldPortfolioAdapter",
    "DynamicFoldPortfolioRegistry",
    "DynamicFoldSearchAttempt",
    "DynamicFoldSearchCandidate",
    "DynamicFoldSearchError",
    "DynamicFoldSearchObjective",
    "DynamicFoldSearchResult",
    "FailureDisposition",
    "SCHEMA",
]
