"""Method registry and durable novelty ledger for automatic folding.

The registry is deliberately scientific rather than cosmetic: a new runner
name, seed, output path, or reimplementation does not reopen a method whose
semantic lane has already failed.  Exact candidates can still be retried after
the child state or evidence epoch materially changes, while curated lane/family
closures prevent the project from cycling through settled experiments.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .dynamic_circuit_ir import JsonScalar, canonical_json_bytes, canonical_sha256


LEDGER_SCHEMA = "neural-forge-dynamic-fold-ledger-v1"
METHOD_SCHEMA = "neural-forge-dynamic-fold-method-v1"


_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.RLock] = {}


def _path_lock(path: Path) -> threading.RLock:
    key = os.path.normcase(str(path.expanduser().absolute()))
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _exclusive_ledger_writer(path: Path):
    """Serialize the read/compare/replace transaction across processes.

    The durable ledger has one writer *at a time*.  The advisory sidecar is
    intentionally persistent; its contents are not authority and a crashed
    writer cannot leave a stale ownership claim behind.
    """

    lock_path = path.with_name(f".{path.name}.writer.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    process_lock = _path_lock(lock_path)
    with process_lock, lock_path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _fsync_directory(path: Path) -> None:
    """Persist a rename on platforms that support directory fsync."""

    directory_flag = getattr(os, "O_DIRECTORY", None)
    if directory_flag is None:
        return
    try:
        descriptor = os.open(path, os.O_RDONLY | directory_flag)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_records_atomic(
    path: Path,
    records: Sequence[LedgerRecord],
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for record in records:
                handle.write(canonical_json_bytes(record.as_dict()))
                handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


class DynamicFoldRegistryError(ValueError):
    """A method, signature, closure, or ledger record is invalid."""


class MethodStatus(str, Enum):
    ACTIVE = "active"
    CONTROL = "control"
    EXPERIMENTAL = "experimental"
    CLOSED = "closed"
    TRAINING_ONLY = "training_only"
    DEFERRED = "deferred"


class ClosureLevel(str, Enum):
    EXACT = "exact"
    CONFIGURATION = "configuration"
    LANE = "lane"
    FAMILY = "family"


class RecordStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    CLOSED_UNCHANGED = "closed_unchanged"
    SUPERSESSION_PENDING = "supersession_pending"
    SUPERSEDED = "superseded"
    FAILED_INFRASTRUCTURE = "failed_infrastructure"


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DynamicFoldRegistryError(f"{name} must be a nonempty string.")
    return value.strip()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DynamicFoldRegistryError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _scalar(value: Any, name: str) -> JsonScalar:
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise DynamicFoldRegistryError(f"{name} must be a JSON scalar.")
    if isinstance(value, float) and not math.isfinite(value):
        raise DynamicFoldRegistryError(f"{name} must be finite.")
    return value


def _metadata(
    value: Mapping[str, JsonScalar] | Iterable[tuple[str, JsonScalar]] | None,
    name: str,
) -> tuple[tuple[str, JsonScalar], ...]:
    if value is None:
        return ()
    items = value.items() if isinstance(value, Mapping) else value
    result = []
    for key, item in items:
        key = _text(key, f"{name} key")
        result.append((key, _scalar(item, f"{name}.{key}")))
    result.sort(key=lambda pair: pair[0])
    if len({key for key, _ in result}) != len(result):
        raise DynamicFoldRegistryError(f"{name} keys must be unique.")
    return tuple(result)


def _strings(values: Iterable[Any], name: str) -> tuple[str, ...]:
    result = tuple(sorted({_text(value, name) for value in values}))
    return result


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DynamicFoldRegistryError(
                f"Ledger JSON contains duplicate key {key!r}."
            )
        result[key] = value
    return result


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise DynamicFoldRegistryError(
            f"{name} keys differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}."
        )


@dataclass(frozen=True, slots=True)
class FoldMethodSpec:
    """One semantically distinct fold mechanism."""

    method_id: str
    version: int
    family_id: str
    status: MethodStatus | str
    source_kinds: tuple[str, ...]
    target_kinds: tuple[str, ...]
    evidence_channels: tuple[str, ...]
    coupled_roles: tuple[str, ...]
    zero_gradient: bool
    standalone: bool
    reason: str
    reopen_requires: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "method_id", _text(self.method_id, "method_id"))
        object.__setattr__(self, "family_id", _text(self.family_id, "family_id"))
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, int)
            or (self.version <= 0)
        ):
            raise DynamicFoldRegistryError("method version must be positive.")
        status = (
            self.status.value if isinstance(self.status, MethodStatus) else self.status
        )
        try:
            status = MethodStatus(status)
        except ValueError as exc:
            raise DynamicFoldRegistryError(
                f"Unknown method status {status!r}."
            ) from exc
        object.__setattr__(self, "status", status.value)
        for name in (
            "source_kinds",
            "target_kinds",
            "evidence_channels",
            "coupled_roles",
            "reopen_requires",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        for name in (
            "source_kinds",
            "target_kinds",
            "evidence_channels",
            "coupled_roles",
        ):
            if not getattr(self, name):
                raise DynamicFoldRegistryError(
                    f"{name} must declare at least one method contract."
                )
        if not isinstance(self.zero_gradient, bool) or not isinstance(
            self.standalone, bool
        ):
            raise DynamicFoldRegistryError(
                "zero_gradient and standalone must be boolean."
            )
        object.__setattr__(self, "reason", _text(self.reason, "method reason"))

    @property
    def selectable(self) -> bool:
        return self.status in {
            MethodStatus.ACTIVE.value,
            MethodStatus.CONTROL.value,
            MethodStatus.EXPERIMENTAL.value,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": METHOD_SCHEMA,
            "method_id": self.method_id,
            "version": self.version,
            "family_id": self.family_id,
            "status": self.status,
            "source_kinds": list(self.source_kinds),
            "target_kinds": list(self.target_kinds),
            "evidence_channels": list(self.evidence_channels),
            "coupled_roles": list(self.coupled_roles),
            "zero_gradient": self.zero_gradient,
            "standalone": self.standalone,
            "reason": self.reason,
            "reopen_requires": list(self.reopen_requires),
        }


class FoldMethodRegistry:
    """Immutable-semantics registry used by planning and execution."""

    def __init__(self, methods: Sequence[FoldMethodSpec] = ()) -> None:
        self._methods: dict[str, FoldMethodSpec] = {}
        for method in methods:
            self.register(method)

    def register(self, method: FoldMethodSpec) -> None:
        if not isinstance(method, FoldMethodSpec):
            raise DynamicFoldRegistryError("method must be FoldMethodSpec.")
        if method.method_id in self._methods:
            raise DynamicFoldRegistryError(
                f"Method {method.method_id!r} is already registered."
            )
        self._methods[method.method_id] = method

    @property
    def methods(self) -> tuple[FoldMethodSpec, ...]:
        return tuple(self._methods[key] for key in sorted(self._methods))

    @property
    def fingerprint(self) -> str:
        return canonical_sha256([value.as_dict() for value in self.methods])

    def get(self, method_id: str) -> FoldMethodSpec:
        try:
            return self._methods[method_id]
        except KeyError as exc:
            raise DynamicFoldRegistryError(
                f"Unknown fold method {method_id!r}."
            ) from exc

    def require_selectable(
        self,
        method_id: str,
        *,
        require_zero_gradient: bool = True,
    ) -> FoldMethodSpec:
        method = self.get(method_id)
        if not method.selectable:
            raise DynamicFoldRegistryError(
                f"Method {method_id!r} is {method.status}: {method.reason}"
            )
        if require_zero_gradient and not method.zero_gradient:
            raise DynamicFoldRegistryError(
                f"Method {method_id!r} is not a zero-gradient fold."
            )
        return method


@dataclass(frozen=True, slots=True)
class ExperimentSignature:
    """Four-level novelty identity for one proposed transaction."""

    method_id: str
    method_version: int
    family_id: str
    parent_graph_fingerprints: tuple[str, ...]
    parent_state_fingerprints: tuple[tuple[str, str], ...]
    child_graph_fingerprint: str
    child_state_fingerprint: str
    evidence_fingerprint: str
    source_unit_ids: tuple[str, ...]
    target_unit_ids: tuple[str, ...]
    topology_transition: str
    selection_policy: str
    fit_objective: str
    promotion_gate: str
    zero_gradient: bool
    configuration: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "method_id",
            "family_id",
            "topology_transition",
            "selection_policy",
            "fit_objective",
            "promotion_gate",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if (
            isinstance(self.method_version, bool)
            or not isinstance(self.method_version, int)
            or self.method_version <= 0
        ):
            raise DynamicFoldRegistryError("method_version must be positive.")
        parent = tuple(
            sorted(
                _digest(value, "parent_graph_fingerprint")
                for value in self.parent_graph_fingerprints
            )
        )
        if not parent:
            raise DynamicFoldRegistryError(
                "At least one parent graph fingerprint is required."
            )
        object.__setattr__(self, "parent_graph_fingerprints", parent)
        parent_states = tuple(
            sorted(
                (
                    _text(name, "parent state name"),
                    _digest(digest, "parent state fingerprint"),
                )
                for name, digest in self.parent_state_fingerprints
            )
        )
        if not parent_states or len(
            {name for name, _digest_value in parent_states}
        ) != len(parent_states):
            raise DynamicFoldRegistryError(
                "Parent state fingerprints must be nonempty and uniquely named."
            )
        object.__setattr__(self, "parent_state_fingerprints", parent_states)
        for name in (
            "child_graph_fingerprint",
            "child_state_fingerprint",
            "evidence_fingerprint",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(
            self, "source_unit_ids", _strings(self.source_unit_ids, "source_unit_id")
        )
        object.__setattr__(
            self, "target_unit_ids", _strings(self.target_unit_ids, "target_unit_id")
        )
        if not self.source_unit_ids or not self.target_unit_ids:
            raise DynamicFoldRegistryError(
                "Experiment signatures require source and target units."
            )
        if not isinstance(self.zero_gradient, bool):
            raise DynamicFoldRegistryError("zero_gradient must be boolean.")
        object.__setattr__(
            self,
            "configuration",
            _metadata(self.configuration, "configuration"),
        )

    @property
    def family_sha256(self) -> str:
        return canonical_sha256(
            {
                "family_id": self.family_id,
                "zero_gradient": self.zero_gradient,
            }
        )

    @property
    def lane_sha256(self) -> str:
        return canonical_sha256(
            {
                "family_sha256": self.family_sha256,
                "method_id": self.method_id,
                "method_version": self.method_version,
                "topology_transition": self.topology_transition,
                "selection_policy": self.selection_policy,
                "fit_objective": self.fit_objective,
                "promotion_gate": self.promotion_gate,
                "zero_gradient": self.zero_gradient,
            }
        )

    @property
    def configuration_sha256(self) -> str:
        return canonical_sha256(
            {
                "lane_sha256": self.lane_sha256,
                "child_graph_fingerprint": self.child_graph_fingerprint,
                "configuration": dict(self.configuration),
            }
        )

    @property
    def candidate_sha256(self) -> str:
        return canonical_sha256(
            {
                "configuration_sha256": self.configuration_sha256,
                "parent_graph_fingerprints": list(self.parent_graph_fingerprints),
                "parent_state_fingerprints": dict(self.parent_state_fingerprints),
                "child_state_fingerprint": self.child_state_fingerprint,
                "evidence_fingerprint": self.evidence_fingerprint,
                "source_unit_ids": list(self.source_unit_ids),
                "target_unit_ids": list(self.target_unit_ids),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "method_version": self.method_version,
            "family_id": self.family_id,
            "parent_graph_fingerprints": list(self.parent_graph_fingerprints),
            "parent_state_fingerprints": dict(self.parent_state_fingerprints),
            "child_graph_fingerprint": self.child_graph_fingerprint,
            "child_state_fingerprint": self.child_state_fingerprint,
            "evidence_fingerprint": self.evidence_fingerprint,
            "source_unit_ids": list(self.source_unit_ids),
            "target_unit_ids": list(self.target_unit_ids),
            "topology_transition": self.topology_transition,
            "selection_policy": self.selection_policy,
            "fit_objective": self.fit_objective,
            "promotion_gate": self.promotion_gate,
            "zero_gradient": self.zero_gradient,
            "configuration": dict(self.configuration),
            "family_sha256": self.family_sha256,
            "lane_sha256": self.lane_sha256,
            "configuration_sha256": self.configuration_sha256,
            "candidate_sha256": self.candidate_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExperimentSignature:
        _exact_keys(
            value,
            {
                "method_id",
                "method_version",
                "family_id",
                "parent_graph_fingerprints",
                "parent_state_fingerprints",
                "child_graph_fingerprint",
                "child_state_fingerprint",
                "evidence_fingerprint",
                "source_unit_ids",
                "target_unit_ids",
                "topology_transition",
                "selection_policy",
                "fit_objective",
                "promotion_gate",
                "zero_gradient",
                "configuration",
                "family_sha256",
                "lane_sha256",
                "configuration_sha256",
                "candidate_sha256",
            },
            "experiment signature",
        )
        configuration = value["configuration"]
        parent_states = value["parent_state_fingerprints"]
        if not isinstance(configuration, Mapping) or not isinstance(
            parent_states, Mapping
        ):
            raise DynamicFoldRegistryError(
                "Experiment signature configuration and parent states must be objects."
            )
        result = cls(
            method_id=value["method_id"],
            method_version=value["method_version"],
            family_id=value["family_id"],
            parent_graph_fingerprints=tuple(value["parent_graph_fingerprints"]),
            parent_state_fingerprints=tuple(parent_states.items()),
            child_graph_fingerprint=value["child_graph_fingerprint"],
            child_state_fingerprint=value["child_state_fingerprint"],
            evidence_fingerprint=value["evidence_fingerprint"],
            source_unit_ids=tuple(value["source_unit_ids"]),
            target_unit_ids=tuple(value["target_unit_ids"]),
            topology_transition=value["topology_transition"],
            selection_policy=value["selection_policy"],
            fit_objective=value["fit_objective"],
            promotion_gate=value["promotion_gate"],
            zero_gradient=value["zero_gradient"],
            configuration=tuple(configuration.items()),
        )
        expected_hashes = {
            "family_sha256": result.family_sha256,
            "lane_sha256": result.lane_sha256,
            "configuration_sha256": result.configuration_sha256,
            "candidate_sha256": result.candidate_sha256,
        }
        for name, expected in expected_hashes.items():
            if value[name] != expected:
                raise DynamicFoldRegistryError(
                    f"Experiment signature {name} does not match its contents."
                )
        return result


@dataclass(frozen=True, slots=True)
class LedgerRecord:
    """Append-only result or curated closure."""

    signature: ExperimentSignature
    closure_level: ClosureLevel | str
    status: RecordStatus | str
    reason_class: str
    reason: str
    metrics: tuple[tuple[str, JsonScalar], ...] = ()
    previous_record_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.signature, ExperimentSignature):
            raise DynamicFoldRegistryError(
                "ledger signature must be ExperimentSignature."
            )
        level = (
            self.closure_level.value
            if isinstance(self.closure_level, ClosureLevel)
            else self.closure_level
        )
        status = (
            self.status.value if isinstance(self.status, RecordStatus) else self.status
        )
        try:
            level = ClosureLevel(level)
            status = RecordStatus(status)
        except ValueError as exc:
            raise DynamicFoldRegistryError(
                "Unknown ledger closure level or status."
            ) from exc
        object.__setattr__(self, "closure_level", level.value)
        object.__setattr__(self, "status", status.value)
        object.__setattr__(
            self, "reason_class", _text(self.reason_class, "reason_class")
        )
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        object.__setattr__(self, "metrics", _metadata(self.metrics, "metrics"))
        if self.previous_record_sha256 is not None:
            object.__setattr__(
                self,
                "previous_record_sha256",
                _digest(self.previous_record_sha256, "previous_record_sha256"),
            )

    @property
    def closure_sha256(self) -> str:
        level = ClosureLevel(self.closure_level)
        if level is ClosureLevel.EXACT:
            return self.signature.candidate_sha256
        if level is ClosureLevel.CONFIGURATION:
            return self.signature.configuration_sha256
        if level is ClosureLevel.LANE:
            return self.signature.lane_sha256
        return self.signature.family_sha256

    @property
    def record_sha256(self) -> str:
        return canonical_sha256(self.as_dict(include_hash=False))

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "schema": LEDGER_SCHEMA,
            "signature": self.signature.as_dict(),
            "closure_level": self.closure_level,
            "closure_sha256": self.closure_sha256,
            "status": self.status,
            "reason_class": self.reason_class,
            "reason": self.reason,
            "metrics": dict(self.metrics),
            "previous_record_sha256": self.previous_record_sha256,
        }
        if include_hash:
            value["record_sha256"] = canonical_sha256(value)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LedgerRecord:
        _exact_keys(
            value,
            {
                "schema",
                "signature",
                "closure_level",
                "closure_sha256",
                "status",
                "reason_class",
                "reason",
                "metrics",
                "previous_record_sha256",
                "record_sha256",
            },
            "ledger record",
        )
        if value["schema"] != LEDGER_SCHEMA:
            raise DynamicFoldRegistryError("Unknown ledger record schema.")
        signature = value["signature"]
        metrics = value["metrics"]
        if not isinstance(signature, Mapping) or not isinstance(metrics, Mapping):
            raise DynamicFoldRegistryError(
                "Ledger signature and metrics must be objects."
            )
        result = cls(
            signature=ExperimentSignature.from_dict(signature),
            closure_level=value["closure_level"],
            status=value["status"],
            reason_class=value["reason_class"],
            reason=value["reason"],
            metrics=tuple(metrics.items()),
            previous_record_sha256=value["previous_record_sha256"],
        )
        if value["closure_sha256"] != result.closure_sha256:
            raise DynamicFoldRegistryError(
                "Ledger closure fingerprint does not match its contents."
            )
        if value["record_sha256"] != result.record_sha256:
            raise DynamicFoldRegistryError(
                "Ledger record fingerprint does not match its contents."
            )
        return result


@dataclass(frozen=True, slots=True)
class LedgerDecision:
    allowed: bool
    reason: str
    blocking_record_sha256: str | None = None
    closure_level: str | None = None


class ExperimentLedger:
    """In-memory view of an append-only canonical JSONL experiment ledger."""

    def __init__(self, records: Sequence[LedgerRecord] = ()) -> None:
        self._records: list[LedgerRecord] = []
        for record in records:
            self.append(record)

    @property
    def records(self) -> tuple[LedgerRecord, ...]:
        return tuple(self._records)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256([value.record_sha256 for value in self._records])

    def _validated_extension(
        self,
        records: Sequence[LedgerRecord],
    ) -> tuple[LedgerRecord, ...]:
        extension = tuple(records)
        previous = None if not self._records else self._records[-1].record_sha256
        for record in extension:
            if not isinstance(record, LedgerRecord):
                raise DynamicFoldRegistryError("record must be LedgerRecord.")
            if record.previous_record_sha256 != previous:
                raise DynamicFoldRegistryError(
                    "Ledger record does not extend the exact append-only hash chain."
                )
            previous = record.record_sha256
        return extension

    def append_batch(self, records: Sequence[LedgerRecord]) -> None:
        """Validate a complete extension before mutating the ledger."""

        extension = self._validated_extension(records)
        self._records.extend(extension)

    def append(self, record: LedgerRecord) -> None:
        self.append_batch((record,))

    def decision(self, signature: ExperimentSignature) -> LedgerDecision:
        identities = {
            ClosureLevel.EXACT.value: signature.candidate_sha256,
            ClosureLevel.CONFIGURATION.value: signature.configuration_sha256,
            ClosureLevel.LANE.value: signature.lane_sha256,
            ClosureLevel.FAMILY.value: signature.family_sha256,
        }
        superseded: set[tuple[str, str]] = set()
        for record in reversed(self._records):
            key = (record.closure_level, record.closure_sha256)
            if record.status == RecordStatus.SUPERSEDED.value:
                superseded.add(key)
                continue
            if key in superseded:
                continue
            if record.status == RecordStatus.FAILED_INFRASTRUCTURE.value:
                continue
            if record.closure_sha256 != identities[record.closure_level]:
                continue
            if record.status == RecordStatus.ACCEPTED.value:
                return LedgerDecision(
                    False,
                    "The exact accepted transaction is already represented.",
                    record.record_sha256,
                    record.closure_level,
                )
            return LedgerDecision(
                False,
                record.reason,
                record.record_sha256,
                record.closure_level,
            )
        return LedgerDecision(True, "No matching closure or prior exact attempt.")

    def supersede_records(
        self,
        record_sha256s: Sequence[str],
        *,
        reason_class: str,
        reason: str,
    ) -> tuple[LedgerRecord, ...]:
        """Append compensating closures for provisional records.

        Artifact publication/search is a wider transaction than one executor
        run.  If a certified candidate is later pruned, these append-only
        records make its provisional acceptances retryable without erasing the
        audit trail.
        """

        requested = tuple(dict.fromkeys(record_sha256s))
        if not requested:
            return ()
        by_hash = {record.record_sha256: record for record in self._records}
        missing = [value for value in requested if value not in by_hash]
        if missing:
            raise DynamicFoldRegistryError(
                "Cannot supersede records absent from the current ledger."
            )
        result = []
        previous = None if not self._records else self._records[-1].record_sha256
        for value in requested:
            source = by_hash[value]
            if source.status not in {
                RecordStatus.ACCEPTED.value,
                RecordStatus.REJECTED.value,
                RecordStatus.SUPERSESSION_PENDING.value,
            }:
                raise DynamicFoldRegistryError(
                    "Only provisional accepted/rejected or rollback-pending "
                    "records may be superseded."
                )
            closure = LedgerRecord(
                signature=source.signature,
                closure_level=source.closure_level,
                status=RecordStatus.SUPERSEDED,
                reason_class=reason_class,
                reason=reason,
                metrics=(),
                previous_record_sha256=previous,
            )
            result.append(closure)
            previous = closure.record_sha256
        self.append_batch(result)
        return tuple(result)

    def prepare_supersession(
        self,
        record_sha256s: Sequence[str],
        *,
        reason_class: str,
        reason: str,
    ) -> tuple[LedgerRecord, ...]:
        """Durably block provisional records before deleting their artifact.

        A later :meth:`supersede_records` call finalizes the rollback and
        reopens the exact candidate.  If artifact deletion or finalization
        fails, the pending record remains a conservative blocking closure.
        """

        requested = tuple(dict.fromkeys(record_sha256s))
        if not requested:
            return ()
        by_hash = {record.record_sha256: record for record in self._records}
        missing = [value for value in requested if value not in by_hash]
        if missing:
            raise DynamicFoldRegistryError(
                "Cannot prepare supersession for records absent from the ledger."
            )
        result = []
        previous = None if not self._records else self._records[-1].record_sha256
        for value in requested:
            source = by_hash[value]
            if source.status not in {
                RecordStatus.ACCEPTED.value,
                RecordStatus.REJECTED.value,
            }:
                raise DynamicFoldRegistryError(
                    "Only provisional accepted/rejected execution records may "
                    "enter supersession-pending state."
                )
            pending = LedgerRecord(
                signature=source.signature,
                closure_level=source.closure_level,
                status=RecordStatus.SUPERSESSION_PENDING,
                reason_class=reason_class,
                reason=reason,
                metrics=(("source_record_sha256", source.record_sha256),),
                previous_record_sha256=previous,
            )
            result.append(pending)
            previous = pending.record_sha256
        self.append_batch(result)
        return tuple(result)

    def write_jsonl(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive_ledger_writer(destination):
            if destination.exists():
                persisted = self.read_jsonl(destination)
                if tuple(
                    value.record_sha256 for value in persisted.records
                ) != tuple(value.record_sha256 for value in self.records):
                    raise DynamicFoldRegistryError(
                        "Refusing a stale ledger rewrite: in-memory and durable "
                        "hash chains differ."
                    )
            _write_records_atomic(destination, self._records)

    @staticmethod
    def read_jsonl(path: str | Path) -> ExperimentLedger:
        """Read a canonical snapshot into the in-memory ledger type.

        DurableExperimentLedger has a path-based constructor, so inherited
        classmethod construction would misinterpret parsed records as a path
        when a durable instance verifies an exported snapshot.
        """

        source = Path(path)
        if not source.is_file():
            raise DynamicFoldRegistryError(f"Ledger file does not exist: {source}.")
        raw = source.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise DynamicFoldRegistryError("Ledger JSONL must end with a newline.")
        records = []
        for line_number, raw_line in enumerate(raw.splitlines(), start=1):
            if not raw_line:
                raise DynamicFoldRegistryError(f"Ledger line {line_number} is blank.")
            try:
                value = json.loads(
                    raw_line.decode("ascii"),
                    object_pairs_hook=_strict_object,
                    parse_constant=lambda token: (_ for _ in ()).throw(
                        DynamicFoldRegistryError(
                            f"Ledger JSON contains non-finite value {token!r}."
                        )
                    ),
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DynamicFoldRegistryError(
                    f"Ledger line {line_number} is not canonical JSON."
                ) from exc
            if not isinstance(value, Mapping):
                raise DynamicFoldRegistryError(
                    f"Ledger line {line_number} must be an object."
                )
            record = LedgerRecord.from_dict(value)
            if canonical_json_bytes(value) != raw_line:
                raise DynamicFoldRegistryError(
                    f"Ledger line {line_number} is not canonically encoded."
                )
            records.append(record)
        return ExperimentLedger(tuple(records))

    def append_jsonl(self, path: str | Path, record: LedgerRecord) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive_ledger_writer(destination):
            if destination.exists():
                persisted = self.read_jsonl(destination)
                if tuple(
                    value.record_sha256 for value in persisted.records
                ) != tuple(value.record_sha256 for value in self.records):
                    raise DynamicFoldRegistryError(
                        "Refusing a stale ledger append: in-memory and durable "
                        "hash chains differ."
                    )
            elif self.records:
                raise DynamicFoldRegistryError(
                    "Cannot append a nonempty in-memory chain to a missing ledger."
                )
            extension = self._validated_extension((record,))
            _write_records_atomic(destination, (*self._records, *extension))
            self._records.extend(extension)


class DurableExperimentLedger(ExperimentLedger):
    """Crash-safe ledger with an exclusive single-writer commit section."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().absolute()
        if self.path.exists() and (
            not self.path.is_file() or self.path.is_symlink()
        ):
            raise DynamicFoldRegistryError(
                "Durable ledger path must be a regular file or be absent."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__()
        with _exclusive_ledger_writer(self.path):
            loaded = (
                ExperimentLedger.read_jsonl(self.path)
                if self.path.is_file()
                else ExperimentLedger()
            )
            self._records.extend(loaded.records)
            if not self.path.exists():
                _write_records_atomic(self.path, self._records)

    def append(self, record: LedgerRecord) -> None:
        self.append_batch((record,))

    def append_batch(self, records: Sequence[LedgerRecord]) -> None:
        """Commit one complete hash-chain extension with one atomic replace."""

        extension = self._validated_extension(records)
        if not extension:
            return
        with _exclusive_ledger_writer(self.path):
            persisted = ExperimentLedger.read_jsonl(self.path)
            if tuple(
                value.record_sha256 for value in persisted.records
            ) != tuple(value.record_sha256 for value in self._records):
                raise DynamicFoldRegistryError(
                    "Refusing a stale durable-ledger append."
                )
            _write_records_atomic(self.path, (*self._records, *extension))
            self._records.extend(extension)


def _method(
    method_id: str,
    family_id: str,
    status: MethodStatus,
    source_kinds: Sequence[str],
    target_kinds: Sequence[str],
    evidence: Sequence[str],
    roles: Sequence[str],
    reason: str,
    *,
    standalone: bool = True,
    zero_gradient: bool = True,
    reopen_requires: Sequence[str] = (),
) -> FoldMethodSpec:
    return FoldMethodSpec(
        method_id=method_id,
        version=1,
        family_id=family_id,
        status=status,
        source_kinds=tuple(source_kinds),
        target_kinds=tuple(target_kinds),
        evidence_channels=tuple(evidence),
        coupled_roles=tuple(roles),
        zero_gradient=zero_gradient,
        standalone=standalone,
        reason=reason,
        reopen_requires=tuple(reopen_requires),
    )


def default_method_registry() -> FoldMethodRegistry:
    """Curated 2026 project knowledge, including settled negative lanes."""

    active = [
        _method(
            "copy_exact",
            "exact_identity",
            MethodStatus.CONTROL,
            ("any",),
            ("same_semantics_same_shape",),
            ("structure",),
            ("all_target_tensor_roles",),
            "Exact tensor-role copy and structural wiring control.",
        ),
        _method(
            "lfm2_closed_loop_layer",
            "coupled_closed_loop_layer",
            MethodStatus.ACTIVE,
            ("attention_or_short_conv", "gated_mlp", "residual"),
            ("lfm2_layer",),
            (
                "parent_qkvo_or_bcx",
                "parent_swiglu",
                "child_state",
                "residual_endpoints",
                "free_rollout",
            ),
            (
                "q_k_v_o_cache",
                "b_c_x_fir_state_output",
                "gate_up_post_swiglu_down",
                "ordered_residual",
            ),
            "Current zero-gradient residual-attention-aware LFM layer transaction.",
        ),
        _method(
            "lfm2_hybrid_residual_attention_model",
            "lfm2_hybrid_residual_attention_model",
            MethodStatus.ACTIVE,
            ("lfm2_hybrid_model",),
            ("lfm2_hybrid_model",),
            (
                "cached_parent_circuit_traces",
                "ordered_source_span_endpoints",
                "attention_residual_maps",
                "post_swiglu",
                "child_state",
                "whole_output_behavior",
            ),
            (
                "q_k_v_o_cache",
                "b_c_x_fir_state_output",
                "gate_up_post_swiglu_down",
                "all_layers_ordered_residual",
            ),
            "One zero-gradient whole-model LFM2 transaction jointly composes "
            "ordered depth spans while folding attention/short-conv and SwiGLU "
            "circuits against cached residual endpoints.",
        ),
        _method(
            "lfm2_producer_first_residual_attention_model",
            "lfm2_producer_first_residual_attention_model",
            MethodStatus.ACTIVE,
            ("lfm2_producer_first_model",),
            ("lfm2_producer_first_model",),
            (
                "sealed_parent_circuit_traces",
                "real_parent_medoid_producers",
                "installed_child_state",
                "sequential_residual_attention_feedback",
                "whole_output_behavior",
            ),
            (
                "q_k_v_o_cache",
                "b_c_x_fir_state_output",
                "gate_up_post_swiglu_down",
                "all_layers_ordered_residual",
            ),
            "One full-depth whole-model transaction installs complete real "
            "parent producers before any child capture, then sequentially "
            "refines and recaptures every LFM2 layer without training.",
        ),
        _method(
            "lfm2_dense_budget_residual_attention_model",
            "lfm2_dense_budget_residual_attention_model",
            MethodStatus.ACTIVE,
            ("lfm2_dense_budget_model",),
            ("lfm2_dense_budget_model",),
            (
                "sealed_parent_circuit_traces",
                "complete_weight_or_circuit_balanced_residual_basis",
                "gqa_rope_attention_transport",
                "coupled_swiglu_transport",
                "installed_child_state",
                "whole_output_behavior",
            ),
            (
                "q_k_v_o_cache",
                "b_c_x_fir_state_output",
                "gate_up_post_swiglu_down",
                "all_layers_ordered_residual",
            ),
            "One same-depth dense blank-first LFM2 transaction fills every "
            "smaller target tensor from a complete parent-derived residual "
            "basis, RoPE/GQA-aware attention maps, coupled SwiGLU maps, and "
            "cached child-closed-loop output repair without training.",
        ),
        _method(
            "neo_native_vision_swiglu_model",
            "native_vision_coupled_swiglu_model",
            MethodStatus.ACTIVE,
            ("neo_native_vision_model",),
            ("neo_native_vision_model",),
            (
                "image_conditioned_post_swiglu",
                "cached_parent_outputs",
                "child_state",
                "residual_endpoints",
            ),
            (
                "q_k_v_o_cache",
                "gate_up_post_swiglu_down",
                "all_layers_ordered_residual",
            ),
            "Atomically copy NEO temporal/height/width attention and all "
            "interfaces exactly while removing one image-conditioned coupled "
            "SwiGLU neuron per decoder layer with cached residual compensation.",
        ),
        _method(
            "coupled_gqa_medoid_endpoint",
            "coupled_attention",
            MethodStatus.ACTIVE,
            ("attention",),
            ("attention",),
            ("q", "k", "v", "context", "residual_endpoint", "rollout"),
            ("q_k", "v_o", "gqa_group", "cache"),
            "Real complete GQA producers plus child-state endpoint feedback.",
        ),
        _method(
            "coupled_swiglu_medoid_endpoint",
            "coupled_gated_mlp",
            MethodStatus.ACTIVE,
            ("gated_mlp",),
            ("gated_mlp",),
            ("gate", "up", "post_swiglu", "residual_endpoint", "rollout"),
            ("gate_up_post_swiglu_down",),
            "Real coupled SwiGLU producers plus child-state endpoint feedback.",
        ),
        _method(
            "coupled_short_conv_channel_endpoint",
            "coupled_recurrent_mixer",
            MethodStatus.ACTIVE,
            ("short_conv",),
            ("short_conv",),
            ("b_c_x", "fir_state", "residual_endpoint", "rollout"),
            ("b_c_x_fir_state_output",),
            "LFM B/C/x, FIR state, and output must move as one circuit.",
        ),
        _method(
            "token_interface_bridge",
            "token_interface",
            MethodStatus.ACTIVE,
            ("token_embedding", "output_head"),
            ("token_embedding", "output_head"),
            ("token_bytes", "parent_logits", "child_logits", "semantic_outputs"),
            ("embedding_tokenizer_output_head_tying",),
            "Identity, strict row, or byte-span token interface compilation.",
        ),
        _method(
            "qwen3_global_codebook_fold",
            "qwen3_global_codebook",
            MethodStatus.ACTIVE,
            ("qwen3_dense_model",),
            ("qwen3_codebook_model",),
            (
                "cached_alignment",
                "cached_codebook_parameters",
                "cached_codebook_topology",
                "exact_token_rows",
                "whole_output_behavior",
            ),
            (
                "cache",
                "embedding_tokenizer_output_head_tying",
                "gate_up_post_swiglu_down",
                "q_k_v_o",
            ),
            "One zero-gradient model-level Qwen3 transaction installs the "
            "globally coupled p35 attention/p36 neuron codebook and exact "
            "token interface.",
        ),
        _method(
            "qwen3_compensated_transaction_replay",
            "qwen3_authenticated_compensated_neuron",
            MethodStatus.ACTIVE,
            ("qwen3_dense_model",),
            ("qwen3_compensated_model",),
            (
                "authenticated_transactions",
                "cached_parent_outputs",
                "cached_candidate_outputs",
                "child_state",
                "whole_output_behavior",
            ),
            (
                "all_layers_ordered_residual",
                "embedding_tokenizer_output_head_tying",
                "gate_up_post_swiglu_down",
                "q_k_v_o_cache",
            ),
            "One zero-gradient model-level Qwen3 transaction replays an "
            "authenticated sequence of compensated multi-anchor neuron "
            "merges while copying every unaffected circuit exactly.",
        ),
        _method(
            "ordered_residual_span_compose",
            "ordered_depth",
            MethodStatus.EXPERIMENTAL,
            ("ordered_layer_span",),
            ("residual_layer",),
            ("span_inputs", "span_outputs", "child_state", "rollout"),
            ("attention_mlp_residual_order",),
            "Depth reduction is permitted only as ordered residual composition.",
            standalone=False,
        ),
        _method(
            "compensated_real_neuron_merge",
            "local_coupled_gated_mlp",
            MethodStatus.ACTIVE,
            ("gated_mlp_neuron",),
            ("gated_mlp_neuron",),
            ("activation", "output_dual", "rollout", "behavior"),
            ("gate_up_down",),
            "Safe local Qwen neuron folds are proven and should be composed adaptively.",
        ),
        _method(
            "attention_codebook_boundary",
            "local_coupled_attention",
            MethodStatus.ACTIVE,
            ("attention_group",),
            ("attention_group",),
            ("activation", "rollout", "behavior"),
            ("q_k_v_o",),
            "The p35 attention boundary is the strongest coherent fold-only incumbent.",
        ),
    ]
    closed_specs = [
        (
            "one_shot_mean_axis_fold",
            "independent_axis_projection",
            "Mean/RMS scale variants remained incoherent at aggressive ratios.",
        ),
        (
            "one_shot_rms_axis_fold",
            "independent_axis_projection",
            "Scale is a stability control, not the missing causal mechanism.",
        ),
        (
            "one_shot_pca_axis_fold",
            "variance_basis_projection",
            "High retained activation variance did not preserve executable circuits.",
        ),
        (
            "dual_orthonormal_axis_fold",
            "variance_basis_projection",
            "Dual/oblique bases retained energy while free-running behavior collapsed.",
        ),
        (
            "activation_subset_axis_fold",
            "sparse_support_only",
            "Real support selection alone removed too much coupled function.",
        ),
        (
            "dense_sinkhorn_axis_fold",
            "transport_support_only",
            "Dense OT with prior costs did not preserve closed-loop behavior.",
        ),
        (
            "path_conjugate_axis_fold",
            "coordinate_pairing_only",
            "Producer/consumer map conjugacy alone did not preserve nonlinear order.",
        ),
        (
            "uniform_depth_codebook",
            "uniform_depth_sharing",
            "Uniform depth/codebook compression crossed a sharp coherence cliff.",
        ),
        (
            "sequential_local_output_fit",
            "local_endpoint_only",
            "Locally accepted closed-form output fits did not compose globally.",
        ),
        (
            "child_action_only_refold",
            "child_action_only",
            "Child-action-only rollout refolding failed to escape collapse.",
        ),
        (
            "shared_token_random_coordinate_ridge",
            "random_coordinate_bridge",
            "Shared-token ridge into random child coordinates did not transfer function.",
        ),
        (
            "raw_value_regression",
            "numeric_internal_match",
            "Excellent V RMSE produced much worse held-out behavior.",
        ),
        (
            "raw_qk_anchor_search",
            "numeric_internal_match",
            "Alternative raw Q/K anchors were worse than the incumbent.",
        ),
        (
            "global_vo_scale_sweep",
            "numeric_internal_match",
            "Global V/O scale searches regressed held-out behavior.",
        ),
    ]
    closed = [
        _method(
            method_id,
            family_id,
            MethodStatus.CLOSED,
            ("any",),
            ("any",),
            ("historical_result",),
            ("incomplete_coupling",),
            reason,
            reopen_requires=(
                "materially_new_coupled_circuit",
                "child_state_feedback",
                "free_running_behavior_gate",
            ),
        )
        for method_id, family_id, reason in closed_specs
    ]
    training = [
        _method(
            "gradient_residual_adapter",
            "training_recovery",
            MethodStatus.TRAINING_ONLY,
            ("any",),
            ("any",),
            ("training_data",),
            ("adapter",),
            "Gradient-trained residual adapters are recovery, not folding.",
            zero_gradient=False,
        ),
        _method(
            "knowledge_distillation",
            "training_recovery",
            MethodStatus.TRAINING_ONLY,
            ("any",),
            ("any",),
            ("teacher_outputs",),
            ("all_trainable_child_parameters",),
            "Distillation begins only after fold-only coherence.",
            zero_gradient=False,
        ),
    ]
    return FoldMethodRegistry((*active, *closed, *training))


__all__ = [
    "ClosureLevel",
    "DurableExperimentLedger",
    "DynamicFoldRegistryError",
    "ExperimentLedger",
    "ExperimentSignature",
    "FoldMethodRegistry",
    "FoldMethodSpec",
    "LEDGER_SCHEMA",
    "LedgerDecision",
    "LedgerRecord",
    "METHOD_SCHEMA",
    "MethodStatus",
    "RecordStatus",
    "default_method_registry",
]
