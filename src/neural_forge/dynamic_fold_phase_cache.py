"""Content-addressed phase caching and chained run events for dynamic folds.

This module is deliberately independent from any model family or folding
implementation.  A phase key commits to all upstream dependencies, the exact
numeric contract under which a result was produced, and the consumer ABI.  A
completed entry is published once, by atomic directory promotion, and is never
updated in place.

The event log is separate from cached payloads.  It is an append-only logical
record stored as uniquely encoded canonical JSONL.  Every event hashes the
previous event, so truncation, reordering, insertion, and mutation fail closed.
"""

from __future__ import annotations

import ctypes
import datetime as dt
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PHASE_KEY_SCHEMA = "neural-forge-dynamic-fold-phase-key-v1"
PHASE_PAYLOAD_INVENTORY_SCHEMA = (
    "neural-forge-dynamic-fold-phase-payload-inventory-v1"
)
PHASE_COMPLETION_SCHEMA = "neural-forge-dynamic-fold-phase-completion-v1"
PHASE_EVENT_SCHEMA = "neural-forge-dynamic-fold-phase-event-v1"
COMPLETE_FILENAME = "complete.json"
PAYLOAD_DIRECTORY_NAME = "payload"

EVENT_BEGIN = "begin"
EVENT_PROGRESS = "progress"
EVENT_HIT = "hit"
EVENT_MISS = "miss"
EVENT_COMPLETE = "complete"
EVENT_FAIL = "fail"
EVENT_HEARTBEAT = "heartbeat"
EVENT_KINDS = frozenset(
    {
        EVENT_BEGIN,
        EVENT_PROGRESS,
        EVENT_HIT,
        EVENT_MISS,
        EVENT_COMPLETE,
        EVENT_FAIL,
        EVENT_HEARTBEAT,
    }
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")
_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z\Z"
)
_GENESIS_SHA256 = "0" * 64
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_EVENT_LINE_BYTES = 4 * 1024 * 1024
_MAX_EVENT_LOG_BYTES = 512 * 1024 * 1024

_PHASE_KEY_BODY_FIELDS = frozenset(
    {
        "schema",
        "phase_name",
        "dependency_sha256s",
        "numeric_contract",
        "numeric_contract_sha256",
        "abi",
        "abi_sha256",
    }
)
_PHASE_KEY_FIELDS = _PHASE_KEY_BODY_FIELDS | {"phase_key_sha256"}
_COMPLETION_BODY_FIELDS = frozenset(
    {
        "schema",
        "phase_key",
        "phase_key_sha256",
        "payload_inventory",
        "result",
        "immutable_by_contract",
    }
)
_COMPLETION_FIELDS = _COMPLETION_BODY_FIELDS | {"completion_sha256"}
_INVENTORY_BODY_FIELDS = frozenset(
    {"schema", "directories", "files", "total_bytes"}
)
_INVENTORY_FIELDS = _INVENTORY_BODY_FIELDS | {"payload_tree_sha256"}
_EVENT_BODY_FIELDS = frozenset(
    {
        "schema",
        "run_id",
        "sequence",
        "timestamp_utc",
        "kind",
        "phase_name",
        "phase_key_sha256",
        "previous_event_sha256",
        "details",
    }
)
_EVENT_FIELDS = _EVENT_BODY_FIELDS | {"event_sha256"}


class DynamicFoldPhaseCacheError(RuntimeError):
    """Base error for phase-cache and phase-event operations."""


class FoldPhaseKeyError(DynamicFoldPhaseCacheError, ValueError):
    """A phase identity is malformed or incomplete."""


class FoldPhaseCacheIntegrityError(DynamicFoldPhaseCacheError):
    """A completed entry is stale, malformed, or has been modified."""


class FoldPhaseCachePublicationError(DynamicFoldPhaseCacheError):
    """A phase entry could not be published immutably."""


class FoldPhaseCacheConcurrencyError(FoldPhaseCachePublicationError):
    """Another writer owns the same phase or event-log lock."""


class FoldPhaseCacheAlreadyCompleteError(FoldPhaseCachePublicationError):
    """The requested content-addressed phase is already complete."""


class FoldPhaseEventLogError(DynamicFoldPhaseCacheError):
    """A chained phase event log is malformed or cannot be updated safely."""


@dataclass(frozen=True, slots=True)
class _VerifiedEventLogTail:
    """Last authenticated event plus the exact file identity it described."""

    next_sequence: int
    event_sha256: str
    stat_signature: tuple[int, int, int, int, int, int]


def _canonical_json_bytes(value: Any, name: str) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise FoldPhaseKeyError(f"{name} must be canonical JSON.") from error


def _canonical_sha256(value: Any, name: str) -> str:
    return hashlib.sha256(_canonical_json_bytes(value, name)).hexdigest()


def _json_value(value: Any, name: str) -> Any:
    return json.loads(_canonical_json_bytes(value, name).decode("ascii"))


def _json_mapping(
    value: Mapping[str, Any],
    name: str,
    *,
    require_nonempty: bool,
) -> dict[str, Any]:
    normalized = _json_value(value, name)
    if not isinstance(normalized, dict):
        raise FoldPhaseKeyError(f"{name} must be a JSON object.")
    if require_nonempty and not normalized:
        raise FoldPhaseKeyError(f"{name} cannot be empty.")
    return normalized


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise FoldPhaseKeyError(
            f"{name} must be a stable identifier without path separators."
        )
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise FoldPhaseKeyError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FoldPhaseEventLogError(f"{name} must be a non-negative integer.")
    return value


def _named_sha256s(value: Mapping[str, str], name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise FoldPhaseKeyError(f"{name} must bind at least one dependency.")
    result: dict[str, str] = {}
    for raw_key, raw_digest in value.items():
        key = _identifier(raw_key, f"{name} key")
        if key in result:
            raise FoldPhaseKeyError(f"{name} keys must be unique.")
        result[key] = _sha256(raw_digest, f"{name}.{key}")
    return dict(sorted(result.items()))


@dataclass(frozen=True, slots=True, init=False)
class FoldPhaseKey:
    """Canonical identity for one reusable dynamic-fold phase result."""

    _body_json: bytes
    phase_key_sha256: str

    def __init__(
        self,
        *,
        phase_name: str,
        dependency_sha256s: Mapping[str, str],
        numeric_contract: Mapping[str, Any],
        abi: Mapping[str, Any],
    ) -> None:
        phase = _identifier(phase_name, "phase_name")
        dependencies = _named_sha256s(
            dependency_sha256s,
            "dependency_sha256s",
        )
        numeric = _json_mapping(
            numeric_contract,
            "numeric_contract",
            require_nonempty=True,
        )
        abi_value = _json_mapping(abi, "abi", require_nonempty=True)
        body = {
            "schema": PHASE_KEY_SCHEMA,
            "phase_name": phase,
            "dependency_sha256s": dependencies,
            "numeric_contract": numeric,
            "numeric_contract_sha256": _canonical_sha256(
                numeric,
                "numeric_contract",
            ),
            "abi": abi_value,
            "abi_sha256": _canonical_sha256(abi_value, "abi"),
        }
        body_json = _canonical_json_bytes(body, "phase key")
        object.__setattr__(self, "_body_json", body_json)
        object.__setattr__(
            self,
            "phase_key_sha256",
            hashlib.sha256(body_json).hexdigest(),
        )

    @property
    def phase_name(self) -> str:
        return str(self.as_dict()["phase_name"])

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FoldPhaseKey:
        value = _json_mapping(payload, "phase key", require_nonempty=True)
        if set(value) != _PHASE_KEY_FIELDS:
            raise FoldPhaseKeyError(
                "Phase key does not contain exactly the v1 fields."
            )
        if value.get("schema") != PHASE_KEY_SCHEMA:
            raise FoldPhaseKeyError("Phase key schema is unsupported.")
        result = cls(
            phase_name=value["phase_name"],
            dependency_sha256s=value["dependency_sha256s"],
            numeric_contract=value["numeric_contract"],
            abi=value["abi"],
        )
        claimed = _sha256(value["phase_key_sha256"], "phase_key_sha256")
        if claimed != result.phase_key_sha256 or value != result.as_dict():
            raise FoldPhaseKeyError(
                "Phase key or one of its derived contract hashes does not verify."
            )
        return result

    def as_dict(self) -> dict[str, Any]:
        result = json.loads(self._body_json.decode("ascii"))
        result["phase_key_sha256"] = self.phase_key_sha256
        return result


@dataclass(frozen=True, slots=True)
class FoldPhaseEvent:
    """One verified record in a chained canonical phase event log."""

    run_id: str
    sequence: int
    timestamp_utc: str
    kind: str
    phase_name: str
    phase_key_sha256: str
    previous_event_sha256: str
    details: Mapping[str, Any]
    event_sha256: str

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        sequence: int,
        timestamp_utc: str,
        kind: str,
        phase_key: FoldPhaseKey,
        previous_event_sha256: str,
        details: Mapping[str, Any] | None = None,
    ) -> FoldPhaseEvent:
        if not isinstance(phase_key, FoldPhaseKey):
            raise FoldPhaseEventLogError("phase_key must be a FoldPhaseKey.")
        body = _event_body(
            run_id=run_id,
            sequence=sequence,
            timestamp_utc=timestamp_utc,
            kind=kind,
            phase_name=phase_key.phase_name,
            phase_key_sha256=phase_key.phase_key_sha256,
            previous_event_sha256=previous_event_sha256,
            details={} if details is None else details,
        )
        return cls._from_body(body, _canonical_sha256(body, "phase event"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FoldPhaseEvent:
        try:
            value = _json_mapping(payload, "phase event", require_nonempty=True)
        except FoldPhaseKeyError as error:
            raise FoldPhaseEventLogError(str(error)) from error
        if set(value) != _EVENT_FIELDS:
            raise FoldPhaseEventLogError(
                "Phase event does not contain exactly the v1 fields."
            )
        claimed = _sha256_for_event(value.pop("event_sha256"), "event_sha256")
        body = _event_body(
            run_id=value.get("run_id"),
            sequence=value.get("sequence"),
            timestamp_utc=value.get("timestamp_utc"),
            kind=value.get("kind"),
            phase_name=value.get("phase_name"),
            phase_key_sha256=value.get("phase_key_sha256"),
            previous_event_sha256=value.get("previous_event_sha256"),
            details=value.get("details"),
        )
        if value.get("schema") != PHASE_EVENT_SCHEMA:
            raise FoldPhaseEventLogError("Phase event schema is unsupported.")
        actual = _canonical_sha256(body, "phase event")
        if claimed != actual:
            raise FoldPhaseEventLogError("Phase event self-hash does not verify.")
        return cls._from_body(body, actual)

    @classmethod
    def _from_body(
        cls,
        body: Mapping[str, Any],
        event_sha256: str,
    ) -> FoldPhaseEvent:
        return cls(
            run_id=str(body["run_id"]),
            sequence=int(body["sequence"]),
            timestamp_utc=str(body["timestamp_utc"]),
            kind=str(body["kind"]),
            phase_name=str(body["phase_name"]),
            phase_key_sha256=str(body["phase_key_sha256"]),
            previous_event_sha256=str(body["previous_event_sha256"]),
            details=dict(body["details"]),
            event_sha256=event_sha256,
        )

    def as_dict(self) -> dict[str, Any]:
        body = _event_body(
            run_id=self.run_id,
            sequence=self.sequence,
            timestamp_utc=self.timestamp_utc,
            kind=self.kind,
            phase_name=self.phase_name,
            phase_key_sha256=self.phase_key_sha256,
            previous_event_sha256=self.previous_event_sha256,
            details=self.details,
        )
        return {**body, "event_sha256": self.event_sha256}


def _event_body(
    *,
    run_id: Any,
    sequence: Any,
    timestamp_utc: Any,
    kind: Any,
    phase_name: Any,
    phase_key_sha256: Any,
    previous_event_sha256: Any,
    details: Any,
) -> dict[str, Any]:
    try:
        run = _identifier(run_id, "run_id")
        phase = _identifier(phase_name, "phase_name")
        phase_digest = _sha256(phase_key_sha256, "phase_key_sha256")
        previous = _sha256(previous_event_sha256, "previous_event_sha256")
        normalized_details = _json_mapping(
            details,
            "event details",
            require_nonempty=False,
        )
    except FoldPhaseKeyError as error:
        raise FoldPhaseEventLogError(str(error)) from error
    index = _nonnegative_int(sequence, "event sequence")
    if not isinstance(kind, str) or kind not in EVENT_KINDS:
        raise FoldPhaseEventLogError(f"Unsupported phase event kind: {kind!r}.")
    if not isinstance(timestamp_utc, str) or not _TIMESTAMP.fullmatch(timestamp_utc):
        raise FoldPhaseEventLogError(
            "timestamp_utc must be canonical UTC with six fractional digits."
        )
    try:
        parsed = dt.datetime.fromisoformat(timestamp_utc[:-1] + "+00:00")
    except ValueError as error:
        raise FoldPhaseEventLogError("timestamp_utc is invalid.") from error
    if parsed.tzinfo != dt.timezone.utc:
        raise FoldPhaseEventLogError("timestamp_utc must be UTC.")
    return {
        "schema": PHASE_EVENT_SCHEMA,
        "run_id": run,
        "sequence": index,
        "timestamp_utc": timestamp_utc,
        "kind": kind,
        "phase_name": phase,
        "phase_key_sha256": phase_digest,
        "previous_event_sha256": previous,
        "details": normalized_details,
    }


def _sha256_for_event(value: Any, name: str) -> str:
    try:
        return _sha256(value, name)
    except FoldPhaseKeyError as error:
        raise FoldPhaseEventLogError(str(error)) from error


def _utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


class FoldPhaseEventLog:
    """Durable canonical JSONL event chain for one compiler run."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        run_id: str,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self.path = _absolute_local_path(path, "event log")
        try:
            self.run_id = _identifier(run_id, "run_id")
        except FoldPhaseKeyError as error:
            raise FoldPhaseEventLogError(str(error)) from error
        self._clock = _utc_now if clock is None else clock
        if not callable(self._clock):
            raise FoldPhaseEventLogError("clock must be callable.")
        self._verified_tail: _VerifiedEventLogTail | None = None

    def read(self) -> tuple[FoldPhaseEvent, ...]:
        return read_fold_phase_events(self.path, expected_run_id=self.run_id)

    def append(
        self,
        kind: str,
        phase_key: FoldPhaseKey,
        details: Mapping[str, Any] | None = None,
    ) -> FoldPhaseEvent:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _assert_no_reparse_ancestors(self.path.parent, FoldPhaseEventLogError)
        lock = _acquire_lock(
            self.path.parent / f".{self.path.name}.lock",
            FoldPhaseEventLogError,
        )
        try:
            tail = self._load_verified_tail_locked()
            timestamp = self._clock()
            event = FoldPhaseEvent.create(
                run_id=self.run_id,
                sequence=tail.next_sequence if tail is not None else 0,
                timestamp_utc=timestamp,
                kind=kind,
                phase_key=phase_key,
                previous_event_sha256=(
                    _GENESIS_SHA256 if tail is None else tail.event_sha256
                ),
                details=details,
            )
            line = _canonical_json_bytes(event.as_dict(), "phase event") + b"\n"
            if len(line) > _MAX_EVENT_LINE_BYTES:
                raise FoldPhaseEventLogError("Phase event exceeds its size limit.")
            current_size = 0 if tail is None else tail.stat_signature[3]
            if current_size + len(line) > _MAX_EVENT_LOG_BYTES:
                raise FoldPhaseEventLogError("Phase event log exceeds its size limit.")
            if tail is None:
                _atomic_create_bytes(
                    self.path,
                    line,
                    FoldPhaseEventLogError,
                )
                raw = _read_limited(
                    self.path,
                    _MAX_EVENT_LOG_BYTES,
                    "event log",
                    FoldPhaseEventLogError,
                )
                if raw != line:
                    raise FoldPhaseEventLogError(
                        "Newly created phase event log does not verify."
                    )
                signature = _stat_signature(self.path.lstat())
            else:
                signature = _append_event_line(
                    self.path,
                    line,
                    expected_signature=tail.stat_signature,
                )
            self._verified_tail = _VerifiedEventLogTail(
                next_sequence=event.sequence + 1,
                event_sha256=event.event_sha256,
                stat_signature=signature,
            )
            return event
        finally:
            _release_lock(lock, FoldPhaseEventLogError)

    def _load_verified_tail_locked(self) -> _VerifiedEventLogTail | None:
        """Return a tail valid for the currently locked on-disk log.

        The common case is an unchanged stat identity and requires no prefix
        read.  A different identity is treated as an external writer or
        mutation: the complete chain is revalidated, and it must extend the
        tail already authenticated by this instance.  That extension check is
        what makes a valid-looking truncation or replacement fail closed.
        """

        cached = self._verified_tail
        if not _lexists(self.path):
            if cached is not None:
                raise FoldPhaseEventLogError(
                    "Phase event log was removed or truncated after verification."
                )
            return None
        _assert_ordinary_file(
            self.path,
            "event log",
            FoldPhaseEventLogError,
        )
        signature = _stat_signature(self.path.lstat())
        if cached is not None and signature == cached.stat_signature:
            return cached

        raw = _read_limited(
            self.path,
            _MAX_EVENT_LOG_BYTES,
            "event log",
            FoldPhaseEventLogError,
        )
        if _stat_signature(self.path.lstat()) != signature:
            raise FoldPhaseEventLogError(
                "Phase event log changed while its chain was being verified."
            )
        events = _parse_event_log(raw, expected_run_id=self.run_id)
        if cached is not None:
            if len(events) < cached.next_sequence:
                raise FoldPhaseEventLogError(
                    "Phase event log was truncated after verification."
                )
            if (
                cached.next_sequence > 0
                and events[cached.next_sequence - 1].event_sha256
                != cached.event_sha256
            ):
                raise FoldPhaseEventLogError(
                    "Phase event log no longer extends its verified prefix."
                )
        verified = _VerifiedEventLogTail(
            next_sequence=len(events),
            event_sha256=(
                _GENESIS_SHA256 if not events else events[-1].event_sha256
            ),
            stat_signature=signature,
        )
        self._verified_tail = verified
        return verified

    def begin(
        self,
        phase_key: FoldPhaseKey,
        details: Mapping[str, Any] | None = None,
    ) -> FoldPhaseEvent:
        return self.append(EVENT_BEGIN, phase_key, details)

    def progress(
        self,
        phase_key: FoldPhaseKey,
        details: Mapping[str, Any] | None = None,
    ) -> FoldPhaseEvent:
        return self.append(EVENT_PROGRESS, phase_key, details)

    def hit(
        self,
        phase_key: FoldPhaseKey,
        details: Mapping[str, Any] | None = None,
    ) -> FoldPhaseEvent:
        return self.append(EVENT_HIT, phase_key, details)

    def miss(
        self,
        phase_key: FoldPhaseKey,
        details: Mapping[str, Any] | None = None,
    ) -> FoldPhaseEvent:
        return self.append(EVENT_MISS, phase_key, details)

    def complete(
        self,
        phase_key: FoldPhaseKey,
        details: Mapping[str, Any] | None = None,
    ) -> FoldPhaseEvent:
        return self.append(EVENT_COMPLETE, phase_key, details)

    def fail(
        self,
        phase_key: FoldPhaseKey,
        details: Mapping[str, Any] | None = None,
    ) -> FoldPhaseEvent:
        return self.append(EVENT_FAIL, phase_key, details)

    def heartbeat(
        self,
        phase_key: FoldPhaseKey,
        details: Mapping[str, Any] | None = None,
    ) -> FoldPhaseEvent:
        return self.append(EVENT_HEARTBEAT, phase_key, details)


def read_fold_phase_events(
    path: str | os.PathLike[str],
    *,
    expected_run_id: str | None = None,
) -> tuple[FoldPhaseEvent, ...]:
    """Read and fully validate a canonical phase event chain."""

    source = _absolute_local_path(path, "event log")
    _assert_no_reparse_ancestors(source.parent, FoldPhaseEventLogError)
    if not _lexists(source):
        return ()
    _assert_ordinary_file(source, "event log", FoldPhaseEventLogError)
    raw = _read_limited(
        source,
        _MAX_EVENT_LOG_BYTES,
        "event log",
        FoldPhaseEventLogError,
    )
    return _parse_event_log(raw, expected_run_id=expected_run_id)


def _parse_event_log(
    raw: bytes,
    *,
    expected_run_id: str | None,
) -> tuple[FoldPhaseEvent, ...]:
    if not raw:
        return ()
    if not raw.endswith(b"\n"):
        raise FoldPhaseEventLogError("Phase event log ends with a partial event.")
    if expected_run_id is not None:
        try:
            run = _identifier(expected_run_id, "expected_run_id")
        except FoldPhaseKeyError as error:
            raise FoldPhaseEventLogError(str(error)) from error
    else:
        run = None
    events: list[FoldPhaseEvent] = []
    previous = _GENESIS_SHA256
    for index, line in enumerate(raw.splitlines(keepends=True)):
        if len(line) > _MAX_EVENT_LINE_BYTES:
            raise FoldPhaseEventLogError("Phase event line exceeds its size limit.")
        payload = _strict_canonical_json_bytes(
            line,
            f"phase event {index}",
            FoldPhaseEventLogError,
        )
        event = FoldPhaseEvent.from_dict(payload)
        if event.sequence != index:
            raise FoldPhaseEventLogError("Phase event sequence is not contiguous.")
        if event.previous_event_sha256 != previous:
            raise FoldPhaseEventLogError("Phase event hash chain is broken.")
        if run is None:
            run = event.run_id
        if event.run_id != run:
            raise FoldPhaseEventLogError("Phase event log mixes compiler run IDs.")
        events.append(event)
        previous = event.event_sha256
    return tuple(events)


@dataclass(frozen=True, slots=True)
class VerifiedFoldPhaseCacheEntry:
    """A complete phase payload verified against its expected phase key."""

    path: Path
    payload_path: Path
    phase_key: FoldPhaseKey
    payload_inventory: Mapping[str, Any]
    result: Mapping[str, Any]
    completion_sha256: str


class DynamicFoldPhaseCache:
    """Immutable content-addressed storage for expensive fold phases."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = _absolute_local_path(root, "phase cache root")

    def entry_path(self, phase_key: FoldPhaseKey) -> Path:
        if not isinstance(phase_key, FoldPhaseKey):
            raise FoldPhaseKeyError("phase_key must be a FoldPhaseKey.")
        return self.root / phase_key.phase_name / phase_key.phase_key_sha256

    def lookup(
        self,
        phase_key: FoldPhaseKey,
        *,
        events: FoldPhaseEventLog | None = None,
    ) -> VerifiedFoldPhaseCacheEntry | None:
        path = self.entry_path(phase_key)
        _assert_no_reparse_ancestors(
            path.parent,
            FoldPhaseCacheIntegrityError,
        )
        if not _lexists(path):
            if events is not None:
                events.miss(phase_key, {"entry_path": os.fspath(path)})
            return None
        try:
            entry = _verify_phase_entry(path, expected_key=phase_key)
        except FoldPhaseCacheIntegrityError as error:
            if events is not None:
                events.fail(
                    phase_key,
                    {
                        "operation": "lookup",
                        "error_type": type(error).__name__,
                        "message": str(error),
                    },
                )
            raise
        if events is not None:
            events.hit(
                phase_key,
                {
                    "completion_sha256": entry.completion_sha256,
                    "entry_path": os.fspath(path),
                },
            )
        return entry

    def begin(
        self,
        phase_key: FoldPhaseKey,
        *,
        events: FoldPhaseEventLog | None = None,
    ) -> FoldPhaseCacheTransaction:
        """Acquire a new phase transaction.

        Callers should normally call :meth:`lookup` first.  ``begin`` still
        checks under an exclusive per-key lock so two misses cannot publish the
        same immutable destination.
        """

        if not isinstance(phase_key, FoldPhaseKey):
            raise FoldPhaseKeyError("phase_key must be a FoldPhaseKey.")
        parent = self.root / phase_key.phase_name
        parent.mkdir(parents=True, exist_ok=True)
        _assert_no_reparse_ancestors(parent, FoldPhaseCachePublicationError)
        destination = self.entry_path(phase_key)
        lock = _acquire_lock(
            parent / f".{phase_key.phase_key_sha256}.lock",
            FoldPhaseCacheConcurrencyError,
        )
        try:
            if _lexists(destination):
                entry = _verify_phase_entry(destination, expected_key=phase_key)
                if events is not None:
                    events.hit(
                        phase_key,
                        {
                            "completion_sha256": entry.completion_sha256,
                            "entry_path": os.fspath(destination),
                            "operation": "begin",
                        },
                    )
                raise FoldPhaseCacheAlreadyCompleteError(
                    f"Immutable phase entry already exists: {destination}"
                )
            staging = Path(
                tempfile.mkdtemp(
                    prefix=f".{phase_key.phase_key_sha256}.",
                    suffix=".staging",
                    dir=parent,
                )
            )
            payload = staging / PAYLOAD_DIRECTORY_NAME
            payload.mkdir()
            transaction = FoldPhaseCacheTransaction(
                phase_key=phase_key,
                destination=destination,
                staging=staging,
                payload_path=payload,
                lock=lock,
                events=events,
            )
            lock = None
            if events is not None:
                try:
                    events.begin(
                        phase_key,
                        {"entry_path": os.fspath(destination)},
                    )
                except BaseException:
                    transaction._discard()
                    raise
            return transaction
        finally:
            if lock is not None:
                _release_lock(lock, FoldPhaseCachePublicationError)


class FoldPhaseCacheTransaction:
    """One exclusive staging transaction for a content-addressed phase."""

    def __init__(
        self,
        *,
        phase_key: FoldPhaseKey,
        destination: Path,
        staging: Path,
        payload_path: Path,
        lock: tuple[Path, int, tuple[int, int]],
        events: FoldPhaseEventLog | None,
    ) -> None:
        self.phase_key = phase_key
        self.destination = destination
        self.staging_path = staging
        self.payload_path = payload_path
        self._lock = lock
        self._events = events
        self._state = "active"

    def __enter__(self) -> FoldPhaseCacheTransaction:
        if self._state != "active":
            raise FoldPhaseCachePublicationError(
                "Phase transaction is no longer active."
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        del traceback
        if self._state != "active":
            return False
        if exc_type is None:
            error = FoldPhaseCachePublicationError(
                "Phase transaction exited without complete()."
            )
            self._record_failure(error)
            self._discard()
            raise error
        self._record_failure(exc)
        self._discard()
        return False

    def progress(self, details: Mapping[str, Any]) -> FoldPhaseEvent | None:
        self._assert_active()
        if self._events is None:
            return None
        return self._events.progress(self.phase_key, details)

    def heartbeat(
        self,
        details: Mapping[str, Any] | None = None,
    ) -> FoldPhaseEvent | None:
        self._assert_active()
        if self._events is None:
            return None
        return self._events.heartbeat(self.phase_key, details)

    def complete(
        self,
        *,
        result: Mapping[str, Any] | None = None,
    ) -> VerifiedFoldPhaseCacheEntry:
        self._assert_active()
        normalized_result = _json_mapping(
            {} if result is None else result,
            "phase result",
            require_nonempty=False,
        )
        try:
            inventory = _payload_inventory(self.payload_path)
            body = {
                "schema": PHASE_COMPLETION_SCHEMA,
                "phase_key": self.phase_key.as_dict(),
                "phase_key_sha256": self.phase_key.phase_key_sha256,
                "payload_inventory": inventory,
                "result": normalized_result,
                "immutable_by_contract": True,
            }
            completion_sha256 = _canonical_sha256(body, "phase completion")
            completion = {**body, "completion_sha256": completion_sha256}
            _atomic_create_bytes(
                self.staging_path / COMPLETE_FILENAME,
                _canonical_json_bytes(completion, "phase completion") + b"\n",
                FoldPhaseCachePublicationError,
            )
            _fsync_tree(self.staging_path)
            staged = _verify_phase_entry(
                self.staging_path,
                expected_key=self.phase_key,
            )
            if staged.completion_sha256 != completion_sha256:
                raise FoldPhaseCachePublicationError(
                    "Staged completion changed before publication."
                )
            _rename_directory_noreplace(self.staging_path, self.destination)
            _fsync_directory(self.destination.parent)
            self._state = "complete"
            self.staging_path = self.destination
            entry = _verify_phase_entry(
                self.destination,
                expected_key=self.phase_key,
            )
            if self._events is not None:
                self._events.complete(
                    self.phase_key,
                    {
                        "completion_sha256": entry.completion_sha256,
                        "payload_tree_sha256": entry.payload_inventory[
                            "payload_tree_sha256"
                        ],
                        "total_bytes": entry.payload_inventory["total_bytes"],
                    },
                )
            self._release()
            return entry
        except BaseException:
            if self._state == "complete":
                self._release()
            raise

    def _assert_active(self) -> None:
        if self._state != "active":
            raise FoldPhaseCachePublicationError(
                "Phase transaction is no longer active."
            )

    def _record_failure(self, error: BaseException) -> None:
        if self._events is None:
            return
        try:
            self._events.fail(
                self.phase_key,
                {
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
        except BaseException:
            # Never replace the phase's causal exception with telemetry failure.
            return

    def _discard(self) -> None:
        if self._state == "active":
            self._state = "failed"
        try:
            if self.staging_path != self.destination and _lexists(self.staging_path):
                shutil.rmtree(self.staging_path)
        finally:
            self._release()

    def _release(self) -> None:
        if self._lock is None:
            return
        lock = self._lock
        self._lock = None
        _release_lock(lock, FoldPhaseCachePublicationError)


def _verify_phase_entry(
    path: Path,
    *,
    expected_key: FoldPhaseKey,
) -> VerifiedFoldPhaseCacheEntry:
    _assert_ordinary_directory(path, "phase entry", FoldPhaseCacheIntegrityError)
    try:
        entries = {entry.name for entry in os.scandir(path)}
    except OSError as error:
        raise FoldPhaseCacheIntegrityError("Cannot enumerate phase entry.") from error
    if entries != {PAYLOAD_DIRECTORY_NAME, COMPLETE_FILENAME}:
        raise FoldPhaseCacheIntegrityError(
            "Phase entry contains missing or foreign root entries."
        )
    payload_path = path / PAYLOAD_DIRECTORY_NAME
    completion_path = path / COMPLETE_FILENAME
    _assert_ordinary_directory(
        payload_path,
        "phase payload",
        FoldPhaseCacheIntegrityError,
    )
    _assert_ordinary_file(
        completion_path,
        "phase completion marker",
        FoldPhaseCacheIntegrityError,
    )
    raw = _read_limited(
        completion_path,
        _MAX_JSON_BYTES,
        "phase completion marker",
        FoldPhaseCacheIntegrityError,
    )
    completion = _strict_canonical_json_bytes(
        raw,
        "phase completion marker",
        FoldPhaseCacheIntegrityError,
    )
    if set(completion) != _COMPLETION_FIELDS:
        raise FoldPhaseCacheIntegrityError(
            "Phase completion marker does not contain exactly the v1 fields."
        )
    if completion.get("schema") != PHASE_COMPLETION_SCHEMA:
        raise FoldPhaseCacheIntegrityError("Phase completion schema is unsupported.")
    try:
        stored_key = FoldPhaseKey.from_dict(completion.get("phase_key"))
        claimed_key = _sha256(
            completion.get("phase_key_sha256"),
            "phase_key_sha256",
        )
        claimed_completion = _sha256(
            completion.get("completion_sha256"),
            "completion_sha256",
        )
    except (FoldPhaseKeyError, TypeError) as error:
        raise FoldPhaseCacheIntegrityError(
            "Phase completion contains an invalid identity digest."
        ) from error
    if (
        stored_key.as_dict() != expected_key.as_dict()
        or claimed_key != expected_key.phase_key_sha256
        or stored_key.phase_key_sha256 != claimed_key
    ):
        raise FoldPhaseCacheIntegrityError(
            "Phase completion is stale or bound to a different phase key."
        )
    body = {key: value for key, value in completion.items() if key != "completion_sha256"}
    if claimed_completion != _canonical_sha256(body, "phase completion"):
        raise FoldPhaseCacheIntegrityError(
            "Phase completion marker self-hash does not verify."
        )
    inventory = _validate_inventory(completion.get("payload_inventory"))
    actual_inventory = _payload_inventory(payload_path)
    if inventory != actual_inventory:
        raise FoldPhaseCacheIntegrityError(
            "Phase payload differs from its immutable completion inventory."
        )
    try:
        result = _json_mapping(
            completion.get("result"),
            "phase result",
            require_nonempty=False,
        )
    except FoldPhaseKeyError as error:
        raise FoldPhaseCacheIntegrityError("Phase result is malformed.") from error
    if completion.get("immutable_by_contract") is not True:
        raise FoldPhaseCacheIntegrityError(
            "Phase completion is not declared immutable."
        )
    return VerifiedFoldPhaseCacheEntry(
        path=path,
        payload_path=payload_path,
        phase_key=stored_key,
        payload_inventory=inventory,
        result=result,
        completion_sha256=claimed_completion,
    )


def _validate_inventory(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FoldPhaseCacheIntegrityError("Payload inventory must be an object.")
    try:
        inventory = _json_mapping(
            value,
            "payload inventory",
            require_nonempty=True,
        )
    except FoldPhaseKeyError as error:
        raise FoldPhaseCacheIntegrityError("Payload inventory is malformed.") from error
    if set(inventory) != _INVENTORY_FIELDS:
        raise FoldPhaseCacheIntegrityError(
            "Payload inventory does not contain exactly the v1 fields."
        )
    if inventory.get("schema") != PHASE_PAYLOAD_INVENTORY_SCHEMA:
        raise FoldPhaseCacheIntegrityError("Payload inventory schema is unsupported.")
    directories = inventory.get("directories")
    files = inventory.get("files")
    if not isinstance(directories, list) or not isinstance(files, list):
        raise FoldPhaseCacheIntegrityError(
            "Payload inventory directories and files must be arrays."
        )
    _validate_inventory_records(directories, files)
    total = inventory.get("total_bytes")
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise FoldPhaseCacheIntegrityError("Payload total_bytes is invalid.")
    if total != sum(record["size_bytes"] for record in files):
        raise FoldPhaseCacheIntegrityError("Payload total_bytes does not verify.")
    body = {key: inventory[key] for key in _INVENTORY_BODY_FIELDS}
    try:
        tree_hash = _sha256(
            inventory.get("payload_tree_sha256"),
            "payload_tree_sha256",
        )
    except FoldPhaseKeyError as error:
        raise FoldPhaseCacheIntegrityError(
            "Payload tree digest is malformed."
        ) from error
    if tree_hash != _canonical_sha256(body, "payload inventory"):
        raise FoldPhaseCacheIntegrityError("Payload tree digest does not verify.")
    return inventory


def _validate_inventory_records(
    directories: list[Any],
    files: list[Any],
) -> None:
    directory_paths: list[str] = []
    for record in directories:
        if not isinstance(record, dict) or set(record) != {"path", "mode"}:
            raise FoldPhaseCacheIntegrityError(
                "Payload directory inventory record is malformed."
            )
        path = _relative_payload_path(record.get("path"))
        mode = record.get("mode")
        if isinstance(mode, bool) or not isinstance(mode, int) or mode < 0:
            raise FoldPhaseCacheIntegrityError(
                "Payload directory mode is malformed."
            )
        directory_paths.append(path)
    file_paths: list[str] = []
    for record in files:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "mode",
            "size_bytes",
            "sha256",
        }:
            raise FoldPhaseCacheIntegrityError(
                "Payload file inventory record is malformed."
            )
        path = _relative_payload_path(record.get("path"))
        mode = record.get("mode")
        size = record.get("size_bytes")
        if (
            isinstance(mode, bool)
            or not isinstance(mode, int)
            or mode < 0
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise FoldPhaseCacheIntegrityError(
                "Payload file size or mode is malformed."
            )
        try:
            _sha256(record.get("sha256"), "payload file sha256")
        except FoldPhaseKeyError as error:
            raise FoldPhaseCacheIntegrityError(
                "Payload file digest is malformed."
            ) from error
        file_paths.append(path)
    if directory_paths != sorted(directory_paths) or len(set(directory_paths)) != len(
        directory_paths
    ):
        raise FoldPhaseCacheIntegrityError(
            "Payload directory inventory is not unique and sorted."
        )
    if file_paths != sorted(file_paths) or len(set(file_paths)) != len(file_paths):
        raise FoldPhaseCacheIntegrityError(
            "Payload file inventory is not unique and sorted."
        )
    if set(directory_paths) & set(file_paths):
        raise FoldPhaseCacheIntegrityError(
            "Payload inventory path cannot be both a file and directory."
        )


def _relative_payload_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise FoldPhaseCacheIntegrityError("Payload path is invalid.")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise FoldPhaseCacheIntegrityError("Payload path escapes its root.")
    if path.as_posix() != value:
        raise FoldPhaseCacheIntegrityError("Payload path is not canonical POSIX form.")
    return value


def _payload_inventory(root: Path) -> dict[str, Any]:
    _assert_ordinary_directory(root, "phase payload", FoldPhaseCacheIntegrityError)
    directories: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    try:
        paths = sorted(root.rglob("*"), key=lambda value: value.as_posix())
    except OSError as error:
        raise FoldPhaseCacheIntegrityError("Cannot enumerate phase payload.") from error
    for path in paths:
        if _is_reparse(path):
            raise FoldPhaseCacheIntegrityError(
                "Phase payload contains a symlink or reparse point."
            )
        try:
            metadata = path.lstat()
        except OSError as error:
            raise FoldPhaseCacheIntegrityError(
                "Cannot inspect phase payload entry."
            ) from error
        relative = path.relative_to(root).as_posix()
        if stat.S_ISDIR(metadata.st_mode):
            directories.append(
                {"path": relative, "mode": stat.S_IMODE(metadata.st_mode)}
            )
        elif stat.S_ISREG(metadata.st_mode):
            if int(metadata.st_nlink) != 1:
                raise FoldPhaseCacheIntegrityError(
                    "Phase payload contains a hard-linked file."
                )
            files.append(
                {
                    "path": relative,
                    "mode": stat.S_IMODE(metadata.st_mode),
                    "size_bytes": int(metadata.st_size),
                    "sha256": _stable_file_sha256(path, metadata),
                }
            )
        else:
            raise FoldPhaseCacheIntegrityError(
                "Phase payload contains a non-file filesystem entry."
            )
    body = {
        "schema": PHASE_PAYLOAD_INVENTORY_SCHEMA,
        "directories": directories,
        "files": files,
        "total_bytes": sum(record["size_bytes"] for record in files),
    }
    return {
        **body,
        "payload_tree_sha256": _canonical_sha256(body, "payload inventory"),
    }


def _stable_file_sha256(path: Path, initial: os.stat_result) -> str:
    expected = _stat_signature(initial)
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            if _stat_signature(os.fstat(descriptor)) != expected:
                raise FoldPhaseCacheIntegrityError(
                    "Phase payload file changed while opening."
                )
            while True:
                block = os.read(descriptor, 8 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
            if _stat_signature(os.fstat(descriptor)) != expected:
                raise FoldPhaseCacheIntegrityError(
                    "Phase payload file changed while hashing."
                )
        finally:
            os.close(descriptor)
        if _stat_signature(path.stat()) != expected:
            raise FoldPhaseCacheIntegrityError(
                "Phase payload file was replaced while hashing."
            )
    except FoldPhaseCacheIntegrityError:
        raise
    except OSError as error:
        raise FoldPhaseCacheIntegrityError(
            "Cannot hash phase payload file."
        ) from error
    return digest.hexdigest()


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(stat.S_IFMT(value.st_mode)),
    )


def _strict_canonical_json_bytes(
    raw: bytes,
    name: str,
    error_type: type[DynamicFoldPhaseCacheError],
) -> dict[str, Any]:
    if not raw or len(raw) > _MAX_JSON_BYTES:
        raise error_type(f"{name} is empty or exceeds its size limit.")

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise error_type(f"{name} contains duplicate key {key!r}.")
            result[key] = value
        return result

    def invalid_constant(value: str) -> Any:
        raise error_type(f"{name} contains non-finite value {value!r}.")

    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
    except DynamicFoldPhaseCacheError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise error_type(f"{name} is not canonical ASCII JSON.") from error
    if not isinstance(value, dict):
        raise error_type(f"{name} must be a JSON object.")
    try:
        canonical = _canonical_json_bytes(value, name) + b"\n"
    except FoldPhaseKeyError as error:
        raise error_type(f"{name} is not canonical JSON.") from error
    if raw != canonical:
        raise error_type(f"{name} is not in its unique canonical encoding.")
    return value


def _absolute_local_path(value: str | os.PathLike[str], name: str) -> Path:
    raw = os.fspath(value)
    if not raw or "://" in raw:
        raise DynamicFoldPhaseCacheError(f"{name} must be a local filesystem path.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return Path(os.path.abspath(path))


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _is_reparse(path: Path) -> bool:
    try:
        value = path.lstat()
    except OSError:
        return False
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(value, "st_file_attributes", 0)
    return bool(path.is_symlink() or (flag and attributes & flag))


def _assert_no_reparse_ancestors(
    path: Path,
    error_type: type[DynamicFoldPhaseCacheError],
) -> None:
    chain: list[Path] = []
    current = path
    while True:
        chain.append(current)
        if current.parent == current:
            break
        current = current.parent
    for candidate in reversed(chain):
        if not _lexists(candidate):
            continue
        if _is_reparse(candidate):
            raise error_type("Cache paths cannot traverse symlinks or reparse points.")
        if not candidate.is_dir():
            raise error_type("Cache path ancestor is not an ordinary directory.")


def _assert_ordinary_directory(
    path: Path,
    name: str,
    error_type: type[DynamicFoldPhaseCacheError],
) -> None:
    if not _lexists(path) or _is_reparse(path) or not path.is_dir():
        raise error_type(f"{name} must be an ordinary directory.")


def _assert_ordinary_file(
    path: Path,
    name: str,
    error_type: type[DynamicFoldPhaseCacheError],
) -> None:
    if not _lexists(path) or _is_reparse(path) or not path.is_file():
        raise error_type(f"{name} must be an ordinary file.")
    try:
        if int(path.stat().st_nlink) != 1:
            raise error_type(f"{name} must not be hard-linked.")
    except OSError as error:
        raise error_type(f"Cannot inspect {name}.") from error


def _read_limited(
    path: Path,
    limit: int,
    name: str,
    error_type: type[DynamicFoldPhaseCacheError],
) -> bytes:
    descriptor: int | None = None
    try:
        before = path.lstat()
        expected = _stat_signature(before)
        if int(before.st_nlink) != 1 or not stat.S_ISREG(before.st_mode):
            raise error_type(f"{name} must be an ordinary unlinked file.")
        if before.st_size > limit:
            raise error_type(f"{name} exceeds its size limit.")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        if _stat_signature(os.fstat(descriptor)) != expected:
            raise error_type(f"{name} changed while opening.")
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining > 0:
            block = os.read(descriptor, min(8 * 1024 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw = b"".join(chunks)
        if len(raw) > limit:
            raise error_type(f"{name} exceeds its size limit.")
        if _stat_signature(os.fstat(descriptor)) != expected:
            raise error_type(f"{name} changed while reading.")
        if _stat_signature(path.lstat()) != expected:
            raise error_type(f"{name} was replaced while reading.")
        return raw
    except DynamicFoldPhaseCacheError:
        raise
    except OSError as error:
        raise error_type(f"Cannot read {name}.") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _atomic_create_bytes(
    path: Path,
    payload: bytes,
    error_type: type[DynamicFoldPhaseCacheError],
) -> None:
    if _lexists(path):
        raise error_type(f"Immutable file already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        _write_descriptor(descriptor, payload, error_type)
        os.close(descriptor)
        descriptor = -1
        _rename_file_noreplace(temporary, path, error_type)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if _lexists(temporary):
            temporary.unlink()


def _append_event_line(
    path: Path,
    line: bytes,
    *,
    expected_signature: tuple[int, int, int, int, int, int],
) -> tuple[int, int, int, int, int, int]:
    """Durably append one already-canonical event without rereading its prefix."""

    descriptor: int | None = None
    expected_size = expected_signature[3]
    flags = (
        os.O_RDWR
        | os.O_APPEND
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        before = path.lstat()
        if _stat_signature(before) != expected_signature:
            raise FoldPhaseEventLogError(
                "Phase event log changed before append."
            )
        if int(before.st_nlink) != 1 or not stat.S_ISREG(before.st_mode):
            raise FoldPhaseEventLogError(
                "event log must be an ordinary unlinked file."
            )
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if _stat_signature(opened) != expected_signature:
            raise FoldPhaseEventLogError(
                "Phase event log changed while opening for append."
            )

        offset = 0
        while offset < len(line):
            written = os.write(descriptor, line[offset:])
            if written <= 0:
                raise FoldPhaseEventLogError(
                    "Could not append the complete phase event."
                )
            offset += written
        os.fsync(descriptor)

        appended = os.fstat(descriptor)
        if (
            int(appended.st_dev) != expected_signature[0]
            or int(appended.st_ino) != expected_signature[1]
            or int(appended.st_nlink) != 1
            or int(appended.st_size) != expected_size + len(line)
            or int(stat.S_IFMT(appended.st_mode)) != expected_signature[5]
        ):
            raise FoldPhaseEventLogError(
                "Phase event log identity or size changed during append."
            )
        os.lseek(descriptor, expected_size, os.SEEK_SET)
        observed = bytearray()
        while len(observed) < len(line):
            block = os.read(descriptor, len(line) - len(observed))
            if not block:
                break
            observed.extend(block)
        if bytes(observed) != line:
            raise FoldPhaseEventLogError(
                "Appended phase event bytes do not verify."
            )

        os.close(descriptor)
        descriptor = None
        final = path.lstat()
        if (
            int(final.st_dev) != expected_signature[0]
            or int(final.st_ino) != expected_signature[1]
            or int(final.st_nlink) != 1
            or int(final.st_size) != expected_size + len(line)
            or int(stat.S_IFMT(final.st_mode)) != expected_signature[5]
        ):
            raise FoldPhaseEventLogError(
                "Phase event log was replaced after append."
            )
        return _stat_signature(final)
    except DynamicFoldPhaseCacheError:
        raise
    except OSError as error:
        raise FoldPhaseEventLogError(
            "Cannot durably append phase event."
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _atomic_replace_bytes(
    path: Path,
    payload: bytes,
    error_type: type[DynamicFoldPhaseCacheError],
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        _write_descriptor(descriptor, payload, error_type)
        os.close(descriptor)
        descriptor = -1
        if _lexists(path):
            _assert_ordinary_file(path, "event log", error_type)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except OSError as error:
        raise error_type("Cannot atomically replace event log.") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if _lexists(temporary):
            temporary.unlink()


def _write_descriptor(
    descriptor: int,
    payload: bytes,
    error_type: type[DynamicFoldPhaseCacheError],
) -> None:
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise error_type("Cannot durably write cache metadata.") from error


def _rename_file_noreplace(
    source: Path,
    destination: Path,
    error_type: type[DynamicFoldPhaseCacheError],
) -> None:
    if _lexists(destination):
        raise error_type(f"Immutable file already exists: {destination}")
    try:
        os.rename(source, destination)
    except OSError as error:
        raise error_type("Atomic immutable file publication failed.") from error


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    if _lexists(destination):
        raise FoldPhaseCacheAlreadyCompleteError(
            f"Immutable phase entry already exists: {destination}"
        )
    if os.name == "nt":
        try:
            os.rename(source, destination)
        except FileExistsError as error:
            raise FoldPhaseCacheAlreadyCompleteError(
                "Concurrent writer won the immutable phase destination."
            ) from error
        except OSError as error:
            raise FoldPhaseCachePublicationError(
                "Atomic phase publication failed."
            ) from error
        return
    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise FoldPhaseCachePublicationError(
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
        if result == 0:
            return
        code = ctypes.get_errno()
        if code in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FoldPhaseCacheAlreadyCompleteError(
                "Concurrent writer won the immutable phase destination."
            )
        raise FoldPhaseCachePublicationError(
            f"Atomic phase publication failed with errno {code}."
        )
    raise FoldPhaseCachePublicationError(
        "Atomic no-clobber publication is unsupported on this platform."
    )


def _acquire_lock(
    path: Path,
    error_type: type[DynamicFoldPhaseCacheError],
) -> tuple[Path, int, tuple[int, int]]:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        return path, descriptor, (int(metadata.st_dev), int(metadata.st_ino))
    except FileExistsError as error:
        raise error_type(f"Exclusive cache lock already exists: {path}") from error
    except OSError as error:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
            try:
                path.unlink()
            except OSError:
                pass
        raise error_type(f"Cannot acquire exclusive cache lock: {path}") from error


def _release_lock(
    lock: tuple[Path, int, tuple[int, int]],
    error_type: type[DynamicFoldPhaseCacheError],
) -> None:
    path, descriptor, identity = lock
    try:
        os.close(descriptor)
        try:
            metadata = path.stat()
        except FileNotFoundError:
            return
        if (int(metadata.st_dev), int(metadata.st_ino)) != identity:
            raise error_type("Cache lock path was replaced while held.")
        path.unlink()
        _fsync_directory(path.parent)
    except DynamicFoldPhaseCacheError:
        raise
    except OSError as error:
        raise error_type("Cannot safely release cache lock.") from error


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if _is_reparse(path):
            raise FoldPhaseCachePublicationError(
                "Cannot publish a phase payload containing a reparse point."
            )
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
            try:
                os.fsync(descriptor)
            except OSError:
                if os.name != "nt":
                    raise
            finally:
                os.close(descriptor)
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
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "COMPLETE_FILENAME",
    "DynamicFoldPhaseCache",
    "DynamicFoldPhaseCacheError",
    "EVENT_BEGIN",
    "EVENT_COMPLETE",
    "EVENT_FAIL",
    "EVENT_HEARTBEAT",
    "EVENT_HIT",
    "EVENT_KINDS",
    "EVENT_MISS",
    "EVENT_PROGRESS",
    "FoldPhaseCacheAlreadyCompleteError",
    "FoldPhaseCacheConcurrencyError",
    "FoldPhaseCacheIntegrityError",
    "FoldPhaseCachePublicationError",
    "FoldPhaseCacheTransaction",
    "FoldPhaseEvent",
    "FoldPhaseEventLog",
    "FoldPhaseEventLogError",
    "FoldPhaseKey",
    "FoldPhaseKeyError",
    "PAYLOAD_DIRECTORY_NAME",
    "PHASE_COMPLETION_SCHEMA",
    "PHASE_EVENT_SCHEMA",
    "PHASE_KEY_SCHEMA",
    "PHASE_PAYLOAD_INVENTORY_SCHEMA",
    "VerifiedFoldPhaseCacheEntry",
    "read_fold_phase_events",
]
