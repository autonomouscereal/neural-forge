"""Family-neutral immutable caching for complete child replay phases.

The generic layer owns only publication, reuse, race handling, and telemetry.
Model-family codecs remain responsible for constructing a content address and
for authenticating every semantic record and tensor in their replay object.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generic, Protocol, TypeVar

from .dynamic_fold_phase_cache import (
    DynamicFoldPhaseCache,
    FoldPhaseCacheAlreadyCompleteError,
    FoldPhaseCacheIntegrityError,
    FoldPhaseEventLog,
    FoldPhaseKey,
    VerifiedFoldPhaseCacheEntry,
)

DYNAMIC_CHILD_REPLAY_PHASE_IDENTITY_SCHEMA = (
    "neural-forge-dynamic-child-replay-phase-identity-v1"
)
DYNAMIC_CHILD_REPLAY_PHASE_RESULT_SCHEMA = (
    "neural-forge-dynamic-child-replay-phase-result-v1"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")

ReplayT = TypeVar("ReplayT")
ChildReplayPhaseProgressCallback = Callable[[str, Mapping[str, Any]], None]


class DynamicChildReplayPhaseCacheError(RuntimeError):
    """The generic child-replay phase contract is malformed."""


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
        raise DynamicChildReplayPhaseCacheError(
            f"{name} must be finite canonical JSON."
        ) from exc


def _canonical_mapping(value: Any, name: str) -> dict[str, Any]:
    normalized = json.loads(_canonical_json_bytes(value, name).decode("ascii"))
    if not isinstance(normalized, dict):
        raise DynamicChildReplayPhaseCacheError(f"{name} must be a mapping.")
    return normalized


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
        raise DynamicChildReplayPhaseCacheError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise DynamicChildReplayPhaseCacheError(
            f"{name} must be a stable identifier without path separators."
        )
    return value


@dataclass(frozen=True, slots=True)
class DynamicChildReplayPhaseIdentity:
    """Portable family-codec request bound to one generic phase key."""

    phase_key: FoldPhaseKey
    request: Mapping[str, Any]
    codec_id: str
    request_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.phase_key, FoldPhaseKey):
            raise TypeError("phase_key must be FoldPhaseKey.")
        codec_id = _identifier(self.codec_id, "codec_id")
        body = _canonical_mapping(self.request, "child replay phase request")
        if body.get("schema") != DYNAMIC_CHILD_REPLAY_PHASE_IDENTITY_SCHEMA:
            raise DynamicChildReplayPhaseCacheError(
                "Child replay phase request schema is unsupported."
            )
        if body.get("phase_key_sha256") != self.phase_key.phase_key_sha256:
            raise DynamicChildReplayPhaseCacheError(
                "Child replay phase request differs from its phase key."
            )
        if body.get("codec_id") != codec_id:
            raise DynamicChildReplayPhaseCacheError(
                "Child replay phase request differs from its codec."
            )
        expected = hashlib.sha256(
            _canonical_json_bytes(body, "child replay phase request")
        ).hexdigest()
        if self.request_sha256 not in {"", expected}:
            raise DynamicChildReplayPhaseCacheError(
                "Child replay phase request SHA-256 is invalid."
            )
        object.__setattr__(self, "codec_id", codec_id)
        object.__setattr__(self, "request", _deep_freeze(body))
        object.__setattr__(self, "request_sha256", expected)

    def as_dict(self) -> dict[str, Any]:
        return {
            **_deep_thaw(self.request),
            "request_sha256": self.request_sha256,
        }


@dataclass(frozen=True, slots=True)
class DynamicChildReplayPhaseCacheContext:
    """Optional immutable-cache and progress handles for child replay."""

    cache: DynamicFoldPhaseCache
    events: FoldPhaseEventLog | None = None
    progress: ChildReplayPhaseProgressCallback | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cache, DynamicFoldPhaseCache):
            raise TypeError("cache must be DynamicFoldPhaseCache.")
        if self.events is not None and not isinstance(self.events, FoldPhaseEventLog):
            raise TypeError("events must be FoldPhaseEventLog or None.")
        if self.progress is not None and not callable(self.progress):
            raise TypeError("progress must be callable or None.")


class DynamicChildReplayCodec(Protocol[ReplayT]):
    """Family codec for one complete replay value."""

    codec_id: str

    def authenticate(
        self,
        value: ReplayT,
        identity: DynamicChildReplayPhaseIdentity,
    ) -> str:
        """Return the authenticated replay digest or raise."""

    def write(
        self,
        value: ReplayT,
        payload_path: Path,
        identity: DynamicChildReplayPhaseIdentity,
    ) -> Mapping[str, Any]:
        """Write an immutable codec payload and return canonical receipt data."""

    def load(
        self,
        entry: VerifiedFoldPhaseCacheEntry,
        identity: DynamicChildReplayPhaseIdentity,
    ) -> ReplayT:
        """Load and authenticate one already inventory-verified entry."""


@dataclass(frozen=True, slots=True)
class AuthenticatedDynamicChildReplayPhase(Generic[ReplayT]):
    """One authenticated cache hit or cold publication result."""

    replay: ReplayT
    entry: VerifiedFoldPhaseCacheEntry
    cache_hit: bool
    producer_invoked: bool
    publication_race_lost: bool
    replay_sha256: str


def _progress(
    context: DynamicChildReplayPhaseCacheContext,
    stage: str,
    **details: Any,
) -> None:
    if context.progress is not None:
        context.progress(stage, MappingProxyType(dict(details)))


def _validate_codec(
    codec: DynamicChildReplayCodec[Any],
    identity: DynamicChildReplayPhaseIdentity,
) -> None:
    if getattr(codec, "codec_id", None) != identity.codec_id:
        raise DynamicChildReplayPhaseCacheError(
            "Child replay codec differs from the phase identity."
        )
    for name in ("authenticate", "write", "load"):
        if not callable(getattr(codec, name, None)):
            raise DynamicChildReplayPhaseCacheError(
                f"Child replay codec has no callable {name}()."
            )


def _completion_result(
    identity: DynamicChildReplayPhaseIdentity,
    replay_sha256: str,
    codec_result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": DYNAMIC_CHILD_REPLAY_PHASE_RESULT_SCHEMA,
        "codec_id": identity.codec_id,
        "request_sha256": identity.request_sha256,
        "replay_sha256": _digest(replay_sha256, "replay_sha256"),
        "codec_result": _canonical_mapping(codec_result, "codec result"),
    }


def _authenticate_entry_result(
    entry: VerifiedFoldPhaseCacheEntry,
    identity: DynamicChildReplayPhaseIdentity,
) -> dict[str, Any]:
    result = _canonical_mapping(entry.result, "phase completion result")
    if (
        set(result)
        != {
            "schema",
            "codec_id",
            "request_sha256",
            "replay_sha256",
            "codec_result",
        }
        or result.get("schema") != DYNAMIC_CHILD_REPLAY_PHASE_RESULT_SCHEMA
        or result.get("codec_id") != identity.codec_id
        or result.get("request_sha256") != identity.request_sha256
    ):
        raise FoldPhaseCacheIntegrityError(
            "Child replay phase completion differs from its request."
        )
    try:
        _digest(result.get("replay_sha256"), "replay_sha256")
    except DynamicChildReplayPhaseCacheError as exc:
        raise FoldPhaseCacheIntegrityError(
            "Child replay phase completion has an invalid replay digest."
        ) from exc
    if not isinstance(result.get("codec_result"), dict):
        raise FoldPhaseCacheIntegrityError(
            "Child replay phase completion has no codec receipt."
        )
    return result


def _load_entry(
    context: DynamicChildReplayPhaseCacheContext,
    identity: DynamicChildReplayPhaseIdentity,
    codec: DynamicChildReplayCodec[ReplayT],
    entry: VerifiedFoldPhaseCacheEntry,
) -> tuple[ReplayT, str]:
    result = _authenticate_entry_result(entry, identity)
    value = codec.load(entry, identity)
    replay_sha256 = _digest(
        codec.authenticate(value, identity),
        "authenticated replay SHA-256",
    )
    if replay_sha256 != result["replay_sha256"]:
        raise FoldPhaseCacheIntegrityError(
            "Loaded child replay differs from the completion result."
        )
    reverified = context.cache.lookup(identity.phase_key)
    if (
        reverified is None
        or reverified.completion_sha256 != entry.completion_sha256
        or dict(reverified.payload_inventory) != dict(entry.payload_inventory)
        or dict(reverified.result) != dict(entry.result)
    ):
        raise FoldPhaseCacheIntegrityError(
            "Child replay phase changed while it was being loaded."
        )
    return value, replay_sha256


def load_or_build_dynamic_child_replay_phase(
    context: DynamicChildReplayPhaseCacheContext,
    identity: DynamicChildReplayPhaseIdentity,
    *,
    codec: DynamicChildReplayCodec[ReplayT],
    producer: Callable[[], ReplayT],
) -> AuthenticatedDynamicChildReplayPhase[ReplayT]:
    """Return a hot hit or atomically publish one complete child replay."""

    if not isinstance(context, DynamicChildReplayPhaseCacheContext):
        raise TypeError("context must be DynamicChildReplayPhaseCacheContext.")
    if not isinstance(identity, DynamicChildReplayPhaseIdentity):
        raise TypeError("identity must be DynamicChildReplayPhaseIdentity.")
    if not callable(producer):
        raise TypeError("producer must be callable.")
    _validate_codec(codec, identity)
    key = identity.phase_key
    _progress(
        context,
        "dynamic_child_replay_phase_cache_lookup",
        phase_key_sha256=key.phase_key_sha256,
        codec_id=identity.codec_id,
    )
    entry = context.cache.lookup(key, events=context.events)
    if entry is not None:
        replay, replay_sha256 = _load_entry(context, identity, codec, entry)
        _progress(
            context,
            "dynamic_child_replay_phase_cache_hit",
            phase_key_sha256=key.phase_key_sha256,
            completion_sha256=entry.completion_sha256,
            producer_invoked=False,
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
    )
    _progress(
        context,
        "dynamic_child_replay_phase_producer_started",
        phase_key_sha256=key.phase_key_sha256,
    )
    cold = producer()
    cold_sha256 = _digest(
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
    )
    try:
        transaction = context.cache.begin(key, events=context.events)
    except FoldPhaseCacheAlreadyCompleteError:
        raced = context.cache.lookup(key, events=context.events)
        if raced is None:  # pragma: no cover - defensive concurrent deletion
            raise FoldPhaseCacheIntegrityError(
                "Concurrent child-replay publication disappeared."
            )
        _raced_replay, replay_sha256 = _load_entry(
            context,
            identity,
            codec,
            raced,
        )
        if replay_sha256 != cold_sha256:
            raise FoldPhaseCacheIntegrityError(
                "Concurrent child-replay publication differs from the local "
                "producer result."
            )
        if _digest(
            codec.authenticate(cold, identity),
            "post-race local replay SHA-256",
        ) != cold_sha256:
            raise FoldPhaseCacheIntegrityError(
                "Local child replay changed while the concurrent winner was "
                "authenticated."
            )
        _progress(
            context,
            "dynamic_child_replay_phase_publish_race_resumed",
            phase_key_sha256=key.phase_key_sha256,
            completion_sha256=raced.completion_sha256,
        )
        return AuthenticatedDynamicChildReplayPhase(
            # The concurrent winner proved the same sealed value. Preserve
            # this invocation's unsealed execution telemetry by returning the
            # local value, exactly as the ordinary cold-publication path does.
            replay=cold,
            entry=raced,
            cache_hit=True,
            producer_invoked=True,
            publication_race_lost=True,
            replay_sha256=replay_sha256,
        )

    with transaction:
        codec_result = codec.write(cold, transaction.payload_path, identity)
        entry = transaction.complete(
            result=_completion_result(
                identity,
                cold_sha256,
                codec_result,
            )
        )
    replay, replay_sha256 = _load_entry(context, identity, codec, entry)
    if replay_sha256 != cold_sha256:
        raise FoldPhaseCacheIntegrityError(
            "Cold and reloaded child replay digests differ."
        )
    if _digest(
        codec.authenticate(cold, identity),
        "post-publication cold replay SHA-256",
    ) != cold_sha256:
        raise FoldPhaseCacheIntegrityError(
            "Cold child replay changed while its payload was published."
        )
    _progress(
        context,
        "dynamic_child_replay_phase_published",
        phase_key_sha256=key.phase_key_sha256,
        completion_sha256=entry.completion_sha256,
        replay_sha256=replay_sha256,
        payload_bytes=entry.payload_inventory["total_bytes"],
        cold_reload_exact=True,
    )
    return AuthenticatedDynamicChildReplayPhase(
        # Preserve non-semantic producer telemetry (for example sharding wall
        # clock data) on the cold invocation. The reloaded value above proved
        # the durable bytes exact; hot hits intentionally carry no historical
        # runtime timing.
        replay=cold,
        entry=entry,
        cache_hit=False,
        producer_invoked=True,
        publication_race_lost=False,
        replay_sha256=replay_sha256,
    )


__all__ = [
    "DYNAMIC_CHILD_REPLAY_PHASE_IDENTITY_SCHEMA",
    "DYNAMIC_CHILD_REPLAY_PHASE_RESULT_SCHEMA",
    "AuthenticatedDynamicChildReplayPhase",
    "ChildReplayPhaseProgressCallback",
    "DynamicChildReplayCodec",
    "DynamicChildReplayPhaseCacheContext",
    "DynamicChildReplayPhaseCacheError",
    "DynamicChildReplayPhaseIdentity",
    "load_or_build_dynamic_child_replay_phase",
]
