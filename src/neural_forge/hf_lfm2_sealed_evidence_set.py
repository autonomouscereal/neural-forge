"""One ordered sealed evidence set: a base cache plus on-policy extensions.

A closed-form LFM2 writer historically consumed exactly one sealed circuit
cache.  On-policy evidence rounds (``hf_lfm2_on_policy_evidence_round``) seal
*additional* immutable caches that extend that corpus with states the child
actually reached.  This module is the read-only aggregate a solve pass binds to
instead of a bare cache:

* it owns the selective authenticated sessions of every member under one
  fail-closed lifecycle, so a half-opened set never leaks a pinned descriptor;
* it iterates every batch across every member with full provenance, so a
  consumer can always say which sealed cache and which sealed batch content
  hash a row came from;
* it publishes one combined evidence identity digest, which is exactly the
  digest the production runner already folds into every derived phase-cache
  key, so a base-only solve can never be silently reused for a base+extension
  request;
* it keeps the two hard rules in one place: **every extension row is a fit
  row** (a child-visited state must never become a ``selection`` row, which
  would let the child pick its own frontier, nor a ``validation`` row, which
  would stop that split being held out), and **no member is ever executed** —
  every member is an already sealed, immutable cache, so binding an evidence
  set performs zero parent forwards.

With no extensions the set is a pass-through: its cache key, corpus
fingerprint, splits, and batch inventory are byte-identical to the base cache's
own, and it opens nothing the caller did not already open.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from .dynamic_circuit_ir import canonical_sha256
from .hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceCache,
    Lfm2CircuitTraceSelectiveAuthenticatedSession,
)


# The combined identity schema is shared with the production catalog runner so
# both sides derive the same digest from one definition.
ON_POLICY_EVIDENCE_IDENTITY_SCHEMA = (
    "neural-forge-production-lfm2-on-policy-evidence-identity-v1"
)
SEALED_EVIDENCE_SET_SCHEMA = "neural-forge-lfm2-sealed-evidence-set-v1"
SEALED_EVIDENCE_SET_CACHE_KEY_SCHEMA = (
    "neural-forge-lfm2-sealed-evidence-set-cache-key-v1"
)
BASE_EVIDENCE_ROLE = "base"
EXTENSION_EVIDENCE_ROLE = "on_policy_extension"
_CORPUS_LIST_FIELDS = (
    "forced_decode_token_counts",
    "ordered_batch_input_sha256",
    "prompt_token_counts",
)
_CORPUS_FIELDS = frozenset({"batch_count", "schema", *_CORPUS_LIST_FIELDS})
_SPLIT_LABELS = frozenset({"fit", "selection", "validation"})


class Lfm2SealedEvidenceSetError(RuntimeError):
    """A sealed evidence set could not be formed or held open."""


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2SealedEvidenceSetError(f"{name} must be a lowercase sha256 digest.")
    return value


def lfm2_on_policy_evidence_identity_sha256(
    *,
    base_cache_key: str,
    extension_cache_keys: Sequence[str],
    extension_batch_counts: Sequence[int],
    round_manifest_sha256: Sequence[str],
) -> str | None:
    """Digest every extension cache and round manifest one solve binds to.

    Returns ``None`` when no round is consumed, which keeps every derived
    phase-cache key byte-identical to the pre-extension revision.  This is the
    single definition of that digest: the runner's sealed-split identity, its
    balanced-basis phase dependency, and this aggregate all call it.
    """

    manifests = tuple(str(value) for value in round_manifest_sha256)
    if not manifests:
        return None
    keys = tuple(str(value) for value in extension_cache_keys)
    counts = [int(value) for value in extension_batch_counts]
    if len(keys) != len(manifests) or len(counts) != len(manifests):
        raise Lfm2SealedEvidenceSetError(
            "On-policy extension cache keys, batch counts, and round manifest "
            "digests must describe the same ordered rounds."
        )
    return canonical_sha256(
        {
            "schema": ON_POLICY_EVIDENCE_IDENTITY_SCHEMA,
            "base_cache_key": str(base_cache_key),
            "extension_cache_keys": list(keys),
            "extension_batch_counts": counts,
            "round_manifest_sha256": list(manifests),
            "round_count": len(manifests),
            "all_extension_rows_are_fit_rows": True,
        }
    )


@dataclass(frozen=True, slots=True)
class Lfm2SealedEvidenceRequest:
    """One sealed cache offered to an evidence set with its split assignment."""

    cache: Lfm2CircuitTraceCache
    splits: tuple[str, ...]
    split_sha256: str
    round_manifest_sha256: str | None = None
    all_rows_are_fit: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.cache, Lfm2CircuitTraceCache):
            raise Lfm2SealedEvidenceSetError(
                "Sealed evidence members must be Lfm2CircuitTraceCache handles."
            )
        object.__setattr__(self, "splits", tuple(str(v) for v in self.splits))
        _digest(self.split_sha256, "split_sha256")
        if self.round_manifest_sha256 is not None:
            _digest(self.round_manifest_sha256, "round_manifest_sha256")
        if not isinstance(self.all_rows_are_fit, bool):
            raise Lfm2SealedEvidenceSetError("all_rows_are_fit must be bool.")
        if len(self.splits) != self.cache.batch_count or not self.splits:
            raise Lfm2SealedEvidenceSetError(
                "Sealed evidence split assignment does not cover its cache."
            )
        unexpected = sorted(set(self.splits).difference(_SPLIT_LABELS))
        if unexpected:
            raise Lfm2SealedEvidenceSetError(
                f"Unsupported sealed evidence split labels: {unexpected}."
            )
        if self.all_rows_are_fit and set(self.splits) != {"fit"}:
            raise Lfm2SealedEvidenceSetError(
                "Every row an on-policy extension contributes must be a fit "
                "row; selection would let the child pick its own frontier and "
                "validation would stop being held out."
            )


@dataclass(frozen=True, slots=True)
class Lfm2SealedEvidenceMember:
    """One authenticated member of an ordered sealed evidence set."""

    role: str
    ordinal: int
    cache: Lfm2CircuitTraceCache
    splits: tuple[str, ...]
    split_sha256: str
    batch_offset: int
    round_manifest_sha256: str | None
    session: Lfm2CircuitTraceSelectiveAuthenticatedSession | None = None
    session_owned: bool = False

    @property
    def cache_key(self) -> str:
        return self.cache.cache_key

    @property
    def batch_count(self) -> int:
        return self.cache.batch_count

    @property
    def all_rows_are_fit(self) -> bool:
        return set(self.splits) == {"fit"}

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "ordinal": self.ordinal,
            "path": str(self.cache.path),
            "cache_key": self.cache_key,
            "corpus_sha256": self.cache.corpus_fingerprint,
            "batch_offset": self.batch_offset,
            "batch_count": self.batch_count,
            "ordered_splits": list(self.splits),
            "split_sha256": self.split_sha256,
            "round_manifest_sha256": self.round_manifest_sha256,
            "all_rows_are_fit_rows": self.all_rows_are_fit,
            "selective_session_sha256": (
                None if self.session is None else self.session.evidence_sha256
            ),
            "selective_session_owned_by_evidence_set": self.session_owned,
        }


@dataclass(frozen=True, slots=True)
class Lfm2SealedEvidenceBatch:
    """One sealed batch with the provenance a consumer must be able to cite."""

    merged_index: int
    member_ordinal: int
    member_role: str
    member_cache_key: str
    member_batch_index: int
    input_sha256: str
    split: str

    @property
    def is_fit(self) -> bool:
        return self.split == "fit"

    def as_dict(self) -> dict[str, Any]:
        return {
            "merged_index": self.merged_index,
            "member_ordinal": self.member_ordinal,
            "member_role": self.member_role,
            "member_cache_key": self.member_cache_key,
            "member_batch_index": self.member_batch_index,
            "input_sha256": self.input_sha256,
            "split": self.split,
        }


def _corpus_body(cache: Lfm2CircuitTraceCache) -> dict[str, Any]:
    """Return one sealed corpus body, proven against its own sealed digest.

    The merged corpus fingerprint is derived from the members' own sealed
    corpus bodies rather than restating the cache module's recipe.  Recomputing
    each member's sealed digest from its own body proves that derivation on
    real data and fails closed if the cache module ever changes shape.
    """

    corpus = dict(cache.manifest["corpus"])
    sealed = corpus.pop("corpus_sha256", None)
    if set(corpus) != _CORPUS_FIELDS or not isinstance(sealed, str):
        raise Lfm2SealedEvidenceSetError(
            f"Sealed cache {cache.cache_key} has an unsupported corpus inventory."
        )
    if canonical_sha256(corpus) != sealed:
        raise Lfm2SealedEvidenceSetError(
            f"Sealed cache {cache.cache_key} corpus digest is not reproducible."
        )
    if int(corpus["batch_count"]) != cache.batch_count or any(
        len(corpus[name]) != cache.batch_count for name in _CORPUS_LIST_FIELDS
    ):
        raise Lfm2SealedEvidenceSetError(
            f"Sealed cache {cache.cache_key} corpus inventory is inconsistent."
        )
    return corpus


class Lfm2SealedEvidenceSet:
    """An ordered ``(base, *on-policy extensions)`` sealed evidence aggregate."""

    __slots__ = (
        "_batches",
        "_cache_key",
        "_closed",
        "_corpus_fingerprint",
        "_evidence_identity_sha256",
        "_members",
    )

    def __init__(
        self,
        members: Sequence[Lfm2SealedEvidenceMember],
        *,
        cache_key: str,
        corpus_fingerprint: str,
        evidence_identity_sha256: str | None,
        batches: Sequence[Lfm2SealedEvidenceBatch],
    ) -> None:
        self._members = tuple(members)
        self._cache_key = cache_key
        self._corpus_fingerprint = corpus_fingerprint
        self._evidence_identity_sha256 = evidence_identity_sha256
        self._batches = tuple(batches)
        self._closed = False

    # ---------------------------------------------------------------- opening

    @classmethod
    def open(
        cls,
        base: Lfm2SealedEvidenceRequest,
        extensions: Sequence[Lfm2SealedEvidenceRequest] = (),
        *,
        base_session: Lfm2CircuitTraceSelectiveAuthenticatedSession | None = None,
        open_sessions: bool = True,
        open_base_session: bool = True,
    ) -> Lfm2SealedEvidenceSet:
        """Bind one base cache plus ordered extensions under one lifecycle.

        Every selective session this call opens is closed again if any later
        member fails to open, so a partially constructed set never survives and
        never leaks an authenticated descriptor.  ``base_session`` is a session
        the caller already opened: it is held but never owned, so the caller's
        existing cleanup keeps closing it exactly once.  ``open_base_session``
        stays available for callers that deliberately defer the base payload
        session until a payload is first required.
        """

        if not isinstance(base, Lfm2SealedEvidenceRequest):
            raise Lfm2SealedEvidenceSetError(
                "The base member must be an Lfm2SealedEvidenceRequest."
            )
        if base.round_manifest_sha256 is not None or base.all_rows_are_fit:
            raise Lfm2SealedEvidenceSetError(
                "The base member is teacher-forced evidence and carries no "
                "round manifest."
            )
        if isinstance(extensions, (str, bytes)) or not isinstance(
            extensions,
            Sequence,
        ):
            raise TypeError("extensions must be a sequence of requests.")
        ordered = tuple(extensions)
        for request in ordered:
            if not isinstance(request, Lfm2SealedEvidenceRequest):
                raise Lfm2SealedEvidenceSetError(
                    "Extension members must be Lfm2SealedEvidenceRequest values."
                )
            if request.round_manifest_sha256 is None or not request.all_rows_are_fit:
                raise Lfm2SealedEvidenceSetError(
                    "An on-policy extension member requires its sealed round "
                    "manifest digest and an all-fit split assignment."
                )
        if base_session is not None and (
            base_session.cache_key != base.cache.cache_key
            or base_session.path != base.cache.path
            or len(base_session) != base.cache.batch_count
        ):
            raise Lfm2SealedEvidenceSetError(
                "The supplied base selective session belongs to another cache."
            )

        requests = (base, *ordered)
        seen_paths = {base.cache.path}
        seen_keys = {base.cache.cache_key}
        for request in ordered:
            if request.cache.path in seen_paths:
                raise Lfm2SealedEvidenceSetError(
                    "Sealed evidence members must occupy distinct paths."
                )
            if request.cache.cache_key in seen_keys:
                raise Lfm2SealedEvidenceSetError(
                    "Sealed evidence members must have distinct cache keys."
                )
            if request.cache.parent_fingerprint != base.cache.parent_fingerprint:
                raise Lfm2SealedEvidenceSetError(
                    "Every sealed evidence member must extend the same parent."
                )
            if (
                request.cache.implementation_fingerprint
                != base.cache.implementation_fingerprint
                or request.cache.layer_topology_fingerprint
                != base.cache.layer_topology_fingerprint
            ):
                raise Lfm2SealedEvidenceSetError(
                    "Sealed evidence members disagree on implementation or "
                    "layer topology; one closed-form solve cannot consume them."
                )
            seen_paths.add(request.cache.path)
            seen_keys.add(request.cache.cache_key)

        corpora = tuple(_corpus_body(request.cache) for request in requests)
        ordered_inputs = [
            str(value)
            for corpus in corpora
            for value in corpus["ordered_batch_input_sha256"]
        ]
        # Checked before anything is acquired: a duplicated batch would double
        # count one sealed state, and regenerating already sealed evidence is a
        # closed path.
        if len(set(ordered_inputs)) != len(ordered_inputs):
            raise Lfm2SealedEvidenceSetError(
                "Sealed evidence members repeat a batch input; regenerating "
                "already sealed evidence is a closed path."
            )
        opened: list[Lfm2CircuitTraceSelectiveAuthenticatedSession] = []
        members: list[Lfm2SealedEvidenceMember] = []
        try:
            offset = 0
            for ordinal, (request, corpus) in enumerate(zip(requests, corpora)):
                session = base_session if ordinal == 0 else None
                owned = False
                may_open = open_sessions and (ordinal > 0 or open_base_session)
                if session is None and may_open:
                    session = request.cache.selective_authenticated_session()
                    opened.append(session)
                    owned = True
                members.append(
                    Lfm2SealedEvidenceMember(
                        role=(
                            BASE_EVIDENCE_ROLE
                            if ordinal == 0
                            else EXTENSION_EVIDENCE_ROLE
                        ),
                        ordinal=ordinal,
                        cache=request.cache,
                        splits=request.splits,
                        split_sha256=request.split_sha256,
                        batch_offset=offset,
                        round_manifest_sha256=request.round_manifest_sha256,
                        session=session,
                        session_owned=owned,
                    )
                )
                offset += int(corpus["batch_count"])
        except BaseException as exc:
            for session in reversed(opened):
                try:
                    session.close()
                except BaseException as close_exc:
                    exc.add_note(
                        "Sealed evidence session cleanup also failed: "
                        f"{type(close_exc).__qualname__}: {close_exc}"
                    )
            raise

        merged_corpus = {
            "schema": corpora[0]["schema"],
            "batch_count": offset,
            **{
                name: [
                    value
                    for corpus in corpora
                    for value in corpus[name]
                ]
                for name in _CORPUS_LIST_FIELDS
            },
        }
        extension_keys = tuple(member.cache_key for member in members[1:])
        extension_counts = tuple(member.batch_count for member in members[1:])
        round_manifests = tuple(
            str(member.round_manifest_sha256) for member in members[1:]
        )
        identity = lfm2_on_policy_evidence_identity_sha256(
            base_cache_key=base.cache.cache_key,
            extension_cache_keys=extension_keys,
            extension_batch_counts=extension_counts,
            round_manifest_sha256=round_manifests,
        )
        cache_key = (
            base.cache.cache_key
            if identity is None
            else canonical_sha256(
                {
                    "schema": SEALED_EVIDENCE_SET_CACHE_KEY_SCHEMA,
                    "base_cache_key": base.cache.cache_key,
                    "ordered_member_cache_keys": [
                        member.cache_key for member in members
                    ],
                    "on_policy_evidence_identity_sha256": identity,
                }
            )
        )
        corpus_fingerprint = (
            base.cache.corpus_fingerprint
            if identity is None
            else canonical_sha256(merged_corpus)
        )
        batches = [
            Lfm2SealedEvidenceBatch(
                merged_index=member.batch_offset + index,
                member_ordinal=member.ordinal,
                member_role=member.role,
                member_cache_key=member.cache_key,
                member_batch_index=index,
                input_sha256=str(
                    corpus["ordered_batch_input_sha256"][index]
                ),
                split=member.splits[index],
            )
            for member, corpus in zip(members, corpora)
            for index in range(member.batch_count)
        ]
        if [value.input_sha256 for value in batches] != ordered_inputs:
            for session in reversed(opened):
                session.close()
            raise Lfm2SealedEvidenceSetError(
                "Sealed evidence batch inventory is inconsistent."
            )
        return cls(
            members,
            cache_key=cache_key,
            corpus_fingerprint=corpus_fingerprint,
            evidence_identity_sha256=identity,
            batches=batches,
        )

    # ------------------------------------------------------------- properties

    @property
    def members(self) -> tuple[Lfm2SealedEvidenceMember, ...]:
        return self._members

    @property
    def base(self) -> Lfm2SealedEvidenceMember:
        return self._members[0]

    @property
    def base_cache(self) -> Lfm2CircuitTraceCache:
        return self._members[0].cache

    @property
    def extension_members(self) -> tuple[Lfm2SealedEvidenceMember, ...]:
        return self._members[1:]

    @property
    def caches(self) -> tuple[Lfm2CircuitTraceCache, ...]:
        return tuple(member.cache for member in self._members)

    @property
    def extension_caches(self) -> tuple[Lfm2CircuitTraceCache, ...]:
        return tuple(member.cache for member in self._members[1:])

    @property
    def sessions(
        self,
    ) -> tuple[Lfm2CircuitTraceSelectiveAuthenticatedSession | None, ...]:
        return tuple(member.session for member in self._members)

    @property
    def has_extension_evidence(self) -> bool:
        return len(self._members) > 1

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def member_count(self) -> int:
        return len(self._members)

    @property
    def batch_count(self) -> int:
        return len(self._batches)

    @property
    def cache_key(self) -> str:
        """One fingerprint for the exact evidence a solve pass binds to."""

        return self._cache_key

    @property
    def corpus_fingerprint(self) -> str:
        return self._corpus_fingerprint

    @property
    def parent_fingerprint(self) -> str:
        return self.base_cache.parent_fingerprint

    @property
    def evidence_identity_sha256(self) -> str | None:
        return self._evidence_identity_sha256

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(value.split for value in self._batches)

    @property
    def split_sha256s(self) -> tuple[str, ...]:
        return tuple(member.split_sha256 for member in self._members)

    @property
    def extension_split_sha256s(self) -> tuple[str, ...]:
        return tuple(member.split_sha256 for member in self._members[1:])

    @property
    def extension_cache_keys(self) -> tuple[str, ...]:
        return tuple(member.cache_key for member in self._members[1:])

    @property
    def extension_batch_counts(self) -> tuple[int, ...]:
        return tuple(member.batch_count for member in self._members[1:])

    @property
    def round_manifest_sha256(self) -> tuple[str, ...]:
        return tuple(
            str(member.round_manifest_sha256) for member in self._members[1:]
        )

    @property
    def round_count(self) -> int:
        return len(self._members) - 1

    def __len__(self) -> int:
        return len(self._batches)

    # ------------------------------------------------------------- consumption

    def fit_caches(self) -> tuple[Lfm2CircuitTraceCache, ...]:
        """Every member contributing fit rows, in canonical evidence order.

        This is the accessor a closed-form writer iterates.  It exists so the
        "no parent forward" guard lives in one place: every value it returns is
        an already sealed immutable cache, never a model.
        """

        self._assert_open()
        return tuple(
            member.cache for member in self._members if "fit" in member.splits
        )

    def iter_batches(self) -> Iterator[Lfm2SealedEvidenceBatch]:
        """Iterate every sealed batch with its member and content provenance."""

        self._assert_open()
        return iter(self._batches)

    def batch(self, merged_index: int) -> Lfm2SealedEvidenceBatch:
        if isinstance(merged_index, bool) or not isinstance(merged_index, int):
            raise TypeError("merged_index must be an integer.")
        if merged_index < 0 or merged_index >= len(self._batches):
            raise IndexError(
                f"merged_index must be in [0, {len(self._batches)}), "
                f"got {merged_index}."
            )
        return self._batches[merged_index]

    def member_for_batch(self, merged_index: int) -> Lfm2SealedEvidenceMember:
        return self._members[self.batch(merged_index).member_ordinal]

    def session_for(
        self,
        member: Lfm2SealedEvidenceMember,
    ) -> Lfm2CircuitTraceSelectiveAuthenticatedSession | None:
        self._assert_open()
        if not any(value is member for value in self._members):
            raise Lfm2SealedEvidenceSetError(
                "The requested member does not belong to this evidence set."
            )
        return member.session

    # ---------------------------------------------------------------- lifecycle

    def _assert_open(self) -> None:
        if self._closed:
            raise Lfm2SealedEvidenceSetError("Sealed evidence set is closed.")

    def verify(self) -> Lfm2SealedEvidenceSet:
        """Recheck every held session and every member binding, fail closed."""

        self._assert_open()
        for member in self._members:
            if member.session is not None:
                member.session.verify()
                if (
                    member.session.cache_key != member.cache_key
                    or member.session.path != member.cache.path
                    or len(member.session) != member.batch_count
                ):
                    raise Lfm2SealedEvidenceSetError(
                        "A sealed evidence session no longer matches its member."
                    )
            elif member.role != BASE_EVIDENCE_ROLE:
                raise Lfm2SealedEvidenceSetError(
                    "An on-policy extension member lost its authenticated "
                    "session."
                )
        return self

    def close(self) -> None:
        """Close every session this set opened; never the caller's own."""

        if self._closed:
            return
        error: BaseException | None = None
        for member in reversed(self._members):
            if member.session is None or not member.session_owned:
                continue
            try:
                member.session.close()
            except BaseException as exc:  # noqa: BLE001 - aggregate and re-raise
                if error is None:
                    error = exc
                else:
                    error.add_note(
                        "Sealed evidence session close also failed: "
                        f"{type(exc).__qualname__}: {exc}"
                    )
        self._closed = True
        if error is not None:
            raise error

    def __enter__(self) -> Lfm2SealedEvidenceSet:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ report

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SEALED_EVIDENCE_SET_SCHEMA,
            "cache_key": self.cache_key,
            "base_cache_key": self.base_cache.cache_key,
            "corpus_sha256": self.corpus_fingerprint,
            "parent_identity_sha256": self.parent_fingerprint,
            "member_count": self.member_count,
            "batch_count": self.batch_count,
            "round_count": self.round_count,
            "on_policy_evidence_identity_sha256": (
                self.evidence_identity_sha256
            ),
            "members": [member.as_dict() for member in self._members],
            "ordered_splits": list(self.splits),
            "all_extension_rows_are_fit_rows": True,
            "extension_rows_are_never_review_rows": True,
            "parent_forward_calls": 0,
            "training_steps": 0,
        }

    def receipt_fields(self) -> dict[str, Any]:
        """Compact identities a run receipt records for this evidence set."""

        return {
            "evidence_set_cache_key": self.cache_key,
            "evidence_set_corpus_sha256": self.corpus_fingerprint,
            "base_cache_key": self.base_cache.cache_key,
            "member_count": self.member_count,
            "batch_count": self.batch_count,
            "base_batch_count": self.base.batch_count,
            "extension_cache_keys": list(self.extension_cache_keys),
            "extension_batch_counts": list(self.extension_batch_counts),
            "extension_split_sha256": list(self.extension_split_sha256s),
            "base_split_sha256": self.base.split_sha256,
            "round_manifest_sha256": list(self.round_manifest_sha256),
            "on_policy_evidence_identity_sha256": (
                self.evidence_identity_sha256
            ),
            "fit_batch_count": sum(
                1 for value in self._batches if value.is_fit
            ),
            "extension_fit_batch_count": sum(
                1
                for value in self._batches
                if value.member_role == EXTENSION_EVIDENCE_ROLE
            ),
            "extension_selection_or_validation_batch_count": sum(
                1
                for value in self._batches
                if value.member_role == EXTENSION_EVIDENCE_ROLE
                and not value.is_fit
            ),
            "all_extension_rows_are_fit_rows": True,
            "parent_forward_calls": 0,
        }


__all__ = [
    "BASE_EVIDENCE_ROLE",
    "EXTENSION_EVIDENCE_ROLE",
    "ON_POLICY_EVIDENCE_IDENTITY_SCHEMA",
    "SEALED_EVIDENCE_SET_CACHE_KEY_SCHEMA",
    "SEALED_EVIDENCE_SET_SCHEMA",
    "Lfm2SealedEvidenceBatch",
    "Lfm2SealedEvidenceMember",
    "Lfm2SealedEvidenceRequest",
    "Lfm2SealedEvidenceSet",
    "Lfm2SealedEvidenceSetError",
    "lfm2_on_policy_evidence_identity_sha256",
]
