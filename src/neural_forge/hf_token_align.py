from __future__ import annotations

import hashlib
import json
import math
import operator
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import Mapping
from typing import Protocol
from typing import Sequence


class HFTokenizerProtocol(Protocol):
    """Small tokenizer surface required by byte-faithful alignment."""

    def __len__(self) -> int: ...

    def decode(self, token_ids: Sequence[int], **kwargs: Any) -> str: ...

    def encode(self, text: str, **kwargs: Any) -> Sequence[int]: ...


class AlignmentClass(str, Enum):
    EXACT_TOKEN = "exact-token"
    UTF8_BYTE_SPAN = "utf8-byte-span"
    UNALIGNABLE = "unalignable"


@dataclass(frozen=True)
class AlignmentCandidate:
    classification: AlignmentClass
    donor_token_ids: tuple[int, ...]
    origins: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.classification is AlignmentClass.UNALIGNABLE:
            raise ValueError("An unalignable row cannot have a donor candidate.")
        if not self.donor_token_ids:
            raise ValueError("A donor candidate must contain at least one token ID.")
        if any(token_id < 0 for token_id in self.donor_token_ids):
            raise ValueError("Donor token IDs must be non-negative.")
        if not self.origins:
            raise ValueError(
                "A donor candidate must record at least one discovery origin."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification.value,
            "donor_token_ids": list(self.donor_token_ids),
            "origins": list(self.origins),
        }


@dataclass(frozen=True)
class TokenAlignment:
    target_token_id: int
    target_text: str | None
    target_utf8_hex: str | None
    classification: AlignmentClass
    candidates: tuple[AlignmentCandidate, ...]
    reason: str | None = None
    diagnostics: tuple[str, ...] = ()
    candidate_search_truncated: bool = False

    def __post_init__(self) -> None:
        if self.target_token_id < 0:
            raise ValueError("Target token ID must be non-negative.")
        if self.classification is AlignmentClass.UNALIGNABLE:
            if self.candidates:
                raise ValueError("An unalignable row cannot contain candidates.")
            if not self.reason:
                raise ValueError("An unalignable row must record a reason.")
        elif not self.candidates:
            raise ValueError("An aligned row must contain at least one candidate.")
        elif any(
            candidate.classification is not self.classification
            for candidate in self.candidates
        ):
            raise ValueError("All candidates must match the row classification.")

    @property
    def is_ambiguous(self) -> bool:
        return len(self.candidates) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_token_id": self.target_token_id,
            "target_text": self.target_text,
            "target_utf8_hex": self.target_utf8_hex,
            "classification": self.classification.value,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "candidate_count": len(self.candidates),
            "ambiguous": self.is_ambiguous,
            "candidate_search_truncated": self.candidate_search_truncated,
            "reason": self.reason,
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True)
class TokenSurfaceIssue:
    token_id: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"token_id": self.token_id, "reason": self.reason}


@dataclass(frozen=True)
class AlignmentCoverage:
    target_token_count: int
    exact_token_count: int
    utf8_byte_span_count: int
    unalignable_count: int
    aligned_count: int
    unique_aligned_count: int
    ambiguous_count: int
    truncated_search_count: int
    coverage_fraction: float
    unique_coverage_fraction: float
    exact_fraction: float
    utf8_byte_span_fraction: float
    unalignable_fraction: float
    ambiguity_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_token_count": self.target_token_count,
            "exact_token_count": self.exact_token_count,
            "utf8_byte_span_count": self.utf8_byte_span_count,
            "unalignable_count": self.unalignable_count,
            "aligned_count": self.aligned_count,
            "unique_aligned_count": self.unique_aligned_count,
            "ambiguous_count": self.ambiguous_count,
            "truncated_search_count": self.truncated_search_count,
            "coverage_fraction": self.coverage_fraction,
            "unique_coverage_fraction": self.unique_coverage_fraction,
            "exact_fraction": self.exact_fraction,
            "utf8_byte_span_fraction": self.utf8_byte_span_fraction,
            "unalignable_fraction": self.unalignable_fraction,
            "ambiguity_fraction": self.ambiguity_fraction,
        }


@dataclass(frozen=True)
class TokenizerAlignmentReport:
    donor_tokenizer_class: str
    target_tokenizer_class: str
    donor_vocab_size: int
    target_vocab_size: int
    include_special_tokens: bool
    max_span_tokens: int
    max_span_candidates: int | None
    donor_special_token_ids: tuple[int, ...]
    target_special_token_ids: tuple[int, ...]
    donor_surface_issues: tuple[TokenSurfaceIssue, ...]
    donor_collision_surface_count: int
    donor_collision_token_count: int
    alignments: tuple[TokenAlignment, ...]
    coverage: AlignmentCoverage

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "direction": "donor-to-target",
            "donor_tokenizer_class": self.donor_tokenizer_class,
            "target_tokenizer_class": self.target_tokenizer_class,
            "donor_vocab_size": self.donor_vocab_size,
            "target_vocab_size": self.target_vocab_size,
            "include_special_tokens": self.include_special_tokens,
            "max_span_tokens": self.max_span_tokens,
            "max_span_candidates": self.max_span_candidates,
            "donor_special_token_ids": list(self.donor_special_token_ids),
            "target_special_token_ids": list(self.target_special_token_ids),
            "donor_surface_issues": [
                issue.to_dict() for issue in self.donor_surface_issues
            ],
            "donor_collision_surface_count": self.donor_collision_surface_count,
            "donor_collision_token_count": self.donor_collision_token_count,
            "coverage": self.coverage.to_dict(),
            "alignments": [alignment.to_dict() for alignment in self.alignments],
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TransportAlignmentThresholds:
    """Acceptance limits for using the alignment to initialize embeddings or heads."""

    min_coverage_fraction: float = 0.95
    min_unique_coverage_fraction: float = 0.95
    max_unalignable_fraction: float = 0.05
    max_ambiguity_fraction: float = 0.0
    allow_truncated_candidate_search: bool = False
    require_nonempty_target: bool = True

    def __post_init__(self) -> None:
        for name in (
            "min_coverage_fraction",
            "min_unique_coverage_fraction",
            "max_unalignable_fraction",
            "max_ambiguity_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1].")

    @classmethod
    def strict(cls) -> TransportAlignmentThresholds:
        return cls(
            min_coverage_fraction=1.0,
            min_unique_coverage_fraction=1.0,
            max_unalignable_fraction=0.0,
            max_ambiguity_fraction=0.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_coverage_fraction": self.min_coverage_fraction,
            "min_unique_coverage_fraction": self.min_unique_coverage_fraction,
            "max_unalignable_fraction": self.max_unalignable_fraction,
            "max_ambiguity_fraction": self.max_ambiguity_fraction,
            "allow_truncated_candidate_search": self.allow_truncated_candidate_search,
            "require_nonempty_target": self.require_nonempty_target,
        }


@dataclass(frozen=True)
class AlignmentGateResult:
    accepted: bool
    failures: tuple[str, ...]
    thresholds: TransportAlignmentThresholds
    coverage: AlignmentCoverage

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "failures": list(self.failures),
            "thresholds": self.thresholds.to_dict(),
            "coverage": self.coverage.to_dict(),
        }


class AlignmentRejectedError(ValueError):
    def __init__(self, result: AlignmentGateResult) -> None:
        self.result = result
        detail = "; ".join(result.failures) or "alignment rejected"
        super().__init__(detail)


@dataclass(frozen=True)
class StrictTokenRowMapping:
    """Auditable resolution of one physical target vocabulary row."""

    target_row_index: int
    donor_row_index: int
    resolution: str
    special_token_roles: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_row_index": self.target_row_index,
            "donor_row_index": self.donor_row_index,
            "resolution": self.resolution,
            "special_token_roles": list(self.special_token_roles),
        }


@dataclass(frozen=True)
class StrictTokenTransportProfile:
    """Complete one-row-to-one-row vocabulary transport contract."""

    donor_tokenizer_class: str
    target_tokenizer_class: str
    donor_vocab_size: int
    target_vocab_size: int
    donor_row_capacity: int
    target_row_capacity: int
    alignment_fingerprint: str
    row_mappings: tuple[StrictTokenRowMapping, ...]

    def __post_init__(self) -> None:
        target_rows = tuple(item.target_row_index for item in self.row_mappings)
        if target_rows != tuple(range(self.target_row_capacity)):
            raise ValueError(
                "Strict token transport mappings must cover every target row in order."
            )
        if any(
            item.donor_row_index < 0 or item.donor_row_index >= self.donor_row_capacity
            for item in self.row_mappings
        ):
            raise ValueError(
                "Strict token transport mappings must stay within donor capacity."
            )

    @property
    def target_to_donor_row_indices(self) -> tuple[int, ...]:
        return tuple(item.donor_row_index for item in self.row_mappings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "profile": "strict-cross-tokenizer-one-row-v1",
            "direction": "target-row-to-donor-row",
            "donor_tokenizer_class": self.donor_tokenizer_class,
            "target_tokenizer_class": self.target_tokenizer_class,
            "donor_vocab_size": self.donor_vocab_size,
            "target_vocab_size": self.target_vocab_size,
            "donor_row_capacity": self.donor_row_capacity,
            "target_row_capacity": self.target_row_capacity,
            "alignment_fingerprint": self.alignment_fingerprint,
            "target_to_donor_row_indices": list(self.target_to_donor_row_indices),
            "row_mappings": [item.to_dict() for item in self.row_mappings],
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class StrictTokenTransportRejectedError(ValueError):
    """Raised before transport when a strict row mapping cannot be proven."""

    def __init__(self, failures: Sequence[str]) -> None:
        self.failures = tuple(failures)
        detail = "; ".join(self.failures) or "strict token transport rejected"
        super().__init__(detail)


@dataclass
class _ByteTrieNode:
    children: dict[int, _ByteTrieNode] = field(default_factory=dict)
    token_ids: list[int] = field(default_factory=list)


def build_token_alignment(
    donor_tokenizer: HFTokenizerProtocol,
    target_tokenizer: HFTokenizerProtocol,
    *,
    include_special_tokens: bool = False,
    max_span_tokens: int = 8,
    max_span_candidates: int | None = 1024,
) -> TokenizerAlignmentReport:
    """Build a deterministic, byte-exact target-row to donor-row alignment.

    A single donor token with exactly equal decoded UTF-8 bytes is preferred.
    If none exists, candidates are donor token spans whose full decode round
    trips to the target bytes. All candidates in the selected class are kept.
    A bounded search never silently chooses one candidate: it marks the row as
    truncated so the default transport gate rejects it.
    """

    if max_span_tokens < 2:
        raise ValueError("max_span_tokens must be at least 2.")
    if max_span_candidates is not None and max_span_candidates < 1:
        raise ValueError("max_span_candidates must be positive or None.")

    donor_vocab_size = _tokenizer_length(donor_tokenizer, "donor")
    target_vocab_size = _tokenizer_length(target_tokenizer, "target")
    donor_special_ids = _special_token_ids(donor_tokenizer, donor_vocab_size)
    target_special_ids = _special_token_ids(target_tokenizer, target_vocab_size)

    donor_surfaces: dict[bytes, list[int]] = {}
    donor_surface_issues: list[TokenSurfaceIssue] = []
    for token_id in range(donor_vocab_size):
        if token_id in donor_special_ids and not include_special_tokens:
            donor_surface_issues.append(
                TokenSurfaceIssue(token_id, "special-token-excluded")
            )
            continue
        text, decode_reason = _decode_token(donor_tokenizer, token_id)
        if decode_reason is not None:
            donor_surface_issues.append(TokenSurfaceIssue(token_id, decode_reason))
            continue
        surface, utf8_reason = _utf8_bytes(text)
        if utf8_reason is not None:
            donor_surface_issues.append(TokenSurfaceIssue(token_id, utf8_reason))
            continue
        donor_surfaces.setdefault(surface, []).append(token_id)

    for token_ids in donor_surfaces.values():
        token_ids.sort()
    trie = _build_byte_trie(donor_surfaces)
    collision_groups = [
        token_ids for token_ids in donor_surfaces.values() if len(token_ids) > 1
    ]

    alignments = []
    for target_id in range(target_vocab_size):
        alignment = _align_target_token(
            donor_tokenizer,
            target_tokenizer,
            target_id,
            donor_vocab_size,
            donor_special_ids,
            target_special_ids,
            donor_surfaces,
            trie,
            include_special_tokens=include_special_tokens,
            max_span_tokens=max_span_tokens,
            max_span_candidates=max_span_candidates,
        )
        alignments.append(alignment)

    alignment_tuple = tuple(alignments)
    return TokenizerAlignmentReport(
        donor_tokenizer_class=type(donor_tokenizer).__name__,
        target_tokenizer_class=type(target_tokenizer).__name__,
        donor_vocab_size=donor_vocab_size,
        target_vocab_size=target_vocab_size,
        include_special_tokens=bool(include_special_tokens),
        max_span_tokens=max_span_tokens,
        max_span_candidates=max_span_candidates,
        donor_special_token_ids=tuple(sorted(donor_special_ids)),
        target_special_token_ids=tuple(sorted(target_special_ids)),
        donor_surface_issues=tuple(donor_surface_issues),
        donor_collision_surface_count=len(collision_groups),
        donor_collision_token_count=sum(len(group) for group in collision_groups),
        alignments=alignment_tuple,
        coverage=_coverage(alignment_tuple),
    )


def evaluate_alignment_for_transport(
    report: TokenizerAlignmentReport,
    thresholds: TransportAlignmentThresholds | None = None,
) -> AlignmentGateResult:
    """Evaluate transport thresholds without mutating or resolving ambiguity."""

    limits = thresholds or TransportAlignmentThresholds()
    coverage = report.coverage
    failures: list[str] = []
    if limits.require_nonempty_target and coverage.target_token_count == 0:
        failures.append("target vocabulary is empty")
    _append_minimum_failure(
        failures,
        "coverage_fraction",
        coverage.coverage_fraction,
        limits.min_coverage_fraction,
    )
    _append_minimum_failure(
        failures,
        "unique_coverage_fraction",
        coverage.unique_coverage_fraction,
        limits.min_unique_coverage_fraction,
    )
    _append_maximum_failure(
        failures,
        "unalignable_fraction",
        coverage.unalignable_fraction,
        limits.max_unalignable_fraction,
    )
    _append_maximum_failure(
        failures,
        "ambiguity_fraction",
        coverage.ambiguity_fraction,
        limits.max_ambiguity_fraction,
    )
    if coverage.truncated_search_count and not limits.allow_truncated_candidate_search:
        failures.append(
            "candidate search was truncated for "
            f"{coverage.truncated_search_count} target token(s)"
        )
    return AlignmentGateResult(
        accepted=not failures,
        failures=tuple(failures),
        thresholds=limits,
        coverage=coverage,
    )


def require_alignment_for_transport(
    report: TokenizerAlignmentReport,
    thresholds: TransportAlignmentThresholds | None = None,
) -> AlignmentGateResult:
    """Return an accepted gate result or raise with every deterministic failure."""

    result = evaluate_alignment_for_transport(report, thresholds)
    if not result.accepted:
        raise AlignmentRejectedError(result)
    return result


def build_strict_token_transport_profile(
    donor_tokenizer: HFTokenizerProtocol,
    target_tokenizer: HFTokenizerProtocol,
    *,
    donor_row_capacity: int | None = None,
    target_row_capacity: int | None = None,
) -> StrictTokenTransportProfile:
    """Prove a complete target-row to donor-row mapping without mutation.

    Physical target rows must be exactly the contiguous tokenizer vocabulary.
    Donor capacity may contain padding after its contiguous tokenizer rows.
    BOS, EOS, PAD, and UNK rows resolve by matching semantic role; every other
    row must have one and only one exact-token donor candidate.
    """

    failures: list[str] = []
    try:
        donor_vocab_size = _tokenizer_length(donor_tokenizer, "donor")
    except ValueError as exc:
        failures.append(str(exc))
        donor_vocab_size = 0
    try:
        target_vocab_size = _tokenizer_length(target_tokenizer, "target")
    except ValueError as exc:
        failures.append(str(exc))
        target_vocab_size = 0

    donor_capacity = _strict_row_capacity(
        donor_row_capacity,
        donor_vocab_size,
        "donor",
        failures,
    )
    target_capacity = _strict_row_capacity(
        target_row_capacity,
        target_vocab_size,
        "target",
        failures,
    )
    if donor_capacity < donor_vocab_size:
        failures.append(
            f"donor row capacity {donor_capacity} is below tokenizer vocabulary "
            f"size {donor_vocab_size}"
        )
    if target_capacity != target_vocab_size:
        failures.append(
            f"target row capacity {target_capacity} must equal tokenizer vocabulary "
            f"size {target_vocab_size}"
        )
    if target_capacity == 0:
        failures.append("target vocabulary is empty")

    _validate_contiguous_vocab_ids(
        donor_tokenizer,
        donor_vocab_size,
        "donor",
        failures,
    )
    _validate_contiguous_vocab_ids(
        target_tokenizer,
        target_vocab_size,
        "target",
        failures,
    )
    donor_roles = _strict_special_token_roles(
        donor_tokenizer,
        donor_vocab_size,
        "donor",
        failures,
    )
    target_roles = _strict_special_token_roles(
        target_tokenizer,
        target_vocab_size,
        "target",
        failures,
    )
    if failures:
        raise StrictTokenTransportRejectedError(failures)

    report = build_token_alignment(
        donor_tokenizer,
        target_tokenizer,
        include_special_tokens=True,
    )
    semantic_rows, unresolved_semantic_rows = _strict_semantic_row_mappings(
        donor_roles,
        target_roles,
        failures,
    )
    row_mappings: list[StrictTokenRowMapping] = []
    for alignment in report.alignments:
        target_id = alignment.target_token_id
        semantic = semantic_rows.get(target_id)
        if semantic is not None:
            donor_id, roles = semantic
            row_mappings.append(
                StrictTokenRowMapping(
                    target_row_index=target_id,
                    donor_row_index=donor_id,
                    resolution="special-token-role",
                    special_token_roles=roles,
                )
            )
            continue
        if target_id in unresolved_semantic_rows:
            continue
        if alignment.classification is AlignmentClass.UTF8_BYTE_SPAN:
            failures.append(
                f"target row {target_id} resolves only to a donor token span"
            )
            continue
        if alignment.classification is AlignmentClass.UNALIGNABLE:
            failures.append(
                f"target row {target_id} is unalignable: {alignment.reason}"
            )
            continue
        if alignment.is_ambiguous:
            donor_ids = tuple(
                candidate.donor_token_ids[0]
                for candidate in alignment.candidates
                if len(candidate.donor_token_ids) == 1
            )
            failures.append(
                f"target row {target_id} is ambiguous across donor rows {donor_ids}"
            )
            continue
        candidate = alignment.candidates[0]
        if (
            candidate.classification is not AlignmentClass.EXACT_TOKEN
            or len(candidate.donor_token_ids) != 1
        ):
            failures.append(
                f"target row {target_id} does not resolve to exactly one donor row"
            )
            continue
        donor_id = candidate.donor_token_ids[0]
        if donor_id >= donor_capacity:
            failures.append(
                f"target row {target_id} resolves to donor row {donor_id}, outside "
                f"capacity {donor_capacity}"
            )
            continue
        row_mappings.append(
            StrictTokenRowMapping(
                target_row_index=target_id,
                donor_row_index=donor_id,
                resolution="exact-token",
            )
        )

    mapped_target_ids = {item.target_row_index for item in row_mappings}
    missing_rows = sorted(set(range(target_capacity)) - mapped_target_ids)
    if missing_rows and not failures:
        failures.append(f"target rows have no donor mapping: {missing_rows}")
    if failures:
        raise StrictTokenTransportRejectedError(failures)

    return StrictTokenTransportProfile(
        donor_tokenizer_class=type(donor_tokenizer).__name__,
        target_tokenizer_class=type(target_tokenizer).__name__,
        donor_vocab_size=donor_vocab_size,
        target_vocab_size=target_vocab_size,
        donor_row_capacity=donor_capacity,
        target_row_capacity=target_capacity,
        alignment_fingerprint=report.fingerprint(),
        row_mappings=tuple(row_mappings),
    )


def _align_target_token(
    donor_tokenizer: HFTokenizerProtocol,
    target_tokenizer: HFTokenizerProtocol,
    target_id: int,
    donor_vocab_size: int,
    donor_special_ids: set[int],
    target_special_ids: set[int],
    donor_surfaces: Mapping[bytes, list[int]],
    trie: _ByteTrieNode,
    *,
    include_special_tokens: bool,
    max_span_tokens: int,
    max_span_candidates: int | None,
) -> TokenAlignment:
    if target_id in target_special_ids and not include_special_tokens:
        return TokenAlignment(
            target_token_id=target_id,
            target_text=None,
            target_utf8_hex=None,
            classification=AlignmentClass.UNALIGNABLE,
            candidates=(),
            reason="special-token-excluded",
        )

    target_text, decode_reason = _decode_token(target_tokenizer, target_id)
    if decode_reason is not None:
        return TokenAlignment(
            target_token_id=target_id,
            target_text=None,
            target_utf8_hex=None,
            classification=AlignmentClass.UNALIGNABLE,
            candidates=(),
            reason=decode_reason,
        )
    target_bytes, utf8_reason = _utf8_bytes(target_text)
    if utf8_reason is not None:
        return TokenAlignment(
            target_token_id=target_id,
            target_text=target_text if utf8_reason != "not-valid-utf8" else None,
            target_utf8_hex=None,
            classification=AlignmentClass.UNALIGNABLE,
            candidates=(),
            reason=utf8_reason,
        )

    exact_ids = tuple(donor_surfaces.get(target_bytes, ()))
    if exact_ids:
        candidates = tuple(
            AlignmentCandidate(
                classification=AlignmentClass.EXACT_TOKEN,
                donor_token_ids=(donor_id,),
                origins=("single-token-decoded-utf8",),
            )
            for donor_id in exact_ids
        )
        return TokenAlignment(
            target_token_id=target_id,
            target_text=target_text,
            target_utf8_hex=target_bytes.hex(),
            classification=AlignmentClass.EXACT_TOKEN,
            candidates=candidates,
        )

    diagnostics: list[str] = []
    candidate_origins: dict[tuple[int, ...], set[str]] = {}
    encoded_ids, encode_reason = _encode_text(donor_tokenizer, target_text)
    if encode_reason is not None:
        diagnostics.append(encode_reason)
    else:
        valid, validation_reason = _validate_donor_sequence(
            donor_tokenizer,
            encoded_ids,
            target_bytes,
            donor_vocab_size,
            donor_special_ids,
            include_special_tokens=include_special_tokens,
            max_span_tokens=max_span_tokens,
        )
        if valid and len(encoded_ids) >= 2:
            candidate_origins.setdefault(encoded_ids, set()).add(
                "donor-encode-roundtrip"
            )
        elif validation_reason is not None:
            diagnostics.append(validation_reason)

    span_sequences, truncated = _enumerate_surface_spans(
        target_bytes,
        trie,
        max_span_tokens=max_span_tokens,
        max_span_candidates=max_span_candidates,
    )
    for sequence in span_sequences:
        valid, _ = _validate_donor_sequence(
            donor_tokenizer,
            sequence,
            target_bytes,
            donor_vocab_size,
            donor_special_ids,
            include_special_tokens=include_special_tokens,
            max_span_tokens=max_span_tokens,
        )
        if valid:
            candidate_origins.setdefault(sequence, set()).add(
                "decoded-utf8-segmentation"
            )

    ordered_sequences = sorted(candidate_origins, key=lambda item: (len(item), item))
    if max_span_candidates is not None and len(ordered_sequences) > max_span_candidates:
        encoded_candidate = encoded_ids if encoded_ids in candidate_origins else None
        retained = ordered_sequences[:max_span_candidates]
        if encoded_candidate is not None and encoded_candidate not in retained:
            retained[-1] = encoded_candidate
            retained.sort(key=lambda item: (len(item), item))
        ordered_sequences = retained
        truncated = True

    if ordered_sequences:
        candidates = tuple(
            AlignmentCandidate(
                classification=AlignmentClass.UTF8_BYTE_SPAN,
                donor_token_ids=sequence,
                origins=tuple(sorted(candidate_origins[sequence])),
            )
            for sequence in ordered_sequences
        )
        return TokenAlignment(
            target_token_id=target_id,
            target_text=target_text,
            target_utf8_hex=target_bytes.hex(),
            classification=AlignmentClass.UTF8_BYTE_SPAN,
            candidates=candidates,
            diagnostics=tuple(sorted(set(diagnostics))),
            candidate_search_truncated=truncated,
        )

    return TokenAlignment(
        target_token_id=target_id,
        target_text=target_text,
        target_utf8_hex=target_bytes.hex(),
        classification=AlignmentClass.UNALIGNABLE,
        candidates=(),
        reason="no-byte-faithful-donor-representation",
        diagnostics=tuple(sorted(set(diagnostics))),
        candidate_search_truncated=truncated,
    )


def _tokenizer_length(tokenizer: HFTokenizerProtocol, label: str) -> int:
    try:
        size = operator.index(len(tokenizer))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{label} tokenizer must have an integer vocabulary length."
        ) from exc
    if size < 0:
        raise ValueError(f"{label} tokenizer vocabulary length cannot be negative.")
    return size


def _strict_row_capacity(
    value: int | None,
    default: int,
    label: str,
    failures: list[str],
) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        failures.append(f"{label} row capacity must be a non-negative integer")
        return default
    try:
        capacity = operator.index(value)
    except (TypeError, ValueError, OverflowError):
        failures.append(f"{label} row capacity must be a non-negative integer")
        return default
    if capacity < 0:
        failures.append(f"{label} row capacity must be a non-negative integer")
        return default
    return capacity


def _validate_contiguous_vocab_ids(
    tokenizer: HFTokenizerProtocol,
    vocab_size: int,
    label: str,
    failures: list[str],
) -> None:
    getter = getattr(tokenizer, "get_vocab", None)
    if not callable(getter):
        failures.append(
            f"{label} tokenizer must expose get_vocab() for strict ID auditing"
        )
        return
    try:
        vocabulary = getter()
    except Exception as exc:
        failures.append(
            f"{label} tokenizer get_vocab() failed with {type(exc).__name__}"
        )
        return
    if not isinstance(vocabulary, Mapping):
        failures.append(f"{label} tokenizer get_vocab() must return a mapping")
        return

    token_ids: list[int] = []
    invalid_count = 0
    for value in vocabulary.values():
        if isinstance(value, bool):
            invalid_count += 1
            continue
        try:
            token_ids.append(operator.index(value))
        except (TypeError, ValueError, OverflowError):
            invalid_count += 1
    if invalid_count:
        failures.append(
            f"{label} vocabulary contains {invalid_count} non-integer token ID(s)"
        )

    actual_ids = set(token_ids)
    expected_ids = set(range(vocab_size))
    missing = sorted(expected_ids - actual_ids)
    outside = sorted(actual_ids - expected_ids)
    if missing:
        failures.append(f"{label} vocabulary has token ID holes: {missing}")
    if outside:
        failures.append(
            f"{label} vocabulary has IDs outside [0, {vocab_size}): {outside}"
        )


_STRICT_SPECIAL_TOKEN_ROLES = (
    ("BOS", "bos_token_id"),
    ("EOS", "eos_token_id"),
    ("PAD", "pad_token_id"),
    ("UNK", "unk_token_id"),
)


def _strict_special_token_roles(
    tokenizer: HFTokenizerProtocol,
    vocab_size: int,
    label: str,
    failures: list[str],
) -> dict[str, tuple[int, ...]]:
    roles: dict[str, tuple[int, ...]] = {}
    for role, attribute in _STRICT_SPECIAL_TOKEN_ROLES:
        value = getattr(tokenizer, attribute, None)
        if value is None:
            roles[role] = ()
            continue
        values = value if isinstance(value, (list, tuple, set)) else (value,)
        token_ids: list[int] = []
        invalid = False
        for item in values:
            if isinstance(item, bool):
                invalid = True
                continue
            try:
                token_id = operator.index(item)
            except (TypeError, ValueError, OverflowError):
                invalid = True
                continue
            if token_id < 0 or token_id >= vocab_size:
                invalid = True
                continue
            token_ids.append(token_id)
        if invalid:
            failures.append(
                f"{label} {role} token ID must identify a tokenizer vocabulary row"
            )
        roles[role] = tuple(sorted(set(token_ids)))
    return roles


def _strict_semantic_row_mappings(
    donor_roles: Mapping[str, tuple[int, ...]],
    target_roles: Mapping[str, tuple[int, ...]],
    failures: list[str],
) -> tuple[dict[int, tuple[int, tuple[str, ...]]], set[int]]:
    assignments: dict[int, list[tuple[str, int]]] = {}
    unresolved_rows: set[int] = set()
    for role, _ in _STRICT_SPECIAL_TOKEN_ROLES:
        target_ids = target_roles.get(role, ())
        if not target_ids:
            continue
        donor_ids = donor_roles.get(role, ())
        if not donor_ids:
            failures.append(f"target {role} role has no donor {role} row")
            unresolved_rows.update(target_ids)
            continue
        if len(donor_ids) != 1:
            failures.append(f"donor {role} role is ambiguous across rows {donor_ids}")
            unresolved_rows.update(target_ids)
            continue
        for target_id in target_ids:
            assignments.setdefault(target_id, []).append((role, donor_ids[0]))

    resolved: dict[int, tuple[int, tuple[str, ...]]] = {}
    for target_id in sorted(assignments):
        items = assignments[target_id]
        donor_ids = tuple(sorted({donor_id for _, donor_id in items}))
        roles = tuple(role for role, _ in items)
        if len(donor_ids) != 1:
            failures.append(
                f"target row {target_id} has conflicting semantic roles {roles} "
                f"mapping to donor rows {donor_ids}"
            )
            unresolved_rows.add(target_id)
            continue
        resolved[target_id] = (donor_ids[0], roles)
    return resolved, unresolved_rows


def _special_token_ids(tokenizer: HFTokenizerProtocol, vocab_size: int) -> set[int]:
    result = set()
    for value in getattr(tokenizer, "all_special_ids", ()) or ():
        try:
            token_id = operator.index(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if 0 <= token_id < vocab_size:
            result.add(token_id)
    return result


def _decode_token(
    tokenizer: HFTokenizerProtocol,
    token_id: int,
) -> tuple[str | None, str | None]:
    return _decode_ids(tokenizer, (token_id,))


def _decode_ids(
    tokenizer: HFTokenizerProtocol,
    token_ids: tuple[int, ...],
) -> tuple[str | None, str | None]:
    attempts = (
        {"skip_special_tokens": False, "clean_up_tokenization_spaces": False},
        {"clean_up_tokenization_spaces": False},
        {},
    )
    last_type_error: TypeError | None = None
    for kwargs in attempts:
        try:
            decoded = tokenizer.decode(list(token_ids), **kwargs)
        except TypeError as exc:
            last_type_error = exc
            continue
        except (
            Exception
        ) as exc:  # Tokenizer implementations expose heterogeneous failures.
            return None, f"decode-error:{type(exc).__name__}"
        if not isinstance(decoded, str):
            return None, "decoded-value-is-not-text"
        return decoded, None
    error_name = (
        type(last_type_error).__name__ if last_type_error is not None else "TypeError"
    )
    return None, f"decode-error:{error_name}"


def _utf8_bytes(text: str | None) -> tuple[bytes, str | None]:
    if text is None:
        return b"", "decoded-value-is-not-text"
    if not text:
        return b"", "empty-decoded-text"
    if "\ufffd" in text:
        return b"", "replacement-character-is-not-byte-faithful"
    try:
        return text.encode("utf-8", errors="strict"), None
    except UnicodeEncodeError:
        return b"", "not-valid-utf8"


def _encode_text(
    tokenizer: HFTokenizerProtocol,
    text: str,
) -> tuple[tuple[int, ...], str | None]:
    encoder = getattr(tokenizer, "encode", None)
    if callable(encoder):
        try:
            encoded = encoder(text, add_special_tokens=False)
        except TypeError:
            return (), "donor-encode-does-not-support-add-special-tokens-false"
        except Exception as exc:
            return (), f"donor-encode-error:{type(exc).__name__}"
    elif callable(tokenizer):
        try:
            encoded = tokenizer(text, add_special_tokens=False)
        except Exception as exc:
            return (), f"donor-call-error:{type(exc).__name__}"
    else:
        return (), "donor-tokenizer-has-no-encoder"

    try:
        token_ids = _coerce_input_ids(encoded)
    except (TypeError, ValueError, OverflowError) as exc:
        return (), f"invalid-donor-encoding:{type(exc).__name__}"
    if not token_ids:
        return (), "donor-encoding-is-empty"
    return token_ids, None


def _coerce_input_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError("Tokenizer mapping has no input_ids.")
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise TypeError("Tokenizer IDs must be a sequence.")
    result = []
    for item in value:
        if isinstance(item, bool):
            raise TypeError("Boolean token IDs are invalid.")
        result.append(operator.index(item))
    return tuple(result)


def _validate_donor_sequence(
    donor_tokenizer: HFTokenizerProtocol,
    sequence: tuple[int, ...],
    target_bytes: bytes,
    donor_vocab_size: int,
    donor_special_ids: set[int],
    *,
    include_special_tokens: bool,
    max_span_tokens: int,
) -> tuple[bool, str | None]:
    if not sequence:
        return False, "donor-candidate-is-empty"
    if len(sequence) > max_span_tokens:
        return False, "donor-candidate-exceeds-max-span-tokens"
    if any(token_id < 0 or token_id >= donor_vocab_size for token_id in sequence):
        return False, "donor-candidate-token-id-out-of-range"
    if not include_special_tokens and any(
        token_id in donor_special_ids for token_id in sequence
    ):
        return False, "donor-candidate-contains-excluded-special-token"
    decoded, decode_reason = _decode_ids(donor_tokenizer, sequence)
    if decode_reason is not None:
        return False, f"donor-candidate-{decode_reason}"
    decoded_bytes, utf8_reason = _utf8_bytes(decoded)
    if utf8_reason is not None:
        return False, f"donor-candidate-{utf8_reason}"
    if decoded_bytes != target_bytes:
        return False, "donor-candidate-roundtrip-byte-mismatch"
    return True, None


def _build_byte_trie(surfaces: Mapping[bytes, list[int]]) -> _ByteTrieNode:
    root = _ByteTrieNode()
    for surface, token_ids in sorted(surfaces.items(), key=lambda item: item[0]):
        node = root
        for byte in surface:
            node = node.children.setdefault(byte, _ByteTrieNode())
        node.token_ids.extend(sorted(token_ids))
    return root


def _enumerate_surface_spans(
    target_bytes: bytes,
    trie: _ByteTrieNode,
    *,
    max_span_tokens: int,
    max_span_candidates: int | None,
) -> tuple[tuple[tuple[int, ...], ...], bool]:
    results: set[tuple[int, ...]] = set()
    truncated = False
    stop_after = None if max_span_candidates is None else max_span_candidates + 1

    def visit(offset: int, prefix: tuple[int, ...]) -> None:
        nonlocal truncated
        if stop_after is not None and len(results) >= stop_after:
            truncated = True
            return
        if offset == len(target_bytes):
            if len(prefix) >= 2:
                results.add(prefix)
            return
        if len(prefix) >= max_span_tokens:
            return
        node = trie
        for cursor in range(offset, len(target_bytes)):
            node = node.children.get(target_bytes[cursor])
            if node is None:
                break
            for token_id in node.token_ids:
                visit(cursor + 1, prefix + (token_id,))
                if stop_after is not None and len(results) >= stop_after:
                    truncated = True
                    return

    visit(0, ())
    ordered = sorted(results, key=lambda item: (len(item), item))
    if max_span_candidates is not None and len(ordered) > max_span_candidates:
        ordered = ordered[:max_span_candidates]
        truncated = True
    return tuple(ordered), truncated


def _coverage(alignments: tuple[TokenAlignment, ...]) -> AlignmentCoverage:
    total = len(alignments)
    exact = sum(
        item.classification is AlignmentClass.EXACT_TOKEN for item in alignments
    )
    spans = sum(
        item.classification is AlignmentClass.UTF8_BYTE_SPAN for item in alignments
    )
    unalignable = sum(
        item.classification is AlignmentClass.UNALIGNABLE for item in alignments
    )
    aligned = exact + spans
    ambiguous = sum(item.is_ambiguous for item in alignments)
    unique_aligned = sum(
        item.classification is not AlignmentClass.UNALIGNABLE and not item.is_ambiguous
        for item in alignments
    )
    truncated = sum(item.candidate_search_truncated for item in alignments)

    def fraction(count: int) -> float:
        return float(count / total) if total else 0.0

    return AlignmentCoverage(
        target_token_count=total,
        exact_token_count=exact,
        utf8_byte_span_count=spans,
        unalignable_count=unalignable,
        aligned_count=aligned,
        unique_aligned_count=unique_aligned,
        ambiguous_count=ambiguous,
        truncated_search_count=truncated,
        coverage_fraction=fraction(aligned),
        unique_coverage_fraction=fraction(unique_aligned),
        exact_fraction=fraction(exact),
        utf8_byte_span_fraction=fraction(spans),
        unalignable_fraction=fraction(unalignable),
        ambiguity_fraction=fraction(ambiguous),
    )


def _append_minimum_failure(
    failures: list[str],
    name: str,
    actual: float,
    required: float,
) -> None:
    if actual + 1e-12 < required:
        failures.append(f"{name}={actual:.6f} is below required {required:.6f}")


def _append_maximum_failure(
    failures: list[str],
    name: str,
    actual: float,
    allowed: float,
) -> None:
    if actual > allowed + 1e-12:
        failures.append(f"{name}={actual:.6f} exceeds allowed {allowed:.6f}")
