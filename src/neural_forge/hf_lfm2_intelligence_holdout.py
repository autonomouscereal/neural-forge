"""Sealed parent/candidate evidence for the independent LFM2 holdout.

This module does not score intelligence itself.  It binds the deterministic
holdout manifest and generation metadata into the existing
``hf_lfm2_intelligence_gate`` types so promotion continues to use one scoring
implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from . import hf_lfm2_intelligence_gate as _gate
from .dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from .hf_lfm2_intelligence_gate import (
    CachedParentBaseline,
    CandidateOutput,
    GenerationEvidence,
    IntelligenceHoldoutManifest,
    Lfm2IntelligenceGateError,
    load_cached_parent_rows,
    load_lfm2_intelligence_holdout_manifest,
    seal_cached_parent_rows,
)


PARENT_ENVELOPE_SCHEMA = (
    "neural-forge-lfm2-intelligence-holdout-parent-envelope-v1"
)
CANDIDATE_OUTPUT_SCHEMA = (
    "neural-forge-lfm2-intelligence-holdout-candidate-output-v1"
)


class Lfm2IntelligenceHoldoutError(Lfm2IntelligenceGateError):
    """Holdout envelope or candidate evidence is invalid."""


def _repetition_metadata(token_ids: Sequence[int]) -> dict[str, Any]:
    tokens = tuple(int(value) for value in token_ids)
    bigrams = tuple(zip(tokens, tokens[1:]))
    counts: dict[tuple[int, int], int] = {}
    for value in bigrams:
        counts[value] = counts.get(value, 0) + 1
    longest = 0
    current = 0
    previous: int | None = None
    for value in tokens:
        if value == previous:
            current += 1
        else:
            current = 1
            previous = value
        longest = max(longest, current)
    return {
        "token_count": len(tokens),
        "unique_token_ratio": (
            1.0 if not tokens else len(set(tokens)) / len(tokens)
        ),
        "maximum_repeated_bigram_share": (
            0.0 if not bigrams else max(counts.values()) / len(bigrams)
        ),
        "longest_identical_token_run": longest,
    }


def _generation(
    token_ids: Any,
    *,
    generation_limit: int,
    eos_token_ids: frozenset[int],
    fragment_like: bool,
    name: str,
) -> GenerationEvidence:
    if (
        not isinstance(token_ids, (list, tuple))
        or len(token_ids) > generation_limit
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in token_ids
        )
    ):
        raise Lfm2IntelligenceHoldoutError(
            f"{name} token IDs are malformed or exceed the generation limit."
        )
    tokens = tuple(int(value) for value in token_ids)
    eos_indices = [
        index for index, token_id in enumerate(tokens) if token_id in eos_token_ids
    ]
    if eos_indices and eos_indices != [len(tokens) - 1]:
        raise Lfm2IntelligenceHoldoutError(
            f"{name} contains EOS before the final generated token."
        )
    return GenerationEvidence(
        generated_token_count=len(tokens),
        generation_limit=generation_limit,
        eos_observed=bool(eos_indices),
        fragment_like=fragment_like,
        token_ids=tokens,
    )


def _generation_protocol(
    *,
    eos_token_ids: Sequence[int],
) -> dict[str, Any]:
    eos = tuple(sorted(set(eos_token_ids)))
    if (
        not eos
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in eos
        )
    ):
        raise Lfm2IntelligenceHoldoutError(
            "eos_token_ids must contain non-negative integer IDs."
        )
    return {
        "do_sample": False,
        "use_cache": True,
        "add_special_tokens": True,
        "per_case_generation_limits": True,
        "eos_token_ids": list(eos),
    }


def _normalize_generated_rows(
    manifest: IntelligenceHoldoutManifest,
    generated_rows: Sequence[Mapping[str, Any]],
    *,
    eos_token_ids: Sequence[int],
) -> tuple[dict[str, Any], ...]:
    if (
        isinstance(generated_rows, (str, bytes))
        or not isinstance(generated_rows, Sequence)
    ):
        raise Lfm2IntelligenceHoldoutError(
            "generated_rows must be a sequence."
        )
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    for raw in generated_rows:
        if not isinstance(raw, Mapping) or set(raw) != {
            "case_id",
            "input_token_count",
            "output",
            "output_token_ids",
        }:
            raise Lfm2IntelligenceHoldoutError(
                "Generated parent row fields differ from the holdout schema."
            )
        case_id = _gate._text(raw["case_id"], "generated case_id")
        if case_id in rows_by_id:
            raise Lfm2IntelligenceHoldoutError(
                "Generated parent case IDs are duplicated."
            )
        rows_by_id[case_id] = raw
    if set(rows_by_id) != set(manifest.case_map):
        raise Lfm2IntelligenceHoldoutError(
            "Generated parent rows do not cover the exact holdout manifest."
        )
    eos = frozenset(_generation_protocol(eos_token_ids=eos_token_ids)["eos_token_ids"])
    normalized = []
    for spec in manifest.cases:
        raw = rows_by_id[spec.case_id]
        input_count = _gate._positive_int(
            raw["input_token_count"],
            f"{spec.case_id} input_token_count",
        )
        output = _gate._text(
            raw["output"],
            f"{spec.case_id} parent output",
            allow_empty=True,
        )
        fragment_like = (
            bool(output.strip()) and len(output.strip()) < 12
        )
        generation = _generation(
            raw["output_token_ids"],
            generation_limit=spec.generation_limit,
            eos_token_ids=eos,
            fragment_like=fragment_like,
            name=f"{spec.case_id} parent",
        )
        declared_capability = (
            None if spec.capability is None else spec.capability.as_dict()
        )
        capability_demonstrated = bool(
            spec.capability is not None and spec.capability.passes(output)
        )
        effective_capability = (
            declared_capability if capability_demonstrated else None
        )
        normalized.append(
            {
                "case_id": spec.case_id,
                "prompt": spec.prompt,
                "prompt_sha256": _gate._prompt_sha256(spec.prompt),
                "category": spec.category,
                "split": spec.split,
                "longer_prefix": spec.longer_prefix,
                "input_token_count": input_count,
                "parent_output": output,
                "parent_output_token_ids": list(generation.token_ids),
                "generation": generation.as_dict(),
                "repetition": _repetition_metadata(generation.token_ids),
                "validator": (
                    None
                    if spec.validator is None
                    else spec.validator.as_dict()
                ),
                "declared_capability": declared_capability,
                "effective_capability": effective_capability,
                "capability_demonstrated_by_parent": capability_demonstrated,
                "manual_review_required": spec.manual_review_required,
            }
        )
    return tuple(normalized)


def seal_lfm2_intelligence_holdout_parent_envelope(
    manifest: IntelligenceHoldoutManifest,
    generated_rows: Sequence[Mapping[str, Any]],
    *,
    parent_state_fingerprint: str,
    parent_model_fingerprint: str,
    tokenizer_fingerprint: str,
    eos_token_ids: Sequence[int],
) -> dict[str, Any]:
    """Seal exactly one generated response for every independent holdout case."""

    if not isinstance(manifest, IntelligenceHoldoutManifest):
        raise TypeError("manifest must be IntelligenceHoldoutManifest.")
    state = _gate._digest(
        parent_state_fingerprint,
        "parent_state_fingerprint",
    )
    model = _gate._digest(
        parent_model_fingerprint,
        "parent_model_fingerprint",
    )
    tokenizer = _gate._digest(
        tokenizer_fingerprint,
        "tokenizer_fingerprint",
    )
    protocol = _generation_protocol(eos_token_ids=eos_token_ids)
    rows = _normalize_generated_rows(
        manifest,
        generated_rows,
        eos_token_ids=protocol["eos_token_ids"],
    )
    cache_key = canonical_sha256(
        {
            "schema": "neural-forge-lfm2-intelligence-holdout-cache-key-v1",
            "manifest_sha256": manifest.manifest_sha256,
            "parent_state_fingerprint": state,
            "parent_model_fingerprint": model,
            "tokenizer_fingerprint": tokenizer,
            "generation": protocol,
        }
    )
    core = {
        "schema": PARENT_ENVELOPE_SCHEMA,
        "manifest_sha256": manifest.manifest_sha256,
        "manifest_document_sha256": manifest.document_sha256,
        "parent_cache_key": cache_key,
        "parent_state_fingerprint": state,
        "parent_model_fingerprint": model,
        "tokenizer_fingerprint": tokenizer,
        "generation": protocol,
        "case_count": len(rows),
        "rows": list(rows),
        "parent_outputs_generated_once": True,
        "parent_generate_invocations": len(rows),
        "parent_forward_passes_after_seal": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "network_access": False,
        "hub_loader_used": False,
        "exact_parent_wording_promotion_criterion": False,
        "exact_parent_token_promotion_criterion": False,
    }
    return _gate._canonical_copy(
        {**core, "envelope_sha256": canonical_sha256(core)},
        "sealed LFM2 intelligence holdout parent envelope",
    )


@dataclass(frozen=True, slots=True)
class Lfm2IntelligenceHoldoutParentEnvelope:
    manifest: IntelligenceHoldoutManifest
    parent_cache_key: str
    parent_state_fingerprint: str
    parent_model_fingerprint: str
    tokenizer_fingerprint: str
    eos_token_ids: tuple[int, ...]
    rows: tuple[Mapping[str, Any], ...]
    envelope_sha256: str
    document_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, IntelligenceHoldoutManifest):
            raise Lfm2IntelligenceHoldoutError(
                "Envelope manifest is not typed."
            )
        for name in (
            "parent_cache_key",
            "parent_state_fingerprint",
            "parent_model_fingerprint",
            "tokenizer_fingerprint",
            "envelope_sha256",
            "document_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _gate._digest(getattr(self, name), name),
            )
        eos = tuple(self.eos_token_ids)
        if not eos or len(set(eos)) != len(eos):
            raise Lfm2IntelligenceHoldoutError(
                "Envelope EOS IDs are invalid."
            )
        object.__setattr__(self, "eos_token_ids", eos)
        rows = tuple(MappingProxyType(dict(value)) for value in self.rows)
        if len(rows) != len(self.manifest.cases):
            raise Lfm2IntelligenceHoldoutError(
                "Envelope row count differs from the manifest."
            )
        object.__setattr__(self, "rows", rows)


def load_lfm2_intelligence_holdout_parent_envelope(
    source: Mapping[str, Any] | str | Path,
    *,
    expected_envelope_sha256: str,
    expected_manifest_sha256: str,
    expected_parent_state_fingerprint: str,
    expected_parent_model_fingerprint: str,
    expected_tokenizer_fingerprint: str,
) -> Lfm2IntelligenceHoldoutParentEnvelope:
    """Authenticate a canonical one-time parent envelope and all row metadata."""

    document = _gate._document_from(source, "holdout parent envelope")
    core = dict(document)
    seal = core.pop("envelope_sha256", None)
    manifest = load_lfm2_intelligence_holdout_manifest(
        expected_manifest_sha256=expected_manifest_sha256
    )
    expected_identities = {
        "parent_state_fingerprint": _gate._digest(
            expected_parent_state_fingerprint,
            "expected_parent_state_fingerprint",
        ),
        "parent_model_fingerprint": _gate._digest(
            expected_parent_model_fingerprint,
            "expected_parent_model_fingerprint",
        ),
        "tokenizer_fingerprint": _gate._digest(
            expected_tokenizer_fingerprint,
            "expected_tokenizer_fingerprint",
        ),
    }
    if (
        core.get("schema") != PARENT_ENVELOPE_SCHEMA
        or seal
        != _gate._digest(
            expected_envelope_sha256,
            "expected_envelope_sha256",
        )
        or seal != canonical_sha256(core)
        or core.get("manifest_sha256") != manifest.manifest_sha256
        or core.get("manifest_document_sha256") != manifest.document_sha256
        or any(core.get(name) != value for name, value in expected_identities.items())
        or core.get("case_count") != len(manifest.cases)
        or core.get("parent_outputs_generated_once") is not True
        or core.get("parent_generate_invocations") != len(manifest.cases)
        or core.get("parent_forward_passes_after_seal") != 0
        or core.get("optimizer_steps") != 0
        or core.get("backward_calls") != 0
        or core.get("network_access") is not False
        or core.get("hub_loader_used") is not False
        or core.get("exact_parent_wording_promotion_criterion") is not False
        or core.get("exact_parent_token_promotion_criterion") is not False
    ):
        raise Lfm2IntelligenceHoldoutError(
            "Holdout parent envelope seal, identity, count, or zero-reuse "
            "declaration is invalid."
        )
    protocol = core.get("generation")
    if not isinstance(protocol, Mapping):
        raise Lfm2IntelligenceHoldoutError(
            "Holdout generation protocol is missing."
        )
    expected_protocol = _generation_protocol(
        eos_token_ids=protocol.get("eos_token_ids", ())
    )
    if dict(protocol) != expected_protocol:
        raise Lfm2IntelligenceHoldoutError(
            "Holdout parent generation protocol is not deterministic."
        )
    rows = core.get("rows")
    if not isinstance(rows, list):
        raise Lfm2IntelligenceHoldoutError(
            "Holdout parent envelope rows are invalid."
        )
    generated_rows = [
        {
            "case_id": row.get("case_id") if isinstance(row, Mapping) else None,
            "input_token_count": (
                row.get("input_token_count")
                if isinstance(row, Mapping)
                else None
            ),
            "output": (
                row.get("parent_output")
                if isinstance(row, Mapping)
                else None
            ),
            "output_token_ids": (
                row.get("parent_output_token_ids")
                if isinstance(row, Mapping)
                else None
            ),
        }
        for row in rows
    ]
    reconstructed = seal_lfm2_intelligence_holdout_parent_envelope(
        manifest,
        generated_rows,
        parent_state_fingerprint=expected_identities[
            "parent_state_fingerprint"
        ],
        parent_model_fingerprint=expected_identities[
            "parent_model_fingerprint"
        ],
        tokenizer_fingerprint=expected_identities["tokenizer_fingerprint"],
        eos_token_ids=expected_protocol["eos_token_ids"],
    )
    if canonical_json_bytes(reconstructed) != canonical_json_bytes(document):
        raise Lfm2IntelligenceHoldoutError(
            "Holdout parent envelope is not in canonical typed form."
        )
    return Lfm2IntelligenceHoldoutParentEnvelope(
        manifest=manifest,
        parent_cache_key=str(core["parent_cache_key"]),
        parent_state_fingerprint=expected_identities[
            "parent_state_fingerprint"
        ],
        parent_model_fingerprint=expected_identities[
            "parent_model_fingerprint"
        ],
        tokenizer_fingerprint=expected_identities["tokenizer_fingerprint"],
        eos_token_ids=tuple(expected_protocol["eos_token_ids"]),
        rows=tuple(rows),
        envelope_sha256=str(seal),
        document_sha256=canonical_sha256(document),
    )


def baseline_from_lfm2_intelligence_holdout_envelope(
    envelope: Lfm2IntelligenceHoldoutParentEnvelope,
) -> CachedParentBaseline:
    """Adapt the holdout envelope to the existing cached intelligence gate."""

    if not isinstance(envelope, Lfm2IntelligenceHoldoutParentEnvelope):
        raise TypeError(
            "envelope must be Lfm2IntelligenceHoldoutParentEnvelope."
        )
    rows = [
        {
            "case_id": str(row["case_id"]),
            "prompt": str(row["prompt"]),
            "category": str(row["category"]),
            "split": str(row["split"]),
            "parent_output": str(row["parent_output"]),
            "generation": dict(row["generation"]),
            "validator": row["validator"],
            "capability": row["effective_capability"],
            "manual_review_required": bool(row["manual_review_required"]),
        }
        for row in envelope.rows
    ]
    baseline_document = seal_cached_parent_rows(
        rows,
        parent_cache_key=envelope.parent_cache_key,
        parent_state_fingerprint=envelope.parent_state_fingerprint,
        tokenizer_fingerprint=envelope.tokenizer_fingerprint,
    )
    return load_cached_parent_rows(
        baseline_document,
        expected_document_sha256=canonical_sha256(baseline_document),
        expected_parent_cache_key=envelope.parent_cache_key,
        expected_parent_state_fingerprint=envelope.parent_state_fingerprint,
        expected_tokenizer_fingerprint=envelope.tokenizer_fingerprint,
    )


def candidate_outputs_from_lfm2_holdout_rows(
    manifest: IntelligenceHoldoutManifest,
    rows: Sequence[Mapping[str, Any]],
    *,
    eos_token_ids: Sequence[int],
) -> tuple[CandidateOutput, ...]:
    """Type candidate rows with manifest limits; this function has no parent."""

    if not isinstance(manifest, IntelligenceHoldoutManifest):
        raise TypeError("manifest must be IntelligenceHoldoutManifest.")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise Lfm2IntelligenceHoldoutError(
            "Candidate rows must be a sequence."
        )
    row_map: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise Lfm2IntelligenceHoldoutError(
                "Candidate row must be an object."
            )
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in row_map:
            raise Lfm2IntelligenceHoldoutError(
                "Candidate case IDs are missing or duplicated."
            )
        row_map[case_id] = row
    if set(row_map) != set(manifest.case_map):
        raise Lfm2IntelligenceHoldoutError(
            "Candidate rows do not cover the exact holdout manifest."
        )
    eos = frozenset(_generation_protocol(eos_token_ids=eos_token_ids)["eos_token_ids"])
    candidates = []
    for spec in manifest.cases:
        row = row_map[spec.case_id]
        output = row.get("output")
        tokens = row.get("output_token_ids")
        limit = row.get("generation_limit")
        if not isinstance(output, str) or limit != spec.generation_limit:
            raise Lfm2IntelligenceHoldoutError(
                f"Candidate {spec.case_id} output or generation limit is invalid."
            )
        generation = _generation(
            tokens,
            generation_limit=spec.generation_limit,
            eos_token_ids=eos,
            fragment_like=(
                bool(output.strip()) and len(output.strip()) < 12
            ),
            name=f"{spec.case_id} candidate",
        )
        topical = row.get("topical_relevance")
        if topical is not None and not isinstance(topical, bool):
            raise Lfm2IntelligenceHoldoutError(
                f"Candidate {spec.case_id} topical diagnostic is invalid."
            )
        candidates.append(
            CandidateOutput(
                case_id=spec.case_id,
                output=output,
                generation=generation,
                topical_relevance=topical,
            )
        )
    return tuple(candidates)


def seal_lfm2_intelligence_holdout_candidate_outputs(
    manifest: IntelligenceHoldoutManifest,
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_state_fingerprint: str,
    tokenizer_fingerprint: str,
    eos_token_ids: Sequence[int],
) -> dict[str, Any]:
    """Optionally seal reusable candidate rows without any parent identity."""

    candidates = candidate_outputs_from_lfm2_holdout_rows(
        manifest,
        rows,
        eos_token_ids=eos_token_ids,
    )
    state = _gate._digest(
        candidate_state_fingerprint,
        "candidate_state_fingerprint",
    )
    tokenizer = _gate._digest(
        tokenizer_fingerprint,
        "tokenizer_fingerprint",
    )
    normalized_rows = [
        {
            "case_id": item.case_id,
            "output": item.output,
            "output_token_ids": list(item.generation.token_ids),
            "generation_limit": item.generation.generation_limit,
            "topical_relevance": item.topical_relevance,
            "repetition": _repetition_metadata(item.generation.token_ids),
        }
        for item in candidates
    ]
    core = {
        "schema": CANDIDATE_OUTPUT_SCHEMA,
        "manifest_sha256": manifest.manifest_sha256,
        "candidate_state_fingerprint": state,
        "tokenizer_fingerprint": tokenizer,
        "rows": normalized_rows,
        "candidate_generate_invocations": len(normalized_rows),
        "parent_model_loaded": False,
        "parent_forward_passes": 0,
        "parent_generate_invocations": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "network_access": False,
    }
    return _gate._canonical_copy(
        {**core, "candidate_outputs_sha256": canonical_sha256(core)},
        "sealed holdout candidate outputs",
    )


@dataclass(frozen=True, slots=True)
class Lfm2IntelligenceHoldoutCandidateEvidence:
    manifest: IntelligenceHoldoutManifest
    candidate_state_fingerprint: str
    tokenizer_fingerprint: str
    rows: tuple[Mapping[str, Any], ...]
    candidates: tuple[CandidateOutput, ...]
    candidate_outputs_sha256: str
    document_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, IntelligenceHoldoutManifest):
            raise Lfm2IntelligenceHoldoutError(
                "Candidate evidence manifest is not typed."
            )
        for name in (
            "candidate_state_fingerprint",
            "tokenizer_fingerprint",
            "candidate_outputs_sha256",
            "document_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _gate._digest(getattr(self, name), name),
            )
        rows = tuple(MappingProxyType(dict(value)) for value in self.rows)
        candidates = tuple(self.candidates)
        if (
            len(rows) != len(self.manifest.cases)
            or len(candidates) != len(self.manifest.cases)
            or any(not isinstance(item, CandidateOutput) for item in candidates)
        ):
            raise Lfm2IntelligenceHoldoutError(
                "Candidate evidence count or typed candidates are invalid."
            )
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "candidates", candidates)


def load_lfm2_intelligence_holdout_candidate_outputs(
    source: Mapping[str, Any] | str | Path,
    *,
    expected_candidate_outputs_sha256: str,
    expected_manifest_sha256: str,
    expected_candidate_state_fingerprint: str,
    expected_tokenizer_fingerprint: str,
    eos_token_ids: Sequence[int],
) -> Lfm2IntelligenceHoldoutCandidateEvidence:
    """Authenticate a reusable candidate-output artifact without a parent."""

    document = _gate._document_from(source, "holdout candidate outputs")
    core = dict(document)
    seal = core.pop("candidate_outputs_sha256", None)
    manifest = load_lfm2_intelligence_holdout_manifest(
        expected_manifest_sha256=expected_manifest_sha256
    )
    state = _gate._digest(
        expected_candidate_state_fingerprint,
        "expected_candidate_state_fingerprint",
    )
    tokenizer = _gate._digest(
        expected_tokenizer_fingerprint,
        "expected_tokenizer_fingerprint",
    )
    expected_seal = _gate._digest(
        expected_candidate_outputs_sha256,
        "expected_candidate_outputs_sha256",
    )
    rows = core.get("rows")
    if (
        core.get("schema") != CANDIDATE_OUTPUT_SCHEMA
        or seal != expected_seal
        or seal != canonical_sha256(core)
        or core.get("manifest_sha256") != manifest.manifest_sha256
        or core.get("candidate_state_fingerprint") != state
        or core.get("tokenizer_fingerprint") != tokenizer
        or not isinstance(rows, list)
        or core.get("candidate_generate_invocations") != len(manifest.cases)
        or core.get("parent_model_loaded") is not False
        or core.get("parent_forward_passes") != 0
        or core.get("parent_generate_invocations") != 0
        or core.get("optimizer_steps") != 0
        or core.get("backward_calls") != 0
        or core.get("network_access") is not False
    ):
        raise Lfm2IntelligenceHoldoutError(
            "Candidate output seal, identity, count, or zero-parent "
            "declaration is invalid."
        )
    candidates = candidate_outputs_from_lfm2_holdout_rows(
        manifest,
        rows,
        eos_token_ids=eos_token_ids,
    )
    reconstructed = seal_lfm2_intelligence_holdout_candidate_outputs(
        manifest,
        rows,
        candidate_state_fingerprint=state,
        tokenizer_fingerprint=tokenizer,
        eos_token_ids=eos_token_ids,
    )
    if canonical_json_bytes(reconstructed) != canonical_json_bytes(document):
        raise Lfm2IntelligenceHoldoutError(
            "Candidate outputs are not in canonical typed form."
        )
    return Lfm2IntelligenceHoldoutCandidateEvidence(
        manifest=manifest,
        candidate_state_fingerprint=state,
        tokenizer_fingerprint=tokenizer,
        rows=tuple(rows),
        candidates=candidates,
        candidate_outputs_sha256=str(seal),
        document_sha256=canonical_sha256(document),
    )


__all__ = [
    "CANDIDATE_OUTPUT_SCHEMA",
    "PARENT_ENVELOPE_SCHEMA",
    "Lfm2IntelligenceHoldoutError",
    "Lfm2IntelligenceHoldoutCandidateEvidence",
    "Lfm2IntelligenceHoldoutParentEnvelope",
    "baseline_from_lfm2_intelligence_holdout_envelope",
    "candidate_outputs_from_lfm2_holdout_rows",
    "load_lfm2_intelligence_holdout_candidate_outputs",
    "load_lfm2_intelligence_holdout_parent_envelope",
    "seal_lfm2_intelligence_holdout_candidate_outputs",
    "seal_lfm2_intelligence_holdout_parent_envelope",
]
