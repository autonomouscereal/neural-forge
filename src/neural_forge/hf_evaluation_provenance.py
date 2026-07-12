"""Exact token-path provenance for :func:`evaluate_loaded_text_model`.

The evaluator in :mod:`neural_forge.hf_tools` has four distinct scoring paths:

* perplexity passes ``input_ids`` as both inputs and labels;
* each multiple-choice answer tokenizes ``prompt + choice`` and masks labels at
  the separately-tokenized prompt length;
* generation and behavior prompts share deterministic, left-padded generation
  batches; and
* suite limits are applied after the normal evaluator suite normalization.

This module deliberately reproduces those calls instead of introducing a
second, superficially similar tokenizer path.  All public records are frozen
and contain hashes only; no mutable token tensors are retained.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from neural_forge.hf_tools import _limit_sequence, _load_eval_suite


EVALUATION_PROVENANCE_SCHEMA = "neural-forge-loaded-text-eval-provenance-v1"
DISJOINTNESS_SCHEMA = "neural-forge-eval-path-disjointness-v1"
EVALUATION_IDENTITY_SCHEMA = "neural-forge-loaded-text-eval-identity-v1"
_GENERATION_BATCH_SIZE = 8
_MIN_MEANINGFUL_PREFIX_TOKENS = 8
_TOKEN_SEQUENCE_HASH_NAMESPACE = b"neural-forge-token-sequence-v1\0"


class HFEvaluationProvenanceError(ValueError):
    """Raised when exact evaluator provenance cannot be established."""


class HFEvaluationPathOverlapError(HFEvaluationProvenanceError):
    """Raised when two nominally independent domains share a token path."""


class HFEvaluationCountError(HFEvaluationProvenanceError):
    """Raised when an evaluator report violates its bound item-count contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor):
        raise HFEvaluationProvenanceError("Tokenizer outputs must be torch tensors.")
    cpu = value.detach().to(device="cpu").contiguous()
    header = _canonical_json(
        {"dtype": str(cpu.dtype), "shape": [int(item) for item in cpu.shape]}
    ).encode("utf-8")
    raw = memoryview(cpu.view(torch.uint8).numpy()).tobytes()
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(raw)
    return digest.hexdigest()


def token_sequence_sha256(token_ids: Tensor | Sequence[int]) -> str:
    """Hash one ordered, unpadded token sequence using a stable int64 form."""

    if isinstance(token_ids, Tensor):
        values = token_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
    else:
        try:
            values = [operator.index(item) for item in token_ids]
        except (TypeError, ValueError, OverflowError) as exc:
            raise HFEvaluationProvenanceError(
                "Token sequences must contain integer token IDs."
            ) from exc
    digest = hashlib.sha256(_TOKEN_SEQUENCE_HASH_NAMESPACE)
    try:
        for token_id in values:
            digest.update(struct.pack("<q", int(token_id)))
    except (struct.error, OverflowError) as exc:
        raise HFEvaluationProvenanceError(
            "Token IDs must fit a signed 64-bit fixed-width encoding."
        ) from exc
    return digest.hexdigest()


def _source_text_sha256(text: str) -> str:
    return _canonical_sha256(str(text))


def _meaningful_token_prefix_sha256s(token_ids: Tensor) -> tuple[str, ...]:
    """Bind every non-trivial prefix so truncation cannot hide reuse.

    Prefixes shorter than eight tokens are intentionally excluded: a shared
    BOS plus a few boilerplate tokens is not meaningful evidence leakage.  An
    identical short path is still caught by the full content hash, and an
    identical short source is caught by the raw source-text hash.
    """

    values = token_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    count = int(values.numel())
    if count < _MIN_MEANINGFUL_PREFIX_TOKENS:
        return ()
    # Fixed-width token encoding makes copied incremental SHA states exact and
    # keeps this O(tokens), rather than re-hashing every growing tensor prefix.
    digest = hashlib.sha256(_TOKEN_SEQUENCE_HASH_NAMESPACE)
    result: list[str] = []
    for length, token_id in enumerate(values.tolist(), start=1):
        digest.update(struct.pack("<q", int(token_id)))
        if length >= _MIN_MEANINGFUL_PREFIX_TOKENS:
            result.append(digest.copy().hexdigest())
    return tuple(result)


def _rank_two_input_ids(encoded: Any, call_name: str) -> Tensor:
    if not isinstance(encoded, Mapping):
        raise HFEvaluationProvenanceError(
            f"{call_name} tokenizer result must be a mapping."
        )
    input_ids = encoded.get("input_ids")
    if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
        raise HFEvaluationProvenanceError(
            f"{call_name} tokenizer result requires rank-2 tensor input_ids."
        )
    return input_ids.detach().to(device="cpu")


def _mapping_tensor_hashes(encoded: Mapping[str, Any], call_name: str) -> tuple["NamedHash", ...]:
    result: list[NamedHash] = []
    for key in sorted(encoded):
        value = encoded[key]
        if not isinstance(value, Tensor):
            raise HFEvaluationProvenanceError(
                f"{call_name} tokenizer output {key!r} is not a tensor."
            )
        result.append(NamedHash(str(key), _tensor_sha256(value)))
    if not result:
        raise HFEvaluationProvenanceError(
            f"{call_name} tokenizer result cannot be empty."
        )
    return tuple(result)


def _row_tensor_hashes(
    encoded: Mapping[str, Any], row_index: int, call_name: str
) -> tuple["NamedHash", ...]:
    result: list[NamedHash] = []
    for key in sorted(encoded):
        value = encoded[key]
        if not isinstance(value, Tensor) or value.ndim < 1:
            raise HFEvaluationProvenanceError(
                f"{call_name} tokenizer output {key!r} is not a batched tensor."
            )
        if row_index >= int(value.shape[0]):
            raise HFEvaluationProvenanceError(
                f"{call_name} tokenizer output {key!r} has too few rows."
            )
        result.append(NamedHash(str(key), _tensor_sha256(value[row_index])))
    return tuple(result)


def _hash_named_hashes(values: Sequence["NamedHash"]) -> str:
    return _canonical_sha256([item.as_dict() for item in values])


def _category(item: Mapping[str, Any] | None, fallback: str) -> str:
    if item is None or item.get("category") is None:
        return fallback
    return str(item["category"])


def _limit(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise HFEvaluationProvenanceError(f"{name} must be non-negative.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HFEvaluationProvenanceError(f"{name} must be non-negative.") from exc
    if result < 0:
        raise HFEvaluationProvenanceError(f"{name} must be non-negative.")
    return result


@dataclass(frozen=True, slots=True)
class NamedHash:
    """One named SHA-256 binding."""

    name: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.name or len(self.sha256) != 64:
            raise HFEvaluationProvenanceError("Named hashes require a name and SHA-256.")

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class TokenizerCallProvenance:
    """The literal keyword path used for one evaluator tokenizer call."""

    call_kind: str
    keyword_arguments: tuple[tuple[str, str], ...]
    batch_size: int
    output_hashes: tuple[NamedHash, ...]
    call_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "call_kind": self.call_kind,
            "keyword_arguments": [list(item) for item in self.keyword_arguments],
            "batch_size": self.batch_size,
            "output_hashes": [item.as_dict() for item in self.output_hashes],
            "call_sha256": self.call_sha256,
        }


def _tokenizer_call(
    call_kind: str,
    keyword_arguments: Sequence[tuple[str, str]],
    batch_size: int,
    output_hashes: Sequence[NamedHash],
) -> TokenizerCallProvenance:
    payload = {
        "call_kind": str(call_kind),
        "keyword_arguments": [list(item) for item in keyword_arguments],
        "batch_size": int(batch_size),
        "output_hashes": [item.as_dict() for item in output_hashes],
    }
    return TokenizerCallProvenance(
        call_kind=str(call_kind),
        keyword_arguments=tuple((str(key), str(value)) for key, value in keyword_arguments),
        batch_size=int(batch_size),
        output_hashes=tuple(output_hashes),
        call_sha256=_canonical_sha256(payload),
    )


@dataclass(frozen=True, slots=True)
class ChoiceScoringSequenceProvenance:
    """Exact inputs and effective causal labels for one answer choice."""

    choice_index: int
    choice_text_sha256: str
    prompt_token_count: int
    label_mask_boundary: int
    full_token_count: int
    loss_scale_token_count: int
    scored_label_count: int
    tokenizer_call: TokenizerCallProvenance
    input_hashes: tuple[NamedHash, ...]
    raw_source_text_sha256: str
    token_content_sha256: str
    token_prefix_sha256s: tuple[str, ...]
    exact_path_sha256: str

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["tokenizer_call"] = self.tokenizer_call.as_dict()
        result["input_hashes"] = [item.as_dict() for item in self.input_hashes]
        return result


@dataclass(frozen=True, slots=True)
class ScoredMetricItemProvenance:
    """One immutable record for one item emitted by the evaluator report."""

    domain: Literal["perplexity", "choice", "generation", "behavior"]
    item_id: str
    category: str
    source_sha256: str
    input_sha256: str
    input_hashes: tuple[NamedHash, ...]
    tokenizer_calls: tuple[TokenizerCallProvenance, ...]
    prompt_text_sha256: str
    raw_source_text_sha256s: tuple[str, ...]
    token_content_sha256s: tuple[str, ...]
    token_prefix_sha256s: tuple[str, ...]
    exact_path_sha256s: tuple[str, ...]
    input_token_count: int
    scored_label_count: int
    expected_answer: int | None = None
    generation_batch_index: int | None = None
    generation_batch_row: int | None = None
    choice_sequences: tuple[ChoiceScoringSequenceProvenance, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "item_id": self.item_id,
            "category": self.category,
            "source_sha256": self.source_sha256,
            "input_sha256": self.input_sha256,
            "input_hashes": [item.as_dict() for item in self.input_hashes],
            "tokenizer_calls": [item.as_dict() for item in self.tokenizer_calls],
            "prompt_text_sha256": self.prompt_text_sha256,
            "raw_source_text_sha256s": list(self.raw_source_text_sha256s),
            "token_content_sha256s": list(self.token_content_sha256s),
            "token_prefix_sha256s": list(self.token_prefix_sha256s),
            "exact_path_sha256s": list(self.exact_path_sha256s),
            "input_token_count": self.input_token_count,
            "scored_label_count": self.scored_label_count,
            "expected_answer": self.expected_answer,
            "generation_batch_index": self.generation_batch_index,
            "generation_batch_row": self.generation_batch_row,
            "choice_sequences": [item.as_dict() for item in self.choice_sequences],
        }


@dataclass(frozen=True, slots=True)
class ScoredItemCountContract:
    """Counts and ordered report IDs expected from one bound evaluation."""

    perplexity_items: int
    choice_items: int
    generation_items: int
    behavior_items: int
    total_metric_items: int
    choice_scoring_sequences: int
    choice_ids: tuple[str, ...]
    generation_ids: tuple[str, ...]
    behavior_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in ("choice_ids", "generation_ids", "behavior_ids"):
            result[name] = list(result[name])
        return result


@dataclass(frozen=True, slots=True)
class EvaluationIdentityBinding:
    """Identity payload callers must pass to ``evaluate_loaded_text_model``."""

    schema: str
    evaluator_provenance_schema: str
    suite_source: str
    suite_content_sha256: str
    scored_items_sha256: str
    evaluation_aggregate_sha256: str
    count_contract_sha256: str
    choice_scoring_sequences: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LoadedTextEvaluationProvenance:
    """Aggregate immutable binding for one evaluator invocation."""

    schema: str
    suite_source: str
    suite_content_sha256: str
    max_new_tokens: int
    max_text_evals: int | None
    max_choice_probes: int | None
    max_generation_prompts: int | None
    max_behavior_probes: int | None
    tokenizer_truncation: Literal["omitted"]
    tokenizer_max_length: Literal["omitted"]
    generation_do_sample: Literal[False]
    generation_pad_token_id: int | None
    generation_batch_size: int
    scored_items: tuple[ScoredMetricItemProvenance, ...]
    count_contract: ScoredItemCountContract
    scored_items_sha256: str
    aggregate_sha256: str
    expected_identity_binding: EvaluationIdentityBinding

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "suite_source": self.suite_source,
            "suite_content_sha256": self.suite_content_sha256,
            "max_new_tokens": self.max_new_tokens,
            "max_text_evals": self.max_text_evals,
            "max_choice_probes": self.max_choice_probes,
            "max_generation_prompts": self.max_generation_prompts,
            "max_behavior_probes": self.max_behavior_probes,
            "tokenizer_truncation": self.tokenizer_truncation,
            "tokenizer_max_length": self.tokenizer_max_length,
            "generation_do_sample": self.generation_do_sample,
            "generation_pad_token_id": self.generation_pad_token_id,
            "generation_batch_size": self.generation_batch_size,
            "scored_items": [item.as_dict() for item in self.scored_items],
            "count_contract": self.count_contract.as_dict(),
            "scored_items_sha256": self.scored_items_sha256,
            "aggregate_sha256": self.aggregate_sha256,
            "expected_identity_binding": self.expected_identity_binding.as_dict(),
        }

    def evaluation_identity_binding(self) -> dict[str, Any]:
        """Return the exact payload for ``evaluate_loaded_text_model``."""

        return self.expected_identity_binding.as_dict()

    def expected_identity_dict(self) -> dict[str, Any]:
        """Backward-readable alias for :meth:`evaluation_identity_binding`."""

        return self.evaluation_identity_binding()

    def to_path_domain(self, name: str) -> "TokenPathDomain":
        exact = sorted(
            {value for item in self.scored_items for value in item.exact_path_sha256s}
        )
        content = sorted(
            {value for item in self.scored_items for value in item.token_content_sha256s}
        )
        raw_source = sorted(
            {
                value
                for item in self.scored_items
                for value in item.raw_source_text_sha256s
            }
        )
        prefixes = sorted(
            {value for item in self.scored_items for value in item.token_prefix_sha256s}
        )
        return _path_domain(str(name), exact, content, raw_source, prefixes)


@dataclass(frozen=True, slots=True)
class TokenPathDomain:
    """Hash-only domain used for strict train/select/evaluate separation."""

    name: str
    exact_path_sha256s: tuple[str, ...]
    token_content_sha256s: tuple[str, ...]
    raw_source_text_sha256s: tuple[str, ...]
    token_prefix_sha256s: tuple[str, ...]
    aggregate_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "exact_path_sha256s": list(self.exact_path_sha256s),
            "token_content_sha256s": list(self.token_content_sha256s),
            "raw_source_text_sha256s": list(self.raw_source_text_sha256s),
            "token_prefix_sha256s": list(self.token_prefix_sha256s),
            "aggregate_sha256": self.aggregate_sha256,
        }


@dataclass(frozen=True, slots=True)
class EvaluationPathDisjointnessReport:
    """Successful six-pair disjointness proof for the four required domains."""

    schema: str
    domains: tuple[TokenPathDomain, ...]
    pairwise_checks: tuple[tuple[str, str, int, int, int, int, int], ...]
    all_paths_disjoint: bool
    aggregate_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "domains": [item.as_dict() for item in self.domains],
            "pairwise_checks": [
                {
                    "left": left,
                    "right": right,
                    "exact_path_overlap_count": exact,
                    "token_content_overlap_count": content,
                    "raw_source_text_overlap_count": raw_source,
                    "left_full_content_in_right_prefix_overlap_count": left_in_right,
                    "right_full_content_in_left_prefix_overlap_count": right_in_left,
                    "token_prefix_overlap_count": left_in_right + right_in_left,
                }
                for (
                    left,
                    right,
                    exact,
                    content,
                    raw_source,
                    left_in_right,
                    right_in_left,
                ) in self.pairwise_checks
            ],
            "all_paths_disjoint": self.all_paths_disjoint,
            "aggregate_sha256": self.aggregate_sha256,
        }


@dataclass(frozen=True, slots=True)
class EvaluationCountVerification:
    """Immutable success record for a report identity/count comparison."""

    expected_total: int
    observed_total: int
    ordered_ids_match: bool
    identity_binding_match: bool
    suite_content_match: bool
    choice_scoring_sequences: int
    report_sha256: str


def _path_domain(
    name: str,
    exact_hashes: Iterable[str],
    content_hashes: Iterable[str],
    raw_source_hashes: Iterable[str],
    prefix_hashes: Iterable[str],
) -> TokenPathDomain:
    exact = tuple(sorted(set(str(item) for item in exact_hashes)))
    content = tuple(sorted(set(str(item) for item in content_hashes)))
    raw_source = tuple(sorted(set(str(item) for item in raw_source_hashes)))
    prefixes = tuple(sorted(set(str(item) for item in prefix_hashes)))
    if not name or not exact or not content or not raw_source:
        raise HFEvaluationProvenanceError(
            "A token-path domain requires a name and non-empty exact/content/source "
            "hashes."
        )
    payload = {
        "name": name,
        "exact_path_sha256s": list(exact),
        "token_content_sha256s": list(content),
        "raw_source_text_sha256s": list(raw_source),
        "token_prefix_sha256s": list(prefixes),
    }
    return TokenPathDomain(
        name,
        exact,
        content,
        raw_source,
        prefixes,
        _canonical_sha256(payload),
    )


def _perplexity_item(tokenizer: Any, text: str, index: int) -> ScoredMetricItemProvenance:
    encoded = tokenizer(str(text), return_tensors="pt")
    input_ids = _rank_two_input_ids(encoded, "perplexity")
    labels = input_ids.clone()
    effective_inputs = input_ids[:, :-1]
    effective_labels = labels[:, 1:]
    hashes = (
        NamedHash("model_input_ids", _tensor_sha256(input_ids)),
        NamedHash("model_labels", _tensor_sha256(labels)),
        NamedHash("effective_causal_input_ids", _tensor_sha256(effective_inputs)),
        NamedHash("effective_causal_label_ids", _tensor_sha256(effective_labels)),
    )
    call = _tokenizer_call(
        "perplexity_text",
        (("return_tensors", "pt"),),
        1,
        _mapping_tensor_hashes(encoded, "perplexity"),
    )
    content_hash = token_sequence_sha256(input_ids[0])
    prefix_hashes = _meaningful_token_prefix_sha256s(input_ids[0])
    text_sha = _source_text_sha256(text)
    exact_path = _canonical_sha256(
        {"input_hashes": [item.as_dict() for item in hashes]}
    )
    input_sha = _canonical_sha256(
        {"tokenizer_call": call.call_sha256, "exact_path": exact_path}
    )
    return ScoredMetricItemProvenance(
        domain="perplexity",
        item_id=f"text_{index:04d}",
        category="text",
        source_sha256=_canonical_sha256(str(text)),
        input_sha256=input_sha,
        input_hashes=hashes,
        tokenizer_calls=(call,),
        prompt_text_sha256=text_sha,
        raw_source_text_sha256s=(text_sha,),
        token_content_sha256s=(content_hash,),
        token_prefix_sha256s=prefix_hashes,
        exact_path_sha256s=(exact_path,),
        input_token_count=int(input_ids.shape[1]),
        scored_label_count=max(0, int(input_ids.shape[1]) - 1),
    )


def _choice_item(
    tokenizer: Any, probe: Mapping[str, Any]
) -> ScoredMetricItemProvenance:
    prompt = str(probe["prompt"])
    prompt_encoded = tokenizer(prompt, return_tensors="pt")
    prompt_ids = _rank_two_input_ids(prompt_encoded, "choice prompt")
    prompt_len = int(prompt_ids.shape[1])
    prompt_call = _tokenizer_call(
        "choice_prompt_boundary",
        (("return_tensors", "pt"),),
        1,
        _mapping_tensor_hashes(prompt_encoded, "choice prompt"),
    )
    content_hashes = [token_sequence_sha256(prompt_ids[0])]
    prefix_hashes = list(_meaningful_token_prefix_sha256s(prompt_ids[0]))
    prompt_text_sha = _source_text_sha256(prompt)
    raw_source_hashes = [prompt_text_sha]
    sequences: list[ChoiceScoringSequenceProvenance] = []
    for choice_index, choice_value in enumerate(probe["choices"]):
        choice = str(choice_value)
        encoded = tokenizer(prompt + choice, return_tensors="pt")
        full_ids = _rank_two_input_ids(encoded, "choice sequence")
        labels = full_ids.clone()
        labels[:, :prompt_len] = -100
        active_labels = labels[:, 1:] != -100
        effective_inputs = full_ids[:, :-1][active_labels]
        effective_labels = labels[:, 1:][active_labels]
        scored_label_count = int(active_labels.sum().item())
        loss_scale_token_count = int((labels != -100).sum().item())
        hashes = (
            NamedHash("model_input_ids", _tensor_sha256(full_ids)),
            NamedHash("model_labels", _tensor_sha256(labels)),
            NamedHash("effective_causal_input_ids", _tensor_sha256(effective_inputs)),
            NamedHash("effective_causal_label_ids", _tensor_sha256(effective_labels)),
        )
        call = _tokenizer_call(
            "choice_prompt_plus_choice",
            (("return_tensors", "pt"),),
            1,
            _mapping_tensor_hashes(encoded, "choice sequence"),
        )
        content_hash = token_sequence_sha256(full_ids[0])
        content_hashes.append(content_hash)
        choice_source_sha = _source_text_sha256(prompt + choice)
        raw_source_hashes.append(choice_source_sha)
        choice_prefix_hashes = _meaningful_token_prefix_sha256s(full_ids[0])
        prefix_hashes.extend(choice_prefix_hashes)
        exact_path = _canonical_sha256(
            {
                "label_mask_boundary": prompt_len,
                "input_hashes": [item.as_dict() for item in hashes],
            }
        )
        sequences.append(
            ChoiceScoringSequenceProvenance(
                choice_index=choice_index,
                choice_text_sha256=_canonical_sha256(choice),
                prompt_token_count=prompt_len,
                label_mask_boundary=prompt_len,
                full_token_count=int(full_ids.shape[1]),
                loss_scale_token_count=loss_scale_token_count,
                scored_label_count=scored_label_count,
                tokenizer_call=call,
                input_hashes=hashes,
                raw_source_text_sha256=choice_source_sha,
                token_content_sha256=content_hash,
                token_prefix_sha256s=choice_prefix_hashes,
                exact_path_sha256=exact_path,
            )
        )
    if not sequences:
        raise HFEvaluationProvenanceError("A choice probe must have at least one choice.")
    sequence_hashes = tuple(item.exact_path_sha256 for item in sequences)
    hashes = (
        NamedHash("prompt_boundary_input_ids", _tensor_sha256(prompt_ids)),
        NamedHash("choice_scoring_sequences", _canonical_sha256(list(sequence_hashes))),
    )
    input_sha = _canonical_sha256(
        {
            "prompt_call": prompt_call.call_sha256,
            "choice_paths": list(sequence_hashes),
        }
    )
    return ScoredMetricItemProvenance(
        domain="choice",
        item_id=str(probe["id"]),
        category=_category(probe, "uncategorized"),
        source_sha256=_canonical_sha256(dict(probe)),
        input_sha256=input_sha,
        input_hashes=hashes,
        tokenizer_calls=(prompt_call,) + tuple(item.tokenizer_call for item in sequences),
        prompt_text_sha256=prompt_text_sha,
        raw_source_text_sha256s=tuple(raw_source_hashes),
        token_content_sha256s=tuple(content_hashes),
        token_prefix_sha256s=tuple(sorted(set(prefix_hashes))),
        exact_path_sha256s=sequence_hashes,
        input_token_count=prompt_len,
        scored_label_count=sum(item.scored_label_count for item in sequences),
        expected_answer=int(probe["answer"]),
        choice_sequences=tuple(sequences),
    )


@dataclass(frozen=True, slots=True)
class _GenerationInputPath:
    prompt: str
    batch_index: int
    batch_row: int
    input_token_count: int
    tokenizer_call: TokenizerCallProvenance
    input_hashes: tuple[NamedHash, ...]
    raw_source_text_sha256: str
    content_sha256: str
    token_prefix_sha256s: tuple[str, ...]
    exact_path_sha256: str


def _generation_input_paths(
    tokenizer: Any, prompts: Sequence[str]
) -> dict[str, _GenerationInputPath]:
    if not prompts:
        return {}
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(
        tokenizer, "eos_token_id", None
    ) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    old_padding_side = getattr(tokenizer, "padding_side", None)
    if old_padding_side is not None:
        tokenizer.padding_side = "left"
    result: dict[str, _GenerationInputPath] = {}
    try:
        for batch_index, start in enumerate(range(0, len(prompts), _GENERATION_BATCH_SIZE)):
            batch_prompts = list(prompts[start : start + _GENERATION_BATCH_SIZE])
            encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True)
            input_ids = _rank_two_input_ids(encoded, "generation batch")
            if int(input_ids.shape[0]) != len(batch_prompts):
                raise HFEvaluationProvenanceError(
                    "Generation tokenizer row count does not match prompt count."
                )
            batch_hashes = _mapping_tensor_hashes(encoded, "generation batch")
            call = _tokenizer_call(
                "generation_left_padded_batch",
                (("return_tensors", "pt"), ("padding", "True")),
                len(batch_prompts),
                batch_hashes,
            )
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None and (
                not isinstance(attention_mask, Tensor)
                or tuple(attention_mask.shape) != tuple(input_ids.shape)
            ):
                raise HFEvaluationProvenanceError(
                    "Generation attention_mask shape does not match input_ids."
                )
            for row, prompt in enumerate(batch_prompts):
                row_hashes = _row_tensor_hashes(encoded, row, "generation batch")
                if isinstance(attention_mask, Tensor):
                    active = input_ids[row][attention_mask[row].to(dtype=torch.bool)]
                else:
                    active = input_ids[row]
                content_sha = token_sequence_sha256(active)
                prefix_hashes = _meaningful_token_prefix_sha256s(active)
                exact_path = _canonical_sha256(
                    {
                        "batch_call_sha256": call.call_sha256,
                        "batch_row": row,
                        "row_hashes": [item.as_dict() for item in row_hashes],
                    }
                )
                result[prompt] = _GenerationInputPath(
                    prompt=prompt,
                    batch_index=batch_index,
                    batch_row=row,
                    input_token_count=int(active.numel()),
                    tokenizer_call=call,
                    input_hashes=row_hashes
                    + (NamedHash("full_batch_model_inputs", _hash_named_hashes(batch_hashes)),),
                    raw_source_text_sha256=_source_text_sha256(prompt),
                    content_sha256=content_sha,
                    token_prefix_sha256s=prefix_hashes,
                    exact_path_sha256=exact_path,
                )
    finally:
        if old_padding_side is not None:
            tokenizer.padding_side = old_padding_side
    return result


def _generated_metric_item(
    domain: Literal["generation", "behavior"],
    spec: Mapping[str, Any],
    path: _GenerationInputPath,
) -> ScoredMetricItemProvenance:
    input_sha = _canonical_sha256(
        {
            "model_input_path": path.exact_path_sha256,
            "max_new_tokens_is_bound_at_suite_level": True,
        }
    )
    return ScoredMetricItemProvenance(
        domain=domain,
        item_id=str(spec["id"]),
        category=_category(spec, "uncategorized"),
        source_sha256=_canonical_sha256(dict(spec)),
        input_sha256=input_sha,
        input_hashes=path.input_hashes,
        tokenizer_calls=(path.tokenizer_call,),
        prompt_text_sha256=path.raw_source_text_sha256,
        raw_source_text_sha256s=(path.raw_source_text_sha256,),
        token_content_sha256s=(path.content_sha256,),
        token_prefix_sha256s=path.token_prefix_sha256s,
        exact_path_sha256s=(path.exact_path_sha256,),
        input_token_count=path.input_token_count,
        scored_label_count=0,
        generation_batch_index=path.batch_index,
        generation_batch_row=path.batch_row,
    )


def bind_loaded_text_evaluation_provenance(
    tokenizer: Any,
    eval_suite: str | Mapping[str, Any] | Any | None = None,
    *,
    max_new_tokens: int = 96,
    max_text_evals: int | None = None,
    max_choice_probes: int | None = None,
    max_generation_prompts: int | None = None,
    max_behavior_probes: int | None = None,
) -> LoadedTextEvaluationProvenance:
    """Bind the exact paths scored by ``evaluate_loaded_text_model``.

    Evaluation tokenization intentionally has no ``max_length`` or
    ``truncation`` keyword: that is the evaluator's current literal contract.
    Collector prompt domains use :func:`bind_prompt_path_domain` instead.
    """

    limits = {
        "max_text_evals": _limit(max_text_evals, "max_text_evals"),
        "max_choice_probes": _limit(max_choice_probes, "max_choice_probes"),
        "max_generation_prompts": _limit(
            max_generation_prompts, "max_generation_prompts"
        ),
        "max_behavior_probes": _limit(max_behavior_probes, "max_behavior_probes"),
    }
    if isinstance(max_new_tokens, bool):
        raise HFEvaluationProvenanceError("max_new_tokens must be positive.")
    try:
        normalized_max_new_tokens = operator.index(max_new_tokens)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HFEvaluationProvenanceError("max_new_tokens must be positive.") from exc
    if normalized_max_new_tokens <= 0:
        raise HFEvaluationProvenanceError("max_new_tokens must be positive.")

    suite = _load_eval_suite(eval_suite)
    text_evals = _limit_sequence(suite["text_evals"], limits["max_text_evals"])
    choices = _limit_sequence(suite["choice_probes"], limits["max_choice_probes"])
    generations = _limit_sequence(
        suite["generation_prompts"], limits["max_generation_prompts"]
    )
    behaviors = _limit_sequence(
        suite["behavior_probes"], limits["max_behavior_probes"]
    )

    # Match the pad-token mutation performed before evaluate_loaded_text_model's
    # no-grad block.  Generation batching performs the same guard once more.
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(
        tokenizer, "eos_token_id", None
    ) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    items: list[ScoredMetricItemProvenance] = []
    items.extend(
        _perplexity_item(tokenizer, str(text), index)
        for index, text in enumerate(text_evals)
    )
    items.extend(_choice_item(tokenizer, probe) for probe in choices)

    requested_prompts: list[str] = []
    seen_prompts: set[str] = set()
    for spec in generations:
        prompt = str(spec["prompt"])
        if prompt not in seen_prompts:
            seen_prompts.add(prompt)
            requested_prompts.append(prompt)
    for probe in behaviors:
        prompt = str(probe["prompt"])
        if prompt not in seen_prompts:
            seen_prompts.add(prompt)
            requested_prompts.append(prompt)
    generation_paths = _generation_input_paths(tokenizer, requested_prompts)
    items.extend(
        _generated_metric_item("generation", spec, generation_paths[str(spec["prompt"])])
        for spec in generations
    )
    items.extend(
        _generated_metric_item("behavior", probe, generation_paths[str(probe["prompt"])])
        for probe in behaviors
    )

    count_contract = ScoredItemCountContract(
        perplexity_items=len(text_evals),
        choice_items=len(choices),
        generation_items=len(generations),
        behavior_items=len(behaviors),
        total_metric_items=len(text_evals) + len(choices) + len(generations) + len(behaviors),
        choice_scoring_sequences=sum(len(item["choices"]) for item in choices),
        choice_ids=tuple(str(item["id"]) for item in choices),
        generation_ids=tuple(str(item["id"]) for item in generations),
        behavior_ids=tuple(str(item["id"]) for item in behaviors),
    )
    suite_payload = {
        "text_evals": list(text_evals),
        "choice_probes": list(choices),
        "generation_prompts": list(generations),
        "behavior_probes": list(behaviors),
    }
    suite_content_sha = _canonical_sha256(suite_payload)
    scored_items_sha = _canonical_sha256([item.as_dict() for item in items])
    aggregate_payload = {
        "schema": EVALUATION_PROVENANCE_SCHEMA,
        "suite_source": str(suite["source"]),
        "suite_content_sha256": suite_content_sha,
        "max_new_tokens": normalized_max_new_tokens,
        **limits,
        "tokenizer_truncation": "omitted",
        "tokenizer_max_length": "omitted",
        "generation_do_sample": False,
        "generation_pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "generation_batch_size": _GENERATION_BATCH_SIZE,
        "scored_items_sha256": scored_items_sha,
        "count_contract": count_contract.as_dict(),
    }
    aggregate_sha = _canonical_sha256(aggregate_payload)
    identity_binding = EvaluationIdentityBinding(
        schema=EVALUATION_IDENTITY_SCHEMA,
        evaluator_provenance_schema=EVALUATION_PROVENANCE_SCHEMA,
        suite_source=str(suite["source"]),
        suite_content_sha256=suite_content_sha,
        scored_items_sha256=scored_items_sha,
        evaluation_aggregate_sha256=aggregate_sha,
        count_contract_sha256=_canonical_sha256(count_contract.as_dict()),
        choice_scoring_sequences=count_contract.choice_scoring_sequences,
    )
    return LoadedTextEvaluationProvenance(
        schema=EVALUATION_PROVENANCE_SCHEMA,
        suite_source=str(suite["source"]),
        suite_content_sha256=suite_content_sha,
        max_new_tokens=normalized_max_new_tokens,
        max_text_evals=limits["max_text_evals"],
        max_choice_probes=limits["max_choice_probes"],
        max_generation_prompts=limits["max_generation_prompts"],
        max_behavior_probes=limits["max_behavior_probes"],
        tokenizer_truncation="omitted",
        tokenizer_max_length="omitted",
        generation_do_sample=False,
        generation_pad_token_id=getattr(tokenizer, "pad_token_id", None),
        generation_batch_size=_GENERATION_BATCH_SIZE,
        scored_items=tuple(items),
        count_contract=count_contract,
        scored_items_sha256=scored_items_sha,
        aggregate_sha256=aggregate_sha,
        expected_identity_binding=identity_binding,
    )


def bind_prompt_path_domain(
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    name: str,
    max_length: int,
) -> TokenPathDomain:
    """Bind the runner's one-row calibration/sensitivity tokenizer path.

    This matches its explicit ``truncation=True``, ``max_length=N``, and
    ``add_special_tokens=True`` call, while evaluator paths remain untruncated.
    """

    if isinstance(max_length, bool):
        raise HFEvaluationProvenanceError("max_length must be positive.")
    try:
        length = operator.index(max_length)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HFEvaluationProvenanceError("max_length must be positive.") from exc
    if length <= 0:
        raise HFEvaluationProvenanceError("max_length must be positive.")
    exact: list[str] = []
    content: list[str] = []
    raw_source: list[str] = []
    prefixes: list[str] = []
    for prompt in prompts:
        encoded = tokenizer(
            str(prompt),
            return_tensors="pt",
            truncation=True,
            max_length=length,
            add_special_tokens=True,
        )
        input_ids = _rank_two_input_ids(encoded, name)
        if int(input_ids.shape[0]) != 1 or int(input_ids.shape[1]) == 0:
            raise HFEvaluationProvenanceError(
                "Collector prompts must tokenize to one non-empty row."
            )
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            active = input_ids[0]
        elif isinstance(attention_mask, Tensor) and tuple(attention_mask.shape) == tuple(
            input_ids.shape
        ):
            active = input_ids[0][attention_mask[0].to(dtype=torch.bool)]
        else:
            raise HFEvaluationProvenanceError(
                "Collector attention_mask shape does not match input_ids."
            )
        if not int(active.numel()):
            raise HFEvaluationProvenanceError(
                "Collector prompts must contain at least one active token."
            )
        row_hashes = _mapping_tensor_hashes(encoded, name)
        exact.append(
            _canonical_sha256(
                {
                    "keyword_arguments": [
                        ["return_tensors", "pt"],
                        ["truncation", "True"],
                        ["max_length", str(length)],
                        ["add_special_tokens", "True"],
                    ],
                    "output_hashes": [item.as_dict() for item in row_hashes],
                }
            )
        )
        content.append(token_sequence_sha256(active))
        raw_source.append(_source_text_sha256(str(prompt)))
        prefixes.extend(_meaningful_token_prefix_sha256s(active))
    return _path_domain(str(name), exact, content, raw_source, prefixes)


def evaluation_identity_binding(
    provenance: LoadedTextEvaluationProvenance,
) -> dict[str, Any]:
    """Stable function form of the evaluator ``identity_binding`` payload."""

    if not isinstance(provenance, LoadedTextEvaluationProvenance):
        raise HFEvaluationProvenanceError(
            "evaluation_identity_binding requires loaded-text provenance."
        )
    return provenance.evaluation_identity_binding()


def assert_strict_evaluation_path_disjointness(
    *,
    calibration: TokenPathDomain,
    sensitivity: TokenPathDomain,
    candidate: LoadedTextEvaluationProvenance | TokenPathDomain,
    transaction: LoadedTextEvaluationProvenance | TokenPathDomain,
) -> EvaluationPathDisjointnessReport:
    """Prove all six calibration/sensitivity/candidate/transaction pairs disjoint."""

    candidate_domain = (
        candidate.to_path_domain("candidate")
        if isinstance(candidate, LoadedTextEvaluationProvenance)
        else _path_domain(
            "candidate",
            candidate.exact_path_sha256s,
            candidate.token_content_sha256s,
            candidate.raw_source_text_sha256s,
            candidate.token_prefix_sha256s,
        )
    )
    transaction_domain = (
        transaction.to_path_domain("transaction")
        if isinstance(transaction, LoadedTextEvaluationProvenance)
        else _path_domain(
            "transaction",
            transaction.exact_path_sha256s,
            transaction.token_content_sha256s,
            transaction.raw_source_text_sha256s,
            transaction.token_prefix_sha256s,
        )
    )
    calibration_domain = _path_domain(
        "calibration",
        calibration.exact_path_sha256s,
        calibration.token_content_sha256s,
        calibration.raw_source_text_sha256s,
        calibration.token_prefix_sha256s,
    )
    sensitivity_domain = _path_domain(
        "sensitivity",
        sensitivity.exact_path_sha256s,
        sensitivity.token_content_sha256s,
        sensitivity.raw_source_text_sha256s,
        sensitivity.token_prefix_sha256s,
    )
    domains = (
        calibration_domain,
        sensitivity_domain,
        candidate_domain,
        transaction_domain,
    )
    checks: list[tuple[str, str, int, int, int, int, int]] = []
    overlaps: list[dict[str, Any]] = []
    for left_index, left in enumerate(domains):
        for right in domains[left_index + 1 :]:
            exact_overlap = sorted(
                set(left.exact_path_sha256s) & set(right.exact_path_sha256s)
            )
            content_overlap = sorted(
                set(left.token_content_sha256s) & set(right.token_content_sha256s)
            )
            raw_source_overlap = sorted(
                set(left.raw_source_text_sha256s)
                & set(right.raw_source_text_sha256s)
            )
            left_full_in_right_prefix = sorted(
                set(left.token_content_sha256s) & set(right.token_prefix_sha256s)
            )
            right_full_in_left_prefix = sorted(
                set(right.token_content_sha256s) & set(left.token_prefix_sha256s)
            )
            checks.append(
                (
                    left.name,
                    right.name,
                    len(exact_overlap),
                    len(content_overlap),
                    len(raw_source_overlap),
                    len(left_full_in_right_prefix),
                    len(right_full_in_left_prefix),
                )
            )
            if (
                exact_overlap
                or content_overlap
                or raw_source_overlap
                or left_full_in_right_prefix
                or right_full_in_left_prefix
            ):
                overlaps.append(
                    {
                        "left": left.name,
                        "right": right.name,
                        "exact_path_overlap_count": len(exact_overlap),
                        "token_content_overlap_count": len(content_overlap),
                        "raw_source_text_overlap_count": len(raw_source_overlap),
                        "left_full_content_in_right_prefix_overlap_count": len(
                            left_full_in_right_prefix
                        ),
                        "right_full_content_in_left_prefix_overlap_count": len(
                            right_full_in_left_prefix
                        ),
                        "token_prefix_overlap_count": len(
                            left_full_in_right_prefix
                        )
                        + len(right_full_in_left_prefix),
                        "exact_path_overlap_sample": exact_overlap[:3],
                        "token_content_overlap_sample": content_overlap[:3],
                        "raw_source_text_overlap_sample": raw_source_overlap[:3],
                        "left_full_content_in_right_prefix_overlap_sample": (
                            left_full_in_right_prefix[:3]
                        ),
                        "right_full_content_in_left_prefix_overlap_sample": (
                            right_full_in_left_prefix[:3]
                        ),
                    }
                )
    if overlaps:
        raise HFEvaluationPathOverlapError(
            "Evaluation token paths are not strictly disjoint: "
            + _canonical_json(overlaps)
        )
    payload = {
        "schema": DISJOINTNESS_SCHEMA,
        "domains": [item.as_dict() for item in domains],
        "pairwise_checks": [list(item) for item in checks],
        "all_paths_disjoint": True,
    }
    return EvaluationPathDisjointnessReport(
        schema=DISJOINTNESS_SCHEMA,
        domains=domains,
        pairwise_checks=tuple(checks),
        all_paths_disjoint=True,
        aggregate_sha256=_canonical_sha256(payload),
    )


def assert_evaluation_report_count_contract(
    provenance: LoadedTextEvaluationProvenance,
    report: Mapping[str, Any],
) -> EvaluationCountVerification:
    """Fail closed unless a report is the exact bound evaluator invocation.

    Counts and IDs are necessary but deliberately insufficient.  The evaluator
    must echo the precomputed identity payload, and every report field that can
    be tied back to a scored suite item is checked independently.
    """

    contract = provenance.count_contract
    expected_binding = provenance.evaluation_identity_binding()
    observed_binding = report.get("identity_binding")
    if not isinstance(observed_binding, Mapping):
        raise HFEvaluationCountError(
            "Evaluator report is missing the required evaluation identity binding."
        )
    try:
        binding_matches = _canonical_sha256(dict(observed_binding)) == _canonical_sha256(
            expected_binding
        )
    except (TypeError, ValueError) as exc:
        raise HFEvaluationCountError(
            "Evaluator identity binding is not canonical JSON data."
        ) from exc
    if not binding_matches:
        raise HFEvaluationCountError(
            "Evaluator identity binding does not match the bound suite source, "
            "content aggregate, or token paths."
        )
    try:
        report_counts = report["eval_suite"]
        ppl_items = report["perplexity"]["items"]
        choice_items = report["choice_accuracy"]["items"]
        generations = report["generations"]
        generation_diagnostics = report["generation_diagnostics"]
        generation_summary = report["generation_health"]
        behavior_items = report["behavior_suite"]["items"]
        behavior_summary = report["behavior_suite"]
    except (KeyError, TypeError) as exc:
        raise HFEvaluationCountError(
            "Evaluator report is missing count-contract fields."
        ) from exc
    if not isinstance(report_counts, Mapping) or str(report_counts.get("source")) != str(
        provenance.suite_source
    ):
        raise HFEvaluationCountError(
            "Evaluator suite source does not match the identity-bound suite."
        )
    observed = {
        "perplexity": len(ppl_items),
        "choice": len(choice_items),
        "generation": len(generations),
        "behavior": len(behavior_items),
    }
    expected = {
        "perplexity": contract.perplexity_items,
        "choice": contract.choice_items,
        "generation": contract.generation_items,
        "behavior": contract.behavior_items,
    }
    advertised = {
        "perplexity": report_counts.get("text_eval_count"),
        "choice": report_counts.get("choice_probe_count"),
        "generation": report_counts.get("generation_prompt_count"),
        "behavior": report_counts.get("behavior_probe_count"),
    }
    if observed != expected or advertised != expected:
        raise HFEvaluationCountError(
            f"Evaluator item-count contract failed: expected={expected}, "
            f"advertised={advertised}, observed={observed}."
        )
    observed_ids = {
        "choice": tuple(str(item.get("id")) for item in choice_items),
        "generation": tuple(str(item.get("id")) for item in generations),
        "behavior": tuple(str(item.get("id")) for item in behavior_items),
    }
    expected_ids = {
        "choice": contract.choice_ids,
        "generation": contract.generation_ids,
        "behavior": contract.behavior_ids,
    }
    if observed_ids != expected_ids:
        raise HFEvaluationCountError(
            f"Evaluator ordered item IDs changed: expected={expected_ids}, "
            f"observed={observed_ids}."
        )

    expected_items = {
        domain: tuple(item for item in provenance.scored_items if item.domain == domain)
        for domain in ("perplexity", "choice", "generation", "behavior")
    }
    for value in ppl_items:
        if isinstance(value, bool):
            raise HFEvaluationCountError("Perplexity items must be finite numbers.")
        try:
            normalized = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise HFEvaluationCountError(
                "Perplexity items must be finite numbers."
            ) from exc
        if not math.isfinite(normalized):
            raise HFEvaluationCountError("Perplexity items must be finite numbers.")

    observed_choice_sequences = 0
    observed_choice_correct = 0
    for expected_item, observed_item in zip(
        expected_items["choice"], choice_items, strict=True
    ):
        if not isinstance(observed_item, Mapping):
            raise HFEvaluationCountError("Choice report items must be objects.")
        choices = observed_item.get("choices")
        scores = observed_item.get("scores")
        if (
            not isinstance(choices, Sequence)
            or isinstance(choices, (str, bytes))
            or not isinstance(scores, Sequence)
            or isinstance(scores, (str, bytes))
        ):
            raise HFEvaluationCountError(
                "Each choice item requires choice text and score sequences."
            )
        expected_choice_hashes = tuple(
            item.choice_text_sha256 for item in expected_item.choice_sequences
        )
        observed_choice_hashes = tuple(
            _source_text_sha256(str(choice)) for choice in choices
        )
        if observed_choice_hashes != expected_choice_hashes:
            raise HFEvaluationCountError(
                f"Choice text hashes changed for {expected_item.item_id!r}."
            )
        if len(scores) != len(expected_choice_hashes):
            raise HFEvaluationCountError(
                f"Choice score count changed for {expected_item.item_id!r}."
            )
        normalized_scores: list[float] = []
        for score in scores:
            if isinstance(score, bool):
                raise HFEvaluationCountError("Choice scores must be finite numbers.")
            try:
                normalized_score = float(score)
            except (TypeError, ValueError, OverflowError) as exc:
                raise HFEvaluationCountError(
                    "Choice scores must be finite numbers."
                ) from exc
            if not math.isfinite(normalized_score):
                raise HFEvaluationCountError("Choice scores must be finite numbers.")
            normalized_scores.append(normalized_score)
        answer = observed_item.get("answer")
        try:
            normalized_answer = operator.index(answer)
        except (TypeError, ValueError, OverflowError) as exc:
            raise HFEvaluationCountError(
                f"Choice answer changed for {expected_item.item_id!r}."
            ) from exc
        if isinstance(answer, bool) or normalized_answer != expected_item.expected_answer:
            raise HFEvaluationCountError(
                f"Choice answer changed for {expected_item.item_id!r}."
            )
        prediction = observed_item.get("prediction")
        try:
            normalized_prediction = operator.index(prediction)
        except (TypeError, ValueError, OverflowError) as exc:
            raise HFEvaluationCountError(
                f"Choice prediction is inconsistent with scores for "
                f"{expected_item.item_id!r}."
            ) from exc
        expected_prediction = max(
            range(len(normalized_scores)), key=lambda index: normalized_scores[index]
        )
        if isinstance(prediction, bool) or normalized_prediction != expected_prediction:
            raise HFEvaluationCountError(
                f"Choice prediction is inconsistent with scores for "
                f"{expected_item.item_id!r}."
            )
        expected_correct = expected_prediction == expected_item.expected_answer
        if observed_item.get("correct") is not expected_correct:
            raise HFEvaluationCountError(
                f"Choice correctness is inconsistent for {expected_item.item_id!r}."
            )
        observed_choice_correct += int(expected_correct)
        observed_choice_sequences += len(normalized_scores)

    if observed_choice_sequences != contract.choice_scoring_sequences:
        raise HFEvaluationCountError(
            "Choice scoring-sequence count does not match the bound contract."
        )
    choice_summary = report.get("choice_accuracy")
    if not isinstance(choice_summary, Mapping):
        raise HFEvaluationCountError("Choice summary is missing.")
    if (
        choice_summary.get("correct") != observed_choice_correct
        or choice_summary.get("total") != contract.choice_items
    ):
        raise HFEvaluationCountError("Choice summary counts are inconsistent.")
    expected_accuracy = observed_choice_correct / max(1, contract.choice_items)
    try:
        observed_accuracy = float(choice_summary.get("accuracy"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise HFEvaluationCountError("Choice summary accuracy is invalid.") from exc
    if not math.isfinite(observed_accuracy) or observed_accuracy != expected_accuracy:
        raise HFEvaluationCountError("Choice summary accuracy is inconsistent.")

    for domain, report_items in (
        ("generation", generations),
        ("behavior", behavior_items),
    ):
        for expected_item, observed_item in zip(
            expected_items[domain], report_items, strict=True
        ):
            if not isinstance(observed_item, Mapping):
                raise HFEvaluationCountError(f"{domain} report items must be objects.")
            if str(observed_item.get("id")) != expected_item.item_id:
                raise HFEvaluationCountError(f"{domain} item ID changed.")
            prompt = observed_item.get("prompt")
            if not isinstance(prompt, str) or _source_text_sha256(
                prompt
            ) != expected_item.prompt_text_sha256:
                raise HFEvaluationCountError(
                    f"{domain} prompt changed for {expected_item.item_id!r}."
                )

    if (
        not isinstance(generation_diagnostics, Sequence)
        or isinstance(generation_diagnostics, (str, bytes))
        or len(generation_diagnostics) != contract.generation_items
    ):
        raise HFEvaluationCountError(
            "Generation diagnostics do not match the bound item count."
        )
    generation_correct = 0
    for expected_item, diagnostic in zip(
        expected_items["generation"], generation_diagnostics, strict=True
    ):
        if not isinstance(diagnostic, Mapping):
            raise HFEvaluationCountError("Generation diagnostics must be objects.")
        if str(diagnostic.get("id")) != expected_item.item_id:
            raise HFEvaluationCountError("Generation diagnostic item ID changed.")
        prompt = diagnostic.get("prompt")
        if not isinstance(prompt, str) or _source_text_sha256(
            prompt
        ) != expected_item.prompt_text_sha256:
            raise HFEvaluationCountError(
                f"Generation diagnostic prompt changed for {expected_item.item_id!r}."
            )
        passed = diagnostic.get("passed")
        if not isinstance(passed, bool):
            raise HFEvaluationCountError(
                "Generation diagnostic pass flags must be booleans."
            )
        generation_correct += int(passed)

    behavior_correct = 0
    for observed_item in behavior_items:
        if not isinstance(observed_item, Mapping):
            raise HFEvaluationCountError("Behavior report items must be objects.")
        passed = observed_item.get("passed")
        if not isinstance(passed, bool):
            raise HFEvaluationCountError("Behavior pass flags must be booleans.")
        behavior_correct += int(passed)

    for name, summary, correct, total in (
        (
            "Generation",
            generation_summary,
            generation_correct,
            contract.generation_items,
        ),
        ("Behavior", behavior_summary, behavior_correct, contract.behavior_items),
    ):
        if not isinstance(summary, Mapping):
            raise HFEvaluationCountError(f"{name} summary is missing.")
        if summary.get("correct") != correct or summary.get("total") != total:
            raise HFEvaluationCountError(f"{name} summary counts are inconsistent.")
        try:
            pass_rate = float(summary.get("pass_rate"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise HFEvaluationCountError(f"{name} summary pass rate is invalid.") from exc
        expected_pass_rate = correct / max(1, total)
        if not math.isfinite(pass_rate) or pass_rate != expected_pass_rate:
            raise HFEvaluationCountError(f"{name} summary pass rate is inconsistent.")

    observed_total = sum(observed.values())
    if observed_total != contract.total_metric_items:
        raise HFEvaluationCountError(
            "Evaluator total metric-item count changed after per-domain checks."
        )
    return EvaluationCountVerification(
        expected_total=contract.total_metric_items,
        observed_total=observed_total,
        ordered_ids_match=True,
        identity_binding_match=True,
        suite_content_match=True,
        choice_scoring_sequences=observed_choice_sequences,
        report_sha256=_canonical_sha256(dict(report)),
    )


__all__ = [
    "DISJOINTNESS_SCHEMA",
    "EVALUATION_IDENTITY_SCHEMA",
    "EVALUATION_PROVENANCE_SCHEMA",
    "ChoiceScoringSequenceProvenance",
    "EvaluationCountVerification",
    "EvaluationIdentityBinding",
    "EvaluationPathDisjointnessReport",
    "HFEvaluationCountError",
    "HFEvaluationPathOverlapError",
    "HFEvaluationProvenanceError",
    "LoadedTextEvaluationProvenance",
    "NamedHash",
    "ScoredItemCountContract",
    "ScoredMetricItemProvenance",
    "TokenPathDomain",
    "TokenizerCallProvenance",
    "assert_evaluation_report_count_contract",
    "assert_strict_evaluation_path_disjointness",
    "bind_loaded_text_evaluation_provenance",
    "bind_prompt_path_domain",
    "evaluation_identity_binding",
    "token_sequence_sha256",
]
