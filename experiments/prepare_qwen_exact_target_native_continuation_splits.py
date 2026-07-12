#!/usr/bin/env python3
"""Prepare deterministic continuation splits from the original exact contract.

This builder is deliberately model-free.  It authenticates the original exact
compiler's 24/12/16 fit/candidate/sealed artifacts, reads the existing pinned
2,906-row corpus, and uses only the pinned tokenizer to produce:

* fit: the original 24 prompts replayed first, then 72 fresh corpus prompts;
* candidate: 24 fresh prompts disjoint from every original and selected prompt;
* sealed: the original 16 prompts unchanged first, then 16 fresh prompts; and
* old candidate: a structurally non-runnable diagnostic-only artifact.

Fresh prompts are selected in stable source-content-hash order.  Exact prompt
IDs and every token prefix of at least eight tokens are excluded globally, so
the prior candidate and sealed domains cannot leak into continuation fitting.
All artifacts use canonical LF JSON and content/file SHA-256 manifests.  No
model class, checkpoint, weight tensor, forward pass, or runner is accessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import operator
import os
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SPLIT_SCHEMA = "neural-forge-target-native-behavior-split-v1"
CONTINUATION_SELECTION_SCHEMA = (
    "neural-forge-exact-target-native-continuation-selection-v1"
)
CONTINUATION_MANIFEST_SCHEMA = (
    "neural-forge-exact-target-native-continuation-manifest-v1"
)
OLD_CANDIDATE_DIAGNOSTIC_SCHEMA = (
    "neural-forge-exact-target-native-old-candidate-diagnostic-v1"
)
CONTINUATION_BUILD_SCHEMA = "neural-forge-exact-target-native-continuation-build-v1"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINAL_SPLIT_DIR = Path(
    "/mnt/ai_data/neural-forge/prereg/qwen_exact_target_native_stage1_20260712"
)
DEFAULT_CORPUS = (
    REPO_ROOT
    / "runs"
    / "qwen4b_to_smollm135m_transport_random_20260627"
    / "calibration_mix_plus_recovery_drills_20260702.jsonl"
)
DEFAULT_TOKENIZER = Path(
    "/home/cereal/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-4B-Base/snapshots/"
    "906bfd4b4dc7f14ee4320094d8b41684abff8539"
)
DEFAULT_OUTPUT = Path(
    "/mnt/ai_data/neural-forge/prereg/qwen_exact_target_native_stage1_continuation_v1"
)

PINNED_CORPUS_SHA256 = (
    "ba2dd1202752e2dd2b9b981c52501060ee367e220e5fb73ce7f8c320e2bbdd90"
)
PINNED_CORPUS_LINE_COUNT = 2_906
PINNED_TOKENIZER_JSON_SHA256 = (
    "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506"
)
PINNED_ORIGINAL_SPLIT_SHA256S = {
    "fit": "04bf059c78f91d9edca6915b5fb6b352442f598b8f79b34bb9f9ba4eac86063a",
    "candidate": ("723549d5c5fdf23a25d68c5cdcd9fb28b3c00daa1fc2c1facdf71f599b7e699d"),
    "sealed": "8927bbe951aab8b05721b68af41703cc61f5e6d7f58845ee3cffef74cfb76414",
}

ORIGINAL_COUNTS = {"fit": 24, "candidate": 12, "sealed": 16}
FRESH_COUNTS = {"fit": 72, "candidate": 24, "sealed": 16}
OUTPUT_COUNTS = {"fit": 96, "candidate": 24, "sealed": 32}
MINIMUM_MEANINGFUL_PREFIX_TOKENS = 8
MAX_PROMPT_TOKENS = 128
_TOKEN_SEQUENCE_HASH_NAMESPACE = b"neural-forge-token-sequence-v1\0"
_SHA256_HEX = frozenset("0123456789abcdef")


class ContinuationSplitPreparationError(RuntimeError):
    """The model-free continuation split contract cannot be authenticated."""


def _sha256(value: Any, name: str) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(character not in _SHA256_HEX for character in result):
        raise ContinuationSplitPreparationError(f"{name} must be a SHA-256.")
    return result


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ContinuationSplitPreparationError(
            "Continuation provenance must be finite canonical JSON."
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _strict_json(path: Path) -> Mapping[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContinuationSplitPreparationError(
                    f"Duplicate JSON key {key!r} in {path.name}."
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContinuationSplitPreparationError(
            f"Could not read strict JSON from {path}."
        ) from error
    if not isinstance(value, Mapping):
        raise ContinuationSplitPreparationError(
            f"{path.name} must contain a JSON object."
        )
    return value


def _token_ids(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ContinuationSplitPreparationError(
            f"{name} must be a nonempty token sequence."
        )
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise ContinuationSplitPreparationError(
                f"{name}[{index}] must be a nonnegative integer."
            )
        try:
            token = operator.index(item)
        except TypeError as error:
            raise ContinuationSplitPreparationError(
                f"{name}[{index}] must be a nonnegative integer."
            ) from error
        if token < 0 or token > 2**63 - 1:
            raise ContinuationSplitPreparationError(
                f"{name}[{index}] must fit a nonnegative signed int64."
            )
        result.append(int(token))
    if not result:
        raise ContinuationSplitPreparationError(f"{name} cannot be empty.")
    return tuple(result)


def _token_sequence_sha256(token_ids: Sequence[int]) -> str:
    tokens = _token_ids(token_ids, "token_ids")
    digest = hashlib.sha256(_TOKEN_SEQUENCE_HASH_NAMESPACE)
    for token in tokens:
        digest.update(struct.pack("<q", token))
    return digest.hexdigest()


def _meaningful_prefixes(tokens: tuple[int, ...]) -> frozenset[tuple[int, ...]]:
    return frozenset(
        tokens[:length]
        for length in range(MINIMUM_MEANINGFUL_PREFIX_TOKENS, len(tokens) + 1)
    )


@dataclass(frozen=True, slots=True)
class _PromptRecord:
    payload: Mapping[str, Any]
    token_ids: tuple[int, ...]
    prompt_id: str
    prefixes: frozenset[tuple[int, ...]]


def _prompt_record(value: Any, name: str) -> _PromptRecord:
    if not isinstance(value, Mapping):
        raise ContinuationSplitPreparationError(f"{name} must be an object.")
    tokens = _token_ids(value.get("token_ids", value.get("input_ids")), name)
    digest = _token_sequence_sha256(tokens)
    prompt_id = f"prompt-{digest}"
    if value.get("prompt_id") != prompt_id:
        raise ContinuationSplitPreparationError(
            f"{name}.prompt_id does not match its token content."
        )
    if value.get("token_ids_sha256") != digest:
        raise ContinuationSplitPreparationError(
            f"{name}.token_ids_sha256 does not match its token content."
        )
    return _PromptRecord(
        payload=dict(value),
        token_ids=tokens,
        prompt_id=prompt_id,
        prefixes=_meaningful_prefixes(tokens),
    )


@dataclass(frozen=True, slots=True)
class _OriginalSplit:
    name: str
    artifact_sha256: str
    payload: Mapping[str, Any]
    prompts: tuple[_PromptRecord, ...]


def _load_original_split(
    split_dir: Path,
    name: str,
    *,
    expected_sha256: str,
) -> _OriginalSplit:
    path = split_dir / f"{name}.json"
    expected = _sha256(expected_sha256, f"original_split_sha256s.{name}")
    if not path.is_file() or _file_sha256(path) != expected:
        raise ContinuationSplitPreparationError(
            f"Original exact {name} split hash changed."
        )
    payload = _strict_json(path)
    if payload.get("schema") != SPLIT_SCHEMA or payload.get("split") != name:
        raise ContinuationSplitPreparationError(
            f"Original exact {name} split schema/name mismatch."
        )
    if name == "sealed" and payload.get("sealed_before_model_access") is not True:
        raise ContinuationSplitPreparationError(
            "Original sealed split lacks its pre-model seal."
        )
    raw_prompts = payload.get("prompts")
    if not isinstance(raw_prompts, list) or len(raw_prompts) != ORIGINAL_COUNTS[name]:
        raise ContinuationSplitPreparationError(
            f"Original exact {name} split count changed."
        )
    prompts = tuple(
        _prompt_record(value, f"original.{name}.prompts[{index}]")
        for index, value in enumerate(raw_prompts)
    )
    if len({value.prompt_id for value in prompts}) != len(prompts):
        raise ContinuationSplitPreparationError(
            f"Original exact {name} split contains duplicate prompts."
        )
    return _OriginalSplit(
        name=name,
        artifact_sha256=expected,
        payload=payload,
        prompts=prompts,
    )


def _assert_globally_disjoint(
    domains: Mapping[str, Sequence[_PromptRecord]],
    *,
    label: str,
) -> None:
    prompt_owners: dict[str, str] = {}
    prefix_owners: dict[tuple[int, ...], str] = {}
    for role, prompts in domains.items():
        for prompt in prompts:
            previous = prompt_owners.get(prompt.prompt_id)
            if previous is not None:
                raise ContinuationSplitPreparationError(
                    f"{label} prompt overlap between {previous} and {role}."
                )
            prompt_owners[prompt.prompt_id] = role
            for prefix in prompt.prefixes:
                previous = prefix_owners.get(prefix)
                if previous is not None:
                    raise ContinuationSplitPreparationError(
                        f"{label} meaningful prefix overlap between {previous} and {role}."
                    )
                prefix_owners[prefix] = role


def _messages_to_prompt(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return None
    pieces: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        role = message.get("role")
        prefix = (
            f"{str(role).strip()}: " if isinstance(role, str) and role.strip() else ""
        )
        pieces.append(prefix + content.strip())
    return "\n".join(pieces) or None


def _row_prompt(item: Any) -> str | None:
    if isinstance(item, str):
        return item.strip() or None
    if not isinstance(item, Mapping):
        return None
    messages = _messages_to_prompt(item.get("messages"))
    if messages:
        return messages
    input_json = item.get("input_json")
    if isinstance(input_json, Mapping):
        messages = _messages_to_prompt(input_json.get("messages"))
        if messages:
            return messages
    for key in ("text", "prompt", "content"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _load_corpus(
    path: Path,
    *,
    expected_sha256: str,
    expected_line_count: int,
) -> tuple[dict[str, Any], ...]:
    expected = _sha256(expected_sha256, "corpus_sha256")
    if not path.is_file() or _file_sha256(path) != expected:
        raise ContinuationSplitPreparationError("Pinned canonical corpus hash changed.")
    if (
        isinstance(expected_line_count, bool)
        or not isinstance(expected_line_count, int)
        or expected_line_count <= 0
    ):
        raise ContinuationSplitPreparationError(
            "corpus_line_count must be a positive integer."
        )
    records: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    nonempty_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            nonempty_lines += 1
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ContinuationSplitPreparationError(
                    f"Invalid canonical JSONL at line {line_number}."
                ) from error
            text = _row_prompt(item)
            if text is None:
                raise ContinuationSplitPreparationError(
                    f"Canonical corpus row {line_number} has no prompt text."
                )
            if text in seen_texts:
                continue
            seen_texts.add(text)
            records.append(
                {
                    "source_line_number": line_number,
                    "source_prompt_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                    "text": text,
                }
            )
    if nonempty_lines != expected_line_count:
        raise ContinuationSplitPreparationError(
            f"Canonical corpus has {nonempty_lines}, not {expected_line_count}, rows."
        )
    return tuple(
        sorted(
            records,
            key=lambda value: (
                value["source_prompt_sha256"],
                value["source_line_number"],
            ),
        )
    )


def _load_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - dependency boundary
        raise RuntimeError(
            "Transformers is required to tokenize fresh prompts."
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ContinuationSplitPreparationError(
                "Pinned tokenizer has no pad/eos token."
            )
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _tokenize(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    if not isinstance(encoded, Mapping):
        raise ContinuationSplitPreparationError(
            "Pinned tokenizer returned a non-mapping result."
        )
    return _token_ids(encoded.get("input_ids"), "tokenizer.input_ids")


def _fresh_prompt_payload(
    record: Mapping[str, Any],
    tokens: tuple[int, ...],
) -> dict[str, Any]:
    digest = _token_sequence_sha256(tokens)
    return {
        "token_ids": list(tokens),
        "prompt_id": f"prompt-{digest}",
        "token_ids_sha256": digest,
        "source_line_number": int(record["source_line_number"]),
        "source_prompt_sha256": str(record["source_prompt_sha256"]),
    }


def _select_fresh_domains(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    originals: Mapping[str, _OriginalSplit],
    max_prompt_tokens: int,
) -> tuple[dict[str, list[_PromptRecord]], dict[str, Any]]:
    if (
        isinstance(max_prompt_tokens, bool)
        or not isinstance(max_prompt_tokens, int)
        or max_prompt_tokens < MINIMUM_MEANINGFUL_PREFIX_TOKENS
    ):
        raise ContinuationSplitPreparationError(
            "max_prompt_tokens must be an integer of at least eight."
        )
    used_prompt_ids = {
        prompt.prompt_id for split in originals.values() for prompt in split.prompts
    }
    used_prefixes = {
        prefix
        for split in originals.values()
        for prompt in split.prompts
        for prefix in prompt.prefixes
    }
    selected: dict[str, list[_PromptRecord]] = {
        "fit": [],
        "candidate": [],
        "sealed": [],
    }
    roles = ("fit", "candidate", "sealed")
    role_index = 0
    stats: dict[str, Any] = {
        "records_considered": 0,
        "skipped_over_token_limit": 0,
        "skipped_exact_prompt_collision": 0,
        "skipped_meaningful_prefix_collision": 0,
        "accepted_by_role": {name: 0 for name in roles},
        "completion_record_ordinal_by_role": {},
    }
    for record in records:
        if role_index == len(roles):
            break
        stats["records_considered"] += 1
        tokens = _tokenize(tokenizer, str(record["text"]))
        if len(tokens) > max_prompt_tokens:
            stats["skipped_over_token_limit"] += 1
            continue
        digest = _token_sequence_sha256(tokens)
        prompt_id = f"prompt-{digest}"
        prefixes = _meaningful_prefixes(tokens)
        if prompt_id in used_prompt_ids:
            stats["skipped_exact_prompt_collision"] += 1
            continue
        if prefixes & used_prefixes:
            stats["skipped_meaningful_prefix_collision"] += 1
            continue
        role = roles[role_index]
        prompt = _PromptRecord(
            payload=_fresh_prompt_payload(record, tokens),
            token_ids=tokens,
            prompt_id=prompt_id,
            prefixes=prefixes,
        )
        selected[role].append(prompt)
        used_prompt_ids.add(prompt_id)
        used_prefixes.update(prefixes)
        stats["accepted_by_role"][role] += 1
        if len(selected[role]) == FRESH_COUNTS[role]:
            stats["completion_record_ordinal_by_role"][role] = stats[
                "records_considered"
            ]
            role_index += 1
    actual = {name: len(values) for name, values in selected.items()}
    if actual != FRESH_COUNTS:
        raise ContinuationSplitPreparationError(
            f"Could not select the required fresh continuation prompts: {actual!r}."
        )
    return selected, stats


def _component_sha256(prompts: Sequence[Mapping[str, Any]]) -> str:
    return _canonical_sha256([dict(value) for value in prompts])


def _content_bound(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = _canonical_sha256(result)
    return result


def _split_payload(
    role: str,
    prompts: Sequence[Mapping[str, Any]],
    *,
    original_hashes: Mapping[str, str],
    corpus_sha256: str,
    corpus_line_count: int,
    tokenizer_sha256: str,
    components: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema": SPLIT_SCHEMA,
        "split": role,
        "sealed_before_model_access": role == "sealed",
        "prompts": [dict(value) for value in prompts],
        "source": {
            "corpus_sha256": corpus_sha256,
            "corpus_line_count": corpus_line_count,
            "tokenizer_json_sha256": tokenizer_sha256,
            "original_exact_split_sha256s": dict(original_hashes),
        },
        "selection": {
            "schema": CONTINUATION_SELECTION_SCHEMA,
            "global_order": "source_prompt_sha256, then source_line_number",
            "components": [dict(value) for value in components],
            "cross_domain_minimum_prefix_tokens": (MINIMUM_MEANINGFUL_PREFIX_TOKENS),
            "fresh_prompts_exclude_all_original_fit_candidate_sealed": True,
            "fresh_prompts_exclude_all_previously_selected_fresh_prompts": True,
            "old_candidate_usage": "diagnostic-only",
            "exact_text_deduplicated": True,
            "colliding_rows_skipped": True,
            "prompt_text_modified": False,
            "synthetic_replicas": 0,
        },
        "raw_prompt_text_retained": False,
        "model_accessed": False,
    }
    return _content_bound(payload, "split_content_sha256")


def prepare_continuation_splits(
    *,
    output_dir: Path,
    original_split_dir: Path = DEFAULT_ORIGINAL_SPLIT_DIR,
    corpus_path: Path = DEFAULT_CORPUS,
    tokenizer_path: Path = DEFAULT_TOKENIZER,
    original_split_sha256s: Mapping[str, str] | None = None,
    corpus_sha256: str = PINNED_CORPUS_SHA256,
    corpus_line_count: int = PINNED_CORPUS_LINE_COUNT,
    tokenizer_json_sha256: str = PINNED_TOKENIZER_JSON_SHA256,
    tokenizer_loader: Callable[[Path], Any] = _load_tokenizer,
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
) -> dict[str, Any]:
    """Write one path-independent, deterministic continuation split package."""

    output = output_dir.expanduser().resolve()
    if output.exists():
        raise ContinuationSplitPreparationError("output_dir must not already exist.")
    split_dir = original_split_dir.expanduser().resolve()
    corpus = corpus_path.expanduser().resolve()
    tokenizer_root = tokenizer_path.expanduser().resolve()
    expected_original = dict(
        PINNED_ORIGINAL_SPLIT_SHA256S
        if original_split_sha256s is None
        else original_split_sha256s
    )
    if set(expected_original) != {"fit", "candidate", "sealed"}:
        raise ContinuationSplitPreparationError(
            "original_split_sha256s must contain fit, candidate, and sealed."
        )
    originals = {
        name: _load_original_split(
            split_dir,
            name,
            expected_sha256=expected_original[name],
        )
        for name in ("fit", "candidate", "sealed")
    }
    _assert_globally_disjoint(
        {name: value.prompts for name, value in originals.items()},
        label="Original exact compiler contract",
    )
    corpus_digest = _sha256(corpus_sha256, "corpus_sha256")
    tokenizer_digest = _sha256(tokenizer_json_sha256, "tokenizer_json_sha256")
    tokenizer_json = tokenizer_root / "tokenizer.json"
    if not tokenizer_json.is_file() or _file_sha256(tokenizer_json) != tokenizer_digest:
        raise ContinuationSplitPreparationError("Pinned tokenizer JSON hash changed.")
    records = _load_corpus(
        corpus,
        expected_sha256=corpus_digest,
        expected_line_count=corpus_line_count,
    )
    tokenizer = tokenizer_loader(tokenizer_root)
    fresh, selection_stats = _select_fresh_domains(
        records,
        tokenizer,
        originals=originals,
        max_prompt_tokens=max_prompt_tokens,
    )

    output_records = {
        "fit": (*originals["fit"].prompts, *fresh["fit"]),
        "candidate": tuple(fresh["candidate"]),
        "sealed": (*originals["sealed"].prompts, *fresh["sealed"]),
    }
    if {name: len(values) for name, values in output_records.items()} != OUTPUT_COUNTS:
        raise ContinuationSplitPreparationError(
            "Continuation output counts do not match the sealed contract."
        )
    _assert_globally_disjoint(
        output_records,
        label="Continuation output contract",
    )
    diagnostic_and_outputs = {
        "old-candidate-diagnostic": originals["candidate"].prompts,
        **output_records,
    }
    _assert_globally_disjoint(
        diagnostic_and_outputs,
        label="Old candidate diagnostic isolation",
    )

    old_fit_payloads = [dict(value.payload) for value in originals["fit"].prompts]
    old_candidate_payloads = [
        dict(value.payload) for value in originals["candidate"].prompts
    ]
    old_sealed_payloads = [dict(value.payload) for value in originals["sealed"].prompts]
    fresh_payloads = {
        name: [dict(value.payload) for value in values]
        for name, values in fresh.items()
    }
    component_hashes = {
        "original_fit_replay": _component_sha256(old_fit_payloads),
        "fresh_fit": _component_sha256(fresh_payloads["fit"]),
        "old_candidate_diagnostic": _component_sha256(old_candidate_payloads),
        "fresh_candidate": _component_sha256(fresh_payloads["candidate"]),
        "original_sealed_untouched": _component_sha256(old_sealed_payloads),
        "fresh_sealed": _component_sha256(fresh_payloads["sealed"]),
    }
    split_prompts = {
        "fit": [*old_fit_payloads, *fresh_payloads["fit"]],
        "candidate": fresh_payloads["candidate"],
        "sealed": [*old_sealed_payloads, *fresh_payloads["sealed"]],
    }
    split_components = {
        "fit": (
            {
                "name": "original-fit-replay",
                "count": ORIGINAL_COUNTS["fit"],
                "prompt_manifest_sha256": component_hashes["original_fit_replay"],
            },
            {
                "name": "fresh-corpus-fit",
                "count": FRESH_COUNTS["fit"],
                "prompt_manifest_sha256": component_hashes["fresh_fit"],
            },
        ),
        "candidate": (
            {
                "name": "fresh-corpus-candidate",
                "count": FRESH_COUNTS["candidate"],
                "prompt_manifest_sha256": component_hashes["fresh_candidate"],
            },
        ),
        "sealed": (
            {
                "name": "original-sealed-untouched",
                "count": ORIGINAL_COUNTS["sealed"],
                "prompt_manifest_sha256": component_hashes["original_sealed_untouched"],
            },
            {
                "name": "fresh-corpus-sealed",
                "count": FRESH_COUNTS["sealed"],
                "prompt_manifest_sha256": component_hashes["fresh_sealed"],
            },
        ),
    }
    split_payloads = {
        role: _split_payload(
            role,
            split_prompts[role],
            original_hashes=expected_original,
            corpus_sha256=corpus_digest,
            corpus_line_count=corpus_line_count,
            tokenizer_sha256=tokenizer_digest,
            components=split_components[role],
        )
        for role in ("fit", "candidate", "sealed")
    }
    diagnostic_payload = _content_bound(
        {
            "schema": OLD_CANDIDATE_DIAGNOSTIC_SCHEMA,
            "role": "diagnostic-only",
            "source_split": "candidate",
            "source_artifact_sha256": expected_original["candidate"],
            "prompt_count": ORIGINAL_COUNTS["candidate"],
            "prompts": old_candidate_payloads,
            "prompt_manifest_sha256": component_hashes["old_candidate_diagnostic"],
            "eligible_for_fit": False,
            "eligible_for_candidate_selection": False,
            "eligible_for_sealed_selection": False,
            "eligible_for_checkpoint_promotion": False,
            "raw_prompt_text_retained": False,
            "model_accessed": False,
        },
        "diagnostic_content_sha256",
    )

    output.mkdir(parents=True)
    artifact_bindings: dict[str, dict[str, Any]] = {}
    for role in ("fit", "candidate", "sealed"):
        filename = f"{role}.json"
        path = output / filename
        _write_json_exclusive(path, split_payloads[role])
        artifact_bindings[role] = {
            "filename": filename,
            "sha256": _file_sha256(path),
            "prompt_count": OUTPUT_COUNTS[role],
            "split_content_sha256": split_payloads[role]["split_content_sha256"],
        }
    diagnostic_name = "old_candidate_diagnostic.json"
    diagnostic_path = output / diagnostic_name
    _write_json_exclusive(diagnostic_path, diagnostic_payload)
    artifact_bindings["old_candidate_diagnostic"] = {
        "filename": diagnostic_name,
        "sha256": _file_sha256(diagnostic_path),
        "prompt_count": ORIGINAL_COUNTS["candidate"],
        "diagnostic_content_sha256": diagnostic_payload["diagnostic_content_sha256"],
        "usage": "diagnostic-only",
    }

    manifest_payload = _content_bound(
        {
            "schema": CONTINUATION_MANIFEST_SCHEMA,
            "original_exact_compiler_split_sha256s": dict(expected_original),
            "corpus": {
                "sha256": corpus_digest,
                "line_count": corpus_line_count,
                "unique_text_count": len(records),
            },
            "tokenizer_json_sha256": tokenizer_digest,
            "builder_sha256": _file_sha256(Path(__file__).resolve()),
            "counts": {
                "original": dict(ORIGINAL_COUNTS),
                "fresh": dict(FRESH_COUNTS),
                "active_output": dict(OUTPUT_COUNTS),
                "old_candidate_diagnostic": ORIGINAL_COUNTS["candidate"],
            },
            "components": component_hashes,
            "artifacts": artifact_bindings,
            "selection": {
                "schema": CONTINUATION_SELECTION_SCHEMA,
                "global_order": "source_prompt_sha256, then source_line_number",
                "max_prompt_tokens": max_prompt_tokens,
                "minimum_meaningful_prefix_tokens": (MINIMUM_MEANINGFUL_PREFIX_TOKENS),
                "statistics": selection_stats,
            },
            "contract": {
                "behavior_fit": "original 24 replay + 72 fresh",
                "active_candidate": "24 fresh",
                "sealed": "original untouched 16 + 16 fresh",
                "old_candidate": "diagnostic-only",
                "fresh_excludes_every_original_prompt_and_prefix": True,
                "fresh_excludes_every_previously_selected_prompt_and_prefix": True,
                "cross_active_split_exact_prompt_overlap": 0,
                "cross_active_split_meaningful_prefix_overlap": 0,
                "old_candidate_overlap_with_active_outputs": 0,
                "original_fit_records_preserved_exactly": True,
                "original_sealed_records_preserved_exactly": True,
            },
            "model_access": {
                "model_class_imported": False,
                "model_created": False,
                "model_weights_loaded": False,
                "model_forward_calls": 0,
                "tokenizer_loaded": True,
            },
        },
        "manifest_content_sha256",
    )
    manifest_path = output / "continuation_manifest.json"
    _write_json_exclusive(manifest_path, manifest_payload)
    report = _content_bound(
        {
            "schema": CONTINUATION_BUILD_SCHEMA,
            "artifacts": artifact_bindings,
            "continuation_manifest": {
                "filename": manifest_path.name,
                "sha256": _file_sha256(manifest_path),
                "manifest_content_sha256": manifest_payload["manifest_content_sha256"],
            },
            "model_accessed": False,
            "model_class_imported": False,
            "model_created": False,
            "model_weights_loaded": False,
            "model_forward_calls": 0,
            "tokenizer_loaded": True,
            "training_launched": False,
            "checkpoint_created": False,
            "ready_for_preregistration_binding": True,
        },
        "build_content_sha256",
    )
    _write_json_exclusive(output / "build_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--original-split-dir", type=Path, default=DEFAULT_ORIGINAL_SPLIT_DIR
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-prompt-tokens", type=int, default=MAX_PROMPT_TOKENS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = prepare_continuation_splits(
        output_dir=args.output_dir,
        original_split_dir=args.original_split_dir,
        corpus_path=args.corpus,
        tokenizer_path=args.tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
