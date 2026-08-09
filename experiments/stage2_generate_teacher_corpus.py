#!/usr/bin/env python3
"""Generate the stage-2 teacher self-generation corpus, one sealed shard at a time.

The corpus is specified by ``docs/STAGE2_FIRST_MEASUREMENT_DESIGN_2026-08-08.md``
section 2.3: ancestral samples from the local ``LFM2.5-230M`` teacher at
temperature 1.0 and top-p 0.995, 1,024 donor tokens per sequence, half seeded
from a 64-token prefix drawn out of the program's own prompt estates and half
unconditional from BOS, with the frozen V5 review surface screened out of every
prefix and every generated sequence.

This module is data preparation only.  It loads the teacher for sampling, and it
never trains, fine-tunes, prunes, quantizes, or writes into any sealed stage-1
run directory.  Every shard is an independent unit sealed by its own manifest,
so a killed run resumes at the first missing shard and a completed shard is
never rewritten.  The top-level corpus manifest seals the whole inventory.

Subcommands
-----------
``estate``       Build the sealed prefix estate bundle from named local sources.
``generate``     Sample shards on one GPU, sealing each shard as it completes.
``seal``         Re-read every shard manifest and write the corpus manifest.
``self-test``    Run the CPU manifest/screen checks with no model and no GPU.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import random
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


PREFIX_ESTATE_SCHEMA = "neural-forge-stage2-prefix-estate-v1"
REVIEW_SCREEN_SCHEMA = "neural-forge-stage2-review-screen-v1"
SHARD_MANIFEST_SCHEMA = "neural-forge-stage2-corpus-shard-v1"
CORPUS_MANIFEST_SCHEMA = "neural-forge-stage2-corpus-manifest-v1"
PROGRESS_SCHEMA = "neural-forge-stage2-corpus-progress-v1"

# Design section 2.3: the donor tokenizer's defined rows only. The undefined
# physical rows must never be scored and must never appear in a stored sequence.
EXPECTED_DEFINED_TOKEN_IDS = 64402
EXPECTED_UNDEFINED_TOKEN_IDS = 1134
EXPECTED_PHYSICAL_VOCAB_ROWS = 65536

# Sealed provenance the corpus is required to bind to.  A run that cannot
# reproduce these refuses rather than quietly distilling from a different
# checkpoint or screening against a different review surface.
TEACHER_SHARD_SHA256 = (
    "f630da86651136c9aee893b04b7542007e90fdd718355358e57e7ecc31517cfd"
)
RESERVED_REVIEW_SOURCE_SHA256 = {
    "research/lfm2_semantic_whole_response_prompts_v1.json": (
        "0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b"
    ),
    "research/lfm2_causal_support_future_manual_review_v1.json": (
        "27b8f88b2e038c74e9f15b372eda0930e01ce24a0a142c2240125ff829c11559"
    ),
}
RESERVED_REVIEW_DIGEST_SET_SHA256 = (
    "b9d8ee479cf2f062f104e2a8d2b0b8ba245c8afc7772e30404f6c24fc26388d9"
)
CALIBRATION_DIGEST_SET_SHA256 = (
    "88ff81dd3a844102da4f7ab4a95ec7b9610fb77e8975fc18166626958278776d"
)
CALIBRATION_DIGEST_COUNT = 552

# A reserved prompt shorter than this is not used as a containment needle: the
# review surface's shortest message is 65 characters, so nothing is lost, and a
# short needle would drop ordinary teacher text for no contamination reason.
MINIMUM_CONTAINMENT_NEEDLE_CHARACTERS = 48


class Stage2TeacherCorpusError(ValueError):
    """A corpus input, screen, shard, or seal could not be authenticated."""


# --------------------------------------------------------------------------
# canonical bytes and atomic files
# --------------------------------------------------------------------------


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Stage2TeacherCorpusError(
            "Stage-2 corpus evidence must be finite canonical JSON."
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024**2):
                digest.update(chunk)
    except OSError as exc:
        raise Stage2TeacherCorpusError(f"Cannot hash {path}.") from exc
    return digest.hexdigest()


def _regular_file(path: Path, name: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise Stage2TeacherCorpusError(f"{name} does not exist: {path}.") from exc
    if not resolved.is_file():
        raise Stage2TeacherCorpusError(f"{name} must be a regular file: {path}.")
    return resolved


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Publish one file by rename so a killed run leaves no half shard."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            # The corpus lands on a CIFS mount that may refuse fsync; the
            # rename below is still the publication boundary.
            pass
    temporary.replace(path)


def _atomic_write_text(path: Path, value: str) -> None:
    _atomic_write_bytes(path, value.encode("utf-8"))


def _sealed_document(body: Mapping[str, Any], seal_key: str) -> dict[str, Any]:
    document = dict(body)
    document.pop(seal_key, None)
    document[seal_key] = _canonical_sha256(document)
    return document


def _verify_seal(document: Mapping[str, Any], seal_key: str, name: str) -> None:
    recorded = document.get(seal_key)
    body = {key: value for key, value in document.items() if key != seal_key}
    if recorded != _canonical_sha256(body):
        raise Stage2TeacherCorpusError(f"{name} seal does not cover its body.")


# --------------------------------------------------------------------------
# the review screen
# --------------------------------------------------------------------------


def _normalized_whitespace(value: str) -> str:
    """Collapse whitespace so a rewrapped copy of a review prompt still hits."""

    return " ".join(value.split())


def _reserved_renderings(path: Path) -> list[str]:
    """Every plain-text rendering of one reserved review-prompt document.

    This mirrors ``tests/test_lfm2_on_policy_fit_prompts.py`` exactly: the
    routed set is chat messages and the future set is flat text, so a sequence
    could collide with a single message or with a transcript of the whole
    exchange.  Both are compared.
    """

    document = json.loads(path.read_text(encoding="utf-8"))
    texts: list[str] = []
    for row in document["prompts"]:
        if "text" in row:
            texts.append(row["text"])
            continue
        messages = row["messages"]
        texts.extend(message["content"] for message in messages)
        texts.append("\n\n".join(m["content"] for m in messages))
        texts.append("\n".join(f"{m['role']}: {m['content']}" for m in messages))
        texts.append(
            "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)
        )
    return texts


def _digest_set(texts: Iterable[str]) -> set[str]:
    """Exact and whitespace-stripped digests, the round's comparison rule."""

    digests: set[str] = set()
    for value in texts:
        digests.add(_text_sha256(value))
        digests.add(_text_sha256(value.strip()))
    return digests


def _calibration_texts(
    calibration_path: Path, parent_outputs_path: Path
) -> list[str]:
    """The 540 texts behind the fit pack's 552-digest disjointness set."""

    texts: list[str] = []
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    for row in calibration:
        texts.extend(row.get("target_input_texts") or ())
        texts.extend(row.get("target_forced_decode_texts") or ())
    outputs = json.loads(parent_outputs_path.read_text(encoding="utf-8"))

    def collect(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                collect(item)
        elif isinstance(node, Mapping):
            prompt = node.get("prompt")
            if isinstance(prompt, str):
                texts.append(prompt)
            for value in node.values():
                collect(value)

    collect(outputs)
    return texts


@dataclass(frozen=True)
class ReviewScreen:
    """The contamination screen applied to every prefix and every sequence."""

    reserved_digests: frozenset[str]
    reserved_needles: tuple[str, ...]
    reserved_needles_normalized: tuple[str, ...]
    calibration_digests: frozenset[str]
    provenance: Mapping[str, Any]

    def reserved_drop_reasons(self, text: str) -> tuple[str, ...]:
        """The frozen-review screen alone: digest match, then containment.

        This is the screen that applies to prefixes as well as to generated
        sequences.  Containment is the load-bearing half: an exact digest can
        only fire when a whole sequence *is* a review prompt, while a teacher
        sample that quotes one is caught by the substring pass.
        """

        reasons: list[str] = []
        stripped = text.strip()
        if (
            _text_sha256(text) in self.reserved_digests
            or _text_sha256(stripped) in self.reserved_digests
        ):
            reasons.append("reserved_review_digest")
        normalized = _normalized_whitespace(text)
        if any(needle in text for needle in self.reserved_needles) or any(
            needle in normalized for needle in self.reserved_needles_normalized
        ):
            reasons.append("reserved_review_containment")
        return tuple(reasons)

    def drop_reasons(self, text: str) -> tuple[str, ...]:
        """Every independent reason this generated sequence stays out.

        The calibration half is not applied to prefixes: the design draws
        prefixes out of that same bridge-folding corpus, so screening them
        against it would delete the prefix estate.
        """

        reasons = list(self.reserved_drop_reasons(text))
        stripped = text.strip()
        if (
            _text_sha256(text) in self.calibration_digests
            or _text_sha256(stripped) in self.calibration_digests
        ):
            reasons.append("calibration_corpus_digest")
        return tuple(reasons)


def build_review_screen(
    *,
    routed_review_path: Path,
    future_review_path: Path,
    calibration_path: Path | None,
    parent_outputs_path: Path | None,
    verify_sealed_digests: bool = True,
) -> ReviewScreen:
    routed = _regular_file(routed_review_path, "routed review prompt pack")
    future = _regular_file(future_review_path, "future review prompt pack")
    sources = {
        "research/lfm2_semantic_whole_response_prompts_v1.json": _file_sha256(routed),
        "research/lfm2_causal_support_future_manual_review_v1.json": _file_sha256(
            future
        ),
    }
    if verify_sealed_digests and sources != RESERVED_REVIEW_SOURCE_SHA256:
        raise Stage2TeacherCorpusError(
            "Reserved review prompt packs do not match the sealed digests the "
            "fit-prompt pack recorded; the screen would not cover the frozen "
            "review surface."
        )

    renderings = _reserved_renderings(routed) + _reserved_renderings(future)
    reserved_digests = _digest_set(renderings)
    reserved_digest_set_sha256 = _canonical_sha256(sorted(reserved_digests))
    if (
        verify_sealed_digests
        and reserved_digest_set_sha256 != RESERVED_REVIEW_DIGEST_SET_SHA256
    ):
        raise Stage2TeacherCorpusError(
            "Reserved review digest set does not reproduce the sealed set sha."
        )

    needles = sorted(
        {
            value
            for value in renderings
            if len(value) >= MINIMUM_CONTAINMENT_NEEDLE_CHARACTERS
        }
    )
    normalized_needles = sorted(
        {
            _normalized_whitespace(value)
            for value in needles
            if len(_normalized_whitespace(value))
            >= MINIMUM_CONTAINMENT_NEEDLE_CHARACTERS
        }
    )

    calibration_digests: set[str] = set()
    calibration_provenance: dict[str, Any] = {
        "applied_to_generated_sequences": False,
        "applied_to_prefixes": False,
        "not_applied_to_prefixes_because": (
            "the design draws prefixes from this same bridge-folding corpus, so "
            "screening prefixes against it would delete the prefix estate"
        ),
        "available": False,
    }
    if calibration_path is not None and parent_outputs_path is not None:
        calibration = _regular_file(calibration_path, "bridge folding corpus")
        parent_outputs = _regular_file(parent_outputs_path, "parent outputs pack")
        calibration_digests = _digest_set(
            _calibration_texts(calibration, parent_outputs)
        )
        calibration_set_sha256 = _canonical_sha256(sorted(calibration_digests))
        if (
            verify_sealed_digests
            and calibration_set_sha256 != CALIBRATION_DIGEST_SET_SHA256
        ):
            raise Stage2TeacherCorpusError(
                "Calibration disjointness digest set does not reproduce the "
                "sealed 552-digest set sha the fit-prompt pack recorded."
            )
        calibration_provenance = {
            "applied_to_generated_sequences": True,
            "applied_to_prefixes": False,
            "not_applied_to_prefixes_because": (
                "the design draws prefixes from this same bridge-folding "
                "corpus, so screening prefixes against it would delete the "
                "prefix estate"
            ),
            "available": True,
            "digest_count": len(calibration_digests),
            "digest_set_sha256": calibration_set_sha256,
            "sources": {
                calibration.name: _file_sha256(calibration),
                parent_outputs.name: _file_sha256(parent_outputs),
            },
        }

    provenance = {
        "schema": REVIEW_SCREEN_SCHEMA,
        "calibration_corpus_screen": calibration_provenance,
        "comparison": "utf8-sha256-exact-and-stripped-plus-substring-containment",
        "containment_minimum_characters": MINIMUM_CONTAINMENT_NEEDLE_CHARACTERS,
        "reserved_review_screen": {
            "digest_count": len(reserved_digests),
            "digest_set_sha256": reserved_digest_set_sha256,
            "needle_count": len(needles),
            "normalized_needle_count": len(normalized_needles),
            "rendering_count": len(renderings),
            "sources": sources,
        },
    }
    return ReviewScreen(
        reserved_digests=frozenset(reserved_digests),
        reserved_needles=tuple(needles),
        reserved_needles_normalized=tuple(normalized_needles),
        calibration_digests=frozenset(calibration_digests),
        provenance=provenance,
    )


def review_probe_texts(routed_review_path: Path, count: int) -> tuple[str, ...]:
    """Synthetic contaminated candidates used to prove the screen is live.

    Each probe is ordinary filler with one real V5 review message spliced into
    the middle, which is exactly the shape the screen exists to catch.
    """

    document = json.loads(
        _regular_file(routed_review_path, "routed review prompt pack").read_text(
            encoding="utf-8"
        )
    )
    contents: list[str] = []
    for row in document["prompts"]:
        if "text" in row:
            contents.append(row["text"])
            continue
        contents.extend(message["content"] for message in row["messages"])
    if not contents:
        raise Stage2TeacherCorpusError("The routed review pack carries no prompts.")
    probes: list[str] = []
    for index in range(max(0, int(count))):
        content = contents[index % len(contents)]
        probes.append(
            "The following passage continues a longer document.\n"
            f"{content}\n"
            "The document then moves on to an unrelated topic."
        )
    return tuple(probes)


# --------------------------------------------------------------------------
# the prefix estate
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PrefixEstate:
    texts: tuple[str, ...]
    provenance: Mapping[str, Any]
    estate_sha256: str


def _repository_corpus_texts(
    evidence_path: Path, repository_root: Path
) -> tuple[list[str], dict[str, Any]]:
    """Text of the a7/a8 corpus files whose bytes still match sealed evidence.

    Files that changed since the sealed run are skipped and counted rather than
    silently substituted, so the estate is exactly the sealed corpus intersected
    with what still exists byte-identically.
    """

    evidence = json.loads(
        _regular_file(evidence_path, "corpus selection evidence").read_text(
            encoding="utf-8"
        )
    )
    rows = evidence.get("files")
    if not isinstance(rows, list) or not rows:
        raise Stage2TeacherCorpusError(
            f"Selection evidence {evidence_path} carries no file inventory."
        )
    texts: list[str] = []
    verified = mismatched = missing = undecodable = 0
    for row in rows:
        candidate = repository_root / str(row["path"])
        if not candidate.is_file():
            missing += 1
            continue
        if _file_sha256(candidate) != row["sha256"]:
            mismatched += 1
            continue
        try:
            texts.append(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            undecodable += 1
            continue
        verified += 1
    statistics = {
        "evidence_sha256": _file_sha256(evidence_path),
        "inventory_file_count": len(rows),
        "sha256_mismatched_file_count": mismatched,
        "missing_file_count": missing,
        "undecodable_file_count": undecodable,
        "verified_file_count": verified,
    }
    return texts, statistics


def build_prefix_estate(
    *,
    bridge_folding_corpus: Path | None,
    fit_prompt_pack: Path | None,
    repository_corpora: Sequence[tuple[str, Path, Path]],
    screen: ReviewScreen,
) -> PrefixEstate:
    """Assemble every prefix source into one screened, sealed estate."""

    texts: list[str] = []
    sources: list[dict[str, Any]] = []
    screened_out = 0

    def admit(name: str, candidates: Sequence[str], extra: Mapping[str, Any]) -> None:
        nonlocal screened_out
        admitted = 0
        rejected = 0
        for value in candidates:
            if not isinstance(value, str) or not value.strip():
                continue
            if screen.reserved_drop_reasons(value):
                rejected += 1
                continue
            texts.append(value)
            admitted += 1
        screened_out += rejected
        sources.append(
            {
                "name": name,
                "admitted_text_count": admitted,
                "screened_out_text_count": rejected,
                **dict(extra),
            }
        )

    if bridge_folding_corpus is not None:
        path = _regular_file(bridge_folding_corpus, "bridge folding corpus")
        rows = json.loads(path.read_text(encoding="utf-8"))
        candidates: list[str] = []
        for row in rows:
            candidates.extend(row.get("target_input_texts") or ())
            candidates.extend(row.get("target_forced_decode_texts") or ())
        admit(
            "bridge_folding_corpus",
            candidates,
            {"path": str(path), "sha256": _file_sha256(path)},
        )

    if fit_prompt_pack is not None:
        path = _regular_file(fit_prompt_pack, "fit prompt pack")
        document = json.loads(path.read_text(encoding="utf-8"))
        admit(
            "on_policy_fit_prompts",
            [row["prompt"] for row in document["prompts"]],
            {
                "corpus_sha256": document.get("corpus_sha256"),
                "path": str(path),
                "sha256": _file_sha256(path),
            },
        )

    for name, evidence_path, repository_root in repository_corpora:
        corpus_texts, statistics = _repository_corpus_texts(
            _regular_file(evidence_path, f"{name} selection evidence"),
            repository_root,
        )
        admit(name, corpus_texts, {"path": str(evidence_path), **statistics})

    if not texts:
        raise Stage2TeacherCorpusError("The prefix estate is empty.")

    provenance = {
        "schema": PREFIX_ESTATE_SCHEMA,
        "review_screen": dict(screen.provenance),
        "screened_out_text_count": screened_out,
        "sources": sources,
        "text_count": len(texts),
    }
    estate_sha256 = _canonical_sha256(
        {"provenance": provenance, "texts": list(texts)}
    )
    return PrefixEstate(
        texts=tuple(texts), provenance=provenance, estate_sha256=estate_sha256
    )


def write_prefix_estate(estate: PrefixEstate, path: Path) -> str:
    document = _sealed_document(
        {
            "estate_sha256": estate.estate_sha256,
            "provenance": dict(estate.provenance),
            "texts": list(estate.texts),
        },
        "bundle_sha256",
    )
    _atomic_write_bytes(path, _canonical_json(document))
    return document["bundle_sha256"]


def read_prefix_estate(path: Path) -> PrefixEstate:
    document = json.loads(
        _regular_file(path, "prefix estate bundle").read_text(encoding="utf-8")
    )
    _verify_seal(document, "bundle_sha256", "prefix estate bundle")
    estate = PrefixEstate(
        texts=tuple(document["texts"]),
        provenance=document["provenance"],
        estate_sha256=document["estate_sha256"],
    )
    recomputed = _canonical_sha256(
        {"provenance": dict(estate.provenance), "texts": list(estate.texts)}
    )
    if recomputed != estate.estate_sha256:
        raise Stage2TeacherCorpusError("Prefix estate digest does not cover its body.")
    return estate


# --------------------------------------------------------------------------
# curation
# --------------------------------------------------------------------------


def three_gram_repetition_fraction(ids: Sequence[int]) -> float:
    """Fraction of a sequence's 3-grams that are repeats of an earlier 3-gram.

    Design section 2.3 drops sequences above 0.5.  A sequence too short to hold
    a 3-gram has no measurable repetition and scores 0.0.
    """

    total = len(ids) - 2
    if total <= 0:
        return 0.0
    seen = {(ids[i], ids[i + 1], ids[i + 2]) for i in range(total)}
    return 1.0 - (len(seen) / total)


# --------------------------------------------------------------------------
# shard manifests
# --------------------------------------------------------------------------


def shard_paths(shard_root: Path, shard_index: int) -> dict[str, Path]:
    stem = f"shard-{int(shard_index):05d}"
    return {
        "tokens": shard_root / f"{stem}.tokens.u16",
        "text": shard_root / f"{stem}.text.jsonl",
        "manifest": shard_root / f"{stem}.manifest.json",
    }


def seal_shard_manifest(body: Mapping[str, Any]) -> dict[str, Any]:
    return _sealed_document(body, "manifest_sha256")


def load_shard_manifest(path: Path) -> dict[str, Any]:
    document = json.loads(
        _regular_file(path, "shard manifest").read_text(encoding="utf-8")
    )
    _verify_seal(document, "manifest_sha256", f"shard manifest {path.name}")
    if document.get("schema") != SHARD_MANIFEST_SCHEMA:
        raise Stage2TeacherCorpusError(
            f"Shard manifest {path.name} carries an unexpected schema."
        )
    return document


def shard_is_complete(
    shard_root: Path, shard_index: int, *, verify_digests: bool = False
) -> bool:
    """A shard counts as done only when its manifest seals files that exist.

    The manifest is written last, by rename, so its presence is the commit
    record and resume can trust it without re-hashing 8 MB per shard.
    """

    paths = shard_paths(shard_root, shard_index)
    if not paths["manifest"].is_file():
        return False
    try:
        manifest = load_shard_manifest(paths["manifest"])
    except (Stage2TeacherCorpusError, json.JSONDecodeError, OSError):
        return False
    for key in ("tokens", "text"):
        path = paths[key]
        if not path.is_file():
            return False
        if path.stat().st_size != manifest[f"{key}_bytes"]:
            return False
        if verify_digests and _file_sha256(path) != manifest[f"{key}_sha256"]:
            return False
    return True


# --------------------------------------------------------------------------
# the corpus manifest
# --------------------------------------------------------------------------


def build_corpus_manifest(
    *,
    shard_root: Path,
    run_provenance: Mapping[str, Any],
    verify_digests: bool = False,
) -> dict[str, Any]:
    manifests = sorted(shard_root.glob("shard-*.manifest.json"))
    if not manifests:
        raise Stage2TeacherCorpusError(f"No shard manifests under {shard_root}.")
    inventory: list[dict[str, Any]] = []
    totals = {"train": 0, "holdout": 0}
    rows_by_split = {"train": 0, "holdout": 0}
    aggregate: dict[str, int] = {}
    for path in manifests:
        manifest = load_shard_manifest(path)
        split = manifest["split"]
        if split not in totals:
            raise Stage2TeacherCorpusError(f"Unknown shard split {split!r}.")
        if verify_digests:
            paths = shard_paths(shard_root, manifest["shard_index"])
            for key in ("tokens", "text"):
                if _file_sha256(paths[key]) != manifest[f"{key}_sha256"]:
                    raise Stage2TeacherCorpusError(
                        f"{paths[key].name} does not match its sealed digest."
                    )
        totals[split] += manifest["token_count"]
        rows_by_split[split] += manifest["row_count"]
        for key, value in manifest["filter_statistics"].items():
            aggregate[key] = aggregate.get(key, 0) + int(value)
        inventory.append(
            {
                "manifest_sha256": manifest["manifest_sha256"],
                "row_count": manifest["row_count"],
                "shard_index": manifest["shard_index"],
                "shard_seed": manifest["shard_seed"],
                "split": split,
                "text_sha256": manifest["text_sha256"],
                "token_count": manifest["token_count"],
                "tokens_sha256": manifest["tokens_sha256"],
            }
        )
    indices = [row["shard_index"] for row in inventory]
    if indices != sorted(indices) or len(set(indices)) != len(indices):
        raise Stage2TeacherCorpusError("Shard indices are duplicated or unordered.")
    body = {
        "schema": CORPUS_MANIFEST_SCHEMA,
        "aggregate_filter_statistics": dict(sorted(aggregate.items())),
        "contiguous_shard_indices": indices == list(range(len(indices))),
        "holdout_row_count": rows_by_split["holdout"],
        "holdout_token_count": totals["holdout"],
        "run": dict(run_provenance),
        "sealed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "shard_count": len(inventory),
        "shards": inventory,
        "total_token_count": totals["train"] + totals["holdout"],
        "train_row_count": rows_by_split["train"],
        "train_token_count": totals["train"],
        "verified_shard_file_digests": bool(verify_digests),
    }
    return _sealed_document(body, "corpus_manifest_sha256")


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationConfigRecord:
    sequence_length: int
    prefix_tokens: int
    prefix_fraction: float
    temperature: float
    top_p: float
    batch_size: int
    repetition_drop_threshold: float
    dtype: str
    device_name: str
    bos_token_id: int
    pad_token_id: int
    defined_token_id_count: int
    undefined_token_id_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "ancestral_sampling": True,
            "batch_size": self.batch_size,
            "bos_token_id": self.bos_token_id,
            "continues_through_eos": True,
            "defined_token_id_count": self.defined_token_id_count,
            "device_name": self.device_name,
            "dtype": self.dtype,
            "pad_token_id": self.pad_token_id,
            "prefix_fraction": self.prefix_fraction,
            "prefix_tokens": self.prefix_tokens,
            "repetition_drop_threshold": self.repetition_drop_threshold,
            "repetition_penalty": 1.0,
            "sequence_length": self.sequence_length,
            "temperature": self.temperature,
            "top_k_disabled": True,
            "top_p": self.top_p,
            "undefined_token_id_count": self.undefined_token_id_count,
        }


class _PrefixSampler:
    """Draw 64-token windows from the estate without crossing a document edge."""

    def __init__(self, documents: Sequence[Any], prefix_tokens: int) -> None:
        self._documents: list[Any] = []
        self._weights: list[int] = []
        cumulative = 0
        self._cumulative: list[int] = []
        skipped = 0
        for index, tokens in enumerate(documents):
            windows = len(tokens) - prefix_tokens + 1
            if windows <= 0:
                skipped += 1
                continue
            self._documents.append((index, tokens))
            self._weights.append(windows)
            cumulative += windows
            self._cumulative.append(cumulative)
        if not self._documents:
            raise Stage2TeacherCorpusError(
                "No estate document is long enough to yield a prefix window."
            )
        self.total_windows = cumulative
        self.skipped_document_count = skipped
        self.prefix_tokens = prefix_tokens

    def draw(self, rng: random.Random) -> tuple[int, int, Any]:
        target = rng.randrange(self.total_windows)
        slot = bisect.bisect_right(self._cumulative, target)
        base = 0 if slot == 0 else self._cumulative[slot - 1]
        offset = target - base
        document_index, tokens = self._documents[slot]
        return document_index, offset, tokens[offset : offset + self.prefix_tokens]


def _load_teacher(model_path: Path, device: str, dtype_name: str):
    import torch  # noqa: PLC0415 - torch is only needed for generation
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    vocabulary = tokenizer.get_vocab()
    ids = sorted(vocabulary.values())
    if ids != list(range(len(ids))):
        raise Stage2TeacherCorpusError(
            "The donor tokenizer's defined ids are not contiguous from zero; "
            "the undefined-row mask assumes a contiguous defined prefix."
        )
    defined = len(ids)
    configuration = json.loads((model_path / "config.json").read_text("utf-8"))
    physical = int(configuration["vocab_size"])
    undefined = physical - defined
    if (
        defined != EXPECTED_DEFINED_TOKEN_IDS
        or undefined != EXPECTED_UNDEFINED_TOKEN_IDS
        or physical != EXPECTED_PHYSICAL_VOCAB_ROWS
    ):
        raise Stage2TeacherCorpusError(
            "Donor vocabulary does not match the sealed contract "
            f"({defined} defined / {undefined} undefined / {physical} rows)."
        )

    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    if dtype_name not in dtypes:
        raise Stage2TeacherCorpusError(f"Unsupported dtype {dtype_name!r}.")
    torch_dtype = dtypes[dtype_name]
    if torch_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise Stage2TeacherCorpusError(
            "The selected device has no usable bfloat16; pass --dtype float16."
        )
    model = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=torch_dtype)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return tokenizer, model, defined, undefined, physical


def _generation_config(
    *, new_tokens: int, temperature: float, top_p: float, pad_token_id: int
):
    from transformers import GenerationConfig  # noqa: PLC0415

    # Built from scratch, never inherited: the shipped generation_config.json
    # carries temperature 0.1, top_k 50, and repetition_penalty 1.05, none of
    # which is the design's sampler.
    return GenerationConfig(
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=0,
        repetition_penalty=1.0,
        min_new_tokens=int(new_tokens),
        max_new_tokens=int(new_tokens),
        eos_token_id=None,
        pad_token_id=int(pad_token_id),
        use_cache=True,
        num_beams=1,
        renormalize_logits=False,
    )


def _undefined_row_processor(defined: int, physical: int):
    import torch  # noqa: PLC0415
    from transformers import LogitsProcessor  # noqa: PLC0415

    class _MaskUndefinedRows(LogitsProcessor):
        """The undefined physical rows must never be scored (goal doc)."""

        def __call__(self, input_ids, scores):
            if scores.shape[-1] > defined:
                scores[..., defined:physical] = torch.finfo(scores.dtype).min
            return scores

    from transformers import LogitsProcessorList  # noqa: PLC0415

    return LogitsProcessorList([_MaskUndefinedRows()])


def generate_shard(
    *,
    shard_index: int,
    split: str,
    rows_per_shard: int,
    shard_root: Path,
    tokenizer,
    model,
    prefix_sampler: _PrefixSampler,
    screen: ReviewScreen,
    config: GenerationConfigRecord,
    probes: Sequence[str],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Sample, screen, and seal exactly one shard."""

    import numpy  # noqa: PLC0415
    import torch  # noqa: PLC0415

    started = time.time()
    shard_seed = int(shard_index)
    torch.manual_seed(shard_seed)
    rng = random.Random(shard_seed)

    device = next(model.parameters()).device
    unconditional_config = _generation_config(
        new_tokens=config.sequence_length - 1,
        temperature=config.temperature,
        top_p=config.top_p,
        pad_token_id=config.pad_token_id,
    )
    seeded_config = _generation_config(
        new_tokens=config.sequence_length - 1 - config.prefix_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        pad_token_id=config.pad_token_id,
    )
    processors = _undefined_row_processor(
        config.defined_token_id_count,
        config.defined_token_id_count + config.undefined_token_id_count,
    )

    statistics = {
        "candidate_count": 0,
        "accepted_count": 0,
        "dropped_calibration_corpus_digest": 0,
        "dropped_repetition": 0,
        "dropped_reserved_review_containment": 0,
        "dropped_reserved_review_digest": 0,
        "dropped_total": 0,
        "dropped_undefined_token_id": 0,
        "review_probe_dropped": 0,
        "review_probe_injected": 0,
        "surplus_over_kind_quota": 0,
    }
    accepted_ids: list[Any] = []
    accepted_rows: list[dict[str, Any]] = []
    repetition_sum = 0.0

    # The design fixes the mix at 50% unconditional / 50% prefix-seeded. Without
    # a per-kind quota the shard's tail is filled by whichever kind is sampled
    # first, which skewed the first trial shard to 2,298/1,798. Quotas make the
    # mix exact, and skipping a filled kind keeps the cost the same.
    quota = {
        "prefix": rows_per_shard // 2,
        "bos": rows_per_shard - rows_per_shard // 2,
    }
    accepted_by_kind = {"bos": 0, "prefix": 0}

    half = max(1, config.batch_size // 2)
    iteration_cap = 8 * (rows_per_shard // max(1, half) + 1)
    iterations = 0
    while len(accepted_ids) < rows_per_shard:
        iterations += 1
        if iterations > iteration_cap:
            raise Stage2TeacherCorpusError(
                f"Shard {shard_index} did not fill in {iteration_cap} iterations; "
                "the curation filter is rejecting almost everything."
            )
        batches: list[tuple[str, Any, Any, list[dict[str, Any]]]] = []

        if accepted_by_kind["bos"] < quota["bos"]:
            unconditional = torch.full(
                (half, 1), config.bos_token_id, dtype=torch.long, device=device
            )
            batches.append(
                (
                    "bos",
                    unconditional,
                    unconditional_config,
                    [{"prefix_document_index": None, "prefix_offset": None}] * half,
                )
            )

        if accepted_by_kind["prefix"] < quota["prefix"]:
            seeded_rows: list[list[int]] = []
            seeded_meta: list[dict[str, Any]] = []
            for _ in range(half):
                document_index, offset, window = prefix_sampler.draw(rng)
                seeded_rows.append([config.bos_token_id, *(int(x) for x in window)])
                seeded_meta.append(
                    {"prefix_document_index": document_index, "prefix_offset": offset}
                )
            seeded = torch.tensor(seeded_rows, dtype=torch.long, device=device)
            batches.append(("prefix", seeded, seeded_config, seeded_meta))

        for kind, inputs, generation_config, metadata in batches:
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    generation_config=generation_config,
                    logits_processor=processors,
                )
            sequences = outputs.detach().to("cpu").numpy().astype(numpy.int64)
            if sequences.shape[1] != config.sequence_length:
                raise Stage2TeacherCorpusError(
                    f"Generated {sequences.shape[1]} tokens, expected "
                    f"{config.sequence_length}."
                )
            texts = tokenizer.batch_decode(
                sequences.tolist(), skip_special_tokens=False
            )
            for row_index in range(sequences.shape[0]):
                ids = sequences[row_index]
                text = texts[row_index]
                statistics["candidate_count"] += 1
                reasons: list[str] = []
                if int(ids.max()) >= config.defined_token_id_count:
                    reasons.append("undefined_token_id")
                repetition = three_gram_repetition_fraction(ids.tolist())
                if repetition > config.repetition_drop_threshold:
                    reasons.append("repetition")
                reasons.extend(screen.drop_reasons(text))
                if reasons:
                    statistics["dropped_total"] += 1
                    for reason in reasons:
                        key = f"dropped_{reason}"
                        statistics[key] = statistics.get(key, 0) + 1
                    continue
                if accepted_by_kind[kind] >= quota[kind]:
                    statistics["surplus_over_kind_quota"] += 1
                    continue
                accepted_by_kind[kind] += 1
                repetition_sum += repetition
                accepted_ids.append(ids.astype(numpy.uint16))
                accepted_rows.append(
                    {
                        "repetition_fraction": round(repetition, 6),
                        "row": len(accepted_ids) - 1,
                        "seed_kind": kind,
                        "text": text,
                        **metadata[row_index],
                    }
                )
                statistics["accepted_count"] += 1

        for probe in probes:
            statistics["review_probe_injected"] += 1
            statistics["candidate_count"] += 1
            if "reserved_review_containment" in screen.drop_reasons(probe):
                statistics["review_probe_dropped"] += 1
                statistics["dropped_total"] += 1
                statistics["dropped_reserved_review_containment"] += 1
                continue
            raise Stage2TeacherCorpusError(
                "A spliced V5 review prompt survived the contamination screen; "
                "the screen is not protecting the frozen review surface."
            )

    tokens = numpy.stack(accepted_ids[:rows_per_shard]).astype(numpy.uint16)
    rows = accepted_rows[:rows_per_shard]
    paths = shard_paths(shard_root, shard_index)
    token_bytes = tokens.tobytes(order="C")
    text_payload = (
        "\n".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True, allow_nan=False)
            for row in rows
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(paths["tokens"], token_bytes)
    _atomic_write_bytes(paths["text"], text_payload)

    elapsed = time.time() - started
    body = {
        "schema": SHARD_MANIFEST_SCHEMA,
        "accepted_mean_repetition_fraction": round(
            repetition_sum / max(1, len(rows)), 6
        ),
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(elapsed, 3),
        "filter_statistics": dict(sorted(statistics.items())),
        "generation": config.as_dict(),
        "row_count": int(tokens.shape[0]),
        "run": dict(provenance),
        "seed_kind_counts": dict(sorted(accepted_by_kind.items())),
        "seed_kind_quota": dict(sorted(quota.items())),
        "sequence_length": int(tokens.shape[1]),
        "shard_index": int(shard_index),
        "shard_seed": shard_seed,
        "split": split,
        "text_bytes": len(text_payload),
        "text_sha256": hashlib.sha256(text_payload).hexdigest(),
        "token_count": int(tokens.shape[0] * tokens.shape[1]),
        "token_dtype": "uint16-little-endian",
        "tokens_bytes": len(token_bytes),
        "tokens_per_second_accepted": round(
            (tokens.shape[0] * tokens.shape[1]) / max(elapsed, 1e-9), 2
        ),
        "tokens_per_second_sampled": round(
            (statistics["candidate_count"] - statistics["review_probe_injected"])
            * config.sequence_length
            / max(elapsed, 1e-9),
            2,
        ),
        "tokens_sha256": hashlib.sha256(token_bytes).hexdigest(),
        "torch_manual_seed": shard_seed,
        "prefix_rng_seed": shard_seed,
    }
    manifest = seal_shard_manifest(body)
    _atomic_write_bytes(paths["manifest"], _canonical_json(manifest))
    return manifest


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------


def _run_estate(args: argparse.Namespace) -> int:
    screen = build_review_screen(
        routed_review_path=args.routed_review,
        future_review_path=args.future_review,
        calibration_path=args.bridge_folding_corpus,
        parent_outputs_path=args.parent_outputs,
        verify_sealed_digests=not args.allow_unsealed_screen,
    )
    corpora: list[tuple[str, Path, Path]] = []
    for name, evidence, root in args.repository_corpus or ():
        corpora.append((name, Path(evidence), Path(root)))
    estate = build_prefix_estate(
        bridge_folding_corpus=args.bridge_folding_corpus,
        fit_prompt_pack=args.fit_prompt_pack,
        repository_corpora=corpora,
        screen=screen,
    )
    bundle_sha256 = write_prefix_estate(estate, args.output)
    print(
        json.dumps(
            {
                "bundle_sha256": bundle_sha256,
                "estate_sha256": estate.estate_sha256,
                "output": str(args.output),
                "screened_out_text_count": estate.provenance[
                    "screened_out_text_count"
                ],
                "sources": estate.provenance["sources"],
                "text_count": estate.provenance["text_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _write_progress(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path, json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    )


def _run_generate(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    shard_root = output_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.teacher).expanduser().resolve()
    teacher_shard = model_path / "model.safetensors"
    teacher_sha256 = _file_sha256(_regular_file(teacher_shard, "teacher shard"))
    if not args.allow_unpinned_teacher and teacher_sha256 != TEACHER_SHARD_SHA256:
        raise Stage2TeacherCorpusError(
            "Teacher checkpoint digest does not equal the stage-1 parent digest "
            "recorded in the sealed run receipts; distilling from a different "
            "checkpoint would void the comparison."
        )
    tokenizer_sha256 = _file_sha256(
        _regular_file(model_path / "tokenizer.json", "teacher tokenizer")
    )

    screen = build_review_screen(
        routed_review_path=args.routed_review,
        future_review_path=args.future_review,
        calibration_path=args.bridge_folding_corpus,
        parent_outputs_path=args.parent_outputs,
        verify_sealed_digests=not args.allow_unsealed_screen,
    )
    estate = read_prefix_estate(Path(args.prefix_estate))

    tokenizer, model, defined, undefined, physical = _load_teacher(
        model_path, args.device, args.dtype
    )

    import torch  # noqa: PLC0415

    if model.device.type == "cuda":
        device_name = torch.cuda.get_device_name(model.device)
    else:
        device_name = str(model.device)

    cache_path = output_root / "inputs" / "prefix_estate.tokens.npz"
    documents = _tokenized_estate(tokenizer, estate, cache_path, args.prefix_tokens)
    prefix_sampler = _PrefixSampler(documents, args.prefix_tokens)

    config = GenerationConfigRecord(
        sequence_length=args.sequence_length,
        prefix_tokens=args.prefix_tokens,
        prefix_fraction=0.5,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
        repetition_drop_threshold=args.repetition_drop_threshold,
        dtype=args.dtype,
        device_name=device_name,
        bos_token_id=int(tokenizer.bos_token_id),
        pad_token_id=int(tokenizer.pad_token_id),
        defined_token_id_count=defined,
        undefined_token_id_count=undefined,
    )
    provenance = {
        "design_document": "docs/STAGE2_FIRST_MEASUREMENT_DESIGN_2026-08-08.md",
        "design_document_sha256": args.design_sha256,
        "estate_sha256": estate.estate_sha256,
        "generator_sha256": _file_sha256(Path(__file__).resolve()),
        "prefix_estate_window_count": prefix_sampler.total_windows,
        "review_screen": dict(screen.provenance),
        "stage": "stage2",
        "teacher_path": str(model_path),
        "teacher_shard_sha256": teacher_sha256,
        "teacher_tokenizer_sha256": tokenizer_sha256,
        "physical_vocab_rows": physical,
    }

    probes = review_probe_texts(args.routed_review, args.inject_review_probes)
    total_shards = args.train_shards + args.holdout_shards
    progress_path = output_root / "progress.json"
    run_started = time.time()
    generated = 0
    generated_tokens = 0

    for shard_index in range(total_shards):
        # The held-out split takes the low indices so that raising
        # --train-shards later is purely additive: no already-sealed shard ever
        # changes split, and a reduced target is a strict prefix of the full one.
        split = "holdout" if shard_index < args.holdout_shards else "train"
        if args.resume and shard_is_complete(
            shard_root, shard_index, verify_digests=args.verify_resume
        ):
            continue
        manifest = generate_shard(
            shard_index=shard_index,
            split=split,
            rows_per_shard=args.rows_per_shard,
            shard_root=shard_root,
            tokenizer=tokenizer,
            model=model,
            prefix_sampler=prefix_sampler,
            screen=screen,
            config=config,
            probes=probes,
            provenance=provenance,
        )
        generated += 1
        generated_tokens += manifest["token_count"]
        elapsed = time.time() - run_started
        rate = generated_tokens / max(elapsed, 1e-9)
        remaining = total_shards - shard_index - 1
        _write_progress(
            progress_path,
            {
                "schema": PROGRESS_SCHEMA,
                "elapsed_seconds": round(elapsed, 1),
                "last_shard_index": shard_index,
                "last_shard_tokens_per_second": manifest[
                    "tokens_per_second_accepted"
                ],
                "measured_tokens_per_second": round(rate, 2),
                "projected_remaining_hours": round(
                    remaining
                    * args.rows_per_shard
                    * args.sequence_length
                    / max(rate, 1e-9)
                    / 3600.0,
                    2,
                ),
                "shards_generated_this_process": generated,
                "shards_total": total_shards,
                "tokens_generated_this_process": generated_tokens,
                "updated_at_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
            },
        )
        print(
            f"shard {shard_index:05d} {split} "
            f"tokens={manifest['token_count']} "
            f"tok/s={manifest['tokens_per_second_accepted']} "
            f"dropped={manifest['filter_statistics']['dropped_total']} "
            f"sha={manifest['tokens_sha256'][:16]}",
            flush=True,
        )
        if args.max_shards and generated >= args.max_shards:
            break

    if args.seal_on_completion and all(
        shard_is_complete(shard_root, index) for index in range(total_shards)
    ):
        manifest = build_corpus_manifest(
            shard_root=shard_root,
            run_provenance={**provenance, "generation": config.as_dict()},
            verify_digests=args.verify_seal,
        )
        _atomic_write_bytes(
            output_root / "corpus_manifest.json", _canonical_json(manifest)
        )
        print(
            f"corpus_manifest_sha256={manifest['corpus_manifest_sha256']}",
            flush=True,
        )
    return 0


def _tokenized_estate(tokenizer, estate: PrefixEstate, cache_path: Path, prefix_tokens: int):
    """Tokenize the estate once and cache it, keyed by the estate digest."""

    import numpy  # noqa: PLC0415

    sidecar = cache_path.with_suffix(".json")
    if cache_path.is_file() and sidecar.is_file():
        marker = json.loads(sidecar.read_text(encoding="utf-8"))
        if marker.get("estate_sha256") == estate.estate_sha256:
            payload = numpy.load(cache_path)
            flat = payload["tokens"]
            offsets = payload["offsets"]
            return [
                flat[offsets[index] : offsets[index + 1]]
                for index in range(len(offsets) - 1)
            ]

    documents: list[Any] = []
    for text in estate.texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        documents.append(numpy.asarray(ids, dtype=numpy.int32))
    offsets = numpy.zeros(len(documents) + 1, dtype=numpy.int64)
    for index, tokens in enumerate(documents):
        offsets[index + 1] = offsets[index] + len(tokens)
    flat = (
        numpy.concatenate(documents)
        if documents
        else numpy.zeros(0, dtype=numpy.int32)
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp.npz")
    numpy.savez(temporary, tokens=flat, offsets=offsets)
    temporary.replace(cache_path)
    _atomic_write_text(
        sidecar,
        json.dumps(
            {
                "estate_sha256": estate.estate_sha256,
                "document_count": len(documents),
                "prefix_tokens": int(prefix_tokens),
                "token_count": int(offsets[-1]),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return documents


def _run_seal(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    run_provenance = {}
    marker = output_root / "shards"
    manifests = sorted(marker.glob("shard-*.manifest.json"))
    if manifests:
        first = load_shard_manifest(manifests[0])
        run_provenance = {**first["run"], "generation": first["generation"]}
    manifest = build_corpus_manifest(
        shard_root=marker,
        run_provenance=run_provenance,
        verify_digests=args.verify_seal,
    )
    _atomic_write_bytes(
        output_root / "corpus_manifest.json", _canonical_json(manifest)
    )
    print(
        json.dumps(
            {
                "corpus_manifest_sha256": manifest["corpus_manifest_sha256"],
                "shard_count": manifest["shard_count"],
                "total_token_count": manifest["total_token_count"],
                "train_token_count": manifest["train_token_count"],
                "holdout_token_count": manifest["holdout_token_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_self_test(args: argparse.Namespace) -> int:
    """CPU-only checks of the screen and manifest logic; no model, no GPU."""

    screen = build_review_screen(
        routed_review_path=args.routed_review,
        future_review_path=args.future_review,
        calibration_path=args.bridge_folding_corpus,
        parent_outputs_path=args.parent_outputs,
        verify_sealed_digests=not args.allow_unsealed_screen,
    )
    probes = review_probe_texts(args.routed_review, 16)
    dropped = sum(
        1 for probe in probes if "reserved_review_containment" in screen.drop_reasons(probe)
    )
    clean = "The quick brown fox jumps over the lazy dog. " * 40
    results = {
        "clean_text_drop_reasons": list(screen.drop_reasons(clean)),
        "probe_count": len(probes),
        "probe_dropped_count": dropped,
        "reserved_digest_count": screen.provenance["reserved_review_screen"][
            "digest_count"
        ],
        "reserved_digest_set_sha256": screen.provenance["reserved_review_screen"][
            "digest_set_sha256"
        ],
        "calibration_digest_count": screen.provenance["calibration_corpus_screen"].get(
            "digest_count"
        ),
        "calibration_digest_set_sha256": screen.provenance[
            "calibration_corpus_screen"
        ].get("digest_set_sha256"),
        "repetition_all_same": three_gram_repetition_fraction([7] * 1024),
        "repetition_all_distinct": three_gram_repetition_fraction(list(range(1024))),
    }
    if dropped != len(probes) or results["clean_text_drop_reasons"]:
        raise Stage2TeacherCorpusError(f"Self-test failed: {results}")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_screen_arguments(parser: argparse.ArgumentParser) -> None:
    root = _repository_root()
    parser.add_argument(
        "--routed-review",
        type=Path,
        default=root / "research" / "lfm2_semantic_whole_response_prompts_v1.json",
    )
    parser.add_argument(
        "--future-review",
        type=Path,
        default=root / "research" / "lfm2_causal_support_future_manual_review_v1.json",
    )
    parser.add_argument("--bridge-folding-corpus", type=Path)
    parser.add_argument("--parent-outputs", type=Path)
    parser.add_argument("--allow-unsealed-screen", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    root = _repository_root()

    estate = subparsers.add_parser("estate", help="build the sealed prefix estate")
    _add_screen_arguments(estate)
    estate.add_argument(
        "--fit-prompt-pack",
        type=Path,
        default=root / "experiments" / "lfm2_on_policy_fit_prompts_v1.json",
    )
    estate.add_argument(
        "--repository-corpus",
        nargs=3,
        action="append",
        metavar=("NAME", "SELECTION_EVIDENCE", "REPOSITORY_ROOT"),
    )
    estate.add_argument("--output", type=Path, required=True)
    estate.set_defaults(handler=_run_estate)

    generate = subparsers.add_parser("generate", help="sample and seal shards")
    _add_screen_arguments(generate)
    generate.add_argument("--teacher", type=Path, required=True)
    generate.add_argument("--prefix-estate", type=Path, required=True)
    generate.add_argument("--output-root", type=Path, required=True)
    generate.add_argument("--device", default="cuda:0")
    generate.add_argument("--dtype", default="float16")
    generate.add_argument("--sequence-length", type=int, default=1024)
    generate.add_argument("--prefix-tokens", type=int, default=64)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--top-p", type=float, default=0.995)
    generate.add_argument("--batch-size", type=int, default=512)
    generate.add_argument("--rows-per-shard", type=int, default=4096)
    generate.add_argument("--train-shards", type=int, default=262)
    generate.add_argument("--holdout-shards", type=int, default=5)
    generate.add_argument("--repetition-drop-threshold", type=float, default=0.5)
    generate.add_argument("--inject-review-probes", type=int, default=1)
    generate.add_argument("--max-shards", type=int, default=0)
    generate.add_argument("--design-sha256", default="")
    generate.add_argument("--allow-unpinned-teacher", action="store_true")
    generate.add_argument("--no-resume", dest="resume", action="store_false")
    generate.add_argument("--verify-resume", action="store_true")
    generate.add_argument("--verify-seal", action="store_true")
    generate.add_argument(
        "--no-seal-on-completion", dest="seal_on_completion", action="store_false"
    )
    generate.set_defaults(handler=_run_generate, resume=True, seal_on_completion=True)

    seal = subparsers.add_parser("seal", help="seal the corpus manifest")
    seal.add_argument("--output-root", type=Path, required=True)
    seal.add_argument("--verify-seal", action="store_true")
    seal.set_defaults(handler=_run_seal)

    self_test = subparsers.add_parser("self-test", help="CPU screen/manifest checks")
    _add_screen_arguments(self_test)
    self_test.set_defaults(handler=_run_self_test)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except Stage2TeacherCorpusError as exc:
        print(f"stage2-corpus error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
