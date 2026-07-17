#!/usr/bin/env python3
"""Compile broad target-tokenized folding trajectories without a parent run.

The output is consumed by ``prepare_lfm2_tokenizer_bridge_evidence.py``.  It
extracts prompt/response pairs from explicit local JSON files, canonicalizes
bounded input and continuation text with the already compiled target tokenizer,
assigns deterministic whole-record splits, and seals every source and output.
It never loads a model, executes a parent, trains, prunes, or quantizes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROW_SCHEMA = "neural-forge-lfm2-bridge-trajectory-text-v1"
RECEIPT_SCHEMA = "neural-forge-lfm2-bridge-folding-corpus-v1"


class Lfm2BridgeFoldingCorpusError(ValueError):
    """A local source, tokenizer boundary, or output seal is invalid."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2BridgeFoldingCorpusError(
            "Folding-corpus provenance must be finite canonical JSON."
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2BridgeFoldingCorpusError(
            f"{name} must be a positive integer."
        )
    return value


def _local_file(path: Path, name: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise Lfm2BridgeFoldingCorpusError(f"{name} does not exist.") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise Lfm2BridgeFoldingCorpusError(
            f"{name} must be a regular local file."
        )
    return resolved


def _local_directory(path: Path, name: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise Lfm2BridgeFoldingCorpusError(f"{name} does not exist.") from exc
    if not resolved.is_dir() or resolved.is_symlink():
        raise Lfm2BridgeFoldingCorpusError(
            f"{name} must be a real local directory."
        )
    return resolved


def _atomic_write_new(path: Path, payload: bytes) -> None:
    destination = path.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise Lfm2BridgeFoldingCorpusError(
            f"Refusing to replace existing output {destination}."
        )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_local_tokenizer(directory: Path) -> Any:
    try:
        from experiments import run_lfm2_residual_attention_fold as residual
    except ImportError:  # Direct execution from the experiments directory.
        import run_lfm2_residual_attention_fold as residual

    return residual.load_local_tokenizer(directory)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2BridgeFoldingCorpusError(
            f"Source is not valid UTF-8 JSON: {path}."
        ) from exc


def _message_pairs(messages: Any) -> Iterable[tuple[str, str]]:
    if not isinstance(messages, list):
        return
    prefix: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str) or not content:
            continue
        if role == "assistant" and prefix:
            yield "\n\n".join(prefix), content
        prefix.append(f"{role}: {content}")


def _pairs_from_value(value: Any) -> Iterable[tuple[str, str, int]]:
    records = value if isinstance(value, list) else [value]
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            continue
        prompt = row.get("prompt")
        response = row.get("response")
        if isinstance(prompt, str) and prompt and isinstance(response, str) and response:
            yield prompt, response, index
            continue
        for prompt_text, response_text in _message_pairs(row.get("messages")):
            yield prompt_text, response_text, index


def _encode(tokenizer: Any, text: str, *, add_special_tokens: bool) -> tuple[int, ...]:
    try:
        values = tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
        )
    except Exception as exc:
        raise Lfm2BridgeFoldingCorpusError(
            "Target tokenizer rejected local corpus text."
        ) from exc
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes, bytearray))
        or not values
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        )
    ):
        raise Lfm2BridgeFoldingCorpusError(
            "Target tokenizer returned invalid token IDs."
        )
    return tuple(int(value) for value in values)


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        value = tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        value = tokenizer.decode(list(token_ids), skip_special_tokens=True)
    if not isinstance(value, str) or not value:
        raise Lfm2BridgeFoldingCorpusError(
            "Target tokenizer returned empty decoded text."
        )
    return value


def _bounded_input_text(
    tokenizer: Any,
    text: str,
    *,
    minimum_tokens: int,
    maximum_tokens: int,
) -> tuple[str, tuple[int, ...]] | None:
    original = _encode(tokenizer, text, add_special_tokens=True)
    if minimum_tokens <= len(original) <= maximum_tokens:
        return text, original
    if len(original) < minimum_tokens:
        return None
    # Use a suffix so long reasoning/code rows contribute their conclusion and
    # local dependencies. Re-tokenize the decoded suffix and accept only a
    # canonical bounded result; no assumed BPE boundary equivalence.
    start = max(0, len(original) - maximum_tokens)
    while start < len(original):
        candidate = _decode(tokenizer, original[start:])
        canonical = _encode(tokenizer, candidate, add_special_tokens=True)
        if minimum_tokens <= len(canonical) <= maximum_tokens:
            return candidate, canonical
        start += 1
    return None


def _bounded_forced_text(
    tokenizer: Any,
    text: str,
    *,
    maximum_tokens: int,
) -> tuple[str, tuple[int, ...]] | None:
    values = _encode(tokenizer, text, add_special_tokens=False)
    for count in range(min(maximum_tokens, len(values)), 0, -1):
        candidate = _decode(tokenizer, values[:count])
        canonical = _encode(tokenizer, candidate, add_special_tokens=False)
        if canonical == values[:count]:
            return candidate, canonical
    return None


def _split_for_rank(rank: int) -> str:
    bucket = rank % 10
    if bucket < 2:
        return "selection"
    if bucket < 8:
        return "fit"
    return "validation"


def compile_bridge_folding_corpus(
    *,
    target_tokenizer_directory: Path,
    source_json_paths: Sequence[Path],
    output_path: Path,
    receipt_path: Path,
    maximum_rows: int,
    minimum_input_tokens: int,
    maximum_input_tokens: int,
    maximum_forced_tokens: int,
    bridge_artifact_loader: Any | None = None,
    tokenizer_semantics_fingerprint: Any | None = None,
) -> dict[str, Any]:
    """Compile deterministic whole-record folding trajectories and seals."""

    row_limit = _positive_int(maximum_rows, "maximum_rows")
    minimum_input = _positive_int(minimum_input_tokens, "minimum_input_tokens")
    maximum_input = _positive_int(maximum_input_tokens, "maximum_input_tokens")
    maximum_forced = _positive_int(maximum_forced_tokens, "maximum_forced_tokens")
    if minimum_input > maximum_input:
        raise Lfm2BridgeFoldingCorpusError(
            "minimum_input_tokens cannot exceed maximum_input_tokens."
        )
    if bridge_artifact_loader is None or tokenizer_semantics_fingerprint is None:
        from neural_forge.hf_tokenizer_bridge import (
            fingerprint_hf_tokenizer_semantics,
            load_strict_tokenizer_bridge_artifact,
        )

        bridge_artifact_loader = load_strict_tokenizer_bridge_artifact
        tokenizer_semantics_fingerprint = fingerprint_hf_tokenizer_semantics

    target_root = _local_directory(
        target_tokenizer_directory,
        "target_tokenizer_directory",
    )
    artifact = bridge_artifact_loader(target_root)
    artifact.verify_files()
    tokenizer = _load_local_tokenizer(target_root)
    target_semantics = tokenizer_semantics_fingerprint(tokenizer)
    if target_semantics != artifact.target_tokenizer_semantics_sha256:
        raise Lfm2BridgeFoldingCorpusError(
            "Live target tokenizer differs from the sealed bridge."
        )
    sources = tuple(
        sorted(
            {
                _local_file(path, "source_json").resolve()
                for path in source_json_paths
            },
            key=str,
        )
    )
    if not sources:
        raise Lfm2BridgeFoldingCorpusError(
            "At least one explicit local JSON source is required."
        )

    candidates: dict[str, dict[str, Any]] = {}
    rejected = 0
    for source in sources:
        for prompt, response, source_index in _pairs_from_value(_read_json(source)):
            try:
                bounded_input = _bounded_input_text(
                    tokenizer,
                    prompt,
                    minimum_tokens=minimum_input,
                    maximum_tokens=maximum_input,
                )
                bounded_forced = _bounded_forced_text(
                    tokenizer,
                    response,
                    maximum_tokens=maximum_forced,
                )
            except Lfm2BridgeFoldingCorpusError:
                rejected += 1
                continue
            if bounded_input is None or bounded_forced is None:
                rejected += 1
                continue
            input_text, input_ids = bounded_input
            forced_text, forced_ids = bounded_forced
            identity = {
                "input_text": input_text,
                "forced_decode_text": forced_text,
            }
            digest = _canonical_sha256(identity)
            candidates.setdefault(
                digest,
                {
                    "digest": digest,
                    "input_text": input_text,
                    "forced_decode_text": forced_text,
                    "input_tokens": len(input_ids),
                    "forced_tokens": len(forced_ids),
                    "source_path": str(source),
                    "source_index": source_index,
                },
            )
    selected = sorted(candidates.values(), key=lambda item: item["digest"])[:row_limit]
    if len(selected) < 3:
        raise Lfm2BridgeFoldingCorpusError(
            "Local sources yielded fewer than three canonical trajectories."
        )

    rows = []
    split_counts = {"selection": 0, "fit": 0, "validation": 0}
    for rank, candidate in enumerate(selected):
        split = _split_for_rank(rank)
        split_counts[split] += 1
        rows.append(
            {
                "schema": ROW_SCHEMA,
                "id": f"broad-{rank:06d}-{candidate['digest'][:16]}",
                "split": split,
                "input_text": candidate["input_text"],
                "forced_decode_text": candidate["forced_decode_text"],
            }
        )
    if any(count <= 0 for count in split_counts.values()):
        raise Lfm2BridgeFoldingCorpusError(
            "Compiled rows do not populate every required whole-record split."
        )

    source_seals = [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in sources
    ]
    row_payload = _canonical_json(rows) + b"\n"
    _atomic_write_new(output_path, row_payload)
    input_counts = [int(item["input_tokens"]) for item in selected]
    forced_counts = [int(item["forced_tokens"]) for item in selected]
    body = {
        "schema": RECEIPT_SCHEMA,
        "status": "compiled",
        "target_tokenizer_directory": str(target_root),
        "target_tokenizer_semantics_sha256": target_semantics,
        "tokenizer_bridge_artifact_sha256": artifact.artifact_sha256,
        "tokenizer_bridge_profile_sha256": artifact.profile_fingerprint,
        "source_files": source_seals,
        "source_file_count": len(source_seals),
        "candidate_count": len(candidates),
        "rejected_pair_count": rejected,
        "row_count": len(rows),
        "split_counts": split_counts,
        "selection": {
            "ordering": "canonical-pair-sha256",
            "maximum_rows": row_limit,
            "minimum_input_tokens": minimum_input,
            "maximum_input_tokens": maximum_input,
            "maximum_forced_tokens": maximum_forced,
        },
        "observed_tokens": {
            "minimum_input": min(input_counts),
            "maximum_input": max(input_counts),
            "minimum_forced": min(forced_counts),
            "maximum_forced": max(forced_counts),
        },
        "trajectory_file": str(output_path.expanduser().absolute()),
        "trajectory_file_sha256": hashlib.sha256(row_payload).hexdigest(),
        "model_loaded": False,
        "parent_forward_calls": 0,
        "parent_generation_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "model_parameters_pruned": 0,
        "model_parameters_quantized": 0,
        "network_access": False,
    }
    receipt = {**body, "receipt_sha256": _canonical_sha256(body)}
    _atomic_write_new(receipt_path, _canonical_json(receipt) + b"\n")
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-tokenizer-directory",
        required=True,
        type=Path,
    )
    parser.add_argument("--source-json", action="append", default=[], type=Path)
    parser.add_argument(
        "--source-directory",
        action="append",
        default=[],
        type=Path,
        help=(
            "Recursively include files matching *_rows_*.json from a local "
            "curated-review directory."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--maximum-rows", type=int, default=128)
    parser.add_argument("--minimum-input-tokens", type=int, default=32)
    parser.add_argument("--maximum-input-tokens", type=int, default=256)
    parser.add_argument("--maximum-forced-tokens", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sources = list(args.source_json)
        for directory in args.source_directory:
            root = _local_directory(directory, "source_directory")
            sources.extend(
                path
                for path in root.rglob("*_rows_*.json")
                if path.is_file() and not path.is_symlink()
            )
        receipt = compile_bridge_folding_corpus(
            target_tokenizer_directory=args.target_tokenizer_directory,
            source_json_paths=sources,
            output_path=args.output,
            receipt_path=args.receipt,
            maximum_rows=args.maximum_rows,
            minimum_input_tokens=args.minimum_input_tokens,
            maximum_input_tokens=args.maximum_input_tokens,
            maximum_forced_tokens=args.maximum_forced_tokens,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": RECEIPT_SCHEMA,
                    "status": "failed_closed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=os.sys.stderr,
        )
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
