#!/usr/bin/env python3
"""Compile a sealed smaller LFM2 tokenizer from independent local text.

This is tokenizer-interface compilation, not model pruning.  It never loads a
model, edits a learned Parameter, executes a parent, or uses a network
identifier.  Every physical target token row maps to one authenticated donor
row so the dense blank-child compiler can transport embedding and language-head
operators without inventing random vocabulary rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

RECEIPT_SCHEMA = "neural-forge-lfm2-target-tokenizer-compilation-v1"


class Lfm2TargetTokenizerCompilationError(ValueError):
    """A local corpus or target-tokenizer compilation is invalid."""


def _load_local_tokenizer(directory: Path) -> Any:
    try:
        from experiments import run_lfm2_residual_attention_fold as residual
    except ImportError:  # Direct execution from the experiments directory.
        import run_lfm2_residual_attention_fold as residual

    return residual.load_local_tokenizer(directory)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2TargetTokenizerCompilationError(
            f"{name} must be a positive integer."
        )
    return value


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
        raise Lfm2TargetTokenizerCompilationError(
            "Receipt payload is not canonical JSON."
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise Lfm2TargetTokenizerCompilationError(
            f"Cannot hash local corpus file {path}."
        ) from exc
    return digest.hexdigest()


def _regular_local_file(path: Path, name: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise Lfm2TargetTokenizerCompilationError(
            f"{name} must be a regular local file."
        )
    return resolved


def resolve_donor_row_capacity(
    donor_tokenizer_directory: Path,
    *,
    explicit_capacity: int | None,
    tokenizer_size: int,
) -> int:
    """Resolve physical embedding rows separately from lexical token count."""

    lexical_size = _positive_int(tokenizer_size, "tokenizer_size")
    if explicit_capacity is not None:
        capacity = _positive_int(
            explicit_capacity,
            "donor_row_capacity",
        )
    else:
        config_path = donor_tokenizer_directory / "config.json"
        if not config_path.is_file() or config_path.is_symlink():
            raise Lfm2TargetTokenizerCompilationError(
                "A physical donor_row_capacity is required when the local "
                "tokenizer directory has no regular config.json."
            )
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Lfm2TargetTokenizerCompilationError(
                "The local donor config.json is unreadable or invalid."
            ) from exc
        if not isinstance(config, Mapping):
            raise Lfm2TargetTokenizerCompilationError(
                "The local donor config.json must be an object."
            )
        capacity = _positive_int(
            config.get("vocab_size"),
            "config.vocab_size",
        )
    if capacity < lexical_size:
        raise Lfm2TargetTokenizerCompilationError(
            "Physical donor row capacity cannot be below tokenizer size."
        )
    return capacity


def _string_leaves(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        if value:
            yield value
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            yield from _string_leaves(value[key])
        return
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for item in value:
            yield from _string_leaves(item)


def _jsonl_texts(path: Path) -> Iterator[str]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise Lfm2TargetTokenizerCompilationError(
                        f"{path} line {line_number} is not valid JSON."
                    ) from exc
                yield from _string_leaves(value)
    except UnicodeDecodeError as exc:
        raise Lfm2TargetTokenizerCompilationError(
            f"{path} is not UTF-8 JSONL."
        ) from exc


def _parquet_texts(path: Path) -> Iterator[str]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - remote runtime dependency
        raise Lfm2TargetTokenizerCompilationError(
            "pyarrow is required for direct local Parquet calibration."
        ) from exc
    try:
        source = parquet.ParquetFile(path)
        for batch in source.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                yield from _string_leaves(row)
    except (OSError, ValueError, RuntimeError) as exc:
        raise Lfm2TargetTokenizerCompilationError(
            f"Cannot read local Parquet calibration {path}."
        ) from exc


def collect_calibration_texts(
    *,
    jsonl_paths: Sequence[Path],
    parquet_paths: Sequence[Path],
    maximum_texts: int,
    maximum_characters: int,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Collect deterministic, deduplicated string leaves from local files."""

    text_limit = _positive_int(maximum_texts, "maximum_texts")
    character_limit = _positive_int(
        maximum_characters,
        "maximum_characters",
    )
    sources: list[tuple[str, Path, Any]] = []
    for path in jsonl_paths:
        resolved = _regular_local_file(Path(path), "jsonl calibration")
        sources.append(("jsonl", resolved, _jsonl_texts))
    for path in parquet_paths:
        resolved = _regular_local_file(Path(path), "parquet calibration")
        sources.append(("parquet", resolved, _parquet_texts))
    sources.sort(key=lambda value: (value[0], str(value[1])))
    if not sources:
        raise Lfm2TargetTokenizerCompilationError(
            "At least one independent local JSONL or Parquet corpus is required."
        )

    texts: list[str] = []
    seen: set[str] = set()
    character_count = 0
    limit_reached = False
    for _kind, path, reader in sources:
        for text in reader(path):
            if text in seen:
                continue
            remaining = character_limit - character_count
            if remaining <= 0 or len(texts) >= text_limit:
                limit_reached = True
                break
            truncated = len(text) > remaining
            selected = text[:remaining] if truncated else text
            if not selected:
                limit_reached = True
                break
            seen.add(selected)
            texts.append(selected)
            character_count += len(selected)
            if truncated:
                limit_reached = True
                break
        if limit_reached:
            break
    if not texts:
        raise Lfm2TargetTokenizerCompilationError(
            "The local calibration sources contained no text."
        )
    corpus_body = {
        "schema": "neural-forge-local-tokenizer-calibration-corpus-v1",
        "sources": [
            {
                "kind": kind,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for kind, path, _reader in sources
        ],
        "selection_order": "kind_then_resolved_path_then_source_row",
        "string_leaf_order": "mapping_keys_sorted_sequences_preserved",
        "deduplicated": True,
        "maximum_texts": text_limit,
        "maximum_characters": character_limit,
        "selected_texts": len(texts),
        "selected_characters": character_count,
        "limit_reached": limit_reached,
        "selected_text_sha256": _canonical_sha256(texts),
        "evaluation_prompts_used": False,
        "parent_outputs_used": False,
        "model_parameters_used": False,
    }
    return tuple(texts), {
        **corpus_body,
        "corpus_sha256": _canonical_sha256(corpus_body),
    }


def _tokenization_diagnostics(
    donor_tokenizer: Any,
    target_tokenizer: Any,
    texts: Sequence[str],
    *,
    maximum_texts: int = 10_000,
) -> dict[str, Any]:
    selected = tuple(texts[:maximum_texts])
    donor_tokens = 0
    target_tokens = 0
    roundtrip_mismatches = 0
    for text in selected:
        donor_ids = donor_tokenizer.encode(text, add_special_tokens=False)
        target_ids = target_tokenizer.encode(text, add_special_tokens=False)
        donor_tokens += len(donor_ids)
        target_tokens += len(target_ids)
        decoded = target_tokenizer.decode(
            target_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded != text:
            roundtrip_mismatches += 1
    return {
        "diagnostic_texts": len(selected),
        "donor_token_count": donor_tokens,
        "target_token_count": target_tokens,
        "target_to_donor_token_count_ratio": (
            None if donor_tokens == 0 else target_tokens / donor_tokens
        ),
        "target_roundtrip_mismatches": roundtrip_mismatches,
        "target_roundtrip_exact": roundtrip_mismatches == 0,
        "behavioral_normalization_retention_proven": False,
    }


def _write_atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise Lfm2TargetTokenizerCompilationError(
            f"Receipt temporary path already exists: {temporary}."
        )
    payload = _canonical_json(value) + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def compile_target_tokenizer(
    *,
    donor_tokenizer_directory: Path,
    output_directory: Path,
    receipt_path: Path,
    target_vocab_size: int,
    donor_row_capacity: int | None,
    jsonl_paths: Sequence[Path],
    parquet_paths: Sequence[Path],
    maximum_texts: int,
    maximum_characters: int,
) -> dict[str, Any]:
    from neural_forge.hf_tokenizer_bridge import (
        compile_hf_target_tokenizer_bridge,
        fingerprint_hf_tokenizer_semantics,
        load_strict_tokenizer_bridge_artifact,
    )

    donor_root = donor_tokenizer_directory.expanduser().resolve(strict=True)
    if donor_root.is_symlink() or not donor_root.is_dir():
        raise Lfm2TargetTokenizerCompilationError(
            "donor_tokenizer_directory must be a real local directory."
        )
    output = output_directory.expanduser().resolve()
    if output.exists():
        raise Lfm2TargetTokenizerCompilationError(
            "output_directory must not already exist."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    target_size = _positive_int(target_vocab_size, "target_vocab_size")
    texts, corpus_report = collect_calibration_texts(
        jsonl_paths=jsonl_paths,
        parquet_paths=parquet_paths,
        maximum_texts=maximum_texts,
        maximum_characters=maximum_characters,
    )
    donor = _load_local_tokenizer(donor_root)
    donor_semantics = fingerprint_hf_tokenizer_semantics(donor)
    physical_donor_rows = resolve_donor_row_capacity(
        donor_root,
        explicit_capacity=donor_row_capacity,
        tokenizer_size=len(donor),
    )
    if target_size >= len(donor):
        raise Lfm2TargetTokenizerCompilationError(
            "target_vocab_size must be smaller than the donor tokenizer."
        )

    staging = output.with_name(f".{output.name}.{os.getpid()}.staging")
    if staging.exists():
        raise Lfm2TargetTokenizerCompilationError(
            f"Tokenizer staging directory already exists: {staging}."
        )
    try:
        result = compile_hf_target_tokenizer_bridge(
            donor,
            staging,
            target_vocab_size=target_size,
            calibration_texts=texts,
            donor_row_capacity=physical_donor_rows,
        )
        result.artifact.verify_live_tokenizers(donor, result.tokenizer)
        staging.replace(output)
        target = _load_local_tokenizer(output)
        artifact = load_strict_tokenizer_bridge_artifact(
            output,
            donor_tokenizer=donor,
            target_tokenizer=target,
            expected_artifact_sha256=result.artifact.artifact_sha256,
        )
        diagnostics = _tokenization_diagnostics(donor, target, texts)
        body = {
            "schema": RECEIPT_SCHEMA,
            "status": "compiled",
            "donor_tokenizer_directory": str(donor_root),
            "target_tokenizer_directory": str(output),
            "donor_vocabulary_size": len(donor),
            "donor_physical_row_capacity": physical_donor_rows,
            "donor_padding_rows": physical_donor_rows - len(donor),
            "target_vocabulary_size": len(target),
            "donor_tokenizer_semantics_sha256": donor_semantics,
            "target_tokenizer_semantics_sha256": (
                artifact.target_tokenizer_semantics_sha256
            ),
            "bridge_artifact_sha256": artifact.artifact_sha256,
            "bridge_profile_sha256": artifact.profile_fingerprint,
            "selected_unique_donor_rows": len(
                set(artifact.transport_profile.target_to_donor_row_indices)
            ),
            "retained_merge_count": len(result.retained_merges),
            "calibration_corpus": corpus_report,
            "tokenization_diagnostics": diagnostics,
            "model_loaded": False,
            "parent_forward_calls": 0,
            "parent_generation_calls": 0,
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "model_parameters_pruned": 0,
            "model_parameters_quantized": 0,
            "random_target_token_rows": 0,
            "network_access": False,
            "behavioral_normalization_retention_proven": False,
            "next_required_gate": (
                "sealed mapped-sequence parent versus canonical-parent "
                "whole-output normalization"
            ),
        }
        receipt = {**body, "receipt_sha256": _canonical_sha256(body)}
        _write_atomic_json(receipt_path, receipt)
        return receipt
    finally:
        if staging.exists():
            if staging.parent != output.parent or staging.name != (
                f".{output.name}.{os.getpid()}.staging"
            ):
                raise Lfm2TargetTokenizerCompilationError(
                    "Refusing to remove an unverified staging path."
                )
            shutil.rmtree(staging)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile and seal a smaller direct-local LFM2 tokenizer bridge "
            "from independent local corpus files."
        )
    )
    parser.add_argument(
        "--donor-tokenizer-directory",
        required=True,
        type=Path,
    )
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--target-vocab-size", required=True, type=int)
    parser.add_argument(
        "--donor-row-capacity",
        type=int,
        help=(
            "Physical embedding/head row capacity. Defaults to the local "
            "config.json vocab_size, which may exceed lexical tokenizer size."
        ),
    )
    parser.add_argument("--jsonl", action="append", default=[], type=Path)
    parser.add_argument("--parquet", action="append", default=[], type=Path)
    parser.add_argument("--maximum-texts", type=int, default=100_000)
    parser.add_argument("--maximum-characters", type=int, default=32_000_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = compile_target_tokenizer(
            donor_tokenizer_directory=args.donor_tokenizer_directory,
            output_directory=args.output_directory,
            receipt_path=args.receipt,
            target_vocab_size=args.target_vocab_size,
            donor_row_capacity=args.donor_row_capacity,
            jsonl_paths=args.jsonl,
            parquet_paths=args.parquet,
            maximum_texts=args.maximum_texts,
            maximum_characters=args.maximum_characters,
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
            file=sys.stderr,
        )
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
