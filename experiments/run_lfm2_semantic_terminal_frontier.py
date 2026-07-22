#!/usr/bin/env python3
"""Resume one LFM2 preterminal capsule and materialize its complete W2 frontier.

This runner performs no parent forward passes, gradient updates, pruning,
quantization, sparsification, or numeric-threshold promotion.  Numeric evidence
may determine the deterministic order stored by the compiler, but every
retained structural candidate is written as a standalone dense child for
whole-response review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from experiments.run_lfm2_residual_attention_fold import (  # noqa: E402
    load_local_tokenizer,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_semantic_postfold_frontier import (  # noqa: E402
    Lfm2SemanticTerminalFrontierSourceRoots,
    compile_lfm2_semantic_terminal_frontier_bank_from_capsule,
    load_lfm2_semantic_terminal_frontier_bank,
    materialize_lfm2_semantic_terminal_frontier_candidate,
)


RUN_SCHEMA = "neural-forge-lfm2-semantic-terminal-frontier-run-v1"
PROGRESS_SCHEMA = "neural-forge-lfm2-semantic-terminal-frontier-progress-v1"
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")
_ZERO_OPERATION_AUDIT = {
    "training_steps": 0,
    "optimizer_steps": 0,
    "backward_calls": 0,
    "parent_forward_passes": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
    "parameters_sparsified": 0,
    "sidecar_parameters": 0,
}


class Lfm2SemanticTerminalFrontierRunnerError(RuntimeError):
    """The complete preterminal-resume transaction could not be performed."""


def _absolute(value: str | os.PathLike[str]) -> Path:
    return Path(value).expanduser().absolute()


def _existing_real_directory(value: str | os.PathLike[str], *, label: str) -> Path:
    lexical = _absolute(value)
    if lexical.is_symlink():
        raise Lfm2SemanticTerminalFrontierRunnerError(f"{label} cannot be a link.")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise Lfm2SemanticTerminalFrontierRunnerError(f"{label} is missing.") from exc
    if resolved != lexical or not resolved.is_dir():
        raise Lfm2SemanticTerminalFrontierRunnerError(
            f"{label} must be one canonical real directory."
        )
    return resolved


def _new_path(value: str | os.PathLike[str], *, label: str) -> Path:
    path = _absolute(value)
    if path.exists() or path.is_symlink():
        raise Lfm2SemanticTerminalFrontierRunnerError(
            f"Refusing to replace existing {label}: {path}."
        )
    parent = _existing_real_directory(path.parent, label=f"{label} parent")
    return parent / path.name


def _overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _require_write_disjoint(
    *,
    readonly: Sequence[Path],
    writes: Sequence[Path],
) -> None:
    """Keep every write isolated while allowing legitimate shared read roots."""

    for index, left in enumerate(writes):
        for right in writes[index + 1 :]:
            if _overlap(left, right):
                raise Lfm2SemanticTerminalFrontierRunnerError(
                    f"Frontier transaction paths overlap: {left} and {right}."
                )
        for right in readonly:
            if _overlap(left, right):
                raise Lfm2SemanticTerminalFrontierRunnerError(
                    f"Frontier transaction paths overlap: {left} and {right}."
                )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _candidate_component(index: int, candidate_id: str) -> str:
    cleaned = _SAFE_COMPONENT.sub("-", candidate_id).strip(".-")[:56] or "candidate"
    identity = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:12]
    return f"{index:03d}-{cleaned}-{identity}"


def _source_roots_from_manifest(
    manifest: Mapping[str, Any],
) -> Lfm2SemanticTerminalFrontierSourceRoots:
    authenticated = manifest.get("authenticated_source_roots")
    if not isinstance(authenticated, Mapping):
        raise Lfm2SemanticTerminalFrontierRunnerError(
            "Frontier bank omitted its source-root inventory."
        )
    role_paths = authenticated.get("role_paths")
    if not isinstance(role_paths, Mapping):
        raise Lfm2SemanticTerminalFrontierRunnerError(
            "Frontier source-root roles are malformed."
        )

    def one(role: str) -> Path:
        values = role_paths.get(role)
        if not isinstance(values, list) or len(values) != 1:
            raise Lfm2SemanticTerminalFrontierRunnerError(
                f"Frontier source role {role!r} must name exactly one directory."
            )
        return Path(values[0])

    def many(role: str) -> tuple[Path, ...]:
        values = role_paths.get(role)
        if not isinstance(values, list) or not values:
            raise Lfm2SemanticTerminalFrontierRunnerError(
                f"Frontier source role {role!r} must be nonempty."
            )
        return tuple(Path(value) for value in values)

    return Lfm2SemanticTerminalFrontierSourceRoots(
        preterminal_capsule_directory=one("preterminal_capsule"),
        semantic_cache_directory=one("semantic_cache"),
        tokenizer_bridge_directory=one("tokenizer_bridge"),
        parent_model_directory=one("parent_model"),
        donor_tokenizer_directories=many("donor_tokenizer"),
        target_tokenizer_directories=many("target_tokenizer"),
    )


def run_complete_frontier(
    *,
    preterminal_capsule_directory: Path,
    expected_preterminal_receipt_file_sha256: str,
    expected_preterminal_receipt_sha256: str,
    expected_preterminal_bindings_sha256: str,
    cache_directory: Path,
    expected_cache_key: str,
    bridge_directory: Path,
    expected_bridge_artifact_sha256: str,
    donor_tokenizer_directory: Path,
    parent_model_directory: Path,
    frontier_bank_directory: Path,
    materializations_directory: Path,
    receipt_json: Path,
    device: str,
    dtype: str = "bfloat16",
) -> dict[str, Any]:
    """Compile, retain, and materialize every structural terminal candidate."""

    readonly = tuple(
        _existing_real_directory(path, label=label)
        for path, label in (
            (preterminal_capsule_directory, "preterminal capsule"),
            (cache_directory, "semantic cache"),
            (bridge_directory, "tokenizer bridge"),
            (donor_tokenizer_directory, "donor tokenizer"),
            (parent_model_directory, "parent model"),
        )
    )
    capsule, cache, bridge, donor_directory, parent = readonly
    bank_output = _new_path(frontier_bank_directory, label="frontier bank")
    materializations = _new_path(
        materializations_directory,
        label="frontier materializations",
    )
    receipt = _new_path(receipt_json, label="frontier run receipt")
    _require_write_disjoint(
        readonly=(capsule, cache, bridge, donor_directory, parent),
        writes=(bank_output, materializations, receipt),
    )

    donor_tokenizer = load_local_tokenizer(donor_directory)
    compilation = compile_lfm2_semantic_terminal_frontier_bank_from_capsule(
        preterminal_capsule_directory=capsule,
        expected_preterminal_receipt_file_sha256=(
            expected_preterminal_receipt_file_sha256
        ),
        expected_preterminal_receipt_sha256=expected_preterminal_receipt_sha256,
        expected_preterminal_bindings_sha256=expected_preterminal_bindings_sha256,
        cache_directory=cache,
        expected_cache_key=expected_cache_key,
        bridge_directory=bridge,
        expected_bridge_artifact_sha256=expected_bridge_artifact_sha256,
        donor_tokenizer=donor_tokenizer,
        parent_model_directory=parent,
        output_directory=bank_output,
        device=device,
        dtype=dtype,
    )
    bank_reference = compilation.bank_reference
    bank = load_lfm2_semantic_terminal_frontier_bank(
        bank_reference.path,
        expected_receipt_file_sha256=bank_reference.receipt_file_sha256,
        expected_receipt_sha256=bank_reference.receipt_sha256,
        expected_preterminal_capsule_receipt_sha256=(
            bank_reference.preterminal_capsule_receipt_sha256
        ),
        expected_terminal_result_sha256=bank_reference.terminal_result_sha256,
    )
    if tuple(bank.reference.candidate_ids) != tuple(bank_reference.candidate_ids):
        raise Lfm2SemanticTerminalFrontierRunnerError(
            "Reloaded frontier candidate inventory changed."
        )
    source_roots = _source_roots_from_manifest(bank.manifest)
    materializations.mkdir(mode=0o700)
    progress_path = materializations / "progress.json"
    completed: list[dict[str, Any]] = []
    candidate_ids = tuple(bank_reference.candidate_ids)
    for index, candidate_id in enumerate(candidate_ids, start=1):
        destination = materializations / _candidate_component(index, candidate_id)
        reference = materialize_lfm2_semantic_terminal_frontier_candidate(
            preterminal_capsule_directory=capsule,
            expected_preterminal_receipt_file_sha256=(
                expected_preterminal_receipt_file_sha256
            ),
            expected_preterminal_receipt_sha256=(expected_preterminal_receipt_sha256),
            expected_preterminal_bindings_sha256=(expected_preterminal_bindings_sha256),
            terminal_frontier_bank_directory=bank_reference.path,
            expected_terminal_frontier_bank_receipt_file_sha256=(
                bank_reference.receipt_file_sha256
            ),
            expected_terminal_frontier_bank_receipt_sha256=(
                bank_reference.receipt_sha256
            ),
            expected_terminal_result_sha256=bank_reference.terminal_result_sha256,
            expected_source_roots=source_roots,
            candidate_id=candidate_id,
            output_directory=destination,
            device=device,
            dtype=dtype,
        )
        completed.append(reference.as_dict())
        progress_body = {
            "schema": PROGRESS_SCHEMA,
            "status": "running" if index < len(candidate_ids) else "complete",
            "terminal_frontier_bank": bank_reference.as_dict(),
            "candidate_count": len(candidate_ids),
            "materialized_count": len(completed),
            "remaining_count": len(candidate_ids) - len(completed),
            "candidate_ids": list(candidate_ids),
            "materializations": list(completed),
            "all_candidates_retained_without_numeric_veto": True,
            "operation_audit": dict(_ZERO_OPERATION_AUDIT),
        }
        _atomic_json(
            progress_path,
            {**progress_body, "progress_sha256": canonical_sha256(progress_body)},
        )

    body = {
        "schema": RUN_SCHEMA,
        "status": "complete",
        "preterminal_capsule": compilation.preterminal_capsule_reference.as_dict(),
        "terminal_frontier_bank": bank_reference.as_dict(),
        "terminal_frontier_compilation": compilation.as_dict(),
        "source_roots": source_roots.as_dict(),
        "materializations_directory": str(materializations),
        "progress_json": str(progress_path),
        "candidate_count": len(candidate_ids),
        "materialized_count": len(completed),
        "candidate_ids": list(candidate_ids),
        "materializations": completed,
        "all_retained_candidates_materialized": len(completed) == len(candidate_ids),
        "candidate_metric_threshold_used_for_materialization": False,
        "numeric_metrics_are_diagnostics_only": True,
        "whole_response_intelligence_review_is_authoritative": True,
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
    }
    document = {**body, "run_sha256": canonical_sha256(body)}
    _atomic_json(receipt, document)
    return document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resume one LFM2 preterminal capsule, export the complete terminal "
            "frontier, and materialize every retained candidate without training."
        )
    )
    parser.add_argument("--preterminal-capsule", required=True, type=Path)
    parser.add_argument("--expected-preterminal-receipt-file-sha256", required=True)
    parser.add_argument("--expected-preterminal-receipt-sha256", required=True)
    parser.add_argument("--expected-preterminal-bindings-sha256", required=True)
    parser.add_argument("--trace-cache", required=True, type=Path)
    parser.add_argument("--expected-cache-key", required=True)
    parser.add_argument("--bridge-directory", required=True, type=Path)
    parser.add_argument("--expected-bridge-artifact-sha256", required=True)
    parser.add_argument("--donor-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--parent-model-directory", required=True, type=Path)
    parser.add_argument("--frontier-bank", required=True, type=Path)
    parser.add_argument("--materializations", required=True, type=Path)
    parser.add_argument("--receipt-json", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_complete_frontier(
        preterminal_capsule_directory=args.preterminal_capsule,
        expected_preterminal_receipt_file_sha256=(
            args.expected_preterminal_receipt_file_sha256
        ),
        expected_preterminal_receipt_sha256=(args.expected_preterminal_receipt_sha256),
        expected_preterminal_bindings_sha256=(
            args.expected_preterminal_bindings_sha256
        ),
        cache_directory=args.trace_cache,
        expected_cache_key=args.expected_cache_key,
        bridge_directory=args.bridge_directory,
        expected_bridge_artifact_sha256=(args.expected_bridge_artifact_sha256),
        donor_tokenizer_directory=args.donor_tokenizer_directory,
        parent_model_directory=args.parent_model_directory,
        frontier_bank_directory=args.frontier_bank,
        materializations_directory=args.materializations,
        receipt_json=args.receipt_json,
        device=args.device,
        dtype=args.dtype,
    )
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
