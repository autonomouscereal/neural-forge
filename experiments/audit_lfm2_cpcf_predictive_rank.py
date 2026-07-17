#!/usr/bin/env python3
"""Audit CPCF predictive rank from one existing sealed LFM2 trace cache.

This command is intentionally evidence-only.  It cannot load a parent model,
execute a parent forward, regenerate traces, train, prune, quantize, or
materialize a child.  It verifies an already completed circuit cache, streams
its cached evidence through the CPCF rank adapter, and atomically publishes a
self-hashed JSON report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ARTIFACT_SCHEMA = "neural-forge-lfm2-cpcf-predictive-rank-artifact-v1"
ARTIFACT_SEAL = "artifact_sha256"


class Lfm2CpcfPredictiveRankAuditCliError(ValueError):
    """The cache request or immutable report publication is invalid."""


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
        raise Lfm2CpcfPredictiveRankAuditCliError(
            "The CPCF rank artifact must be finite canonical JSON."
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
        raise Lfm2CpcfPredictiveRankAuditCliError(
            f"Cannot hash published report {path}."
        ) from exc
    return digest.hexdigest()


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Lfm2CpcfPredictiveRankAuditCliError(
                f"Published report contains duplicate JSON key {key!r}."
            )
        result[key] = value
    return result


def _sealed_artifact(body: Mapping[str, Any]) -> dict[str, Any]:
    if ARTIFACT_SEAL in body:
        raise Lfm2CpcfPredictiveRankAuditCliError(
            f"Artifact body already contains {ARTIFACT_SEAL!r}."
        )
    result = dict(body)
    result[ARTIFACT_SEAL] = _canonical_sha256(result)
    return result


def _verify_artifact(value: Mapping[str, Any]) -> None:
    if value.get("schema") != ARTIFACT_SCHEMA:
        raise Lfm2CpcfPredictiveRankAuditCliError(
            "Published report has the wrong artifact schema."
        )
    observed = value.get(ARTIFACT_SEAL)
    if not isinstance(observed, str) or len(observed) != 64:
        raise Lfm2CpcfPredictiveRankAuditCliError(
            "Published report has no valid artifact seal."
        )
    body = dict(value)
    body.pop(ARTIFACT_SEAL)
    if _canonical_sha256(body) != observed:
        raise Lfm2CpcfPredictiveRankAuditCliError(
            "Published report does not match its artifact seal."
        )


def _read_published_artifact(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Lfm2CpcfPredictiveRankAuditCliError(
            "Existing report must be a regular file."
        )
    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=_strict_pairs)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise Lfm2CpcfPredictiveRankAuditCliError(
            "Existing report is unreadable or invalid JSON."
        ) from exc
    if not isinstance(value, dict):
        raise Lfm2CpcfPredictiveRankAuditCliError(
            "Existing report must contain a JSON object."
        )
    if raw != _canonical_json(value) + b"\n":
        raise Lfm2CpcfPredictiveRankAuditCliError(
            "Existing report is not canonical JSON."
        )
    _verify_artifact(value)
    return value


def _publish_artifact(path: Path, artifact: Mapping[str, Any]) -> str:
    """Publish once, or authenticate and reuse byte-identical contents."""

    _verify_artifact(artifact)
    destination = path.expanduser().absolute()
    if destination.exists():
        existing = _read_published_artifact(destination)
        if existing != dict(artifact):
            raise Lfm2CpcfPredictiveRankAuditCliError(
                "Refusing to replace a different immutable CPCF report."
            )
        return _file_sha256(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.stage-{uuid.uuid4().hex}"
    )
    payload = _canonical_json(dict(artifact)) + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except OSError as exc:
        raise Lfm2CpcfPredictiveRankAuditCliError(
            f"Could not atomically publish report {destination}."
        ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return _file_sha256(destination)


def _candidate_widths(values: Sequence[int]) -> tuple[int, ...]:
    widths = tuple(values)
    if (
        not widths
        or any(
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            for width in widths
        )
        or len(set(widths)) != len(widths)
    ):
        raise Lfm2CpcfPredictiveRankAuditCliError(
            "Candidate widths must be unique positive integers."
        )
    return widths


def run_audit(
    *,
    cache_directory: Path,
    output_path: Path,
    candidate_widths: Sequence[int],
) -> dict[str, Any]:
    """Verify a sealed cache, run the CPCF audit, and publish its report."""

    from neural_forge.hf_lfm2_circuit_trace_cache import (
        open_lfm2_circuit_trace_cache,
    )
    from neural_forge.hf_lfm2_cpcf_cached_evidence import (
        Lfm2CpcfEvidenceConfig,
        audit_lfm2_cpcf_predictive_rank,
    )

    requested_widths = _candidate_widths(candidate_widths)
    cache_path = cache_directory.expanduser().absolute()
    if cache_path.is_symlink() or not cache_path.is_dir():
        raise Lfm2CpcfPredictiveRankAuditCliError(
            "cache_directory must be an existing regular directory."
        )
    cache_path = cache_path.resolve(strict=True)
    cache = open_lfm2_circuit_trace_cache(cache_path, verify=True)
    config = Lfm2CpcfEvidenceConfig(predictive_energy_fraction=0.90)
    audit = audit_lfm2_cpcf_predictive_rank(
        cache,
        config=config,
        candidate_widths=requested_widths,
    )
    cache.verify()
    body = {
        "schema": ARTIFACT_SCHEMA,
        "cache_evidence": {
            "cache_directory": str(cache_path),
            "cache_key": cache.cache_key,
            "batch_count": cache.batch_count,
            "cache_opened_with_full_verification": True,
            "cache_verified_after_streaming": True,
            "existing_sealed_cache_only": True,
            "cache_regenerated": False,
        },
        "audit": audit.as_dict(),
        "claim_boundary": {
            "predictive_rank_feasibility_measured": True,
            "parameter_compaction_proven": False,
            "standalone_child_materialized": False,
            "standalone_child_coherence_measured": False,
            "intelligence_retention_measured": False,
            "attention_installability_proven": False,
            "scope": "cached_residual_width_predictive_rank_evidence_only",
        },
    }
    artifact = _sealed_artifact(body)
    output = output_path.expanduser().absolute()
    file_sha256 = _publish_artifact(output, artifact)
    return {
        "status": "complete",
        "output_path": str(output),
        "file_sha256": file_sha256,
        "artifact_sha256": artifact[ARTIFACT_SEAL],
        "audit_report_sha256": audit.report_sha256,
        "cache_key": audit.cache_key,
        "batch_count": cache.batch_count,
        "layer_count": len(audit.boundaries),
        "source_width": audit.source_width,
        "minimum_uniform_width_at_90_predictive_energy": (
            audit.minimum_uniform_width_at_90_predictive_energy
        ),
        "maximum_uniform_residual_width_compression_ratio": (
            audit.maximum_uniform_residual_width_compression_ratio
        ),
        "candidate_widths": {
            str(width): dict(report)
            for width, report in audit.candidate_widths.items()
        },
        "full_compilation_ready": audit.full_compilation_ready,
        "claim_scope": "predictive_rank_feasibility_only",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify an existing sealed LFM2 circuit cache and atomically "
            "publish its CPCF 90%-predictive-energy rank audit."
        )
    )
    parser.add_argument("--cache-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--candidate-widths",
        required=True,
        nargs="+",
        type=int,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_audit(
            cache_directory=args.cache_directory,
            output_path=args.output,
            candidate_widths=args.candidate_widths,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": ARTIFACT_SCHEMA,
                    "status": "failed_closed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
