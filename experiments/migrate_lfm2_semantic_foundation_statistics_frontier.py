#!/usr/bin/env python3
"""Persist a frontier companion for one pinned legacy LFM2 foundation.

This entrypoint is deliberately offline and model-free.  It authenticates one
complete semantic cache, tokenizer bridge, and direct-local parent checkpoint;
reconstructs only the mapped parent tied table; migrates the pinned compact
statistics; and publishes one immutable canonical companion JSON document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any


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
from neural_forge.dynamic_circuit_ir import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_long_horizon_terminal_refold import (  # noqa: E402
    load_lfm2_parent_final_rmsnorm_from_local_safetensors,
)
from neural_forge.hf_lfm2_semantic_foundation_statistics import (  # noqa: E402
    LFM2_SEMANTIC_FOUNDATION_INTERFACE_FRONTIER_COMPANION_SCHEMA,
    Lfm2SemanticFoundationInterfaceFrontierCompanion,
    migrate_lfm2_semantic_foundation_interface_statistics_frontier,
)
from neural_forge.hf_lfm2_semantic_postfold import (  # noqa: E402
    load_lfm2_mapped_parent_tied_table_from_local_safetensors,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    load_strict_tokenizer_bridge_artifact,
)


_MAX_CANONICAL_JSON_BYTES = 64 * 1024 * 1024
_REPORT_SCHEMA = (
    "neural-forge-lfm2-semantic-foundation-frontier-companion-migration-cli-v1"
)
_ZERO_OPERATION_AUDIT = MappingProxyType(
    {
        "parent_model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
        "parent_model_generation_calls": 0,
        "child_model_objects_constructed": 0,
        "child_model_forward_calls": 0,
        "child_model_generation_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "cache_regeneration_calls": 0,
        "trace_capture_calls": 0,
        "network_requests": 0,
    }
)
_EXPECTED_COMPANION_OPERATION_AUDIT = MappingProxyType(
    {
        "compiler_cache_open_calls": 0,
        "trace_safetensor_open_calls": 0,
        "selective_provider_open_calls": 0,
        "parent_model_forward_calls": 0,
        "parent_model_generation_calls": 0,
        "child_model_forward_calls": 0,
        "child_model_generation_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "dense_candidate_checkpoints_persisted": 0,
    }
)


class Lfm2SemanticFoundationFrontierMigrationCliError(RuntimeError):
    """The pinned offline migration or immutable publication failed."""


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2SemanticFoundationFrontierMigrationCliError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return canonical_json_bytes(dict(value)) + b"\n"
    except (TypeError, ValueError) as exc:
        raise Lfm2SemanticFoundationFrontierMigrationCliError(
            "Receipt is not representable as canonical JSON."
        ) from exc


def _new_output_file(value: str | os.PathLike[str]) -> Path:
    candidate = Path(value).expanduser()
    if candidate.exists() or candidate.is_symlink():
        raise Lfm2SemanticFoundationFrontierMigrationCliError(
            f"Refusing to replace existing companion output: {candidate}."
        )
    parent = candidate.parent
    if not parent.is_dir() or parent.is_symlink():
        raise Lfm2SemanticFoundationFrontierMigrationCliError(
            "Companion output parent must be an ordinary existing directory."
        )
    return parent.resolve() / candidate.name


def _read_pinned_canonical_json(
    value: str | os.PathLike[str],
    *,
    expected_canonical_sha256: str,
    expected_file_sha256: str,
    seal_field: str,
    name: str,
) -> tuple[Mapping[str, Any], str]:
    """Read one canonical sealed document once and enforce both external pins."""

    canonical_pin = _digest(expected_canonical_sha256, f"{name} canonical pin")
    file_pin = _digest(expected_file_sha256, f"{name} file pin")
    path = Path(value).expanduser()
    if not path.is_file() or path.is_symlink():
        raise Lfm2SemanticFoundationFrontierMigrationCliError(
            f"{name} must be an ordinary local file."
        )
    try:
        with path.open("rb") as handle:
            raw = handle.read(_MAX_CANONICAL_JSON_BYTES + 1)
        parsed = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2SemanticFoundationFrontierMigrationCliError(
            f"Could not read canonical {name}."
        ) from exc
    if len(raw) > _MAX_CANONICAL_JSON_BYTES or not isinstance(parsed, dict):
        raise Lfm2SemanticFoundationFrontierMigrationCliError(
            f"{name} is not a bounded JSON object."
        )
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if file_sha256 != file_pin:
        raise Lfm2SemanticFoundationFrontierMigrationCliError(
            f"{name} differs from its external file SHA-256 pin."
        )
    if raw != _canonical_bytes(parsed):
        raise Lfm2SemanticFoundationFrontierMigrationCliError(
            f"{name} is not canonical JSON."
        )
    body = dict(parsed)
    claimed = _digest(body.pop(seal_field, None), f"{name} {seal_field}")
    if claimed != canonical_pin or canonical_sha256(body) != canonical_pin:
        raise Lfm2SemanticFoundationFrontierMigrationCliError(
            f"{name} differs from its external canonical SHA-256 pin."
        )
    return MappingProxyType(parsed), file_sha256


def _verify_sealed_mapping(
    value: Any,
    *,
    seal_field: str,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2SemanticFoundationFrontierMigrationCliError(
            f"{name} is not a mapping."
        )
    body = dict(value)
    claimed = _digest(body.pop(seal_field, None), f"{name} {seal_field}")
    if canonical_sha256(body) != claimed:
        raise Lfm2SemanticFoundationFrontierMigrationCliError(
            f"{name} seal is invalid."
        )
    _canonical_bytes(value)
    return MappingProxyType(dict(value))


def _exclusive_atomic_canonical_json(
    path: Path,
    value: Mapping[str, Any],
) -> tuple[str, str]:
    """Publish canonical bytes atomically with a race-free no-replace link."""

    payload = _canonical_bytes(value)
    document_sha256 = canonical_sha256(dict(value))
    file_sha256 = hashlib.sha256(payload).hexdigest()
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
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise Lfm2SemanticFoundationFrontierMigrationCliError(
                "A concurrent writer already published the companion output."
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    try:
        published = path.read_bytes()
    except OSError as exc:
        raise Lfm2SemanticFoundationFrontierMigrationCliError(
            "Could not verify the published companion output."
        ) from exc
    if published != payload:
        raise Lfm2SemanticFoundationFrontierMigrationCliError(
            "Published companion bytes failed immediate canonical verification."
        )
    return document_sha256, file_sha256


def _validated_companion_receipt(
    value: Lfm2SemanticFoundationInterfaceFrontierCompanion,
    *,
    expected_artifact_sha256: str,
    expected_foundation_receipt_sha256: str,
    expected_foundation_receipt_file_sha256: str,
) -> Mapping[str, Any]:
    if not isinstance(
        value,
        Lfm2SemanticFoundationInterfaceFrontierCompanion,
    ):
        raise Lfm2SemanticFoundationFrontierMigrationCliError(
            "Migration did not return a frontier companion."
        )
    receipt = _verify_sealed_mapping(
        value.companion_receipt,
        seal_field="companion_sha256",
        name="frontier companion receipt",
    )
    if (
        receipt.get("schema")
        != LFM2_SEMANTIC_FOUNDATION_INTERFACE_FRONTIER_COMPANION_SCHEMA
        or receipt.get("statistics_artifact_sha256")
        != expected_artifact_sha256
        or receipt.get("legacy_foundation_receipt_sha256")
        != expected_foundation_receipt_sha256
        or receipt.get("legacy_foundation_receipt_file_sha256")
        != expected_foundation_receipt_file_sha256
        or receipt.get("operation_audit")
        != dict(_EXPECTED_COMPANION_OPERATION_AUDIT)
        or receipt.get("numeric_metrics_have_final_intelligence_authority") is not False
        or receipt.get("whole_response_review_selects_final_candidate") is not True
    ):
        raise Lfm2SemanticFoundationFrontierMigrationCliError(
            "Frontier companion does not retain the pinned legacy bindings and "
            "zero-operation contract."
        )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-statistics-artifact-directory",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--expected-legacy-statistics-artifact-sha256",
        required=True,
    )
    parser.add_argument("--semantic-cache-directory", required=True, type=Path)
    parser.add_argument("--expected-semantic-cache-key", required=True)
    parser.add_argument("--bridge-directory", required=True, type=Path)
    parser.add_argument("--expected-bridge-artifact-sha256", required=True)
    parser.add_argument("--donor-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--target-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--parent-model-directory", required=True, type=Path)
    parser.add_argument(
        "--legacy-foundation-receipt-json",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--expected-legacy-foundation-receipt-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-legacy-foundation-receipt-file-sha256",
        required=True,
    )
    parser.add_argument("--output-companion-json", required=True, type=Path)
    parser.add_argument(
        "--parent-final-norm-tensor-key",
        default="model.embedding_norm.weight",
    )
    parser.add_argument("--parent-norm-epsilon-config-key", default="norm_eps")
    parser.add_argument(
        "--parent-tied-tensor-key",
        default="model.embed_tokens.weight",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifact_sha256 = _digest(
        args.expected_legacy_statistics_artifact_sha256,
        "expected legacy statistics artifact",
    )
    cache_key = _digest(args.expected_semantic_cache_key, "expected semantic cache")
    bridge_sha256 = _digest(
        args.expected_bridge_artifact_sha256,
        "expected bridge artifact",
    )
    foundation_receipt_sha256 = _digest(
        args.expected_legacy_foundation_receipt_sha256,
        "expected legacy foundation receipt",
    )
    foundation_receipt_file_sha256 = _digest(
        args.expected_legacy_foundation_receipt_file_sha256,
        "expected legacy foundation receipt file",
    )
    output = _new_output_file(args.output_companion_json)
    foundation_receipt, observed_foundation_file_sha256 = (
        _read_pinned_canonical_json(
            args.legacy_foundation_receipt_json,
            expected_canonical_sha256=foundation_receipt_sha256,
            expected_file_sha256=foundation_receipt_file_sha256,
            seal_field="receipt_sha256",
            name="legacy foundation receipt",
        )
    )

    cache = open_lfm2_circuit_trace_cache(
        args.semantic_cache_directory,
        verify=True,
    )
    if cache.cache_key != cache_key:
        raise Lfm2SemanticFoundationFrontierMigrationCliError(
            "Semantic cache differs from its external identity pin."
        )
    cache_manifest = cache.manifest
    cache_manifest_sha256 = _digest(
        cache_manifest.get("manifest_sha256"),
        "semantic cache manifest",
    )

    donor_tokenizer = load_local_tokenizer(args.donor_tokenizer_directory)
    target_tokenizer = load_local_tokenizer(args.target_tokenizer_directory)
    bridge = load_strict_tokenizer_bridge_artifact(
        args.bridge_directory,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        expected_artifact_sha256=bridge_sha256,
    )
    if bridge.artifact_sha256 != bridge_sha256:
        raise Lfm2SemanticFoundationFrontierMigrationCliError(
            "Tokenizer bridge differs from its external artifact pin."
        )

    parent_final_norm = load_lfm2_parent_final_rmsnorm_from_local_safetensors(
        cache,
        args.parent_model_directory,
        tensor_key=args.parent_final_norm_tensor_key,
        epsilon_config_key=args.parent_norm_epsilon_config_key,
    )
    final_norm_sha256 = _digest(
        parent_final_norm.result_sha256,
        "authenticated parent final norm load",
    )
    loaded_table = load_lfm2_mapped_parent_tied_table_from_local_safetensors(
        args.parent_model_directory,
        cache=cache,
        bridge=bridge,
        authenticated_parent_final_norm=parent_final_norm,
        tensor_key=args.parent_tied_tensor_key,
        include_source_lineage_report=True,
    )
    if not isinstance(loaded_table, tuple) or len(loaded_table) != 3:
        raise Lfm2SemanticFoundationFrontierMigrationCliError(
            "Direct-local mapped tied-table load omitted exact source lineage."
        )
    source_table, mapped_receipt_value, source_report_value = loaded_table
    mapped_receipt = _verify_sealed_mapping(
        mapped_receipt_value,
        seal_field="result_sha256",
        name="mapped tied-table receipt",
    )
    source_report = _verify_sealed_mapping(
        source_report_value,
        seal_field="result_sha256",
        name="mapped tied-table direct-local source report",
    )
    if (
        mapped_receipt.get("cache_key") != cache_key
        or mapped_receipt.get("cache_parent_fingerprint")
        != cache.parent_fingerprint
        or mapped_receipt.get("bridge_artifact_sha256") != bridge_sha256
        or mapped_receipt.get("parent_final_norm_load_sha256")
        != final_norm_sha256
        or source_report.get("source_checkpoint_exactly_matches_authenticated_parent")
        is not True
        or source_report.get("parent_model_objects_constructed") != 0
        or source_report.get("parent_model_forward_calls") != 0
        or source_report.get("network_access_performed") is not False
    ):
        raise Lfm2SemanticFoundationFrontierMigrationCliError(
            "Mapped parent tied table lacks exact pinned direct-local lineage."
        )

    migrated = migrate_lfm2_semantic_foundation_interface_statistics_frontier(
        args.legacy_statistics_artifact_directory,
        source_table,
        expected_artifact_sha256=artifact_sha256,
        verify_runtime=True,
        legacy_foundation_receipt=foundation_receipt,
        legacy_foundation_receipt_file_sha256=(
            observed_foundation_file_sha256
        ),
    )
    companion_receipt = _validated_companion_receipt(
        migrated,
        expected_artifact_sha256=artifact_sha256,
        expected_foundation_receipt_sha256=foundation_receipt_sha256,
        expected_foundation_receipt_file_sha256=foundation_receipt_file_sha256,
    )
    companion_sha256 = _digest(
        companion_receipt.get("companion_sha256"),
        "frontier companion canonical receipt",
    )
    companion_document_sha256, companion_file_sha256 = (
        _exclusive_atomic_canonical_json(output, companion_receipt)
    )
    persisted, observed_companion_file_sha256 = _read_pinned_canonical_json(
        output,
        expected_canonical_sha256=companion_sha256,
        expected_file_sha256=companion_file_sha256,
        seal_field="companion_sha256",
        name="persisted frontier companion",
    )
    if dict(persisted) != dict(companion_receipt):
        raise Lfm2SemanticFoundationFrontierMigrationCliError(
            "Persisted companion differs from the authenticated migration result."
        )

    body = {
        "schema": _REPORT_SCHEMA,
        "status": "frontier-companion-persisted",
        "legacy_statistics_artifact_sha256": artifact_sha256,
        "semantic_cache_key": cache_key,
        "semantic_cache_manifest_sha256": cache_manifest_sha256,
        "semantic_cache_parent_fingerprint": cache.parent_fingerprint,
        "bridge_artifact_sha256": bridge_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "legacy_foundation_receipt_sha256": foundation_receipt_sha256,
        "legacy_foundation_receipt_file_sha256": (
            foundation_receipt_file_sha256
        ),
        "parent_final_norm_load_sha256": final_norm_sha256,
        "mapped_parent_tied_table_receipt_sha256": mapped_receipt["result_sha256"],
        "mapped_parent_tied_table_source_sha256": source_report["result_sha256"],
        "companion_output": str(output),
        "companion_canonical_sha256": companion_sha256,
        "companion_document_sha256": companion_document_sha256,
        "companion_file_sha256": observed_companion_file_sha256,
        "frontier_sha256": companion_receipt["frontier_sha256"],
        "frontier_materializable_candidate_indices": companion_receipt[
            "frontier_materializable_candidate_indices"
        ],
        "frontier_whole_response_review_order_candidate_indices": (
            companion_receipt[
                "frontier_whole_response_review_order_candidate_indices"
            ]
        ),
        "numeric_metrics_have_final_intelligence_authority": False,
        "whole_response_review_selects_final_candidate": True,
        "offline_direct_local_only": True,
        "authentication_audit": {
            "complete_semantic_cache_opened_and_verified": True,
            "tokenizer_bridge_opened_and_verified": True,
            "parent_checkpoint_exact_identity_reconstructed": True,
            "legacy_foundation_receipt_read_calls": 1,
            "companion_published_without_replacement": True,
        },
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
    }
    return {**body, "report_sha256": canonical_sha256(body)}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
