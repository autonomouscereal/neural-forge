"""Compact authenticated replay evidence for the LFM2 semantic foundation.

The parent circuit cache is an immutable evidence source, not an efficient
rehydration format.  This module publishes the exact second moment, scalar
metric reductions, and causal endpoint rows needed to replay the foundation's
tied-interface solve without opening any parent trace safetensor.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from neural_forge.hf_tied_causal_interface_anchor_fold import (
    TiedCausalAnchorResidualFold,
    TiedCausalAnchorSplit,
    fit_tied_causal_interface_anchor_residual,
)
from neural_forge.hf_tied_causal_interface_fold import (
    TiedCausalInterfaceFold,
    TiedCausalInterfaceMetricWitness,
    TiedCausalInterfaceSufficientStatistics,
    fit_tied_causal_interface_from_statistics,
)


LFM2_SEMANTIC_FOUNDATION_INTERFACE_STATISTICS_SCHEMA = (
    "neural-forge-lfm2-semantic-foundation-interface-sufficient-statistics-v1"
)
LFM2_SEMANTIC_FOUNDATION_INTERFACE_STATISTICS_STATUS = (
    "authenticated-compact-sufficient-statistics"
)
LFM2_SEMANTIC_FOUNDATION_INTERFACE_STATISTICS_PAYLOAD_SCHEMA = (
    "neural-forge-lfm2-semantic-foundation-interface-statistics-payload-v1"
)
LFM2_SEMANTIC_FOUNDATION_INTERFACE_FRONTIER_COMPANION_SCHEMA = (
    "neural-forge-lfm2-semantic-foundation-interface-frontier-companion-v1"
)

_FILES = frozenset({"manifest.json", "statistics.safetensors"})
_SPLITS = ("selection", "fit", "validation")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_TENSOR_HASH_SCHEMA = "dtype-nul-shape-nul-raw-v1"
_TENSOR_NAMES = frozenset(
    {
        "base.fit_second_moment_upper",
        "base.fit_metric_f64",
        "base.fit_metric_i64",
        "base.validation_metric_f64",
        "base.validation_metric_i64",
        "anchor.source_rows_bfloat16",
        "anchor.prefix_positions",
        "anchor.target_next_token_ids",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "status",
        "artifact_sha256",
        "payload_file",
        "payload_file_sha256",
        "payload_file_bytes",
        "payload_tensor_hash_schema",
        "payload_tensors",
        "base_statistics",
        "covariance_packing",
        "anchor_splits",
        "anchor_policy",
        "bindings",
        "runtime",
        "implementation",
        "operation_audit",
    }
)
_BINDING_FIELDS = frozenset(
    {
        "foundation_receipt_sha256",
        "foundation_receipt_file_sha256",
        "interface_solve_sha256",
        "compiler_cache_key",
        "cache_parent_fingerprint",
        "cache_manifest_sha256",
        "bridge_artifact_sha256",
        "trajectory_corpus_sha256",
        "trace_adapter_sha256",
        "selective_catalog_preparation_receipt_sha256",
        "selective_catalog_binding_sha256",
        "selective_catalog_manifest_sha256",
        "selective_catalog_artifact_sha256",
        "final_norm_authentication_sha256",
        "source_tied_table_sha256",
        "target_hidden_width",
        "target_norm_epsilon",
        "base_interface_report_sha256",
        "causal_anchor_report_sha256",
        "active_interface_report_sha256",
        "active_target_weight_sha256",
        "producer_encoder_sha256",
    }
)
_ANCHOR_CALL_FIELDS = frozenset(
    {
        "ridge_candidates",
        "margin_candidates",
        "maximum_relative_table_drift",
        "target_norm_epsilon",
        "metric_chunk_rows",
        "maximum_fit_anchor_rows",
        "minimum_mapped_parent_behavior_retention",
    }
)
_ANCHOR_POLICY_SEMANTIC_FIELDS = frozenset(
    {
        "minimum_mapped_parent_behavior_retention_is_diagnostic_review_goal",
        "minimum_mapped_parent_behavior_retention_used_as_candidate_eligibility_gate",
        "maximum_relative_table_drift_is_diagnostic_review_goal",
        "maximum_relative_table_drift_used_as_candidate_eligibility_gate",
        "base_control_used_only_for_structural_fallback",
        "structural_safety_is_only_candidate_veto",
    }
)
_ANCHOR_POLICY_FIELDS = _ANCHOR_CALL_FIELDS | _ANCHOR_POLICY_SEMANTIC_FIELDS
_FRONTIER_ONLY_ANCHOR_REPORT_FIELDS = frozenset(
    {
        "anchor_frontier",
        "anchor_frontier_sha256",
        "candidate_recipe",
        "candidate_recipe_sha256",
        "compatibility_default_candidate_index",
        "materializable_candidate_indices",
        "whole_response_review_order_candidate_indices",
        "all_structurally_valid_candidates_retained",
        "numeric_metrics_used_for_candidate_admission",
        "numeric_metrics_used_for_review_order_only",
        "numeric_metrics_have_final_intelligence_authority",
        "whole_response_review_selects_final_candidate",
        "lazy_candidate_materialization_supported",
        "selection_split_selects_compatibility_default_only",
        "selection_split_discards_nondefault_structural_candidates",
        "structurally_valid_candidate_indices",
        "maximum_fit_anchor_rows_is_per_solver_batch_scheduling_bound",
        "maximum_fit_anchor_rows_is_legacy_diagnostic_only",
        "global_solution_independent_of_maximum_fit_anchor_rows",
        "maximum_fit_anchor_rows_used_as_evidence_rejection_gate",
        "fit_anchor_rows_scheduled",
        "fit_anchor_rows_rejected",
        "fit_solver_batch_count",
    }
)
_FRONTIER_ONLY_CANDIDATE_REPORT_FIELDS = frozenset(
    {
        "fit_solver_schedule",
        "fit_solver_batch_count",
        "fit_solver_batch_row_counts",
        "maximum_fit_anchor_rows_per_solver_batch",
        "fit_rows_rejected_by_solver_batch_bound",
        "fit_solver_batch_bound_is_resource_scheduling_only",
        "maximum_fit_anchor_rows_is_legacy_diagnostic_only",
        "global_solution_independent_of_maximum_fit_anchor_rows",
    }
)
_LEGACY_ANCHOR_CANDIDATE_SELECTION_POLICY = (
    "all_finite_dense_shape_dtype_device_correct_effectful_analytic_"
    "residuals_eligible_then_deterministic_diagnostic_ordering_by_"
    "parent_raw_centered_energy_parent_top1_base_energy_and_drift_"
    "with_base_only_if_all_analytic_candidates_invalid_or_identical"
)


class Lfm2SemanticFoundationStatisticsError(ValueError):
    """A sufficient-statistics artifact is malformed or unauthenticated."""


@dataclass(frozen=True, slots=True)
class Lfm2SemanticFoundationInterfaceStatisticsArtifact:
    path: Path
    artifact_sha256: str
    manifest_file_sha256: str
    base_statistics: TiedCausalInterfaceSufficientStatistics
    anchor_splits: Mapping[str, TiedCausalAnchorSplit]
    anchor_policy: Mapping[str, Any]
    bindings: Mapping[str, Any]
    runtime: Mapping[str, Any]
    manifest: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Lfm2SemanticFoundationInterfaceStatisticsReplay:
    interface: TiedCausalInterfaceFold
    anchor_fold: TiedCausalAnchorResidualFold
    solve_report: Mapping[str, Any]
    replay_report: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Lfm2SemanticFoundationInterfaceFrontierCompanion:
    """A frontier-aware derivative of one immutable legacy foundation receipt."""

    interface: TiedCausalInterfaceFold
    anchor_fold: TiedCausalAnchorResidualFold
    legacy_solve_report: Mapping[str, Any]
    companion_receipt: Mapping[str, Any]


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2SemanticFoundationStatisticsError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _canonical_file_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(dict(value)) + b"\n"


def _verify_sealed(value: Any, *, field: str, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2SemanticFoundationStatisticsError(f"{name} is not a mapping.")
    body = dict(value)
    claimed = _digest(body.pop(field, None), f"{name} {field}")
    if canonical_sha256(body) != claimed:
        raise Lfm2SemanticFoundationStatisticsError(f"{name} seal is invalid.")
    return {**body, field: claimed}


def _ordinary_directory(path: str | os.PathLike[str], name: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_dir() or candidate.is_symlink():
        raise Lfm2SemanticFoundationStatisticsError(
            f"{name} must be an ordinary directory."
        )
    return candidate.resolve()


def _ordinary_file(path: Path, name: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise Lfm2SemanticFoundationStatisticsError(f"{name} must be an ordinary file.")
    return path.resolve()


def _read_canonical_manifest(path: Path) -> tuple[dict[str, Any], str]:
    path = _ordinary_file(path, "statistics manifest")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2SemanticFoundationStatisticsError(
            "Could not parse the statistics manifest."
        ) from exc
    if (
        len(raw) > _MAX_MANIFEST_BYTES
        or not isinstance(value, dict)
        or raw != _canonical_file_bytes(value)
    ):
        raise Lfm2SemanticFoundationStatisticsError(
            "Statistics manifest is not bounded canonical JSON."
        )
    return value, hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> tuple[int, str]:
    size = 0
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise Lfm2SemanticFoundationStatisticsError(
            f"Could not hash {path.name}."
        ) from exc
    return size, digest.hexdigest()


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            [int(size) for size in tensor.shape],
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _foundation_tensor_sha256(value: Any) -> str:
    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
    }
    digest = hashlib.sha256(canonical_json_bytes(descriptor))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tensor_descriptor(name: str, value: Any) -> dict[str, Any]:
    tensor = value.detach().to(device="cpu").contiguous()
    return {
        "name": name,
        "dtype": str(tensor.dtype),
        "shape": [int(size) for size in tensor.shape],
        "numel": int(tensor.numel()),
        "size_bytes": int(tensor.numel() * tensor.element_size()),
        "tensor_sha256": _tensor_sha256(tensor),
    }


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for semantic-foundation statistics."
        ) from exc
    return torch


def _runtime_fingerprint() -> dict[str, Any]:
    torch = _torch()
    config = torch.__config__.show()
    return {
        "schema": "neural-forge-fp64-solver-runtime-v1",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "byteorder": sys.byteorder,
        "torch_version": str(torch.__version__),
        "torch_git_version": str(getattr(torch.version, "git_version", "")),
        "torch_build_config_sha256": hashlib.sha256(config.encode("utf-8")).hexdigest(),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "torch_deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "torch_default_dtype": str(torch.get_default_dtype()),
        "solver_device": "cpu",
        "solver_dtype": "torch.float64",
    }


def _implementation_fingerprint() -> dict[str, Any]:
    sources = {
        "semantic_foundation_statistics": Path(__file__).resolve(),
        "tied_causal_interface": Path(
            inspect.getsourcefile(fit_tied_causal_interface_from_statistics) or ""
        ).resolve(),
        "tied_causal_anchor": Path(
            inspect.getsourcefile(fit_tied_causal_interface_anchor_residual) or ""
        ).resolve(),
    }
    records = {}
    for name, path in sources.items():
        if not path.is_file() or path.is_symlink():
            raise Lfm2SemanticFoundationStatisticsError(
                f"Statistics implementation source {name!r} is unavailable."
            )
        records[name] = {
            "filename": path.name,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {
        "schema": "neural-forge-foundation-statistics-implementation-v1",
        "sources": records,
        "implementation_sha256": canonical_sha256(records),
    }


def _metric_tensors(
    witness: TiedCausalInterfaceMetricWitness,
) -> tuple[Any, Any]:
    torch = _torch()
    return (
        torch.tensor(
            [
                witness.weighted_metric_rows,
                witness.parent_centered_square,
                witness.raw_error_square,
                witness.normalized_error_square,
            ],
            dtype=torch.float64,
        ),
        torch.tensor(
            [witness.rows, witness.raw_top1, witness.normalized_top1],
            dtype=torch.int64,
        ),
    )


def _metric_witness(float_values: Any, integer_values: Any) -> Any:
    if tuple(float_values.shape) != (4,) or tuple(integer_values.shape) != (3,):
        raise Lfm2SemanticFoundationStatisticsError(
            "Metric witness tensor shapes are invalid."
        )
    return TiedCausalInterfaceMetricWitness(
        rows=int(integer_values[0].item()),
        weighted_metric_rows=float(float_values[0].item()),
        parent_centered_square=float(float_values[1].item()),
        raw_error_square=float(float_values[2].item()),
        normalized_error_square=float(float_values[3].item()),
        raw_top1=int(integer_values[1].item()),
        normalized_top1=int(integer_values[2].item()),
    )


def _pack_second_moment(value: Any) -> Any:
    torch = _torch()
    width = int(value.shape[0])
    indices = torch.triu_indices(width, width)
    return value[indices[0], indices[1]].contiguous()


def _unpack_second_moment(value: Any, *, width: int) -> Any:
    torch = _torch()
    expected = width * (width + 1) // 2
    if value.dtype != torch.float64 or tuple(value.shape) != (expected,):
        raise Lfm2SemanticFoundationStatisticsError(
            "Packed fit second moment has an invalid dtype or shape."
        )
    indices = torch.triu_indices(width, width)
    result = torch.zeros((width, width), dtype=torch.float64)
    result[indices[0], indices[1]] = value
    result[indices[1], indices[0]] = value
    return result.contiguous()


def _foundation_bindings(
    foundation_receipt: Mapping[str, Any],
    *,
    foundation_receipt_file_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    receipt = _verify_sealed(
        foundation_receipt,
        field="receipt_sha256",
        name="foundation receipt",
    )
    solve = _verify_sealed(
        receipt.get("interface_solve"),
        field="result_sha256",
        name="foundation interface solve",
    )
    base = _verify_sealed(
        solve.get("base_interface_report"),
        field="report_sha256",
        name="base interface report",
    )
    anchor = _verify_sealed(
        solve.get("causal_anchor_report"),
        field="report_sha256",
        name="causal anchor report",
    )
    active = _verify_sealed(
        solve.get("active_interface_report"),
        field="report_sha256",
        name="active interface report",
    )
    bindings = {
        "foundation_receipt_sha256": receipt["receipt_sha256"],
        "foundation_receipt_file_sha256": _digest(
            foundation_receipt_file_sha256,
            "foundation receipt file SHA-256",
        ),
        "interface_solve_sha256": solve["result_sha256"],
        "compiler_cache_key": solve["compiler_cache_key"],
        "cache_parent_fingerprint": solve["cache_parent_fingerprint"],
        "cache_manifest_sha256": solve["cache_manifest_sha256"],
        "bridge_artifact_sha256": solve["bridge_artifact_sha256"],
        "trajectory_corpus_sha256": solve["trajectory_corpus_sha256"],
        "trace_adapter_sha256": solve["trace_adapter_sha256"],
        "selective_catalog_preparation_receipt_sha256": solve[
            "selective_catalog_preparation_receipt_sha256"
        ],
        "selective_catalog_binding_sha256": solve["selective_catalog_binding_sha256"],
        "selective_catalog_manifest_sha256": solve["selective_catalog_manifest_sha256"],
        "selective_catalog_artifact_sha256": solve["selective_catalog_artifact_sha256"],
        "final_norm_authentication_sha256": solve["final_norm_authentication_sha256"],
        "source_tied_table_sha256": solve["source_tied_table_sha256"],
        "target_hidden_width": solve["target_hidden_width"],
        "target_norm_epsilon": solve["target_norm_epsilon"],
        "base_interface_report_sha256": base["report_sha256"],
        "causal_anchor_report_sha256": anchor["report_sha256"],
        "active_interface_report_sha256": active["report_sha256"],
        "active_target_weight_sha256": solve["active_target_weight_sha256"],
        "producer_encoder_sha256": solve["producer_encoder_sha256"],
    }
    for name, value in bindings.items():
        if name in {"target_hidden_width", "target_norm_epsilon"}:
            continue
        _digest(value, f"statistics binding {name}")
    if set(bindings) != _BINDING_FIELDS:
        raise Lfm2SemanticFoundationStatisticsError(
            "Foundation statistics bindings differ from the immutable schema."
        )
    policy = {name: anchor[name] for name in sorted(_ANCHOR_POLICY_FIELDS)}
    if (
        policy["minimum_mapped_parent_behavior_retention_is_diagnostic_review_goal"]
        is not True
        or policy[
            "minimum_mapped_parent_behavior_retention_used_as_candidate_eligibility_gate"
        ]
        is not False
        or policy["maximum_relative_table_drift_is_diagnostic_review_goal"] is not True
        or policy["maximum_relative_table_drift_used_as_candidate_eligibility_gate"]
        is not False
        or policy["base_control_used_only_for_structural_fallback"] is not True
        or policy["structural_safety_is_only_candidate_veto"] is not True
    ):
        raise Lfm2SemanticFoundationStatisticsError(
            "Foundation anchor policy restored a numeric eligibility gate."
        )
    return bindings, policy, anchor


def _legacy_anchor_report_from_frontier(
    frontier_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a frontier-aware report onto the exact pre-frontier v1 receipt."""

    report = _verify_sealed(
        frontier_report,
        field="report_sha256",
        name="frontier causal-anchor report",
    )
    if (
        report.get("anchor_frontier_sha256") is None
        or report.get("fit_anchor_row_ceiling_satisfied") is not True
        or report.get("fit_solver_batch_count") != 1
    ):
        raise Lfm2SemanticFoundationStatisticsError(
            "Only a single-batch frontier report has an exact legacy projection."
        )
    body = dict(report)
    body.pop("report_sha256")
    for name in _FRONTIER_ONLY_ANCHOR_REPORT_FIELDS:
        body.pop(name, None)
    reports = body.get("candidate_reports")
    if not isinstance(reports, list):
        raise Lfm2SemanticFoundationStatisticsError(
            "Frontier causal-anchor candidate reports are invalid."
        )
    legacy_candidates = []
    for value in reports:
        if not isinstance(value, Mapping):
            raise Lfm2SemanticFoundationStatisticsError(
                "Frontier causal-anchor candidate report is invalid."
            )
        candidate = dict(value)
        for name in _FRONTIER_ONLY_CANDIDATE_REPORT_FIELDS:
            candidate.pop(name, None)
        legacy_candidates.append(candidate)
    body["candidate_reports"] = legacy_candidates
    body["candidate_selection_policy"] = (
        _LEGACY_ANCHOR_CANDIDATE_SELECTION_POLICY
    )
    return {**body, "report_sha256": canonical_sha256(body)}


def _reauthenticate_artifact_splits(
    artifact: Lfm2SemanticFoundationInterfaceStatisticsArtifact,
) -> Mapping[str, TiedCausalAnchorSplit]:
    splits: dict[str, TiedCausalAnchorSplit] = {}
    if set(artifact.anchor_splits) != set(_SPLITS):
        raise Lfm2SemanticFoundationStatisticsError(
            "Statistics artifact anchor split inventory is invalid."
        )
    for name in _SPLITS:
        supplied = artifact.anchor_splits[name]
        if not isinstance(supplied, TiedCausalAnchorSplit):
            raise Lfm2SemanticFoundationStatisticsError(
                "Statistics artifact anchor split type is invalid."
            )
        splits[name] = TiedCausalAnchorSplit(
            name=name,
            row_ids=supplied.row_ids,
            prefix_positions=supplied.prefix_positions,
            source_producer_rows=supplied.source_producer_rows,
            target_next_token_ids=supplied.target_next_token_ids,
            split_sha256=supplied.split_sha256,
        )
    return MappingProxyType(splits)


def _reauthenticate_base_statistics(
    artifact: Lfm2SemanticFoundationInterfaceStatisticsArtifact,
) -> TiedCausalInterfaceSufficientStatistics:
    supplied = artifact.base_statistics
    if not isinstance(supplied, TiedCausalInterfaceSufficientStatistics):
        raise Lfm2SemanticFoundationStatisticsError(
            "Statistics artifact base statistics type is invalid."
        )
    try:
        return TiedCausalInterfaceSufficientStatistics(
            fit_second_moment=supplied.fit_second_moment,
            fit_metric_witness=supplied.fit_metric_witness,
            validation_metric_witness=supplied.validation_metric_witness,
            source_table_sha256=supplied.source_table_sha256,
            fit_producer_rows_sha256=supplied.fit_producer_rows_sha256,
            validation_producer_rows_sha256=(
                supplied.validation_producer_rows_sha256
            ),
            target_width=supplied.target_width,
            preserve_vocabulary_mean=supplied.preserve_vocabulary_mean,
            target_norm_epsilon=supplied.target_norm_epsilon,
            minimum_validation_top1_agreement=(
                supplied.minimum_validation_top1_agreement
            ),
            metric_chunk_rows=supplied.metric_chunk_rows,
            statistics_sha256=supplied.statistics_sha256,
        )
    except (TypeError, ValueError) as exc:
        raise Lfm2SemanticFoundationStatisticsError(
            "Statistics artifact base evidence failed reauthentication."
        ) from exc


def _payload_tensors(
    base_statistics: TiedCausalInterfaceSufficientStatistics,
    anchor_splits: Mapping[str, TiedCausalAnchorSplit],
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = _torch()
    if set(anchor_splits) != set(_SPLITS) or any(
        not isinstance(anchor_splits[name], TiedCausalAnchorSplit) for name in _SPLITS
    ):
        raise Lfm2SemanticFoundationStatisticsError(
            "Exactly selection, fit, and validation anchor splits are required."
        )
    rows = []
    positions = []
    targets = []
    offsets = [0]
    reports: dict[str, Any] = {}
    row_ids: dict[str, Any] = {}
    for name in _SPLITS:
        split = anchor_splits[name]
        compact = split.source_producer_rows.to(dtype=torch.bfloat16)
        if not torch.equal(compact.to(dtype=torch.float64), split.source_producer_rows):
            raise Lfm2SemanticFoundationStatisticsError(
                f"{name} anchor rows are not losslessly representable as bfloat16."
            )
        rows.append(compact.contiguous())
        positions.append(split.prefix_positions)
        targets.append(split.target_next_token_ids)
        offsets.append(offsets[-1] + split.row_count)
        reports[name] = split.as_dict()
        row_ids[name] = list(split.row_ids)
    fit_f64, fit_i64 = _metric_tensors(base_statistics.fit_metric_witness)
    validation_f64, validation_i64 = _metric_tensors(
        base_statistics.validation_metric_witness
    )
    tensors = {
        "base.fit_second_moment_upper": _pack_second_moment(
            base_statistics.fit_second_moment
        ),
        "base.fit_metric_f64": fit_f64,
        "base.fit_metric_i64": fit_i64,
        "base.validation_metric_f64": validation_f64,
        "base.validation_metric_i64": validation_i64,
        "anchor.source_rows_bfloat16": torch.cat(rows, dim=0).contiguous(),
        "anchor.prefix_positions": torch.cat(positions, dim=0).contiguous(),
        "anchor.target_next_token_ids": torch.cat(targets, dim=0).contiguous(),
    }
    anchor_manifest = {
        "split_order": list(_SPLITS),
        "split_offsets": offsets,
        "row_ids": row_ids,
        "split_reports": reports,
        "combined_row_count": offsets[-1],
        "source_width": base_statistics.source_width,
        "stored_source_dtype": "torch.bfloat16",
        "compiled_source_dtype": "torch.float64",
        "bfloat16_to_float64_lossless": True,
    }
    return tensors, anchor_manifest


def _save_payload(path: Path, tensors: Mapping[str, Any]) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "safetensors is required for semantic-foundation statistics."
        ) from exc
    contract = canonical_json_bytes(
        {"schema": LFM2_SEMANTIC_FOUNDATION_INTERFACE_STATISTICS_PAYLOAD_SCHEMA}
    ).decode("ascii")
    try:
        save_file(
            dict(sorted(tensors.items())), str(path), metadata={"contract": contract}
        )
        with path.open("r+b") as handle:
            os.fsync(handle.fileno())
    except (OSError, RuntimeError, ValueError) as exc:
        raise Lfm2SemanticFoundationStatisticsError(
            "Could not write the sufficient-statistics payload."
        ) from exc


def _load_payload(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    path = _ordinary_file(path, "statistics payload")
    expected_size = manifest.get("payload_file_bytes")
    expected_sha = _digest(
        manifest.get("payload_file_sha256"),
        "statistics payload SHA-256",
    )
    size, observed_sha = _file_sha256(path)
    if size != expected_size or observed_sha != expected_sha:
        raise Lfm2SemanticFoundationStatisticsError(
            "Statistics payload differs from its manifest."
        )
    try:
        payload = path.read_bytes()
        if len(payload) < 8:
            raise ValueError("truncated safetensors payload")
        header_size = int.from_bytes(payload[:8], "little", signed=False)
        header = json.loads(payload[8 : 8 + header_size].decode("utf-8"))
        metadata = header.get("__metadata__")
        contract = canonical_json_bytes(
            {"schema": (LFM2_SEMANTIC_FOUNDATION_INTERFACE_STATISTICS_PAYLOAD_SCHEMA)}
        ).decode("ascii")
        if metadata != {"contract": contract}:
            raise ValueError("invalid safetensors contract")
        from safetensors.torch import load

        tensors = load(payload)
    except (
        OSError,
        RuntimeError,
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise Lfm2SemanticFoundationStatisticsError(
            "Statistics payload is not a valid authenticated safetensors file."
        ) from exc
    if set(tensors) != _TENSOR_NAMES:
        raise Lfm2SemanticFoundationStatisticsError(
            "Statistics payload tensor inventory is invalid."
        )
    entries = [_tensor_descriptor(name, tensors[name]) for name in sorted(tensors)]
    if entries != manifest.get("payload_tensors"):
        raise Lfm2SemanticFoundationStatisticsError(
            "Statistics payload tensor descriptors are invalid."
        )
    return tensors


def write_lfm2_semantic_foundation_interface_statistics(
    artifact_store_directory: str | os.PathLike[str],
    *,
    base_statistics: TiedCausalInterfaceSufficientStatistics,
    anchor_splits: Mapping[str, TiedCausalAnchorSplit],
    foundation_receipt: Mapping[str, Any],
    foundation_receipt_file_sha256: str,
) -> Lfm2SemanticFoundationInterfaceStatisticsArtifact:
    """Publish one immutable content-addressed sufficient-statistics artifact."""

    if not isinstance(base_statistics, TiedCausalInterfaceSufficientStatistics):
        raise TypeError(
            "base_statistics must be TiedCausalInterfaceSufficientStatistics."
        )
    root = Path(artifact_store_directory).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    root = _ordinary_directory(root, "statistics artifact store")
    bindings, anchor_policy, expected_anchor = _foundation_bindings(
        foundation_receipt,
        foundation_receipt_file_sha256=foundation_receipt_file_sha256,
    )
    tensors, anchor_manifest = _payload_tensors(base_statistics, anchor_splits)
    if expected_anchor.get("split_inventory") != anchor_manifest["split_reports"]:
        raise Lfm2SemanticFoundationStatisticsError(
            "Anchor splits differ from the foundation receipt."
        )
    staging = Path(tempfile.mkdtemp(prefix=".foundation-statistics-", dir=root))
    published = False
    try:
        payload_path = staging / "statistics.safetensors"
        _save_payload(payload_path, tensors)
        payload_size, payload_sha = _file_sha256(payload_path)
        body = {
            "schema": LFM2_SEMANTIC_FOUNDATION_INTERFACE_STATISTICS_SCHEMA,
            "status": LFM2_SEMANTIC_FOUNDATION_INTERFACE_STATISTICS_STATUS,
            "payload_file": "statistics.safetensors",
            "payload_file_sha256": payload_sha,
            "payload_file_bytes": payload_size,
            "payload_tensor_hash_schema": _TENSOR_HASH_SCHEMA,
            "payload_tensors": [
                _tensor_descriptor(name, tensors[name]) for name in sorted(tensors)
            ],
            "base_statistics": base_statistics.as_dict(),
            "covariance_packing": {
                "method": "row_major_upper_triangle_including_diagonal",
                "source_width": base_statistics.source_width,
                "packed_values": int(
                    base_statistics.source_width
                    * (base_statistics.source_width + 1)
                    // 2
                ),
                "reconstructed_dtype": "torch.float64",
                "exact_bit_symmetric": True,
            },
            "anchor_splits": anchor_manifest,
            "anchor_policy": anchor_policy,
            "bindings": bindings,
            "runtime": _runtime_fingerprint(),
            "implementation": _implementation_fingerprint(),
            "operation_audit": {
                "parent_forward_calls": 0,
                "child_forward_calls": 0,
                "training_steps": 0,
                "optimizer_steps": 0,
                "backward_calls": 0,
                "parameters_pruned": 0,
                "parameters_quantized": 0,
            },
        }
        artifact_sha = canonical_sha256(body)
        manifest = {**body, "artifact_sha256": artifact_sha}
        manifest_path = staging / "manifest.json"
        raw = _canonical_file_bytes(manifest)
        with manifest_path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        destination = root / artifact_sha
        if destination.exists() or destination.is_symlink():
            existing = open_lfm2_semantic_foundation_interface_statistics(
                destination,
                expected_artifact_sha256=artifact_sha,
            )
            if dict(existing.manifest) != manifest:
                raise Lfm2SemanticFoundationStatisticsError(
                    "An existing statistics artifact differs at the same key."
                )
            return existing
        os.replace(staging, destination)
        published = True
        return open_lfm2_semantic_foundation_interface_statistics(
            destination,
            expected_artifact_sha256=artifact_sha,
        )
    finally:
        if not published and staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)


def _open_lfm2_semantic_foundation_interface_statistics(
    artifact_directory: str | os.PathLike[str],
    *,
    expected_artifact_sha256: str,
    verify_runtime: bool = True,
    allow_legacy_implementation_for_frontier_migration: bool = False,
) -> Lfm2SemanticFoundationInterfaceStatisticsArtifact:
    """Open and fully authenticate a compact interface-statistics artifact."""

    expected = _digest(expected_artifact_sha256, "statistics artifact SHA-256")
    directory = _ordinary_directory(artifact_directory, "statistics artifact")
    if {entry.name for entry in directory.iterdir()} != _FILES or any(
        entry.is_symlink() for entry in directory.iterdir()
    ):
        raise Lfm2SemanticFoundationStatisticsError(
            "Statistics artifact file inventory is invalid."
        )
    manifest, manifest_file_sha = _read_canonical_manifest(directory / "manifest.json")
    if set(manifest) != _MANIFEST_FIELDS:
        raise Lfm2SemanticFoundationStatisticsError(
            "Statistics manifest fields differ from the immutable schema."
        )
    authenticated = _verify_sealed(
        manifest,
        field="artifact_sha256",
        name="statistics manifest",
    )
    if (
        authenticated["artifact_sha256"] != expected
        or authenticated.get("schema")
        != LFM2_SEMANTIC_FOUNDATION_INTERFACE_STATISTICS_SCHEMA
        or authenticated.get("status")
        != LFM2_SEMANTIC_FOUNDATION_INTERFACE_STATISTICS_STATUS
        or authenticated.get("payload_file") != "statistics.safetensors"
        or authenticated.get("payload_tensor_hash_schema") != _TENSOR_HASH_SCHEMA
    ):
        raise Lfm2SemanticFoundationStatisticsError(
            "Statistics manifest identity or contract is invalid."
        )
    if (
        set(authenticated.get("bindings", {})) != _BINDING_FIELDS
        or set(authenticated.get("anchor_policy", {})) != _ANCHOR_POLICY_FIELDS
    ):
        raise Lfm2SemanticFoundationStatisticsError(
            "Statistics bindings or anchor policy are invalid."
        )
    anchor_policy = authenticated["anchor_policy"]
    if (
        anchor_policy.get(
            "minimum_mapped_parent_behavior_retention_is_diagnostic_review_goal"
        )
        is not True
        or anchor_policy.get(
            "minimum_mapped_parent_behavior_retention_used_as_candidate_eligibility_gate"
        )
        is not False
        or anchor_policy.get("maximum_relative_table_drift_is_diagnostic_review_goal")
        is not True
        or anchor_policy.get(
            "maximum_relative_table_drift_used_as_candidate_eligibility_gate"
        )
        is not False
        or anchor_policy.get("base_control_used_only_for_structural_fallback")
        is not True
        or anchor_policy.get("structural_safety_is_only_candidate_veto") is not True
    ):
        raise Lfm2SemanticFoundationStatisticsError(
            "Statistics anchor policy restored a numeric eligibility gate."
        )
    if verify_runtime and authenticated.get("runtime") != _runtime_fingerprint():
        raise Lfm2SemanticFoundationStatisticsError(
            "Statistics solver runtime differs from the producing runtime."
        )
    if type(allow_legacy_implementation_for_frontier_migration) is not bool:
        raise TypeError(
            "allow_legacy_implementation_for_frontier_migration must be boolean."
        )
    if (
        authenticated.get("implementation") != _implementation_fingerprint()
        and not allow_legacy_implementation_for_frontier_migration
    ):
        raise Lfm2SemanticFoundationStatisticsError(
            "Statistics solver implementation differs from the producing implementation."
        )
    tensors = _load_payload(directory / "statistics.safetensors", authenticated)
    base_record = authenticated.get("base_statistics")
    if not isinstance(base_record, Mapping):
        raise Lfm2SemanticFoundationStatisticsError(
            "Base sufficient-statistics record is invalid."
        )
    width = base_record.get("source_width")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise Lfm2SemanticFoundationStatisticsError(
            "Base sufficient-statistics source width is invalid."
        )
    covariance = _unpack_second_moment(
        tensors["base.fit_second_moment_upper"],
        width=width,
    )
    fit_witness = _metric_witness(
        tensors["base.fit_metric_f64"],
        tensors["base.fit_metric_i64"],
    )
    validation_witness = _metric_witness(
        tensors["base.validation_metric_f64"],
        tensors["base.validation_metric_i64"],
    )
    statistics = TiedCausalInterfaceSufficientStatistics(
        fit_second_moment=covariance,
        fit_metric_witness=fit_witness,
        validation_metric_witness=validation_witness,
        source_table_sha256=base_record.get("source_table_sha256"),
        fit_producer_rows_sha256=base_record.get("fit_producer_rows_sha256"),
        validation_producer_rows_sha256=base_record.get(
            "validation_producer_rows_sha256"
        ),
        target_width=base_record.get("target_width"),
        preserve_vocabulary_mean=base_record.get("preserve_vocabulary_mean"),
        target_norm_epsilon=base_record.get("target_norm_epsilon"),
        minimum_validation_top1_agreement=base_record.get(
            "minimum_validation_top1_agreement"
        ),
        metric_chunk_rows=base_record.get("metric_chunk_rows"),
        statistics_sha256=base_record.get("statistics_sha256"),
    )
    if statistics.as_dict() != dict(base_record):
        raise Lfm2SemanticFoundationStatisticsError(
            "Base sufficient statistics differ from their manifest."
        )
    anchor_record = authenticated.get("anchor_splits")
    if (
        not isinstance(anchor_record, Mapping)
        or anchor_record.get("split_order") != list(_SPLITS)
        or anchor_record.get("source_width") != width
        or anchor_record.get("stored_source_dtype") != "torch.bfloat16"
        or anchor_record.get("compiled_source_dtype") != "torch.float64"
        or anchor_record.get("bfloat16_to_float64_lossless") is not True
    ):
        raise Lfm2SemanticFoundationStatisticsError("Anchor split manifest is invalid.")
    offsets = anchor_record.get("split_offsets")
    row_ids = anchor_record.get("row_ids")
    reports = anchor_record.get("split_reports")
    source_rows = tensors["anchor.source_rows_bfloat16"]
    positions = tensors["anchor.prefix_positions"]
    targets = tensors["anchor.target_next_token_ids"]
    if (
        not isinstance(offsets, list)
        or len(offsets) != len(_SPLITS) + 1
        or offsets[0] != 0
        or offsets != sorted(offsets)
        or offsets[-1] != int(source_rows.shape[0])
        or offsets[-1] != int(positions.numel())
        or offsets[-1] != int(targets.numel())
        or anchor_record.get("combined_row_count") != offsets[-1]
        or not isinstance(row_ids, Mapping)
        or not isinstance(reports, Mapping)
        or set(row_ids) != set(_SPLITS)
        or set(reports) != set(_SPLITS)
    ):
        raise Lfm2SemanticFoundationStatisticsError(
            "Anchor split offsets or row identities are invalid."
        )
    splits: dict[str, TiedCausalAnchorSplit] = {}
    for index, name in enumerate(_SPLITS):
        start, stop = offsets[index], offsets[index + 1]
        split = TiedCausalAnchorSplit(
            name=name,
            row_ids=tuple(row_ids[name]),
            prefix_positions=positions[start:stop],
            source_producer_rows=source_rows[start:stop],
            target_next_token_ids=targets[start:stop],
            split_sha256=reports[name]["split_sha256"],
        )
        if split.as_dict() != reports[name]:
            raise Lfm2SemanticFoundationStatisticsError(
                f"{name} anchor split differs from its manifest."
            )
        splits[name] = split
    return Lfm2SemanticFoundationInterfaceStatisticsArtifact(
        path=directory,
        artifact_sha256=expected,
        manifest_file_sha256=manifest_file_sha,
        base_statistics=statistics,
        anchor_splits=MappingProxyType(splits),
        anchor_policy=MappingProxyType(dict(authenticated["anchor_policy"])),
        bindings=MappingProxyType(dict(authenticated["bindings"])),
        runtime=MappingProxyType(dict(authenticated["runtime"])),
        manifest=MappingProxyType(dict(authenticated)),
    )


def open_lfm2_semantic_foundation_interface_statistics(
    artifact_directory: str | os.PathLike[str],
    *,
    expected_artifact_sha256: str,
    verify_runtime: bool = True,
) -> Lfm2SemanticFoundationInterfaceStatisticsArtifact:
    """Open a current artifact; legacy implementations require migration."""

    return _open_lfm2_semantic_foundation_interface_statistics(
        artifact_directory,
        expected_artifact_sha256=expected_artifact_sha256,
        verify_runtime=verify_runtime,
        allow_legacy_implementation_for_frontier_migration=False,
    )


def replay_lfm2_semantic_foundation_interface_statistics(
    artifact: Lfm2SemanticFoundationInterfaceStatisticsArtifact,
    source_selected_tied_table: Any,
    *,
    foundation_receipt: Mapping[str, Any],
    foundation_receipt_file_sha256: str,
) -> Lfm2SemanticFoundationInterfaceStatisticsReplay:
    """Replay and authenticate the exact v1 interface without opening traces."""

    if not isinstance(artifact, Lfm2SemanticFoundationInterfaceStatisticsArtifact):
        raise TypeError(
            "artifact must be Lfm2SemanticFoundationInterfaceStatisticsArtifact."
        )
    bindings, expected_policy, _expected_anchor = _foundation_bindings(
        foundation_receipt,
        foundation_receipt_file_sha256=foundation_receipt_file_sha256,
    )
    if dict(artifact.bindings) != bindings or dict(artifact.anchor_policy) != (
        expected_policy
    ):
        raise Lfm2SemanticFoundationStatisticsError(
            "Statistics artifact differs from the authenticated foundation receipt."
        )
    receipt = _verify_sealed(
        foundation_receipt,
        field="receipt_sha256",
        name="foundation receipt",
    )
    expected_solve = _verify_sealed(
        receipt.get("interface_solve"),
        field="result_sha256",
        name="foundation interface solve",
    )
    table = source_selected_tied_table
    base = fit_tied_causal_interface_from_statistics(
        table,
        artifact.base_statistics,
    )
    anchor = fit_tied_causal_interface_anchor_residual(
        base,
        table,
        selection_split=artifact.anchor_splits["selection"],
        fit_split=artifact.anchor_splits["fit"],
        validation_split=artifact.anchor_splits["validation"],
        **{name: artifact.anchor_policy[name] for name in sorted(_ANCHOR_CALL_FIELDS)},
    )
    interface = anchor.interface
    if (
        dict(base.report) != expected_solve.get("base_interface_report")
        or dict(anchor.report) != expected_solve.get("causal_anchor_report")
        or dict(interface.report) != expected_solve.get("active_interface_report")
        or _foundation_tensor_sha256(table)
        != expected_solve.get("source_tied_table_sha256")
        or _foundation_tensor_sha256(interface.target_weight)
        != expected_solve.get("active_target_weight_sha256")
        or _foundation_tensor_sha256(interface.producer_encoder)
        != expected_solve.get("producer_encoder_sha256")
    ):
        raise Lfm2SemanticFoundationStatisticsError(
            "Replayed interface differs from the authenticated full-cache solve."
        )
    frontier = anchor.frontier
    if (
        frontier is None
        or anchor.report.get("anchor_frontier_sha256")
        != frontier.frontier_sha256
        or anchor.report.get("materializable_candidate_indices")
        != list(frontier.materializable_candidate_indices)
    ):
        raise Lfm2SemanticFoundationStatisticsError(
            "Replayed interface discarded or altered its anchor frontier."
        )
    body = {
        "schema": (
            "neural-forge-lfm2-semantic-foundation-interface-statistics-replay-v1"
        ),
        "statistics_artifact_sha256": artifact.artifact_sha256,
        "statistics_manifest_file_sha256": artifact.manifest_file_sha256,
        "foundation_receipt_sha256": receipt["receipt_sha256"],
        "foundation_receipt_file_sha256": foundation_receipt_file_sha256,
        "interface_solve_sha256": expected_solve["result_sha256"],
        "base_interface_report_sha256": base.report["report_sha256"],
        "causal_anchor_report_sha256": anchor.report["report_sha256"],
        "active_interface_report_sha256": interface.report["report_sha256"],
        "active_target_weight_sha256": _foundation_tensor_sha256(
            interface.target_weight
        ),
        "producer_encoder_sha256": _foundation_tensor_sha256(
            interface.producer_encoder
        ),
        "causal_anchor_frontier_sha256": frontier.frontier_sha256,
        "causal_anchor_candidate_recipe_sha256": [
            recipe.recipe_sha256 for recipe in frontier.recipes
        ],
        "causal_anchor_materializable_candidate_indices": list(
            frontier.materializable_candidate_indices
        ),
        "causal_anchor_compatibility_default_candidate_index": (
            frontier.compatibility_default_candidate_index
        ),
        "causal_anchor_frontier_bound_to_statistics_and_foundation": True,
        "causal_anchor_numeric_metrics_have_final_intelligence_authority": False,
        "causal_anchor_dense_candidate_checkpoints_persisted": 0,
        "compiler_cache_open_calls": 0,
        "trace_safetensor_open_calls": 0,
        "selective_provider_open_calls": 0,
        "parent_forward_calls": 0,
        "child_forward_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "full_cache_solve_report_reproduced_exactly": True,
    }
    report = {**body, "result_sha256": canonical_sha256(body)}
    return Lfm2SemanticFoundationInterfaceStatisticsReplay(
        interface=interface,
        anchor_fold=anchor,
        solve_report=MappingProxyType(dict(expected_solve)),
        replay_report=MappingProxyType(report),
    )


def authenticate_lfm2_semantic_foundation_interface_frontier_companion(
    companion_receipt: Mapping[str, Any],
    *,
    expected_companion_sha256: str,
    artifact: Lfm2SemanticFoundationInterfaceStatisticsArtifact,
    legacy_foundation_receipt: Mapping[str, Any],
    legacy_foundation_receipt_file_sha256: str,
    source_selected_tied_table: Any,
    anchor_fold: TiedCausalAnchorResidualFold | None = None,
) -> Mapping[str, Any]:
    """Authenticate a derived frontier receipt against immutable old evidence."""

    if not isinstance(artifact, Lfm2SemanticFoundationInterfaceStatisticsArtifact):
        raise TypeError(
            "artifact must be Lfm2SemanticFoundationInterfaceStatisticsArtifact."
        )
    bindings, policy, legacy_anchor = _foundation_bindings(
        legacy_foundation_receipt,
        foundation_receipt_file_sha256=(
            legacy_foundation_receipt_file_sha256
        ),
    )
    if dict(artifact.bindings) != bindings or dict(artifact.anchor_policy) != policy:
        raise Lfm2SemanticFoundationStatisticsError(
            "Frontier companion artifact differs from its legacy receipt."
        )
    legacy_receipt = _verify_sealed(
        legacy_foundation_receipt,
        field="receipt_sha256",
        name="legacy foundation receipt",
    )
    legacy_solve = _verify_sealed(
        legacy_receipt.get("interface_solve"),
        field="result_sha256",
        name="legacy foundation interface solve",
    )
    legacy_active = _verify_sealed(
        legacy_solve.get("active_interface_report"),
        field="report_sha256",
        name="legacy active interface report",
    )
    authenticated = _verify_sealed(
        companion_receipt,
        field="companion_sha256",
        name="frontier companion receipt",
    )
    if authenticated["companion_sha256"] != _digest(
        expected_companion_sha256,
        "frontier companion external SHA-256 pin",
    ):
        raise Lfm2SemanticFoundationStatisticsError(
            "Frontier companion differs from its external SHA-256 pin."
        )
    new_anchor = _verify_sealed(
        authenticated.get("frontier_causal_anchor_report"),
        field="report_sha256",
        name="frontier causal-anchor report",
    )
    frontier_record = new_anchor.get("anchor_frontier")
    operation_audit = authenticated.get("operation_audit")
    expected_zero_audit = {
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
    source_sha256 = _foundation_tensor_sha256(source_selected_tied_table)
    legacy_projection = _legacy_anchor_report_from_frontier(new_anchor)
    recipe_hashes = (
        frontier_record.get("candidate_recipe_sha256")
        if isinstance(frontier_record, Mapping)
        else None
    )
    if (
        authenticated.get("schema")
        != LFM2_SEMANTIC_FOUNDATION_INTERFACE_FRONTIER_COMPANION_SCHEMA
        or authenticated.get("status")
        != "frontier-aware-companion-derived-from-legacy-compact-statistics"
        or authenticated.get("statistics_artifact_sha256")
        != artifact.artifact_sha256
        or authenticated.get("statistics_manifest_file_sha256")
        != artifact.manifest_file_sha256
        or authenticated.get("statistics_implementation_sha256")
        != artifact.manifest.get("implementation", {}).get(
            "implementation_sha256"
        )
        or authenticated.get("frontier_implementation_sha256")
        != _implementation_fingerprint()["implementation_sha256"]
        or authenticated.get("legacy_foundation_receipt_sha256")
        != legacy_receipt["receipt_sha256"]
        or authenticated.get("legacy_foundation_receipt_file_sha256")
        != legacy_foundation_receipt_file_sha256
        or authenticated.get("legacy_interface_solve_sha256")
        != legacy_solve["result_sha256"]
        or authenticated.get("legacy_base_interface_report_sha256")
        != legacy_solve.get("base_interface_report_sha256")
        or legacy_solve.get("base_interface_report_sha256")
        != legacy_solve.get("base_interface_report", {}).get("report_sha256")
        or authenticated.get("legacy_causal_anchor_report_sha256")
        != legacy_anchor["report_sha256"]
        or legacy_solve.get("causal_anchor_report_sha256")
        != legacy_anchor["report_sha256"]
        or authenticated.get("legacy_active_interface_report_sha256")
        != legacy_active["report_sha256"]
        or legacy_solve.get("active_interface_report_sha256")
        != legacy_active["report_sha256"]
        or authenticated.get("source_selected_tied_table_sha256") != source_sha256
        or authenticated.get("legacy_active_target_weight_sha256")
        != legacy_solve.get("active_target_weight_sha256")
        or authenticated.get("frontier_active_target_weight_sha256")
        != legacy_solve.get("active_target_weight_sha256")
        or authenticated.get("frontier_causal_anchor_report_sha256")
        != new_anchor["report_sha256"]
        or authenticated.get("legacy_producer_encoder_sha256")
        != legacy_solve.get("producer_encoder_sha256")
        or authenticated.get("frontier_producer_encoder_sha256")
        != legacy_solve.get("producer_encoder_sha256")
        or legacy_projection != legacy_anchor
        or legacy_active != legacy_anchor
        or new_anchor.get("target_weight_sha256")
        != legacy_anchor.get("target_weight_sha256")
        or new_anchor.get("producer_encoder_sha256")
        != legacy_anchor.get("producer_encoder_sha256")
        or not isinstance(frontier_record, Mapping)
        or authenticated.get("frontier_sha256")
        != frontier_record.get("frontier_sha256")
        or authenticated.get("frontier_candidate_recipe_sha256") != recipe_hashes
        or authenticated.get("frontier_materializable_candidate_indices")
        != frontier_record.get("materializable_candidate_indices")
        or authenticated.get("frontier_compatibility_default_candidate_index")
        != frontier_record.get("compatibility_default_candidate_index")
        or authenticated.get("frontier_whole_response_review_order_candidate_indices")
        != frontier_record.get("whole_response_review_order_candidate_indices")
        or authenticated.get("legacy_report_projection_exact") is not True
        or authenticated.get("active_target_tensor_exactly_byte_identical")
        is not True
        or authenticated.get("numeric_metrics_have_final_intelligence_authority")
        is not False
        or authenticated.get("whole_response_review_selects_final_candidate")
        is not True
        or operation_audit != expected_zero_audit
    ):
        raise Lfm2SemanticFoundationStatisticsError(
            "Frontier companion receipt bindings are invalid."
        )
    if anchor_fold is not None:
        if not isinstance(anchor_fold, TiedCausalAnchorResidualFold):
            raise TypeError("anchor_fold must be TiedCausalAnchorResidualFold.")
        frontier = anchor_fold.frontier
        if (
            frontier is None
            or dict(anchor_fold.report) != new_anchor
            or frontier.frontier_sha256
            != authenticated.get("frontier_sha256")
            or _foundation_tensor_sha256(anchor_fold.interface.target_weight)
            != authenticated.get("frontier_active_target_weight_sha256")
            or _foundation_tensor_sha256(anchor_fold.interface.producer_encoder)
            != authenticated.get("frontier_producer_encoder_sha256")
        ):
            raise Lfm2SemanticFoundationStatisticsError(
                "Frontier companion runtime fold differs from its receipt."
            )
    return MappingProxyType(authenticated)


def migrate_lfm2_semantic_foundation_interface_statistics_frontier(
    artifact: (
        Lfm2SemanticFoundationInterfaceStatisticsArtifact
        | str
        | os.PathLike[str]
    ),
    source_selected_tied_table: Any,
    *,
    expected_artifact_sha256: str | None = None,
    verify_runtime: bool = True,
    legacy_foundation_receipt: Mapping[str, Any],
    legacy_foundation_receipt_file_sha256: str,
) -> Lfm2SemanticFoundationInterfaceFrontierCompanion:
    """Derive a full lazy anchor frontier from authenticated legacy statistics."""

    if expected_artifact_sha256 is None:
        raise Lfm2SemanticFoundationStatisticsError(
            "Frontier migration requires an external legacy artifact SHA-256 pin."
        )
    if isinstance(artifact, Lfm2SemanticFoundationInterfaceStatisticsArtifact):
        if expected_artifact_sha256 != artifact.artifact_sha256:
            raise Lfm2SemanticFoundationStatisticsError(
                "Loaded legacy artifact differs from its external SHA-256 pin."
            )
        artifact_directory = artifact.path
    else:
        artifact_directory = artifact
    artifact = _open_lfm2_semantic_foundation_interface_statistics(
        artifact_directory,
        expected_artifact_sha256=expected_artifact_sha256,
        verify_runtime=verify_runtime,
        allow_legacy_implementation_for_frontier_migration=True,
    )
    bindings, policy, legacy_anchor = _foundation_bindings(
        legacy_foundation_receipt,
        foundation_receipt_file_sha256=(
            legacy_foundation_receipt_file_sha256
        ),
    )
    if dict(artifact.bindings) != bindings or dict(artifact.anchor_policy) != policy:
        raise Lfm2SemanticFoundationStatisticsError(
            "Legacy statistics artifact differs from its foundation receipt."
        )
    if legacy_anchor.get("anchor_frontier_sha256") is not None:
        raise Lfm2SemanticFoundationStatisticsError(
            "A frontier-aware receipt must use normal replay, not migration."
        )
    manifest, manifest_file_sha256 = _read_canonical_manifest(
        artifact.path / "manifest.json"
    )
    verified_manifest = _verify_sealed(
        manifest,
        field="artifact_sha256",
        name="statistics manifest",
    )
    if (
        manifest != dict(artifact.manifest)
        or verified_manifest["artifact_sha256"] != artifact.artifact_sha256
        or manifest_file_sha256 != artifact.manifest_file_sha256
        or verified_manifest.get("bindings") != bindings
        or verified_manifest.get("anchor_policy") != policy
    ):
        raise Lfm2SemanticFoundationStatisticsError(
            "Legacy statistics artifact object differs from its sealed manifest."
        )
    statistics = _reauthenticate_base_statistics(artifact)
    splits = _reauthenticate_artifact_splits(artifact)
    legacy_receipt = _verify_sealed(
        legacy_foundation_receipt,
        field="receipt_sha256",
        name="legacy foundation receipt",
    )
    legacy_solve = _verify_sealed(
        legacy_receipt.get("interface_solve"),
        field="result_sha256",
        name="legacy foundation interface solve",
    )
    legacy_base = _verify_sealed(
        legacy_solve.get("base_interface_report"),
        field="report_sha256",
        name="legacy base interface report",
    )
    legacy_active = _verify_sealed(
        legacy_solve.get("active_interface_report"),
        field="report_sha256",
        name="legacy active interface report",
    )
    table = source_selected_tied_table
    base = fit_tied_causal_interface_from_statistics(table, statistics)
    anchor = fit_tied_causal_interface_anchor_residual(
        base,
        table,
        selection_split=splits["selection"],
        fit_split=splits["fit"],
        validation_split=splits["validation"],
        **{name: policy[name] for name in sorted(_ANCHOR_CALL_FIELDS)},
    )
    frontier = anchor.frontier
    new_report = dict(anchor.report)
    legacy_projection = _legacy_anchor_report_from_frontier(new_report)
    source_sha256 = _foundation_tensor_sha256(table)
    active_target_sha256 = _foundation_tensor_sha256(
        anchor.interface.target_weight
    )
    producer_sha256 = _foundation_tensor_sha256(
        anchor.interface.producer_encoder
    )
    if (
        frontier is None
        or dict(base.report) != legacy_base
        or legacy_projection != legacy_anchor
        or legacy_active != legacy_anchor
        or source_sha256 != legacy_solve.get("source_tied_table_sha256")
        or active_target_sha256
        != legacy_solve.get("active_target_weight_sha256")
        or producer_sha256 != legacy_solve.get("producer_encoder_sha256")
    ):
        raise Lfm2SemanticFoundationStatisticsError(
            "Frontier migration changed legacy reports or active tensor bytes."
        )
    operation_audit = {
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
    body = {
        "schema": LFM2_SEMANTIC_FOUNDATION_INTERFACE_FRONTIER_COMPANION_SCHEMA,
        "status": (
            "frontier-aware-companion-derived-from-legacy-compact-statistics"
        ),
        "statistics_artifact_sha256": artifact.artifact_sha256,
        "statistics_manifest_file_sha256": artifact.manifest_file_sha256,
        "statistics_implementation_sha256": artifact.manifest["implementation"][
            "implementation_sha256"
        ],
        "frontier_implementation_sha256": _implementation_fingerprint()[
            "implementation_sha256"
        ],
        "legacy_foundation_receipt_sha256": legacy_receipt["receipt_sha256"],
        "legacy_foundation_receipt_file_sha256": (
            legacy_foundation_receipt_file_sha256
        ),
        "legacy_interface_solve_sha256": legacy_solve["result_sha256"],
        "legacy_base_interface_report_sha256": legacy_base["report_sha256"],
        "legacy_causal_anchor_report_sha256": legacy_anchor["report_sha256"],
        "legacy_active_interface_report_sha256": legacy_active["report_sha256"],
        "source_selected_tied_table_sha256": source_sha256,
        "legacy_active_target_weight_sha256": legacy_solve[
            "active_target_weight_sha256"
        ],
        "legacy_producer_encoder_sha256": legacy_solve[
            "producer_encoder_sha256"
        ],
        "frontier_causal_anchor_report": new_report,
        "frontier_causal_anchor_report_sha256": new_report["report_sha256"],
        "frontier_sha256": frontier.frontier_sha256,
        "frontier_candidate_recipe_sha256": [
            recipe.recipe_sha256 for recipe in frontier.recipes
        ],
        "frontier_materializable_candidate_indices": list(
            frontier.materializable_candidate_indices
        ),
        "frontier_compatibility_default_candidate_index": (
            frontier.compatibility_default_candidate_index
        ),
        "frontier_whole_response_review_order_candidate_indices": list(
            frontier.whole_response_review_order_candidate_indices
        ),
        "frontier_active_target_weight_sha256": active_target_sha256,
        "frontier_producer_encoder_sha256": producer_sha256,
        "legacy_report_projection_exact": True,
        "active_target_tensor_exactly_byte_identical": True,
        "numeric_metrics_have_final_intelligence_authority": False,
        "whole_response_review_selects_final_candidate": True,
        "operation_audit": operation_audit,
    }
    companion = {**body, "companion_sha256": canonical_sha256(body)}
    authenticated = (
        authenticate_lfm2_semantic_foundation_interface_frontier_companion(
            companion,
            expected_companion_sha256=companion["companion_sha256"],
            artifact=artifact,
            legacy_foundation_receipt=legacy_foundation_receipt,
            legacy_foundation_receipt_file_sha256=(
                legacy_foundation_receipt_file_sha256
            ),
            source_selected_tied_table=table,
            anchor_fold=anchor,
        )
    )
    return Lfm2SemanticFoundationInterfaceFrontierCompanion(
        interface=anchor.interface,
        anchor_fold=anchor,
        legacy_solve_report=MappingProxyType(dict(legacy_solve)),
        companion_receipt=authenticated,
    )


__all__ = [
    "LFM2_SEMANTIC_FOUNDATION_INTERFACE_FRONTIER_COMPANION_SCHEMA",
    "LFM2_SEMANTIC_FOUNDATION_INTERFACE_STATISTICS_SCHEMA",
    "LFM2_SEMANTIC_FOUNDATION_INTERFACE_STATISTICS_STATUS",
    "Lfm2SemanticFoundationInterfaceFrontierCompanion",
    "Lfm2SemanticFoundationInterfaceStatisticsArtifact",
    "Lfm2SemanticFoundationInterfaceStatisticsReplay",
    "Lfm2SemanticFoundationStatisticsError",
    "authenticate_lfm2_semantic_foundation_interface_frontier_companion",
    "migrate_lfm2_semantic_foundation_interface_statistics_frontier",
    "open_lfm2_semantic_foundation_interface_statistics",
    "replay_lfm2_semantic_foundation_interface_statistics",
    "write_lfm2_semantic_foundation_interface_statistics",
]
