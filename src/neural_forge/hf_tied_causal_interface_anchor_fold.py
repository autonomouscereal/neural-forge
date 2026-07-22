"""Closed-form causal-anchor residuals for a folded tied vocabulary.

The base CPCF interface is fitted to mapped-parent logits.  Across tokenizer
boundaries that objective can faithfully preserve the *wrong* free-running
decision: the unchanged parent is executing a noncanonical token sequence.
This module adds an analytic residual to the already folded dense tied table
using coherent, teacher-forced next-token anchors from the sealed trajectory
cache.

The producer encoder is fixed.  For every fit observation, the residual is
asked to place the target next token above the strongest mapped-parent
competitor by a candidate margin.  The sparse linear constraints are solved in
observation space with deterministic FP64 ridge regression.  Every finite,
dense, shape/dtype-correct residual that changes the tied table remains
eligible.  Mapped-parent behavior, table drift, and the configured 90% value
are diagnostic guidance for deterministic review ordering, never eligibility
gates.  Every structurally valid analytic candidate is retained as an
authenticated deterministic recipe and can be materialized lazily.  The
historical metric-ranked candidate remains only the compatibility default; it
has no final intelligence authority.  Target-anchor metrics remain diagnostics.
Validation is evaluated only after that compatibility default is frozen.

No optimizer, backward pass, parent execution, pruning, quantization, or
additional runtime parameter is used.  The result replaces a ``V x H`` tied
table with another ``V x H`` tied table.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.hf_tied_causal_interface_fold import (
    TiedCausalInterfaceFold,
)


TIED_CAUSAL_ANCHOR_SPLIT_SCHEMA = "neural-forge-tied-causal-anchor-split-v1"
TIED_CAUSAL_ANCHOR_RESIDUAL_SCHEMA = "neural-forge-tied-causal-anchor-residual-v1"
TIED_CAUSAL_ANCHOR_CANDIDATE_RECIPE_SCHEMA = (
    "neural-forge-tied-causal-anchor-candidate-recipe-v1"
)
TIED_CAUSAL_ANCHOR_FRONTIER_SCHEMA = "neural-forge-tied-causal-anchor-frontier-v1"
_BASE_INTERFACE_SCHEMA = "neural-forge-tied-causal-interface-fold-v1"
_SPLITS = ("selection", "fit", "validation")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class TiedCausalAnchorFoldError(ValueError):
    """The supplied causal anchors cannot produce a safe tied-table fold."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for tied causal-anchor folding."
        ) from exc
    return torch


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
        raise TiedCausalAnchorFoldError(
            "Causal-anchor metadata must be finite canonical JSON."
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TiedCausalAnchorFoldError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
    }
    digest = hashlib.sha256(_canonical_json(descriptor))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _interface_tensor_sha256(value: Any) -> str:
    """Match the tensor seal used by the base tied-interface report."""

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


def _verified_report(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TiedCausalAnchorFoldError(f"{name} must be a report mapping.")
    report = dict(value)
    body = dict(report)
    seal = body.pop("report_sha256", None)
    if _digest(seal, f"{name}.report_sha256") != _sha(body):
        raise TiedCausalAnchorFoldError(f"{name} report seal is invalid.")
    return report


def _verified_base_interface_report(
    value: Any,
    *,
    base_weight: Any,
    producer_encoder: Any,
    source_table: Any,
) -> dict[str, Any]:
    report = _verified_report(value, "base_interface")
    expected = {
        "schema": _BASE_INTERFACE_SCHEMA,
        "source_width": int(producer_encoder.shape[0]),
        "target_width": int(producer_encoder.shape[1]),
        "vocabulary_size": int(base_weight.shape[0]),
        "target_weight_sha256": _interface_tensor_sha256(base_weight),
        "producer_encoder_sha256": _interface_tensor_sha256(producer_encoder),
        "source_table_sha256": _interface_tensor_sha256(source_table),
    }
    if any(
        report.get(name) != expected_value for name, expected_value in expected.items()
    ):
        raise TiedCausalAnchorFoldError(
            "Base interface report does not authenticate its source table, "
            "target table, producer encoder, or dimensions."
        )
    return report


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TiedCausalAnchorFoldError(f"{name} must be a positive integer.")
    return value


def _fraction(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise TiedCausalAnchorFoldError(f"{name} must be a finite fraction.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TiedCausalAnchorFoldError(f"{name} must be a finite fraction.") from exc
    lower_ok = result > 0.0 if positive else result >= 0.0
    if not math.isfinite(result) or not lower_ok or result > 1.0:
        qualifier = "within (0, 1]" if positive else "within [0, 1]"
        raise TiedCausalAnchorFoldError(f"{name} must be {qualifier}.")
    return result


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TiedCausalAnchorFoldError(f"{name} must be positive and finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TiedCausalAnchorFoldError(f"{name} must be positive and finite.") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise TiedCausalAnchorFoldError(f"{name} must be positive and finite.")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TiedCausalAnchorFoldError(f"{name} must be nonnegative and finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TiedCausalAnchorFoldError(
            f"{name} must be nonnegative and finite."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise TiedCausalAnchorFoldError(f"{name} must be nonnegative and finite.")
    return result


def _float_matrix(
    value: Any,
    name: str,
    *,
    width: int | None = None,
) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 2
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
    ):
        raise TiedCausalAnchorFoldError(
            f"{name} must be a nonempty dense real floating matrix."
        )
    if width is not None and int(value.shape[1]) != width:
        raise TiedCausalAnchorFoldError(f"{name} width must equal {width}.")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise TiedCausalAnchorFoldError(f"{name} contains NaN or infinity.")
    return result.clone()


def _integer_vector(value: Any, name: str) -> Any:
    torch = _torch()
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 1
        or value.numel() == 0
        or value.dtype not in integer_dtypes
    ):
        raise TiedCausalAnchorFoldError(
            f"{name} must be a nonempty dense integer vector."
        )
    result = value.detach().to(device="cpu", dtype=torch.long).contiguous()
    if bool((result < 0).any().item()):
        raise TiedCausalAnchorFoldError(f"{name} contains a negative value.")
    return result.clone()


def _rms_normalize(value: Any, epsilon: float) -> Any:
    return value * (value.square().mean(dim=-1, keepdim=True) + epsilon).rsqrt()


@dataclass(frozen=True, slots=True)
class TiedCausalAnchorSplit:
    """One disjoint set of source producer rows and target next-token anchors."""

    name: str
    row_ids: tuple[str, ...]
    prefix_positions: Any
    source_producer_rows: Any
    target_next_token_ids: Any
    split_sha256: str = ""

    def __post_init__(self) -> None:
        if self.name not in _SPLITS:
            raise TiedCausalAnchorFoldError(
                f"Anchor split name must be one of {_SPLITS!r}."
            )
        row_ids = tuple(self.row_ids)
        if (
            not row_ids
            or any(not isinstance(value, str) or not value for value in row_ids)
            or len(set(row_ids)) != len(row_ids)
        ):
            raise TiedCausalAnchorFoldError(
                "Anchor row IDs must be nonempty unique strings."
            )
        positions = _integer_vector(
            self.prefix_positions,
            "prefix_positions",
        )
        targets = _integer_vector(
            self.target_next_token_ids,
            "target_next_token_ids",
        )
        producers = _float_matrix(
            self.source_producer_rows,
            "source_producer_rows",
        )
        rows = len(row_ids)
        if (
            int(positions.numel()) != rows
            or int(targets.numel()) != rows
            or int(producers.shape[0]) != rows
        ):
            raise TiedCausalAnchorFoldError(
                "Anchor split tensors must contain one observation per row ID."
            )
        body = {
            "schema": TIED_CAUSAL_ANCHOR_SPLIT_SCHEMA,
            "name": self.name,
            "row_ids": list(row_ids),
            "prefix_positions_sha256": _tensor_sha256(positions),
            "source_producer_rows_sha256": _tensor_sha256(producers),
            "target_next_token_ids_sha256": _tensor_sha256(targets),
            "row_count": rows,
            "source_width": int(producers.shape[1]),
            "minimum_prefix_position": int(positions.min().item()),
            "maximum_prefix_position": int(positions.max().item()),
        }
        expected = _sha(body)
        if self.split_sha256 and self.split_sha256 != expected:
            raise TiedCausalAnchorFoldError(f"{self.name} split seal is invalid.")
        object.__setattr__(self, "row_ids", row_ids)
        object.__setattr__(self, "prefix_positions", positions)
        object.__setattr__(self, "source_producer_rows", producers)
        object.__setattr__(self, "target_next_token_ids", targets)
        object.__setattr__(self, "split_sha256", expected)

    @property
    def row_count(self) -> int:
        return len(self.row_ids)

    @property
    def source_width(self) -> int:
        return int(self.source_producer_rows.shape[1])

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": TIED_CAUSAL_ANCHOR_SPLIT_SCHEMA,
            "name": self.name,
            "row_count": self.row_count,
            "source_width": self.source_width,
            "minimum_prefix_position": int(self.prefix_positions.min().item()),
            "maximum_prefix_position": int(self.prefix_positions.max().item()),
            "split_sha256": self.split_sha256,
        }


def _split_inventory(
    selection: TiedCausalAnchorSplit,
    fit: TiedCausalAnchorSplit,
    validation: TiedCausalAnchorSplit,
    *,
    source_width: int,
    vocabulary_size: int,
) -> Mapping[str, TiedCausalAnchorSplit]:
    supplied = {
        "selection": selection,
        "fit": fit,
        "validation": validation,
    }
    if any(
        not isinstance(split, TiedCausalAnchorSplit) or split.name != name
        for name, split in supplied.items()
    ):
        raise TiedCausalAnchorFoldError(
            "Selection, fit, and validation split objects are required."
        )
    all_ids: set[str] = set()
    for name in _SPLITS:
        split = supplied[name]
        if split.source_width != source_width:
            raise TiedCausalAnchorFoldError(
                f"{name} producer width differs from the interface."
            )
        if bool((split.target_next_token_ids >= vocabulary_size).any().item()):
            raise TiedCausalAnchorFoldError(
                f"{name} target token exceeds the tied vocabulary."
            )
        overlap = all_ids.intersection(split.row_ids)
        if overlap:
            raise TiedCausalAnchorFoldError(
                f"Anchor split row IDs are not disjoint: {sorted(overlap)!r}."
            )
        all_ids.update(split.row_ids)
    return MappingProxyType(supplied)


def _candidate_values(
    values: Sequence[float],
    name: str,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TiedCausalAnchorFoldError(f"{name} must be a nonempty sequence.")
    result = tuple(sorted({_positive_float(value, name) for value in values}))
    if not result:
        raise TiedCausalAnchorFoldError(f"{name} must be a nonempty sequence.")
    return result


def _coordinates(
    split: TiedCausalAnchorSplit,
    encoder: Any,
    *,
    norm_epsilon: float,
) -> Any:
    return _rms_normalize(
        split.source_producer_rows @ encoder,
        norm_epsilon,
    ).contiguous()


def _mapped_parent_competitors(
    split: TiedCausalAnchorSplit,
    source_table: Any,
    *,
    chunk_rows: int,
) -> Any:
    torch = _torch()
    competitors = []
    for start in range(0, split.row_count, chunk_rows):
        stop = min(split.row_count, start + chunk_rows)
        logits = split.source_producer_rows[start:stop] @ source_table.T
        targets = split.target_next_token_ids[start:stop]
        logits[
            torch.arange(stop - start, dtype=torch.long),
            targets,
        ] = -torch.inf
        competitors.append(logits.argmax(dim=1))
    return torch.cat(competitors).contiguous()


def _stable_finite_vector_norm(value: Any) -> float:
    """Return a finite diagnostic norm for a finite FP64 tensor."""

    torch = _torch()
    scale = float(value.abs().max().item())
    if scale == 0.0:
        return 0.0
    normalized = float(torch.linalg.vector_norm(value / scale).item())
    result = scale * normalized
    return min(result, torch.finfo(torch.float64).max)


def _fit_residual(
    base_weight: Any,
    fit_coordinates: Any,
    target_ids: Any,
    competitor_ids: Any,
    *,
    ridge: float,
    margin: float,
    maximum_relative_table_drift: float,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    base_target = base_weight.index_select(0, target_ids)
    base_competitor = base_weight.index_select(0, competitor_ids)
    base_margins = (fit_coordinates * (base_target - base_competitor)).sum(dim=1)
    requested = torch.full_like(base_margins, margin) - base_margins

    coordinate_gram = fit_coordinates @ fit_coordinates.T
    target_equal = target_ids[:, None] == target_ids[None, :]
    target_competitor = target_ids[:, None] == competitor_ids[None, :]
    competitor_target = competitor_ids[:, None] == target_ids[None, :]
    competitor_equal = competitor_ids[:, None] == competitor_ids[None, :]
    incidence_gram = (
        target_equal.to(dtype=torch.float64)
        - target_competitor.to(dtype=torch.float64)
        - competitor_target.to(dtype=torch.float64)
        + competitor_equal.to(dtype=torch.float64)
    )
    gram = coordinate_gram * incidence_gram
    diagonal_scale = max(
        float(torch.diagonal(gram).mean().item()),
        torch.finfo(torch.float64).eps,
    )
    regularization = ridge * diagonal_scale
    system = (
        gram
        + torch.eye(
            int(gram.shape[0]),
            dtype=torch.float64,
        )
        * regularization
    )
    try:
        coefficients = torch.linalg.solve(system, requested)
    except RuntimeError as exc:
        raise TiedCausalAnchorFoldError(
            "Causal-anchor ridge system could not be solved."
        ) from exc
    updates = coefficients.unsqueeze(1) * fit_coordinates
    residual = torch.zeros_like(base_weight)
    residual.index_add_(0, target_ids, updates)
    residual.index_add_(0, competitor_ids, -updates)
    if not bool(torch.isfinite(residual).all().item()):
        raise TiedCausalAnchorFoldError(
            "Causal-anchor residual contains NaN or infinity."
        )

    base_norm = _stable_finite_vector_norm(base_weight)
    residual_norm = _stable_finite_vector_norm(residual)
    drift_reference_norm = max(base_norm, torch.finfo(torch.float64).eps)
    relative_drift = residual_norm / drift_reference_norm
    if not math.isfinite(relative_drift):
        relative_drift = torch.finfo(torch.float64).max
    repaired = base_weight + residual
    if tuple(repaired.shape) != tuple(base_weight.shape) or not bool(
        torch.isfinite(repaired).all().item()
    ):
        raise TiedCausalAnchorFoldError(
            "Causal-anchor residual violated its shape or finite contract."
        )
    realized = (
        fit_coordinates
        * (
            repaired.index_select(0, target_ids)
            - repaired.index_select(0, competitor_ids)
        )
    ).sum(dim=1)
    report = {
        "ridge_fraction": ridge,
        "ridge_absolute": regularization,
        "requested_margin": margin,
        "fit_constraint_count": int(target_ids.numel()),
        "raw_relative_table_drift": relative_drift,
        "relative_table_drift": relative_drift,
        "maximum_relative_table_drift": maximum_relative_table_drift,
        "maximum_relative_table_drift_is_diagnostic_review_goal": True,
        "maximum_relative_table_drift_used_as_candidate_eligibility_gate": False,
        "maximum_relative_table_drift_used_to_clip_candidate": False,
        "relative_table_drift_review_goal_met": (
            relative_drift <= maximum_relative_table_drift + 1e-12
        ),
        "relative_table_drift_reference_norm": drift_reference_norm,
        "relative_table_drift_reference_used_epsilon": base_norm == 0.0,
        "drift_scale": 1.0,
        "drift_clipped": False,
        "fit_requested_margin_rmse": float(
            torch.sqrt((realized - margin).square().mean()).item()
        ),
        "fit_realized_mean_anchor_competitor_margin": float(realized.mean().item()),
        "fit_realized_minimum_anchor_competitor_margin": float(realized.min().item()),
        "fit_positive_anchor_competitor_margin_fraction": float(
            (realized > 0.0).to(dtype=torch.float64).mean().item()
        ),
        "target_token_ids_sha256": _tensor_sha256(target_ids),
        "competitor_token_ids_sha256": _tensor_sha256(competitor_ids),
        "requested_adjustments_sha256": _tensor_sha256(requested),
        "coefficient_sha256": _tensor_sha256(coefficients),
        "residual_weight_sha256": _tensor_sha256(residual),
        "target_weight_sha256": _tensor_sha256(repaired),
    }
    return residual, report


def _fit_residual_scheduled(
    base_weight: Any,
    fit_coordinates: Any,
    target_ids: Any,
    competitor_ids: Any,
    *,
    ridge: float,
    margin: float,
    maximum_relative_table_drift: float,
    maximum_solver_rows: int,
) -> tuple[Any, dict[str, Any]]:
    """Run one global solve without rejecting evidence at the legacy row ceiling.

    Sequentially solving row blocks changes the analytic problem and makes the
    folded table depend on an operational chunk size.  ``maximum_solver_rows``
    therefore remains authenticated as a legacy diagnostic ceiling only; it
    never truncates evidence and never partitions the mathematical solve.
    """

    row_count = int(target_ids.numel())
    if (
        int(fit_coordinates.shape[0]) != row_count
        or int(competitor_ids.numel()) != row_count
        or row_count <= 0
    ):
        raise TiedCausalAnchorFoldError(
            "Scheduled causal-anchor solve has inconsistent row evidence."
        )
    maximum_rows = _positive_int(maximum_solver_rows, "maximum_solver_rows")
    residual, report = _fit_residual(
        base_weight,
        fit_coordinates,
        target_ids,
        competitor_ids,
        ridge=ridge,
        margin=margin,
        maximum_relative_table_drift=maximum_relative_table_drift,
    )
    report = {
        **report,
        "fit_solver_schedule": "single-global-dense-observation-solve-v1",
        "fit_solver_batch_count": 1,
        "fit_solver_batch_row_counts": [row_count],
        "maximum_fit_anchor_rows_per_solver_batch": maximum_rows,
        "fit_rows_rejected_by_solver_batch_bound": 0,
        "fit_solver_batch_bound_is_resource_scheduling_only": False,
        "maximum_fit_anchor_rows_is_legacy_diagnostic_only": True,
        "global_solution_independent_of_maximum_fit_anchor_rows": True,
    }
    return residual, report


def _residual_candidate_structure(
    base_weight: Any,
    residual: Any,
) -> tuple[Any | None, dict[str, Any]]:
    """Authenticate structural eligibility without consulting performance."""

    torch = _torch()
    is_tensor = isinstance(residual, torch.Tensor)
    dense = bool(is_tensor and residual.layout == torch.strided)
    shape_matches = bool(
        is_tensor and tuple(residual.shape) == tuple(base_weight.shape)
    )
    dtype_matches = bool(is_tensor and residual.dtype == base_weight.dtype)
    device_matches = bool(is_tensor and residual.device == base_weight.device)
    finite = bool(is_tensor and dense and bool(torch.isfinite(residual).all().item()))
    structurally_valid = bool(
        is_tensor
        and dense
        and shape_matches
        and dtype_matches
        and device_matches
        and finite
    )
    repaired = base_weight + residual if structurally_valid else None
    repaired_finite = bool(
        repaired is not None and bool(torch.isfinite(repaired).all().item())
    )
    structurally_valid = structurally_valid and repaired_finite
    tensor_identical = bool(structurally_valid and torch.equal(repaired, base_weight))
    effectful = structurally_valid and not tensor_identical
    if effectful:
        rejection_reason = None
    elif structurally_valid:
        rejection_reason = "tensor_identical_to_base"
    elif not is_tensor:
        rejection_reason = "residual_is_not_tensor"
    elif not dense:
        rejection_reason = "residual_is_not_dense_strided"
    elif not shape_matches:
        rejection_reason = "residual_shape_differs_from_base"
    elif not dtype_matches:
        rejection_reason = "residual_dtype_differs_from_base"
    elif not device_matches:
        rejection_reason = "residual_device_differs_from_base"
    elif not finite:
        rejection_reason = "residual_contains_nonfinite_values"
    else:
        rejection_reason = "repaired_table_contains_nonfinite_values"
    return repaired, {
        "residual_is_tensor": is_tensor,
        "residual_is_dense_strided": dense,
        "residual_shape_matches_base": shape_matches,
        "residual_dtype_matches_base": dtype_matches,
        "residual_device_matches_base": device_matches,
        "residual_values_finite": finite,
        "repaired_values_finite": repaired_finite,
        "structurally_valid": structurally_valid,
        "tensor_identical_to_base": tensor_identical,
        "effectful_residual": effectful,
        "structurally_eligible": effectful,
        "structural_rejection_reason": rejection_reason,
        "numeric_metrics_used_for_structural_eligibility": False,
    }


def _split_metrics(
    split: TiedCausalAnchorSplit,
    source_table: Any,
    encoder: Any,
    target_weight: Any,
    base_target_weight: Any,
    *,
    norm_epsilon: float,
    chunk_rows: int,
) -> dict[str, Any]:
    torch = _torch()
    anchor_top1 = 0
    positive_margin = 0
    mapped_parent_top1_agreement = 0
    mapped_parent_anchor_top1 = 0
    anchor_margin_sum = 0.0
    minimum_anchor_margin = math.inf
    centered_parent_energy = 0.0
    centered_error_energy = 0.0
    raw_centered_error_energy = 0.0
    base_raw_centered_energy = 0.0
    base_raw_centered_error_energy = 0.0
    for start in range(0, split.row_count, chunk_rows):
        stop = min(split.row_count, start + chunk_rows)
        source = split.source_producer_rows[start:stop]
        target_ids = split.target_next_token_ids[start:stop]
        raw_coordinates = source @ encoder
        coordinates = _rms_normalize(
            raw_coordinates,
            norm_epsilon,
        )
        parent_logits = source @ source_table.T
        target_logits = coordinates @ target_weight.T
        raw_target_logits = raw_coordinates @ target_weight.T
        base_raw_logits = raw_coordinates @ base_target_weight.T
        if not bool(
            torch.isfinite(parent_logits).all().item()
            and torch.isfinite(target_logits).all().item()
            and torch.isfinite(raw_target_logits).all().item()
            and torch.isfinite(base_raw_logits).all().item()
        ):
            raise TiedCausalAnchorFoldError(
                f"{split.name} metrics contain NaN or infinity."
            )
        row_indices = torch.arange(stop - start, dtype=torch.long)
        anchor_logits = target_logits[row_indices, target_ids]
        competitor_logits = target_logits.clone()
        competitor_logits[row_indices, target_ids] = -torch.inf
        margins = anchor_logits - competitor_logits.max(dim=1).values
        target_winner = target_logits.argmax(dim=1)
        parent_winner = parent_logits.argmax(dim=1)
        anchor_top1 += int((target_winner == target_ids).sum().item())
        positive_margin += int((margins > 0.0).sum().item())
        mapped_parent_top1_agreement += int(
            (target_winner == parent_winner).sum().item()
        )
        mapped_parent_anchor_top1 += int((parent_winner == target_ids).sum().item())
        anchor_margin_sum += float(margins.sum().item())
        minimum_anchor_margin = min(
            minimum_anchor_margin,
            float(margins.min().item()),
        )
        parent_centered = parent_logits - parent_logits.mean(
            dim=1,
            keepdim=True,
        )
        target_centered = target_logits - target_logits.mean(
            dim=1,
            keepdim=True,
        )
        raw_target_centered = raw_target_logits - raw_target_logits.mean(
            dim=1,
            keepdim=True,
        )
        base_raw_centered = base_raw_logits - base_raw_logits.mean(
            dim=1,
            keepdim=True,
        )
        centered_parent_energy += float(parent_centered.square().sum().item())
        centered_error_energy += float(
            (target_centered - parent_centered).square().sum().item()
        )
        raw_centered_error_energy += float(
            (raw_target_centered - parent_centered).square().sum().item()
        )
        base_raw_centered_energy += float(base_raw_centered.square().sum().item())
        base_raw_centered_error_energy += float(
            (raw_target_centered - base_raw_centered).square().sum().item()
        )
    denominator = max(
        centered_parent_energy,
        torch.finfo(torch.float64).tiny,
    )
    base_denominator = max(
        base_raw_centered_energy,
        torch.finfo(torch.float64).tiny,
    )
    rows = split.row_count
    return {
        "split": split.name,
        "row_count": rows,
        "split_sha256": split.split_sha256,
        "target_anchor_top1_fraction": anchor_top1 / rows,
        "positive_target_anchor_margin_fraction": positive_margin / rows,
        "mean_target_anchor_margin": anchor_margin_sum / rows,
        "minimum_target_anchor_margin": minimum_anchor_margin,
        "mapped_parent_anchor_top1_fraction": (mapped_parent_anchor_top1 / rows),
        "mapped_parent_top1_agreement": (mapped_parent_top1_agreement / rows),
        "mapped_parent_centered_logit_relative_rmse": math.sqrt(
            centered_error_energy / denominator
        ),
        "mapped_parent_centered_logit_retained_energy_fraction": (
            1.0 - centered_error_energy / denominator
        ),
        "mapped_parent_raw_centered_logit_relative_rmse": math.sqrt(
            raw_centered_error_energy / denominator
        ),
        "mapped_parent_raw_centered_logit_retained_energy_fraction": (
            1.0 - raw_centered_error_energy / denominator
        ),
        "base_interface_raw_centered_logit_relative_rmse": math.sqrt(
            base_raw_centered_error_energy / base_denominator
        ),
        "base_interface_raw_centered_logit_retained_energy_fraction": (
            1.0 - base_raw_centered_error_energy / base_denominator
        ),
        "raw_and_runtime_coordinates_have_same_target_top1": True,
        "runtime_target_coordinates_rms_normalized": True,
        "raw_target_coordinates_used_for_parent_behavior_selection": True,
        "raw_target_coordinates_used_for_parent_behavior_diagnostic_ordering": True,
        "parent_behavior_metrics_used_as_candidate_eligibility_gate": False,
        "exact_output_text_or_token_sequence_matching_used": False,
    }


@dataclass(frozen=True, slots=True)
class TiedCausalAnchorCandidateRecipe:
    """Compact authenticated instructions for one structural anchor candidate."""

    candidate_index: int
    candidate_kind: str
    ridge_fraction: float
    requested_margin: float
    base_interface_report_sha256: str
    base_target_weight_sha256: str
    producer_encoder_sha256: str
    source_tied_table_sha256: str
    fit_split_sha256: str
    fit_coordinates_sha256: str
    fit_target_token_ids_sha256: str
    fit_competitor_token_ids_sha256: str
    candidate_report_sha256: str
    original_residual_weight_sha256: str
    original_target_weight_sha256: str
    original_effectful_residual: bool
    original_tensor_identical_to_base: bool
    maximum_relative_table_drift: float
    maximum_fit_anchor_rows_per_solver_batch: int
    recipe_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.candidate_index, bool)
            or not isinstance(self.candidate_index, int)
            or self.candidate_index < 0
        ):
            raise TiedCausalAnchorFoldError(
                "Anchor candidate index must be a nonnegative integer."
            )
        if not isinstance(self.candidate_kind, str) or not self.candidate_kind:
            raise TiedCausalAnchorFoldError(
                "Anchor candidate kind must be nonempty text."
            )
        ridge = _nonnegative_float(self.ridge_fraction, "ridge_fraction")
        margin = _nonnegative_float(self.requested_margin, "requested_margin")
        drift = _fraction(
            self.maximum_relative_table_drift,
            "maximum_relative_table_drift",
            positive=True,
        )
        maximum_rows = _positive_int(
            self.maximum_fit_anchor_rows_per_solver_batch,
            "maximum_fit_anchor_rows_per_solver_batch",
        )
        for name in (
            "base_interface_report_sha256",
            "base_target_weight_sha256",
            "producer_encoder_sha256",
            "source_tied_table_sha256",
            "fit_split_sha256",
            "fit_coordinates_sha256",
            "fit_target_token_ids_sha256",
            "fit_competitor_token_ids_sha256",
            "candidate_report_sha256",
            "original_residual_weight_sha256",
            "original_target_weight_sha256",
        ):
            _digest(getattr(self, name), name)
        if (
            type(self.original_effectful_residual) is not bool
            or type(self.original_tensor_identical_to_base) is not bool
            or self.original_effectful_residual
            is self.original_tensor_identical_to_base
        ):
            raise TiedCausalAnchorFoldError(
                "A structural anchor recipe must record exactly one of "
                "effectful or tensor-identical."
            )
        body = {
            "schema": TIED_CAUSAL_ANCHOR_CANDIDATE_RECIPE_SCHEMA,
            "candidate_index": self.candidate_index,
            "candidate_kind": self.candidate_kind,
            "ridge_fraction": ridge,
            "requested_margin": margin,
            "base_interface_report_sha256": self.base_interface_report_sha256,
            "base_target_weight_sha256": self.base_target_weight_sha256,
            "producer_encoder_sha256": self.producer_encoder_sha256,
            "source_tied_table_sha256": self.source_tied_table_sha256,
            "fit_split_sha256": self.fit_split_sha256,
            "fit_coordinates_sha256": self.fit_coordinates_sha256,
            "fit_target_token_ids_sha256": self.fit_target_token_ids_sha256,
            "fit_competitor_token_ids_sha256": (
                self.fit_competitor_token_ids_sha256
            ),
            "candidate_report_sha256": self.candidate_report_sha256,
            "original_residual_weight_sha256": (
                self.original_residual_weight_sha256
            ),
            "original_target_weight_sha256": self.original_target_weight_sha256,
            "original_effectful_residual": self.original_effectful_residual,
            "original_tensor_identical_to_base": (
                self.original_tensor_identical_to_base
            ),
            "maximum_relative_table_drift": drift,
            "maximum_fit_anchor_rows_per_solver_batch": maximum_rows,
            "fit_solver_schedule": "single-global-dense-observation-solve-v1",
            "maximum_fit_anchor_rows_is_legacy_diagnostic_only": True,
            "global_solution_independent_of_maximum_fit_anchor_rows": True,
            "structurally_valid_candidate_retained": True,
            "numeric_metrics_used_for_recipe_admission": False,
            "original_output_hashes_are_replay_diagnostics_only": True,
            "parent_model_forward_calls": 0,
            "training_steps": 0,
        }
        expected = _sha(body)
        supplied = str(self.recipe_sha256).strip().lower()
        if supplied and supplied != expected:
            raise TiedCausalAnchorFoldError(
                "Anchor candidate recipe authentication is invalid."
            )
        object.__setattr__(self, "ridge_fraction", ridge)
        object.__setattr__(self, "requested_margin", margin)
        object.__setattr__(self, "maximum_relative_table_drift", drift)
        object.__setattr__(
            self,
            "maximum_fit_anchor_rows_per_solver_batch",
            maximum_rows,
        )
        object.__setattr__(self, "recipe_sha256", expected)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": TIED_CAUSAL_ANCHOR_CANDIDATE_RECIPE_SCHEMA,
            "candidate_index": self.candidate_index,
            "candidate_kind": self.candidate_kind,
            "ridge_fraction": self.ridge_fraction,
            "requested_margin": self.requested_margin,
            "base_interface_report_sha256": self.base_interface_report_sha256,
            "base_target_weight_sha256": self.base_target_weight_sha256,
            "producer_encoder_sha256": self.producer_encoder_sha256,
            "source_tied_table_sha256": self.source_tied_table_sha256,
            "fit_split_sha256": self.fit_split_sha256,
            "fit_coordinates_sha256": self.fit_coordinates_sha256,
            "fit_target_token_ids_sha256": self.fit_target_token_ids_sha256,
            "fit_competitor_token_ids_sha256": (
                self.fit_competitor_token_ids_sha256
            ),
            "candidate_report_sha256": self.candidate_report_sha256,
            "original_residual_weight_sha256": (
                self.original_residual_weight_sha256
            ),
            "original_target_weight_sha256": self.original_target_weight_sha256,
            "original_effectful_residual": self.original_effectful_residual,
            "original_tensor_identical_to_base": (
                self.original_tensor_identical_to_base
            ),
            "maximum_relative_table_drift": self.maximum_relative_table_drift,
            "maximum_fit_anchor_rows_per_solver_batch": (
                self.maximum_fit_anchor_rows_per_solver_batch
            ),
            "fit_solver_schedule": "single-global-dense-observation-solve-v1",
            "maximum_fit_anchor_rows_is_legacy_diagnostic_only": True,
            "global_solution_independent_of_maximum_fit_anchor_rows": True,
            "structurally_valid_candidate_retained": True,
            "numeric_metrics_used_for_recipe_admission": False,
            "original_output_hashes_are_replay_diagnostics_only": True,
            "parent_model_forward_calls": 0,
            "training_steps": 0,
            "recipe_sha256": self.recipe_sha256,
        }


@dataclass(frozen=True, slots=True)
class TiedCausalAnchorFrontier:
    """Authenticated complete anchor-recipe bank with lazy materialization."""

    base_interface: TiedCausalInterfaceFold
    source_tied_table_sha256: str
    fit_split_sha256: str
    fit_coordinates: Any
    fit_target_token_ids: Any
    fit_competitor_token_ids: Any
    recipes: tuple[TiedCausalAnchorCandidateRecipe, ...]
    compatibility_default_candidate_index: int
    diagnostic_order_candidate_indices: tuple[int, ...]
    maximum_relative_table_drift: float
    minimum_mapped_parent_behavior_retention: float
    target_norm_epsilon: float
    maximum_fit_anchor_rows_per_solver_batch: int
    frontier_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.base_interface, TiedCausalInterfaceFold):
            raise TypeError("base_interface must be TiedCausalInterfaceFold.")
        base_report = _verified_report(
            self.base_interface.report,
            "anchor_frontier.base_interface",
        )
        base_report_sha256 = _digest(
            base_report.get("report_sha256"),
            "anchor_frontier.base_interface.report_sha256",
        )
        base_target_sha256 = _tensor_sha256(self.base_interface.target_weight)
        producer_sha256 = _tensor_sha256(self.base_interface.producer_encoder)
        if (
            base_report.get("target_weight_sha256")
            != _interface_tensor_sha256(self.base_interface.target_weight)
            or base_report.get("producer_encoder_sha256")
            != _interface_tensor_sha256(self.base_interface.producer_encoder)
        ):
            raise TiedCausalAnchorFoldError(
                "Anchor frontier base interface tensor lineage is invalid."
            )
        source_sha256 = _digest(
            self.source_tied_table_sha256,
            "source_tied_table_sha256",
        )
        fit_split_sha256 = _digest(self.fit_split_sha256, "fit_split_sha256")
        coordinates = _float_matrix(
            self.fit_coordinates,
            "fit_coordinates",
            width=self.base_interface.target_width,
        )
        targets = _integer_vector(self.fit_target_token_ids, "fit_target_token_ids")
        competitors = _integer_vector(
            self.fit_competitor_token_ids,
            "fit_competitor_token_ids",
        )
        if (
            int(coordinates.shape[0]) != int(targets.numel())
            or int(competitors.numel()) != int(targets.numel())
            or bool((targets >= self.base_interface.vocabulary_size).any().item())
            or bool((competitors >= self.base_interface.vocabulary_size).any().item())
        ):
            raise TiedCausalAnchorFoldError(
                "Anchor frontier fit evidence has incompatible geometry."
            )
        recipes = tuple(self.recipes)
        if not recipes or any(
            not isinstance(value, TiedCausalAnchorCandidateRecipe)
            for value in recipes
        ):
            raise TiedCausalAnchorFoldError(
                "Anchor frontier requires authenticated candidate recipes."
            )
        candidate_indices = tuple(value.candidate_index for value in recipes)
        if candidate_indices != tuple(sorted(set(candidate_indices))) or 0 not in set(
            candidate_indices
        ):
            raise TiedCausalAnchorFoldError(
                "Anchor frontier candidate indices must be unique, sorted, and include base."
            )
        evidence_hashes = {
            "fit_coordinates_sha256": _tensor_sha256(coordinates),
            "fit_target_token_ids_sha256": _tensor_sha256(targets),
            "fit_competitor_token_ids_sha256": _tensor_sha256(competitors),
        }
        drift = _fraction(
            self.maximum_relative_table_drift,
            "maximum_relative_table_drift",
            positive=True,
        )
        behavior_goal = _fraction(
            self.minimum_mapped_parent_behavior_retention,
            "minimum_mapped_parent_behavior_retention",
        )
        norm_epsilon = _positive_float(self.target_norm_epsilon, "target_norm_epsilon")
        maximum_rows = _positive_int(
            self.maximum_fit_anchor_rows_per_solver_batch,
            "maximum_fit_anchor_rows_per_solver_batch",
        )
        if any(
            recipe.base_interface_report_sha256 != base_report_sha256
            or recipe.base_target_weight_sha256 != base_target_sha256
            or recipe.producer_encoder_sha256 != producer_sha256
            or recipe.source_tied_table_sha256 != source_sha256
            or recipe.fit_split_sha256 != fit_split_sha256
            or recipe.fit_coordinates_sha256
            != evidence_hashes["fit_coordinates_sha256"]
            or recipe.fit_target_token_ids_sha256
            != evidence_hashes["fit_target_token_ids_sha256"]
            or recipe.fit_competitor_token_ids_sha256
            != evidence_hashes["fit_competitor_token_ids_sha256"]
            or recipe.maximum_relative_table_drift != drift
            or recipe.maximum_fit_anchor_rows_per_solver_batch != maximum_rows
            for recipe in recipes
        ):
            raise TiedCausalAnchorFoldError(
                "Anchor candidate recipe is not bound to the frontier evidence."
            )
        default_index = self.compatibility_default_candidate_index
        if (
            isinstance(default_index, bool)
            or not isinstance(default_index, int)
            or default_index not in set(candidate_indices)
        ):
            raise TiedCausalAnchorFoldError(
                "Anchor compatibility default is not materializable."
            )
        diagnostic_order = tuple(self.diagnostic_order_candidate_indices)
        if (
            len(set(diagnostic_order)) != len(diagnostic_order)
            or any(index not in set(candidate_indices) for index in diagnostic_order)
        ):
            raise TiedCausalAnchorFoldError(
                "Anchor diagnostic review order is invalid."
            )
        review_order = diagnostic_order + tuple(
            index for index in candidate_indices if index not in set(diagnostic_order)
        )
        body = {
            "schema": TIED_CAUSAL_ANCHOR_FRONTIER_SCHEMA,
            "base_interface_report_sha256": base_report_sha256,
            "base_target_weight_sha256": base_target_sha256,
            "producer_encoder_sha256": producer_sha256,
            "source_tied_table_sha256": source_sha256,
            "fit_split_sha256": fit_split_sha256,
            **evidence_hashes,
            "fit_evidence_row_count": int(targets.numel()),
            "candidate_recipes": [value.as_dict() for value in recipes],
            "candidate_recipe_sha256": [value.recipe_sha256 for value in recipes],
            "materializable_candidate_indices": list(candidate_indices),
            "compatibility_default_candidate_index": default_index,
            "diagnostic_order_candidate_indices": list(diagnostic_order),
            "whole_response_review_order_candidate_indices": list(review_order),
            "maximum_relative_table_drift": drift,
            "minimum_mapped_parent_behavior_retention": behavior_goal,
            "target_norm_epsilon": norm_epsilon,
            "maximum_fit_anchor_rows_per_solver_batch": maximum_rows,
            "maximum_fit_anchor_rows_is_legacy_diagnostic_only": True,
            "global_solution_independent_of_maximum_fit_anchor_rows": True,
            "all_structurally_valid_candidates_retained": True,
            "numeric_metrics_used_for_candidate_admission": False,
            "numeric_metrics_used_for_review_order_only": True,
            "numeric_metrics_have_final_intelligence_authority": False,
            "whole_response_review_selects_final_candidate": True,
            "lazy_materialization_supported": True,
            "parent_model_forward_calls": 0,
            "training_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
        }
        expected = _sha(body)
        supplied = str(self.frontier_sha256).strip().lower()
        if supplied and supplied != expected:
            raise TiedCausalAnchorFoldError(
                "Anchor frontier authentication is invalid."
            )
        object.__setattr__(self, "fit_coordinates", coordinates)
        object.__setattr__(self, "fit_target_token_ids", targets)
        object.__setattr__(self, "fit_competitor_token_ids", competitors)
        object.__setattr__(self, "recipes", recipes)
        object.__setattr__(
            self,
            "diagnostic_order_candidate_indices",
            diagnostic_order,
        )
        object.__setattr__(self, "maximum_relative_table_drift", drift)
        object.__setattr__(
            self,
            "minimum_mapped_parent_behavior_retention",
            behavior_goal,
        )
        object.__setattr__(self, "target_norm_epsilon", norm_epsilon)
        object.__setattr__(
            self,
            "maximum_fit_anchor_rows_per_solver_batch",
            maximum_rows,
        )
        object.__setattr__(self, "frontier_sha256", expected)

    @property
    def materializable_candidate_indices(self) -> tuple[int, ...]:
        return tuple(value.candidate_index for value in self.recipes)

    @property
    def whole_response_review_order_candidate_indices(self) -> tuple[int, ...]:
        diagnostic = self.diagnostic_order_candidate_indices
        return diagnostic + tuple(
            index
            for index in self.materializable_candidate_indices
            if index not in set(diagnostic)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": TIED_CAUSAL_ANCHOR_FRONTIER_SCHEMA,
            "base_interface_report_sha256": self.base_interface.report[
                "report_sha256"
            ],
            "base_target_weight_sha256": _tensor_sha256(
                self.base_interface.target_weight
            ),
            "producer_encoder_sha256": _tensor_sha256(
                self.base_interface.producer_encoder
            ),
            "source_tied_table_sha256": self.source_tied_table_sha256,
            "fit_split_sha256": self.fit_split_sha256,
            "fit_coordinates_sha256": _tensor_sha256(self.fit_coordinates),
            "fit_target_token_ids_sha256": _tensor_sha256(
                self.fit_target_token_ids
            ),
            "fit_competitor_token_ids_sha256": _tensor_sha256(
                self.fit_competitor_token_ids
            ),
            "fit_evidence_row_count": int(self.fit_target_token_ids.numel()),
            "candidate_recipes": [value.as_dict() for value in self.recipes],
            "candidate_recipe_sha256": [
                value.recipe_sha256 for value in self.recipes
            ],
            "materializable_candidate_indices": list(
                self.materializable_candidate_indices
            ),
            "compatibility_default_candidate_index": (
                self.compatibility_default_candidate_index
            ),
            "diagnostic_order_candidate_indices": list(
                self.diagnostic_order_candidate_indices
            ),
            "whole_response_review_order_candidate_indices": list(
                self.whole_response_review_order_candidate_indices
            ),
            "maximum_relative_table_drift": self.maximum_relative_table_drift,
            "minimum_mapped_parent_behavior_retention": (
                self.minimum_mapped_parent_behavior_retention
            ),
            "target_norm_epsilon": self.target_norm_epsilon,
            "maximum_fit_anchor_rows_per_solver_batch": (
                self.maximum_fit_anchor_rows_per_solver_batch
            ),
            "maximum_fit_anchor_rows_is_legacy_diagnostic_only": True,
            "global_solution_independent_of_maximum_fit_anchor_rows": True,
            "all_structurally_valid_candidates_retained": True,
            "numeric_metrics_used_for_candidate_admission": False,
            "numeric_metrics_used_for_review_order_only": True,
            "numeric_metrics_have_final_intelligence_authority": False,
            "whole_response_review_selects_final_candidate": True,
            "lazy_materialization_supported": True,
            "parent_model_forward_calls": 0,
            "training_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "frontier_sha256": self.frontier_sha256,
        }

    def enumerate_recipes(
        self,
        *,
        review_order: bool = True,
    ) -> tuple[Mapping[str, Any], ...]:
        if type(review_order) is not bool:
            raise TypeError("review_order must be boolean.")
        by_index = {value.candidate_index: value for value in self.recipes}
        order = (
            self.whole_response_review_order_candidate_indices
            if review_order
            else self.materializable_candidate_indices
        )
        return tuple(MappingProxyType(by_index[index].as_dict()) for index in order)

    def materialize(self, candidate_index: int) -> "TiedCausalAnchorResidualFold":
        if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
            raise TiedCausalAnchorFoldError(
                "Anchor materialization candidate index must be an integer."
            )
        by_index = {value.candidate_index: value for value in self.recipes}
        recipe = by_index.get(candidate_index)
        if recipe is None:
            raise TiedCausalAnchorFoldError(
                f"Anchor candidate {candidate_index} is not materializable."
            )
        torch = _torch()
        base_weight = self.base_interface.target_weight
        if recipe.candidate_index == 0:
            residual = torch.zeros_like(base_weight)
            solver_report: Mapping[str, Any] = MappingProxyType(
                {
                    "fit_solver_schedule": "base-control-no-solve",
                    "fit_solver_batch_count": 0,
                    "fit_rows_rejected_by_solver_batch_bound": 0,
                }
            )
        else:
            residual, solver_report = _fit_residual_scheduled(
                base_weight,
                self.fit_coordinates,
                self.fit_target_token_ids,
                self.fit_competitor_token_ids,
                ridge=recipe.ridge_fraction,
                margin=recipe.requested_margin,
                maximum_relative_table_drift=(
                    recipe.maximum_relative_table_drift
                ),
                maximum_solver_rows=(
                    recipe.maximum_fit_anchor_rows_per_solver_batch
                ),
            )
        target_weight, structure = _residual_candidate_structure(
            base_weight,
            residual,
        )
        if target_weight is None or structure["structurally_valid"] is not True:
            raise TiedCausalAnchorFoldError(
                "Lazy anchor materialization violated structural safety."
            )
        base_report = dict(self.base_interface.report)
        base_report_sha256 = base_report["report_sha256"]
        residual_sha256 = _tensor_sha256(residual)
        target_sha256 = _tensor_sha256(target_weight)
        if (
            residual_sha256 != recipe.original_residual_weight_sha256
            or target_sha256 != recipe.original_target_weight_sha256
        ):
            raise TiedCausalAnchorFoldError(
                "Lazy anchor materialization changed the authenticated candidate bytes."
            )
        frontier = self.as_dict()
        body = {
            "schema": TIED_CAUSAL_ANCHOR_RESIDUAL_SCHEMA,
            "method": "authenticated-lazy-anchor-frontier-recipe-materialization",
            "source_width": self.base_interface.source_width,
            "target_width": self.base_interface.target_width,
            "vocabulary_size": self.base_interface.vocabulary_size,
            "base_dense_tied_parameter_count": int(base_weight.numel()),
            "repaired_dense_tied_parameter_count": int(target_weight.numel()),
            "additional_runtime_parameter_count": 0,
            "dense_same_shape_tied_table": True,
            "base_interface_report_sha256": base_report_sha256,
            "base_interface_lineage": {
                "report": base_report,
                "report_sha256": base_report_sha256,
                "target_weight_sha256": base_report["target_weight_sha256"],
                "producer_encoder_sha256": base_report[
                    "producer_encoder_sha256"
                ],
                "source_table_sha256": base_report["source_table_sha256"],
                "tensor_identity_verified": True,
            },
            "producer_encoder_fixed": True,
            "producer_encoder_sha256": _tensor_sha256(
                self.base_interface.producer_encoder
            ),
            "source_tied_table_sha256": self.source_tied_table_sha256,
            "base_target_weight_sha256": _tensor_sha256(base_weight),
            "residual_weight_sha256": residual_sha256,
            "target_weight_sha256": target_sha256,
            "anchor_frontier": frontier,
            "anchor_frontier_sha256": self.frontier_sha256,
            "candidate_recipe": recipe.as_dict(),
            "candidate_recipe_sha256": recipe.recipe_sha256,
            "selected_candidate_index": recipe.candidate_index,
            "compatibility_default_candidate_index": (
                self.compatibility_default_candidate_index
            ),
            "selected_candidate_is_base_control": recipe.candidate_index == 0,
            "active_anchor_residual_selected": bool(
                structure["effectful_residual"]
            ),
            "selected_ridge_fraction": recipe.ridge_fraction,
            "selected_requested_margin": recipe.requested_margin,
            "original_residual_weight_sha256": (
                recipe.original_residual_weight_sha256
            ),
            "original_target_weight_sha256": recipe.original_target_weight_sha256,
            "materialized_residual_matches_original_hash_diagnostic": (
                True
            ),
            "materialized_target_matches_original_hash_diagnostic": (
                True
            ),
            "original_output_hashes_used_as_admission_gate": False,
            "original_output_hashes_used_as_materialization_authentication": True,
            "materialization_structure": structure,
            "solver_report": dict(solver_report),
            "maximum_relative_table_drift": self.maximum_relative_table_drift,
            "maximum_relative_table_drift_is_diagnostic_review_goal": True,
            "maximum_relative_table_drift_used_as_candidate_eligibility_gate": False,
            "minimum_mapped_parent_behavior_retention": (
                self.minimum_mapped_parent_behavior_retention
            ),
            "minimum_mapped_parent_behavior_retention_is_diagnostic_review_goal": (
                True
            ),
            "minimum_mapped_parent_behavior_retention_used_as_candidate_eligibility_gate": (
                False
            ),
            "base_control_used_only_for_structural_fallback": True,
            "structural_safety_is_only_candidate_veto": True,
            "numeric_thresholds_used_for_candidate_eligibility": False,
            "numeric_metrics_used_for_base_fallback": False,
            "numeric_metrics_used_for_review_order_only": True,
            "numeric_metrics_have_final_intelligence_authority": False,
            "whole_response_review_selects_final_candidate": True,
            "lazy_frontier_materialization": True,
            "operation_audit": {
                "parent_forward_count": 0,
                "parent_generation_count": 0,
                "parent_cache_regeneration_count": 0,
                "target_model_forward_count": 0,
                "training_step_count": 0,
                "optimizer_step_count": 0,
                "backward_call_count": 0,
                "parameters_pruned": 0,
                "parameters_quantized": 0,
            },
            "numeric_parent_weight_matching_objective": False,
            "exact_output_text_or_token_sequence_matching_objective": False,
        }
        report = {**body, "report_sha256": _sha(body)}
        interface = TiedCausalInterfaceFold(
            target_weight=target_weight,
            producer_encoder=self.base_interface.producer_encoder,
            report=report,
        )
        return TiedCausalAnchorResidualFold(
            interface=interface,
            residual_weight=residual,
            report=report,
            frontier=self,
        )


@dataclass(frozen=True, slots=True)
class TiedCausalAnchorResidualFold:
    """A repaired interface plus its same-shaped tied-table residual."""

    interface: TiedCausalInterfaceFold
    residual_weight: Any
    report: Mapping[str, Any]
    frontier: TiedCausalAnchorFrontier | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.interface, TiedCausalInterfaceFold):
            raise TypeError("interface must be TiedCausalInterfaceFold.")
        residual = _float_matrix(
            self.residual_weight,
            "residual_weight",
        )
        if tuple(residual.shape) != tuple(self.interface.target_weight.shape):
            raise TiedCausalAnchorFoldError(
                "Residual and repaired tied-table shapes differ."
            )
        report = dict(self.report)
        seal = report.get("report_sha256")
        body = dict(report)
        body.pop("report_sha256", None)
        lineage = report.get("base_interface_lineage")
        if not isinstance(lineage, Mapping):
            raise TiedCausalAnchorFoldError(
                "Causal-anchor base-interface lineage is missing."
            )
        base_report = _verified_report(
            lineage.get("report"),
            "base_interface_lineage.report",
        )
        base_report_sha256 = _digest(
            base_report.get("report_sha256"),
            "base_interface_lineage.report.report_sha256",
        )
        base_target_sha256 = _digest(
            base_report.get("target_weight_sha256"),
            "base_interface_lineage.report.target_weight_sha256",
        )
        base_producer_sha256 = _digest(
            base_report.get("producer_encoder_sha256"),
            "base_interface_lineage.report.producer_encoder_sha256",
        )
        base_source_sha256 = _digest(
            base_report.get("source_table_sha256"),
            "base_interface_lineage.report.source_table_sha256",
        )
        producer_sha256 = _interface_tensor_sha256(self.interface.producer_encoder)
        if (
            _digest(seal, "report_sha256") != _sha(body)
            or dict(self.interface.report) != report
            or report.get("target_weight_sha256")
            != _tensor_sha256(self.interface.target_weight)
            or report.get("residual_weight_sha256") != _tensor_sha256(residual)
            or report.get("producer_encoder_sha256")
            != _tensor_sha256(self.interface.producer_encoder)
            or report.get("base_interface_report_sha256") != base_report_sha256
            or lineage.get("report_sha256") != base_report_sha256
            or lineage.get("target_weight_sha256") != base_target_sha256
            or lineage.get("producer_encoder_sha256") != base_producer_sha256
            or lineage.get("source_table_sha256") != base_source_sha256
            or lineage.get("producer_encoder_sha256") != producer_sha256
            or base_report.get("schema") != _BASE_INTERFACE_SCHEMA
            or base_report.get("source_width") != self.interface.source_width
            or base_report.get("target_width") != self.interface.target_width
            or base_report.get("vocabulary_size") != self.interface.vocabulary_size
            or lineage.get("tensor_identity_verified") is not True
        ):
            raise TiedCausalAnchorFoldError(
                "Causal-anchor residual report, tensor seal, or base "
                "ancestry is invalid."
            )
        frontier = self.frontier
        if frontier is not None:
            if not isinstance(frontier, TiedCausalAnchorFrontier):
                raise TypeError("frontier must be TiedCausalAnchorFrontier.")
            frontier_report = report.get("anchor_frontier")
            candidate_index = report.get("selected_candidate_index")
            recipes = {
                recipe.candidate_index: recipe for recipe in frontier.recipes
            }
            recipe = recipes.get(candidate_index)
            supplied_recipe = report.get("candidate_recipe")
            candidate_reports = report.get("candidate_reports")
            candidate_report_bound = True
            if candidate_reports is not None:
                candidate_report_bound = False
                if (
                    isinstance(candidate_reports, Sequence)
                    and not isinstance(candidate_reports, (str, bytes))
                    and recipe is not None
                ):
                    matching_reports = [
                        value
                        for value in candidate_reports
                        if isinstance(value, Mapping)
                        and value.get("candidate_index") == candidate_index
                    ]
                    candidate_report_bound = bool(
                        len(matching_reports) == 1
                        and _sha(dict(matching_reports[0]))
                        == recipe.candidate_report_sha256
                    )
            if (
                not isinstance(frontier_report, Mapping)
                or dict(frontier_report) != frontier.as_dict()
                or report.get("anchor_frontier_sha256")
                != frontier.frontier_sha256
                or recipe is None
                or report.get("candidate_recipe_sha256")
                != recipe.recipe_sha256
                or not isinstance(supplied_recipe, Mapping)
                or dict(supplied_recipe) != recipe.as_dict()
                or not candidate_report_bound
                or report.get("compatibility_default_candidate_index")
                != frontier.compatibility_default_candidate_index
                or base_report_sha256
                != frontier.base_interface.report.get("report_sha256")
                or report.get("source_tied_table_sha256")
                != frontier.source_tied_table_sha256
                or producer_sha256
                != _interface_tensor_sha256(
                    frontier.base_interface.producer_encoder
                )
            ):
                raise TiedCausalAnchorFoldError(
                    "Causal-anchor result is not bound to its candidate frontier."
                )
        object.__setattr__(self, "residual_weight", residual)
        object.__setattr__(self, "report", MappingProxyType(report))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)

    def enumerate_candidate_recipes(
        self,
        *,
        review_order: bool = True,
    ) -> tuple[Mapping[str, Any], ...]:
        """Enumerate every retained recipe without allocating target tables."""

        if self.frontier is None:
            raise TiedCausalAnchorFoldError(
                "This compatibility result does not carry an anchor frontier."
            )
        return self.frontier.enumerate_recipes(review_order=review_order)

    def materialize_candidate(
        self,
        candidate_index: int,
    ) -> "TiedCausalAnchorResidualFold":
        """Materialize one retained candidate while leaving all others compact."""

        if self.frontier is None:
            raise TiedCausalAnchorFoldError(
                "This compatibility result does not carry an anchor frontier."
            )
        return self.frontier.materialize(candidate_index)


def fit_tied_causal_interface_anchor_residual(
    base_interface: TiedCausalInterfaceFold,
    source_tied_table: Any,
    *,
    selection_split: TiedCausalAnchorSplit,
    fit_split: TiedCausalAnchorSplit,
    validation_split: TiedCausalAnchorSplit,
    ridge_candidates: Sequence[float] = (1e-8, 1e-6, 1e-4),
    margin_candidates: Sequence[float] = (0.25, 0.5, 1.0),
    maximum_relative_table_drift: float = 0.25,
    target_norm_epsilon: float = 1e-5,
    metric_chunk_rows: int = 256,
    maximum_fit_anchor_rows: int = 4096,
    minimum_mapped_parent_behavior_retention: float = 0.90,
) -> TiedCausalAnchorResidualFold:
    """Fit and diagnostically order same-shape tied-table anchor residuals."""

    torch = _torch()
    if not isinstance(base_interface, TiedCausalInterfaceFold):
        raise TypeError("base_interface must be TiedCausalInterfaceFold.")
    source_table = _float_matrix(
        source_tied_table,
        "source_tied_table",
        width=base_interface.source_width,
    )
    base_weight = _float_matrix(
        base_interface.target_weight,
        "base_interface.target_weight",
        width=base_interface.target_width,
    )
    encoder = _float_matrix(
        base_interface.producer_encoder,
        "base_interface.producer_encoder",
        width=base_interface.target_width,
    )
    if int(source_table.shape[0]) != int(base_weight.shape[0]):
        raise TiedCausalAnchorFoldError(
            "Source and target tied tables must use the same vocabulary order."
        )
    vocabulary_size = int(base_weight.shape[0])
    if vocabulary_size < 2:
        raise TiedCausalAnchorFoldError(
            "Causal-anchor folding requires at least two vocabulary rows."
        )
    base_report = _verified_base_interface_report(
        base_interface.report,
        base_weight=base_weight,
        producer_encoder=encoder,
        source_table=source_table,
    )
    base_report_sha256 = _digest(
        base_report.get("report_sha256"),
        "base_interface.report_sha256",
    )
    splits = _split_inventory(
        selection_split,
        fit_split,
        validation_split,
        source_width=base_interface.source_width,
        vocabulary_size=vocabulary_size,
    )
    ridges = _candidate_values(ridge_candidates, "ridge_candidates")
    margins = _candidate_values(margin_candidates, "margin_candidates")
    drift_limit = _fraction(
        maximum_relative_table_drift,
        "maximum_relative_table_drift",
        positive=True,
    )
    norm_epsilon = _positive_float(
        target_norm_epsilon,
        "target_norm_epsilon",
    )
    chunk_rows = _positive_int(metric_chunk_rows, "metric_chunk_rows")
    maximum_fit_rows = _positive_int(
        maximum_fit_anchor_rows,
        "maximum_fit_anchor_rows",
    )
    behavior_review_goal = _fraction(
        minimum_mapped_parent_behavior_retention,
        "minimum_mapped_parent_behavior_retention",
    )
    fit_coordinates = _coordinates(
        splits["fit"],
        encoder,
        norm_epsilon=norm_epsilon,
    )
    fit_competitors = _mapped_parent_competitors(
        splits["fit"],
        source_table,
        chunk_rows=chunk_rows,
    )
    zero_residual = torch.zeros_like(base_weight)
    base_selection_metrics = _split_metrics(
        splits["selection"],
        source_table,
        encoder,
        base_weight,
        base_weight,
        norm_epsilon=norm_epsilon,
        chunk_rows=chunk_rows,
    )
    base_parent_top1 = float(base_selection_metrics["mapped_parent_top1_agreement"])
    base_parent_energy = float(
        base_selection_metrics[
            "mapped_parent_raw_centered_logit_retained_energy_fraction"
        ]
    )
    diagnostic_goal_parent_top1 = base_parent_top1 * behavior_review_goal
    diagnostic_goal_parent_energy = (
        base_parent_energy * behavior_review_goal
        if base_parent_energy > 0.0
        else base_parent_energy
    )

    def diagnostic_score(
        metrics: Mapping[str, Any],
        *,
        relative_table_drift: float,
    ) -> tuple[float, ...]:
        return (
            float(metrics["mapped_parent_raw_centered_logit_retained_energy_fraction"]),
            float(metrics["mapped_parent_top1_agreement"]),
            float(
                metrics["base_interface_raw_centered_logit_retained_energy_fraction"]
            ),
            -relative_table_drift,
        )

    def behavior_review(
        metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        parent_top1 = float(metrics["mapped_parent_top1_agreement"])
        parent_energy = float(
            metrics["mapped_parent_raw_centered_logit_retained_energy_fraction"]
        )
        base_energy = float(
            metrics["base_interface_raw_centered_logit_retained_energy_fraction"]
        )
        parent_top1_met = parent_top1 + 1e-12 >= diagnostic_goal_parent_top1
        parent_energy_met = parent_energy + 1e-12 >= diagnostic_goal_parent_energy
        base_energy_met = base_energy + 1e-12 >= behavior_review_goal
        return {
            "mapped_parent_behavior_retention_review_goal": behavior_review_goal,
            "diagnostic_goal_mapped_parent_top1_agreement": (
                diagnostic_goal_parent_top1
            ),
            "diagnostic_goal_mapped_parent_raw_centered_logit_retained_energy_fraction": (
                diagnostic_goal_parent_energy
            ),
            "diagnostic_goal_base_interface_raw_centered_logit_retained_energy_fraction": (
                behavior_review_goal
            ),
            "mapped_parent_top1_review_goal_met": parent_top1_met,
            "mapped_parent_raw_centered_energy_review_goal_met": parent_energy_met,
            "base_interface_raw_centered_energy_review_goal_met": base_energy_met,
            "all_mapped_parent_behavior_review_goals_met": (
                parent_top1_met and parent_energy_met and base_energy_met
            ),
            "mapped_parent_behavior_review_goal_used_as_eligibility_gate": False,
        }

    _base_repaired, base_structure = _residual_candidate_structure(
        base_weight,
        zero_residual,
    )
    base_candidate_report = {
        "candidate_index": 0,
        "candidate_kind": "unmodified_base_interface",
        "is_base_control": True,
        "ridge_fraction": 0.0,
        "ridge_absolute": 0.0,
        "requested_margin": 0.0,
        "fit_constraint_count": 0,
        "raw_relative_table_drift": 0.0,
        "relative_table_drift": 0.0,
        "maximum_relative_table_drift": drift_limit,
        "maximum_relative_table_drift_is_diagnostic_review_goal": True,
        "maximum_relative_table_drift_used_as_candidate_eligibility_gate": False,
        "maximum_relative_table_drift_used_to_clip_candidate": False,
        "relative_table_drift_review_goal_met": True,
        "drift_scale": 1.0,
        "drift_clipped": False,
        "residual_weight_sha256": _tensor_sha256(zero_residual),
        "target_weight_sha256": _tensor_sha256(base_weight),
        "selection_metrics": base_selection_metrics,
        "selection_metrics_computed": True,
        **behavior_review(base_selection_metrics),
        **base_structure,
        "base_fallback_eligible": False,
        "candidate_eligible_for_selection": False,
        "selection_diagnostic_score": list(
            diagnostic_score(
                base_selection_metrics,
                relative_table_drift=0.0,
            )
        ),
        "selection_behavior_score": list(
            diagnostic_score(
                base_selection_metrics,
                relative_table_drift=0.0,
            )
        ),
        "target_anchor_metrics_used_for_candidate_selection": False,
        "selection_split_used_for_candidate_parameters": False,
        "validation_split_read": False,
    }
    candidate_reports = [base_candidate_report]
    analytic_candidates = []
    for ridge in ridges:
        for margin in margins:
            candidate_index = len(candidate_reports)
            try:
                residual, fit_report = _fit_residual_scheduled(
                    base_weight,
                    fit_coordinates,
                    splits["fit"].target_next_token_ids,
                    fit_competitors,
                    ridge=ridge,
                    margin=margin,
                    maximum_relative_table_drift=drift_limit,
                    maximum_solver_rows=maximum_fit_rows,
                )
            except TiedCausalAnchorFoldError as exc:
                _repaired, structure = _residual_candidate_structure(
                    base_weight,
                    None,
                )
                candidate_reports.append(
                    {
                        "candidate_index": candidate_index,
                        "candidate_kind": "anchor_residual",
                        "is_base_control": False,
                        "ridge_fraction": ridge,
                        "ridge_absolute": None,
                        "requested_margin": margin,
                        "fit_solver_succeeded": False,
                        "fit_solver_error": str(exc),
                        "selection_metrics": None,
                        "selection_metrics_computed": False,
                        **structure,
                        "base_fallback_eligible": False,
                        "candidate_eligible_for_selection": False,
                        "selection_diagnostic_score": None,
                        "selection_behavior_score": None,
                        "mapped_parent_behavior_retention_review_goal": (
                            behavior_review_goal
                        ),
                        "mapped_parent_behavior_review_goal_used_as_eligibility_gate": (
                            False
                        ),
                        "target_anchor_metrics_used_for_candidate_selection": False,
                        "selection_split_used_for_candidate_parameters": False,
                        "validation_split_read": False,
                    }
                )
                continue
            repaired, structure = _residual_candidate_structure(
                base_weight,
                residual,
            )
            common_report = {
                "candidate_index": candidate_index,
                "candidate_kind": "anchor_residual",
                "is_base_control": False,
                **fit_report,
                "fit_solver_succeeded": True,
                **structure,
                "base_fallback_eligible": False,
                "candidate_eligible_for_selection": bool(
                    structure["structurally_eligible"]
                ),
                "mapped_parent_behavior_retention_review_goal": (behavior_review_goal),
                "mapped_parent_behavior_review_goal_used_as_eligibility_gate": False,
                "target_anchor_metrics_used_for_candidate_selection": False,
                "selection_split_used_for_candidate_parameters": False,
                "validation_split_read": False,
            }
            if not structure["structurally_eligible"]:
                candidate_reports.append(
                    {
                        **common_report,
                        "selection_metrics": None,
                        "selection_metrics_computed": False,
                        "selection_diagnostic_score": None,
                        "selection_behavior_score": None,
                    }
                )
                continue
            assert repaired is not None
            selection_metrics = _split_metrics(
                splits["selection"],
                source_table,
                encoder,
                repaired,
                base_weight,
                norm_epsilon=norm_epsilon,
                chunk_rows=chunk_rows,
            )
            score = diagnostic_score(
                selection_metrics,
                relative_table_drift=float(fit_report["relative_table_drift"]),
            )
            candidate_report = {
                **common_report,
                "selection_metrics": selection_metrics,
                "selection_metrics_computed": True,
                **behavior_review(selection_metrics),
                "selection_diagnostic_score": list(score),
                "selection_behavior_score": list(score),
            }
            candidate_reports.append(candidate_report)
            analytic_candidates.append((score, candidate_index, residual, repaired))
    diagnostic_order = sorted(
        analytic_candidates,
        key=lambda value: (value[0], -value[1]),
        reverse=True,
    )
    if diagnostic_order:
        _score, selected_index, selected_residual, selected_weight = diagnostic_order[0]
    else:
        selected_index = 0
        selected_residual = zero_residual
        selected_weight = base_weight
        base_candidate_report["base_fallback_eligible"] = True
        base_candidate_report["candidate_eligible_for_selection"] = True

    source_tied_table_sha256 = _tensor_sha256(source_table)
    base_target_weight_sha256 = _tensor_sha256(base_weight)
    producer_encoder_sha256 = _tensor_sha256(encoder)
    fit_coordinates_sha256 = _tensor_sha256(fit_coordinates)
    fit_target_token_ids_sha256 = _tensor_sha256(
        splits["fit"].target_next_token_ids
    )
    fit_competitor_token_ids_sha256 = _tensor_sha256(fit_competitors)
    recipes = tuple(
        TiedCausalAnchorCandidateRecipe(
            candidate_index=int(candidate["candidate_index"]),
            candidate_kind=str(candidate["candidate_kind"]),
            ridge_fraction=float(candidate["ridge_fraction"]),
            requested_margin=float(candidate["requested_margin"]),
            base_interface_report_sha256=base_report_sha256,
            base_target_weight_sha256=base_target_weight_sha256,
            producer_encoder_sha256=producer_encoder_sha256,
            source_tied_table_sha256=source_tied_table_sha256,
            fit_split_sha256=splits["fit"].split_sha256,
            fit_coordinates_sha256=fit_coordinates_sha256,
            fit_target_token_ids_sha256=fit_target_token_ids_sha256,
            fit_competitor_token_ids_sha256=(
                fit_competitor_token_ids_sha256
            ),
            candidate_report_sha256=_sha(candidate),
            original_residual_weight_sha256=str(
                candidate["residual_weight_sha256"]
            ),
            original_target_weight_sha256=str(candidate["target_weight_sha256"]),
            original_effectful_residual=bool(candidate["effectful_residual"]),
            original_tensor_identical_to_base=bool(
                candidate["tensor_identical_to_base"]
            ),
            maximum_relative_table_drift=drift_limit,
            maximum_fit_anchor_rows_per_solver_batch=maximum_fit_rows,
        )
        for candidate in candidate_reports
        if candidate["structurally_valid"]
    )
    frontier = TiedCausalAnchorFrontier(
        base_interface=base_interface,
        source_tied_table_sha256=source_tied_table_sha256,
        fit_split_sha256=splits["fit"].split_sha256,
        fit_coordinates=fit_coordinates,
        fit_target_token_ids=splits["fit"].target_next_token_ids,
        fit_competitor_token_ids=fit_competitors,
        recipes=recipes,
        compatibility_default_candidate_index=selected_index,
        diagnostic_order_candidate_indices=tuple(
            int(candidate[1]) for candidate in diagnostic_order
        ),
        maximum_relative_table_drift=drift_limit,
        minimum_mapped_parent_behavior_retention=behavior_review_goal,
        target_norm_epsilon=norm_epsilon,
        maximum_fit_anchor_rows_per_solver_batch=maximum_fit_rows,
    )
    selected_recipe = {
        recipe.candidate_index: recipe for recipe in recipes
    }[selected_index]

    # Validation is intentionally first touched after the winner is frozen.
    split_metrics = {
        name: _split_metrics(
            splits[name],
            source_table,
            encoder,
            selected_weight,
            base_weight,
            norm_epsilon=norm_epsilon,
            chunk_rows=chunk_rows,
        )
        for name in _SPLITS
    }
    selected_report = candidate_reports[selected_index]
    selected_relative_drift = float(selected_report["relative_table_drift"])
    base_parameters = int(base_weight.numel())
    body = {
        "schema": TIED_CAUSAL_ANCHOR_RESIDUAL_SCHEMA,
        "method": (
            "fixed_producer_coordinates_sparse_target_vs_mapped_competitor_"
            "observation_space_ridge"
        ),
        "source_width": base_interface.source_width,
        "target_width": base_interface.target_width,
        "vocabulary_size": vocabulary_size,
        "base_dense_tied_parameter_count": base_parameters,
        "repaired_dense_tied_parameter_count": int(selected_weight.numel()),
        "additional_runtime_parameter_count": 0,
        "dense_same_shape_tied_table": True,
        "base_interface_report_sha256": base_report_sha256,
        "base_interface_lineage": {
            "report": base_report,
            "report_sha256": base_report_sha256,
            "target_weight_sha256": base_report["target_weight_sha256"],
            "producer_encoder_sha256": base_report["producer_encoder_sha256"],
            "source_table_sha256": base_report["source_table_sha256"],
            "tensor_identity_verified": True,
        },
        "producer_encoder_fixed": True,
        "producer_encoder_sha256": producer_encoder_sha256,
        "source_tied_table_sha256": source_tied_table_sha256,
        "base_target_weight_sha256": base_target_weight_sha256,
        "residual_weight_sha256": _tensor_sha256(selected_residual),
        "target_weight_sha256": _tensor_sha256(selected_weight),
        "anchor_frontier": frontier.as_dict(),
        "anchor_frontier_sha256": frontier.frontier_sha256,
        "candidate_recipe": selected_recipe.as_dict(),
        "candidate_recipe_sha256": selected_recipe.recipe_sha256,
        "compatibility_default_candidate_index": selected_index,
        "materializable_candidate_indices": list(
            frontier.materializable_candidate_indices
        ),
        "whole_response_review_order_candidate_indices": list(
            frontier.whole_response_review_order_candidate_indices
        ),
        "all_structurally_valid_candidates_retained": True,
        "numeric_metrics_used_for_candidate_admission": False,
        "numeric_metrics_used_for_review_order_only": True,
        "numeric_metrics_have_final_intelligence_authority": False,
        "whole_response_review_selects_final_candidate": True,
        "lazy_candidate_materialization_supported": True,
        "split_inventory": {name: splits[name].as_dict() for name in _SPLITS},
        "split_row_ids_disjoint": True,
        "fit_split_used_for_residual_parameters": True,
        "selection_split_used_for_residual_parameters": False,
        "selection_split_used_for_ridge_margin_selection": True,
        "selection_split_selects_compatibility_default_only": True,
        "selection_split_discards_nondefault_structural_candidates": False,
        "selection_split_used_for_diagnostic_ordering": True,
        "selection_split_used_for_numeric_eligibility_gating": False,
        "validation_split_used_for_residual_parameters": False,
        "validation_split_used_for_ridge_margin_selection": False,
        "validation_values_used_for_parameters_or_selection": False,
        "validation_metrics_first_computed_after_candidate_frozen": True,
        "ridge_candidates": list(ridges),
        "margin_candidates": list(margins),
        "base_control_candidate_included": True,
        "base_control_candidate_index": 0,
        "base_control_used_only_for_structural_fallback": True,
        "base_control_fallback_triggered": not bool(diagnostic_order),
        "base_control_fallback_reason": (
            "all_analytic_residual_candidates_structurally_invalid_or_tensor_identical"
            if not diagnostic_order
            else None
        ),
        "minimum_mapped_parent_behavior_retention": behavior_review_goal,
        "minimum_mapped_parent_behavior_retention_is_diagnostic_review_goal": True,
        "minimum_mapped_parent_behavior_retention_used_as_candidate_eligibility_gate": (
            False
        ),
        "candidate_selection_policy": (
            "all_finite_dense_shape_dtype_device_correct_effectful_analytic_"
            "residuals_eligible_then_deterministic_diagnostic_ordering_by_"
            "parent_raw_centered_energy_parent_top1_base_energy_and_drift_"
            "with_base_only_if_all_analytic_candidates_invalid_or_identical_"
            "for_compatibility_default_only_all_structural_recipes_retained"
        ),
        "structural_safety_is_only_candidate_veto": True,
        "numeric_metrics_used_for_structural_eligibility": False,
        "numeric_thresholds_used_for_candidate_eligibility": False,
        "numeric_metrics_used_for_base_fallback": False,
        "numeric_metrics_used_for_deterministic_diagnostic_ordering": True,
        "diagnostic_ordering_tie_break": "lowest_candidate_index",
        "diagnostic_candidate_indices_best_to_worst": [
            int(candidate[1]) for candidate in diagnostic_order
        ],
        "target_anchor_metrics_diagnostic_only": True,
        "target_anchor_metrics_used_for_candidate_selection": False,
        "structurally_eligible_candidate_indices": [
            int(candidate["candidate_index"])
            for candidate in candidate_reports
            if candidate["structurally_eligible"]
        ],
        "structurally_valid_candidate_indices": [
            int(candidate["candidate_index"])
            for candidate in candidate_reports
            if candidate["structurally_valid"]
        ],
        "eligible_candidate_indices": [
            int(candidate["candidate_index"])
            for candidate in candidate_reports
            if candidate["candidate_eligible_for_selection"]
        ],
        "structurally_rejected_candidate_indices": [
            int(candidate["candidate_index"])
            for candidate in candidate_reports[1:]
            if not candidate["structurally_valid"]
        ],
        "tensor_identical_candidate_indices": [
            int(candidate["candidate_index"])
            for candidate in candidate_reports[1:]
            if candidate["tensor_identical_to_base"]
        ],
        "candidate_reports": candidate_reports,
        "selected_candidate_index": selected_index,
        "selected_candidate_is_base_control": bool(selected_report["is_base_control"]),
        "active_anchor_residual_selected": not bool(selected_report["is_base_control"]),
        "selected_ridge_fraction": selected_report["ridge_fraction"],
        "selected_requested_margin": selected_report["requested_margin"],
        "relative_table_drift": selected_relative_drift,
        "maximum_relative_table_drift": drift_limit,
        "maximum_relative_table_drift_is_diagnostic_review_goal": True,
        "maximum_relative_table_drift_used_as_candidate_eligibility_gate": False,
        "maximum_relative_table_drift_used_to_clip_candidate": False,
        "relative_table_drift_review_goal_met": (
            selected_relative_drift <= drift_limit + 1e-12
        ),
        "drift_bound_satisfied": (selected_relative_drift <= drift_limit + 1e-12),
        "drift_bound_enforced": False,
        "split_metrics": split_metrics,
        "target_norm_epsilon": norm_epsilon,
        "metric_chunk_rows": chunk_rows,
        "maximum_fit_anchor_rows": maximum_fit_rows,
        "maximum_fit_anchor_rows_is_per_solver_batch_scheduling_bound": False,
        "maximum_fit_anchor_rows_is_legacy_diagnostic_only": True,
        "global_solution_independent_of_maximum_fit_anchor_rows": True,
        "maximum_fit_anchor_rows_used_as_evidence_rejection_gate": False,
        "fit_anchor_row_ceiling_satisfied": (
            splits["fit"].row_count <= maximum_fit_rows
        ),
        "fit_anchor_rows_scheduled": splits["fit"].row_count,
        "fit_anchor_rows_rejected": 0,
        "fit_solver_batch_count": 1,
        "operation_audit": {
            "parent_forward_count": 0,
            "parent_generation_count": 0,
            "parent_cache_regeneration_count": 0,
            "target_model_forward_count": 0,
            "training_step_count": 0,
            "optimizer_step_count": 0,
            "backward_call_count": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
        },
        "numeric_parent_weight_matching_objective": False,
        "mapped_parent_logit_retention_measured": True,
        "target_next_token_anchor_objective": True,
        "exact_output_text_or_token_sequence_matching_objective": False,
    }
    report = {**body, "report_sha256": _sha(body)}
    repaired_interface = TiedCausalInterfaceFold(
        target_weight=selected_weight,
        producer_encoder=encoder,
        report=report,
    )
    return TiedCausalAnchorResidualFold(
        interface=repaired_interface,
        residual_weight=selected_residual,
        report=report,
        frontier=frontier,
    )


def enumerate_tied_causal_anchor_candidates(
    value: TiedCausalAnchorResidualFold | TiedCausalAnchorFrontier,
    *,
    review_order: bool = True,
) -> tuple[Mapping[str, Any], ...]:
    """Enumerate compact anchor recipes without materializing their tables."""

    if isinstance(value, TiedCausalAnchorResidualFold):
        return value.enumerate_candidate_recipes(review_order=review_order)
    if isinstance(value, TiedCausalAnchorFrontier):
        return value.enumerate_recipes(review_order=review_order)
    raise TypeError(
        "value must be TiedCausalAnchorResidualFold or TiedCausalAnchorFrontier."
    )


def materialize_tied_causal_anchor_candidate(
    value: TiedCausalAnchorResidualFold | TiedCausalAnchorFrontier,
    candidate_index: int,
) -> TiedCausalAnchorResidualFold:
    """Materialize one retained anchor candidate for decoded review."""

    if isinstance(value, TiedCausalAnchorResidualFold):
        return value.materialize_candidate(candidate_index)
    if isinstance(value, TiedCausalAnchorFrontier):
        return value.materialize(candidate_index)
    raise TypeError(
        "value must be TiedCausalAnchorResidualFold or TiedCausalAnchorFrontier."
    )


__all__ = [
    "TIED_CAUSAL_ANCHOR_CANDIDATE_RECIPE_SCHEMA",
    "TIED_CAUSAL_ANCHOR_FRONTIER_SCHEMA",
    "TIED_CAUSAL_ANCHOR_RESIDUAL_SCHEMA",
    "TIED_CAUSAL_ANCHOR_SPLIT_SCHEMA",
    "TiedCausalAnchorCandidateRecipe",
    "TiedCausalAnchorFoldError",
    "TiedCausalAnchorFrontier",
    "TiedCausalAnchorResidualFold",
    "TiedCausalAnchorSplit",
    "enumerate_tied_causal_anchor_candidates",
    "fit_tied_causal_interface_anchor_residual",
    "materialize_tied_causal_anchor_candidate",
]
