"""Closed-form causal-anchor residuals for a folded tied vocabulary.

The base CPCF interface is fitted to mapped-parent logits.  Across tokenizer
boundaries that objective can faithfully preserve the *wrong* free-running
decision: the unchanged parent is executing a noncanonical token sequence.
This module adds a bounded residual to the already folded dense tied table
using coherent, teacher-forced next-token anchors from the sealed trajectory
cache.

The producer encoder is fixed.  For every fit observation, the residual is
asked to place the target next token above the strongest mapped-parent
competitor by a candidate margin.  The sparse linear constraints are solved in
observation space with deterministic FP64 ridge regression.  The unmodified
base interface is an explicit same-row control.  Residual candidates must
preserve mapped-parent behavior relative to that control, and candidates are
ranked only by mapped-parent logit behavior and drift.  Target-anchor metrics
remain diagnostics.  Validation is evaluated only after the winner is frozen.

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

    base_norm = float(torch.linalg.vector_norm(base_weight).item())
    if not math.isfinite(base_norm) or base_norm <= 0.0:
        raise TiedCausalAnchorFoldError(
            "The base tied table must have positive finite energy."
        )
    residual_norm = float(torch.linalg.vector_norm(residual).item())
    raw_relative_drift = residual_norm / base_norm
    scale = 1.0
    if raw_relative_drift > maximum_relative_table_drift:
        scale = maximum_relative_table_drift / raw_relative_drift
        residual.mul_(scale)
        residual_norm *= scale
    relative_drift = residual_norm / base_norm
    repaired = base_weight + residual
    if (
        tuple(repaired.shape) != tuple(base_weight.shape)
        or not bool(torch.isfinite(repaired).all().item())
        or relative_drift > maximum_relative_table_drift + 1e-12
    ):
        raise TiedCausalAnchorFoldError(
            "Causal-anchor residual violated its shape, finite, or drift bound."
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
        "raw_relative_table_drift": raw_relative_drift,
        "relative_table_drift": relative_drift,
        "maximum_relative_table_drift": maximum_relative_table_drift,
        "drift_scale": scale,
        "drift_clipped": scale < 1.0,
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
        "exact_output_text_or_token_sequence_matching_used": False,
    }


@dataclass(frozen=True, slots=True)
class TiedCausalAnchorResidualFold:
    """A repaired interface plus its same-shaped tied-table residual."""

    interface: TiedCausalInterfaceFold
    residual_weight: Any
    report: Mapping[str, Any]

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
        object.__setattr__(self, "residual_weight", residual)
        object.__setattr__(self, "report", MappingProxyType(report))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


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
    """Fit and select one bounded same-shape tied-table anchor residual."""

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
    behavior_retention = _fraction(
        minimum_mapped_parent_behavior_retention,
        "minimum_mapped_parent_behavior_retention",
    )
    if splits["fit"].row_count > maximum_fit_rows:
        raise TiedCausalAnchorFoldError(
            "Fit anchor count exceeds the dense observation-space safety "
            f"ceiling: {splits['fit'].row_count} > {maximum_fit_rows}."
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
    required_parent_top1 = base_parent_top1 * behavior_retention
    required_parent_energy = (
        base_parent_energy * behavior_retention
        if base_parent_energy > 0.0
        else base_parent_energy
    )

    def behavior_score(
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
        "drift_scale": 1.0,
        "drift_clipped": False,
        "residual_weight_sha256": _tensor_sha256(zero_residual),
        "target_weight_sha256": _tensor_sha256(base_weight),
        "selection_metrics": base_selection_metrics,
        "minimum_required_mapped_parent_top1_agreement": required_parent_top1,
        "minimum_required_mapped_parent_raw_centered_logit_retained_energy_fraction": (
            required_parent_energy
        ),
        "minimum_required_base_interface_raw_centered_logit_retained_energy_fraction": (
            behavior_retention
        ),
        "mapped_parent_behavior_eligible": True,
        "selection_behavior_score": list(
            behavior_score(
                base_selection_metrics,
                relative_table_drift=0.0,
            )
        ),
        "target_anchor_metrics_used_for_candidate_selection": False,
        "selection_split_used_for_candidate_parameters": False,
        "validation_split_read": False,
    }
    candidate_reports = [base_candidate_report]
    candidates = [
        (
            behavior_score(
                base_selection_metrics,
                relative_table_drift=0.0,
            ),
            0,
            zero_residual,
            base_weight,
        )
    ]
    for ridge in ridges:
        for margin in margins:
            residual, fit_report = _fit_residual(
                base_weight,
                fit_coordinates,
                splits["fit"].target_next_token_ids,
                fit_competitors,
                ridge=ridge,
                margin=margin,
                maximum_relative_table_drift=drift_limit,
            )
            repaired = base_weight + residual
            selection_metrics = _split_metrics(
                splits["selection"],
                source_table,
                encoder,
                repaired,
                base_weight,
                norm_epsilon=norm_epsilon,
                chunk_rows=chunk_rows,
            )
            candidate_index = len(candidate_reports)
            parent_top1 = float(selection_metrics["mapped_parent_top1_agreement"])
            parent_energy = float(
                selection_metrics[
                    "mapped_parent_raw_centered_logit_retained_energy_fraction"
                ]
            )
            base_energy = float(
                selection_metrics[
                    "base_interface_raw_centered_logit_retained_energy_fraction"
                ]
            )
            eligible = (
                parent_top1 + 1e-12 >= required_parent_top1
                and parent_energy + 1e-12 >= required_parent_energy
                and base_energy + 1e-12 >= behavior_retention
            )
            score = behavior_score(
                selection_metrics,
                relative_table_drift=float(fit_report["relative_table_drift"]),
            )
            candidate_report = {
                "candidate_index": candidate_index,
                "candidate_kind": "anchor_residual",
                "is_base_control": False,
                **fit_report,
                "selection_metrics": selection_metrics,
                "minimum_required_mapped_parent_top1_agreement": (required_parent_top1),
                "minimum_required_mapped_parent_raw_centered_logit_retained_energy_fraction": (
                    required_parent_energy
                ),
                "minimum_required_base_interface_raw_centered_logit_retained_energy_fraction": (
                    behavior_retention
                ),
                "mapped_parent_behavior_eligible": eligible,
                "selection_behavior_score": list(score),
                "target_anchor_metrics_used_for_candidate_selection": False,
                "selection_split_used_for_candidate_parameters": False,
                "validation_split_read": False,
            }
            candidate_reports.append(candidate_report)
            if eligible:
                candidates.append((score, candidate_index, residual, repaired))
    _score, selected_index, selected_residual, selected_weight = max(
        candidates,
        key=lambda value: (value[0], -value[1]),
    )

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
        "producer_encoder_sha256": _tensor_sha256(encoder),
        "source_tied_table_sha256": _tensor_sha256(source_table),
        "base_target_weight_sha256": _tensor_sha256(base_weight),
        "residual_weight_sha256": _tensor_sha256(selected_residual),
        "target_weight_sha256": _tensor_sha256(selected_weight),
        "split_inventory": {name: splits[name].as_dict() for name in _SPLITS},
        "split_row_ids_disjoint": True,
        "fit_split_used_for_residual_parameters": True,
        "selection_split_used_for_residual_parameters": False,
        "selection_split_used_for_ridge_margin_selection": True,
        "validation_split_used_for_residual_parameters": False,
        "validation_split_used_for_ridge_margin_selection": False,
        "validation_values_used_for_parameters_or_selection": False,
        "validation_metrics_first_computed_after_candidate_frozen": True,
        "ridge_candidates": list(ridges),
        "margin_candidates": list(margins),
        "base_control_candidate_included": True,
        "base_control_candidate_index": 0,
        "minimum_mapped_parent_behavior_retention": behavior_retention,
        "candidate_selection_policy": (
            "eligible_if_same_row_parent_top1_parent_raw_centered_energy_and_"
            "base_raw_centered_energy_preserve_configured_fraction_then_rank_"
            "by_parent_raw_centered_energy_parent_top1_base_energy_and_drift"
        ),
        "target_anchor_metrics_diagnostic_only": True,
        "target_anchor_metrics_used_for_candidate_selection": False,
        "eligible_candidate_indices": [
            int(candidate["candidate_index"])
            for candidate in candidate_reports
            if candidate["mapped_parent_behavior_eligible"]
        ],
        "candidate_reports": candidate_reports,
        "selected_candidate_index": selected_index,
        "selected_candidate_is_base_control": bool(selected_report["is_base_control"]),
        "active_anchor_residual_selected": not bool(selected_report["is_base_control"]),
        "selected_ridge_fraction": selected_report["ridge_fraction"],
        "selected_requested_margin": selected_report["requested_margin"],
        "relative_table_drift": selected_relative_drift,
        "maximum_relative_table_drift": drift_limit,
        "drift_bound_satisfied": (selected_relative_drift <= drift_limit + 1e-12),
        "split_metrics": split_metrics,
        "target_norm_epsilon": norm_epsilon,
        "metric_chunk_rows": chunk_rows,
        "maximum_fit_anchor_rows": maximum_fit_rows,
        "fit_anchor_row_ceiling_satisfied": True,
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
    )


__all__ = [
    "TIED_CAUSAL_ANCHOR_RESIDUAL_SCHEMA",
    "TIED_CAUSAL_ANCHOR_SPLIT_SCHEMA",
    "TiedCausalAnchorFoldError",
    "TiedCausalAnchorResidualFold",
    "TiedCausalAnchorSplit",
    "fit_tied_causal_interface_anchor_residual",
]
