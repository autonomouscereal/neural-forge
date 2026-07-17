"""Closed-form tied input/output interface folding from causal evidence.

A tied language-model vocabulary is both the input embedding table and the
output classifier.  Projecting its source weights through an arbitrary hidden
basis is therefore unnecessarily restrictive: a blank target is free to use
new coordinates as long as one dense table serves both roles.

This module chooses that table from the *observed output operator*.  It finds
the target-width vocabulary subspace that best preserves mapped-parent logits
on fit trajectories, keeps the vocabulary-wide mean as one explicit
softmax-invariant coordinate, and exposes the corresponding source-to-target
producer map.  The resulting target table can be installed directly as a
blank child's tied embedding/LM-head parameter.

Only deterministic FP64 eigendecompositions and matrix products are used.
There is no optimizer, backward pass, quantization, pruning, parent execution,
or mutation of a model.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


_SCHEMA = "neural-forge-tied-causal-interface-fold-v1"


class TiedCausalInterfaceFoldError(ValueError):
    """The supplied evidence cannot produce a valid tied interface fold."""


class TiedCausalInterfaceCapacityError(TiedCausalInterfaceFoldError):
    """The requested target width misses an explicit held-out quality floor."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for tied causal-interface folding."
        ) from exc
    return torch


def _matrix(value: Any, name: str, *, width: int | None = None) -> Any:
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
        raise TiedCausalInterfaceFoldError(
            f"{name} must be a nonempty dense real floating matrix."
        )
    if width is not None and int(value.shape[1]) != width:
        raise TiedCausalInterfaceFoldError(
            f"{name} width must equal {width}."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise TiedCausalInterfaceFoldError(
            f"{name} contains NaN or infinity."
        )
    return result


def _weights(value: Any | None, rows: int, name: str) -> Any:
    torch = _torch()
    if value is None:
        return torch.ones(rows, dtype=torch.float64)
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 1
        or int(value.numel()) != rows
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
    ):
        raise TiedCausalInterfaceFoldError(
            f"{name} must contain one positive floating value per row."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if (
        not bool(torch.isfinite(result).all().item())
        or bool((result <= 0.0).any().item())
    ):
        raise TiedCausalInterfaceFoldError(
            f"{name} must contain finite positive values."
        )
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TiedCausalInterfaceFoldError(
            f"{name} must be a positive integer."
        )
    return value


def _fraction(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TiedCausalInterfaceFoldError(
            f"{name} must be a finite fraction within [0, 1]."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TiedCausalInterfaceFoldError(
            f"{name} must be a finite fraction within [0, 1]."
        ) from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise TiedCausalInterfaceFoldError(
            f"{name} must be a finite fraction within [0, 1]."
        )
    return result


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


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


def _canonicalize_columns(value: Any) -> Any:
    result = value.clone()
    for column_index in range(int(result.shape[1])):
        column = result[:, column_index]
        pivot = int(column.abs().argmax().item())
        if float(column[pivot].item()) < 0.0:
            result[:, column_index].mul_(-1.0)
    return result


def _rms_normalize(value: Any, epsilon: float) -> Any:
    return value * (
        value.square().mean(dim=-1, keepdim=True) + epsilon
    ).rsqrt()


def _metric_report(
    producer_rows: Any,
    source_table: Any,
    producer_encoder: Any,
    target_table: Any,
    *,
    row_weights: Any,
    chunk_rows: int,
    target_norm_epsilon: float,
) -> dict[str, Any]:
    torch = _torch()
    parent_centered_square = 0.0
    raw_error_square = 0.0
    normalized_error_square = 0.0
    raw_top1 = 0
    normalized_top1 = 0
    rows = int(producer_rows.shape[0])
    vocabulary = int(source_table.shape[0])
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        source = producer_rows[start:stop]
        weights = row_weights[start:stop].unsqueeze(1)
        parent = source @ source_table.T
        coordinates = source @ producer_encoder
        raw = coordinates @ target_table.T
        normalized = (
            _rms_normalize(coordinates, target_norm_epsilon) @ target_table.T
        )
        parent_centered = parent - parent.mean(dim=1, keepdim=True)
        raw_centered = raw - raw.mean(dim=1, keepdim=True)
        normalized_centered = normalized - normalized.mean(
            dim=1, keepdim=True
        )
        parent_centered_square += float(
            (weights * parent_centered.square()).sum().item()
        )
        raw_error_square += float(
            (weights * (raw_centered - parent_centered).square()).sum().item()
        )
        normalized_error_square += float(
            (
                weights
                * (normalized_centered - parent_centered).square()
            )
            .sum()
            .item()
        )
        parent_winner = parent.argmax(dim=1)
        raw_top1 += int((raw.argmax(dim=1) == parent_winner).sum().item())
        normalized_top1 += int(
            (normalized.argmax(dim=1) == parent_winner).sum().item()
        )
    denominator = max(
        parent_centered_square,
        torch.finfo(torch.float64).tiny,
    )
    return {
        "rows": rows,
        "vocabulary_size": vocabulary,
        "centered_parent_logit_energy": parent_centered_square,
        "raw_centered_relative_rmse": math.sqrt(
            raw_error_square / denominator
        ),
        "raw_centered_retained_energy_fraction": (
            1.0 - raw_error_square / denominator
        ),
        "raw_parent_top1_agreement": raw_top1 / rows,
        "rmsnorm_centered_relative_rmse": math.sqrt(
            normalized_error_square / denominator
        ),
        "rmsnorm_centered_retained_energy_fraction": (
            1.0 - normalized_error_square / denominator
        ),
        "rmsnorm_parent_top1_agreement": normalized_top1 / rows,
        "rowwise_positive_rmsnorm_scaling_preserves_child_order": (
            raw_top1 == normalized_top1
        ),
        "weighted_metric_rows": float(row_weights.sum().item()),
    }


@dataclass(frozen=True, slots=True)
class TiedCausalInterfaceFold:
    """One target-shaped tied table and its causal producer coordinate map."""

    target_weight: Any
    producer_encoder: Any
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        weight = _matrix(self.target_weight, "target_weight")
        encoder = _matrix(self.producer_encoder, "producer_encoder")
        if int(weight.shape[1]) != int(encoder.shape[1]):
            raise TiedCausalInterfaceFoldError(
                "Target table and producer encoder widths differ."
            )
        object.__setattr__(self, "target_weight", weight.clone())
        object.__setattr__(self, "producer_encoder", encoder.clone())
        object.__setattr__(
            self,
            "report",
            MappingProxyType(dict(self.report)),
        )

    @property
    def source_width(self) -> int:
        return int(self.producer_encoder.shape[0])

    @property
    def target_width(self) -> int:
        return int(self.producer_encoder.shape[1])

    @property
    def vocabulary_size(self) -> int:
        return int(self.target_weight.shape[0])

    def encode_producer_rows(
        self,
        source_rows: Any,
        *,
        rms_normalize: bool = False,
        norm_epsilon: float = 1e-5,
    ) -> Any:
        rows = _matrix(
            source_rows,
            "source_rows",
            width=self.source_width,
        )
        result = rows @ self.producer_encoder
        if rms_normalize:
            if not math.isfinite(norm_epsilon) or norm_epsilon <= 0.0:
                raise TiedCausalInterfaceFoldError(
                    "norm_epsilon must be positive and finite."
                )
            result = _rms_normalize(result, norm_epsilon)
        return result.contiguous()

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def fit_tied_causal_interface(
    source_tied_table: Any,
    fit_producer_rows: Any,
    *,
    target_width: int,
    validation_producer_rows: Any,
    fit_weights: Any | None = None,
    validation_weights: Any | None = None,
    preserve_vocabulary_mean: bool = True,
    target_norm_epsilon: float = 1e-5,
    minimum_validation_top1_agreement: float = 0.0,
    metric_chunk_rows: int = 512,
) -> TiedCausalInterfaceFold:
    """Fit a dense target tied table from mapped-parent output behavior.

    ``source_tied_table`` must already be in the target vocabulary order (for
    example, after a strict target-token to donor-row bridge).  Fit and
    validation producer rows are the parent's final normalized hidden states.
    """

    torch = _torch()
    table = _matrix(source_tied_table, "source_tied_table")
    fit = _matrix(
        fit_producer_rows,
        "fit_producer_rows",
        width=int(table.shape[1]),
    )
    validation = _matrix(
        validation_producer_rows,
        "validation_producer_rows",
        width=int(table.shape[1]),
    )
    if fit.data_ptr() == validation.data_ptr():
        raise TiedCausalInterfaceFoldError(
            "Fit and validation producer rows must be disjoint objects."
        )
    target_width = _positive_int(target_width, "target_width")
    metric_chunk_rows = _positive_int(metric_chunk_rows, "metric_chunk_rows")
    source_width = int(table.shape[1])
    vocabulary = int(table.shape[0])
    if target_width >= source_width:
        raise TiedCausalInterfaceFoldError(
            "The target tied interface must contract the source hidden width."
        )
    if target_width > vocabulary:
        raise TiedCausalInterfaceFoldError(
            "target_width cannot exceed the target vocabulary size."
        )
    if not isinstance(preserve_vocabulary_mean, bool):
        raise TiedCausalInterfaceFoldError(
            "preserve_vocabulary_mean must be boolean."
        )
    if (
        not math.isfinite(target_norm_epsilon)
        or target_norm_epsilon <= 0.0
    ):
        raise TiedCausalInterfaceFoldError(
            "target_norm_epsilon must be positive and finite."
        )
    required_top1 = _fraction(
        minimum_validation_top1_agreement,
        "minimum_validation_top1_agreement",
    )
    fit_weight = _weights(fit_weights, int(fit.shape[0]), "fit_weights")
    validation_weight = _weights(
        validation_weights,
        int(validation.shape[0]),
        "validation_weights",
    )
    explicit_mean_columns = int(preserve_vocabulary_mean)
    predictive_rank = target_width - explicit_mean_columns
    if predictive_rank <= 0:
        raise TiedCausalInterfaceFoldError(
            "Mean-preserving tied folding requires target_width >= 2."
        )

    mean = (
        table.mean(dim=0)
        if preserve_vocabulary_mean
        else torch.zeros(source_width, dtype=torch.float64)
    )
    centered = table - mean
    covariance = fit.T @ (fit_weight.unsqueeze(1) * fit)
    covariance = 0.5 * (covariance + covariance.T)
    covariance_values, covariance_vectors = torch.linalg.eigh(covariance)
    covariance_values = covariance_values.clamp_min(0.0)
    covariance_root = (
        covariance_vectors
        @ torch.diag(torch.sqrt(covariance_values))
        @ covariance_vectors.T
    )
    compact = covariance_root @ centered.T
    gram = compact @ compact.T
    gram = 0.5 * (gram + gram.T)
    values, vectors = torch.linalg.eigh(gram)
    order = torch.arange(source_width - 1, -1, -1, dtype=torch.long)
    values = values.index_select(0, order).clamp_min(0.0)
    vectors = vectors.index_select(1, order)
    maximum = float(values[0].item())
    tolerance = (
        max(maximum, 1.0)
        * torch.finfo(torch.float64).eps
        * max(source_width, vocabulary)
    )
    positive_rank = int((values > tolerance).sum().item())
    if predictive_rank > positive_rank:
        raise TiedCausalInterfaceFoldError(
            "Requested target width exceeds the numerical causal-logit rank: "
            f"{predictive_rank} > {positive_rank}."
        )
    selected_vectors = vectors[:, :predictive_rank]
    singular = torch.sqrt(values[:predictive_rank])
    vocabulary_vectors = (
        compact.T @ selected_vectors
    ) / singular.unsqueeze(0)
    vocabulary_vectors = _canonicalize_columns(vocabulary_vectors)
    orthogonality_error = float(
        torch.linalg.matrix_norm(
            vocabulary_vectors.T @ vocabulary_vectors
            - torch.eye(predictive_rank, dtype=torch.float64)
        ).item()
    )
    if orthogonality_error > 1e-7:
        raise TiedCausalInterfaceFoldError(
            "Causal vocabulary vectors are not numerically orthonormal."
        )

    component_rms = float(torch.sqrt(table.square().mean()).item())
    if component_rms <= 0.0:
        raise TiedCausalInterfaceFoldError(
            "The source tied table has zero energy."
        )
    column_scale = component_rms * math.sqrt(vocabulary)
    target_parts = []
    encoder_parts = []
    if preserve_vocabulary_mean:
        target_parts.append(
            torch.full(
                (vocabulary, 1),
                component_rms,
                dtype=torch.float64,
            )
        )
        encoder_parts.append((mean / component_rms).unsqueeze(1))
    target_parts.append(vocabulary_vectors * column_scale)
    encoder_parts.append(
        centered.T @ vocabulary_vectors / column_scale
    )
    target = torch.cat(target_parts, dim=1).contiguous()
    encoder = torch.cat(encoder_parts, dim=1).contiguous()
    if (
        tuple(target.shape) != (vocabulary, target_width)
        or tuple(encoder.shape) != (source_width, target_width)
        or not bool(torch.isfinite(target).all().item())
        or not bool(torch.isfinite(encoder).all().item())
    ):
        raise TiedCausalInterfaceFoldError(
            "The compiled tied interface has invalid shape or values."
        )

    fit_metrics = _metric_report(
        fit,
        table,
        encoder,
        target,
        row_weights=fit_weight,
        chunk_rows=metric_chunk_rows,
        target_norm_epsilon=target_norm_epsilon,
    )
    validation_metrics = _metric_report(
        validation,
        table,
        encoder,
        target,
        row_weights=validation_weight,
        chunk_rows=metric_chunk_rows,
        target_norm_epsilon=target_norm_epsilon,
    )
    observed_top1 = float(
        validation_metrics["rmsnorm_parent_top1_agreement"]
    )
    if observed_top1 + 1e-12 < required_top1:
        raise TiedCausalInterfaceCapacityError(
            "Target tied-interface width retains only "
            f"{observed_top1:.6f} held-out parent top-1 agreement; "
            f"{required_top1:.6f} is required."
        )

    total_energy = float(values.sum().item())
    selected_energy = float(values[:predictive_rank].sum().item())
    body = {
        "schema": _SCHEMA,
        "method": (
            "weighted_causal_logit_eigenspace_with_explicit_vocabulary_mean"
            if preserve_vocabulary_mean
            else "weighted_causal_logit_eigenspace"
        ),
        "source_width": source_width,
        "target_width": target_width,
        "vocabulary_size": vocabulary,
        "predictive_rank": predictive_rank,
        "numerical_causal_logit_rank": positive_rank,
        "fit_rows": int(fit.shape[0]),
        "validation_rows": int(validation.shape[0]),
        "fit_validation_disjoint_objects": True,
        "preserve_vocabulary_mean": preserve_vocabulary_mean,
        "softmax_invariant_mean_coordinate": preserve_vocabulary_mean,
        "retained_weighted_causal_logit_energy_fraction": (
            1.0 if total_energy <= 0.0 else selected_energy / total_energy
        ),
        "minimum_validation_top1_agreement": required_top1,
        "fit": fit_metrics,
        "validation": validation_metrics,
        "target_norm_epsilon": target_norm_epsilon,
        "target_dense_parameter_count": int(target.numel()),
        "mapped_source_dense_parameter_count": int(table.numel()),
        "interface_parameter_reduction_fraction": (
            1.0 - int(target.numel()) / int(table.numel())
        ),
        "vocabulary_vector_orthogonality_error": orthogonality_error,
        "source_table_sha256": _tensor_sha256(table),
        "fit_producer_rows_sha256": _tensor_sha256(fit),
        "validation_producer_rows_sha256": _tensor_sha256(validation),
        "target_weight_sha256": _tensor_sha256(target),
        "producer_encoder_sha256": _tensor_sha256(encoder),
        "blank_target_values_read": False,
        "parent_forward_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "quantization_used": False,
        "pruning_used": False,
        "numeric_parent_weight_matching_objective": False,
        "functional_mapped_parent_logit_objective": True,
    }
    report = {**body, "report_sha256": _canonical_sha256(body)}
    return TiedCausalInterfaceFold(
        target_weight=target,
        producer_encoder=encoder,
        report=report,
    )


__all__ = [
    "TiedCausalInterfaceCapacityError",
    "TiedCausalInterfaceFold",
    "TiedCausalInterfaceFoldError",
    "fit_tied_causal_interface",
]
