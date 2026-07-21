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
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


_SCHEMA = "neural-forge-tied-causal-interface-fold-v1"
TIED_CAUSAL_INTERFACE_SUFFICIENT_STATISTICS_SCHEMA = (
    "neural-forge-tied-causal-interface-sufficient-statistics-v1"
)
TIED_CAUSAL_INTERFACE_METRIC_WITNESS_SCHEMA = (
    "neural-forge-tied-causal-interface-metric-witness-v1"
)
TIED_CAUSAL_INTERFACE_SELF_DUAL_SCHEMA = (
    "neural-forge-tied-causal-interface-self-dual-gauge-v1"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


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
        raise TiedCausalInterfaceFoldError(f"{name} width must equal {width}.")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise TiedCausalInterfaceFoldError(f"{name} contains NaN or infinity.")
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
    if not bool(torch.isfinite(result).all().item()) or bool(
        (result <= 0.0).any().item()
    ):
        raise TiedCausalInterfaceFoldError(
            f"{name} must contain finite positive values."
        )
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TiedCausalInterfaceFoldError(f"{name} must be a positive integer.")
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
    return value * (value.square().mean(dim=-1, keepdim=True) + epsilon).rsqrt()


@dataclass(frozen=True, slots=True)
class TiedCausalInterfaceMetricWitness:
    """Exact scalar reductions needed to reconstruct one legacy metric report."""

    rows: int
    weighted_metric_rows: float
    parent_centered_square: float
    raw_error_square: float
    normalized_error_square: float
    raw_top1: int
    normalized_top1: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.rows, bool)
            or not isinstance(self.rows, int)
            or self.rows <= 0
        ):
            raise TiedCausalInterfaceFoldError("Metric witness rows must be positive.")
        for name in (
            "weighted_metric_rows",
            "parent_centered_square",
            "raw_error_square",
            "normalized_error_square",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TiedCausalInterfaceFoldError(
                    f"Metric witness {name} must be a finite scalar."
                )
            normalized = float(value)
            if not math.isfinite(normalized) or normalized < 0.0:
                raise TiedCausalInterfaceFoldError(
                    f"Metric witness {name} must be finite and nonnegative."
                )
            if name == "weighted_metric_rows" and normalized <= 0.0:
                raise TiedCausalInterfaceFoldError(
                    "Metric witness weighted rows must be positive."
                )
            object.__setattr__(self, name, normalized)
        for name in ("raw_top1", "normalized_top1"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > self.rows
            ):
                raise TiedCausalInterfaceFoldError(
                    f"Metric witness {name} must be within the row count."
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": TIED_CAUSAL_INTERFACE_METRIC_WITNESS_SCHEMA,
            "rows": self.rows,
            "weighted_metric_rows": self.weighted_metric_rows,
            "parent_centered_square": self.parent_centered_square,
            "raw_error_square": self.raw_error_square,
            "normalized_error_square": self.normalized_error_square,
            "raw_top1": self.raw_top1,
            "normalized_top1": self.normalized_top1,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TiedCausalInterfaceMetricWitness":
        if not isinstance(value, Mapping) or value.get("schema") != (
            TIED_CAUSAL_INTERFACE_METRIC_WITNESS_SCHEMA
        ):
            raise TiedCausalInterfaceFoldError("Metric witness schema is invalid.")
        expected = {
            "schema",
            "rows",
            "weighted_metric_rows",
            "parent_centered_square",
            "raw_error_square",
            "normalized_error_square",
            "raw_top1",
            "normalized_top1",
        }
        if set(value) != expected:
            raise TiedCausalInterfaceFoldError("Metric witness fields are invalid.")
        return cls(
            rows=value["rows"],
            weighted_metric_rows=value["weighted_metric_rows"],
            parent_centered_square=value["parent_centered_square"],
            raw_error_square=value["raw_error_square"],
            normalized_error_square=value["normalized_error_square"],
            raw_top1=value["raw_top1"],
            normalized_top1=value["normalized_top1"],
        )


@dataclass(frozen=True, slots=True)
class TiedCausalInterfaceSufficientStatistics:
    """Compact evidence sufficient to replay one tied-interface base solve."""

    fit_second_moment: Any
    fit_metric_witness: TiedCausalInterfaceMetricWitness
    validation_metric_witness: TiedCausalInterfaceMetricWitness
    source_table_sha256: str
    fit_producer_rows_sha256: str
    validation_producer_rows_sha256: str
    target_width: int
    preserve_vocabulary_mean: bool = True
    target_norm_epsilon: float = 1e-5
    minimum_validation_top1_agreement: float = 0.0
    metric_chunk_rows: int = 512
    statistics_sha256: str = ""

    def __post_init__(self) -> None:
        torch = _torch()
        moment = _matrix(self.fit_second_moment, "fit_second_moment")
        if int(moment.shape[0]) != int(moment.shape[1]) or not torch.equal(
            moment, moment.T
        ):
            raise TiedCausalInterfaceFoldError(
                "Fit second moment must be an exactly symmetric square matrix."
            )
        if not isinstance(self.fit_metric_witness, TiedCausalInterfaceMetricWitness):
            raise TiedCausalInterfaceFoldError("Fit metric witness is invalid.")
        if not isinstance(
            self.validation_metric_witness, TiedCausalInterfaceMetricWitness
        ):
            raise TiedCausalInterfaceFoldError("Validation metric witness is invalid.")
        for name in (
            "source_table_sha256",
            "fit_producer_rows_sha256",
            "validation_producer_rows_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise TiedCausalInterfaceFoldError(f"{name} is not a SHA-256 digest.")
        target_width = _positive_int(self.target_width, "target_width")
        if target_width >= int(moment.shape[0]):
            raise TiedCausalInterfaceFoldError(
                "The target tied interface must contract the source hidden width."
            )
        if not isinstance(self.preserve_vocabulary_mean, bool):
            raise TiedCausalInterfaceFoldError(
                "preserve_vocabulary_mean must be boolean."
            )
        epsilon = float(self.target_norm_epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise TiedCausalInterfaceFoldError(
                "target_norm_epsilon must be positive and finite."
            )
        required_top1 = _fraction(
            self.minimum_validation_top1_agreement,
            "minimum_validation_top1_agreement",
        )
        chunk_rows = _positive_int(self.metric_chunk_rows, "metric_chunk_rows")
        object.__setattr__(self, "fit_second_moment", moment.clone())
        object.__setattr__(self, "target_width", target_width)
        object.__setattr__(self, "target_norm_epsilon", epsilon)
        object.__setattr__(self, "minimum_validation_top1_agreement", required_top1)
        object.__setattr__(self, "metric_chunk_rows", chunk_rows)
        body = self._body()
        expected = _canonical_sha256(body)
        if self.statistics_sha256 and self.statistics_sha256 != expected:
            raise TiedCausalInterfaceFoldError(
                "Tied-interface sufficient-statistics seal is invalid."
            )
        object.__setattr__(self, "statistics_sha256", expected)

    @property
    def source_width(self) -> int:
        return int(self.fit_second_moment.shape[0])

    def _body(self) -> dict[str, Any]:
        return {
            "schema": TIED_CAUSAL_INTERFACE_SUFFICIENT_STATISTICS_SCHEMA,
            "source_width": int(self.fit_second_moment.shape[0]),
            "target_width": self.target_width,
            "preserve_vocabulary_mean": self.preserve_vocabulary_mean,
            "target_norm_epsilon": self.target_norm_epsilon,
            "minimum_validation_top1_agreement": (
                self.minimum_validation_top1_agreement
            ),
            "metric_chunk_rows": self.metric_chunk_rows,
            "fit_second_moment_sha256": _tensor_sha256(self.fit_second_moment),
            "fit_metric_witness": self.fit_metric_witness.as_dict(),
            "validation_metric_witness": self.validation_metric_witness.as_dict(),
            "source_table_sha256": self.source_table_sha256,
            "fit_producer_rows_sha256": self.fit_producer_rows_sha256,
            "validation_producer_rows_sha256": (self.validation_producer_rows_sha256),
            "fit_validation_disjoint_objects": True,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "statistics_sha256": self.statistics_sha256}


def _metric_witness(
    producer_rows: Any,
    source_table: Any,
    producer_encoder: Any,
    target_table: Any,
    *,
    row_weights: Any,
    chunk_rows: int,
    target_norm_epsilon: float,
) -> TiedCausalInterfaceMetricWitness:
    parent_centered_square = 0.0
    raw_error_square = 0.0
    normalized_error_square = 0.0
    raw_top1 = 0
    normalized_top1 = 0
    rows = int(producer_rows.shape[0])
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        source = producer_rows[start:stop]
        weights = row_weights[start:stop].unsqueeze(1)
        parent = source @ source_table.T
        coordinates = source @ producer_encoder
        raw = coordinates @ target_table.T
        normalized = _rms_normalize(coordinates, target_norm_epsilon) @ target_table.T
        parent_centered = parent - parent.mean(dim=1, keepdim=True)
        raw_centered = raw - raw.mean(dim=1, keepdim=True)
        normalized_centered = normalized - normalized.mean(dim=1, keepdim=True)
        parent_centered_square += float(
            (weights * parent_centered.square()).sum().item()
        )
        raw_error_square += float(
            (weights * (raw_centered - parent_centered).square()).sum().item()
        )
        normalized_error_square += float(
            (weights * (normalized_centered - parent_centered).square()).sum().item()
        )
        parent_winner = parent.argmax(dim=1)
        raw_top1 += int((raw.argmax(dim=1) == parent_winner).sum().item())
        normalized_top1 += int((normalized.argmax(dim=1) == parent_winner).sum().item())
    return TiedCausalInterfaceMetricWitness(
        rows=rows,
        weighted_metric_rows=float(row_weights.sum().item()),
        parent_centered_square=parent_centered_square,
        raw_error_square=raw_error_square,
        normalized_error_square=normalized_error_square,
        raw_top1=raw_top1,
        normalized_top1=normalized_top1,
    )


def _metric_report_from_witness(
    witness: TiedCausalInterfaceMetricWitness,
    *,
    vocabulary_size: int,
) -> dict[str, Any]:
    torch = _torch()
    if not isinstance(witness, TiedCausalInterfaceMetricWitness):
        raise TiedCausalInterfaceFoldError("Metric witness is invalid.")
    rows = witness.rows
    parent_centered_square = witness.parent_centered_square
    raw_error_square = witness.raw_error_square
    normalized_error_square = witness.normalized_error_square
    raw_top1 = witness.raw_top1
    normalized_top1 = witness.normalized_top1
    denominator = max(
        parent_centered_square,
        torch.finfo(torch.float64).tiny,
    )
    return {
        "rows": rows,
        "vocabulary_size": vocabulary_size,
        "centered_parent_logit_energy": parent_centered_square,
        "raw_centered_relative_rmse": math.sqrt(raw_error_square / denominator),
        "raw_centered_retained_energy_fraction": (1.0 - raw_error_square / denominator),
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
        "weighted_metric_rows": witness.weighted_metric_rows,
    }


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
    return _metric_report_from_witness(
        _metric_witness(
            producer_rows,
            source_table,
            producer_encoder,
            target_table,
            row_weights=row_weights,
            chunk_rows=chunk_rows,
            target_norm_epsilon=target_norm_epsilon,
        ),
        vocabulary_size=int(source_table.shape[0]),
    )


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


def whiten_tied_causal_interface(
    interface: TiedCausalInterfaceFold,
) -> TiedCausalInterfaceFold:
    """Move a tied fold into one orthonormal, input/output-consistent gauge.

    For target table ``T`` and source-to-child encoder ``E``, any invertible
    target-coordinate transform can be applied reciprocally without changing
    the represented source-to-vocabulary operator.  With the thin singular
    value decomposition ``E = U diag(s) V.T``, this routine chooses the unique
    full-column-rank polar decomposition::

        B = U @ V.T
        R = V @ diag(s) @ V.T
        T_self_dual = T @ R

    so ``E = B @ R``, ``B.T @ B = I``, and
    ``B @ T_self_dual.T = E @ T.T``.  Computing the polar factors directly
    from the thin SVD avoids squaring the encoder's condition number through
    ``E.T @ E``.  The same full-rank basis can therefore serve the child's
    input embeddings, residual-state encoder, and tied output head without
    discarding the reciprocal scaling that made the original output
    factorization valid.
    """

    torch = _torch()
    if not isinstance(interface, TiedCausalInterfaceFold):
        raise TypeError("interface must be TiedCausalInterfaceFold.")
    target = _matrix(interface.target_weight, "interface.target_weight")
    encoder = _matrix(interface.producer_encoder, "interface.producer_encoder")
    source_report = dict(interface.report)
    source_report_body = dict(source_report)
    source_report_sha256 = source_report_body.pop("report_sha256", None)
    if (
        not isinstance(source_report_sha256, str)
        or len(source_report_sha256) != 64
        or _canonical_sha256(source_report_body) != source_report_sha256
    ):
        raise TiedCausalInterfaceFoldError(
            "Source tied interface report is not canonically authenticated."
        )

    with torch.no_grad():
        polar_left, singular_values, polar_right_h = torch.linalg.svd(
            encoder,
            full_matrices=False,
        )
        maximum = float(singular_values.max().item())
        rank_tolerance = (
            max(maximum, 1.0)
            * torch.finfo(torch.float64).eps
            * max(int(encoder.shape[0]), int(encoder.shape[1]))
        )
        if maximum <= 0.0 or bool((singular_values <= rank_tolerance).any().item()):
            raise TiedCausalInterfaceFoldError(
                "Self-dual whitening requires a full-column-rank producer encoder."
            )
        self_dual_encoder = (polar_left @ polar_right_h).contiguous()
        right_stretch = (
            polar_right_h.T @ torch.diag(singular_values) @ polar_right_h
        ).contiguous()
        self_dual_target = (target @ right_stretch).contiguous()
        identity = torch.eye(int(encoder.shape[1]), dtype=torch.float64)
        orthogonality_error = float(
            torch.linalg.matrix_norm(
                self_dual_encoder.T @ self_dual_encoder - identity,
                ord=2,
            ).item()
        )
        source_operator = encoder @ target.T
        self_dual_operator = self_dual_encoder @ self_dual_target.T
        operator_residual = self_dual_operator - source_operator
        reconstructed_encoder = self_dual_encoder @ right_stretch
        reconstruction_relative_error = float(
            torch.linalg.vector_norm(reconstructed_encoder - encoder).item()
            / max(
                float(torch.linalg.vector_norm(encoder).item()),
                torch.finfo(torch.float64).tiny,
            )
        )
        operator_relative_error = float(
            torch.linalg.vector_norm(operator_residual).item()
            / max(
                float(torch.linalg.vector_norm(source_operator).item()),
                torch.finfo(torch.float64).tiny,
            )
        )
        operator_maximum_error = float(operator_residual.abs().max().item())
        source_mean_operator = encoder @ target.mean(dim=0)
        self_dual_mean_operator = self_dual_encoder @ self_dual_target.mean(dim=0)
        mean_operator_maximum_error = float(
            (self_dual_mean_operator - source_mean_operator).abs().max().item()
        )
        condition_number = float((singular_values.max() / singular_values.min()).item())
    numerical_tolerance = max(
        1e-10,
        100.0
        * torch.finfo(torch.float64).eps
        * max(int(encoder.shape[0]), int(encoder.shape[1]))
        * max(1.0, condition_number),
    )
    if (
        not bool(torch.isfinite(self_dual_encoder).all().item())
        or not bool(torch.isfinite(self_dual_target).all().item())
        or orthogonality_error > numerical_tolerance
        or reconstruction_relative_error > numerical_tolerance
        or operator_relative_error > numerical_tolerance
        or mean_operator_maximum_error
        > numerical_tolerance
        * max(
            float(source_mean_operator.abs().max().item()),
            torch.finfo(torch.float64).tiny,
        )
    ):
        raise TiedCausalInterfaceFoldError(
            "Self-dual tied-interface whitening failed its orthogonality or "
            "operator-preservation invariant."
        )

    body = {
        **source_report_body,
        "schema": TIED_CAUSAL_INTERFACE_SELF_DUAL_SCHEMA,
        "method": "thin_svd_polar_self_dual_tied_interface_gauge",
        "source_interface_schema": source_report.get("schema"),
        "source_interface_report_sha256": source_report_sha256,
        "source_target_weight_sha256": _tensor_sha256(target),
        "source_producer_encoder_sha256": _tensor_sha256(encoder),
        "target_weight_sha256": _tensor_sha256(self_dual_target),
        "producer_encoder_sha256": _tensor_sha256(self_dual_encoder),
        "producer_encoder_singular_values_sha256": _tensor_sha256(singular_values),
        "producer_encoder_right_stretch_sha256": _tensor_sha256(right_stretch),
        "producer_encoder_numerical_rank": int(singular_values.numel()),
        "producer_encoder_rank_tolerance": rank_tolerance,
        "encoder_condition_number": condition_number,
        "encoder_polar_reconstruction_relative_error": (reconstruction_relative_error),
        "self_dual_encoder_orthogonality_error": orthogonality_error,
        "source_to_vocabulary_operator_relative_error": operator_relative_error,
        "source_to_vocabulary_operator_maximum_absolute_error": (
            operator_maximum_error
        ),
        "vocabulary_mean_operator_maximum_absolute_error": (
            mean_operator_maximum_error
        ),
        "source_to_vocabulary_operator_preserved": True,
        "full_vocabulary_rows_preserved": True,
        "full_target_width_preserved": True,
        "vocabulary_mean_subspace_preserved": True,
        "vocabulary_mean_coordinate_may_mix": True,
        "one_global_residual_gauge_available": True,
        "parent_forward_calls": 0,
        "child_forward_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "quantization_used": False,
        "pruning_used": False,
    }
    report = {**body, "report_sha256": _canonical_sha256(body)}
    return TiedCausalInterfaceFold(
        target_weight=self_dual_target,
        producer_encoder=self_dual_encoder,
        report=report,
    )


@dataclass(frozen=True, slots=True)
class _SecondMomentSolution:
    target_weight: Any
    producer_encoder: Any
    spectrum: Any
    predictive_rank: int
    numerical_rank: int
    vocabulary_vector_orthogonality_error: float


def _solve_tied_causal_interface_second_moment(
    source_table: Any,
    fit_second_moment: Any,
    *,
    target_width: int,
    preserve_vocabulary_mean: bool,
) -> _SecondMomentSolution:
    torch = _torch()
    source_width = int(source_table.shape[1])
    vocabulary = int(source_table.shape[0])
    explicit_mean_columns = int(preserve_vocabulary_mean)
    predictive_rank = target_width - explicit_mean_columns
    if predictive_rank <= 0:
        raise TiedCausalInterfaceFoldError(
            "Mean-preserving tied folding requires target_width >= 2."
        )

    mean = (
        source_table.mean(dim=0)
        if preserve_vocabulary_mean
        else torch.zeros(source_width, dtype=torch.float64)
    )
    centered = source_table - mean
    covariance_values, covariance_vectors = torch.linalg.eigh(fit_second_moment)
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
    vocabulary_vectors = (compact.T @ selected_vectors) / singular.unsqueeze(0)
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

    component_rms = float(torch.sqrt(source_table.square().mean()).item())
    if component_rms <= 0.0:
        raise TiedCausalInterfaceFoldError("The source tied table has zero energy.")
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
    encoder_parts.append(centered.T @ vocabulary_vectors / column_scale)
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
    return _SecondMomentSolution(
        target_weight=target,
        producer_encoder=encoder,
        spectrum=values,
        predictive_rank=predictive_rank,
        numerical_rank=positive_rank,
        vocabulary_vector_orthogonality_error=orthogonality_error,
    )


def _finish_tied_causal_interface_from_statistics(
    source_table: Any,
    statistics: TiedCausalInterfaceSufficientStatistics,
    solution: _SecondMomentSolution,
) -> TiedCausalInterfaceFold:
    target = solution.target_weight
    encoder = solution.producer_encoder
    vocabulary = int(source_table.shape[0])
    fit_metrics = _metric_report_from_witness(
        statistics.fit_metric_witness,
        vocabulary_size=vocabulary,
    )
    validation_metrics = _metric_report_from_witness(
        statistics.validation_metric_witness,
        vocabulary_size=vocabulary,
    )
    required_top1 = statistics.minimum_validation_top1_agreement
    observed_top1 = float(validation_metrics["rmsnorm_parent_top1_agreement"])
    if observed_top1 + 1e-12 < required_top1:
        raise TiedCausalInterfaceCapacityError(
            "Target tied-interface width retains only "
            f"{observed_top1:.6f} held-out parent top-1 agreement; "
            f"{required_top1:.6f} is required."
        )

    total_energy = float(solution.spectrum.sum().item())
    selected_energy = float(solution.spectrum[: solution.predictive_rank].sum().item())
    body = {
        "schema": _SCHEMA,
        "method": (
            "weighted_causal_logit_eigenspace_with_explicit_vocabulary_mean"
            if statistics.preserve_vocabulary_mean
            else "weighted_causal_logit_eigenspace"
        ),
        "source_width": statistics.source_width,
        "target_width": statistics.target_width,
        "vocabulary_size": vocabulary,
        "predictive_rank": solution.predictive_rank,
        "numerical_causal_logit_rank": solution.numerical_rank,
        "fit_rows": statistics.fit_metric_witness.rows,
        "validation_rows": statistics.validation_metric_witness.rows,
        "fit_validation_disjoint_objects": True,
        "preserve_vocabulary_mean": statistics.preserve_vocabulary_mean,
        "softmax_invariant_mean_coordinate": statistics.preserve_vocabulary_mean,
        "retained_weighted_causal_logit_energy_fraction": (
            1.0 if total_energy <= 0.0 else selected_energy / total_energy
        ),
        "minimum_validation_top1_agreement": required_top1,
        "fit": fit_metrics,
        "validation": validation_metrics,
        "target_norm_epsilon": statistics.target_norm_epsilon,
        "target_dense_parameter_count": int(target.numel()),
        "mapped_source_dense_parameter_count": int(source_table.numel()),
        "interface_parameter_reduction_fraction": (
            1.0 - int(target.numel()) / int(source_table.numel())
        ),
        "vocabulary_vector_orthogonality_error": (
            solution.vocabulary_vector_orthogonality_error
        ),
        "source_table_sha256": _tensor_sha256(source_table),
        "fit_producer_rows_sha256": statistics.fit_producer_rows_sha256,
        "validation_producer_rows_sha256": (statistics.validation_producer_rows_sha256),
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


def fit_tied_causal_interface_from_statistics(
    source_tied_table: Any,
    statistics: TiedCausalInterfaceSufficientStatistics,
) -> TiedCausalInterfaceFold:
    """Replay a tied-interface fit without retaining any full producer rows."""

    if not isinstance(statistics, TiedCausalInterfaceSufficientStatistics):
        raise TypeError("statistics must be TiedCausalInterfaceSufficientStatistics.")
    table = _matrix(source_tied_table, "source_tied_table")
    if int(table.shape[1]) != statistics.source_width:
        raise TiedCausalInterfaceFoldError(
            "Source tied-table width differs from the sufficient statistics."
        )
    if _tensor_sha256(table) != statistics.source_table_sha256:
        raise TiedCausalInterfaceFoldError(
            "Source tied table differs from the sufficient-statistics binding."
        )
    if statistics.target_width > int(table.shape[0]):
        raise TiedCausalInterfaceFoldError(
            "target_width cannot exceed the target vocabulary size."
        )
    solution = _solve_tied_causal_interface_second_moment(
        table,
        statistics.fit_second_moment,
        target_width=statistics.target_width,
        preserve_vocabulary_mean=statistics.preserve_vocabulary_mean,
    )
    return _finish_tied_causal_interface_from_statistics(
        table,
        statistics,
        solution,
    )


def fit_tied_causal_interface_with_statistics(
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
) -> tuple[TiedCausalInterfaceFold, TiedCausalInterfaceSufficientStatistics]:
    """Fit a dense tied table and return its exact compact replay evidence.

    ``source_tied_table`` must already be in the target vocabulary order (for
    example, after a strict target-token to donor-row bridge).  Fit and
    validation producer rows are the parent's final normalized hidden states.
    """

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
        raise TiedCausalInterfaceFoldError("preserve_vocabulary_mean must be boolean.")
    if not math.isfinite(target_norm_epsilon) or target_norm_epsilon <= 0.0:
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
    covariance = fit.T @ (fit_weight.unsqueeze(1) * fit)
    covariance = 0.5 * (covariance + covariance.T)
    solution = _solve_tied_causal_interface_second_moment(
        table,
        covariance,
        target_width=target_width,
        preserve_vocabulary_mean=preserve_vocabulary_mean,
    )
    fit_witness = _metric_witness(
        fit,
        table,
        solution.producer_encoder,
        solution.target_weight,
        row_weights=fit_weight,
        chunk_rows=metric_chunk_rows,
        target_norm_epsilon=target_norm_epsilon,
    )
    validation_witness = _metric_witness(
        validation,
        table,
        solution.producer_encoder,
        solution.target_weight,
        row_weights=validation_weight,
        chunk_rows=metric_chunk_rows,
        target_norm_epsilon=target_norm_epsilon,
    )
    statistics = TiedCausalInterfaceSufficientStatistics(
        fit_second_moment=covariance,
        fit_metric_witness=fit_witness,
        validation_metric_witness=validation_witness,
        source_table_sha256=_tensor_sha256(table),
        fit_producer_rows_sha256=_tensor_sha256(fit),
        validation_producer_rows_sha256=_tensor_sha256(validation),
        target_width=target_width,
        preserve_vocabulary_mean=preserve_vocabulary_mean,
        target_norm_epsilon=target_norm_epsilon,
        minimum_validation_top1_agreement=required_top1,
        metric_chunk_rows=metric_chunk_rows,
    )
    folded = _finish_tied_causal_interface_from_statistics(
        table,
        statistics,
        solution,
    )
    return folded, statistics


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
    """Fit a dense target tied table from mapped-parent output behavior."""

    folded, _statistics = fit_tied_causal_interface_with_statistics(
        source_tied_table,
        fit_producer_rows,
        target_width=target_width,
        validation_producer_rows=validation_producer_rows,
        fit_weights=fit_weights,
        validation_weights=validation_weights,
        preserve_vocabulary_mean=preserve_vocabulary_mean,
        target_norm_epsilon=target_norm_epsilon,
        minimum_validation_top1_agreement=minimum_validation_top1_agreement,
        metric_chunk_rows=metric_chunk_rows,
    )
    return folded


__all__ = [
    "TIED_CAUSAL_INTERFACE_METRIC_WITNESS_SCHEMA",
    "TIED_CAUSAL_INTERFACE_SELF_DUAL_SCHEMA",
    "TIED_CAUSAL_INTERFACE_SUFFICIENT_STATISTICS_SCHEMA",
    "TiedCausalInterfaceCapacityError",
    "TiedCausalInterfaceFold",
    "TiedCausalInterfaceFoldError",
    "TiedCausalInterfaceMetricWitness",
    "TiedCausalInterfaceSufficientStatistics",
    "fit_tied_causal_interface",
    "fit_tied_causal_interface_from_statistics",
    "fit_tied_causal_interface_with_statistics",
    "whiten_tied_causal_interface",
]
