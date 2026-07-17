"""Closed-form causal circuit folding for severe blank-target contraction.

The existing dense-width transport path uses one residual basis to conjugate
many unlike operators.  That is only function preserving when the reachable
state lies in the retained subspace *and* the operator family is closed under
that basis change.  Diagonal RMSNorm gains and depthwise FIR kernels are not
closed under a general dense change of basis.  Q/K score geometry and SwiGLU
features are nonlinear, so weight projection is not a functional compiler for
them either.

This module provides isolated, family-neutral primitives for a different
compiler:

* boundary coordinates are selected by predictive future/action evidence,
  rather than raw weight or residual energy;
* Q/K is fitted directly to cached centered attention scores;
* V/O is fitted jointly to cached residual contributions;
* SwiGLU and multiplicative short-convolution channels are compiled as
  functional feature dictionaries, with the final residual writer solved
  against the complete parent contribution; and
* impossible target widths fail a predictive-rank gate before a blank child is
  installed.

Every fit is a deterministic linear solve, eigendecomposition, SVD, or a fixed
sequence of those operations.  The module never executes a parent, constructs
an optimizer, invokes backward, quantizes a tensor, or mutates a model.  It
does not claim that an arbitrary information-rich parent can retain ninety
percent capability in an arbitrarily tiny target; the rank and parameter
audits make that boundary explicit.
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


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SCHEMA = "neural-forge-closed-form-causal-circuit-fold-v1"


class ClosedFormCausalCircuitFoldError(ValueError):
    """Cached evidence or a target contract cannot support a valid fold."""


class ClosedFormCausalCircuitCapacityError(ClosedFormCausalCircuitFoldError):
    """The requested target width cannot retain the sealed predictive rank."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional native dependency
        raise RuntimeError(
            "PyTorch is required for closed-form causal circuit folding."
        ) from exc
    return torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ClosedFormCausalCircuitFoldError(
            f"{name} must be a positive integer."
        )
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ClosedFormCausalCircuitFoldError(
            f"{name} must be a finite nonnegative float."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ClosedFormCausalCircuitFoldError(
            f"{name} must be a finite nonnegative float."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise ClosedFormCausalCircuitFoldError(
            f"{name} must be a finite nonnegative float."
        )
    return result


def _unit_fraction(value: Any, name: str) -> float:
    result = _nonnegative_float(value, name)
    if result > 1.0:
        raise ClosedFormCausalCircuitFoldError(
            f"{name} must be within [0, 1]."
        )
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ClosedFormCausalCircuitFoldError(
            f"{name} must be nonempty text."
        )
    return value.strip()


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ClosedFormCausalCircuitFoldError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


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
        raise ClosedFormCausalCircuitFoldError(
            f"{name} must be a nonempty dense real floating matrix."
        )
    if width is not None and int(value.shape[1]) != width:
        raise ClosedFormCausalCircuitFoldError(
            f"{name} width must equal {width}."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise ClosedFormCausalCircuitFoldError(
            f"{name} contains NaN or infinity."
        )
    return result


def _tensor(value: Any, name: str, *, dimensions: int) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != dimensions
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
    ):
        raise ClosedFormCausalCircuitFoldError(
            f"{name} must be a nonempty dense real floating tensor with "
            f"{dimensions} dimensions."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise ClosedFormCausalCircuitFoldError(
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
        raise ClosedFormCausalCircuitFoldError(
            f"{name} must contain one positive floating value per row."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if (
        not bool(torch.isfinite(result).all().item())
        or bool((result <= 0.0).any().item())
    ):
        raise ClosedFormCausalCircuitFoldError(
            f"{name} must contain finite positive values."
        )
    return result


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        _canonical_sha256([int(size) for size in tensor.shape]).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _relative_rmse(prediction: Any, target: Any, weights: Any) -> float:
    torch = _torch()
    scale = torch.sqrt(weights).reshape(-1, *([1] * (target.ndim - 1)))
    error = torch.linalg.vector_norm((prediction - target) * scale)
    denominator = torch.linalg.vector_norm(target * scale).clamp_min(1e-30)
    return float((error / denominator).item())


def _energy_retention(prediction: Any, target: Any, weights: Any) -> float:
    scale = weights.reshape(-1, *([1] * (target.ndim - 1)))
    target_energy = (scale * target.square()).sum().clamp_min(1e-30)
    error_energy = (scale * (prediction - target).square()).sum()
    return float((1.0 - error_energy / target_energy).item())


def _ridge_coefficients(
    inputs: Any,
    targets: Any,
    weights: Any,
    ridge: float,
) -> Any:
    torch = _torch()
    ridge = _nonnegative_float(ridge, "ridge")
    if (
        inputs.ndim != 2
        or targets.ndim != 2
        or int(inputs.shape[0]) != int(targets.shape[0])
        or int(weights.numel()) != int(inputs.shape[0])
    ):
        raise ClosedFormCausalCircuitFoldError(
            "Ridge inputs, targets, and weights are row-incompatible."
        )
    root = torch.sqrt(weights).unsqueeze(1)
    design = inputs * root
    response = targets * root
    gram = design.T @ design
    regularized = gram + ridge * torch.eye(
        int(gram.shape[0]),
        dtype=torch.float64,
    )
    return torch.linalg.solve(regularized, design.T @ response).contiguous()


def _canonicalize_columns(value: Any) -> Any:
    result = value.clone()
    for column_index in range(int(result.shape[1])):
        column = result[:, column_index]
        pivot = int(column.abs().argmax().item())
        if float(column[pivot].item()) < 0.0:
            result[:, column_index].mul_(-1.0)
    return result


@dataclass(frozen=True, slots=True)
class CausalFeatureBlock:
    """One role-balanced future/action view aligned to residual-state rows."""

    name: str
    role: str
    values: Any
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "feature name"))
        object.__setattr__(self, "role", _text(self.role, "feature role"))
        object.__setattr__(
            self,
            "values",
            _matrix(self.values, f"feature block {self.name}").clone(),
        )
        weight = _nonnegative_float(self.weight, f"{self.name}.weight")
        if weight == 0.0:
            raise ClosedFormCausalCircuitFoldError(
                f"{self.name}.weight must be positive."
            )
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True, slots=True)
class PredictiveBoundaryMap:
    """Layer-boundary encoder/decoder selected by causal future evidence."""

    encoder: Any
    decoder: Any
    feature_predictors: Mapping[str, Any]
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        encoder = _matrix(self.encoder, "boundary encoder")
        decoder = _matrix(self.decoder, "boundary decoder")
        if int(encoder.shape[1]) != int(decoder.shape[0]):
            raise ClosedFormCausalCircuitFoldError(
                "Boundary encoder and decoder latent widths differ."
            )
        if int(encoder.shape[0]) != int(decoder.shape[1]):
            raise ClosedFormCausalCircuitFoldError(
                "Boundary encoder and decoder source widths differ."
            )
        predictors = {
            _text(name, "feature predictor name"): _matrix(
                value,
                f"feature predictor {name}",
            ).clone()
            for name, value in self.feature_predictors.items()
        }
        if any(
            int(value.shape[0]) != int(encoder.shape[1])
            for value in predictors.values()
        ):
            raise ClosedFormCausalCircuitFoldError(
                "A feature predictor has the wrong latent input width."
            )
        object.__setattr__(self, "encoder", encoder.clone())
        object.__setattr__(self, "decoder", decoder.clone())
        object.__setattr__(
            self,
            "feature_predictors",
            MappingProxyType(predictors),
        )
        object.__setattr__(
            self,
            "report",
            MappingProxyType(dict(self.report)),
        )

    @property
    def source_width(self) -> int:
        return int(self.encoder.shape[0])

    @property
    def target_width(self) -> int:
        return int(self.encoder.shape[1])

    def encode(self, source_rows: Any) -> Any:
        rows = _matrix(source_rows, "source_rows", width=self.source_width)
        return (rows @ self.encoder).contiguous()

    def decode(self, target_rows: Any) -> Any:
        rows = _matrix(target_rows, "target_rows", width=self.target_width)
        return (rows @ self.decoder).contiguous()

    def predict_feature(self, name: str, source_rows: Any) -> Any:
        key = _text(name, "feature name")
        try:
            predictor = self.feature_predictors[key]
        except KeyError as exc:
            raise ClosedFormCausalCircuitFoldError(
                f"No predictor was fitted for feature {key!r}."
            ) from exc
        return (self.encode(source_rows) @ predictor).contiguous()

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def _feature_table(
    blocks: Sequence[CausalFeatureBlock],
    *,
    rows: int,
    expected: Mapping[str, CausalFeatureBlock] | None = None,
) -> dict[str, CausalFeatureBlock]:
    result: dict[str, CausalFeatureBlock] = {}
    for block in blocks:
        if not isinstance(block, CausalFeatureBlock):
            raise ClosedFormCausalCircuitFoldError(
                "Feature blocks must use CausalFeatureBlock."
            )
        if block.name in result:
            raise ClosedFormCausalCircuitFoldError(
                f"Duplicate feature block {block.name!r}."
            )
        if int(block.values.shape[0]) != rows:
            raise ClosedFormCausalCircuitFoldError(
                f"Feature block {block.name!r} has the wrong row count."
            )
        result[block.name] = block
    if not result:
        raise ClosedFormCausalCircuitFoldError(
            "At least one causal feature block is required."
        )
    if expected is not None:
        if set(result) != set(expected):
            raise ClosedFormCausalCircuitFoldError(
                "Fit and validation feature names differ."
            )
        for name, block in result.items():
            reference = expected[name]
            if (
                block.role != reference.role
                or int(block.values.shape[1]) != int(reference.values.shape[1])
                or block.weight != reference.weight
            ):
                raise ClosedFormCausalCircuitFoldError(
                    f"Fit and validation feature contract differs for {name!r}."
                )
    return result


def fit_predictive_boundary_map(
    fit_states: Any,
    fit_features: Sequence[CausalFeatureBlock],
    *,
    target_width: int,
    validation_states: Any,
    validation_features: Sequence[CausalFeatureBlock],
    fit_weights: Any | None = None,
    validation_weights: Any | None = None,
    state_reconstruction_weight: float = 0.25,
    ridge: float = 1e-6,
    minimum_predictive_energy_fraction: float = 0.90,
    minimum_validation_block_retention: float = 0.0,
    enforce_validation_block_floor: bool = True,
) -> PredictiveBoundaryMap:
    """Fit one predictive-state boundary map from disjoint cached rows.

    The eigensystem is the whitened cross-moment between residual states and a
    role-balanced concatenation of future/action evidence.  The source state
    itself is included as a lower-weight view so the map cannot preserve a
    scalar action probe while destroying every residual coordinate.
    """

    torch = _torch()
    fit = _matrix(fit_states, "fit_states")
    validation = _matrix(
        validation_states,
        "validation_states",
        width=int(fit.shape[1]),
    )
    if fit.data_ptr() == validation.data_ptr():
        raise ClosedFormCausalCircuitFoldError(
            "Fit and validation state tensors must be disjoint objects."
        )
    target_width = _positive_int(target_width, "target_width")
    source_width = int(fit.shape[1])
    if target_width > source_width:
        raise ClosedFormCausalCircuitFoldError(
            "Predictive boundary expansion is not a fold."
        )
    fit_table = _feature_table(fit_features, rows=int(fit.shape[0]))
    validation_table = _feature_table(
        validation_features,
        rows=int(validation.shape[0]),
        expected=fit_table,
    )
    fit_weight = _weights(fit_weights, int(fit.shape[0]), "fit_weights")
    validation_weight = _weights(
        validation_weights,
        int(validation.shape[0]),
        "validation_weights",
    )
    state_weight = _nonnegative_float(
        state_reconstruction_weight,
        "state_reconstruction_weight",
    )
    ridge = _nonnegative_float(ridge, "ridge")
    required_energy = _unit_fraction(
        minimum_predictive_energy_fraction,
        "minimum_predictive_energy_fraction",
    )
    required_validation = _unit_fraction(
        minimum_validation_block_retention,
        "minimum_validation_block_retention",
    )
    if not isinstance(enforce_validation_block_floor, bool):
        raise ClosedFormCausalCircuitFoldError(
            "enforce_validation_block_floor must be bool."
        )

    normalized_views = []
    view_report = []
    if state_weight > 0.0:
        scale = float(
            torch.sqrt(
                (fit_weight.unsqueeze(1) * fit.square()).sum()
                / (fit_weight.sum() * source_width)
            )
            .clamp_min(1e-30)
            .item()
        )
        normalized_views.append(math.sqrt(state_weight) * fit / scale)
        view_report.append(
            {
                "name": "__state__",
                "role": "residual_reconstruction",
                "width": source_width,
                "weight": state_weight,
                "fit_scale": scale,
            }
        )
    for block in fit_table.values():
        width = int(block.values.shape[1])
        scale = float(
            torch.sqrt(
                (
                    fit_weight.unsqueeze(1) * block.values.square()
                ).sum()
                / (fit_weight.sum() * width)
            )
            .clamp_min(1e-30)
            .item()
        )
        normalized_views.append(
            math.sqrt(block.weight) * block.values / scale
        )
        view_report.append(
            {
                "name": block.name,
                "role": block.role,
                "width": width,
                "weight": block.weight,
                "fit_scale": scale,
            }
        )
    if not normalized_views:
        raise ClosedFormCausalCircuitFoldError(
            "No positive predictive view was supplied."
        )
    observables = torch.cat(normalized_views, dim=1)
    root = torch.sqrt(fit_weight / fit_weight.sum()).unsqueeze(1)
    weighted_states = fit * root
    weighted_observables = observables * root
    covariance = weighted_states.T @ weighted_states
    eigenvalues, eigenvectors = torch.linalg.eigh(
        covariance
        + ridge * torch.eye(source_width, dtype=torch.float64)
    )
    inverse_root = (
        eigenvectors
        @ torch.diag(eigenvalues.clamp_min(1e-30).rsqrt())
        @ eigenvectors.T
    )
    cross = weighted_states.T @ weighted_observables
    predictive_gram = inverse_root @ cross @ cross.T @ inverse_root
    predictive_gram = 0.5 * (predictive_gram + predictive_gram.T)
    values, vectors = torch.linalg.eigh(predictive_gram)
    order = torch.arange(source_width - 1, -1, -1, dtype=torch.long)
    values = values.index_select(0, order).clamp_min(0.0)
    vectors = vectors.index_select(1, order)
    vectors = _canonicalize_columns(vectors)
    encoder = (inverse_root @ vectors[:, :target_width]).contiguous()

    total_predictive_energy = float(values.sum().item())
    if total_predictive_energy <= 1e-30:
        raise ClosedFormCausalCircuitFoldError(
            "Predictive evidence has zero usable cross-moment energy."
        )
    cumulative = torch.cumsum(values, dim=0) / values.sum()
    required_width = int(
        torch.nonzero(
            cumulative >= required_energy,
            as_tuple=False,
        )[0].item()
    ) + 1
    retained_energy = float(cumulative[target_width - 1].item())
    if retained_energy + 1e-12 < required_energy:
        raise ClosedFormCausalCircuitCapacityError(
            "Requested target width retains only "
            f"{retained_energy:.6f} predictive energy; width "
            f"{required_width} is the minimum sealed-evidence width for "
            f"{required_energy:.6f}."
        )

    fit_latent = fit @ encoder
    validation_latent = validation @ encoder
    decoder = _ridge_coefficients(
        fit_latent,
        fit,
        fit_weight,
        ridge,
    )
    decoded_validation = validation_latent @ decoder
    state_retention = _energy_retention(
        decoded_validation,
        validation,
        validation_weight,
    )
    predictors: dict[str, Any] = {}
    heldout_blocks: dict[str, Any] = {}
    minimum_observed = 1.0
    for name, block in fit_table.items():
        predictor = _ridge_coefficients(
            fit_latent,
            block.values,
            fit_weight,
            ridge,
        )
        prediction = validation_latent @ predictor
        target = validation_table[name].values
        retention = _energy_retention(
            prediction,
            target,
            validation_weight,
        )
        minimum_observed = min(minimum_observed, retention)
        heldout_blocks[name] = {
            "role": block.role,
            "retained_energy_fraction": retention,
            "relative_rmse": _relative_rmse(
                prediction,
                target,
                validation_weight,
            ),
        }
        predictors[name] = predictor
    if (
        enforce_validation_block_floor
        and minimum_observed + 1e-12 < required_validation
    ):
        raise ClosedFormCausalCircuitCapacityError(
            "The predictive boundary map failed its held-out causal feature "
            f"floor: {minimum_observed:.6f} < {required_validation:.6f}."
        )

    report_body = {
        "schema": _SCHEMA,
        "component": "predictive_boundary_map",
        "method": "whitened_role_balanced_future_cross_moment",
        "source_width": source_width,
        "target_width": target_width,
        "fit_rows": int(fit.shape[0]),
        "validation_rows": int(validation.shape[0]),
        "fit_validation_disjoint_objects": True,
        "views": view_report,
        "minimum_predictive_energy_fraction": required_energy,
        "retained_predictive_energy_fraction": retained_energy,
        "minimum_width_at_predictive_energy_floor": required_width,
        "validation_state_retained_energy_fraction": state_retention,
        "validation_state_relative_rmse": _relative_rmse(
            decoded_validation,
            validation,
            validation_weight,
        ),
        "minimum_validation_block_retention": required_validation,
        "validation_block_floor_enforced": (
            enforce_validation_block_floor
        ),
        "minimum_observed_validation_block_retention": minimum_observed,
        "validation_blocks": heldout_blocks,
        "ridge": ridge,
        "uncentered_residual_coordinates": True,
        "numeric_weight_matching_objective": False,
        "fit_states_sha256": _tensor_sha256(fit),
        "validation_states_sha256": _tensor_sha256(validation),
        "encoder_sha256": _tensor_sha256(encoder),
        "decoder_sha256": _tensor_sha256(decoder),
    }
    report = {
        **report_body,
        "report_sha256": _canonical_sha256(report_body),
    }
    return PredictiveBoundaryMap(
        encoder=encoder,
        decoder=decoder,
        feature_predictors=predictors,
        report=report,
    )


@dataclass(frozen=True, slots=True)
class DiagonalOperatorClosureAudit:
    """Exact loss incurred by forcing a transformed diagonal to stay diagonal."""

    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "report",
            MappingProxyType(dict(self.report)),
        )

    @property
    def maximum_off_diagonal_energy_fraction(self) -> float:
        return float(self.report["maximum_off_diagonal_energy_fraction"])

    @property
    def closed(self) -> bool:
        return bool(self.report["closed"])

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def audit_diagonal_operator_closure(
    basis: Any,
    diagonal_operators: Any,
    *,
    tolerance: float = 1e-10,
) -> DiagonalOperatorClosureAudit:
    """Measure off-diagonal terms discarded by ``diag(B.T diag(d) B)``."""

    torch = _torch()
    matrix = _matrix(basis, "basis")
    diagonals = _matrix(
        diagonal_operators,
        "diagonal_operators",
        width=int(matrix.shape[0]),
    )
    tolerance = _nonnegative_float(tolerance, "tolerance")
    fractions = []
    for diagonal in diagonals:
        transformed = matrix.T @ (diagonal.unsqueeze(1) * matrix)
        diagonal_part = torch.diag(torch.diagonal(transformed))
        total = transformed.square().sum().clamp_min(1e-30)
        off = (transformed - diagonal_part).square().sum()
        fractions.append(float((off / total).item()))
    maximum = max(fractions)
    mean = sum(fractions) / len(fractions)
    report_body = {
        "schema": _SCHEMA,
        "component": "diagonal_operator_closure_audit",
        "source_width": int(matrix.shape[0]),
        "target_width": int(matrix.shape[1]),
        "operator_count": int(diagonals.shape[0]),
        "off_diagonal_energy_fractions": fractions,
        "mean_off_diagonal_energy_fraction": mean,
        "maximum_off_diagonal_energy_fraction": maximum,
        "tolerance": tolerance,
        "closed": maximum <= tolerance,
        "basis_sha256": _tensor_sha256(matrix),
        "operators_sha256": _tensor_sha256(diagonals),
    }
    return DiagonalOperatorClosureAudit(
        {
            **report_body,
            "report_sha256": _canonical_sha256(report_body),
        }
    )


@dataclass(frozen=True, slots=True)
class ClosedFormRmsNormFold:
    weight: Any
    epsilon: float
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        torch = _torch()
        weight = self.weight
        if (
            not isinstance(weight, torch.Tensor)
            or weight.ndim != 1
            or not torch.is_floating_point(weight)
        ):
            raise ClosedFormCausalCircuitFoldError(
                "RMSNorm weight must be a floating vector."
            )
        object.__setattr__(
            self,
            "weight",
            weight.detach().to(device="cpu", dtype=torch.float64).clone(),
        )
        object.__setattr__(
            self,
            "epsilon",
            _nonnegative_float(self.epsilon, "epsilon"),
        )
        object.__setattr__(
            self,
            "report",
            MappingProxyType(dict(self.report)),
        )

    def predict(self, child_rows: Any) -> Any:
        rows = _matrix(
            child_rows,
            "child_rows",
            width=int(self.weight.numel()),
        )
        normalized = rows * (
            rows.square().mean(dim=1, keepdim=True) + self.epsilon
        ).rsqrt()
        return (normalized * self.weight.unsqueeze(0)).contiguous()


def fit_rmsnorm_diagonal(
    child_rows: Any,
    target_normalized_rows: Any,
    *,
    epsilon: float,
    sample_weights: Any | None = None,
    ridge: float = 1e-6,
) -> ClosedFormRmsNormFold:
    """Fit target-coordinate RMSNorm gains instead of projecting source gains."""

    rows = _matrix(child_rows, "child_rows")
    targets = _matrix(
        target_normalized_rows,
        "target_normalized_rows",
        width=int(rows.shape[1]),
    )
    if int(rows.shape[0]) != int(targets.shape[0]):
        raise ClosedFormCausalCircuitFoldError(
            "RMSNorm evidence rows differ."
        )
    epsilon = _nonnegative_float(epsilon, "epsilon")
    ridge = _nonnegative_float(ridge, "ridge")
    weights = _weights(sample_weights, int(rows.shape[0]), "sample_weights")
    normalized = rows * (
        rows.square().mean(dim=1, keepdim=True) + epsilon
    ).rsqrt()
    numerator = (weights.unsqueeze(1) * normalized * targets).sum(dim=0)
    denominator = (
        weights.unsqueeze(1) * normalized.square()
    ).sum(dim=0) + ridge
    gains = numerator / denominator
    prediction = normalized * gains.unsqueeze(0)
    body = {
        "schema": _SCHEMA,
        "component": "target_coordinate_rmsnorm_fit",
        "width": int(rows.shape[1]),
        "rows": int(rows.shape[0]),
        "epsilon": epsilon,
        "ridge": ridge,
        "relative_rmse": _relative_rmse(prediction, targets, weights),
        "retained_energy_fraction": _energy_retention(
            prediction,
            targets,
            weights,
        ),
        "source_gain_projection_used": False,
        "weight_sha256": _tensor_sha256(gains),
    }
    return ClosedFormRmsNormFold(
        weight=gains,
        epsilon=epsilon,
        report={**body, "report_sha256": _canonical_sha256(body)},
    )


@dataclass(frozen=True, slots=True)
class BilinearAttentionScoreFold:
    query_weight: Any
    key_weight: Any
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        query = _matrix(self.query_weight, "query_weight")
        key = _matrix(self.key_weight, "key_weight")
        if int(query.shape[0]) != int(key.shape[0]):
            raise ClosedFormCausalCircuitFoldError(
                "Q/K fitted head dimensions differ."
            )
        object.__setattr__(self, "query_weight", query.clone())
        object.__setattr__(self, "key_weight", key.clone())
        object.__setattr__(
            self,
            "report",
            MappingProxyType(dict(self.report)),
        )

    @property
    def head_dim(self) -> int:
        return int(self.query_weight.shape[0])

    def predict(self, query_rows: Any, key_rows: Any) -> Any:
        query = _matrix(
            query_rows,
            "query_rows",
            width=int(self.query_weight.shape[1]),
        )
        key = _matrix(
            key_rows,
            "key_rows",
            width=int(self.key_weight.shape[1]),
        )
        if int(query.shape[0]) != int(key.shape[0]):
            raise ClosedFormCausalCircuitFoldError(
                "Q/K score pair counts differ."
            )
        return (
            (query @ self.query_weight.T)
            * (key @ self.key_weight.T)
        ).sum(dim=1)


def fit_bilinear_attention_scores(
    query_rows: Any,
    key_rows: Any,
    centered_target_scores: Any,
    *,
    head_dim: int,
    sample_weights: Any | None = None,
    ridge: float = 1e-6,
    maximum_pairs: int = 2048,
) -> BilinearAttentionScoreFold:
    """Solve cached QK score pairs in dual form, then rank-factor the kernel."""

    torch = _torch()
    query = _matrix(query_rows, "query_rows")
    key = _matrix(key_rows, "key_rows")
    if int(query.shape[0]) != int(key.shape[0]):
        raise ClosedFormCausalCircuitFoldError(
            "Q/K score evidence pair counts differ."
        )
    rows = int(query.shape[0])
    maximum_pairs = _positive_int(maximum_pairs, "maximum_pairs")
    if rows > maximum_pairs:
        raise ClosedFormCausalCircuitFoldError(
            f"Score pair count {rows} exceeds the dual solve ceiling "
            f"{maximum_pairs}."
        )
    head_dim = _positive_int(head_dim, "head_dim")
    if head_dim > min(int(query.shape[1]), int(key.shape[1]), rows):
        raise ClosedFormCausalCircuitFoldError(
            "head_dim exceeds the evidence/kernel rank ceiling."
        )
    if (
        not isinstance(centered_target_scores, torch.Tensor)
        or centered_target_scores.ndim != 1
        or int(centered_target_scores.numel()) != rows
    ):
        raise ClosedFormCausalCircuitFoldError(
            "centered_target_scores must contain one value per Q/K pair."
        )
    scores = (
        centered_target_scores.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
    )
    if not bool(torch.isfinite(scores).all().item()):
        raise ClosedFormCausalCircuitFoldError(
            "centered_target_scores are non-finite."
        )
    weights = _weights(sample_weights, rows, "sample_weights")
    ridge = _nonnegative_float(ridge, "ridge")
    roots = torch.sqrt(weights)
    gram = (query @ query.T) * (key @ key.T)
    weighted_gram = roots.unsqueeze(1) * gram * roots.unsqueeze(0)
    alpha = torch.linalg.solve(
        weighted_gram + ridge * torch.eye(rows, dtype=torch.float64),
        roots * scores,
    )
    effective = query.T @ (
        (roots * alpha).unsqueeze(1) * key
    )
    left, singular, right = torch.linalg.svd(
        effective,
        full_matrices=False,
    )
    singular = singular[:head_dim].clamp_min(0.0)
    root_singular = torch.sqrt(singular)
    query_weight = (
        left[:, :head_dim] * root_singular.unsqueeze(0)
    ).T.contiguous()
    key_weight = (
        right[:head_dim, :] * root_singular.unsqueeze(1)
    ).contiguous()
    prediction = (
        (query @ query_weight.T) * (key @ key_weight.T)
    ).sum(dim=1)
    body = {
        "schema": _SCHEMA,
        "component": "bilinear_attention_score_fold",
        "method": "dual_kernel_ridge_then_truncated_svd",
        "pair_count": rows,
        "query_width": int(query.shape[1]),
        "key_width": int(key.shape[1]),
        "target_head_dim": head_dim,
        "ridge": ridge,
        "relative_rmse": _relative_rmse(
            prediction.unsqueeze(1),
            scores.unsqueeze(1),
            weights,
        ),
        "retained_score_energy_fraction": _energy_retention(
            prediction.unsqueeze(1),
            scores.unsqueeze(1),
            weights,
        ),
        "attention_scale_absorbed_into_factors": True,
        "raw_qk_weight_matching_objective": False,
        "query_weight_sha256": _tensor_sha256(query_weight),
        "key_weight_sha256": _tensor_sha256(key_weight),
    }
    return BilinearAttentionScoreFold(
        query_weight=query_weight,
        key_weight=key_weight,
        report={**body, "report_sha256": _canonical_sha256(body)},
    )


@dataclass(frozen=True, slots=True)
class AttentionValueOutputFold:
    value_weight: Any
    output_weight: Any
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        value = _matrix(self.value_weight, "value_weight")
        output = _matrix(self.output_weight, "output_weight")
        if int(value.shape[0]) != int(output.shape[1]):
            raise ClosedFormCausalCircuitFoldError(
                "V/O fitted channel widths differ."
            )
        object.__setattr__(self, "value_weight", value.clone())
        object.__setattr__(self, "output_weight", output.clone())
        object.__setattr__(
            self,
            "report",
            MappingProxyType(dict(self.report)),
        )

    def predict(self, attended_input_rows: Any) -> Any:
        rows = _matrix(
            attended_input_rows,
            "attended_input_rows",
            width=int(self.value_weight.shape[1]),
        )
        return (
            rows @ self.value_weight.T @ self.output_weight.T
        ).contiguous()


def fit_attention_value_output(
    attended_input_rows: Any,
    target_residual_deltas: Any,
    *,
    value_width: int,
    sample_weights: Any | None = None,
    ridge: float = 1e-6,
) -> AttentionValueOutputFold:
    """Fit the effective fixed-attention residual writer and rank-factor V/O."""

    contexts = _matrix(attended_input_rows, "attended_input_rows")
    targets = _matrix(target_residual_deltas, "target_residual_deltas")
    if int(contexts.shape[0]) != int(targets.shape[0]):
        raise ClosedFormCausalCircuitFoldError(
            "V/O context and residual row counts differ."
        )
    value_width = _positive_int(value_width, "value_width")
    if value_width > min(int(contexts.shape[1]), int(targets.shape[1])):
        raise ClosedFormCausalCircuitFoldError(
            "value_width exceeds the effective residual-writer rank ceiling."
        )
    weights = _weights(
        sample_weights,
        int(contexts.shape[0]),
        "sample_weights",
    )
    ridge = _nonnegative_float(ridge, "ridge")
    effective = _ridge_coefficients(contexts, targets, weights, ridge)
    left, singular, right = _torch().linalg.svd(
        effective,
        full_matrices=False,
    )
    singular = singular[:value_width].clamp_min(0.0)
    root = _torch().sqrt(singular)
    value_weight = (
        left[:, :value_width] * root.unsqueeze(0)
    ).T.contiguous()
    output_weight = (
        right[:value_width, :].T * root.unsqueeze(0)
    ).contiguous()
    prediction = contexts @ value_weight.T @ output_weight.T
    body = {
        "schema": _SCHEMA,
        "component": "attention_value_output_fold",
        "method": "fixed_attention_effective_writer_ridge_svd",
        "rows": int(contexts.shape[0]),
        "input_width": int(contexts.shape[1]),
        "output_width": int(targets.shape[1]),
        "value_width": value_width,
        "ridge": ridge,
        "relative_rmse": _relative_rmse(prediction, targets, weights),
        "retained_residual_energy_fraction": _energy_retention(
            prediction,
            targets,
            weights,
        ),
        "raw_value_weight_matching_objective": False,
        "value_weight_sha256": _tensor_sha256(value_weight),
        "output_weight_sha256": _tensor_sha256(output_weight),
    }
    return AttentionValueOutputFold(
        value_weight=value_weight,
        output_weight=output_weight,
        report={**body, "report_sha256": _canonical_sha256(body)},
    )


def _pivot_function_columns(
    values: Any,
    count: int,
    weights: Any,
) -> tuple[int, ...]:
    torch = _torch()
    count = _positive_int(count, "count")
    if count > int(values.shape[1]):
        raise ClosedFormCausalCircuitFoldError(
            "Cannot select more functional columns than the source exposes."
        )
    weighted = values * torch.sqrt(weights).unsqueeze(1)
    residual = weighted.clone()
    available = torch.ones(int(values.shape[1]), dtype=torch.bool)
    selected = []
    basis = []
    for _ in range(count):
        energies = residual.square().sum(dim=0)
        energies = torch.where(
            available,
            energies,
            torch.full_like(energies, -1.0),
        )
        index = int(energies.argmax().item())
        selected.append(index)
        available[index] = False
        vector = residual[:, index]
        norm = torch.linalg.vector_norm(vector)
        if float(norm.item()) <= 1e-30:
            continue
        direction = vector / norm
        basis.append(direction)
        residual = residual - direction.unsqueeze(1) * (
            direction.unsqueeze(0) @ residual
        )
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class FunctionalSwiGLUFold:
    gate_weight: Any
    up_weight: Any
    down_weight: Any
    selected_source_neurons: tuple[int, ...]
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        gate = _matrix(self.gate_weight, "gate_weight")
        up = _matrix(self.up_weight, "up_weight")
        down = _matrix(self.down_weight, "down_weight")
        if gate.shape != up.shape or int(gate.shape[0]) != int(down.shape[1]):
            raise ClosedFormCausalCircuitFoldError(
                "SwiGLU fitted tensor shapes are inconsistent."
            )
        selected = tuple(
            _positive_int(index + 1, "selected neuron") - 1
            for index in self.selected_source_neurons
        )
        if len(selected) != int(gate.shape[0]) or len(set(selected)) != len(
            selected
        ):
            raise ClosedFormCausalCircuitFoldError(
                "SwiGLU selected-neuron identity is inconsistent."
            )
        object.__setattr__(self, "gate_weight", gate.clone())
        object.__setattr__(self, "up_weight", up.clone())
        object.__setattr__(self, "down_weight", down.clone())
        object.__setattr__(self, "selected_source_neurons", selected)
        object.__setattr__(
            self,
            "report",
            MappingProxyType(dict(self.report)),
        )

    def predict(self, child_input_rows: Any) -> Any:
        torch = _torch()
        rows = _matrix(
            child_input_rows,
            "child_input_rows",
            width=int(self.gate_weight.shape[1]),
        )
        features = torch.nn.functional.silu(
            rows @ self.gate_weight.T
        ) * (rows @ self.up_weight.T)
        return (features @ self.down_weight.T).contiguous()


def fit_functional_swiglu(
    child_input_rows: Any,
    source_gate_rows: Any,
    source_up_rows: Any,
    complete_target_residual_rows: Any,
    *,
    target_neurons: int,
    sample_weights: Any | None = None,
    ridge: float = 1e-6,
) -> FunctionalSwiGLUFold:
    """Compile a smaller SwiGLU dictionary against the complete FFN write."""

    torch = _torch()
    inputs = _matrix(child_input_rows, "child_input_rows")
    gates = _matrix(source_gate_rows, "source_gate_rows")
    ups = _matrix(
        source_up_rows,
        "source_up_rows",
        width=int(gates.shape[1]),
    )
    targets = _matrix(
        complete_target_residual_rows,
        "complete_target_residual_rows",
    )
    rows = int(inputs.shape[0])
    if (
        int(gates.shape[0]) != rows
        or int(ups.shape[0]) != rows
        or int(targets.shape[0]) != rows
    ):
        raise ClosedFormCausalCircuitFoldError(
            "SwiGLU evidence row counts differ."
        )
    target_neurons = _positive_int(target_neurons, "target_neurons")
    if target_neurons > int(gates.shape[1]):
        raise ClosedFormCausalCircuitFoldError(
            "SwiGLU expansion is outside this fold."
        )
    weights = _weights(sample_weights, rows, "sample_weights")
    ridge = _nonnegative_float(ridge, "ridge")
    all_gate_weights = _ridge_coefficients(inputs, gates, weights, ridge).T
    all_up_weights = _ridge_coefficients(inputs, ups, weights, ridge).T
    realizable_features = torch.nn.functional.silu(
        inputs @ all_gate_weights.T
    ) * (inputs @ all_up_weights.T)
    selected = _pivot_function_columns(
        realizable_features,
        target_neurons,
        weights,
    )
    index = torch.tensor(selected, dtype=torch.long)
    gate_weight = all_gate_weights.index_select(0, index).contiguous()
    up_weight = all_up_weights.index_select(0, index).contiguous()
    child_features = torch.nn.functional.silu(
        inputs @ gate_weight.T
    ) * (inputs @ up_weight.T)
    down_weight = _ridge_coefficients(
        child_features,
        targets,
        weights,
        ridge,
    ).T.contiguous()
    prediction = child_features @ down_weight.T
    source_features = torch.nn.functional.silu(gates) * ups
    body = {
        "schema": _SCHEMA,
        "component": "functional_swiglu_fold",
        "method": (
            "closed_form_producer_realizability_pivot_complete_residual_ridge"
        ),
        "rows": rows,
        "input_width": int(inputs.shape[1]),
        "source_neurons": int(gates.shape[1]),
        "target_neurons": target_neurons,
        "selected_source_neurons": list(selected),
        "all_source_gate_up_rows_consumed": True,
        "complete_parent_residual_target_consumed": True,
        "source_feature_matrix_sha256": _tensor_sha256(source_features),
        "ridge": ridge,
        "relative_rmse": _relative_rmse(prediction, targets, weights),
        "retained_residual_energy_fraction": _energy_retention(
            prediction,
            targets,
            weights,
        ),
        "numeric_swiglu_weight_matching_objective": False,
        "gate_weight_sha256": _tensor_sha256(gate_weight),
        "up_weight_sha256": _tensor_sha256(up_weight),
        "down_weight_sha256": _tensor_sha256(down_weight),
    }
    return FunctionalSwiGLUFold(
        gate_weight=gate_weight,
        up_weight=up_weight,
        down_weight=down_weight,
        selected_source_neurons=selected,
        report={**body, "report_sha256": _canonical_sha256(body)},
    )


def _causal_depthwise_context(
    b_values: Any,
    c_values: Any,
    x_values: Any,
    kernels: Any,
) -> Any:
    torch = _torch()
    batch, sequence, channels = (
        int(b_values.shape[0]),
        int(b_values.shape[1]),
        int(b_values.shape[2]),
    )
    if (
        tuple(c_values.shape) != (batch, sequence, channels)
        or tuple(x_values.shape) != (batch, sequence, channels)
        or int(kernels.shape[0]) != channels
    ):
        raise ClosedFormCausalCircuitFoldError(
            "Short-convolution functional evidence has incompatible shapes."
        )
    gated = b_values * x_values
    filtered = torch.zeros_like(gated)
    length = int(kernels.shape[1])
    for lag in range(length):
        tap = kernels[:, length - 1 - lag]
        if lag == 0:
            filtered = filtered + gated * tap.reshape(1, 1, channels)
        elif lag < sequence:
            filtered[:, lag:, :] = (
                filtered[:, lag:, :]
                + gated[:, :-lag, :] * tap.reshape(1, 1, channels)
            )
    return (c_values * filtered).contiguous()


@dataclass(frozen=True, slots=True)
class FunctionalShortConvFold:
    input_weight: Any
    depthwise_kernel: Any
    output_weight: Any
    selected_source_channels: tuple[int, ...]
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        input_weight = _matrix(self.input_weight, "input_weight")
        kernel = _matrix(self.depthwise_kernel, "depthwise_kernel")
        output = _matrix(self.output_weight, "output_weight")
        if (
            int(input_weight.shape[0]) != 3 * int(kernel.shape[0])
            or int(output.shape[1]) != int(kernel.shape[0])
        ):
            raise ClosedFormCausalCircuitFoldError(
                "Short-convolution fitted tensor shapes are inconsistent."
            )
        selected = tuple(self.selected_source_channels)
        if len(selected) != int(kernel.shape[0]) or len(set(selected)) != len(
            selected
        ):
            raise ClosedFormCausalCircuitFoldError(
                "Short-convolution selected-channel identity is inconsistent."
            )
        object.__setattr__(self, "input_weight", input_weight.clone())
        object.__setattr__(self, "depthwise_kernel", kernel.clone())
        object.__setattr__(self, "output_weight", output.clone())
        object.__setattr__(self, "selected_source_channels", selected)
        object.__setattr__(
            self,
            "report",
            MappingProxyType(dict(self.report)),
        )

    def predict(self, child_input_sequences: Any) -> Any:
        inputs = _tensor(
            child_input_sequences,
            "child_input_sequences",
            dimensions=3,
        )
        if int(inputs.shape[2]) != int(self.input_weight.shape[1]):
            raise ClosedFormCausalCircuitFoldError(
                "Short-convolution child input width differs."
            )
        projected = inputs @ self.input_weight.T
        b_values, c_values, x_values = projected.chunk(3, dim=2)
        context = _causal_depthwise_context(
            b_values,
            c_values,
            x_values,
            self.depthwise_kernel,
        )
        return (context @ self.output_weight.T).contiguous()


def fit_functional_short_conv(
    child_input_sequences: Any,
    source_b_values: Any,
    source_c_values: Any,
    source_x_values: Any,
    source_depthwise_kernels: Any,
    complete_target_residual_sequences: Any,
    *,
    target_channels: int,
    sample_weights: Any | None = None,
    ridge: float = 1e-6,
) -> FunctionalShortConvFold:
    """Compile LFM2 B/C/x/FIR channels by their cached residual function."""

    torch = _torch()
    inputs = _tensor(
        child_input_sequences,
        "child_input_sequences",
        dimensions=3,
    )
    b_values = _tensor(source_b_values, "source_b_values", dimensions=3)
    c_values = _tensor(source_c_values, "source_c_values", dimensions=3)
    x_values = _tensor(source_x_values, "source_x_values", dimensions=3)
    kernels = _matrix(
        source_depthwise_kernels,
        "source_depthwise_kernels",
    )
    targets = _tensor(
        complete_target_residual_sequences,
        "complete_target_residual_sequences",
        dimensions=3,
    )
    batch, sequence, input_width = (
        int(inputs.shape[0]),
        int(inputs.shape[1]),
        int(inputs.shape[2]),
    )
    channels = int(b_values.shape[2])
    expected = (batch, sequence, channels)
    if (
        tuple(b_values.shape) != expected
        or tuple(c_values.shape) != expected
        or tuple(x_values.shape) != expected
        or int(kernels.shape[0]) != channels
        or tuple(targets.shape[:2]) != (batch, sequence)
    ):
        raise ClosedFormCausalCircuitFoldError(
            "Short-convolution cached evidence shapes differ."
        )
    target_channels = _positive_int(target_channels, "target_channels")
    if target_channels > channels:
        raise ClosedFormCausalCircuitFoldError(
            "Short-convolution expansion is outside this fold."
        )
    row_count = batch * sequence
    weights = _weights(sample_weights, row_count, "sample_weights")
    ridge = _nonnegative_float(ridge, "ridge")
    flat_inputs = inputs.reshape(row_count, input_width)
    flat_b = b_values.reshape(row_count, channels)
    flat_c = c_values.reshape(row_count, channels)
    flat_x = x_values.reshape(row_count, channels)
    b_weight = _ridge_coefficients(
        flat_inputs,
        flat_b,
        weights,
        ridge,
    ).T
    c_weight = _ridge_coefficients(
        flat_inputs,
        flat_c,
        weights,
        ridge,
    ).T
    x_weight = _ridge_coefficients(
        flat_inputs,
        flat_x,
        weights,
        ridge,
    ).T
    realized_b = (flat_inputs @ b_weight.T).reshape(batch, sequence, channels)
    realized_c = (flat_inputs @ c_weight.T).reshape(batch, sequence, channels)
    realized_x = (flat_inputs @ x_weight.T).reshape(batch, sequence, channels)
    realized_context = _causal_depthwise_context(
        realized_b,
        realized_c,
        realized_x,
        kernels,
    )
    selected = _pivot_function_columns(
        realized_context.reshape(row_count, channels),
        target_channels,
        weights,
    )
    index = torch.tensor(selected, dtype=torch.long)
    selected_b = b_weight.index_select(0, index)
    selected_c = c_weight.index_select(0, index)
    selected_x = x_weight.index_select(0, index)
    selected_kernels = kernels.index_select(0, index).contiguous()
    input_weight = torch.cat(
        (selected_b, selected_c, selected_x),
        dim=0,
    ).contiguous()
    selected_context = realized_context.index_select(2, index)
    output_weight = _ridge_coefficients(
        selected_context.reshape(row_count, target_channels),
        targets.reshape(row_count, int(targets.shape[2])),
        weights,
        ridge,
    ).T.contiguous()
    prediction = selected_context @ output_weight.T
    complete_source_context = _causal_depthwise_context(
        b_values,
        c_values,
        x_values,
        kernels,
    )
    body = {
        "schema": _SCHEMA,
        "component": "functional_short_conv_fold",
        "method": (
            "closed_form_bcx_realizability_context_pivot_complete_residual_ridge"
        ),
        "batch": batch,
        "sequence": sequence,
        "input_width": input_width,
        "output_width": int(targets.shape[2]),
        "source_channels": channels,
        "target_channels": target_channels,
        "fir_length": int(kernels.shape[1]),
        "selected_source_channels": list(selected),
        "all_source_bcx_rows_consumed": True,
        "complete_parent_residual_target_consumed": True,
        "complete_source_context_sha256": _tensor_sha256(
            complete_source_context
        ),
        "ridge": ridge,
        "relative_rmse": _relative_rmse(
            prediction.reshape(row_count, int(targets.shape[2])),
            targets.reshape(row_count, int(targets.shape[2])),
            weights,
        ),
        "retained_residual_energy_fraction": _energy_retention(
            prediction.reshape(row_count, int(targets.shape[2])),
            targets.reshape(row_count, int(targets.shape[2])),
            weights,
        ),
        "dense_basis_kernel_diagonalization_used": False,
        "input_weight_sha256": _tensor_sha256(input_weight),
        "depthwise_kernel_sha256": _tensor_sha256(selected_kernels),
        "output_weight_sha256": _tensor_sha256(output_weight),
    }
    return FunctionalShortConvFold(
        input_weight=input_weight,
        depthwise_kernel=selected_kernels,
        output_weight=output_weight,
        selected_source_channels=selected,
        report={**body, "report_sha256": _canonical_sha256(body)},
    )


@dataclass(frozen=True, slots=True)
class DenseParameterSite:
    """One declared target Parameter path and its optional tied alias group."""

    name: str
    shape: tuple[int, ...]
    semantic_role: str
    alias_group: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "site name"))
        shape = tuple(
            _positive_int(value, f"{self.name}.shape")
            for value in self.shape
        )
        if not shape:
            raise ClosedFormCausalCircuitFoldError(
                f"{self.name}.shape cannot be empty."
            )
        object.__setattr__(self, "shape", shape)
        object.__setattr__(
            self,
            "semantic_role",
            _text(self.semantic_role, f"{self.name}.semantic_role"),
        )
        if self.alias_group is not None:
            object.__setattr__(
                self,
                "alias_group",
                _text(self.alias_group, f"{self.name}.alias_group"),
            )

    @property
    def elements(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True, slots=True)
class FoldingOnlyOperationAudit:
    parent_forward_calls: int = 0
    parent_generation_calls: int = 0
    parent_trace_regeneration_calls: int = 0
    training_steps: int = 0
    optimizer_steps: int = 0
    backward_calls: int = 0
    pruned_parent_parameters: int = 0
    quantized_target_parameters: int = 0
    external_adapter_parameters: int = 0

    @property
    def all_zero(self) -> bool:
        return all(
            getattr(self, name) == 0
            for name in (
                "parent_forward_calls",
                "parent_generation_calls",
                "parent_trace_regeneration_calls",
                "training_steps",
                "optimizer_steps",
                "backward_calls",
                "pruned_parent_parameters",
                "quantized_target_parameters",
                "external_adapter_parameters",
            )
        )

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "parent_forward_calls": self.parent_forward_calls,
            "parent_generation_calls": self.parent_generation_calls,
            "parent_trace_regeneration_calls": (
                self.parent_trace_regeneration_calls
            ),
            "training_steps": self.training_steps,
            "optimizer_steps": self.optimizer_steps,
            "backward_calls": self.backward_calls,
            "pruned_parent_parameters": self.pruned_parent_parameters,
            "quantized_target_parameters": self.quantized_target_parameters,
            "external_adapter_parameters": self.external_adapter_parameters,
            "all_zero": self.all_zero,
        }


@dataclass(frozen=True, slots=True)
class CausalCircuitFoldIntegrationContract:
    """Typed boundary between sealed evidence, target synthesis, and install."""

    parent_cache_sha256: str
    fit_split_sha256: str
    validation_split_sha256: str
    target_topology_sha256: str
    target_parameter_count: int
    target_sites: tuple[DenseParameterSite, ...]
    minimum_predictive_energy_fraction: float = 0.90
    minimum_validation_feature_retention: float = 0.90

    def __post_init__(self) -> None:
        for name in (
            "parent_cache_sha256",
            "fit_split_sha256",
            "validation_split_sha256",
            "target_topology_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name),
            )
        if self.fit_split_sha256 == self.validation_split_sha256:
            raise ClosedFormCausalCircuitFoldError(
                "Fit and validation split identities must differ."
            )
        target_count = _positive_int(
            self.target_parameter_count,
            "target_parameter_count",
        )
        sites = tuple(self.target_sites)
        if (
            not sites
            or any(not isinstance(site, DenseParameterSite) for site in sites)
            or len({site.name for site in sites}) != len(sites)
        ):
            raise ClosedFormCausalCircuitFoldError(
                "Target sites must be nonempty and uniquely named."
            )
        grouped: dict[str, DenseParameterSite] = {}
        for site in sites:
            key = site.alias_group or site.name
            previous = grouped.get(key)
            if previous is not None and previous.shape != site.shape:
                raise ClosedFormCausalCircuitFoldError(
                    f"Tied alias group {key!r} changes shape."
                )
            grouped.setdefault(key, site)
        declared = sum(site.elements for site in grouped.values())
        if declared != target_count:
            raise ClosedFormCausalCircuitFoldError(
                "Target site accounting does not equal the exact requested "
                f"parameter count: {declared} != {target_count}."
            )
        object.__setattr__(self, "target_parameter_count", target_count)
        object.__setattr__(self, "target_sites", sites)
        object.__setattr__(
            self,
            "minimum_predictive_energy_fraction",
            _unit_fraction(
                self.minimum_predictive_energy_fraction,
                "minimum_predictive_energy_fraction",
            ),
        )
        object.__setattr__(
            self,
            "minimum_validation_feature_retention",
            _unit_fraction(
                self.minimum_validation_feature_retention,
                "minimum_validation_feature_retention",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "parent_cache_sha256": self.parent_cache_sha256,
            "fit_split_sha256": self.fit_split_sha256,
            "validation_split_sha256": self.validation_split_sha256,
            "target_topology_sha256": self.target_topology_sha256,
            "target_parameter_count": self.target_parameter_count,
            "target_sites": [
                {
                    "name": site.name,
                    "shape": list(site.shape),
                    "semantic_role": site.semantic_role,
                    "alias_group": site.alias_group,
                    "elements": site.elements,
                }
                for site in self.target_sites
            ],
            "minimum_predictive_energy_fraction": (
                self.minimum_predictive_energy_fraction
            ),
            "minimum_validation_feature_retention": (
                self.minimum_validation_feature_retention
            ),
        }


@dataclass(frozen=True, slots=True)
class DenseParameterAudit:
    target_parameter_count: int
    counted_by_role: Mapping[str, int]
    tensor_sha256: Mapping[str, str]
    tied_alias_groups: Mapping[str, tuple[str, ...]]
    report_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "counted_by_role",
            MappingProxyType(dict(self.counted_by_role)),
        )
        object.__setattr__(
            self,
            "tensor_sha256",
            MappingProxyType(dict(self.tensor_sha256)),
        )
        object.__setattr__(
            self,
            "tied_alias_groups",
            MappingProxyType(dict(self.tied_alias_groups)),
        )
        object.__setattr__(
            self,
            "report_sha256",
            _sha256(self.report_sha256, "report_sha256"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_parameter_count": self.target_parameter_count,
            "counted_by_role": dict(self.counted_by_role),
            "tensor_sha256": dict(self.tensor_sha256),
            "tied_alias_groups": {
                key: list(value)
                for key, value in self.tied_alias_groups.items()
            },
            "report_sha256": self.report_sha256,
        }


def audit_dense_target_parameters(
    contract: CausalCircuitFoldIntegrationContract,
    tensors: Mapping[str, Any],
) -> DenseParameterAudit:
    """Prove complete dense target overwrite with no undeclared sidecar state."""

    torch = _torch()
    if not isinstance(contract, CausalCircuitFoldIntegrationContract):
        raise ClosedFormCausalCircuitFoldError(
            "contract has the wrong type."
        )
    values = dict(tensors)
    expected_names = {site.name for site in contract.target_sites}
    if set(values) != expected_names:
        missing = sorted(expected_names.difference(values))
        extra = sorted(set(values).difference(expected_names))
        raise ClosedFormCausalCircuitFoldError(
            f"Target tensor coverage differs; missing={missing}, extra={extra}."
        )
    sites_by_group: dict[str, list[DenseParameterSite]] = {}
    tensor_hashes: dict[str, str] = {}
    identity_groups: dict[int, str] = {}
    for site in contract.target_sites:
        value = values[site.name]
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.device.type == "meta"
            or tuple(int(size) for size in value.shape) != site.shape
            or not torch.is_floating_point(value)
            or torch.is_complex(value)
            or not bool(torch.isfinite(value.detach()).all().item())
        ):
            raise ClosedFormCausalCircuitFoldError(
                f"Target tensor {site.name!r} violates its dense site contract."
            )
        group = site.alias_group or site.name
        sites_by_group.setdefault(group, []).append(site)
        prior = identity_groups.get(id(value))
        if prior is not None and prior != group:
            raise ClosedFormCausalCircuitFoldError(
                "An undeclared tensor alias crosses target parameter groups."
            )
        identity_groups[id(value)] = group
        tensor_hashes[site.name] = _tensor_sha256(value)
    counted_by_role: dict[str, int] = {}
    aliases: dict[str, tuple[str, ...]] = {}
    counted = 0
    for group, sites in sites_by_group.items():
        identities = {id(values[site.name]) for site in sites}
        if len(identities) != 1:
            raise ClosedFormCausalCircuitFoldError(
                f"Declared tied alias group {group!r} is not one tensor object."
            )
        count = sites[0].elements
        roles = "+".join(sorted({site.semantic_role for site in sites}))
        counted_by_role[roles] = counted_by_role.get(roles, 0) + count
        counted += count
        if len(sites) > 1:
            aliases[group] = tuple(sorted(site.name for site in sites))
    if counted != contract.target_parameter_count:
        raise ClosedFormCausalCircuitFoldError(
            "Materialized target parameter count differs from its contract."
        )
    body = {
        "schema": _SCHEMA,
        "component": "dense_target_parameter_audit",
        "target_parameter_count": counted,
        "counted_by_role": counted_by_role,
        "tensor_sha256": tensor_hashes,
        "tied_alias_groups": {
            key: list(value) for key, value in aliases.items()
        },
        "complete_target_overwrite": True,
        "undeclared_parameter_sidecars": 0,
        "quantized_parameters_counted": 0,
        "pruned_parent_parameters_counted": 0,
        "parent_runtime_parameters_counted": 0,
    }
    report_sha256 = _canonical_sha256(body)
    return DenseParameterAudit(
        target_parameter_count=counted,
        counted_by_role=counted_by_role,
        tensor_sha256=tensor_hashes,
        tied_alias_groups=aliases,
        report_sha256=report_sha256,
    )


@dataclass(frozen=True, slots=True)
class CausalCircuitFoldPlan:
    contract: CausalCircuitFoldIntegrationContract
    parameter_audit: DenseParameterAudit
    operation_audit: FoldingOnlyOperationAudit
    component_report_sha256: Mapping[str, str]
    plan_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(
                self.contract,
                CausalCircuitFoldIntegrationContract,
            )
            or not isinstance(self.parameter_audit, DenseParameterAudit)
            or not isinstance(self.operation_audit, FoldingOnlyOperationAudit)
            or not self.operation_audit.all_zero
        ):
            raise ClosedFormCausalCircuitFoldError(
                "A causal circuit plan must be a zero-operation dense fold."
            )
        component_hashes = {
            _text(name, "component name"): _sha256(
                digest,
                f"{name}.report_sha256",
            )
            for name, digest in self.component_report_sha256.items()
        }
        if not component_hashes:
            raise ClosedFormCausalCircuitFoldError(
                "At least one compiled component report is required."
            )
        object.__setattr__(
            self,
            "component_report_sha256",
            MappingProxyType(component_hashes),
        )
        object.__setattr__(
            self,
            "plan_sha256",
            _sha256(self.plan_sha256, "plan_sha256"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "contract": self.contract.as_dict(),
            "parameter_audit": self.parameter_audit.as_dict(),
            "operation_audit": self.operation_audit.as_dict(),
            "component_report_sha256": dict(
                self.component_report_sha256
            ),
            "plan_sha256": self.plan_sha256,
        }


def seal_causal_circuit_fold_plan(
    contract: CausalCircuitFoldIntegrationContract,
    tensors: Mapping[str, Any],
    *,
    component_reports: Mapping[str, Mapping[str, Any]],
    operation_audit: FoldingOnlyOperationAudit | None = None,
) -> CausalCircuitFoldPlan:
    """Seal a complete blank-child materialization after exact accounting."""

    if operation_audit is None:
        operation_audit = FoldingOnlyOperationAudit()
    if not isinstance(operation_audit, FoldingOnlyOperationAudit):
        raise ClosedFormCausalCircuitFoldError(
            "operation_audit has the wrong type."
        )
    if not operation_audit.all_zero:
        raise ClosedFormCausalCircuitFoldError(
            "The plan used a forbidden runtime, recovery, or accounting path."
        )
    parameter_audit = audit_dense_target_parameters(contract, tensors)
    component_hashes: dict[str, str] = {}
    for name, report in component_reports.items():
        key = _text(name, "component name")
        if not isinstance(report, Mapping):
            raise ClosedFormCausalCircuitFoldError(
                f"Component report {key!r} must be a mapping."
            )
        digest = report.get("report_sha256")
        component_hashes[key] = _sha256(
            digest,
            f"{key}.report_sha256",
        )
    body = {
        "schema": _SCHEMA,
        "contract_sha256": _canonical_sha256(contract.as_dict()),
        "parameter_audit_sha256": parameter_audit.report_sha256,
        "operation_audit": operation_audit.as_dict(),
        "component_report_sha256": component_hashes,
        "standalone_dense_child_required": True,
        "parent_at_runtime": False,
        "external_adapter_at_runtime": False,
    }
    return CausalCircuitFoldPlan(
        contract=contract,
        parameter_audit=parameter_audit,
        operation_audit=operation_audit,
        component_report_sha256=component_hashes,
        plan_sha256=_canonical_sha256(body),
    )


__all__ = [
    "AttentionValueOutputFold",
    "BilinearAttentionScoreFold",
    "CausalCircuitFoldIntegrationContract",
    "CausalCircuitFoldPlan",
    "CausalFeatureBlock",
    "ClosedFormCausalCircuitCapacityError",
    "ClosedFormCausalCircuitFoldError",
    "ClosedFormRmsNormFold",
    "DenseParameterAudit",
    "DenseParameterSite",
    "DiagonalOperatorClosureAudit",
    "FoldingOnlyOperationAudit",
    "FunctionalShortConvFold",
    "FunctionalSwiGLUFold",
    "PredictiveBoundaryMap",
    "audit_dense_target_parameters",
    "audit_diagonal_operator_closure",
    "fit_attention_value_output",
    "fit_bilinear_attention_scores",
    "fit_functional_short_conv",
    "fit_functional_swiglu",
    "fit_predictive_boundary_map",
    "fit_rmsnorm_diagonal",
    "seal_causal_circuit_fold_plan",
]
