"""Predictive boundary charts and chart-correct residual-write targets.

Each folded residual boundary may use a different target coordinate chart.
Because a transformer/SSM residual skip is the identity, a child branch must
absorb the chart transition itself:

``child_write = encoded_parent_endpoint - actual_child_pre_state``.

Projecting only the parent residual delta through one global basis omits that
term and is not compositionally correct.  This module makes the transition an
explicit, audited compiler primitive.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.hf_closed_form_causal_circuit_fold import (
    CausalFeatureBlock,
    PredictiveBoundaryMap,
    fit_predictive_boundary_map,
)


_SCHEMA = "neural-forge-cpcf-boundary-chart-v1"
_TRANSPORT_SCHEMA = "neural-forge-cpcf-orthogonal-chart-transport-v1"


class CpcfBoundaryChartError(ValueError):
    """Boundary evidence or a residual transition is invalid."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for CPCF boundary charts."
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
        or value.is_complex()
    ):
        raise CpcfBoundaryChartError(
            f"{name} must be a nonempty dense real floating matrix."
        )
    if width is not None and int(value.shape[1]) != width:
        raise CpcfBoundaryChartError(
            f"{name} width must equal {width}."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise CpcfBoundaryChartError(f"{name} contains NaN or infinity.")
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CpcfBoundaryChartError(f"{name} must be nonempty text.")
    return value.strip()


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
        _canonical_sha256([int(size) for size in tensor.shape]).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


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
        or value.is_complex()
    ):
        raise CpcfBoundaryChartError(
            f"{name} must contain one positive floating value per row."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if (
        not bool(torch.isfinite(result).all().item())
        or bool((result <= 0.0).any().item())
    ):
        raise CpcfBoundaryChartError(
            f"{name} must contain finite positive values."
        )
    return result


def _nonnegative_finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CpcfBoundaryChartError(
            f"{name} must be a finite nonnegative number."
        ) from exc
    if result < 0.0 or result == float("inf") or result != result:
        raise CpcfBoundaryChartError(
            f"{name} must be a finite nonnegative number."
        )
    return result


def _shares_storage(left: Any, right: Any) -> bool:
    torch = _torch()
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return left is right
    if left is right:
        return True
    if left.device != right.device:
        return False
    try:
        return (
            left.untyped_storage().data_ptr()
            == right.untyped_storage().data_ptr()
        )
    except RuntimeError:
        return False


def _require_disjoint_storage(
    fit_value: Any,
    heldout_value: Any,
    name: str,
) -> None:
    if _shares_storage(fit_value, heldout_value):
        raise CpcfBoundaryChartError(
            f"Fit and heldout {name} must not share tensor storage."
        )


def _weighted_rmse(prediction: Any, target: Any, weights: Any) -> float:
    torch = _torch()
    error_energy = (
        weights.unsqueeze(1) * (prediction - target).square()
    ).sum()
    denominator = weights.sum() * int(target.shape[1])
    return float(torch.sqrt(error_energy / denominator).item())


def _maximum_absolute_error(left: Any, right: Any) -> float:
    return float((left - right).abs().max().item())


def _canonical_orthogonal_complement(
    basis: Any,
    *,
    width: int,
    columns: int,
) -> Any:
    """Complete an orthonormal basis using projected coordinate axes."""

    torch = _torch()
    if columns == 0:
        return torch.empty((width, 0), dtype=torch.float64)
    selected: list[Any] = []
    threshold = 64.0 * torch.finfo(torch.float64).eps * max(1, width)
    for coordinate in range(width):
        candidate = torch.zeros(width, dtype=torch.float64)
        candidate[coordinate] = 1.0
        for _ in range(2):
            if int(basis.shape[1]) > 0:
                candidate = candidate - basis @ (basis.T @ candidate)
            if selected:
                complement = torch.stack(selected, dim=1)
                candidate = candidate - complement @ (
                    complement.T @ candidate
                )
        norm = float(torch.linalg.vector_norm(candidate).item())
        if norm <= threshold:
            continue
        candidate = candidate / norm
        pivot = int(candidate.abs().argmax().item())
        if float(candidate[pivot].item()) < 0.0:
            candidate = -candidate
        selected.append(candidate)
        if len(selected) == columns:
            break
    if len(selected) != columns:
        raise CpcfBoundaryChartError(
            "Could not construct a deterministic orthogonal complement."
        )
    return torch.stack(selected, dim=1).contiguous()


def _weighted_orthogonal_procrustes(
    encoded_parent_rows: Any,
    actual_child_rows: Any,
    weights: Any,
) -> tuple[Any, Any, int]:
    """Solve ``encoded_parent_rows @ R ~= actual_child_rows`` in FP64."""

    torch = _torch()
    root = torch.sqrt(weights).unsqueeze(1)
    cross = (encoded_parent_rows * root).T @ (actual_child_rows * root)
    left, singular_values, right_h = torch.linalg.svd(
        cross,
        full_matrices=True,
    )
    largest = float(singular_values.max().item())
    rank_tolerance = (
        max(int(cross.shape[0]), int(cross.shape[1]))
        * torch.finfo(torch.float64).eps
        * largest
    )
    rank = int((singular_values > rank_tolerance).sum().item())
    left_range = left[:, :rank]
    right_range = right_h.T[:, :rank]
    left_null = _canonical_orthogonal_complement(
        left_range,
        width=int(cross.shape[0]),
        columns=int(cross.shape[0]) - rank,
    )
    right_null = _canonical_orthogonal_complement(
        right_range,
        width=int(cross.shape[1]),
        columns=int(cross.shape[1]) - rank,
    )
    rotation = (
        left_range @ right_range.T + left_null @ right_null.T
    ).contiguous()
    return rotation, singular_values.contiguous(), rank


@dataclass(frozen=True, slots=True)
class CpcfBoundaryChart:
    """One named predictive parent-state to child-state coordinate chart."""

    name: str
    mapping: PredictiveBoundaryMap
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "name"))
        if not isinstance(self.mapping, PredictiveBoundaryMap):
            raise TypeError("mapping must be PredictiveBoundaryMap.")
        object.__setattr__(
            self,
            "report",
            MappingProxyType(dict(self.report)),
        )

    @property
    def source_width(self) -> int:
        return self.mapping.source_width

    @property
    def target_width(self) -> int:
        return self.mapping.target_width

    def encode(self, parent_rows: Any) -> Any:
        return self.mapping.encode(parent_rows)

    def encode_shaped(self, parent_states: Any) -> Any:
        torch = _torch()
        if (
            not isinstance(parent_states, torch.Tensor)
            or parent_states.ndim < 2
            or int(parent_states.shape[-1]) != self.source_width
        ):
            raise CpcfBoundaryChartError(
                "parent_states has the wrong rank or source width."
            )
        shape = tuple(int(value) for value in parent_states.shape[:-1])
        rows = _matrix(
            parent_states.reshape(-1, self.source_width),
            "parent_states",
            width=self.source_width,
        )
        return self.encode(rows).reshape(*shape, self.target_width)

    def residual_write_target(
        self,
        *,
        actual_child_pre_state: Any,
        parent_endpoint: Any,
    ) -> tuple[Any, Any]:
        """Return ``(encoded endpoint, endpoint - actual pre-state)``."""

        torch = _torch()
        if not isinstance(actual_child_pre_state, torch.Tensor):
            raise CpcfBoundaryChartError(
                "actual_child_pre_state must be a tensor."
            )
        endpoint = self.encode_shaped(parent_endpoint)
        actual = _matrix(
            actual_child_pre_state.reshape(-1, self.target_width),
            "actual_child_pre_state",
            width=self.target_width,
        ).reshape_as(endpoint)
        if tuple(actual.shape) != tuple(endpoint.shape):
            raise CpcfBoundaryChartError(
                "Actual child state and encoded parent endpoint are misaligned."
            )
        write = endpoint - actual
        if not bool(torch.isfinite(write).all().item()):
            raise CpcfBoundaryChartError(
                "Chart-transition residual target is non-finite."
            )
        return endpoint.contiguous(), write.contiguous()

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


@dataclass(frozen=True, slots=True)
class CpcfBoundaryChartTransport:
    """A gauge-equivalent predictive map aligned to child coordinates."""

    mapping: PredictiveBoundaryMap
    chart: CpcfBoundaryChart | None
    rotation: Any
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.mapping, PredictiveBoundaryMap):
            raise TypeError("mapping must be PredictiveBoundaryMap.")
        if self.chart is not None:
            if not isinstance(self.chart, CpcfBoundaryChart):
                raise TypeError("chart must be CpcfBoundaryChart or None.")
            if self.chart.mapping is not self.mapping:
                raise CpcfBoundaryChartError(
                    "Transported chart and mapping must be the same object."
                )
        rotation = _matrix(
            self.rotation,
            "rotation",
            width=self.mapping.target_width,
        )
        if int(rotation.shape[0]) != self.mapping.target_width:
            raise CpcfBoundaryChartError("rotation must be square.")
        report = dict(self.report)
        seal = report.pop("report_sha256", None)
        if not isinstance(seal, str) or seal != _canonical_sha256(report):
            raise CpcfBoundaryChartError(
                "Chart-transport report seal is invalid."
            )
        report["report_sha256"] = seal
        object.__setattr__(self, "rotation", rotation.clone())
        object.__setattr__(self, "report", MappingProxyType(report))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def fit_cpcf_orthogonal_chart_transport(
    boundary: CpcfBoundaryChart | PredictiveBoundaryMap,
    *,
    fit_actual_child_rows: Any,
    fit_provenance: str,
    fit_parent_states: Any | None = None,
    fit_encoded_parent_rows: Any | None = None,
    fit_weights: Any | None = None,
    heldout_actual_child_rows: Any | None = None,
    heldout_provenance: str | None = None,
    heldout_parent_states: Any | None = None,
    heldout_encoded_parent_rows: Any | None = None,
    heldout_weights: Any | None = None,
    alignment_tolerance: float = 1e-12,
    orthogonality_tolerance: float = 1e-10,
    preservation_tolerance: float = 1e-10,
) -> CpcfBoundaryChartTransport:
    """Align a predictive chart to child coordinates without changing behavior.

    Only fit rows enter the weighted orthogonal Procrustes solve.  Optional
    heldout rows are checked for provenance/storage disjointness and evaluated
    strictly after the rotation is fixed.
    """

    torch = _torch()
    if isinstance(boundary, CpcfBoundaryChart):
        source_chart = boundary
        source_mapping = boundary.mapping
        source_kind = "cpcf_boundary_chart"
    elif isinstance(boundary, PredictiveBoundaryMap):
        source_chart = None
        source_mapping = boundary
        source_kind = "predictive_boundary_map"
    else:
        raise TypeError(
            "boundary must be CpcfBoundaryChart or PredictiveBoundaryMap."
        )
    fit_split = _text(fit_provenance, "fit_provenance")
    alignment_tolerance = _nonnegative_finite_float(
        alignment_tolerance,
        "alignment_tolerance",
    )
    orthogonality_tolerance = _nonnegative_finite_float(
        orthogonality_tolerance,
        "orthogonality_tolerance",
    )
    preservation_tolerance = _nonnegative_finite_float(
        preservation_tolerance,
        "preservation_tolerance",
    )
    if (fit_parent_states is None) == (fit_encoded_parent_rows is None):
        raise CpcfBoundaryChartError(
            "Supply exactly one of fit_parent_states and "
            "fit_encoded_parent_rows."
        )
    if fit_parent_states is not None:
        fit_source_input = fit_parent_states
        fit_source = _matrix(
            fit_parent_states,
            "fit_parent_states",
            width=source_mapping.source_width,
        )
        fit_encoded = source_mapping.encode(fit_source)
        fit_input_kind = "parent_states"
    else:
        fit_source_input = fit_encoded_parent_rows
        fit_source = None
        fit_encoded = _matrix(
            fit_encoded_parent_rows,
            "fit_encoded_parent_rows",
            width=source_mapping.target_width,
        )
        fit_input_kind = "already_encoded_parent_rows"
    fit_actual = _matrix(
        fit_actual_child_rows,
        "fit_actual_child_rows",
        width=source_mapping.target_width,
    )
    if int(fit_encoded.shape[0]) != int(fit_actual.shape[0]):
        raise CpcfBoundaryChartError(
            "Fit encoded-parent and actual-child row counts differ."
        )
    fit_weight = _weights(
        fit_weights,
        int(fit_encoded.shape[0]),
        "fit_weights",
    )

    # The solver boundary is intentionally above every heldout read.
    rotation, singular_values, effective_rank = (
        _weighted_orthogonal_procrustes(
            fit_encoded,
            fit_actual,
            fit_weight,
        )
    )
    fit_pre_rmse = _weighted_rmse(fit_encoded, fit_actual, fit_weight)
    fit_post = fit_encoded @ rotation
    fit_post_rmse = _weighted_rmse(fit_post, fit_actual, fit_weight)
    fit_alignment_limit = fit_pre_rmse + alignment_tolerance * max(
        1.0,
        fit_pre_rmse,
    )
    if fit_post_rmse > fit_alignment_limit:
        raise CpcfBoundaryChartError(
            "Orthogonal chart transport made fit alignment worse: "
            f"{fit_post_rmse:.12g} > {fit_alignment_limit:.12g}."
        )

    heldout_requested = any(
        value is not None
        for value in (
            heldout_actual_child_rows,
            heldout_provenance,
            heldout_parent_states,
            heldout_encoded_parent_rows,
            heldout_weights,
        )
    )
    heldout_encoded = None
    heldout_actual = None
    heldout_weight = None
    heldout_source = None
    heldout_report = None
    heldout_input_kind = None
    heldout_split = None
    if heldout_requested:
        if heldout_actual_child_rows is None or heldout_provenance is None:
            raise CpcfBoundaryChartError(
                "Heldout evaluation requires actual child rows and provenance."
            )
        if (
            heldout_parent_states is None
        ) == (heldout_encoded_parent_rows is None):
            raise CpcfBoundaryChartError(
                "Heldout evaluation requires exactly one parent-state or "
                "already-encoded input."
            )
        heldout_split = _text(
            heldout_provenance,
            "heldout_provenance",
        )
        if heldout_split == fit_split:
            raise CpcfBoundaryChartError(
                "Fit and heldout provenance identifiers must differ."
            )
        heldout_source_input = (
            heldout_parent_states
            if heldout_parent_states is not None
            else heldout_encoded_parent_rows
        )
        _require_disjoint_storage(
            fit_source_input,
            heldout_source_input,
            "parent evidence",
        )
        _require_disjoint_storage(
            fit_actual_child_rows,
            heldout_actual_child_rows,
            "actual child rows",
        )
        if (
            fit_weights is not None
            and heldout_weights is not None
        ):
            _require_disjoint_storage(
                fit_weights,
                heldout_weights,
                "weights",
            )
        if heldout_parent_states is not None:
            heldout_source = _matrix(
                heldout_parent_states,
                "heldout_parent_states",
                width=source_mapping.source_width,
            )
            heldout_encoded = source_mapping.encode(heldout_source)
            heldout_input_kind = "parent_states"
        else:
            heldout_encoded = _matrix(
                heldout_encoded_parent_rows,
                "heldout_encoded_parent_rows",
                width=source_mapping.target_width,
            )
            heldout_input_kind = "already_encoded_parent_rows"
        heldout_actual = _matrix(
            heldout_actual_child_rows,
            "heldout_actual_child_rows",
            width=source_mapping.target_width,
        )
        if int(heldout_encoded.shape[0]) != int(heldout_actual.shape[0]):
            raise CpcfBoundaryChartError(
                "Heldout encoded-parent and actual-child row counts differ."
            )
        heldout_weight = _weights(
            heldout_weights,
            int(heldout_encoded.shape[0]),
            "heldout_weights",
        )
        heldout_pre_rmse = _weighted_rmse(
            heldout_encoded,
            heldout_actual,
            heldout_weight,
        )
        heldout_post_rmse = _weighted_rmse(
            heldout_encoded @ rotation,
            heldout_actual,
            heldout_weight,
        )
        heldout_report = {
            "provenance": heldout_split,
            "provenance_sha256": _canonical_sha256(heldout_split),
            "input_kind": heldout_input_kind,
            "rows": int(heldout_encoded.shape[0]),
            "pre_alignment_rmse": heldout_pre_rmse,
            "post_alignment_rmse": heldout_post_rmse,
            "required_write_rms_reduction": (
                heldout_pre_rmse - heldout_post_rmse
            ),
            "used_by_solver": False,
        }

    identity = torch.eye(source_mapping.target_width, dtype=torch.float64)
    orthogonality_error = max(
        _maximum_absolute_error(rotation.T @ rotation, identity),
        _maximum_absolute_error(rotation @ rotation.T, identity),
    )
    if orthogonality_error > orthogonality_tolerance:
        raise CpcfBoundaryChartError(
            "Chart transport is not orthogonal within tolerance."
        )

    transported_encoder = (source_mapping.encoder @ rotation).contiguous()
    transported_decoder = (rotation.T @ source_mapping.decoder).contiguous()
    transported_predictors = {
        name: (rotation.T @ predictor).contiguous()
        for name, predictor in sorted(
            source_mapping.feature_predictors.items()
        )
    }
    decode_errors = [
        _maximum_absolute_error(
            transported_encoder @ transported_decoder,
            source_mapping.encoder @ source_mapping.decoder,
        ),
        _maximum_absolute_error(
            fit_post @ transported_decoder,
            fit_encoded @ source_mapping.decoder,
        ),
    ]
    if heldout_encoded is not None:
        decode_errors.append(
            _maximum_absolute_error(
                (heldout_encoded @ rotation) @ transported_decoder,
                heldout_encoded @ source_mapping.decoder,
            )
        )
    decode_invariance_error = max(decode_errors)

    predictor_errors: dict[str, float] = {}
    for name, predictor in sorted(
        source_mapping.feature_predictors.items()
    ):
        transported_predictor = transported_predictors[name]
        errors = [
            _maximum_absolute_error(
                transported_encoder @ transported_predictor,
                source_mapping.encoder @ predictor,
            ),
            _maximum_absolute_error(
                fit_post @ transported_predictor,
                fit_encoded @ predictor,
            ),
        ]
        if heldout_encoded is not None:
            errors.append(
                _maximum_absolute_error(
                    (heldout_encoded @ rotation) @ transported_predictor,
                    heldout_encoded @ predictor,
                )
            )
        predictor_errors[name] = max(errors)
    predictor_invariance_error = max(
        predictor_errors.values(),
        default=0.0,
    )
    if (
        decode_invariance_error > preservation_tolerance
        or predictor_invariance_error > preservation_tolerance
    ):
        raise CpcfBoundaryChartError(
            "Gauge transport failed decode or feature-prediction "
            "preservation tolerance."
        )

    pre_write_rms = fit_pre_rmse
    post_write_rms = fit_post_rmse
    reduction = pre_write_rms - post_write_rms
    reduction_fraction = (
        reduction / pre_write_rms if pre_write_rms > 1e-30 else 0.0
    )
    tensor_hashes = {
        "source_encoder": _tensor_sha256(source_mapping.encoder),
        "transported_encoder": _tensor_sha256(transported_encoder),
        "source_decoder": _tensor_sha256(source_mapping.decoder),
        "transported_decoder": _tensor_sha256(transported_decoder),
        "source_feature_predictors": {
            name: _tensor_sha256(value)
            for name, value in sorted(
                source_mapping.feature_predictors.items()
            )
        },
        "transported_feature_predictors": {
            name: _tensor_sha256(value)
            for name, value in transported_predictors.items()
        },
        "rotation": _tensor_sha256(rotation),
        "singular_values": _tensor_sha256(singular_values),
        "fit_encoded_parent_rows": _tensor_sha256(fit_encoded),
        "fit_actual_child_rows": _tensor_sha256(fit_actual),
        "fit_weights": _tensor_sha256(fit_weight),
    }
    if fit_source is not None:
        tensor_hashes["fit_parent_states"] = _tensor_sha256(fit_source)
    if heldout_encoded is not None:
        tensor_hashes.update(
            {
                "heldout_encoded_parent_rows": _tensor_sha256(
                    heldout_encoded
                ),
                "heldout_actual_child_rows": _tensor_sha256(
                    heldout_actual
                ),
                "heldout_weights": _tensor_sha256(heldout_weight),
            }
        )
    if heldout_source is not None:
        tensor_hashes["heldout_parent_states"] = _tensor_sha256(
            heldout_source
        )

    source_mapping_document = source_mapping.as_dict()
    body = {
        "schema": _TRANSPORT_SCHEMA,
        "component": "weighted_orthogonal_procrustes_chart_transport",
        "source_kind": source_kind,
        "source_mapping_document_sha256": _canonical_sha256(
            source_mapping_document
        ),
        "source_chart_document_sha256": (
            _canonical_sha256(source_chart.as_dict())
            if source_chart is not None
            else None
        ),
        "row_convention": "encoded_parent_rows @ rotation ~= actual_child_rows",
        "gauge_transform": {
            "encoder": "source_encoder @ rotation",
            "decoder": "rotation.T @ source_decoder",
            "feature_predictors": (
                "rotation.T @ source_feature_predictor"
            ),
        },
        "source_width": source_mapping.source_width,
        "target_width": source_mapping.target_width,
        "fit_only_solver_provenance": {
            "provenance": fit_split,
            "provenance_sha256": _canonical_sha256(fit_split),
            "input_kind": fit_input_kind,
            "rows": int(fit_encoded.shape[0]),
            "solver_inputs": (
                "fit_encoded_parent_rows",
                "fit_actual_child_rows",
                "fit_weights",
            ),
            "heldout_used_by_solver": False,
        },
        "heldout_evaluation": heldout_report,
        "fit_heldout_provenance_disjoint": (
            heldout_split != fit_split
            if heldout_report is not None
            else None
        ),
        "fit_heldout_storage_disjoint": (
            True if heldout_report is not None else None
        ),
        "fit_pre_alignment_rmse": fit_pre_rmse,
        "fit_post_alignment_rmse": fit_post_rmse,
        "fit_alignment_no_worse": True,
        "fit_alignment_acceptance_limit": fit_alignment_limit,
        "required_write_rms_before": pre_write_rms,
        "required_write_rms_after": post_write_rms,
        "required_write_rms_reduction": reduction,
        "required_write_rms_reduction_fraction": reduction_fraction,
        "effective_cross_covariance_rank": effective_rank,
        "orthogonality_max_abs_error": orthogonality_error,
        "decode_invariance_max_abs_error": decode_invariance_error,
        "feature_predictor_invariance_max_abs_error": (
            predictor_invariance_error
        ),
        "feature_predictor_invariance_by_name": predictor_errors,
        "alignment_tolerance": alignment_tolerance,
        "orthogonality_tolerance": orthogonality_tolerance,
        "preservation_tolerance": preservation_tolerance,
        "tensor_sha256": tensor_hashes,
        "floating_point_dtype": "torch.float64",
        "closed_form_solver": True,
        "numeric_parent_weight_matching_objective": False,
        "parent_forward_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "sparsity_introduced": False,
        "parent_sidecar_required": False,
    }
    report = {**body, "report_sha256": _canonical_sha256(body)}

    mapping_body = {
        "schema": _TRANSPORT_SCHEMA,
        "component": "gauge_transported_predictive_boundary_map",
        "source_mapping_document_sha256": body[
            "source_mapping_document_sha256"
        ],
        "transport_report_sha256": report["report_sha256"],
        "source_width": source_mapping.source_width,
        "target_width": source_mapping.target_width,
        "encoder_sha256": _tensor_sha256(transported_encoder),
        "decoder_sha256": _tensor_sha256(transported_decoder),
        "feature_predictor_sha256": tensor_hashes[
            "transported_feature_predictors"
        ],
        "decode_and_feature_predictions_preserved": True,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    mapping_report = {
        **mapping_body,
        "report_sha256": _canonical_sha256(mapping_body),
    }
    transported_mapping = PredictiveBoundaryMap(
        encoder=transported_encoder,
        decoder=transported_decoder,
        feature_predictors=transported_predictors,
        report=mapping_report,
    )

    transported_chart = None
    if source_chart is not None:
        source_chart_body = source_chart.as_dict()
        source_chart_body.pop("report_sha256", None)
        chart_body = {
            **source_chart_body,
            "predictive_mapping": transported_mapping.as_dict(),
            "orthogonal_coordinate_transport": report,
            "source_chart_document_sha256": body[
                "source_chart_document_sha256"
            ],
        }
        chart_report = {
            **chart_body,
            "report_sha256": _canonical_sha256(chart_body),
        }
        transported_chart = CpcfBoundaryChart(
            name=source_chart.name,
            mapping=transported_mapping,
            report=chart_report,
        )

    return CpcfBoundaryChartTransport(
        mapping=transported_mapping,
        chart=transported_chart,
        rotation=rotation,
        report=report,
    )


def fit_cpcf_boundary_chart(
    name: str,
    fit_parent_states: Any,
    fit_future_coordinates: Any,
    *,
    target_width: int,
    validation_parent_states: Any,
    validation_future_coordinates: Any,
    fit_additional_features: Sequence[CausalFeatureBlock] = (),
    validation_additional_features: Sequence[CausalFeatureBlock] = (),
    fit_weights: Any | None = None,
    validation_weights: Any | None = None,
    state_reconstruction_weight: float = 0.05,
    ridge: float = 1e-6,
    minimum_predictive_energy_fraction: float = 0.90,
    minimum_validation_block_retention: float = 0.0,
    enforce_validation_block_floor: bool = True,
) -> CpcfBoundaryChart:
    """Fit one chart against future/output behavior, not numeric weights."""

    boundary_name = _text(name, "name")
    fit_states = _matrix(fit_parent_states, "fit_parent_states")
    validation_states = _matrix(
        validation_parent_states,
        "validation_parent_states",
        width=int(fit_states.shape[1]),
    )
    fit_future = _matrix(
        fit_future_coordinates,
        "fit_future_coordinates",
    )
    validation_future = _matrix(
        validation_future_coordinates,
        "validation_future_coordinates",
        width=int(fit_future.shape[1]),
    )
    if (
        int(fit_states.shape[0]) != int(fit_future.shape[0])
        or int(validation_states.shape[0])
        != int(validation_future.shape[0])
    ):
        raise CpcfBoundaryChartError(
            "Boundary state and future-coordinate rows are misaligned."
        )
    fit_features = (
        CausalFeatureBlock(
            name="future_output_coordinates",
            role="terminal_action_and_future_behavior",
            values=fit_future,
            weight=1.0,
        ),
        *tuple(fit_additional_features),
    )
    validation_features = (
        CausalFeatureBlock(
            name="future_output_coordinates",
            role="terminal_action_and_future_behavior",
            values=validation_future,
            weight=1.0,
        ),
        *tuple(validation_additional_features),
    )
    mapping = fit_predictive_boundary_map(
        fit_states,
        fit_features,
        target_width=target_width,
        validation_states=validation_states,
        validation_features=validation_features,
        fit_weights=fit_weights,
        validation_weights=validation_weights,
        state_reconstruction_weight=state_reconstruction_weight,
        ridge=ridge,
        minimum_predictive_energy_fraction=(
            minimum_predictive_energy_fraction
        ),
        minimum_validation_block_retention=(
            minimum_validation_block_retention
        ),
        enforce_validation_block_floor=enforce_validation_block_floor,
    )
    body = {
        "schema": _SCHEMA,
        "name": boundary_name,
        "source_width": mapping.source_width,
        "target_width": mapping.target_width,
        "future_coordinate_width": int(fit_future.shape[1]),
        "fit_rows": int(fit_states.shape[0]),
        "validation_rows": int(validation_states.shape[0]),
        "chart_transition_residual_formula": (
            "encoded_parent_endpoint_minus_actual_child_pre_state"
        ),
        "identity_skip_path_accounted_for": True,
        "independent_boundary_coordinates": True,
        "numeric_parent_weight_matching_objective": False,
        "predictive_mapping": mapping.as_dict(),
        "parent_forward_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    report = {**body, "report_sha256": _canonical_sha256(body)}
    return CpcfBoundaryChart(
        name=boundary_name,
        mapping=mapping,
        report=report,
    )


__all__ = [
    "CpcfBoundaryChart",
    "CpcfBoundaryChartError",
    "CpcfBoundaryChartTransport",
    "fit_cpcf_boundary_chart",
    "fit_cpcf_orthogonal_chart_transport",
]
