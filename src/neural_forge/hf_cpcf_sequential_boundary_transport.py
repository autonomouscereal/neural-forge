"""Sequential gauge transport for adjacent CPCF residual boundaries.

An independently fitted chart is only defined up to an orthogonal target
coordinate change.  That freedom must be resolved at the *actual* child state
which precedes the boundary.  Otherwise the next residual branch is asked to
spend finite child capacity crossing an arbitrary chart orientation.

The first use of this rule aligns a post-operator chart to the child state
entering an attention or recurrent operator.  This module supplies the
causally subsequent use: after the operator has been folded and replayed,
align the post-layer chart to the actual child post-operator state before
folding the MLP.

The transform is one weighted orthogonal Procrustes solve over fit evidence.
It is a pure gauge change:

``E' = E R``, ``D' = R.T D``, and ``F' = R.T F``.

Decoded source states and all fitted future/action predictions are invariant.
Optional selection evidence is not materialized until the transport is
already frozen, and is diagnostic only.  Validation evidence is not accepted.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.hf_cpcf_boundary_chart import (
    CpcfBoundaryChart,
    CpcfBoundaryChartTransport,
    fit_cpcf_orthogonal_chart_transport,
)


CPCF_SEQUENTIAL_BOUNDARY_TRANSPORT_SCHEMA = (
    "neural-forge-cpcf-sequential-boundary-transport-v1"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class CpcfSequentialBoundaryTransportError(ValueError):
    """Adjacent boundary charts or replay evidence are invalid."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for CPCF sequential boundary transport."
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
        raise CpcfSequentialBoundaryTransportError(
            "Sequential boundary metadata must be finite canonical JSON."
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CpcfSequentialBoundaryTransportError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CpcfSequentialBoundaryTransportError(
            f"{name} must be nonempty text."
        )
    return value.strip()


def _matrix(value: Any, name: str, *, width: int) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 2
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or value.is_complex()
        or value.requires_grad
        or value.grad_fn is not None
        or int(value.shape[1]) != width
    ):
        raise CpcfSequentialBoundaryTransportError(
            f"{name} must be a detached nonempty dense real matrix with "
            f"width {width}."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise CpcfSequentialBoundaryTransportError(
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
        or value.is_complex()
        or value.requires_grad
        or value.grad_fn is not None
    ):
        raise CpcfSequentialBoundaryTransportError(
            f"{name} must contain one detached positive floating value per row."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if (
        not bool(torch.isfinite(result).all().item())
        or bool((result <= 0.0).any().item())
    ):
        raise CpcfSequentialBoundaryTransportError(
            f"{name} must contain finite positive values."
        )
    return result


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
    }
    digest = hashlib.sha256(_canonical_json(descriptor))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


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


def _weighted_rmse(prediction: Any, target: Any, weights: Any) -> float:
    torch = _torch()
    difference = prediction - target
    energy = (weights.unsqueeze(1) * difference.square()).sum()
    denominator = weights.sum() * int(prediction.shape[1])
    return float(torch.sqrt(energy / denominator).item())


def _chart_document_sha256(chart: CpcfBoundaryChart) -> str:
    return _sha256(chart.as_dict())


def _chart_payload_sha256(chart: CpcfBoundaryChart) -> str:
    return _sha256(
        {
            "document_sha256": _chart_document_sha256(chart),
            "encoder_sha256": _tensor_sha256(chart.mapping.encoder),
            "decoder_sha256": _tensor_sha256(chart.mapping.decoder),
            "feature_predictor_sha256": {
                name: _tensor_sha256(value)
                for name, value in sorted(
                    chart.mapping.feature_predictors.items()
                )
            },
        }
    )


def _selection_requested(values: tuple[Any, ...]) -> bool:
    supplied = tuple(value is not None for value in values)
    if any(supplied) and not all(supplied):
        raise CpcfSequentialBoundaryTransportError(
            "Selection evaluation requires parent states, actual child rows, "
            "weights, and provenance together."
        )
    return all(supplied)


@dataclass(frozen=True, slots=True)
class CpcfSequentialBoundaryTransport:
    """One unchanged upstream chart and one transported downstream chart."""

    upstream_chart: CpcfBoundaryChart
    source_downstream_chart: CpcfBoundaryChart
    downstream_transport: CpcfBoundaryChartTransport
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.upstream_chart, CpcfBoundaryChart):
            raise TypeError("upstream_chart must be CpcfBoundaryChart.")
        if not isinstance(self.source_downstream_chart, CpcfBoundaryChart):
            raise TypeError(
                "source_downstream_chart must be CpcfBoundaryChart."
            )
        if not isinstance(
            self.downstream_transport,
            CpcfBoundaryChartTransport,
        ):
            raise TypeError(
                "downstream_transport must be CpcfBoundaryChartTransport."
            )
        transported = self.downstream_transport.chart
        if transported is None:
            raise CpcfSequentialBoundaryTransportError(
                "The downstream transport must contain a boundary chart."
            )
        report = dict(self.report)
        seal = _digest(
            report.pop("report_sha256", None),
            "report.report_sha256",
        )
        if seal != _sha256(report):
            raise CpcfSequentialBoundaryTransportError(
                "Sequential boundary report seal is invalid."
            )
        if (
            report.get("upstream_chart_payload_sha256")
            != _chart_payload_sha256(self.upstream_chart)
            or report.get("source_downstream_chart_payload_sha256")
            != _chart_payload_sha256(self.source_downstream_chart)
            or report.get("transported_downstream_chart_payload_sha256")
            != _chart_payload_sha256(transported)
            or report.get("downstream_transport_report_sha256")
            != self.downstream_transport.report["report_sha256"]
        ):
            raise CpcfSequentialBoundaryTransportError(
                "Sequential boundary objects differ from their report bindings."
            )
        object.__setattr__(
            self,
            "report",
            MappingProxyType({**report, "report_sha256": seal}),
        )

    @property
    def downstream_chart(self) -> CpcfBoundaryChart:
        chart = self.downstream_transport.chart
        if chart is None:  # pragma: no cover - guarded in __post_init__
            raise RuntimeError("Sequential transport lost its chart.")
        return chart

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def fit_cpcf_sequential_downstream_chart_transport(
    upstream_chart: CpcfBoundaryChart,
    downstream_chart: CpcfBoundaryChart,
    *,
    fit_parent_downstream_states: Any,
    fit_actual_child_upstream_rows: Any,
    fit_provenance: str,
    fit_weights: Any | None = None,
    selection_parent_downstream_states: Any | None = None,
    selection_actual_child_upstream_rows: Any | None = None,
    selection_weights: Any | None = None,
    selection_provenance: str | None = None,
    alignment_tolerance: float = 1e-12,
    orthogonality_tolerance: float = 1e-10,
    preservation_tolerance: float = 1e-10,
) -> CpcfSequentialBoundaryTransport:
    """Align the next chart to actual replay at the preceding boundary.

    The function performs exactly one fit-only SVD.  Selection evidence, when
    supplied, is read only after the returned low-level transport is sealed.
    It can report generalization but cannot alter the rotation.
    """

    if not isinstance(upstream_chart, CpcfBoundaryChart):
        raise TypeError("upstream_chart must be CpcfBoundaryChart.")
    if not isinstance(downstream_chart, CpcfBoundaryChart):
        raise TypeError("downstream_chart must be CpcfBoundaryChart.")
    if upstream_chart is downstream_chart or upstream_chart.name == downstream_chart.name:
        raise CpcfSequentialBoundaryTransportError(
            "Upstream and downstream boundaries must be distinct charts."
        )
    if (
        upstream_chart.source_width != downstream_chart.source_width
        or upstream_chart.target_width != downstream_chart.target_width
    ):
        raise CpcfSequentialBoundaryTransportError(
            "Adjacent chart source and target widths must match."
        )
    fit_split = _text(fit_provenance, "fit_provenance")
    selection_is_requested = _selection_requested(
        (
            selection_parent_downstream_states,
            selection_actual_child_upstream_rows,
            selection_weights,
            selection_provenance,
        )
    )
    if selection_is_requested:
        selection_split = _text(
            selection_provenance,
            "selection_provenance",
        )
        if selection_split == fit_split:
            raise CpcfSequentialBoundaryTransportError(
                "Fit and selection provenance identifiers must differ."
            )
    else:
        selection_split = None

    source_upstream_document_sha = _chart_document_sha256(upstream_chart)
    source_downstream_document_sha = _chart_document_sha256(downstream_chart)
    source_upstream_payload_sha = _chart_payload_sha256(upstream_chart)
    source_downstream_payload_sha = _chart_payload_sha256(downstream_chart)

    fit_parent = _matrix(
        fit_parent_downstream_states,
        "fit_parent_downstream_states",
        width=downstream_chart.source_width,
    )
    fit_actual = _matrix(
        fit_actual_child_upstream_rows,
        "fit_actual_child_upstream_rows",
        width=downstream_chart.target_width,
    )
    if int(fit_parent.shape[0]) != int(fit_actual.shape[0]):
        raise CpcfSequentialBoundaryTransportError(
            "Fit parent and actual-child row counts differ."
        )
    fit_weight = _weights(
        fit_weights,
        int(fit_parent.shape[0]),
        "fit_weights",
    )

    # Solver freeze boundary.  No selection tensor has been inspected above.
    downstream_transport = fit_cpcf_orthogonal_chart_transport(
        downstream_chart,
        fit_actual_child_rows=fit_actual,
        fit_provenance=fit_split,
        fit_parent_states=fit_parent,
        fit_weights=fit_weight,
        alignment_tolerance=alignment_tolerance,
        orthogonality_tolerance=orthogonality_tolerance,
        preservation_tolerance=preservation_tolerance,
    )
    transported_downstream = downstream_transport.chart
    if transported_downstream is None:
        raise CpcfSequentialBoundaryTransportError(
            "Downstream chart transport returned no chart."
        )
    frozen_transport_sha = downstream_transport.report["report_sha256"]

    selection_report = None
    if selection_is_requested:
        if (
            _shares_storage(
                fit_parent_downstream_states,
                selection_parent_downstream_states,
            )
            or _shares_storage(
                fit_actual_child_upstream_rows,
                selection_actual_child_upstream_rows,
            )
            or (
                fit_weights is not None
                and _shares_storage(fit_weights, selection_weights)
            )
        ):
            raise CpcfSequentialBoundaryTransportError(
                "Fit and selection evidence must not share tensor storage."
            )
        selection_parent = _matrix(
            selection_parent_downstream_states,
            "selection_parent_downstream_states",
            width=downstream_chart.source_width,
        )
        selection_actual = _matrix(
            selection_actual_child_upstream_rows,
            "selection_actual_child_upstream_rows",
            width=downstream_chart.target_width,
        )
        if int(selection_parent.shape[0]) != int(selection_actual.shape[0]):
            raise CpcfSequentialBoundaryTransportError(
                "Selection parent and actual-child row counts differ."
            )
        selection_weight = _weights(
            selection_weights,
            int(selection_parent.shape[0]),
            "selection_weights",
        )
        selection_before = downstream_chart.encode(selection_parent)
        selection_after = transported_downstream.encode(selection_parent)
        selection_before_rmse = _weighted_rmse(
            selection_before,
            selection_actual,
            selection_weight,
        )
        selection_after_rmse = _weighted_rmse(
            selection_after,
            selection_actual,
            selection_weight,
        )
        selection_report = {
            "provenance": selection_split,
            "provenance_sha256": _sha256(selection_split),
            "rows": int(selection_parent.shape[0]),
            "required_write_rms_before": selection_before_rmse,
            "required_write_rms_after": selection_after_rmse,
            "required_write_rms_reduction": (
                selection_before_rmse - selection_after_rmse
            ),
            "used_by_solver": False,
            "read_after_transport_freeze": True,
            "parent_states_sha256": _tensor_sha256(selection_parent),
            "actual_child_upstream_rows_sha256": _tensor_sha256(
                selection_actual
            ),
            "weights_sha256": _tensor_sha256(selection_weight),
        }

    if (
        _chart_document_sha256(upstream_chart)
        != source_upstream_document_sha
        or _chart_document_sha256(downstream_chart)
        != source_downstream_document_sha
        or _chart_payload_sha256(upstream_chart)
        != source_upstream_payload_sha
        or _chart_payload_sha256(downstream_chart)
        != source_downstream_payload_sha
        or downstream_transport.report["report_sha256"]
        != frozen_transport_sha
    ):
        raise CpcfSequentialBoundaryTransportError(
            "A source chart or frozen transport changed during evaluation."
        )
    if (
        downstream_transport.report["decode_invariance_max_abs_error"]
        > downstream_transport.report["preservation_tolerance"]
        or downstream_transport.report[
            "feature_predictor_invariance_max_abs_error"
        ]
        > downstream_transport.report["preservation_tolerance"]
    ):
        raise CpcfSequentialBoundaryTransportError(
            "Downstream gauge transport did not preserve chart behavior."
        )

    body = {
        "schema": CPCF_SEQUENTIAL_BOUNDARY_TRANSPORT_SCHEMA,
        "component": "fit_only_sequential_downstream_chart_transport",
        "upstream_boundary_name": upstream_chart.name,
        "source_downstream_boundary_name": downstream_chart.name,
        "source_width": downstream_chart.source_width,
        "target_width": downstream_chart.target_width,
        "fit_provenance": fit_split,
        "fit_provenance_sha256": _sha256(fit_split),
        "fit_rows": int(fit_parent.shape[0]),
        "fit_parent_downstream_states_sha256": _tensor_sha256(fit_parent),
        "fit_actual_child_upstream_rows_sha256": _tensor_sha256(fit_actual),
        "fit_weights_sha256": _tensor_sha256(fit_weight),
        "upstream_chart_document_sha256": source_upstream_document_sha,
        "upstream_chart_payload_sha256": source_upstream_payload_sha,
        "source_downstream_chart_document_sha256": (
            source_downstream_document_sha
        ),
        "source_downstream_chart_payload_sha256": (
            source_downstream_payload_sha
        ),
        "transported_downstream_chart_document_sha256": (
            _chart_document_sha256(transported_downstream)
        ),
        "transported_downstream_chart_payload_sha256": (
            _chart_payload_sha256(transported_downstream)
        ),
        "downstream_transport_report_sha256": frozen_transport_sha,
        "fit_required_write_rms_before": downstream_transport.report[
            "required_write_rms_before"
        ],
        "fit_required_write_rms_after": downstream_transport.report[
            "required_write_rms_after"
        ],
        "fit_required_write_rms_reduction": downstream_transport.report[
            "required_write_rms_reduction"
        ],
        "fit_required_write_rms_reduction_fraction": (
            downstream_transport.report[
                "required_write_rms_reduction_fraction"
            ]
        ),
        "selection_evaluation": selection_report,
        "selection_used_by_solver": False,
        "selection_read_after_transport_freeze": (
            True if selection_report is not None else None
        ),
        "validation_argument_supported": False,
        "validation_rows_read": 0,
        "upstream_chart_same_object": True,
        "upstream_chart_unchanged": True,
        "source_downstream_chart_unchanged": True,
        "downstream_decoded_state_predictions_invariant": True,
        "downstream_feature_predictions_invariant": True,
        "orthogonality_max_abs_error": downstream_transport.report[
            "orthogonality_max_abs_error"
        ],
        "decode_invariance_max_abs_error": downstream_transport.report[
            "decode_invariance_max_abs_error"
        ],
        "feature_predictor_invariance_max_abs_error": downstream_transport.report[
            "feature_predictor_invariance_max_abs_error"
        ],
        "causal_install_order": [
            "fold_and_replay_upstream_operator",
            "freeze_fit_only_downstream_chart_transport",
            "fold_downstream_residual_branch",
        ],
        "residual_write_formula": (
            "transported_downstream_endpoint_minus_"
            "actual_child_upstream_replay"
        ),
        "one_closed_form_svd": True,
        "automatic_retry_or_retuning_performed": False,
        "numeric_parent_weight_matching_objective": False,
        "parent_model_forward_calls": 0,
        "target_model_forward_calls": 0,
        "target_layer_forward_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "sparsity_introduced": False,
        "parent_sidecar_required": False,
    }
    report = MappingProxyType({**body, "report_sha256": _sha256(body)})
    return CpcfSequentialBoundaryTransport(
        upstream_chart=upstream_chart,
        source_downstream_chart=downstream_chart,
        downstream_transport=downstream_transport,
        report=report,
    )


__all__ = [
    "CPCF_SEQUENTIAL_BOUNDARY_TRANSPORT_SCHEMA",
    "CpcfSequentialBoundaryTransport",
    "CpcfSequentialBoundaryTransportError",
    "fit_cpcf_sequential_downstream_chart_transport",
]
