"""Fit-only post-operator chart transport for sequential LFM2 CPCF folds.

The controller rotates one already-fitted post-operator predictive chart into
the coordinate gauge occupied by the actual folded-upstream child.  It is a
closed-form, orthogonal chart change: decoded source states and every fitted
feature prediction remain invariant, while the residual branch is asked to
write a smaller chart-transition error.

Only ``fit`` phases enter the Procrustes solve.  ``selection`` tensors are not
read until the transport is frozen, and validation tensors are never read by
this module.  The post-layer chart remains an independent, unchanged object.
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

from neural_forge.hf_cpcf_boundary_chart import (
    CpcfBoundaryChartTransport,
    fit_cpcf_orthogonal_chart_transport,
)
from neural_forge.hf_lfm2_cpcf_boundary_bundle import (
    LFM2_CPCF_LAYER_CHARTS_SCHEMA,
    Lfm2CpcfLayerBoundaryCharts,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import (
    Lfm2CpcfPhaseStateBank,
    Lfm2CpcfTargetPhase,
)


LFM2_CPCF_RESIDUAL_ERROR_CONTROLLER_SCHEMA = (
    "neural-forge-lfm2-cpcf-residual-error-controller-v1"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CONTROL_SPLITS = ("fit", "selection")


class Lfm2CpcfResidualErrorControllerError(ValueError):
    """The post-operator chart-transport contract is not authenticated."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for LFM2 CPCF residual-error control."
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
        raise Lfm2CpcfResidualErrorControllerError(
            "Residual-error controller metadata must be finite canonical JSON."
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2CpcfResidualErrorControllerError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.numel() == 0
    ):
        raise Lfm2CpcfResidualErrorControllerError(
            "Controller evidence must be a nonempty dense tensor."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
    }
    digest = hashlib.sha256(_canonical_json(descriptor))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _weighted_rmse(
    prediction: Any,
    target: Any,
    weights: Any,
) -> float:
    torch = _torch()
    prediction64 = prediction.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    target64 = target.detach().to(device="cpu", dtype=torch.float64)
    weights64 = weights.detach().to(device="cpu", dtype=torch.float64)
    if (
        prediction64.ndim != 2
        or tuple(prediction64.shape) != tuple(target64.shape)
        or weights64.ndim != 1
        or int(weights64.numel()) != int(prediction64.shape[0])
        or not bool(torch.isfinite(prediction64).all().item())
        or not bool(torch.isfinite(target64).all().item())
        or not bool(torch.isfinite(weights64).all().item())
        or bool((weights64 <= 0.0).any().item())
    ):
        raise Lfm2CpcfResidualErrorControllerError(
            "Weighted controller rows are invalid or misaligned."
        )
    energy = (weights64.unsqueeze(1) * (prediction64 - target64).square()).sum()
    denominator = weights64.sum() * int(prediction64.shape[1])
    return float(torch.sqrt(energy / denominator).item())


def _chart_document_sha256(chart: Any) -> str:
    as_dict = getattr(chart, "as_dict", None)
    if not callable(as_dict):
        raise Lfm2CpcfResidualErrorControllerError(
            "Boundary chart lacks a canonical document."
        )
    return _sha256(as_dict())


def _chart_payload_sha256(chart: Any) -> str:
    mapping = getattr(chart, "mapping", None)
    predictors = getattr(mapping, "feature_predictors", None)
    if mapping is None or not isinstance(predictors, Mapping):
        raise Lfm2CpcfResidualErrorControllerError(
            "Boundary chart lacks an authenticated predictive payload."
        )
    return _sha256(
        {
            "document_sha256": _chart_document_sha256(chart),
            "encoder_sha256": _tensor_sha256(mapping.encoder),
            "decoder_sha256": _tensor_sha256(mapping.decoder),
            "feature_predictor_sha256": {
                name: _tensor_sha256(value)
                for name, value in sorted(predictors.items())
            },
        }
    )


def _row_lineage(
    phase: Lfm2CpcfTargetPhase,
    *,
    split_ordinal: int,
    layer_evidence_sha256: str,
) -> list[dict[str, Any]]:
    first_position = phase.full_prefix_tokens - phase.sampled_suffix_tokens
    return [
        {
            "split": phase.split,
            "split_phase_ordinal": split_ordinal,
            "batch_index": phase.batch_index,
            "record_id": phase.record_id,
            "phase": phase.phase,
            "phase_sha256": phase.phase_sha256,
            "parent_evidence_sha256": (phase.parent_evidence.evidence_sha256),
            "layer_evidence_sha256": layer_evidence_sha256,
            "batch_row_index": batch_row,
            "sampled_suffix_index": suffix_index,
            "full_prefix_position": first_position + suffix_index,
        }
        for batch_row in range(phase.batch_size)
        for suffix_index in range(phase.sampled_suffix_tokens)
    ]


def _phase_evidence_identity_sha256(
    bank: Lfm2CpcfPhaseStateBank,
    *,
    layer_index: int,
) -> str:
    """Fingerprint sealed identities without reading any evidence tensor."""

    inventory = []
    for ordinal, phase in enumerate(bank.phases):
        if len(phase.parent_evidence.layers) <= layer_index:
            raise Lfm2CpcfResidualErrorControllerError(
                f"Phase {phase.identity!r} lacks layer {layer_index}."
            )
        layer = phase.parent_evidence.layers[layer_index]
        inventory.append(
            {
                "ordinal": ordinal,
                "identity": [phase.batch_index, phase.phase],
                "record_id": phase.record_id,
                "split": phase.split,
                "phase_sha256": phase.phase_sha256,
                "parent_evidence_sha256": (phase.parent_evidence.evidence_sha256),
                "layer_evidence_sha256": layer.evidence_sha256,
                "layer_index": layer.layer_index,
                "operator_kind": layer.operator_kind,
                "source_width": layer.source_width,
            }
        )
    return _sha256(inventory)


def _split_rows(
    bank: Lfm2CpcfPhaseStateBank,
    charts: Lfm2CpcfLayerBoundaryCharts,
    *,
    split: str,
) -> tuple[Any, Any, Any, Mapping[str, Any]]:
    """Materialize one split; callers control the strict read boundary."""

    torch = _torch()
    phases = tuple(phase for phase in bank.phases if phase.split == split)
    if not phases:
        raise Lfm2CpcfResidualErrorControllerError(
            f"Residual-error control requires at least one {split!r} phase."
        )
    parent_parts = []
    actual_parts = []
    weight_parts = []
    lineage = []
    phase_inventory = []
    tensor_inventory = []
    phase_mass = 1.0 / len(phases)
    field = "residual.post_operator"
    for split_ordinal, phase in enumerate(phases):
        if (
            phase.identity not in bank.states
            or len(phase.parent_evidence.layers) <= charts.layer_index
        ):
            raise Lfm2CpcfResidualErrorControllerError(
                f"{split} phase {phase.identity!r} is not aligned to layer "
                f"{charts.layer_index}."
            )
        layer = phase.parent_evidence.layers[charts.layer_index]
        if (
            layer.layer_index != charts.layer_index
            or layer.operator_kind != charts.operator_kind
            or layer.source_width != charts.source_width
        ):
            raise Lfm2CpcfResidualErrorControllerError(
                f"{split} phase {phase.identity!r} changed layer topology."
            )
        try:
            parent = layer.tensors[field]
        except KeyError as exc:
            raise Lfm2CpcfResidualErrorControllerError(
                f"{split} phase {phase.identity!r} lacks {field!r}."
            ) from exc
        if (
            not isinstance(parent, torch.Tensor)
            or parent.ndim != 3
            or tuple(parent.shape)
            != (
                phase.batch_size,
                phase.sampled_suffix_tokens,
                charts.source_width,
            )
            or not bool(torch.isfinite(parent).all().item())
        ):
            raise Lfm2CpcfResidualErrorControllerError(
                f"{split} parent post-operator suffix is invalid."
            )
        actual = bank.suffix(phase)
        if (
            tuple(actual.shape)
            != (
                phase.batch_size,
                phase.sampled_suffix_tokens,
                charts.target_width,
            )
            or actual.requires_grad
            or actual.grad_fn is not None
            or not bool(torch.isfinite(actual).all().item())
        ):
            raise Lfm2CpcfResidualErrorControllerError(
                f"{split} actual child pre-operator suffix is invalid."
            )
        rows = phase.batch_size * phase.sampled_suffix_tokens
        row_weight = phase_mass / rows
        parent_rows = (
            parent.detach()
            .to(device="cpu")
            .reshape(
                rows,
                charts.source_width,
            )
            .contiguous()
        )
        actual_rows = (
            actual.detach()
            .to(device="cpu")
            .reshape(
                rows,
                charts.target_width,
            )
            .contiguous()
        )
        weights = torch.full(
            (rows,),
            row_weight,
            dtype=torch.float64,
        )
        parent_parts.append(parent_rows)
        actual_parts.append(actual_rows)
        weight_parts.append(weights)
        phase_lineage = _row_lineage(
            phase,
            split_ordinal=split_ordinal,
            layer_evidence_sha256=layer.evidence_sha256,
        )
        lineage.extend(phase_lineage)
        phase_inventory.append(
            {
                "split_phase_ordinal": split_ordinal,
                "identity": [phase.batch_index, phase.phase],
                "record_id": phase.record_id,
                "phase_sha256": phase.phase_sha256,
                "parent_evidence_sha256": (phase.parent_evidence.evidence_sha256),
                "layer_evidence_sha256": layer.evidence_sha256,
                "rows": rows,
                "row_weight": row_weight,
                "total_phase_mass": float(weights.sum().item()),
            }
        )
        tensor_inventory.append(
            {
                "phase_sha256": phase.phase_sha256,
                "parent_post_operator_sha256": _tensor_sha256(parent),
                "actual_child_pre_operator_sha256": _tensor_sha256(actual),
            }
        )
    parent_rows = torch.cat(parent_parts, dim=0).contiguous()
    actual_rows = torch.cat(actual_parts, dim=0).contiguous()
    weights = torch.cat(weight_parts, dim=0).contiguous()
    row_ids = tuple(_sha256(row) for row in lineage)
    if (
        len(set(row_ids)) != len(row_ids)
        or int(parent_rows.shape[0]) != len(lineage)
        or tuple(parent_rows.shape[1:]) != (charts.source_width,)
        or tuple(actual_rows.shape) != (len(lineage), charts.target_width)
        or int(weights.numel()) != len(lineage)
        or not math.isclose(
            float(weights.sum().item()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or any(
            not math.isclose(
                float(record["total_phase_mass"]),
                phase_mass,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for record in phase_inventory
        )
    ):
        raise Lfm2CpcfResidualErrorControllerError(
            f"{split} row lineage or equal-phase-mass weighting failed."
        )
    report = {
        "split": split,
        "phase_count": len(phases),
        "row_count": len(lineage),
        "ordered_phase_sha256": [phase.phase_sha256 for phase in phases],
        "phase_inventory_sha256": _sha256(phase_inventory),
        "row_lineage_sha256": _sha256(lineage),
        "ordered_row_identity_sha256": _sha256(row_ids),
        "tensor_inventory_sha256": _sha256(tensor_inventory),
        "weights_sha256": _tensor_sha256(weights),
        "equal_total_mass_per_phase": True,
        "unit_total_split_mass": True,
    }
    return parent_rows, actual_rows, weights, MappingProxyType(report)


def _seal_layer_charts(
    source: Lfm2CpcfLayerBoundaryCharts,
    *,
    post_operator: Any,
) -> Lfm2CpcfLayerBoundaryCharts:
    post_operator_sha = _digest(
        post_operator.report.get("report_sha256"),
        "post_operator.report_sha256",
    )
    post_layer_sha = _digest(
        source.post_layer.report.get("report_sha256"),
        "post_layer.report_sha256",
    )
    body = {
        "schema": LFM2_CPCF_LAYER_CHARTS_SCHEMA,
        "layer_index": source.layer_index,
        "operator_kind": source.operator_kind,
        "source_width": post_operator.source_width,
        "target_width": post_operator.target_width,
        "post_operator_report_sha256": post_operator_sha,
        "post_layer_report_sha256": post_layer_sha,
        "independent_boundary_coordinates": True,
    }
    return Lfm2CpcfLayerBoundaryCharts(
        layer_index=source.layer_index,
        operator_kind=source.operator_kind,
        post_operator=post_operator,
        post_layer=source.post_layer,
        layer_charts_sha256=_sha256(body),
    )


@dataclass(frozen=True, slots=True)
class Lfm2CpcfResidualErrorControllerResult:
    """One frozen transport, resealed layer charts, and audit report."""

    boundary_charts: Lfm2CpcfLayerBoundaryCharts
    transport: CpcfBoundaryChartTransport
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(
            self.boundary_charts,
            Lfm2CpcfLayerBoundaryCharts,
        ):
            raise TypeError("boundary_charts must be Lfm2CpcfLayerBoundaryCharts.")
        if not isinstance(self.transport, CpcfBoundaryChartTransport):
            raise TypeError("transport must be CpcfBoundaryChartTransport.")
        report = dict(self.report)
        seal = _digest(
            report.pop("report_sha256", None),
            "report.report_sha256",
        )
        if seal != _sha256(report):
            raise Lfm2CpcfResidualErrorControllerError(
                "Residual-error controller report seal is invalid."
            )
        if (
            report.get("transport_report_sha256")
            != self.transport.report["report_sha256"]
            or report.get("transported_layer_charts_sha256")
            != self.boundary_charts.layer_charts_sha256
            or self.transport.chart is not self.boundary_charts.post_operator
        ):
            raise Lfm2CpcfResidualErrorControllerError(
                "Controller result objects differ from their report bindings."
            )
        object.__setattr__(
            self,
            "report",
            MappingProxyType({**report, "report_sha256": seal}),
        )

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def fit_lfm2_cpcf_post_operator_chart_transport(
    input_bank: Lfm2CpcfPhaseStateBank,
    boundary_charts: Lfm2CpcfLayerBoundaryCharts,
) -> Lfm2CpcfResidualErrorControllerResult:
    """Freeze a fit-only chart transport, then evaluate selection once."""

    if not isinstance(input_bank, Lfm2CpcfPhaseStateBank):
        raise TypeError("input_bank must be Lfm2CpcfPhaseStateBank.")
    if not isinstance(boundary_charts, Lfm2CpcfLayerBoundaryCharts):
        raise TypeError("boundary_charts must be Lfm2CpcfLayerBoundaryCharts.")
    if (
        input_bank.layer_index != boundary_charts.layer_index
        or input_bank.hidden_size != boundary_charts.target_width
    ):
        raise Lfm2CpcfResidualErrorControllerError(
            "Input bank and layer boundary charts are not aligned."
        )
    observed_splits = {phase.split for phase in input_bank.phases}
    if not set(_CONTROL_SPLITS).issubset(observed_splits):
        raise Lfm2CpcfResidualErrorControllerError(
            "Input bank must contain disjoint fit and selection phases."
        )

    source_bank_sha = input_bank.state_bank_sha256
    source_layer_charts_sha = boundary_charts.layer_charts_sha256
    source_post_operator_document_sha = _chart_document_sha256(
        boundary_charts.post_operator
    )
    source_post_layer_document_sha = _chart_document_sha256(boundary_charts.post_layer)
    source_post_operator_payload_sha = _chart_payload_sha256(
        boundary_charts.post_operator
    )
    source_post_layer_payload_sha = _chart_payload_sha256(boundary_charts.post_layer)
    source_phase_evidence_identity_sha = _phase_evidence_identity_sha256(
        input_bank,
        layer_index=boundary_charts.layer_index,
    )

    # Strict solver boundary: no selection or validation tensor is touched
    # before this fit-only transport object has been constructed and sealed.
    fit_parent, fit_actual, fit_weights, fit_lineage = _split_rows(
        input_bank,
        boundary_charts,
        split="fit",
    )
    fit_provenance = (
        "lfm2-cpcf-post-operator-chart-transport/fit/"
        + fit_lineage["row_lineage_sha256"]
    )
    transport = fit_cpcf_orthogonal_chart_transport(
        boundary_charts.post_operator,
        fit_actual_child_rows=fit_actual,
        fit_provenance=fit_provenance,
        fit_parent_states=fit_parent,
        fit_weights=fit_weights,
    )
    if transport.chart is None:
        raise Lfm2CpcfResidualErrorControllerError(
            "Chart transport did not return a transported chart."
        )
    frozen_transport_sha = transport.report["report_sha256"]

    # Selection is materialized and scored only after the fit transport seal
    # exists.  No solver is called again and selection cannot alter rotation.
    (
        selection_parent,
        selection_actual,
        selection_weights,
        selection_lineage,
    ) = _split_rows(
        input_bank,
        boundary_charts,
        split="selection",
    )
    if (
        fit_lineage["ordered_row_identity_sha256"]
        == selection_lineage["ordered_row_identity_sha256"]
        or fit_lineage["row_lineage_sha256"] == selection_lineage["row_lineage_sha256"]
    ):
        raise Lfm2CpcfResidualErrorControllerError(
            "Fit and selection row lineages are not disjoint."
        )
    selection_before = boundary_charts.post_operator.encode(selection_parent)
    selection_after = transport.chart.encode(selection_parent)
    selection_before_rmse = _weighted_rmse(
        selection_before,
        selection_actual,
        selection_weights,
    )
    selection_after_rmse = _weighted_rmse(
        selection_after,
        selection_actual,
        selection_weights,
    )

    transported_charts = _seal_layer_charts(
        boundary_charts,
        post_operator=transport.chart,
    )

    # Re-read only the two admitted controller splits to prove the source
    # tensors remain byte-identical.  Validation evidence stays unread.
    (
        _fit_parent_after,
        _fit_actual_after,
        _fit_weights_after,
        fit_lineage_after,
    ) = _split_rows(input_bank, boundary_charts, split="fit")
    (
        _selection_parent_after,
        _selection_actual_after,
        _selection_weights_after,
        selection_lineage_after,
    ) = _split_rows(input_bank, boundary_charts, split="selection")
    if (
        input_bank.state_bank_sha256 != source_bank_sha
        or boundary_charts.layer_charts_sha256 != source_layer_charts_sha
        or _chart_document_sha256(boundary_charts.post_operator)
        != source_post_operator_document_sha
        or _chart_document_sha256(boundary_charts.post_layer)
        != source_post_layer_document_sha
        or _chart_payload_sha256(boundary_charts.post_operator)
        != source_post_operator_payload_sha
        or _chart_payload_sha256(boundary_charts.post_layer)
        != source_post_layer_payload_sha
        or _phase_evidence_identity_sha256(
            input_bank,
            layer_index=boundary_charts.layer_index,
        )
        != source_phase_evidence_identity_sha
        or dict(fit_lineage_after) != dict(fit_lineage)
        or dict(selection_lineage_after) != dict(selection_lineage)
        or transport.report["report_sha256"] != frozen_transport_sha
        or transported_charts.post_layer is not boundary_charts.post_layer
    ):
        raise Lfm2CpcfResidualErrorControllerError(
            "Controller source bank, charts, evidence, or frozen transport "
            "changed during evaluation."
        )
    if (
        transport.report["decode_invariance_max_abs_error"]
        > transport.report["preservation_tolerance"]
        or transport.report["feature_predictor_invariance_max_abs_error"]
        > transport.report["preservation_tolerance"]
    ):
        raise Lfm2CpcfResidualErrorControllerError(
            "Transport did not preserve decoded state and feature predictions."
        )

    body = {
        "schema": LFM2_CPCF_RESIDUAL_ERROR_CONTROLLER_SCHEMA,
        "component": "fit_only_post_operator_chart_transport_shadow",
        "layer_index": boundary_charts.layer_index,
        "operator_kind": boundary_charts.operator_kind,
        "input_boundary": input_bank.boundary,
        "source_width": boundary_charts.source_width,
        "target_width": boundary_charts.target_width,
        "source_state_bank_sha256": source_bank_sha,
        "source_layer_charts_sha256": source_layer_charts_sha,
        "source_post_operator_document_sha256": (source_post_operator_document_sha),
        "source_post_layer_document_sha256": source_post_layer_document_sha,
        "source_post_operator_payload_sha256": (source_post_operator_payload_sha),
        "source_post_layer_payload_sha256": source_post_layer_payload_sha,
        "source_phase_evidence_identity_sha256": (source_phase_evidence_identity_sha),
        "transport_report_sha256": frozen_transport_sha,
        "transported_layer_charts_sha256": (transported_charts.layer_charts_sha256),
        "transported_post_operator_report_sha256": (
            transported_charts.post_operator.report["report_sha256"]
        ),
        "unchanged_post_layer_report_sha256": (
            transported_charts.post_layer.report["report_sha256"]
        ),
        "fit_lineage": dict(fit_lineage),
        "selection_lineage": dict(selection_lineage),
        "fit_solver_inputs": [
            "parent.residual.post_operator.suffix_rows",
            "actual_child.pre_operator.suffix_rows",
            "equal_phase_mass_weights",
        ],
        "selection_read_after_transport_freeze": True,
        "selection_used_by_solver": False,
        "fit_required_write_rms_before": (
            transport.report["required_write_rms_before"]
        ),
        "fit_required_write_rms_after": (
            transport.report["required_write_rms_after"]
        ),
        "fit_required_write_rms_reduction": (
            transport.report["required_write_rms_reduction"]
        ),
        "fit_required_write_rms_reduction_fraction": (
            transport.report["required_write_rms_reduction_fraction"]
        ),
        "selection_required_write_rms_before": selection_before_rmse,
        "selection_required_write_rms_after": selection_after_rmse,
        "selection_required_write_rms_reduction": (
            selection_before_rmse - selection_after_rmse
        ),
        "selection_required_write_rms_reduction_fraction": (
            (selection_before_rmse - selection_after_rmse)
            / selection_before_rmse
            if selection_before_rmse > 1e-30
            else 0.0
        ),
        "effective_cross_covariance_rank": (
            transport.report["effective_cross_covariance_rank"]
        ),
        "alignment_tolerance": transport.report["alignment_tolerance"],
        "orthogonality_tolerance": (
            transport.report["orthogonality_tolerance"]
        ),
        "preservation_tolerance": (
            transport.report["preservation_tolerance"]
        ),
        "orthogonality_max_abs_error": (
            transport.report["orthogonality_max_abs_error"]
        ),
        "decode_invariance_max_abs_error": (
            transport.report["decode_invariance_max_abs_error"]
        ),
        "feature_predictor_invariance_max_abs_error": (
            transport.report[
                "feature_predictor_invariance_max_abs_error"
            ]
        ),
        "validation_rows_read": 0,
        "validation_used_by_solver": False,
        "fit_only_closed_form_orthogonal_procrustes": True,
        "source_bank_unchanged": True,
        "source_charts_unchanged": True,
        "source_evidence_unchanged": True,
        "post_layer_chart_same_object": True,
        "decoded_state_predictions_invariant": True,
        "feature_predictions_invariant": True,
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
        "automatic_retry_or_retuning_performed": False,
        "forced_shadow_for_controlled_comparison": True,
        "automatic_accept_reject_controller": False,
    }
    report = MappingProxyType({**body, "report_sha256": _sha256(body)})
    return Lfm2CpcfResidualErrorControllerResult(
        boundary_charts=transported_charts,
        transport=transport,
        report=report,
    )


def apply_lfm2_cpcf_post_operator_chart_transport_shadow(
    input_bank: Lfm2CpcfPhaseStateBank,
    boundary_charts: Lfm2CpcfLayerBoundaryCharts,
    *,
    enabled: bool = False,
) -> Lfm2CpcfResidualErrorControllerResult | None:
    """Preserve exact caller semantics while the shadow mode is disabled."""

    if type(enabled) is not bool:
        raise TypeError("enabled must be bool.")
    if not enabled:
        return None
    return fit_lfm2_cpcf_post_operator_chart_transport(
        input_bank,
        boundary_charts,
    )


__all__ = [
    "LFM2_CPCF_RESIDUAL_ERROR_CONTROLLER_SCHEMA",
    "Lfm2CpcfResidualErrorControllerError",
    "Lfm2CpcfResidualErrorControllerResult",
    "apply_lfm2_cpcf_post_operator_chart_transport_shadow",
    "fit_lfm2_cpcf_post_operator_chart_transport",
]
