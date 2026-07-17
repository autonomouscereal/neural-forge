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
    "fit_cpcf_boundary_chart",
]
