#!/usr/bin/env python3
"""Refine a real blank-first structural LFM2 child from its sealed parent cache.

The input child is the already-folded, physically smaller dense h640 target
created by ``run_lfm2_dynamic_structural_compression.py``.  This stage does not
instantiate another random target, alter architecture, execute the parent, or
train.  It deterministically rebuilds the baseline compiler's global linear
residual coordinate map, projects sealed parent trace rows into those same
target coordinates, and analytically recompiles target-shaped:

* Q/K/V/O plus SwiGLU for attention layers, atomically; and
* B/C/x short-conv plus SwiGLU for convolution layers, with a separate
  activation-derived sparse channel assignment and rollback.

The existing baseline child tensors are explicit source-derived priors.  All
compiled values are copied into the target's existing ``Parameter`` objects.
The result is written as a new ordinary dense local checkpoint and accepted
only after construction of a fresh local model from those new bytes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments import run_lfm2_dynamic_structural_compression as baseline  # noqa: E402
from neural_forge.hf_closed_form_fit import (  # noqa: E402
    fit_diagonal_norm,
    measure_fit,
)
from neural_forge.hf_feature_fold import (  # noqa: E402
    FeatureFoldMap,
    build_feature_fold_map,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    fingerprint_lfm2_parent,
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    Lfm2CachedAttentionFoldCase,
    Lfm2CachedMlpFoldCase,
    Lfm2CachedShortConvFoldCase,
    Lfm2SparseResidualAssignment,
    cached_attention_cases_from_lfm2_trace_phases,
    cached_mlp_case_from_lfm2_trace,
    cached_short_conv_case_from_lfm2_trace,
    compose_lfm2_structural_layer_fold,
    compile_lfm2_structural_residual_attention_fold,
    compile_lfm2_structural_residual_mlp_fold,
    compile_lfm2_structural_residual_short_conv_fold,
    install_lfm2_structural_layer_fold,
    install_lfm2_structural_mlp_fold,
    install_lfm2_structural_short_conv_fold,
)


REFINEMENT_SCHEMA = (
    "neural-forge-lfm2-dynamic-structural-residual-attention-refinement-v1"
)
CASE_SCHEMA = "neural-forge-lfm2-sealed-target-coordinate-cases-v1"
REQUIRED_BASELINE_CANDIDATE = "h640_q10_kv5_ff1920"


class Lfm2StructuralRefinementError(RuntimeError):
    """A cache, checkpoint, map, transaction, or reload invariant failed."""


class ResidualCoordinateMap(Protocol):
    """Linear coordinate-map contract shared by sparse and dense mappings."""

    @property
    def source_width(self) -> int: ...

    @property
    def target_width(self) -> int: ...

    @property
    def map_sha256(self) -> str: ...

    def encode(self, source_rows: torch.Tensor) -> torch.Tensor: ...

    def decode(self, target_rows: torch.Tensor) -> torch.Tensor: ...

    def fold_consumer_weight(
        self, source_weight: torch.Tensor
    ) -> torch.Tensor: ...

    def fold_producer_weight(
        self, source_weight: torch.Tensor
    ) -> torch.Tensor: ...


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Lfm2StructuralRefinementError(
            "Refinement metadata is not strict canonical JSON."
        ) from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Lfm2StructuralRefinementError(
            f"Could not read refinement input JSON {path}."
        ) from exc
    if not isinstance(value, Mapping):
        raise Lfm2StructuralRefinementError(f"{path} must contain one object.")
    return dict(value)


def _tensor_descriptor(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": baseline._tensor_sha256(value),
    }


def _state_subset_sha256(
    model: nn.Module,
    *,
    excluded_layers: Sequence[int] = (),
) -> str:
    excluded = tuple(
        f".layers.{int(layer_index)}." for layer_index in excluded_layers
    )
    descriptor = {
        name: _tensor_descriptor(value)
        for name, value in sorted(model.state_dict().items())
        if not any(marker in f".{name}." for marker in excluded)
    }
    return _json_sha256(descriptor)


def _module_parameter_topology(
    module: nn.Module,
) -> dict[str, tuple[int, tuple[int, ...], str, bool]]:
    return {
        name: (
            id(parameter),
            tuple(parameter.shape),
            str(parameter.dtype),
            bool(parameter.requires_grad),
        )
        for name, parameter in module.named_parameters(remove_duplicate=False)
    }


def _module_parameter_state_sha256(module: nn.Module) -> str:
    return _json_sha256(
        {
            name: _tensor_descriptor(parameter)
            for name, parameter in sorted(
                module.named_parameters(remove_duplicate=False)
            )
        }
    )


def _trace_tensor_sha256(
    batches: Sequence[Mapping[str, Any]],
    layer_indices: Sequence[int],
) -> str:
    markers = tuple(
        f".layer.{int(layer_index):03d}." for layer_index in layer_indices
    )
    descriptor: list[dict[str, Any]] = []
    for batch_index, tensors in enumerate(batches):
        for name, value in sorted(tensors.items()):
            if (
                markers
                and not any(marker in name for marker in markers)
            ):
                continue
            if isinstance(value, torch.Tensor):
                descriptor.append(
                    {
                        "batch": batch_index,
                        "name": name,
                        **_tensor_descriptor(value),
                    }
                )
    if not descriptor:
        raise Lfm2StructuralRefinementError(
            "Selected sealed trace batches contain no tensors for the layers."
        )
    return _json_sha256(descriptor)


def _global_coordinate_map_contract(
    mapping: ResidualCoordinateMap,
) -> tuple[int, int, str]:
    """Validate the behavior-bearing global map without assuming its class."""

    source_width = getattr(mapping, "source_width", None)
    target_width = getattr(mapping, "target_width", None)
    map_sha256 = getattr(mapping, "map_sha256", None)
    if (
        isinstance(source_width, bool)
        or not isinstance(source_width, int)
        or isinstance(target_width, bool)
        or not isinstance(target_width, int)
        or source_width <= 0
        or target_width <= 0
        or target_width > source_width
        or not isinstance(map_sha256, str)
        or len(map_sha256) != 64
        or any(value not in "0123456789abcdef" for value in map_sha256)
        or any(
            not callable(getattr(mapping, name, None))
            for name in (
                "encode",
                "decode",
                "fold_consumer_weight",
                "fold_producer_weight",
            )
        )
    ):
        raise Lfm2StructuralRefinementError(
            "Global residual mapping does not implement the linear "
            "coordinate-map protocol."
        )
    return source_width, target_width, map_sha256


def project_with_global_residual_map(
    mapping: ResidualCoordinateMap,
    value: torch.Tensor,
) -> torch.Tensor:
    """Encode any row-major cached residual tensor into target coordinates."""

    source_width, target_width, _map_sha256 = (
        _global_coordinate_map_contract(mapping)
    )
    if not isinstance(value, torch.Tensor) or value.ndim < 1:
        raise Lfm2StructuralRefinementError(
            "A projected cache value must be a non-scalar tensor."
        )
    if int(value.shape[-1]) != source_width:
        raise Lfm2StructuralRefinementError(
            "Cached residual width does not match the rebuilt global map."
        )
    projected = mapping.encode(
        value.detach().to(device="cpu", dtype=torch.float32)
    ).contiguous()
    if tuple(projected.shape[:-1]) != tuple(value.shape[:-1]):
        raise Lfm2StructuralRefinementError(
            "Global residual projection changed non-feature axes."
        )
    if int(projected.shape[-1]) != target_width:
        raise Lfm2StructuralRefinementError(
            "Global residual projection returned the wrong target width."
        )
    return projected


def _core_assignment_from_sparse_channel_map(
    mapping: FeatureFoldMap,
) -> Lfm2SparseResidualAssignment:
    """Convert one explicitly sparse channel map for the short-conv core."""

    if not isinstance(mapping, FeatureFoldMap):
        raise TypeError("mapping must be a FeatureFoldMap.")
    assignment = Lfm2SparseResidualAssignment(
        target_width=mapping.target_width,
        source_to_target=tuple(
            int(value) for value in mapping.assignment.tolist()
        ),
        coefficients=tuple(
            float(value) for value in mapping.coefficients.tolist()
        ),
    )
    if (
        not assignment.full_source_coverage
        or assignment.source_width != mapping.source_width
        or assignment.target_width != mapping.target_width
        or sorted(
            source
            for target in range(mapping.target_width)
            for source in (
                mapping.assignment == target
            ).nonzero(as_tuple=False).reshape(-1).tolist()
        )
        != list(range(mapping.source_width))
    ):
        raise Lfm2StructuralRefinementError(
            "Short-conv channel assignment lost a source or target channel."
        )
    return assignment


def build_short_conv_channel_assignment(
    cases: Sequence[Lfm2CachedShortConvFoldCase],
    *,
    layer_index: int,
    target_width: int,
    global_residual_map_sha256: str,
) -> tuple[Lfm2SparseResidualAssignment, dict[str, Any]]:
    """Build one deterministic full-source B/C/x assignment for a conv layer.

    This assignment is intentionally independent of the global residual
    coordinate basis.  A dense residual basis has no one-destination channel
    interpretation, while the depthwise FIR compiler needs exactly that local
    B/C/x grouping.
    """

    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or layer_index < 0
        or isinstance(target_width, bool)
        or not isinstance(target_width, int)
        or target_width <= 0
        or not isinstance(global_residual_map_sha256, str)
        or len(global_residual_map_sha256) != 64
        or any(
            value not in "0123456789abcdef"
            for value in global_residual_map_sha256
        )
    ):
        raise Lfm2StructuralRefinementError(
            "Short-conv assignment metadata is malformed."
        )
    if not cases or not all(
        isinstance(value, Lfm2CachedShortConvFoldCase) for value in cases
    ):
        raise Lfm2StructuralRefinementError(
            "Short-conv assignment requires sealed B/C/x cases."
        )
    role_rows: dict[str, list[torch.Tensor]] = {
        "b": [],
        "c": [],
        "x": [],
    }
    source_width: int | None = None
    for case_index, case in enumerate(cases):
        for role, attribute in (
            ("b", "parent_b"),
            ("c", "parent_c"),
            ("x", "parent_x"),
        ):
            value = getattr(case, attribute)
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 3
                or not torch.is_floating_point(value)
                or torch.is_complex(value)
                or not bool(torch.isfinite(value.detach()).all())
            ):
                raise Lfm2StructuralRefinementError(
                    f"Short-conv case {case_index} {role} tensor is malformed."
                )
            current_width = int(value.shape[-1])
            source_width = (
                current_width if source_width is None else source_width
            )
            if current_width != source_width:
                raise Lfm2StructuralRefinementError(
                    "Short-conv B/C/x source widths disagree."
                )
            role_rows[role].append(
                value.detach()
                .reshape(-1, current_width)
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
            )
    if source_width is None or target_width >= source_width:
        raise Lfm2StructuralRefinementError(
            "Short-conv assignment must reduce a positive source width."
        )
    calibration = {
        role: torch.cat(values, dim=0).contiguous()
        for role, values in role_rows.items()
    }
    sparse_map = build_feature_fold_map(
        tuple(calibration[role] for role in ("b", "c", "x")),
        target_width,
        source_width=source_width,
        chunk_size=min(256, source_width),
        sketch_size=32,
        balance_strength=0.25,
        allow_sign_flip=True,
    )
    assignment = _core_assignment_from_sparse_channel_map(sparse_map)
    diagnostics = sparse_map.report
    if (
        diagnostics is None
        or not diagnostics.full_source_coverage
        or not diagnostics.full_target_rank
        or not diagnostics.deterministic
        or assignment.source_width != source_width
        or assignment.target_width != target_width
    ):
        raise Lfm2StructuralRefinementError(
            "Activation-derived short-conv assignment lacks full coverage."
        )
    report = {
        "schema": "neural-forge-lfm2-short-conv-channel-assignment-v1",
        "layer_index": layer_index,
        "algorithm": (
            "deterministic-coupled-activation-aware-sparse-bcx-channel-fold"
        ),
        "calibration_roles": ["short_conv.b", "short_conv.c", "short_conv.x"],
        "calibration_case_count": len(cases),
        "calibration_rows_by_role": {
            role: int(value.shape[0])
            for role, value in calibration.items()
        },
        "calibration_sha256_by_role": {
            role: baseline._tensor_sha256(value)
            for role, value in calibration.items()
        },
        "source_width": source_width,
        "target_width": target_width,
        "global_residual_map_sha256": global_residual_map_sha256,
        "short_conv_channel_assignment_sha256": (
            assignment.assignment_sha256
        ),
        "independent_of_global_residual_coordinate_map": True,
        "dense_global_basis_interpreted_as_sparse_assignment": False,
        "full_source_coverage": True,
        "source_channels_dropped": 0,
        "feature_fold_diagnostics": diagnostics.as_dict(),
    }
    return assignment, report


def parse_layer_selection(value: str, layer_count: int) -> tuple[int, ...]:
    if isinstance(layer_count, bool) or layer_count <= 0:
        raise Lfm2StructuralRefinementError("layer_count must be positive.")
    if not isinstance(value, str) or not value.strip():
        raise Lfm2StructuralRefinementError("Layer selection cannot be empty.")
    if value.strip().lower() == "all":
        return tuple(range(layer_count))
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise Lfm2StructuralRefinementError(
                "Layer selection contains an empty component."
            )
        if "-" in part:
            left, right = part.split("-", 1)
            try:
                start, stop = int(left), int(right)
            except ValueError as exc:
                raise Lfm2StructuralRefinementError(
                    f"Invalid layer range {part!r}."
                ) from exc
            if start > stop:
                raise Lfm2StructuralRefinementError(
                    f"Descending layer range {part!r} is not allowed."
                )
            selected.update(range(start, stop + 1))
        else:
            try:
                selected.add(int(part))
            except ValueError as exc:
                raise Lfm2StructuralRefinementError(
                    f"Invalid layer index {part!r}."
                ) from exc
    if not selected or min(selected) < 0 or max(selected) >= layer_count:
        raise Lfm2StructuralRefinementError(
            f"Layer selection must stay within [0, {layer_count - 1}]."
        )
    return tuple(sorted(selected))


def _phases_for_layer(
    tensors: Mapping[str, Any],
    layer_index: int,
    *,
    maximum: int,
) -> tuple[str, ...]:
    marker = f".layer.{layer_index:03d}."
    phases = sorted(
        {
            name.split(".layer.", 1)[0]
            for name in tensors
            if marker in name and not name.startswith("cached_prefill.")
        },
        key=baseline._phase_key,
    )
    if not phases:
        raise Lfm2StructuralRefinementError(
            f"Sealed trace contains no phases for layer {layer_index}."
        )
    if maximum < 0:
        raise Lfm2StructuralRefinementError(
            "max_phases_per_batch cannot be negative."
        )
    if maximum:
        phases = phases[:maximum]
    return tuple(phases)


def _projected_target_trace(
    tensors: Mapping[str, Any],
    phases: Sequence[str],
    layer_index: int,
    mapping: ResidualCoordinateMap,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for phase in phases:
        base = f"{phase}.layer.{layer_index:03d}."
        for suffix in (
            "operator.input",
            "mlp.input",
            "residual.pre",
            "residual.post_operator",
            "residual.post_layer",
        ):
            name = base + suffix
            if name not in tensors:
                raise Lfm2StructuralRefinementError(
                    f"Verified trace is missing {name!r}."
                )
            result[name] = project_with_global_residual_map(
                mapping, tensors[name]
            )
    return result


def _strict_rms_norm_epsilon(module: nn.Module, role: str) -> float:
    for attribute in ("variance_epsilon", "eps", "epsilon"):
        value = getattr(module, attribute, None)
        if value is None:
            continue
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise Lfm2StructuralRefinementError(
                f"{role} RMSNorm epsilon is not numeric."
            ) from exc
        if math.isfinite(result) and result > 0.0:
            return result
        raise Lfm2StructuralRefinementError(
            f"{role} RMSNorm epsilon must be finite and positive."
        )
    raise Lfm2StructuralRefinementError(
        f"{role} RMSNorm exposes no explicit epsilon; refusing to guess."
    )


def _materialized_rms_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Execute the concrete LFM2 RMSNorm equation without a module call."""

    if (
        not isinstance(value, torch.Tensor)
        or value.ndim < 2
        or not isinstance(weight, torch.Tensor)
        or weight.ndim != 1
        or int(value.shape[-1]) != int(weight.shape[0])
    ):
        raise Lfm2StructuralRefinementError(
            "Analytical RMSNorm input/weight shapes are incompatible."
        )
    runtime_weight = weight.detach().to(device="cpu").contiguous()
    runtime_input = value.detach().to(
        device="cpu", dtype=runtime_weight.dtype
    )
    normalized = runtime_input.float()
    normalized = normalized * torch.rsqrt(
        normalized.square().mean(dim=-1, keepdim=True) + float(epsilon)
    )
    result = runtime_weight * normalized.to(dtype=runtime_input.dtype)
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2StructuralRefinementError(
            "Analytical materialized RMSNorm produced NaN or infinity."
        )
    return result.contiguous()


def _fit_metrics_dict(value: Any) -> dict[str, Any]:
    return {
        "rows": int(value.rows),
        "output_width": int(value.output_width),
        "rmse": float(value.rmse),
        "target_rms": float(value.target_rms),
        "relative_rmse": float(value.relative_rmse),
        "mean_cosine_similarity": float(value.mean_cosine_similarity),
        "finite": bool(value.finite),
    }


@dataclass(frozen=True, slots=True)
class FittedTargetRmsNorm:
    role: str
    weight: torch.Tensor
    epsilon: float
    rows: int
    analytical_fit: Mapping[str, Any]
    materialized_fit: Mapping[str, Any]
    input_sha256: str
    desired_output_sha256: str
    prior_weight_sha256: str
    fitted_weight_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "solver": "closed-form-per-channel-rmsnorm-gamma-ls",
            "epsilon": self.epsilon,
            "rows": self.rows,
            "analytical_fit": dict(self.analytical_fit),
            "materialized_fit": dict(self.materialized_fit),
            "input_sha256": self.input_sha256,
            "desired_output_sha256": self.desired_output_sha256,
            "prior_weight_sha256": self.prior_weight_sha256,
            "fitted_weight": _tensor_descriptor(self.weight),
            "fitted_weight_sha256": self.fitted_weight_sha256,
            "target_runtime_dtype_materialized_before_case_assembly": True,
            "model_forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
        }


@dataclass(frozen=True, slots=True)
class FittedLayerTargetNorms:
    operator: FittedTargetRmsNorm
    ffn: FittedTargetRmsNorm

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator_norm": self.operator.as_dict(),
            "ffn_norm": self.ffn.as_dict(),
            "parent_normalized_inputs_used_only_as_target_outputs": True,
            "deployed_design_inputs_recomputed_from_projected_residuals": True,
            "model_forward_calls": 0,
            "training_steps": 0,
        }


def _fit_target_rms_norm(
    *,
    role: str,
    input_values: Sequence[torch.Tensor],
    desired_values: Sequence[torch.Tensor],
    target_norm: nn.Module,
    ridge: float,
) -> FittedTargetRmsNorm:
    if not input_values or len(input_values) != len(desired_values):
        raise Lfm2StructuralRefinementError(
            f"{role} RMSNorm fit needs aligned cached input/target tensors."
        )
    parameter = getattr(target_norm, "weight", None)
    if not isinstance(parameter, nn.Parameter) or parameter.ndim != 1:
        raise Lfm2StructuralRefinementError(
            f"{role} target RMSNorm has no vector Parameter weight."
        )
    epsilon = _strict_rms_norm_epsilon(target_norm, role)
    inputs = torch.cat(
        [
            value.detach().to(device="cpu", dtype=torch.float64).reshape(
                -1, int(value.shape[-1])
            )
            for value in input_values
        ],
        dim=0,
    ).contiguous()
    desired = torch.cat(
        [
            value.detach().to(device="cpu", dtype=torch.float64).reshape(
                -1, int(value.shape[-1])
            )
            for value in desired_values
        ],
        dim=0,
    ).contiguous()
    if (
        tuple(inputs.shape) != tuple(desired.shape)
        or int(inputs.shape[1]) != int(parameter.shape[0])
    ):
        raise Lfm2StructuralRefinementError(
            f"{role} projected RMSNorm pairs do not match target width."
        )
    fit = fit_diagonal_norm(
        inputs,
        desired,
        kind="rms_norm",
        epsilon=epsilon,
        has_bias=False,
        ridge=ridge,
    )
    fitted_weight = fit.weight.to(
        device="cpu", dtype=parameter.dtype
    ).contiguous()
    materialized_predictions = [
        _materialized_rms_norm(
            value, fitted_weight, epsilon=epsilon
        ).reshape(-1, int(fitted_weight.shape[0]))
        for value in input_values
    ]
    materialized_metrics = measure_fit(
        torch.cat(materialized_predictions, dim=0),
        desired,
    )
    if not materialized_metrics.finite:
        raise Lfm2StructuralRefinementError(
            f"{role} materialized RMSNorm fit is non-finite."
        )
    return FittedTargetRmsNorm(
        role=role,
        weight=fitted_weight,
        epsilon=epsilon,
        rows=int(inputs.shape[0]),
        analytical_fit=MappingProxyType(_fit_metrics_dict(fit.train_metrics)),
        materialized_fit=MappingProxyType(
            _fit_metrics_dict(materialized_metrics)
        ),
        input_sha256=baseline._tensor_sha256(inputs),
        desired_output_sha256=baseline._tensor_sha256(desired),
        prior_weight_sha256=baseline._tensor_sha256(
            parameter.detach().to(device="cpu")
        ),
        fitted_weight_sha256=baseline._tensor_sha256(fitted_weight),
    )


def _fit_layer_target_norms(
    projected_traces: Sequence[Mapping[str, torch.Tensor]],
    phase_sequences: Sequence[Sequence[str]],
    *,
    layer_index: int,
    target_layer: nn.Module,
    ridge: float,
) -> FittedLayerTargetNorms:
    operator_inputs: list[torch.Tensor] = []
    operator_targets: list[torch.Tensor] = []
    ffn_inputs: list[torch.Tensor] = []
    ffn_targets: list[torch.Tensor] = []
    for projected, phases in zip(projected_traces, phase_sequences):
        for phase in phases:
            base = f"{phase}.layer.{layer_index:03d}."
            operator_inputs.append(projected[base + "residual.pre"])
            operator_targets.append(projected[base + "operator.input"])
            ffn_inputs.append(projected[base + "residual.post_operator"])
            ffn_targets.append(projected[base + "mlp.input"])
    return FittedLayerTargetNorms(
        operator=_fit_target_rms_norm(
            role=f"layer.{layer_index:03d}.operator_norm",
            input_values=operator_inputs,
            desired_values=operator_targets,
            target_norm=target_layer.operator_norm,
            ridge=ridge,
        ),
        ffn=_fit_target_rms_norm(
            role=f"layer.{layer_index:03d}.ffn_norm",
            input_values=ffn_inputs,
            desired_values=ffn_targets,
            target_norm=target_layer.ffn_norm,
            ridge=ridge,
        ),
    )


def _case_descriptor(value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in fields(value):
        item = getattr(value, field.name)
        if isinstance(item, torch.Tensor):
            result[field.name] = _tensor_descriptor(item)
        elif item is None or isinstance(item, (bool, int, float, str)):
            result[field.name] = item
        else:
            raise Lfm2StructuralRefinementError(
                f"Unsupported sealed case field {field.name!r}."
            )
    return result


@dataclass(frozen=True, slots=True)
class SealedLayerCases:
    layer_index: int
    operator_kind: str
    phase_sequences: tuple[tuple[str, ...], ...]
    attention: tuple[Lfm2CachedAttentionFoldCase, ...]
    mlp: tuple[Lfm2CachedMlpFoldCase, ...]
    short_conv: tuple[Lfm2CachedShortConvFoldCase, ...]
    cases_sha256: str
    target_norms: FittedLayerTargetNorms | None = None

    @property
    def full_multi_phase(self) -> bool:
        return any(len(value) > 1 for value in self.phase_sequences)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": CASE_SCHEMA,
            "layer_index": self.layer_index,
            "operator_kind": self.operator_kind,
            "phase_sequences": [list(value) for value in self.phase_sequences],
            "cache_batch_count": len(self.phase_sequences),
            "attention_case_count": len(self.attention),
            "mlp_case_count": len(self.mlp),
            "short_conv_case_count": len(self.short_conv),
            "multi_phase_cache_used": self.full_multi_phase,
            "cases_sha256": self.cases_sha256,
            "target_rms_norms": (
                None
                if self.target_norms is None
                else self.target_norms.as_dict()
            ),
            "parent_forward_calls": 0,
            "target_forward_calls": 0,
            "trace_regeneration_calls": 0,
        }


def assemble_sealed_layer_cases(
    trace_batches: Sequence[Mapping[str, Any]],
    *,
    layer_index: int,
    is_attention_layer: bool,
    mapping: ResidualCoordinateMap,
    target_layer: nn.Module | None = None,
    norm_ridge: float = 1e-6,
    max_phases_per_batch: int = 0,
) -> SealedLayerCases:
    """Use only public core adapters over verified, already-sealed tensors."""

    if not trace_batches:
        raise Lfm2StructuralRefinementError(
            "At least one sealed trace batch is required."
        )
    attention: list[Lfm2CachedAttentionFoldCase] = []
    mlp: list[Lfm2CachedMlpFoldCase] = []
    short_conv: list[Lfm2CachedShortConvFoldCase] = []
    phase_sequences: list[tuple[str, ...]] = []
    projected_traces: list[dict[str, torch.Tensor]] = []
    for tensors in trace_batches:
        if not isinstance(tensors, Mapping):
            raise Lfm2StructuralRefinementError(
                "Each trace batch must expose a tensor mapping."
            )
        phases = _phases_for_layer(
            tensors,
            layer_index,
            maximum=max_phases_per_batch,
        )
        phase_sequences.append(phases)
        projected_traces.append(
            _projected_target_trace(
                tensors, phases, layer_index, mapping
            )
        )
    target_norms = (
        None
        if target_layer is None
        else _fit_layer_target_norms(
            projected_traces,
            phase_sequences,
            layer_index=layer_index,
            target_layer=target_layer,
            ridge=norm_ridge,
        )
    )
    for tensors, phases, projected in zip(
        trace_batches, phase_sequences, projected_traces
    ):
        if target_norms is not None:
            for phase in phases:
                base = f"{phase}.layer.{layer_index:03d}."
                projected[base + "operator.input"] = _materialized_rms_norm(
                    projected[base + "residual.pre"],
                    target_norms.operator.weight,
                    epsilon=target_norms.operator.epsilon,
                )
                projected[base + "mlp.input"] = _materialized_rms_norm(
                    projected[base + "residual.post_operator"],
                    target_norms.ffn.weight,
                    epsilon=target_norms.ffn.epsilon,
                )
        if is_attention_layer:
            endpoints = {
                phase: projected[
                    f"{phase}.layer.{layer_index:03d}.residual.post_operator"
                ]
                for phase in phases
            }
            attention.extend(
                cached_attention_cases_from_lfm2_trace_phases(
                    tensors,
                    phases=phases,
                    layer_index=layer_index,
                    target_tensors=projected,
                    parent_post_attention_residual_target_space_by_phase=endpoints,
                )
            )
        else:
            for phase in phases:
                base = f"{phase}.layer.{layer_index:03d}."
                short_conv.append(
                    cached_short_conv_case_from_lfm2_trace(
                        tensors,
                        phase=phase,
                        layer_index=layer_index,
                        target_input=projected[base + "operator.input"],
                        parent_post_operator_residual_target_space=projected[
                            base + "residual.post_operator"
                        ],
                        target_pre_operator_residual=projected[
                            base + "residual.pre"
                        ],
                    )
                )
        for phase in phases:
            base = f"{phase}.layer.{layer_index:03d}."
            mlp.append(
                cached_mlp_case_from_lfm2_trace(
                    tensors,
                    phase=phase,
                    layer_index=layer_index,
                    target_input=projected[base + "mlp.input"],
                    parent_post_mlp_residual_target_space=projected[
                        base + "residual.post_layer"
                    ],
                    target_pre_mlp_residual=projected[
                        base + "residual.post_operator"
                    ],
                )
            )
    descriptor = {
        "schema": CASE_SCHEMA,
        "layer_index": layer_index,
        "operator_kind": (
            "full_attention" if is_attention_layer else "short_conv"
        ),
        "phase_sequences": [list(value) for value in phase_sequences],
        "attention": [_case_descriptor(value) for value in attention],
        "mlp": [_case_descriptor(value) for value in mlp],
        "short_conv": [_case_descriptor(value) for value in short_conv],
        "target_rms_norms": (
            None if target_norms is None else target_norms.as_dict()
        ),
    }
    return SealedLayerCases(
        layer_index=layer_index,
        operator_kind=descriptor["operator_kind"],
        phase_sequences=tuple(phase_sequences),
        attention=tuple(attention),
        mlp=tuple(mlp),
        short_conv=tuple(short_conv),
        cases_sha256=_json_sha256(descriptor),
        target_norms=target_norms,
    )


def _clone(value: torch.Tensor | None) -> torch.Tensor | None:
    return (
        None
        if value is None
        else value.detach().to(device="cpu").contiguous().clone()
    )


def _norm_epsilon(attention: nn.Module) -> float:
    for module_name in ("q_layernorm", "k_layernorm"):
        module = getattr(attention, module_name, None)
        for attribute in ("variance_epsilon", "eps", "epsilon"):
            value = getattr(module, attribute, None)
            if value is not None:
                result = float(value)
                if math.isfinite(result) and result > 0.0:
                    return result
    return 1e-6


def _restore_parameters(
    module: nn.Module,
    backup: Mapping[str, torch.Tensor],
) -> None:
    current = dict(module.named_parameters(remove_duplicate=False))
    with torch.no_grad():
        for name, value in backup.items():
            if name not in current:
                raise Lfm2StructuralRefinementError(
                    f"Cannot roll back missing Parameter {name!r}."
                )
            current[name].copy_(
                value.to(
                    device=current[name].device,
                    dtype=current[name].dtype,
                )
            )


def _install_fitted_layer_norms(
    target_layer: nn.Module,
    fitted: FittedLayerTargetNorms,
) -> dict[str, Any]:
    parameters = {
        "operator_norm.weight": getattr(
            getattr(target_layer, "operator_norm", None), "weight", None
        ),
        "ffn_norm.weight": getattr(
            getattr(target_layer, "ffn_norm", None), "weight", None
        ),
    }
    expected = {
        "operator_norm.weight": fitted.operator.weight,
        "ffn_norm.weight": fitted.ffn.weight,
    }
    if any(not isinstance(value, nn.Parameter) for value in parameters.values()):
        raise Lfm2StructuralRefinementError(
            "Target layer does not expose operator/FFN RMSNorm Parameters."
        )
    for name, parameter in parameters.items():
        if (
            tuple(parameter.shape) != tuple(expected[name].shape)
            or parameter.dtype != expected[name].dtype
        ):
            raise Lfm2StructuralRefinementError(
                f"Fitted {name} does not match the prebuilt target Parameter."
            )
    identities = {name: id(value) for name, value in parameters.items()}
    gradient_state = {
        name: (
            bool(value.requires_grad),
            None if value.grad is None else id(value.grad),
            None
            if value.grad is None
            else baseline._tensor_sha256(value.grad),
        )
        for name, value in parameters.items()
    }
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(
                expected[name].to(
                    device=parameter.device, dtype=parameter.dtype
                )
            )
    installed = {
        name: baseline._tensor_sha256(value)
        for name, value in parameters.items()
    }
    expected_hashes = {
        name: baseline._tensor_sha256(value)
        for name, value in expected.items()
    }
    preserved = all(
        id(parameters[name]) == identities[name]
        and (
            bool(parameters[name].requires_grad),
            None
            if parameters[name].grad is None
            else id(parameters[name].grad),
            None
            if parameters[name].grad is None
            else baseline._tensor_sha256(parameters[name].grad),
        )
        == gradient_state[name]
        for name in parameters
    )
    if not preserved or installed != expected_hashes:
        raise Lfm2StructuralRefinementError(
            "Analytical RMSNorm installation changed identity/gradient state "
            "or failed to install the fitted bytes."
        )
    return {
        "accepted": True,
        "parameter_identities_preserved": True,
        "gradient_state_preserved": True,
        "installed_weight_sha256": installed,
        "model_forward_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
    }


def refine_layer_from_sealed_cases(
    source_layer: nn.Module,
    target_layer: nn.Module,
    cases: SealedLayerCases,
    *,
    global_residual_map_sha256: str,
    short_conv_channel_assignment: (
        Lfm2SparseResidualAssignment | None
    ) = None,
    short_conv_channel_assignment_report: Mapping[str, Any] | None = None,
    ridge: float,
    max_prior_delta_ratio: float | None,
    cluster_ridge: float,
    cluster_max_iterations: int,
    cluster_distance_chunk_size: int,
) -> dict[str, Any]:
    """Compile first, then transactionally overwrite one existing target layer."""

    if not isinstance(cases, SealedLayerCases):
        raise TypeError("cases must be SealedLayerCases.")
    source_is_attention = bool(
        getattr(source_layer, "is_attention_layer", False)
    )
    target_is_attention = bool(
        getattr(target_layer, "is_attention_layer", False)
    )
    if source_is_attention != target_is_attention:
        raise Lfm2StructuralRefinementError(
            "Parent and target layer operator kinds differ."
        )
    expected_kind = "full_attention" if target_is_attention else "short_conv"
    if cases.operator_kind != expected_kind:
        raise Lfm2StructuralRefinementError(
            "Sealed cases do not match the target layer operator."
        )
    if (
        not isinstance(global_residual_map_sha256, str)
        or len(global_residual_map_sha256) != 64
        or any(
            value not in "0123456789abcdef"
            for value in global_residual_map_sha256
        )
    ):
        raise Lfm2StructuralRefinementError(
            "global_residual_map_sha256 is not a lowercase SHA-256 digest."
        )
    if target_is_attention:
        if (
            short_conv_channel_assignment is not None
            or short_conv_channel_assignment_report is not None
        ):
            raise Lfm2StructuralRefinementError(
                "Attention refinement cannot consume a short-conv channel "
                "assignment."
            )
    elif (
        not isinstance(
            short_conv_channel_assignment,
            Lfm2SparseResidualAssignment,
        )
        or not isinstance(short_conv_channel_assignment_report, Mapping)
        or short_conv_channel_assignment_report.get(
            "short_conv_channel_assignment_sha256"
        )
        != short_conv_channel_assignment.assignment_sha256
        or short_conv_channel_assignment_report.get(
            "global_residual_map_sha256"
        )
        != global_residual_map_sha256
    ):
        raise Lfm2StructuralRefinementError(
            "Short-conv refinement requires its separately derived, "
            "digest-bound B/C/x channel assignment."
        )
    topology_before = _module_parameter_topology(target_layer)
    target_state_before = _module_parameter_state_sha256(target_layer)
    backup = {
        name: parameter.detach().to(device="cpu").clone()
        for name, parameter in target_layer.named_parameters(
            remove_duplicate=False
        )
    }
    source_state_before = _module_parameter_state_sha256(source_layer)

    source_mlp = source_layer.feed_forward
    target_mlp = target_layer.feed_forward
    mlp_priors = {
        "w1.weight": _clone(target_mlp.w1.weight),
        "w3.weight": _clone(target_mlp.w3.weight),
        "w2.weight": _clone(target_mlp.w2.weight),
    }
    compiled_mlp = compile_lfm2_structural_residual_mlp_fold(
        source_gate_weight=source_mlp.w1.weight,
        source_up_weight=source_mlp.w3.weight,
        source_down_weight=source_mlp.w2.weight,
        target_gate_weight_template=target_mlp.w1.weight,
        target_up_weight_template=target_mlp.w3.weight,
        target_down_weight_template=target_mlp.w2.weight,
        cases=cases.mlp,
        gate_weight_prior=mlp_priors["w1.weight"],
        up_weight_prior=mlp_priors["w3.weight"],
        down_weight_prior=mlp_priors["w2.weight"],
        residual_assignment_sha256=global_residual_map_sha256,
        ridge=ridge,
        max_prior_delta_ratio=max_prior_delta_ratio,
        cluster_ridge=cluster_ridge,
        cluster_max_iterations=cluster_max_iterations,
        cluster_distance_chunk_size=cluster_distance_chunk_size,
    )

    operator_report: dict[str, Any]
    norm_install: dict[str, Any] | None = None
    try:
        if cases.target_norms is not None:
            norm_install = _install_fitted_layer_norms(
                target_layer, cases.target_norms
            )
        if target_is_attention:
            source_attention = source_layer.self_attn
            target_attention = target_layer.self_attn
            attention_priors = {
                "q_proj.weight": _clone(target_attention.q_proj.weight),
                "k_proj.weight": _clone(target_attention.k_proj.weight),
                "v_proj.weight": _clone(target_attention.v_proj.weight),
                "out_proj.weight": _clone(
                    target_attention.out_proj.weight
                ),
            }
            compiled_attention = (
                compile_lfm2_structural_residual_attention_fold(
                    source_query_weight=source_attention.q_proj.weight,
                    source_key_weight=source_attention.k_proj.weight,
                    source_value_weight=source_attention.v_proj.weight,
                    source_output_weight=source_attention.out_proj.weight,
                    target_query_weight_template=target_attention.q_proj.weight,
                    target_key_weight_template=target_attention.k_proj.weight,
                    target_value_weight_template=target_attention.v_proj.weight,
                    target_output_weight_template=(
                        target_attention.out_proj.weight
                    ),
                    cases=cases.attention,
                    head_dim=int(target_attention.head_dim),
                    query_norm_weight=getattr(
                        getattr(target_attention, "q_layernorm", None),
                        "weight",
                        None,
                    ),
                    key_norm_weight=getattr(
                        getattr(target_attention, "k_layernorm", None),
                        "weight",
                        None,
                    ),
                    query_weight_prior=attention_priors["q_proj.weight"],
                    key_weight_prior=attention_priors["k_proj.weight"],
                    value_weight_prior=attention_priors["v_proj.weight"],
                    output_weight_prior=attention_priors[
                        "out_proj.weight"
                    ],
                    residual_assignment_sha256=(
                        global_residual_map_sha256
                    ),
                    norm_epsilon=_norm_epsilon(target_attention),
                    ridge=ridge,
                    max_prior_delta_ratio=max_prior_delta_ratio,
                    cluster_ridge=cluster_ridge,
                    cluster_max_iterations=cluster_max_iterations,
                    cluster_distance_chunk_size=(
                        cluster_distance_chunk_size
                    ),
                )
            )
            compiled_layer = compose_lfm2_structural_layer_fold(
                compiled_attention, compiled_mlp
            )
            install = install_lfm2_structural_layer_fold(
                target_layer, compiled_layer
            )
            if not install.accepted:
                raise Lfm2StructuralRefinementError(
                    "Atomic QKVO plus SwiGLU install was not accepted."
                )
            operator_report = {
                "kind": "full_attention",
                "global_residual_map_sha256": (
                    global_residual_map_sha256
                ),
                "short_conv_channel_assignment_sha256": None,
                "compiled": compiled_attention.as_dict(),
                "install": install.as_dict(),
                "atomic_qkvo_and_swiglu": True,
                "source_derived_prior_state_sha256": _json_sha256(
                    {
                        name: _tensor_descriptor(value)
                        for name, value in {
                            **attention_priors,
                            **{
                                f"feed_forward.{name}": prior
                                for name, prior in mlp_priors.items()
                            },
                        }.items()
                    }
                ),
            }
        else:
            assert isinstance(
                short_conv_channel_assignment,
                Lfm2SparseResidualAssignment,
            )
            assert isinstance(short_conv_channel_assignment_report, Mapping)
            source_conv = source_layer.conv
            target_conv = target_layer.conv
            conv_priors = {
                "in_proj.weight": _clone(target_conv.in_proj.weight),
                "in_proj.bias": _clone(target_conv.in_proj.bias),
                "out_proj.weight": _clone(target_conv.out_proj.weight),
                "out_proj.bias": _clone(target_conv.out_proj.bias),
            }
            compiled_conv = compile_lfm2_structural_residual_short_conv_fold(
                source_in_proj_weight=source_conv.in_proj.weight,
                source_conv_weight=source_conv.conv.weight,
                source_out_proj_weight=source_conv.out_proj.weight,
                source_in_proj_bias=source_conv.in_proj.bias,
                source_conv_bias=source_conv.conv.bias,
                source_out_proj_bias=source_conv.out_proj.bias,
                target_in_proj_weight_template=target_conv.in_proj.weight,
                target_conv_weight_template=target_conv.conv.weight,
                target_out_proj_weight_template=target_conv.out_proj.weight,
                target_in_proj_bias_template=target_conv.in_proj.bias,
                target_conv_bias_template=target_conv.conv.bias,
                target_out_proj_bias_template=target_conv.out_proj.bias,
                cases=cases.short_conv,
                residual_assignment=short_conv_channel_assignment,
                in_proj_weight_prior=conv_priors["in_proj.weight"],
                in_proj_bias_prior=conv_priors["in_proj.bias"],
                out_proj_weight_prior=conv_priors["out_proj.weight"],
                out_proj_bias_prior=conv_priors["out_proj.bias"],
                ridge=ridge,
                max_prior_delta_ratio=max_prior_delta_ratio,
            )
            conv_install = install_lfm2_structural_short_conv_fold(
                target_conv, compiled_conv
            )
            mlp_install = install_lfm2_structural_mlp_fold(
                target_mlp, compiled_mlp
            )
            if not conv_install.accepted or not mlp_install.accepted:
                raise Lfm2StructuralRefinementError(
                    "Short-conv plus SwiGLU transaction was not accepted."
                )
            operator_report = {
                "kind": "short_conv",
                "global_residual_map_sha256": (
                    global_residual_map_sha256
                ),
                "short_conv_channel_assignment_sha256": (
                    short_conv_channel_assignment.assignment_sha256
                ),
                "short_conv_channel_assignment": dict(
                    short_conv_channel_assignment_report
                ),
                "compiled": compiled_conv.as_dict(),
                "short_conv_install": conv_install.as_dict(),
                "mlp_install": mlp_install.as_dict(),
                "atomic_short_conv_and_swiglu_driver_transaction": True,
                "source_derived_prior_state_sha256": _json_sha256(
                    {
                        name: (
                            None
                            if value is None
                            else _tensor_descriptor(value)
                        )
                        for name, value in {
                            **conv_priors,
                            **{
                                f"feed_forward.{name}": prior
                                for name, prior in mlp_priors.items()
                            },
                        }.items()
                    }
                ),
            }
    except Exception:
        _restore_parameters(target_layer, backup)
        if _module_parameter_topology(target_layer) != topology_before:
            raise Lfm2StructuralRefinementError(
                "Layer refinement failed and rollback changed target topology."
            )
        raise

    topology_after = _module_parameter_topology(target_layer)
    source_state_after = _module_parameter_state_sha256(source_layer)
    if topology_after != topology_before:
        _restore_parameters(target_layer, backup)
        raise Lfm2StructuralRefinementError(
            "Refinement replaced a target Parameter or changed shape/dtype."
        )
    if source_state_after != source_state_before:
        _restore_parameters(target_layer, backup)
        raise Lfm2StructuralRefinementError(
            "Refinement changed the immutable parent layer."
        )
    if (
        not compiled_mlp.full_source_coverage
        or compiled_mlp.residual_assignment_sha256
        != global_residual_map_sha256
    ):
        _restore_parameters(target_layer, backup)
        raise Lfm2StructuralRefinementError(
            "Compiled SwiGLU lost source coverage or global-map binding."
        )
    return {
        "layer_index": cases.layer_index,
        "cases": cases.as_dict(),
        "target_parameter_topology_preserved": True,
        "parent_layer_immutable": True,
        "target_state_sha256_before": target_state_before,
        "target_state_sha256_after": _module_parameter_state_sha256(
            target_layer
        ),
        "global_residual_map_sha256": global_residual_map_sha256,
        "short_conv_channel_assignment_sha256": (
            None
            if short_conv_channel_assignment is None
            else short_conv_channel_assignment.assignment_sha256
        ),
        "mlp": compiled_mlp.as_dict(),
        "operator": operator_report,
        "target_rms_norms": (
            None
            if cases.target_norms is None
            else {
                "compiled": cases.target_norms.as_dict(),
                "install": norm_install,
                "transactionally_bound_to_operator_and_mlp_install": True,
            }
        ),
        "full_source_coverage": True,
        "source_units_dropped": 0,
        "baseline_target_values_used_only_as_explicit_priors": True,
        "blank_initialization_residue_used": False,
        "parent_forward_calls": 0,
        "target_forward_calls": 0,
        "trace_regeneration_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }


def load_sealed_trace_batches(
    cache: Any,
    *,
    maximum: int,
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(maximum, bool) or maximum < 0:
        raise Lfm2StructuralRefinementError(
            "max_cache_batches must be non-negative; zero means every "
            "verified sealed batch."
        )
    result: list[Mapping[str, Any]] = []
    for batch_index, batch in enumerate(cache.iter_batches()):
        if maximum and batch_index >= maximum:
            break
        tensors = getattr(batch, "tensors", None)
        if not isinstance(tensors, Mapping) or not tensors:
            raise Lfm2StructuralRefinementError(
                "Verified cache batch has no tensor mapping."
            )
        result.append(MappingProxyType(dict(tensors)))
    if not result:
        raise Lfm2StructuralRefinementError(
            "Verified cache yielded no sealed trace batches."
        )
    return tuple(result)


def refine_selected_layers(
    parent: nn.Module,
    child: nn.Module,
    trace_batches: Sequence[Mapping[str, Any]],
    mapping: ResidualCoordinateMap,
    selected_layers: Sequence[int],
    *,
    max_phases_per_batch: int,
    ridge: float,
    max_prior_delta_ratio: float | None,
    cluster_ridge: float,
    cluster_max_iterations: int,
    cluster_distance_chunk_size: int,
) -> dict[str, Any]:
    _source_width, target_width, global_map_sha256 = (
        _global_coordinate_map_contract(mapping)
    )
    source_layers = baseline._lfm2_layers(parent)
    target_layers = baseline._lfm2_layers(child)
    selected = tuple(int(value) for value in selected_layers)
    if (
        not selected
        or tuple(sorted(set(selected))) != selected
        or min(selected) < 0
        or max(selected) >= len(source_layers)
    ):
        raise Lfm2StructuralRefinementError(
            "selected_layers must be unique, ordered, and in range."
        )
    parent_before = fingerprint_lfm2_parent(parent)
    target_topology_before = baseline.parameter_topology_snapshot(child)
    child_before = baseline.model_state_sha256(child)
    unselected_before = _state_subset_sha256(
        child, excluded_layers=selected
    )
    trace_before = _trace_tensor_sha256(trace_batches, selected)
    layer_reports = []
    short_conv_assignments: dict[str, dict[str, Any]] = {}
    for layer_index in selected:
        source_layer = source_layers[layer_index]
        target_layer = target_layers[layer_index]
        is_attention = bool(
            getattr(source_layer, "is_attention_layer", False)
        )
        layer_cases = assemble_sealed_layer_cases(
            trace_batches,
            layer_index=layer_index,
            is_attention_layer=is_attention,
            mapping=mapping,
            target_layer=target_layer,
            norm_ridge=ridge,
            max_phases_per_batch=max_phases_per_batch,
        )
        short_conv_assignment = None
        short_conv_assignment_report = None
        if not is_attention:
            (
                short_conv_assignment,
                short_conv_assignment_report,
            ) = build_short_conv_channel_assignment(
                layer_cases.short_conv,
                layer_index=layer_index,
                target_width=target_width,
                global_residual_map_sha256=global_map_sha256,
            )
            short_conv_assignments[str(layer_index)] = dict(
                short_conv_assignment_report
            )
        layer_reports.append(
            refine_layer_from_sealed_cases(
                source_layer,
                target_layer,
                layer_cases,
                global_residual_map_sha256=global_map_sha256,
                short_conv_channel_assignment=short_conv_assignment,
                short_conv_channel_assignment_report=(
                    short_conv_assignment_report
                ),
                ridge=ridge,
                max_prior_delta_ratio=max_prior_delta_ratio,
                cluster_ridge=cluster_ridge,
                cluster_max_iterations=cluster_max_iterations,
                cluster_distance_chunk_size=(
                    cluster_distance_chunk_size
                ),
            )
        )
    parent_after = fingerprint_lfm2_parent(parent)
    target_topology_after = baseline.parameter_topology_snapshot(child)
    trace_after = _trace_tensor_sha256(trace_batches, selected)
    unselected_after = _state_subset_sha256(
        child, excluded_layers=selected
    )
    if parent_after != parent_before:
        raise Lfm2StructuralRefinementError(
            "Immutable parent changed during refinement."
        )
    if target_topology_after != target_topology_before:
        raise Lfm2StructuralRefinementError(
            "Target Parameter topology changed during refinement."
        )
    if trace_after != trace_before:
        raise Lfm2StructuralRefinementError(
            "Sealed trace tensors changed during refinement."
        )
    if unselected_after != unselected_before:
        raise Lfm2StructuralRefinementError(
            "Refinement changed state outside selected target layers."
        )
    return {
        "selected_layers": list(selected),
        "all_layers_refined": len(selected) == len(source_layers),
        "layer_reports": layer_reports,
        "baseline_child_state_sha256": child_before,
        "refined_child_state_sha256": baseline.model_state_sha256(child),
        "parent_identity_sha256": parent_before.identity_sha256,
        "parent_immutable": True,
        "target_parameter_objects_shapes_dtypes_unchanged": True,
        "unselected_target_state_unchanged": True,
        "sealed_trace_unchanged": True,
        "sealed_trace_sha256": trace_before,
        "global_residual_map_sha256": global_map_sha256,
        "short_conv_channel_assignments": short_conv_assignments,
        "dense_global_basis_interpreted_as_sparse_assignment": False,
        "source_coverage_fraction": 1.0,
        "source_units_dropped": 0,
        "parent_forward_calls": 0,
        "target_forward_calls": 0,
        "parent_trace_regeneration_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }


def refine_final_embedding_norm(
    parent: nn.Module,
    child: nn.Module,
    trace_batches: Sequence[Mapping[str, Any]],
    mapping: ResidualCoordinateMap,
    *,
    max_phases_per_batch: int,
    ridge: float,
) -> dict[str, Any]:
    """Fit the concrete final LFM2 RMSNorm from the sealed last residual."""

    _source_width, _target_width, global_map_sha256 = (
        _global_coordinate_map_contract(mapping)
    )
    source_layers = baseline._lfm2_layers(parent)
    layer_index = len(source_layers) - 1
    parent_norm = baseline._lfm2_final_norm(parent)
    target_norm = baseline._lfm2_final_norm(child)
    parent_weight = getattr(parent_norm, "weight", None)
    target_weight = getattr(target_norm, "weight", None)
    if not isinstance(parent_weight, nn.Parameter) or not isinstance(
        target_weight, nn.Parameter
    ):
        raise Lfm2StructuralRefinementError(
            "Parent/target final embedding norms must expose Parameters."
        )
    parent_epsilon = _strict_rms_norm_epsilon(
        parent_norm, "parent.embedding_norm"
    )
    target_inputs: list[torch.Tensor] = []
    desired_outputs: list[torch.Tensor] = []
    phase_sequences: list[tuple[str, ...]] = []
    global_phase_tensor_names: set[str] = set()
    for tensors in trace_batches:
        phases = _phases_for_layer(
            tensors,
            layer_index,
            maximum=max_phases_per_batch,
        )
        phase_sequences.append(phases)
        for phase in phases:
            base = f"{phase}.layer.{layer_index:03d}."
            name = base + "residual.post_layer"
            if name not in tensors:
                raise Lfm2StructuralRefinementError(
                    f"Sealed trace is missing final residual {name!r}."
                )
            parent_residual = tensors[name]
            parent_normalized = _materialized_rms_norm(
                parent_residual,
                parent_weight,
                epsilon=parent_epsilon,
            )
            target_inputs.append(
                project_with_global_residual_map(mapping, parent_residual)
            )
            desired_outputs.append(
                project_with_global_residual_map(
                    mapping, parent_normalized
                )
            )
            global_phase_tensor_names.update(
                key
                for key in tensors
                if key.startswith(phase + ".") and ".layer." not in key
            )
    non_logit_global_tensors = sorted(
        name
        for name in global_phase_tensor_names
        if not (
            name.endswith(".logits")
            or name.endswith(".full_prefix_logits")
        )
    )
    fitted = _fit_target_rms_norm(
        role="model.embedding_norm",
        input_values=target_inputs,
        desired_values=desired_outputs,
        target_norm=target_norm,
        ridge=ridge,
    )
    topology_before = _module_parameter_topology(target_norm)
    parent_before = _module_parameter_state_sha256(parent_norm)
    child_before = baseline.model_state_sha256(child)
    backup = target_weight.detach().to(device="cpu").clone()
    try:
        with torch.no_grad():
            target_weight.copy_(
                fitted.weight.to(
                    device=target_weight.device,
                    dtype=target_weight.dtype,
                )
            )
        if (
            _module_parameter_topology(target_norm) != topology_before
            or baseline._tensor_sha256(target_weight)
            != fitted.fitted_weight_sha256
            or _module_parameter_state_sha256(parent_norm) != parent_before
        ):
            raise Lfm2StructuralRefinementError(
                "Final embedding norm install violated identity/content/parent "
                "immutability."
            )
    except Exception:
        with torch.no_grad():
            target_weight.copy_(
                backup.to(
                    device=target_weight.device, dtype=target_weight.dtype
                )
            )
        raise
    return {
        "layer_index_source_residual": layer_index,
        "global_residual_map_sha256": global_map_sha256,
        "phase_sequences": [list(value) for value in phase_sequences],
        "cache_input_field": (
            f"*.layer.{layer_index:03d}.residual.post_layer"
        ),
        "cache_global_phase_tensor_inventory": sorted(
            global_phase_tensor_names
        ),
        "cache_non_logit_global_tensor_inventory": non_logit_global_tensors,
        "explicit_final_normalized_cache_tensor_used": False,
        "explicit_final_normalized_cache_tensor_discovered": False,
        "explicit_final_normalized_cache_schema_field": None,
        "cache_inventory_verified_before_analytical_derivation": True,
        "desired_output_derivation": (
            "analytical-immutable-parent-embedding-norm-over-sealed-final-residual"
            "-then-global-residual-map"
        ),
        "parent_norm_epsilon": parent_epsilon,
        "fit": fitted.as_dict(),
        "install": {
            "accepted": True,
            "parameter_identity_shape_dtype_gradient_state_preserved": True,
            "parent_final_norm_immutable": True,
        },
        "child_state_sha256_before": child_before,
        "child_state_sha256_after": baseline.model_state_sha256(child),
        "parent_forward_calls": 0,
        "target_forward_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }


def save_and_fresh_reload(
    model: nn.Module,
    *,
    source_directory: Path,
    destination: Path,
    report: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    expected_unique_parameters: int,
    save_function: Callable[..., None] = baseline.save_dense_local_checkpoint,
    load_function: Callable[..., nn.Module] = baseline.load_local_lfm2,
    dense_audit_function: Callable[..., Mapping[str, Any]] = (
        baseline.dense_logical_parameter_audit
    ),
) -> tuple[nn.Module, dict[str, Any]]:
    """Save once and construct an independent model from the new local bytes."""

    expected_state = baseline.model_state_sha256(model)
    save_function(model, source_directory, destination, report)
    reloaded = load_function(destination, device=device, dtype=dtype)
    if reloaded is model:
        raise Lfm2StructuralRefinementError(
            "Fresh reload returned the in-memory refined object."
        )
    audit = dict(
        dense_audit_function(reloaded, require_any_nonzero=True)
    )
    reloaded_state = baseline.model_state_sha256(reloaded)
    if (
        audit["unique_logical_parameters"] != expected_unique_parameters
        or reloaded_state != expected_state
    ):
        raise Lfm2StructuralRefinementError(
            "Fresh local reload changed dense parameter count or content."
        )
    return reloaded, {
        "verified": True,
        "state_sha256": reloaded_state,
        "exact_unique_logical_parameters": audit[
            "unique_logical_parameters"
        ],
        "independent_model_object": True,
        "hub_loader_used": False,
        "network_allowed": False,
    }


def _baseline_contract(
    report: Mapping[str, Any],
) -> tuple[baseline.StructuralCandidate, str, str]:
    candidate_payload = report.get("candidate")
    fold = report.get("fold")
    if not isinstance(candidate_payload, Mapping) or not isinstance(
        fold, Mapping
    ):
        raise Lfm2StructuralRefinementError(
            "Baseline report lacks candidate/fold contracts."
        )
    name = candidate_payload.get("name")
    if name != REQUIRED_BASELINE_CANDIDATE:
        raise Lfm2StructuralRefinementError(
            "This refinement proof is bound to the real h640/Q10/KV5/FF1920 "
            "blank-first checkpoint."
        )
    candidate = baseline._CANDIDATES_BY_NAME[name]
    fold_candidate = fold.get("candidate")
    residual = fold.get("global_residual_map")
    if (
        not isinstance(fold_candidate, Mapping)
        or fold_candidate.get("name") != name
        or not isinstance(residual, Mapping)
        or not isinstance(residual.get("map_sha256"), str)
        or not isinstance(fold.get("folded_state_sha256"), str)
    ):
        raise Lfm2StructuralRefinementError(
            "Baseline report lacks a bound folded state/global map."
        )
    if (
        report.get("dtype_or_packing_reduction_claimed") is not False
        or report.get("training_steps") != 0
        or report.get("hub_loader_used") is not False
    ):
        raise Lfm2StructuralRefinementError(
            "Baseline is not the accepted dense, fold-only local checkpoint."
        )
    return (
        candidate,
        str(residual["map_sha256"]),
        str(fold["folded_state_sha256"]),
    )


def execute(args: argparse.Namespace) -> dict[str, Any]:
    parent_directory = baseline._local_model_directory(args.parent_model)
    child_directory = baseline._local_model_directory(args.baseline_child)
    baseline_report = _read_json_object(
        child_directory / "structural-fold-report.json"
    )
    candidate, expected_map_sha, expected_child_state = _baseline_contract(
        baseline_report
    )
    device = torch.device(args.device)
    parent = baseline.load_local_lfm2(
        parent_directory, device=device, dtype=args.dtype
    )
    child = baseline.load_local_lfm2(
        child_directory, device=device, dtype=args.dtype
    )
    parent_fingerprint = fingerprint_lfm2_parent(parent)
    baseline_parent = baseline_report.get("parent")
    if (
        not isinstance(baseline_parent, Mapping)
        or baseline_parent.get("identity_sha256")
        != parent_fingerprint.identity_sha256
    ):
        raise Lfm2StructuralRefinementError(
            "Baseline child is not bound to the loaded local parent."
        )
    if baseline.model_state_sha256(child) != expected_child_state:
        raise Lfm2StructuralRefinementError(
            "Loaded baseline child bytes differ from the accepted fold report."
        )
    baseline.validate_live_structural_topology(parent, child, candidate)
    child_dense = baseline.dense_logical_parameter_audit(
        child, require_any_nonzero=True
    )
    if (
        child_dense["unique_logical_parameters"]
        != candidate.verified_unique_parameters
    ):
        raise Lfm2StructuralRefinementError(
            "Baseline child is not the exact verified h640 dense target."
        )

    cache = open_lfm2_circuit_trace_cache(args.trace_cache, verify=True)
    if cache.parent_fingerprint != parent_fingerprint.identity_sha256:
        raise Lfm2StructuralRefinementError(
            "Sealed trace cache is not bound to the loaded local parent."
        )
    evidence = baseline.extract_cached_fold_evidence(
        cache,
        parent,
        max_rows=args.max_rows,
        max_attention_rows=args.max_attention_rows,
        max_vocab_rows=args.max_vocab_rows,
    )
    baseline_cache = baseline_report.get("cache")
    if (
        not isinstance(baseline_cache, Mapping)
        or baseline_cache.get("cached_evidence_sha256")
        != evidence.evidence_sha256
    ):
        raise Lfm2StructuralRefinementError(
            "Rebuilt cached evidence differs from the baseline compiler input."
        )
    mapping, mapping_report = baseline.build_global_residual_map(
        evidence, candidate
    )
    if mapping_report["map_sha256"] != expected_map_sha:
        raise Lfm2StructuralRefinementError(
            "Deterministically rebuilt global residual map differs from baseline."
        )
    _source_width, _target_width, rebuilt_map_sha256 = (
        _global_coordinate_map_contract(mapping)
    )
    if rebuilt_map_sha256 != expected_map_sha:
        raise Lfm2StructuralRefinementError(
            "Global coordinate-map object digest differs from its baseline "
            "report."
        )
    source_layers = baseline._lfm2_layers(parent)
    selected_layers = parse_layer_selection(
        args.layers, len(source_layers)
    )
    trace_batches = load_sealed_trace_batches(
        cache, maximum=args.max_cache_batches
    )
    refinement = refine_selected_layers(
        parent,
        child,
        trace_batches,
        mapping,
        selected_layers,
        max_phases_per_batch=args.max_phases_per_batch,
        ridge=args.ridge,
        max_prior_delta_ratio=args.max_prior_delta_ratio,
        cluster_ridge=args.cluster_ridge,
        cluster_max_iterations=args.cluster_max_iterations,
        cluster_distance_chunk_size=args.cluster_distance_chunk_size,
    )
    final_norm = refine_final_embedding_norm(
        parent,
        child,
        trace_batches,
        mapping,
        max_phases_per_batch=args.max_phases_per_batch,
        ridge=args.ridge,
    )
    refinement["refined_child_state_sha256_before_final_embedding_norm"] = (
        refinement["refined_child_state_sha256"]
    )
    refinement["refined_child_state_sha256"] = final_norm[
        "child_state_sha256_after"
    ]
    refinement["final_embedding_norm"] = final_norm
    refinement_report: dict[str, Any] = {
        "schema": REFINEMENT_SCHEMA,
        "baseline_checkpoint": {
            "directory": str(child_directory),
            "state_sha256": expected_child_state,
            "report_run_sha256": baseline_report.get("run_sha256"),
            "compiler_created_blank": True,
            "source_derived_target_shaped_priors": True,
        },
        "parent": parent_fingerprint.as_dict(),
        "candidate": candidate.as_dict(),
        "child": {
            "exact_unique_logical_parameters": child_dense[
                "unique_logical_parameters"
            ],
            "compression_metric": "unique-learned-Parameter-elements-only",
            "logical_parameter_reduction_fraction": (
                1.0
                - child_dense["unique_logical_parameters"]
                / baseline.EXPECTED_PARENT_UNIQUE_PARAMETERS
            ),
            "architecture_changed_during_refinement": False,
        },
        "cache": {
            "cache_key": cache.cache_key,
            "corpus_sha256": cache.corpus_fingerprint,
            "evidence_sha256": evidence.evidence_sha256,
            "sealed_batches_used": len(trace_batches),
            "parent_forward_calls": 0,
            "trace_regeneration_calls": 0,
        },
        "global_residual_map": {
            **mapping_report,
            "matches_baseline_map_sha256": True,
            "coordinate_map_protocol_verified": True,
            "dense_basis_interpreted_as_sparse_assignment": False,
            "short_conv_channel_assignments": refinement[
                "short_conv_channel_assignments"
            ],
        },
        "refinement": refinement,
        "dense_floating_checkpoint": True,
        "dtype_or_packing_reduction_claimed": False,
        "hub_loader_used": False,
        "network_allowed": False,
        "parent_forward_calls": 0,
        "target_forward_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    refinement_report["run_sha256"] = _json_sha256(refinement_report)

    # Keep the checkpoint's canonical structural-fold contract consumable by
    # the existing fresh-reload whole-output evaluator.  The refinement is a
    # deterministic continuation of the same blank-first fold, so retain its
    # seed-erasure/complete-overwrite proof while rebinding the folded state to
    # the analytically refined bytes.
    structural_report = copy.deepcopy(baseline_report)
    structural_report.pop("run_sha256", None)
    structural_fold = structural_report.get("fold")
    if not isinstance(structural_fold, dict):
        raise Lfm2StructuralRefinementError(
            "Baseline structural fold contract is not mutable JSON."
        )
    structural_fold["folded_state_sha256"] = refinement[
        "refined_child_state_sha256"
    ]
    structural_fold["residual_attention_refinement"] = {
        "schema": REFINEMENT_SCHEMA,
        "baseline_fold_run_sha256": baseline_report.get("run_sha256"),
        "selected_layers": list(selected_layers),
        "all_layers_refined": refinement["all_layers_refined"],
        "global_residual_map_sha256": rebuilt_map_sha256,
        "short_conv_channel_assignment_sha256_by_layer": {
            layer: value["short_conv_channel_assignment_sha256"]
            for layer, value in refinement[
                "short_conv_channel_assignments"
            ].items()
        },
        "dense_global_basis_interpreted_as_sparse_assignment": False,
        "sealed_trace_sha256": refinement["sealed_trace_sha256"],
        "source_coverage_fraction": 1.0,
        "source_units_dropped": 0,
        "parent_forward_calls": 0,
        "target_forward_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    structural_report["fresh_local_reload"] = {
        "verified": False,
        "pending_refined_checkpoint_reload": True,
    }
    structural_report["run_sha256"] = _json_sha256(structural_report)
    output = Path(args.output).expanduser().absolute()
    reloaded, reload_report = save_and_fresh_reload(
        child,
        source_directory=child_directory,
        destination=output,
        report=structural_report,
        device=device,
        dtype=args.dtype,
        expected_unique_parameters=candidate.verified_unique_parameters,
    )
    baseline.validate_live_structural_topology(
        parent, reloaded, candidate
    )
    structural_report["fresh_local_reload"] = reload_report
    structural_report["run_sha256"] = _json_sha256(
        {
            key: value
            for key, value in structural_report.items()
            if key != "run_sha256"
        }
    )
    (output / "structural-fold-report.json").write_bytes(
        _canonical_json(structural_report) + b"\n"
    )
    refinement_report["fresh_local_reload"] = reload_report
    refinement_report["resulting_structural_fold_run_sha256"] = (
        structural_report["run_sha256"]
    )
    refinement_report["run_sha256"] = _json_sha256(
        {
            key: value
            for key, value in refinement_report.items()
            if key != "run_sha256"
        }
    )
    (output / "structural-refinement-report.json").write_bytes(
        _canonical_json(refinement_report) + b"\n"
    )
    return refinement_report


def _parse_dtype(value: str) -> torch.dtype:
    values = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return values[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            "dtype must be float16, bfloat16, or float32"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-model", required=True)
    parser.add_argument("--baseline-child", required=True)
    parser.add_argument("--trace-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--layers",
        default="all",
        help="all, a comma list (2,4,6), or inclusive ranges (2-6)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", type=_parse_dtype, default=torch.bfloat16
    )
    parser.add_argument(
        "--max-cache-batches",
        type=int,
        default=0,
        help="0 uses every verified sealed cache batch",
    )
    parser.add_argument(
        "--max-phases-per-batch",
        type=int,
        default=0,
        help="0 keeps every sealed prefill/decode phase",
    )
    parser.add_argument("--max-rows", type=int, default=4096)
    parser.add_argument("--max-attention-rows", type=int, default=512)
    parser.add_argument("--max-vocab-rows", type=int, default=2048)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument(
        "--max-prior-delta-ratio", type=float, default=0.5
    )
    parser.add_argument("--cluster-ridge", type=float, default=1e-8)
    parser.add_argument(
        "--cluster-max-iterations", type=int, default=8
    )
    parser.add_argument(
        "--cluster-distance-chunk-size", type=int, default=512
    )
    return parser


def main() -> int:
    report = execute(build_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
