"""Suffix-aware target-native CPCF folding for LFM2 short convolution.

The operator's causal state cannot be fitted from an isolated sampled suffix.
This compiler therefore consumes the actual child's complete operator-input
prefix and the parent's complete authenticated B/C/x histories.  It uses the
prefix to fit and warm the target B/C/x producer and depthwise FIR, then uses
only the authenticated residual suffix to fit the target output writer.

Every source channel is assigned to exactly one target channel.  No source
channel is selected, dropped, zeroed, pruned, or quantized.  All fitting is
deterministic CPU float64 linear algebra under ``torch.no_grad()``; the target
templates contribute only topology and dtype, never initial values.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


LFM2_CPCF_SHORT_CONV_FOLD_SCHEMA = "neural-forge-lfm2-cpcf-short-conv-fold-v1"


class Lfm2CpcfShortConvFoldError(ValueError):
    """CPCF short-convolution evidence or a compiled fold is invalid."""


class Lfm2CpcfShortConvValidationError(Lfm2CpcfShortConvFoldError):
    """A materialized fold failed fit or heldout behavior gates."""

    def __init__(self, message: str, report: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.report = MappingProxyType(dict(report))


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for LFM2 CPCF short-convolution folding."
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
        raise Lfm2CpcfShortConvFoldError(
            "CPCF short-convolution metadata must be finite canonical JSON."
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _dtype_name(value: Any) -> str:
    return str(value.dtype).removeprefix("torch.")


def _tensor_sha256(value: Any, name: str) -> str:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2CpcfShortConvFoldError(
            f"{name} must be a materialized dense tensor for authentication."
        )
    detached = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": _dtype_name(detached),
        "shape": [int(size) for size in detached.shape],
        "layout": str(detached.layout),
    }
    digest = hashlib.sha256(_canonical_json(descriptor))
    digest.update(detached.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tensor_bundle_sha256(values: Mapping[str, Any]) -> str:
    return _sha256(
        {name: _tensor_sha256(value, name) for name, value in sorted(values.items())}
    )


def _finite_float_tensor(
    value: Any,
    name: str,
    *,
    dimensions: int,
) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != dimensions
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or value.is_complex()
    ):
        raise Lfm2CpcfShortConvFoldError(
            f"{name} must be a nonempty dense real floating rank-{dimensions} tensor."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CpcfShortConvFoldError(f"{name} contains NaN or infinity.")
    return result


def _template_spec(
    value: Any,
    name: str,
    *,
    dimensions: int,
) -> tuple[Any, tuple[int, ...]]:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != dimensions
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or value.is_complex()
    ):
        raise Lfm2CpcfShortConvFoldError(
            f"{name} must be a nonempty materialized dense real floating "
            f"rank-{dimensions} target template."
        )
    return value.dtype, tuple(int(size) for size in value.shape)


def _nonnegative_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise Lfm2CpcfShortConvFoldError(f"{name} must be a finite nonnegative number.")
    return float(value)


def _bounded_float(
    value: Any,
    name: str,
    *,
    lower: float,
    upper: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Lfm2CpcfShortConvFoldError(f"{name} must be in [{lower}, {upper}].")
    result = float(value)
    if not math.isfinite(result) or result < lower or result > upper:
        raise Lfm2CpcfShortConvFoldError(f"{name} must be in [{lower}, {upper}].")
    return result


def _case_authentication_body(
    *,
    case_id: str,
    split: str,
    suffix_start: int,
    actual_child_operator_input_history: Any,
    parent_b_history: Any,
    parent_c_history: Any,
    parent_x_history: Any,
    parent_post_operator_residual_target_space_suffix: Any,
    actual_child_pre_operator_residual_suffix: Any,
    valid_prefix_mask: Any | None,
) -> dict[str, Any]:
    tensors = {
        "actual_child_operator_input_history": (actual_child_operator_input_history),
        "parent_b_history": parent_b_history,
        "parent_c_history": parent_c_history,
        "parent_x_history": parent_x_history,
        "parent_post_operator_residual_target_space_suffix": (
            parent_post_operator_residual_target_space_suffix
        ),
        "actual_child_pre_operator_residual_suffix": (
            actual_child_pre_operator_residual_suffix
        ),
    }
    if valid_prefix_mask is not None:
        tensors["valid_prefix_mask"] = valid_prefix_mask
    return {
        "schema": LFM2_CPCF_SHORT_CONV_FOLD_SCHEMA,
        "scope": (
            "full_actual_child_operator_input_and_parent_bcx_prefix_plus_"
            "residual_endpoint_suffix"
        ),
        "case_id": case_id,
        "split": split,
        "suffix_start": suffix_start,
        "tensors": {
            name: _tensor_sha256(value, f"case.{name}")
            for name, value in sorted(tensors.items())
        },
    }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfShortConvFoldCase:
    """One independent causal prefix with an authenticated residual suffix."""

    case_id: str
    split: str
    suffix_start: int
    actual_child_operator_input_history: Any
    parent_b_history: Any
    parent_c_history: Any
    parent_x_history: Any
    parent_post_operator_residual_target_space_suffix: Any
    actual_child_pre_operator_residual_suffix: Any
    valid_prefix_mask: Any | None = None
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise Lfm2CpcfShortConvFoldError("case_id must be nonempty text.")
        split = str(self.split).strip().lower()
        if split not in {"selection", "fit", "validation"}:
            raise Lfm2CpcfShortConvFoldError(
                "split must be 'selection', 'fit', or 'validation'."
            )
        if (
            isinstance(self.suffix_start, bool)
            or not isinstance(self.suffix_start, int)
            or self.suffix_start < 0
        ):
            raise Lfm2CpcfShortConvFoldError(
                "suffix_start must be a nonnegative integer."
            )
        case_id = self.case_id.strip()
        body = _case_authentication_body(
            case_id=case_id,
            split=split,
            suffix_start=self.suffix_start,
            actual_child_operator_input_history=(
                self.actual_child_operator_input_history
            ),
            parent_b_history=self.parent_b_history,
            parent_c_history=self.parent_c_history,
            parent_x_history=self.parent_x_history,
            parent_post_operator_residual_target_space_suffix=(
                self.parent_post_operator_residual_target_space_suffix
            ),
            actual_child_pre_operator_residual_suffix=(
                self.actual_child_pre_operator_residual_suffix
            ),
            valid_prefix_mask=self.valid_prefix_mask,
        )
        digest = _sha256(body)
        if self.authentication_sha256 and self.authentication_sha256 != digest:
            raise Lfm2CpcfShortConvFoldError(
                f"Case {case_id!r} authentication digest does not match its evidence."
            )
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "authentication_sha256", digest)

    def verify_authentication(self) -> bool:
        body = _case_authentication_body(
            case_id=self.case_id,
            split=self.split,
            suffix_start=self.suffix_start,
            actual_child_operator_input_history=(
                self.actual_child_operator_input_history
            ),
            parent_b_history=self.parent_b_history,
            parent_c_history=self.parent_c_history,
            parent_x_history=self.parent_x_history,
            parent_post_operator_residual_target_space_suffix=(
                self.parent_post_operator_residual_target_space_suffix
            ),
            actual_child_pre_operator_residual_suffix=(
                self.actual_child_pre_operator_residual_suffix
            ),
            valid_prefix_mask=self.valid_prefix_mask,
        )
        return _sha256(body) == self.authentication_sha256


@dataclass(frozen=True, slots=True)
class Lfm2CpcfShortConvGatePolicy:
    """Fit and heldout residual-endpoint behavior gates."""

    maximum_fit_endpoint_relative_rmse: float = 1.0
    maximum_validation_endpoint_relative_rmse: float = 1.0
    minimum_fit_endpoint_cosine_similarity: float = 0.0
    minimum_validation_endpoint_cosine_similarity: float = 0.0
    enforce: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "maximum_fit_endpoint_relative_rmse",
            _nonnegative_float(
                self.maximum_fit_endpoint_relative_rmse,
                "maximum_fit_endpoint_relative_rmse",
            ),
        )
        object.__setattr__(
            self,
            "maximum_validation_endpoint_relative_rmse",
            _nonnegative_float(
                self.maximum_validation_endpoint_relative_rmse,
                "maximum_validation_endpoint_relative_rmse",
            ),
        )
        object.__setattr__(
            self,
            "minimum_fit_endpoint_cosine_similarity",
            _bounded_float(
                self.minimum_fit_endpoint_cosine_similarity,
                "minimum_fit_endpoint_cosine_similarity",
                lower=-1.0,
                upper=1.0,
            ),
        )
        object.__setattr__(
            self,
            "minimum_validation_endpoint_cosine_similarity",
            _bounded_float(
                self.minimum_validation_endpoint_cosine_similarity,
                "minimum_validation_endpoint_cosine_similarity",
                lower=-1.0,
                upper=1.0,
            ),
        )
        if not isinstance(self.enforce, bool):
            raise Lfm2CpcfShortConvFoldError("enforce must be boolean.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "maximum_fit_endpoint_relative_rmse": (
                self.maximum_fit_endpoint_relative_rmse
            ),
            "maximum_validation_endpoint_relative_rmse": (
                self.maximum_validation_endpoint_relative_rmse
            ),
            "minimum_fit_endpoint_cosine_similarity": (
                self.minimum_fit_endpoint_cosine_similarity
            ),
            "minimum_validation_endpoint_cosine_similarity": (
                self.minimum_validation_endpoint_cosine_similarity
            ),
            "enforce": self.enforce,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CompiledCpcfShortConvFold:
    """Target-shaped CPCF B/C/x producer, depthwise FIR, and suffix writer."""

    in_proj_weight: Any
    in_proj_bias: Any | None
    conv_weight: Any
    conv_bias: Any | None
    out_proj_weight: Any
    out_proj_bias: Any | None
    source_to_target: tuple[int, ...]
    channel_groups: tuple[tuple[int, ...], ...]
    channel_barycentric_weights: tuple[tuple[float, ...], ...]
    source_hidden_size: int
    target_hidden_size: int
    source_fir_length: int
    target_fir_length: int
    fit_metrics: Mapping[str, Any]
    validation_metrics: Mapping[str, Any]
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fit_metrics",
            MappingProxyType(dict(self.fit_metrics)),
        )
        object.__setattr__(
            self,
            "validation_metrics",
            MappingProxyType(dict(self.validation_metrics)),
        )
        object.__setattr__(
            self,
            "report",
            MappingProxyType(dict(self.report)),
        )

    @property
    def full_source_coverage(self) -> bool:
        return sorted(
            source for group in self.channel_groups for source in group
        ) == list(range(self.source_hidden_size))

    @property
    def behavior_gate_passed(self) -> bool:
        return bool(self.report["behavior_gate"]["passed"])

    def state_dict(self) -> Mapping[str, Any]:
        values = {
            "in_proj.weight": self.in_proj_weight,
            "conv.weight": self.conv_weight,
            "out_proj.weight": self.out_proj_weight,
        }
        if self.in_proj_bias is not None:
            values["in_proj.bias"] = self.in_proj_bias
        if self.conv_bias is not None:
            values["conv.bias"] = self.conv_bias
        if self.out_proj_bias is not None:
            values["out_proj.bias"] = self.out_proj_bias
        return MappingProxyType(
            {
                name: value.detach().to(device="cpu").contiguous().clone()
                for name, value in values.items()
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def _relative_rmse(prediction: Any, target: Any) -> float:
    torch = _torch()
    error = torch.sqrt(torch.mean((prediction - target).square()))
    scale = torch.sqrt(torch.mean(target.square()))
    floor = torch.finfo(torch.float64).eps
    return float((error / torch.clamp(scale, min=floor)).item())


def _cosine_similarity(prediction: Any, target: Any) -> float:
    torch = _torch()
    left = prediction.reshape(-1)
    right = target.reshape(-1)
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    floor = torch.finfo(torch.float64).eps
    if float(left_norm.item()) <= floor and float(right_norm.item()) <= floor:
        return 1.0
    if float(left_norm.item()) <= floor or float(right_norm.item()) <= floor:
        return 0.0
    return float(torch.dot(left, right).div(left_norm * right_norm).item())


def _metrics(prediction: Any, target: Any) -> dict[str, float | int]:
    torch = _torch()
    difference = prediction - target
    return {
        "rows": int(prediction.shape[0]),
        "width": int(prediction.shape[1]),
        "relative_rmse": _relative_rmse(prediction, target),
        "cosine_similarity": _cosine_similarity(prediction, target),
        "maximum_absolute_error": float(torch.max(torch.abs(difference)).item()),
        "mean_absolute_error": float(torch.mean(torch.abs(difference)).item()),
    }


def _ridge_solution(
    design: Any,
    target: Any,
    *,
    ridge: float,
    prior: Any | None = None,
) -> Any:
    torch = _torch()
    if int(design.shape[0]) != int(target.shape[0]):
        raise Lfm2CpcfShortConvFoldError(
            "Closed-form design and target row counts differ."
        )
    if int(design.shape[0]) == 0:
        raise Lfm2CpcfShortConvFoldError(
            "Closed-form fitting requires at least one evidence row."
        )
    columns = int(design.shape[1])
    outputs = int(target.shape[1])
    if prior is None:
        prior = torch.zeros(
            (columns, outputs),
            dtype=torch.float64,
        )
    if tuple(prior.shape) != (columns, outputs):
        raise Lfm2CpcfShortConvFoldError("Closed-form prior has the wrong shape.")
    residual = target - design @ prior
    if ridge > 0.0:
        gram = design.T @ design
        gram = gram + ridge * torch.eye(columns, dtype=torch.float64)
        right = design.T @ residual
        try:
            correction = torch.linalg.solve(gram, right)
        except RuntimeError:
            correction = torch.linalg.pinv(gram) @ right
    else:
        correction = torch.linalg.pinv(design) @ residual
    result = prior + correction
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CpcfShortConvFoldError("Closed-form solve produced NaN or infinity.")
    return result.contiguous()


def _fit_affine(
    inputs: Any,
    targets: Any,
    *,
    fit_bias: bool,
    ridge: float,
) -> tuple[Any, Any | None]:
    torch = _torch()
    if fit_bias:
        design = torch.cat(
            (
                inputs,
                torch.ones(
                    (int(inputs.shape[0]), 1),
                    dtype=torch.float64,
                ),
            ),
            dim=1,
        )
    else:
        design = inputs
    solution = _ridge_solution(design, targets, ridge=ridge)
    if fit_bias:
        return solution[:-1].T.contiguous(), solution[-1].contiguous()
    return solution.T.contiguous(), None


def _apply_affine(inputs: Any, weight: Any, bias: Any | None) -> Any:
    output = inputs @ weight.T
    if bias is not None:
        output = output + bias
    return output


def _resize_fir_rows(kernel: Any, target_length: int) -> Any:
    torch = _torch()
    source_length = int(kernel.shape[1])
    if source_length == target_length:
        return kernel.contiguous().clone()
    if source_length == 1:
        return kernel.expand(-1, target_length).contiguous().clone()
    if target_length == 1:
        return kernel.mean(dim=1, keepdim=True)
    return (
        torch.nn.functional.interpolate(
            kernel.unsqueeze(0),
            size=target_length,
            mode="linear",
            align_corners=True,
        )
        .squeeze(0)
        .contiguous()
    )


def _causal_depthwise_context(
    b_history: Any,
    c_history: Any,
    x_history: Any,
    kernel: Any,
    bias: Any | None,
) -> Any:
    torch = _torch()
    batch, sequence, hidden = map(int, b_history.shape)
    fir_length = int(kernel.shape[1])
    gated = (b_history * x_history).transpose(1, 2).contiguous()
    history = torch.nn.functional.pad(gated, (fir_length - 1, 0))
    filtered = torch.nn.functional.conv1d(
        history,
        kernel.reshape(hidden, 1, fir_length),
        bias=bias,
        stride=1,
        padding=0,
        dilation=1,
        groups=hidden,
    )
    if tuple(filtered.shape) != (batch, hidden, sequence):
        raise Lfm2CpcfShortConvFoldError(
            "Depthwise FIR replay did not preserve the causal prefix length."
        )
    return c_history * filtered.transpose(1, 2)


def _history_statistics(value: Any) -> Any:
    torch = _torch()
    rows = value.reshape(-1, int(value.shape[-1]))
    return torch.stack(
        (
            rows.mean(dim=0),
            rows.square().mean(dim=0).sqrt(),
            rows.abs().mean(dim=0),
            rows.std(dim=0, unbiased=False),
            rows.amin(dim=0),
            rows.amax(dim=0),
        ),
        dim=1,
    )


def _cluster_all_source_channels(
    *,
    histories: Sequence[Mapping[str, Any]],
    parent_contexts: Sequence[Any],
    source_kernel: Any,
    target_hidden: int,
) -> tuple[
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[float, ...], ...],
    Any,
    Any,
]:
    torch = _torch()
    source_hidden = int(source_kernel.shape[0])
    if target_hidden > source_hidden:
        raise Lfm2CpcfShortConvFoldError(
            "A folding target cannot have more short-conv channels than its source."
        )
    role_values = {
        role: torch.cat(
            [value[f"parent_{role}"].reshape(-1, source_hidden) for value in histories],
            dim=0,
        )
        for role in ("b", "c", "x")
    }
    role_values["bx"] = role_values["b"] * role_values["x"]
    role_values["context"] = torch.cat(
        [value.reshape(-1, source_hidden) for value in parent_contexts],
        dim=0,
    )
    features = torch.cat(
        [_history_statistics(value) for value in role_values.values()]
        + [source_kernel],
        dim=1,
    )
    feature_mean = features.mean(dim=0, keepdim=True)
    feature_scale = features.std(dim=0, keepdim=True, unbiased=False)
    feature_scale = torch.clamp(
        feature_scale,
        min=torch.finfo(torch.float64).eps,
    )
    standardized = (features - feature_mean) / feature_scale
    importance = sum(
        value.reshape(-1, source_hidden).square().mean(dim=0)
        for value in role_values.values()
    )
    importance = importance + source_kernel.square().mean(dim=1)
    importance = torch.clamp(
        importance,
        min=torch.finfo(torch.float64).eps,
    )

    remaining = set(range(source_hidden))
    first = max(
        remaining,
        key=lambda index: (float(importance[index].item()), -index),
    )
    seeds = [first]
    remaining.remove(first)
    while len(seeds) < target_hidden:
        next_seed = max(
            remaining,
            key=lambda index: (
                min(
                    float(
                        torch.sum(
                            (standardized[index] - standardized[seed]).square()
                        ).item()
                    )
                    for seed in seeds
                ),
                float(importance[index].item()),
                -index,
            ),
        )
        seeds.append(next_seed)
        remaining.remove(next_seed)

    groups: list[list[int]] = [[seed] for seed in seeds]
    assignment = [-1] * source_hidden
    for target, source in enumerate(seeds):
        assignment[source] = target
    order = sorted(
        remaining,
        key=lambda index: (-float(importance[index].item()), index),
    )
    for source in order:
        distances = [
            float(
                torch.sum((standardized[source] - standardized[seed]).square()).item()
            )
            for seed in seeds
        ]
        target = min(
            range(target_hidden),
            key=lambda index: (
                distances[index],
                len(groups[index]),
                index,
            ),
        )
        groups[target].append(source)
        assignment[source] = target

    group_tuples = tuple(tuple(sorted(group)) for group in groups)
    weights: list[tuple[float, ...]] = []
    for group in group_tuples:
        values = importance[list(group)]
        values = values / values.sum()
        weights.append(tuple(float(value.item()) for value in values))
    if (
        any(value < 0 for value in assignment)
        or sorted(source for group in group_tuples for source in group)
        != list(range(source_hidden))
        or any(not group for group in group_tuples)
    ):
        raise Lfm2CpcfShortConvFoldError(
            "Full-source deterministic channel clustering failed."
        )
    return (
        tuple(assignment),
        group_tuples,
        tuple(weights),
        importance,
        standardized,
    )


def _fold_source_history(
    history: Any,
    groups: Sequence[Sequence[int]],
    weights: Sequence[Sequence[float]],
) -> Any:
    torch = _torch()
    folded = []
    for group, group_weights in zip(groups, weights):
        coefficient = torch.tensor(
            group_weights,
            dtype=torch.float64,
        ).reshape(1, 1, -1)
        folded.append((history[..., list(group)] * coefficient).sum(dim=-1))
    return torch.stack(folded, dim=-1)


def _fold_source_rows(
    rows: Any,
    groups: Sequence[Sequence[int]],
    weights: Sequence[Sequence[float]],
) -> Any:
    torch = _torch()
    folded = []
    for group, group_weights in zip(groups, weights):
        coefficient = torch.tensor(
            group_weights,
            dtype=torch.float64,
        ).reshape(-1, 1)
        folded.append((rows[list(group)] * coefficient).sum(dim=0))
    return torch.stack(folded, dim=0)


def _lag_design(
    b_history: Any,
    c_history: Any,
    x_history: Any,
    *,
    channel: int,
    fir_length: int,
    fit_bias: bool,
) -> Any:
    torch = _torch()
    bx = b_history[..., channel] * x_history[..., channel]
    padded = torch.nn.functional.pad(bx, (fir_length - 1, 0))
    windows = padded.unfold(dimension=1, size=fir_length, step=1)
    design = windows * c_history[..., channel].unsqueeze(-1)
    if fit_bias:
        design = torch.cat(
            (design, c_history[..., channel].unsqueeze(-1)),
            dim=-1,
        )
    return design


def _fit_target_fir(
    *,
    fit_cases: Sequence[Mapping[str, Any]],
    projected_bcx: Sequence[tuple[Any, Any, Any]],
    desired_contexts: Sequence[Any],
    prior_kernel: Any,
    prior_bias: Any | None,
    fit_bias: bool,
    ridge: float,
) -> tuple[Any, Any | None]:
    torch = _torch()
    target_hidden, fir_length = map(int, prior_kernel.shape)
    kernel_rows = []
    bias_rows = []
    for channel in range(target_hidden):
        designs = []
        targets = []
        for case, (b_value, c_value, x_value), desired in zip(
            fit_cases,
            projected_bcx,
            desired_contexts,
        ):
            design = _lag_design(
                b_value,
                c_value,
                x_value,
                channel=channel,
                fir_length=fir_length,
                fit_bias=fit_bias,
            )
            mask = case["valid_mask"].reshape(-1)
            designs.append(design.reshape(-1, int(design.shape[-1]))[mask])
            targets.append(desired[..., channel].reshape(-1, 1)[mask])
        all_design = torch.cat(designs, dim=0)
        all_target = torch.cat(targets, dim=0)
        prior_parts = [prior_kernel[channel]]
        if fit_bias:
            assert prior_bias is not None
            prior_parts.append(prior_bias[channel].reshape(1))
        prior = torch.cat(prior_parts).reshape(-1, 1)
        solution = _ridge_solution(
            all_design,
            all_target,
            ridge=ridge,
            prior=prior,
        ).reshape(-1)
        kernel_rows.append(solution[:fir_length])
        if fit_bias:
            bias_rows.append(solution[-1])
    kernel = torch.stack(kernel_rows, dim=0)
    bias = torch.stack(bias_rows) if fit_bias else None
    return kernel.contiguous(), None if bias is None else bias.contiguous()


def _normalize_cases(
    cases: Sequence[Lfm2CpcfShortConvFoldCase],
    *,
    source_hidden: int,
    target_hidden: int,
) -> list[dict[str, Any]]:
    torch = _torch()
    if not cases or not all(
        isinstance(value, Lfm2CpcfShortConvFoldCase) for value in cases
    ):
        raise Lfm2CpcfShortConvFoldError(
            "cases must contain authenticated Lfm2CpcfShortConvFoldCase instances."
        )
    if len({value.case_id for value in cases}) != len(cases):
        raise Lfm2CpcfShortConvFoldError("case_id values must be unique.")
    normalized = []
    for index, case in enumerate(cases):
        if not case.verify_authentication():
            raise Lfm2CpcfShortConvFoldError(
                f"cases[{index}] {case.case_id!r} failed authentication; its "
                "evidence changed after sealing."
            )
        prefix = f"cases[{index}]"
        target_input = _finite_float_tensor(
            case.actual_child_operator_input_history,
            f"{prefix}.actual_child_operator_input_history",
            dimensions=3,
        )
        parent_b = _finite_float_tensor(
            case.parent_b_history,
            f"{prefix}.parent_b_history",
            dimensions=3,
        )
        parent_c = _finite_float_tensor(
            case.parent_c_history,
            f"{prefix}.parent_c_history",
            dimensions=3,
        )
        parent_x = _finite_float_tensor(
            case.parent_x_history,
            f"{prefix}.parent_x_history",
            dimensions=3,
        )
        endpoint = _finite_float_tensor(
            case.parent_post_operator_residual_target_space_suffix,
            (f"{prefix}.parent_post_operator_residual_target_space_suffix"),
            dimensions=3,
        )
        actual_pre = _finite_float_tensor(
            case.actual_child_pre_operator_residual_suffix,
            f"{prefix}.actual_child_pre_operator_residual_suffix",
            dimensions=3,
        )
        batch, sequence, observed_target = map(int, target_input.shape)
        suffix = int(endpoint.shape[1])
        if (
            observed_target != target_hidden
            or tuple(parent_b.shape) != (batch, sequence, source_hidden)
            or tuple(parent_c.shape) != tuple(parent_b.shape)
            or tuple(parent_x.shape) != tuple(parent_b.shape)
            or tuple(endpoint.shape) != (batch, suffix, target_hidden)
            or tuple(actual_pre.shape) != tuple(endpoint.shape)
            or suffix <= 0
            or case.suffix_start + suffix != sequence
        ):
            raise Lfm2CpcfShortConvFoldError(
                f"{prefix} must align one complete child/parent prefix with "
                "one nonempty terminal residual suffix."
            )
        if case.valid_prefix_mask is None:
            valid_mask = torch.ones(
                (batch, sequence),
                dtype=torch.bool,
            )
        else:
            mask = case.valid_prefix_mask
            if (
                not isinstance(mask, torch.Tensor)
                or mask.layout != torch.strided
                or mask.device.type == "meta"
                or mask.ndim != 2
                or tuple(mask.shape) != (batch, sequence)
            ):
                raise Lfm2CpcfShortConvFoldError(
                    f"{prefix}.valid_prefix_mask must have shape {(batch, sequence)}."
                )
            valid_mask = mask.detach().to(device="cpu", dtype=torch.bool)
        suffix_mask = valid_mask[:, case.suffix_start :]
        if not bool(valid_mask.any().item()) or not bool(suffix_mask.any().item()):
            raise Lfm2CpcfShortConvFoldError(
                f"{prefix} has no valid full-prefix or suffix evidence rows."
            )
        normalized.append(
            {
                "case_id": case.case_id,
                "split": case.split,
                "suffix_start": case.suffix_start,
                "authentication_sha256": case.authentication_sha256,
                "target_input": target_input,
                "parent_b": parent_b,
                "parent_c": parent_c,
                "parent_x": parent_x,
                "parent_endpoint_suffix": endpoint,
                "actual_pre_suffix": actual_pre,
                "valid_mask": valid_mask.contiguous(),
                "suffix_mask": suffix_mask.contiguous(),
            }
        )
    observed_splits = {value["split"] for value in normalized}
    if not {"fit", "validation"}.issubset(observed_splits):
        raise Lfm2CpcfShortConvFoldError(
            "At least one fit case and one heldout validation case are required."
        )
    split_rank = {"selection": 0, "fit": 1, "validation": 2}
    return sorted(
        normalized,
        key=lambda value: (
            split_rank[value["split"]],
            value["case_id"],
            value["authentication_sha256"],
        ),
    )


def _split_metrics(
    cases: Sequence[Mapping[str, Any]],
    *,
    projected_bcx: Sequence[tuple[Any, Any, Any]],
    desired_bcx: Sequence[tuple[Any, Any, Any]],
    predicted_contexts: Sequence[Any],
    desired_contexts: Sequence[Any],
    out_weight: Any,
    out_bias: Any | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predicted_bcx_rows = []
    desired_bcx_rows = []
    predicted_context_rows = []
    desired_context_rows = []
    endpoint_predictions = []
    endpoint_targets = []
    case_reports = []
    for case, predicted_parts, desired_parts, context, desired_context in zip(
        cases,
        projected_bcx,
        desired_bcx,
        predicted_contexts,
        desired_contexts,
    ):
        prefix_mask = case["valid_mask"].reshape(-1)
        predicted_bcx_rows.append(
            _torch()
            .cat(predicted_parts, dim=-1)
            .reshape(-1, 3 * int(context.shape[-1]))[prefix_mask]
        )
        desired_bcx_rows.append(
            _torch()
            .cat(desired_parts, dim=-1)
            .reshape(-1, 3 * int(context.shape[-1]))[prefix_mask]
        )
        predicted_context_rows.append(
            context.reshape(-1, int(context.shape[-1]))[prefix_mask]
        )
        desired_context_rows.append(
            desired_context.reshape(-1, int(context.shape[-1]))[prefix_mask]
        )
        start = int(case["suffix_start"])
        suffix_context = context[:, start:, :]
        branch_write = _apply_affine(
            suffix_context,
            out_weight,
            out_bias,
        )
        endpoint = case["actual_pre_suffix"] + branch_write
        suffix_mask = case["suffix_mask"].reshape(-1)
        endpoint_rows = endpoint.reshape(
            -1,
            int(endpoint.shape[-1]),
        )[suffix_mask]
        target_rows = case["parent_endpoint_suffix"].reshape(
            -1,
            int(endpoint.shape[-1]),
        )[suffix_mask]
        endpoint_predictions.append(endpoint_rows)
        endpoint_targets.append(target_rows)
        case_reports.append(
            {
                "case_id": case["case_id"],
                "authentication_sha256": case["authentication_sha256"],
                "prefix_sequence_length": int(context.shape[1]),
                "suffix_start": start,
                "suffix_sequence_length": int(suffix_context.shape[1]),
                "fir_warmup_tokens_per_sequence": start,
                "output_writer_fit_or_evaluation_rows": int(suffix_mask.sum().item()),
                "endpoint": _metrics(endpoint_rows, target_rows),
            }
        )
    bcx_prediction = _torch().cat(predicted_bcx_rows, dim=0)
    bcx_target = _torch().cat(desired_bcx_rows, dim=0)
    context_prediction = _torch().cat(predicted_context_rows, dim=0)
    context_target = _torch().cat(desired_context_rows, dim=0)
    endpoint_prediction = _torch().cat(endpoint_predictions, dim=0)
    endpoint_target = _torch().cat(endpoint_targets, dim=0)
    return (
        {
            "bcx_full_prefix": _metrics(bcx_prediction, bcx_target),
            "fir_context_full_prefix": _metrics(
                context_prediction,
                context_target,
            ),
            "residual_endpoint_suffix": _metrics(
                endpoint_prediction,
                endpoint_target,
            ),
        },
        case_reports,
    )


def compile_lfm2_cpcf_short_conv_fold(
    *,
    source_fir_weight: Any,
    target_in_proj_weight_template: Any,
    target_conv_weight_template: Any,
    target_out_proj_weight_template: Any,
    cases: Sequence[Lfm2CpcfShortConvFoldCase],
    source_fir_bias: Any | None = None,
    target_in_proj_bias_template: Any | None = None,
    target_conv_bias_template: Any | None = None,
    target_out_proj_bias_template: Any | None = None,
    ridge: float = 1e-8,
    gate_policy: Lfm2CpcfShortConvGatePolicy | None = None,
) -> Lfm2CompiledCpcfShortConvFold:
    """Compile a smaller target-native short-conv from cached causal evidence.

    Fit cases alone determine channel clusters and all target tensors.
    Validation cases are strictly held out until the materialized target
    tensors are evaluated.  Residual endpoint tensors may cover only the final
    suffix; B/C/x and actual child operator inputs must cover the full prefix.
    """

    torch = _torch()
    ridge = _nonnegative_float(ridge, "ridge")
    policy = Lfm2CpcfShortConvGatePolicy() if gate_policy is None else gate_policy
    if not isinstance(policy, Lfm2CpcfShortConvGatePolicy):
        raise TypeError("gate_policy must be Lfm2CpcfShortConvGatePolicy or None.")

    source_inputs = {"source_fir_weight": source_fir_weight}
    if source_fir_bias is not None:
        source_inputs["source_fir_bias"] = source_fir_bias
    target_templates = {
        "in_proj.weight": target_in_proj_weight_template,
        "conv.weight": target_conv_weight_template,
        "out_proj.weight": target_out_proj_weight_template,
    }
    if target_in_proj_bias_template is not None:
        target_templates["in_proj.bias"] = target_in_proj_bias_template
    if target_conv_bias_template is not None:
        target_templates["conv.bias"] = target_conv_bias_template
    if target_out_proj_bias_template is not None:
        target_templates["out_proj.bias"] = target_out_proj_bias_template
    evidence_inputs: dict[str, Any] = {}
    for case in cases:
        if not isinstance(case, Lfm2CpcfShortConvFoldCase):
            continue
        prefix = f"case.{case.case_id}"
        evidence_inputs.update(
            {
                f"{prefix}.target_input": (case.actual_child_operator_input_history),
                f"{prefix}.parent_b": case.parent_b_history,
                f"{prefix}.parent_c": case.parent_c_history,
                f"{prefix}.parent_x": case.parent_x_history,
                f"{prefix}.parent_endpoint_suffix": (
                    case.parent_post_operator_residual_target_space_suffix
                ),
                f"{prefix}.actual_pre_suffix": (
                    case.actual_child_pre_operator_residual_suffix
                ),
            }
        )
        if case.valid_prefix_mask is not None:
            evidence_inputs[f"{prefix}.valid_prefix_mask"] = case.valid_prefix_mask
    source_sha_before = _tensor_bundle_sha256(source_inputs)
    template_sha_before = _tensor_bundle_sha256(target_templates)
    evidence_sha_before = _tensor_bundle_sha256(evidence_inputs)

    raw_source_kernel = _finite_float_tensor(
        source_fir_weight,
        "source_fir_weight",
        dimensions=int(getattr(source_fir_weight, "ndim", -1)),
    )
    if raw_source_kernel.ndim == 3:
        if int(raw_source_kernel.shape[1]) != 1:
            raise Lfm2CpcfShortConvFoldError(
                "source_fir_weight must be depthwise [source, 1, length]."
            )
        source_kernel = raw_source_kernel[:, 0, :]
    elif raw_source_kernel.ndim == 2:
        source_kernel = raw_source_kernel
    else:
        raise Lfm2CpcfShortConvFoldError("source_fir_weight must have rank 2 or 3.")
    source_hidden, source_fir_length = map(int, source_kernel.shape)
    source_bias = None
    if source_fir_bias is not None:
        source_bias = _finite_float_tensor(
            source_fir_bias,
            "source_fir_bias",
            dimensions=1,
        )
        if tuple(source_bias.shape) != (source_hidden,):
            raise Lfm2CpcfShortConvFoldError(
                "source_fir_bias width differs from the source FIR."
            )

    template_dtypes: dict[str, Any] = {}
    template_shapes: dict[str, tuple[int, ...]] = {}
    for name, value in target_templates.items():
        dimensions = (
            3 if name == "conv.weight" else (1 if name.endswith(".bias") else 2)
        )
        dtype, shape = _template_spec(
            value,
            f"target_{name}",
            dimensions=dimensions,
        )
        template_dtypes[name] = dtype
        template_shapes[name] = shape
    target_hidden = int(template_shapes["out_proj.weight"][0])
    if target_hidden >= source_hidden:
        raise Lfm2CpcfShortConvFoldError(
            "Target hidden width must be smaller than source hidden width."
        )
    target_fir_length = int(template_shapes["conv.weight"][2])
    expected_shapes = {
        "in_proj.weight": (3 * target_hidden, target_hidden),
        "conv.weight": (target_hidden, 1, target_fir_length),
        "out_proj.weight": (target_hidden, target_hidden),
    }
    optional_bias_shapes = {
        "in_proj.bias": (3 * target_hidden,),
        "conv.bias": (target_hidden,),
        "out_proj.bias": (target_hidden,),
    }
    for name, shape in optional_bias_shapes.items():
        if name in template_shapes:
            expected_shapes[name] = shape
    if target_fir_length <= 0 or template_shapes != expected_shapes:
        raise Lfm2CpcfShortConvFoldError(
            "Target templates do not form one canonical target-shaped LFM2 "
            "B/C/x depthwise short-convolution operator."
        )

    normalized = _normalize_cases(
        cases,
        source_hidden=source_hidden,
        target_hidden=target_hidden,
    )
    fit_cases = [value for value in normalized if value["split"] == "fit"]
    validation_cases = [value for value in normalized if value["split"] == "validation"]

    with torch.no_grad():
        parent_contexts = [
            _causal_depthwise_context(
                value["parent_b"],
                value["parent_c"],
                value["parent_x"],
                source_kernel,
                source_bias,
            )
            for value in normalized
        ]
        fit_parent_contexts = [
            parent_contexts[index]
            for index, value in enumerate(normalized)
            if value["split"] == "fit"
        ]
        (
            source_to_target,
            channel_groups,
            channel_weights,
            channel_importance,
            channel_features,
        ) = _cluster_all_source_channels(
            histories=fit_cases,
            parent_contexts=fit_parent_contexts,
            source_kernel=source_kernel,
            target_hidden=target_hidden,
        )

        desired_bcx = [
            (
                _fold_source_history(
                    value["parent_b"],
                    channel_groups,
                    channel_weights,
                ),
                _fold_source_history(
                    value["parent_c"],
                    channel_groups,
                    channel_weights,
                ),
                _fold_source_history(
                    value["parent_x"],
                    channel_groups,
                    channel_weights,
                ),
            )
            for value in normalized
        ]
        desired_contexts = [
            _fold_source_history(
                context,
                channel_groups,
                channel_weights,
            )
            for context in parent_contexts
        ]

        fit_input_rows = torch.cat(
            [
                value["target_input"].reshape(-1, target_hidden)[
                    value["valid_mask"].reshape(-1)
                ]
                for value in fit_cases
            ],
            dim=0,
        )
        fit_desired_bcx_rows = torch.cat(
            [
                torch.cat(desired_bcx[index], dim=-1).reshape(
                    -1,
                    3 * target_hidden,
                )[value["valid_mask"].reshape(-1)]
                for index, value in enumerate(normalized)
                if value["split"] == "fit"
            ],
            dim=0,
        )
        in_weight_fp64, in_bias_fp64 = _fit_affine(
            fit_input_rows,
            fit_desired_bcx_rows,
            fit_bias="in_proj.bias" in target_templates,
            ridge=ridge,
        )
        in_weight = (
            in_weight_fp64.to(dtype=template_dtypes["in_proj.weight"])
            .contiguous()
            .clone()
        )
        in_bias = None
        if in_bias_fp64 is not None:
            in_bias = (
                in_bias_fp64.to(dtype=template_dtypes["in_proj.bias"])
                .contiguous()
                .clone()
            )
        materialized_in_weight = in_weight.to(torch.float64)
        materialized_in_bias = None if in_bias is None else in_bias.to(torch.float64)
        projected_bcx = []
        for value in normalized:
            projection = _apply_affine(
                value["target_input"],
                materialized_in_weight,
                materialized_in_bias,
            )
            projected_bcx.append(tuple(projection.chunk(3, dim=-1)))

        resized_source_kernel = _resize_fir_rows(
            source_kernel,
            target_fir_length,
        )
        prior_kernel = _fold_source_rows(
            resized_source_kernel,
            channel_groups,
            channel_weights,
        )
        prior_conv_bias = None
        if "conv.bias" in target_templates:
            if source_bias is None:
                prior_conv_bias = torch.zeros(
                    (target_hidden,),
                    dtype=torch.float64,
                )
            else:
                prior_conv_bias = _fold_source_rows(
                    source_bias.reshape(-1, 1),
                    channel_groups,
                    channel_weights,
                ).reshape(-1)

        fit_projected_bcx = [
            projected_bcx[index]
            for index, value in enumerate(normalized)
            if value["split"] == "fit"
        ]
        fit_desired_contexts = [
            desired_contexts[index]
            for index, value in enumerate(normalized)
            if value["split"] == "fit"
        ]
        conv_kernel_fp64, conv_bias_fp64 = _fit_target_fir(
            fit_cases=fit_cases,
            projected_bcx=fit_projected_bcx,
            desired_contexts=fit_desired_contexts,
            prior_kernel=prior_kernel,
            prior_bias=prior_conv_bias,
            fit_bias="conv.bias" in target_templates,
            ridge=ridge,
        )
        conv_weight = (
            conv_kernel_fp64.reshape(
                target_hidden,
                1,
                target_fir_length,
            )
            .to(dtype=template_dtypes["conv.weight"])
            .contiguous()
            .clone()
        )
        conv_bias = None
        if conv_bias_fp64 is not None:
            conv_bias = (
                conv_bias_fp64.to(dtype=template_dtypes["conv.bias"])
                .contiguous()
                .clone()
            )
        materialized_kernel = conv_weight[:, 0, :].to(torch.float64)
        materialized_conv_bias = (
            None if conv_bias is None else conv_bias.to(torch.float64)
        )
        predicted_contexts = [
            _causal_depthwise_context(
                b_value,
                c_value,
                x_value,
                materialized_kernel,
                materialized_conv_bias,
            )
            for b_value, c_value, x_value in projected_bcx
        ]

        fit_context_suffix_rows = []
        fit_write_rows = []
        for index, value in enumerate(normalized):
            if value["split"] != "fit":
                continue
            start = int(value["suffix_start"])
            suffix_mask = value["suffix_mask"].reshape(-1)
            fit_context_suffix_rows.append(
                predicted_contexts[index][:, start:, :].reshape(
                    -1,
                    target_hidden,
                )[suffix_mask]
            )
            fit_write_rows.append(
                (value["parent_endpoint_suffix"] - value["actual_pre_suffix"]).reshape(
                    -1, target_hidden
                )[suffix_mask]
            )
        writer_inputs = torch.cat(fit_context_suffix_rows, dim=0)
        writer_targets = torch.cat(fit_write_rows, dim=0)
        out_weight_fp64, out_bias_fp64 = _fit_affine(
            writer_inputs,
            writer_targets,
            fit_bias="out_proj.bias" in target_templates,
            ridge=ridge,
        )
        out_weight = (
            out_weight_fp64.to(dtype=template_dtypes["out_proj.weight"])
            .contiguous()
            .clone()
        )
        out_bias = None
        if out_bias_fp64 is not None:
            out_bias = (
                out_bias_fp64.to(dtype=template_dtypes["out_proj.bias"])
                .contiguous()
                .clone()
            )
        materialized_out_weight = out_weight.to(torch.float64)
        materialized_out_bias = None if out_bias is None else out_bias.to(torch.float64)

        fit_indices = [
            index for index, value in enumerate(normalized) if value["split"] == "fit"
        ]
        validation_indices = [
            index
            for index, value in enumerate(normalized)
            if value["split"] == "validation"
        ]
        fit_metrics, fit_case_reports = _split_metrics(
            fit_cases,
            projected_bcx=[projected_bcx[index] for index in fit_indices],
            desired_bcx=[desired_bcx[index] for index in fit_indices],
            predicted_contexts=[predicted_contexts[index] for index in fit_indices],
            desired_contexts=[desired_contexts[index] for index in fit_indices],
            out_weight=materialized_out_weight,
            out_bias=materialized_out_bias,
        )
        validation_metrics, validation_case_reports = _split_metrics(
            validation_cases,
            projected_bcx=[projected_bcx[index] for index in validation_indices],
            desired_bcx=[desired_bcx[index] for index in validation_indices],
            predicted_contexts=[
                predicted_contexts[index] for index in validation_indices
            ],
            desired_contexts=[desired_contexts[index] for index in validation_indices],
            out_weight=materialized_out_weight,
            out_bias=materialized_out_bias,
        )

    source_sha_after = _tensor_bundle_sha256(source_inputs)
    template_sha_after = _tensor_bundle_sha256(target_templates)
    evidence_sha_after = _tensor_bundle_sha256(evidence_inputs)
    source_unchanged = source_sha_before == source_sha_after
    templates_unchanged = template_sha_before == template_sha_after
    evidence_unchanged = evidence_sha_before == evidence_sha_after
    if not source_unchanged or not templates_unchanged or not evidence_unchanged:
        raise Lfm2CpcfShortConvFoldError(
            "Compiler input tensors changed during closed-form folding."
        )

    compiled_state = {
        "in_proj.weight": in_weight,
        "conv.weight": conv_weight,
        "out_proj.weight": out_weight,
    }
    if in_bias is not None:
        compiled_state["in_proj.bias"] = in_bias
    if conv_bias is not None:
        compiled_state["conv.bias"] = conv_bias
    if out_bias is not None:
        compiled_state["out_proj.bias"] = out_bias
    if any(
        tuple(compiled_state[name].shape) != template_shapes[name]
        for name in compiled_state
    ):
        raise Lfm2CpcfShortConvFoldError(
            "Materialized CPCF short-conv tensors differ from target shapes."
        )
    coverage = sorted(source for group in channel_groups for source in group)
    if coverage != list(range(source_hidden)):
        raise Lfm2CpcfShortConvFoldError(
            "Compiled channel provenance does not cover every source channel."
        )

    fit_endpoint = fit_metrics["residual_endpoint_suffix"]
    validation_endpoint = validation_metrics["residual_endpoint_suffix"]
    gate_checks = {
        "fit_endpoint_relative_rmse": (
            float(fit_endpoint["relative_rmse"])
            <= policy.maximum_fit_endpoint_relative_rmse
        ),
        "validation_endpoint_relative_rmse": (
            float(validation_endpoint["relative_rmse"])
            <= policy.maximum_validation_endpoint_relative_rmse
        ),
        "fit_endpoint_cosine_similarity": (
            float(fit_endpoint["cosine_similarity"])
            >= policy.minimum_fit_endpoint_cosine_similarity
        ),
        "validation_endpoint_cosine_similarity": (
            float(validation_endpoint["cosine_similarity"])
            >= policy.minimum_validation_endpoint_cosine_similarity
        ),
    }
    gate_passed = all(gate_checks.values())
    gate_report = {
        "policy": policy.as_dict(),
        "checks": gate_checks,
        "passed": gate_passed,
        "heldout_validation_used_for_fitting": False,
    }

    provenance_body = {
        "source_hidden_size": source_hidden,
        "target_hidden_size": target_hidden,
        "source_to_target": list(source_to_target),
        "channel_groups": [list(group) for group in channel_groups],
        "channel_barycentric_weights": [list(weights) for weights in channel_weights],
        "source_channel_importance": [
            float(value.item()) for value in channel_importance
        ],
        "channel_feature_sha256": _tensor_sha256(
            channel_features,
            "channel_features",
        ),
        "fit_case_authentication_sha256": [
            value["authentication_sha256"] for value in fit_cases
        ],
        "validation_case_authentication_sha256": [
            value["authentication_sha256"] for value in validation_cases
        ],
        "source_channels_selected": 0,
        "source_channels_dropped": 0,
        "source_channels_pruned": 0,
        "full_source_coverage": True,
    }
    target_logical_parameters = sum(
        int(value.numel()) for value in compiled_state.values()
    )
    body = {
        "schema": LFM2_CPCF_SHORT_CONV_FOLD_SCHEMA,
        "source_hidden_size": source_hidden,
        "target_hidden_size": target_hidden,
        "hidden_width_compression_ratio": source_hidden / target_hidden,
        "source_fir_length": source_fir_length,
        "target_fir_length": target_fir_length,
        "target_logical_parameters": target_logical_parameters,
        "fit_case_count": len(fit_cases),
        "validation_case_count": len(validation_cases),
        "fit_metrics": fit_metrics,
        "validation_metrics": validation_metrics,
        "fit_case_reports": fit_case_reports,
        "validation_case_reports": validation_case_reports,
        "output_writer_scope": "authenticated_residual_suffix_only",
        "producer_and_fir_scope": "complete_causal_prefix",
        "fir_state_policy": (
            "zero_at_sequence_origin_then_warm_over_every_prefix_token"
        ),
        "writer_fit_rows": int(writer_inputs.shape[0]),
        "fit_prefix_rows": int(fit_input_rows.shape[0]),
        "fit_fir_warmup_rows": sum(
            int(value["target_input"].shape[0]) * int(value["suffix_start"])
            for value in fit_cases
        ),
        "provenance": {
            **provenance_body,
            "provenance_sha256": _sha256(provenance_body),
        },
        "behavior_gate": gate_report,
        "audit": {
            "source_inputs_sha256_before": source_sha_before,
            "source_inputs_sha256_after": source_sha_after,
            "source_inputs_unchanged": source_unchanged,
            "target_templates_sha256_before": template_sha_before,
            "target_templates_sha256_after": template_sha_after,
            "target_templates_unchanged": templates_unchanged,
            "evidence_sha256_before": evidence_sha_before,
            "evidence_sha256_after": evidence_sha_after,
            "evidence_unchanged": evidence_unchanged,
            "case_authentication_verified": True,
            "blank_target_values_used_by_compiler": False,
            "target_templates_used_for_shape_and_dtype_only": True,
            "returned_tensors_detached": all(
                not value.requires_grad for value in compiled_state.values()
            ),
        },
        "execution": {
            "closed_form_linear_solve": True,
            "deterministic_barycentric_channel_clustering": True,
            "parent_model_forward_calls": 0,
            "target_model_forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
        },
        "compiled_state_sha256": _tensor_bundle_sha256(compiled_state),
    }
    report = {**body, "report_sha256": _sha256(body)}
    if policy.enforce and not gate_passed:
        failed = sorted(name for name, passed in gate_checks.items() if not passed)
        raise Lfm2CpcfShortConvValidationError(
            "Compiled CPCF short-conv failed enforced behavior gates: "
            + ", ".join(failed),
            report,
        )
    return Lfm2CompiledCpcfShortConvFold(
        in_proj_weight=in_weight,
        in_proj_bias=in_bias,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        out_proj_weight=out_weight,
        out_proj_bias=out_bias,
        source_to_target=source_to_target,
        channel_groups=channel_groups,
        channel_barycentric_weights=channel_weights,
        source_hidden_size=source_hidden,
        target_hidden_size=target_hidden,
        source_fir_length=source_fir_length,
        target_fir_length=target_fir_length,
        fit_metrics=fit_metrics,
        validation_metrics=validation_metrics,
        report=report,
    )


__all__ = [
    "LFM2_CPCF_SHORT_CONV_FOLD_SCHEMA",
    "Lfm2CompiledCpcfShortConvFold",
    "Lfm2CpcfShortConvFoldCase",
    "Lfm2CpcfShortConvFoldError",
    "Lfm2CpcfShortConvGatePolicy",
    "Lfm2CpcfShortConvValidationError",
    "compile_lfm2_cpcf_short_conv_fold",
]
