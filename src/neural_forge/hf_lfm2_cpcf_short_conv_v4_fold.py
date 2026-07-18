"""Production orchestration for residual-endpoint LFM2 short-conv v4 folding.

The numerical v4 module deliberately exposes only small, auditable primitives.
This module composes them into one target-shaped fold:

* fit an affine lift from the actual child operator input to the cached parent
  operator input;
* choose signed source channels that are both predictable and visible at the
  residual endpoint;
* transport the live parent B/C/x, FIR, and writer tensors as structural
  priors;
* run a bounded, fit-only coupled CP/Volterra factor refinement for every
  authenticated ridge candidate;
* choose the candidate on selection trajectories only; and
* evaluate the frozen, target-dtype state on fit and heldout validation.

There is no model forward call, optimizer, backward pass, training, pruning,
quantization, or parent-backed fallback in this module.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.hf_lfm2_cpcf_short_conv_fold import (
    Lfm2CpcfShortConvGatePolicy,
    Lfm2CpcfShortConvValidationError,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (
    LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
    Lfm2CpcfShortConvV4SolverPolicy,
    Lfm2CpcfV4BasisCase,
    Lfm2CpcfV4ProjectionCase,
    Lfm2CpcfV4RidgeCase,
    affine_penalty_mask,
    build_signed_endpoint_aware_channel_basis,
    case_balanced_trace_scaled_prior_ridge,
    causal_depthwise_context,
    evaluate_candidate_state_target_dtype,
    rank_short_conv_v4_selection_candidates,
    run_fixed_sampled_cp_variable_projection,
    transport_parent_short_conv_priors,
)


LFM2_CPCF_SHORT_CONV_V4_FOLD_SCHEMA = "neural-forge-lfm2-cpcf-short-conv-fold-v4"
LFM2_CPCF_SHORT_CONV_V4_FAILURE_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-fold-v4-failure-v1"
)


class Lfm2CpcfShortConvV4FoldError(ValueError):
    """The v4 production fold request or compiled artifact is invalid."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for LFM2 short-conv v4 folding."
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
        raise Lfm2CpcfShortConvV4FoldError(
            "V4 fold evidence must be finite canonical JSON."
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _canonical_copy(value: Any, name: str) -> dict[str, Any]:
    try:
        result = json.loads(_canonical_json(value).decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CpcfShortConvV4FoldError(f"{name} is not canonical JSON.") from exc
    if not isinstance(result, dict):
        raise Lfm2CpcfShortConvV4FoldError(f"{name} must be a mapping.")
    return result


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


def _dtype_name(value: Any) -> str:
    return str(value.dtype).removeprefix("torch.")


def _tensor_sha256(value: Any, name: str) -> str:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2CpcfShortConvV4FoldError(
            f"{name} must be a materialized dense tensor."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": _dtype_name(tensor),
        "shape": [int(size) for size in tensor.shape],
        "layout": str(tensor.layout),
    }
    digest = hashlib.sha256(_canonical_json(descriptor))
    digest.update(
        tensor.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes(order="C")
    )
    return digest.hexdigest()


def _snapshot_float(value: Any, name: str, *, dimensions: int) -> Any:
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
        raise Lfm2CpcfShortConvV4FoldError(
            f"{name} must be a nonempty dense real rank-{dimensions} tensor."
        )
    result = value.detach().to(device="cpu").contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CpcfShortConvV4FoldError(f"{name} contains NaN or infinity.")
    return result


def _snapshot_mask(value: Any, name: str) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 2
        or value.numel() == 0
    ):
        raise Lfm2CpcfShortConvV4FoldError(
            f"{name} must be a nonempty dense rank-2 mask."
        )
    return value.detach().to(device="cpu", dtype=torch.bool).contiguous().clone()


def _case_body(
    *,
    case_id: str,
    split: str,
    suffix_start: int,
    tensors: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": LFM2_CPCF_SHORT_CONV_V4_FOLD_SCHEMA,
        "kind": "authenticated_causal_fold_case",
        "case_id": case_id,
        "split": split,
        "suffix_start": suffix_start,
        "tensors": {
            name: _tensor_sha256(value, f"{case_id}.{name}")
            for name, value in sorted(tensors.items())
        },
    }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfShortConvV4FoldCase:
    """One complete target-aligned causal trajectory."""

    case_id: str
    split: str
    suffix_start: int
    actual_child_operator_input_history: Any
    parent_operator_input_history: Any
    parent_b_history: Any
    parent_c_history: Any
    parent_x_history: Any
    parent_post_operator_residual_target_space_suffix: Any
    actual_child_pre_operator_residual_suffix: Any
    valid_prefix_mask: Any
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        case_id = str(self.case_id).strip()
        split = str(self.split).strip().lower()
        if not case_id:
            raise Lfm2CpcfShortConvV4FoldError("case_id must be nonempty text.")
        if split not in {"fit", "selection", "validation"}:
            raise Lfm2CpcfShortConvV4FoldError(
                "split must be fit, selection, or validation."
            )
        if (
            isinstance(self.suffix_start, bool)
            or not isinstance(self.suffix_start, int)
            or self.suffix_start < 0
        ):
            raise Lfm2CpcfShortConvV4FoldError(
                "suffix_start must be a nonnegative integer."
            )
        tensor_names = (
            "actual_child_operator_input_history",
            "parent_operator_input_history",
            "parent_b_history",
            "parent_c_history",
            "parent_x_history",
            "parent_post_operator_residual_target_space_suffix",
            "actual_child_pre_operator_residual_suffix",
        )
        values = {
            name: _snapshot_float(getattr(self, name), name, dimensions=3)
            for name in tensor_names
        }
        mask = _snapshot_mask(self.valid_prefix_mask, "valid_prefix_mask")
        child = values["actual_child_operator_input_history"]
        parent_input = values["parent_operator_input_history"]
        b_value = values["parent_b_history"]
        c_value = values["parent_c_history"]
        x_value = values["parent_x_history"]
        endpoint = values["parent_post_operator_residual_target_space_suffix"]
        actual_pre = values["actual_child_pre_operator_residual_suffix"]
        prefix_shape = tuple(int(size) for size in child.shape[:2])
        if (
            tuple(int(size) for size in parent_input.shape[:2]) != prefix_shape
            or tuple(int(size) for size in b_value.shape[:2]) != prefix_shape
            or tuple(b_value.shape) != tuple(c_value.shape)
            or tuple(b_value.shape) != tuple(x_value.shape)
            or tuple(mask.shape) != prefix_shape
            or self.suffix_start >= int(child.shape[1])
            or tuple(endpoint.shape) != tuple(actual_pre.shape)
            or tuple(int(size) for size in endpoint.shape[:2])
            != (
                int(child.shape[0]),
                int(child.shape[1]) - self.suffix_start,
            )
            or int(endpoint.shape[-1]) != int(child.shape[-1])
        ):
            raise Lfm2CpcfShortConvV4FoldError(
                "V4 fold case tensors have incompatible causal geometry."
            )
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "split", split)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "valid_prefix_mask", mask)
        body = _case_body(
            case_id=case_id,
            split=split,
            suffix_start=self.suffix_start,
            tensors={**values, "valid_prefix_mask": mask},
        )
        expected = _sha(body)
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2CpcfShortConvV4FoldError(
                "V4 fold case authentication is invalid."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def verify_authentication(self) -> bool:
        tensors = {
            name: getattr(self, name)
            for name in (
                "actual_child_operator_input_history",
                "parent_operator_input_history",
                "parent_b_history",
                "parent_c_history",
                "parent_x_history",
                "parent_post_operator_residual_target_space_suffix",
                "actual_child_pre_operator_residual_suffix",
            )
        }
        tensors["valid_prefix_mask"] = self.valid_prefix_mask
        return self.authentication_sha256 == _sha(
            _case_body(
                case_id=self.case_id,
                split=self.split,
                suffix_start=self.suffix_start,
                tensors=tensors,
            )
        )


def _ordered_cases(
    cases: Sequence[Lfm2CpcfShortConvV4FoldCase],
) -> tuple[Lfm2CpcfShortConvV4FoldCase, ...]:
    values = tuple(cases)
    if (
        not values
        or any(not isinstance(value, Lfm2CpcfShortConvV4FoldCase) for value in values)
        or len({value.case_id for value in values}) != len(values)
    ):
        raise Lfm2CpcfShortConvV4FoldError(
            "cases must contain unique authenticated v4 fold cases."
        )
    ordered = tuple(sorted(values, key=lambda value: value.case_id))
    if any(not value.verify_authentication() for value in ordered):
        raise Lfm2CpcfShortConvV4FoldError(
            "A v4 fold case changed after authentication."
        )
    if {value.split for value in ordered} != {
        "fit",
        "selection",
        "validation",
    }:
        raise Lfm2CpcfShortConvV4FoldError(
            "V4 folding requires disjoint fit, selection, and validation cases."
        )
    return ordered


def _source_kernel(value: Any) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor) or value.ndim not in {2, 3}:
        raise Lfm2CpcfShortConvV4FoldError(
            "Source FIR must be rank-2 or depthwise rank-3."
        )
    result = _snapshot_float(
        value,
        "source_fir_weight",
        dimensions=value.ndim,
    )
    if result.ndim == 3:
        if int(result.shape[1]) != 1:
            raise Lfm2CpcfShortConvV4FoldError("Source FIR must be depthwise [S,1,L].")
        result = result[:, 0, :]
    if result.ndim != 2:
        raise Lfm2CpcfShortConvV4FoldError(
            "Source FIR must be rank-2 or depthwise rank-3."
        )
    return result.contiguous()


def _template(
    value: Any,
    name: str,
    *,
    dimensions: int,
) -> tuple[Any, tuple[int, ...]]:
    tensor = _snapshot_float(value, name, dimensions=dimensions)
    return value.dtype, tuple(int(size) for size in tensor.shape)


def _bias_template(
    value: Any | None,
    name: str,
    *,
    width: int,
) -> Any | None:
    if value is None:
        return None
    result = _snapshot_float(value, name, dimensions=1)
    if tuple(result.shape) != (width,):
        raise Lfm2CpcfShortConvV4FoldError(
            f"{name} width differs from its target module."
        )
    return result


def _materialize_bias(
    value: Any | None,
    template: Any | None,
    name: str,
) -> Any | None:
    torch = _torch()
    if template is None:
        if value is not None and bool((value.to(torch.float64) != 0.0).any().item()):
            raise Lfm2CpcfShortConvV4FoldError(
                f"{name} requires a target bias that does not exist."
            )
        return None
    if value is None:
        return torch.zeros_like(template, device="cpu").contiguous()
    result = value.detach().to(device="cpu", dtype=template.dtype).contiguous()
    if tuple(result.shape) != tuple(template.shape):
        raise Lfm2CpcfShortConvV4FoldError(
            f"{name} differs from the target bias shape."
        )
    return result


def _input_lift(
    cases: Sequence[Lfm2CpcfShortConvV4FoldCase],
    *,
    prior_weight: Any,
    policy: Lfm2CpcfShortConvV4SolverPolicy,
) -> tuple[Any, Any, Mapping[str, Any]]:
    torch = _torch()
    ridge_cases = []
    for case in cases:
        if case.split != "fit":
            continue
        child = case.actual_child_operator_input_history.to(torch.float64)
        parent = case.parent_operator_input_history.to(torch.float64)
        mask = case.valid_prefix_mask.reshape(-1)
        design = child.reshape(-1, int(child.shape[-1]))
        design = torch.cat(
            (
                design,
                torch.ones(
                    (int(design.shape[0]), 1),
                    dtype=torch.float64,
                ),
            ),
            dim=1,
        )
        ridge_cases.append(
            Lfm2CpcfV4RidgeCase(
                case_id=case.case_id,
                split="fit",
                design=design,
                target=parent.reshape(-1, int(parent.shape[-1])),
                valid_rows=mask,
            )
        )
    prior = prior_weight.detach().to(device="cpu", dtype=torch.float64)
    prior_solution = torch.cat(
        (
            prior,
            torch.zeros((1, int(prior.shape[1])), dtype=torch.float64),
        ),
        dim=0,
    )
    result = case_balanced_trace_scaled_prior_ridge(
        ridge_cases,
        prior=prior_solution,
        alpha=policy.predictor_ridge_alpha,
        penalty_mask=affine_penalty_mask(
            int(prior.shape[0]),
            fit_bias=True,
        ),
        required_split="fit",
        rank_relative_tolerance=policy.rank_relative_tolerance,
    )
    weight = result.solution[:-1, :].contiguous()
    bias = result.solution[-1, :].contiguous()
    body = {
        "schema": LFM2_CPCF_SHORT_CONV_V4_FOLD_SCHEMA,
        "kind": "actual_child_to_parent_operator_input_affine_lift",
        "fit_only": True,
        "selection_cases_consumed": 0,
        "validation_cases_consumed": 0,
        "ridge_report_sha256": result.diagnostics["report_sha256"],
        "weight_sha256": _tensor_sha256(weight, "input_lift.weight"),
        "bias_sha256": _tensor_sha256(bias, "input_lift.bias"),
        "affine_bias_transported": True,
    }
    return (
        weight,
        bias,
        MappingProxyType({**body, "report_sha256": _sha(body)}),
    )


def _projection_cases(
    cases: Sequence[Lfm2CpcfShortConvV4FoldCase],
    *,
    parent_contexts: Mapping[str, Any],
    encoder: Any,
) -> tuple[Lfm2CpcfV4ProjectionCase, ...]:
    result = []
    for case in cases:
        context = parent_contexts[case.case_id] @ encoder
        desired_write = case.parent_post_operator_residual_target_space_suffix.to(
            context.dtype
        ) - case.actual_child_pre_operator_residual_suffix.to(context.dtype)
        result.append(
            Lfm2CpcfV4ProjectionCase(
                case_id=case.case_id,
                split=case.split,
                child_input_history=(case.actual_child_operator_input_history),
                desired_context_history=context,
                desired_write_suffix=desired_write,
                suffix_start=case.suffix_start,
                valid_prefix_mask=case.valid_prefix_mask,
                valid_suffix_mask=case.valid_prefix_mask[:, case.suffix_start :],
            )
        )
    return tuple(result)


def _state_sha256(values: Mapping[str, Any]) -> str:
    return _sha(
        {
            name: _tensor_sha256(value, name)
            for name, value in sorted(values.items())
            if value is not None
        }
    )


def _gate_report(
    *,
    policy: Lfm2CpcfShortConvGatePolicy,
    fit: Mapping[str, Any],
    validation: Mapping[str, Any],
    full_source_coverage: bool,
) -> dict[str, Any]:
    checks = {
        "fit_endpoint_relative_rmse": (
            float(fit["maximum_case_relative_rmse"])
            <= policy.maximum_fit_endpoint_relative_rmse
        ),
        "validation_endpoint_relative_rmse": (
            float(validation["maximum_case_relative_rmse"])
            <= policy.maximum_validation_endpoint_relative_rmse
        ),
        "fit_endpoint_cosine_similarity": (
            float(fit["minimum_case_cosine_similarity"])
            >= policy.minimum_fit_endpoint_cosine_similarity
        ),
        "validation_endpoint_cosine_similarity": (
            float(validation["minimum_case_cosine_similarity"])
            >= policy.minimum_validation_endpoint_cosine_similarity
        ),
        "fit_context_relative_rmse": (
            fit.get("context_evidence_present") is True
            and float(fit["maximum_case_context_relative_rmse"])
            <= policy.maximum_fit_endpoint_relative_rmse
        ),
        "validation_context_relative_rmse": (
            validation.get("context_evidence_present") is True
            and float(validation["maximum_case_context_relative_rmse"])
            <= policy.maximum_validation_endpoint_relative_rmse
        ),
        "fit_context_cosine_similarity": (
            fit.get("context_evidence_present") is True
            and float(fit["minimum_case_context_cosine_similarity"])
            >= policy.minimum_fit_endpoint_cosine_similarity
        ),
        "validation_context_cosine_similarity": (
            validation.get("context_evidence_present") is True
            and float(validation["minimum_case_context_cosine_similarity"])
            >= policy.minimum_validation_endpoint_cosine_similarity
        ),
        "full_source_numeric_participation": full_source_coverage,
        "actual_fit_target_dtype_replay": (
            fit["actual_parameter_and_intermediate_target_dtype_replay"] is True
        ),
        "actual_validation_target_dtype_replay": (
            validation["actual_parameter_and_intermediate_target_dtype_replay"] is True
        ),
    }
    return {
        "policy": policy.as_dict(),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "passed": all(checks.values()),
        "enforced": policy.enforce,
    }


@dataclass(frozen=True, slots=True)
class Lfm2CompiledCpcfShortConvV4Fold:
    """One materialized, target-shaped residual-endpoint v4 operator."""

    in_proj_weight: Any
    in_proj_bias: Any | None
    conv_weight: Any
    conv_bias: Any | None
    out_proj_weight: Any
    out_proj_bias: Any | None
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name, dimensions in (
            ("in_proj_weight", 2),
            ("conv_weight", 3),
            ("out_proj_weight", 2),
        ):
            object.__setattr__(
                self,
                name,
                _snapshot_float(
                    getattr(self, name),
                    name,
                    dimensions=dimensions,
                ),
            )
        for name in ("in_proj_bias", "conv_bias", "out_proj_bias"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _snapshot_float(value, name, dimensions=1),
                )
        report = _canonical_copy(self.report, "compiled v4 report")
        body = dict(report)
        seal = body.pop("report_sha256", None)
        behavior_gate = report.get("behavior_gate", {})
        gate_admissible = behavior_gate.get("passed") is True or (
            behavior_gate.get("passed") is False
            and behavior_gate.get("enforced") is False
        )
        if (
            seal != _sha(body)
            or report.get("schema") != LFM2_CPCF_SHORT_CONV_V4_FOLD_SCHEMA
            or report.get("compiled_state_sha256") != _state_sha256(self.state_dict())
            or report.get("compiled_short_conv_fold_issued") is not True
            or report.get("promotion_claimed") is not False
            or report.get("promotable_candidate")
            is not (behavior_gate.get("passed") is True)
            or not gate_admissible
        ):
            raise Lfm2CpcfShortConvV4FoldError(
                "Compiled v4 short-conv artifact is not authenticated."
            )
        object.__setattr__(
            self,
            "report",
            _deep_freeze(report),
        )

    @property
    def behavior_gate_passed(self) -> bool:
        return bool(self.report["behavior_gate"]["passed"])

    @property
    def full_source_coverage(self) -> bool:
        return bool(
            self.report["behavior_gate"]["checks"]["full_source_numeric_participation"]
        )

    def state_dict(self) -> Mapping[str, Any]:
        values = {
            "in_proj.weight": self.in_proj_weight.clone(),
            "conv.weight": self.conv_weight.clone(),
            "out_proj.weight": self.out_proj_weight.clone(),
        }
        if self.in_proj_bias is not None:
            values["in_proj.bias"] = self.in_proj_bias.clone()
        if self.conv_bias is not None:
            values["conv.bias"] = self.conv_bias.clone()
        if self.out_proj_bias is not None:
            values["out_proj.bias"] = self.out_proj_bias.clone()
        return MappingProxyType(values)

    def verify_authentication(self) -> bool:
        report = _deep_thaw(self.report)
        body = dict(report)
        supplied = body.pop("report_sha256", None)
        behavior_gate = report.get("behavior_gate", {})
        gate_admissible = behavior_gate.get("passed") is True or (
            behavior_gate.get("passed") is False
            and behavior_gate.get("enforced") is False
        )
        return bool(
            isinstance(supplied, str)
            and supplied == _sha(body)
            and report.get("compiled_state_sha256") == _state_sha256(self.state_dict())
            and report.get("compiled_short_conv_fold_issued") is True
            and report.get("promotion_claimed") is False
            and report.get("promotable_candidate")
            is (behavior_gate.get("passed") is True)
            and gate_admissible
        )

    def as_dict(self) -> dict[str, Any]:
        if not self.verify_authentication():
            raise Lfm2CpcfShortConvV4FoldError(
                "Compiled v4 artifact changed after authentication."
            )
        return _deep_thaw(self.report)


def compile_lfm2_cpcf_short_conv_v4_fold(
    *,
    source_in_proj_weight: Any,
    source_in_proj_bias: Any | None,
    source_fir_weight: Any,
    source_fir_bias: Any | None,
    source_out_proj_weight: Any,
    source_out_proj_bias: Any | None,
    target_in_proj_weight_template: Any,
    target_in_proj_bias_template: Any | None,
    target_conv_weight_template: Any,
    target_conv_bias_template: Any | None,
    target_out_proj_weight_template: Any,
    target_out_proj_bias_template: Any | None,
    output_encoder: Any,
    cases: Sequence[Lfm2CpcfShortConvV4FoldCase],
    solver_policy: Lfm2CpcfShortConvV4SolverPolicy,
    gate_policy: Lfm2CpcfShortConvGatePolicy,
) -> Lfm2CompiledCpcfShortConvV4Fold:
    """Compile one complete v4 operator without executing either model."""

    torch = _torch()
    if not isinstance(solver_policy, Lfm2CpcfShortConvV4SolverPolicy):
        raise TypeError("solver_policy must be Lfm2CpcfShortConvV4SolverPolicy.")
    if not solver_policy.verify_authentication():
        raise Lfm2CpcfShortConvV4FoldError("V4 solver policy authentication failed.")
    if not isinstance(gate_policy, Lfm2CpcfShortConvGatePolicy):
        raise TypeError("gate_policy must be Lfm2CpcfShortConvGatePolicy.")
    ordered = _ordered_cases(cases)
    target_dtype, in_shape = _template(
        target_in_proj_weight_template,
        "target_in_proj_weight_template",
        dimensions=2,
    )
    conv_dtype, conv_shape = _template(
        target_conv_weight_template,
        "target_conv_weight_template",
        dimensions=3,
    )
    out_dtype, out_shape = _template(
        target_out_proj_weight_template,
        "target_out_proj_weight_template",
        dimensions=2,
    )
    target_hidden = int(in_shape[1])
    if (
        in_shape != (3 * target_hidden, target_hidden)
        or conv_shape[:2] != (target_hidden, 1)
        or out_shape != (target_hidden, target_hidden)
        or not (
            target_dtype
            == conv_dtype
            == out_dtype
            == getattr(torch, solver_policy.target_dtype)
        )
    ):
        raise Lfm2CpcfShortConvV4FoldError(
            "Target templates differ from the authenticated v4 dtype/topology."
        )
    in_bias_template = _bias_template(
        target_in_proj_bias_template,
        "target_in_proj_bias_template",
        width=3 * target_hidden,
    )
    conv_bias_template = _bias_template(
        target_conv_bias_template,
        "target_conv_bias_template",
        width=target_hidden,
    )
    out_bias_template = _bias_template(
        target_out_proj_bias_template,
        "target_out_proj_bias_template",
        width=target_hidden,
    )
    source_kernel = _source_kernel(source_fir_weight)
    source_hidden = int(source_kernel.shape[0])
    parent_contexts = {
        case.case_id: causal_depthwise_context(
            case.parent_b_history,
            case.parent_c_history,
            case.parent_x_history,
            source_kernel,
            source_fir_bias,
        )
        for case in ordered
    }
    basis = build_signed_endpoint_aware_channel_basis(
        tuple(
            Lfm2CpcfV4BasisCase(
                case_id=case.case_id,
                split="fit",
                child_inputs=case.actual_child_operator_input_history[
                    :, case.suffix_start :
                ],
                parent_contexts=parent_contexts[case.case_id][:, case.suffix_start :],
                endpoint_writes=(
                    case.parent_post_operator_residual_target_space_suffix
                    - case.actual_child_pre_operator_residual_suffix
                ),
                valid_rows=case.valid_prefix_mask[:, case.suffix_start :],
            )
            for case in ordered
            if case.split == "fit"
        ),
        target_channels=target_hidden,
        policy=solver_policy,
    )
    lift_weight, lift_bias, lift_report = _input_lift(
        ordered,
        prior_weight=basis.decoder,
        policy=solver_policy,
    )
    priors = transport_parent_short_conv_priors(
        basis=basis,
        source_in_proj_weight=source_in_proj_weight,
        source_in_proj_bias=source_in_proj_bias,
        source_fir_kernel=source_kernel,
        source_fir_bias=source_fir_bias,
        source_out_proj_weight=source_out_proj_weight,
        source_out_proj_bias=source_out_proj_bias,
        target_fir_length=int(conv_shape[2]),
        input_lift=lift_weight,
        input_lift_bias=lift_bias,
        output_encoder=output_encoder,
    )
    projections = _projection_cases(
        ordered,
        parent_contexts=parent_contexts,
        encoder=basis.encoder,
    )
    split_cases = {
        split: tuple(case for case in projections if case.split == split)
        for split in ("fit", "selection", "validation")
    }
    candidate_results = {}
    selection_reports = {}
    initial_in_bias = (
        None
        if in_bias_template is None
        else (
            priors.in_proj_bias
            if priors.in_proj_bias is not None
            else torch.zeros(
                (3 * target_hidden,),
                dtype=torch.float64,
            )
        )
    )
    initial_fir_bias = (
        None
        if conv_bias_template is None
        else (
            priors.fir_bias
            if priors.fir_bias is not None
            else torch.zeros((target_hidden,), dtype=torch.float64)
        )
    )
    initial_out_bias = (
        None
        if out_bias_template is None
        else (
            priors.out_proj_bias
            if priors.out_proj_bias is not None
            else torch.zeros((target_hidden,), dtype=torch.float64)
        )
    )
    for index, alpha in enumerate(solver_policy.ridge_alphas):
        candidate_id = f"ridge-alpha-{index:02d}-{alpha:.17g}"
        result = run_fixed_sampled_cp_variable_projection(
            cases=split_cases["fit"],
            initial_in_proj_weight=priors.in_proj_weight,
            initial_in_proj_bias=initial_in_bias,
            initial_fir_kernel=priors.fir_kernel,
            initial_fir_bias=initial_fir_bias,
            initial_writer_weight=priors.out_proj_weight,
            initial_writer_bias=initial_out_bias,
            alpha=alpha,
            policy=solver_policy,
        )
        candidate_results[candidate_id] = result
        selection_reports[candidate_id] = evaluate_candidate_state_target_dtype(
            cases=split_cases["selection"],
            in_proj_weight=result.in_proj_weight,
            in_proj_bias=result.in_proj_bias,
            fir_kernel=result.fir_kernel,
            fir_bias=result.fir_bias,
            writer_weight=result.writer_weight,
            writer_bias=result.writer_bias,
            required_split="selection",
            policy=solver_policy,
        )
    ranking = rank_short_conv_v4_selection_candidates(
        selection_reports,
        policy_sha256=solver_policy.authentication_sha256,
    )
    selected_id = str(ranking["selected_candidate_id"])
    selected = candidate_results[selected_id]
    in_weight = selected.in_proj_weight.to(dtype=target_dtype).contiguous()
    fir_kernel = selected.fir_kernel.to(dtype=target_dtype).contiguous()
    out_weight = selected.writer_weight.to(dtype=target_dtype).contiguous()
    in_bias = _materialize_bias(
        selected.in_proj_bias,
        in_bias_template,
        "in_proj_bias",
    )
    fir_bias = _materialize_bias(
        selected.fir_bias,
        conv_bias_template,
        "conv_bias",
    )
    out_bias = _materialize_bias(
        selected.writer_bias,
        out_bias_template,
        "out_proj_bias",
    )
    evaluations = {
        "selection": selection_reports[selected_id],
        **{
            split: evaluate_candidate_state_target_dtype(
                cases=split_cases[split],
                in_proj_weight=in_weight,
                in_proj_bias=in_bias,
                fir_kernel=fir_kernel,
                fir_bias=fir_bias,
                writer_weight=out_weight,
                writer_bias=out_bias,
                required_split=split,
                policy=solver_policy,
            )
            for split in ("fit", "validation")
        },
    }
    selected_state_sha = selection_reports[selected_id]["candidate_state_sha256"]
    if evaluations["selection"]["candidate_state_sha256"] != selected_state_sha or any(
        report["candidate_state_sha256"] != selected_state_sha
        for report in evaluations.values()
    ):
        raise Lfm2CpcfShortConvV4FoldError(
            "Frozen winner evaluations do not bind the selected candidate state."
        )
    full_source_coverage = bool(
        basis.report["full_source_numeric_participation"]
        and priors.report["source_channels_dropped"] == 0
    )
    gate = _gate_report(
        policy=gate_policy,
        fit=evaluations["fit"],
        validation=evaluations["validation"],
        full_source_coverage=full_source_coverage,
    )
    state = {
        "in_proj.weight": in_weight,
        "conv.weight": fir_kernel.unsqueeze(1).contiguous(),
        "out_proj.weight": out_weight,
        "in_proj.bias": in_bias,
        "conv.bias": fir_bias,
        "out_proj.bias": out_bias,
    }
    compiled_state_sha = _state_sha256(state)
    common = {
        "solver_schema": LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
        "solver_policy": solver_policy.as_dict(),
        "solver_policy_sha256": solver_policy.authentication_sha256,
        "gate_policy": gate_policy.as_dict(),
        "source_hidden_size": source_hidden,
        "target_hidden_size": target_hidden,
        "source_fir_length": int(source_kernel.shape[1]),
        "target_fir_length": int(fir_kernel.shape[1]),
        "hidden_width_compression_ratio": source_hidden / target_hidden,
        "case_authentication_sha256": {
            split: [
                case.authentication_sha256 for case in ordered if case.split == split
            ]
            for split in ("fit", "selection", "validation")
        },
        "basis": dict(basis.report),
        "input_lift": dict(lift_report),
        "transported_priors": dict(priors.report),
        "candidate_fit_reports": {
            candidate_id: dict(result.report)
            for candidate_id, result in candidate_results.items()
        },
        "candidate_selection_reports": {
            candidate_id: dict(report)
            for candidate_id, report in selection_reports.items()
        },
        "selection_ranking": dict(ranking),
        "selected_candidate_id": selected_id,
        "fit_evaluation": dict(evaluations["fit"]),
        "selection_evaluation": dict(evaluations["selection"]),
        "validation_evaluation": dict(evaluations["validation"]),
        "behavior_gate": gate,
        "compiled_state_sha256": compiled_state_sha,
        "source_channels_selected": basis.report["source_channels_selected"],
        "source_channels_dropped": basis.report["source_channels_dropped"],
        "source_channels_pruned": 0,
        "source_channels_quantized": 0,
        "fit_only_parameter_solves": True,
        "selection_used_only_for_candidate_ranking": True,
        "validation_used_only_after_candidate_freeze": True,
        "execution": {
            "parent_model_forward_calls": 0,
            "target_model_forward_calls": 0,
            "generation_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "fallback_paths": 0,
        },
    }
    if gate_policy.enforce and not gate["passed"]:
        failure_body = {
            "schema": LFM2_CPCF_SHORT_CONV_V4_FAILURE_SCHEMA,
            **common,
            "compiled_short_conv_fold_issued": False,
            "promotion_claimed": False,
            "promotable_candidate": False,
            "failure_evidence_only": True,
        }
        failure = {
            **failure_body,
            "report_sha256": _sha(failure_body),
        }
        raise Lfm2CpcfShortConvValidationError(
            "Compiled CPCF short-conv v4 failed enforced behavior gates: "
            + ", ".join(gate["failed_checks"]),
            failure,
        )
    body = {
        "schema": LFM2_CPCF_SHORT_CONV_V4_FOLD_SCHEMA,
        **common,
        "compiled_short_conv_fold_issued": True,
        "promotion_claimed": False,
        "promotable_candidate": bool(gate["passed"]),
    }
    report = {**body, "report_sha256": _sha(body)}
    return Lfm2CompiledCpcfShortConvV4Fold(
        in_proj_weight=in_weight,
        in_proj_bias=in_bias,
        conv_weight=fir_kernel.unsqueeze(1).contiguous(),
        conv_bias=fir_bias,
        out_proj_weight=out_weight,
        out_proj_bias=out_bias,
        report=report,
    )


__all__ = [
    "LFM2_CPCF_SHORT_CONV_V4_FAILURE_SCHEMA",
    "LFM2_CPCF_SHORT_CONV_V4_FOLD_SCHEMA",
    "Lfm2CompiledCpcfShortConvV4Fold",
    "Lfm2CpcfShortConvV4FoldCase",
    "Lfm2CpcfShortConvV4FoldError",
    "compile_lfm2_cpcf_short_conv_v4_fold",
]
