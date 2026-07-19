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
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from neural_forge.hf_lfm2_cpcf_candidate_replay import (
    Lfm2CpcfCandidateReplayEvaluator,
    evaluate_lfm2_cpcf_candidate_states_with_backend,
)
from neural_forge.hf_lfm2_cpcf_short_conv_fold import (
    Lfm2CpcfShortConvGatePolicy,
    Lfm2CpcfShortConvValidationError,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v5 import (
    LFM2_CPCF_SHORT_CONV_V5_SCHEMA,
    verify_lfm2_cpcf_short_conv_v5_report,
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
    fixed_projection_trust_path_fractions,
    interpolate_short_conv_v4_candidate_state,
    rank_short_conv_v4_selection_candidates,
    run_fixed_sampled_cp_variable_projection,
    transport_parent_short_conv_priors,
)


LFM2_CPCF_SHORT_CONV_V4_FOLD_SCHEMA = "neural-forge-lfm2-cpcf-short-conv-fold-v4"
LFM2_CPCF_SHORT_CONV_V5_FOLD_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-fold-v5-scheduled-replay"
)
LFM2_CPCF_SHORT_CONV_V5_SELECTOR_FOLD_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-fold-v5-preinstall-selector-v1"
)
LFM2_CPCF_SHORT_CONV_V5_SELECTOR_REQUEST_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-v5-selector-request-v1"
)
LFM2_CPCF_SHORT_CONV_V5_SELECTOR_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-v5-selector-receipt-v1"
)
LFM2_CPCF_SHORT_CONV_V5_MACROCELL_SELECTOR_FOLD_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-fold-v5-macrocell-selector-v1"
)
LFM2_CPCF_SHORT_CONV_V5_MACROCELL_FRONTIER_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-v5-macrocell-frontier-v1"
)
LFM2_CPCF_SHORT_CONV_V5_MACROCELL_CANDIDATE_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-v5-macrocell-candidate-v1"
)
LFM2_CPCF_SHORT_CONV_V5_MACROCELL_EVALUATION_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-v5-macrocell-evaluation-v1"
)
LFM2_CPCF_SHORT_CONV_V5_MACROCELL_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-v5-macrocell-evidence-v1"
)
LFM2_CPCF_SHORT_CONV_V5_MACROCELL_SELECTOR_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-v5-macrocell-selector-receipt-v1"
)
LFM2_CPCF_SHORT_CONV_V5_MACROCELL_SELECTOR_FAILURE_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-fold-v5-macrocell-selector-failure-v1"
)
LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_FOLD_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-fold-v5-numerical-frontier-v1"
)
LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_REQUEST_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-v5-numerical-frontier-request-v1"
)
LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-v5-numerical-frontier-v1"
)
LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_RAW_CANDIDATE_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-v5-numerical-raw-candidate-v1"
)
LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_PROVENANCE_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-v5-numerical-provenance-v1"
)
LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_POOL_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-v5-numerical-pool-v1"
)
LFM2_CPCF_SHORT_CONV_V4_FAILURE_V1_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-fold-v4-failure-v1"
)
LFM2_CPCF_SHORT_CONV_V4_FAILURE_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-fold-v4-failure-v2"
)
LFM2_CPCF_SHORT_CONV_V5_FAILURE_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-fold-v5-scheduled-replay-failure-v1"
)
LFM2_CPCF_SHORT_CONV_V5_SELECTOR_FAILURE_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-fold-v5-preinstall-selector-failure-v1"
)
LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_FAILURE_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-fold-v5-numerical-frontier-failure-v1"
)


class Lfm2CpcfShortConvV4FoldError(ValueError):
    """The v4 production fold request or compiled artifact is invalid."""


class Lfm2CpcfShortConvV5CandidateSelector(Protocol):
    """Select one retained candidate from an authenticated v5 request."""

    def select(
        self,
        request: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Return a sealed v5 report naming one candidate ID and state SHA."""


class Lfm2CpcfShortConvV5MacrocellCandidateSelector(Protocol):
    """Select one tensor-bearing candidate by replaying its whole macrocell."""

    def select(
        self,
        frontier: "Lfm2CpcfShortConvV5CandidateFrontier",
    ) -> Mapping[str, Any]:
        """Return a sealed fit/selection-only whole-macrocell decision."""


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


def _seal_report(body: Mapping[str, Any]) -> dict[str, Any]:
    detached = _canonical_copy(body, "report")
    return {**detached, "report_sha256": _sha(detached)}


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


_NUMERICAL_SOURCE_TENSOR_ROLES = frozenset(
    {
        "source_in_proj_weight",
        "source_in_proj_bias",
        "source_fir_kernel",
        "source_fir_bias",
        "source_out_proj_weight",
        "source_out_proj_bias",
        "output_encoder",
    }
)
_NUMERICAL_TARGET_TEMPLATE_ROLES = frozenset(
    {
        "in_proj_weight",
        "in_proj_bias",
        "fir_kernel",
        "fir_bias",
        "writer_weight",
        "writer_bias",
    }
)
_NUMERICAL_CANDIDATE_ROLES = (
    "in_proj_weight",
    "in_proj_bias",
    "fir_kernel",
    "fir_bias",
    "writer_weight",
    "writer_bias",
)
_ZERO_EXECUTION = {
    "parent_model_forward_calls": 0,
    "target_model_forward_calls": 0,
    "generation_calls": 0,
    "training_steps": 0,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
}


def _valid_sha256_text(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _optional_tensor_inventory(
    values: Mapping[str, Any | None],
    *,
    prefix: str,
) -> dict[str, str | None]:
    return {
        name: (None if value is None else _tensor_sha256(value, f"{prefix}.{name}"))
        for name, value in sorted(values.items())
    }


def _snapshot_numerical_tensor_mapping(
    values: Mapping[str, Any | None],
    *,
    expected_roles: frozenset[str],
    dimensions: Mapping[str, int],
    name: str,
) -> Mapping[str, Any | None]:
    if not isinstance(values, Mapping) or set(values) != expected_roles:
        raise Lfm2CpcfShortConvV4FoldError(
            f"{name} does not contain the exact audited tensor roles."
        )
    snapshots = {}
    for role in sorted(expected_roles):
        value = values[role]
        if value is None:
            if not role.endswith("_bias"):
                raise Lfm2CpcfShortConvV4FoldError(f"{name}.{role} cannot be absent.")
            snapshots[role] = None
        else:
            snapshots[role] = _snapshot_float(
                value,
                f"{name}.{role}",
                dimensions=dimensions[role],
            )
    return MappingProxyType(snapshots)


@dataclass(frozen=True, slots=True)
class Lfm2CpcfShortConvNumericalFrontierRequest:
    """Detached fit-only evidence for one optional numerical generator."""

    source_tensors: Mapping[str, Any | None]
    cases: tuple[Lfm2CpcfShortConvV4FoldCase, ...]
    parent_contexts: Mapping[str, Any]
    target_templates: Mapping[str, Any | None]
    policy: Lfm2CpcfShortConvV4SolverPolicy
    report: Mapping[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.policy, Lfm2CpcfShortConvV4SolverPolicy):
            raise TypeError("policy must be Lfm2CpcfShortConvV4SolverPolicy.")
        if not self.policy.verify_authentication():
            raise Lfm2CpcfShortConvV4FoldError(
                "Numerical-frontier policy failed authentication."
            )
        source = _snapshot_numerical_tensor_mapping(
            self.source_tensors,
            expected_roles=_NUMERICAL_SOURCE_TENSOR_ROLES,
            dimensions={
                "source_in_proj_weight": 2,
                "source_in_proj_bias": 1,
                "source_fir_kernel": 2,
                "source_fir_bias": 1,
                "source_out_proj_weight": 2,
                "source_out_proj_bias": 1,
                "output_encoder": 2,
            },
            name="source_tensors",
        )
        templates = _snapshot_numerical_tensor_mapping(
            self.target_templates,
            expected_roles=_NUMERICAL_TARGET_TEMPLATE_ROLES,
            dimensions={
                "in_proj_weight": 2,
                "in_proj_bias": 1,
                "fir_kernel": 2,
                "fir_bias": 1,
                "writer_weight": 2,
                "writer_bias": 1,
            },
            name="target_templates",
        )
        values = tuple(self.cases)
        if (
            not values
            or any(
                not isinstance(case, Lfm2CpcfShortConvV4FoldCase)
                or case.split != "fit"
                or not case.verify_authentication()
                for case in values
            )
            or len({case.case_id for case in values}) != len(values)
        ):
            raise Lfm2CpcfShortConvV4FoldError(
                "Numerical-frontier request requires unique authenticated "
                "fit cases only."
            )
        cases = tuple(
            Lfm2CpcfShortConvV4FoldCase(
                case_id=case.case_id,
                split=case.split,
                suffix_start=case.suffix_start,
                actual_child_operator_input_history=(
                    case.actual_child_operator_input_history
                ),
                parent_operator_input_history=(case.parent_operator_input_history),
                parent_b_history=case.parent_b_history,
                parent_c_history=case.parent_c_history,
                parent_x_history=case.parent_x_history,
                parent_post_operator_residual_target_space_suffix=(
                    case.parent_post_operator_residual_target_space_suffix
                ),
                actual_child_pre_operator_residual_suffix=(
                    case.actual_child_pre_operator_residual_suffix
                ),
                valid_prefix_mask=case.valid_prefix_mask,
                authentication_sha256=case.authentication_sha256,
            )
            for case in sorted(values, key=lambda item: item.case_id)
        )
        if not isinstance(self.parent_contexts, Mapping) or set(
            self.parent_contexts
        ) != {case.case_id for case in cases}:
            raise Lfm2CpcfShortConvV4FoldError(
                "Numerical-frontier parent contexts do not match fit cases."
            )
        contexts = MappingProxyType(
            {
                case.case_id: _snapshot_float(
                    self.parent_contexts[case.case_id],
                    f"parent_contexts.{case.case_id}",
                    dimensions=3,
                )
                for case in cases
            }
        )
        source_hidden = int(source["source_fir_kernel"].shape[0])
        target_hidden = int(templates["in_proj_weight"].shape[1])
        if (
            tuple(source["source_in_proj_weight"].shape)
            != (3 * source_hidden, source_hidden)
            or tuple(source["source_out_proj_weight"].shape)
            != (source_hidden, source_hidden)
            or tuple(source["output_encoder"].shape) != (source_hidden, target_hidden)
            or tuple(templates["in_proj_weight"].shape)
            != (3 * target_hidden, target_hidden)
            or tuple(templates["writer_weight"].shape) != (target_hidden, target_hidden)
            or int(templates["fir_kernel"].shape[0]) != target_hidden
        ):
            raise Lfm2CpcfShortConvV4FoldError(
                "Numerical-frontier request tensor topology is inconsistent."
            )
        optional_shapes = {
            "source_in_proj_bias": (3 * source_hidden,),
            "source_fir_bias": (source_hidden,),
            "source_out_proj_bias": (source_hidden,),
        }
        target_optional_shapes = {
            "in_proj_bias": (3 * target_hidden,),
            "fir_bias": (target_hidden,),
            "writer_bias": (target_hidden,),
        }
        if any(
            source[role] is not None and tuple(source[role].shape) != shape
            for role, shape in optional_shapes.items()
        ) or any(
            templates[role] is not None and tuple(templates[role].shape) != shape
            for role, shape in target_optional_shapes.items()
        ):
            raise Lfm2CpcfShortConvV4FoldError(
                "Numerical-frontier request bias topology is inconsistent."
            )
        for case in cases:
            context = contexts[case.case_id]
            if (
                tuple(context.shape[:2]) != tuple(case.parent_b_history.shape[:2])
                or int(context.shape[-1]) != source_hidden
                or int(case.actual_child_operator_input_history.shape[-1])
                != target_hidden
            ):
                raise Lfm2CpcfShortConvV4FoldError(
                    "Numerical-frontier fit context geometry is inconsistent."
                )
        object.__setattr__(self, "source_tensors", source)
        object.__setattr__(self, "target_templates", templates)
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "parent_contexts", contexts)
        object.__setattr__(
            self,
            "report",
            _deep_freeze(_seal_report(self._report_body())),
        )

    def _report_body(self) -> dict[str, Any]:
        source_inventory = _optional_tensor_inventory(
            self.source_tensors,
            prefix="numerical_frontier.source",
        )
        template_inventory = _optional_tensor_inventory(
            self.target_templates,
            prefix="numerical_frontier.target_template",
        )
        context_inventory = {
            case.case_id: _tensor_sha256(
                self.parent_contexts[case.case_id],
                f"numerical_frontier.parent_context.{case.case_id}",
            )
            for case in self.cases
        }
        return {
            "schema": (LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_REQUEST_SCHEMA),
            "kind": "detached_fit_only_numerical_frontier_request",
            "policy_sha256": self.policy.authentication_sha256,
            "target_dtype": self.policy.target_dtype,
            "source_tensor_sha256": source_inventory,
            "source_tensor_inventory_sha256": _sha(source_inventory),
            "fit_case_authentication_sha256": [
                case.authentication_sha256 for case in self.cases
            ],
            "parent_context_sha256": context_inventory,
            "parent_context_inventory_sha256": _sha(context_inventory),
            "target_template_sha256": template_inventory,
            "target_template_inventory_sha256": _sha(template_inventory),
            "source_hidden_size": int(
                self.source_tensors["source_fir_kernel"].shape[0]
            ),
            "target_hidden_size": int(self.target_templates["in_proj_weight"].shape[1]),
            "source_fir_length": int(self.source_tensors["source_fir_kernel"].shape[1]),
            "target_fir_length": int(self.target_templates["fir_kernel"].shape[1]),
            "selection_cases_exposed": False,
            "validation_cases_exposed": False,
            "fit_only": True,
            "execution": dict(_ZERO_EXECUTION),
        }

    @property
    def request_sha256(self) -> str:
        return str(self.report["report_sha256"])

    def verify_authentication(self) -> bool:
        try:
            return bool(
                self.policy.verify_authentication()
                and all(case.verify_authentication() for case in self.cases)
                and _deep_thaw(self.report) == _seal_report(self._report_body())
            )
        except (
            AttributeError,
            KeyError,
            OverflowError,
            TypeError,
            ValueError,
        ):
            return False

    def as_dict(self) -> dict[str, Any]:
        if not self.verify_authentication():
            raise Lfm2CpcfShortConvV4FoldError(
                "Numerical-frontier request changed after authentication."
            )
        return _deep_thaw(self.report)


@dataclass(frozen=True, slots=True)
class Lfm2CpcfShortConvNumericalRawCandidate:
    """One immutable fit-only raw state emitted by a numerical adapter."""

    request_sha256: str
    candidate_id: str
    candidate_kind: str
    in_proj_weight: Any
    in_proj_bias: Any | None
    fir_kernel: Any
    fir_bias: Any | None
    writer_weight: Any
    writer_bias: Any | None
    method_provenance: Mapping[str, Any]
    report: Mapping[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not _valid_sha256_text(self.request_sha256):
            raise Lfm2CpcfShortConvV4FoldError(
                "Raw candidate request_sha256 is invalid."
            )
        candidate_id = str(self.candidate_id).strip()
        candidate_kind = str(self.candidate_kind).strip()
        if not candidate_id or not candidate_kind:
            raise Lfm2CpcfShortConvV4FoldError(
                "Raw candidate id and kind must be nonempty."
            )
        for role, dimensions in (
            ("in_proj_weight", 2),
            ("fir_kernel", 2),
            ("writer_weight", 2),
        ):
            object.__setattr__(
                self,
                role,
                _snapshot_float(
                    getattr(self, role),
                    f"raw_candidate.{role}",
                    dimensions=dimensions,
                ),
            )
        for role in ("in_proj_bias", "fir_bias", "writer_bias"):
            value = getattr(self, role)
            if value is not None:
                object.__setattr__(
                    self,
                    role,
                    _snapshot_float(
                        value,
                        f"raw_candidate.{role}",
                        dimensions=1,
                    ),
                )
        provenance = _canonical_copy(
            self.method_provenance,
            "raw candidate method_provenance",
        )
        if (
            not isinstance(provenance.get("method"), str)
            or not provenance["method"].strip()
        ):
            raise Lfm2CpcfShortConvV4FoldError(
                "Raw candidate provenance must name its method."
            )
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "candidate_kind", candidate_kind)
        object.__setattr__(
            self,
            "method_provenance",
            _deep_freeze(provenance),
        )
        object.__setattr__(
            self,
            "report",
            _deep_freeze(_seal_report(self._report_body())),
        )

    def _live_state(self) -> dict[str, Any | None]:
        return {role: getattr(self, role) for role in _NUMERICAL_CANDIDATE_ROLES}

    def _report_body(self) -> dict[str, Any]:
        provenance = _deep_thaw(self.method_provenance)
        return {
            "schema": (LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_RAW_CANDIDATE_SCHEMA),
            "kind": "authenticated_fit_only_numerical_raw_candidate",
            "request_sha256": self.request_sha256,
            "candidate_id": self.candidate_id,
            "candidate_kind": self.candidate_kind,
            "state_sha256": _state_sha256(self._live_state()),
            "method_provenance": provenance,
            "method_provenance_sha256": _sha(provenance),
            "fit_only": True,
            "selection_cases_consumed": 0,
            "validation_cases_consumed": 0,
            "execution": dict(_ZERO_EXECUTION),
        }

    @property
    def state_sha256(self) -> str:
        return str(self.report["state_sha256"])

    def state_dict(self) -> Mapping[str, Any | None]:
        return MappingProxyType(
            {
                role: (
                    None if getattr(self, role) is None else getattr(self, role).clone()
                )
                for role in _NUMERICAL_CANDIDATE_ROLES
            }
        )

    def verify_authentication(self) -> bool:
        try:
            return _deep_thaw(self.report) == _seal_report(self._report_body())
        except (
            AttributeError,
            KeyError,
            OverflowError,
            TypeError,
            ValueError,
        ):
            return False

    def as_dict(self) -> dict[str, Any]:
        if not self.verify_authentication():
            raise Lfm2CpcfShortConvV4FoldError(
                "Numerical raw candidate changed after authentication."
            )
        return _deep_thaw(self.report)


@dataclass(frozen=True, slots=True)
class Lfm2CpcfShortConvNumericalFrontier:
    """One authenticated fit-only raw frontier plus its context encoder."""

    adapter_id: str
    request_sha256: str
    context_encoder: Any
    raw_candidates: tuple[Lfm2CpcfShortConvNumericalRawCandidate, ...]
    fit_case_authentication_sha256: tuple[str, ...]
    adapter_provenance: Mapping[str, Any]
    report: Mapping[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        adapter_id = str(self.adapter_id).strip()
        if not adapter_id or not _valid_sha256_text(self.request_sha256):
            raise Lfm2CpcfShortConvV4FoldError(
                "Numerical frontier adapter id or request seal is invalid."
            )
        encoder = _snapshot_float(
            self.context_encoder,
            "numerical_frontier.context_encoder",
            dimensions=2,
        )
        candidates = tuple(self.raw_candidates)
        if (
            not candidates
            or any(
                not isinstance(
                    candidate,
                    Lfm2CpcfShortConvNumericalRawCandidate,
                )
                or not candidate.verify_authentication()
                or candidate.request_sha256 != self.request_sha256
                for candidate in candidates
            )
            or len({candidate.candidate_id for candidate in candidates})
            != len(candidates)
        ):
            raise Lfm2CpcfShortConvV4FoldError(
                "Numerical frontier raw candidates are empty, duplicated, "
                "unsealed, or request-mismatched."
            )
        fit_seals = tuple(self.fit_case_authentication_sha256)
        if (
            not fit_seals
            or len(fit_seals) != len(set(fit_seals))
            or any(not _valid_sha256_text(value) for value in fit_seals)
        ):
            raise Lfm2CpcfShortConvV4FoldError(
                "Numerical frontier fit coverage is invalid."
            )
        provenance = _canonical_copy(
            self.adapter_provenance,
            "numerical frontier adapter_provenance",
        )
        if (
            not isinstance(provenance.get("implementation"), str)
            or not provenance["implementation"].strip()
        ):
            raise Lfm2CpcfShortConvV4FoldError(
                "Numerical frontier provenance must name its implementation."
            )
        object.__setattr__(self, "adapter_id", adapter_id)
        object.__setattr__(self, "context_encoder", encoder)
        object.__setattr__(self, "raw_candidates", candidates)
        object.__setattr__(
            self,
            "fit_case_authentication_sha256",
            fit_seals,
        )
        object.__setattr__(
            self,
            "adapter_provenance",
            _deep_freeze(provenance),
        )
        object.__setattr__(
            self,
            "report",
            _deep_freeze(_seal_report(self._report_body())),
        )

    def _numeric_participants(self) -> list[int]:
        participation = self.context_encoder.abs().sum(dim=1)
        return [
            index
            for index, value in enumerate(participation.tolist())
            if float(value) > 0.0
        ]

    def _report_body(self) -> dict[str, Any]:
        candidate_reports = {
            candidate.candidate_id: candidate.as_dict()
            for candidate in self.raw_candidates
        }
        participants = self._numeric_participants()
        source_channels = int(self.context_encoder.shape[0])
        provenance = _deep_thaw(self.adapter_provenance)
        return {
            "schema": (LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_SCHEMA),
            "kind": "authenticated_fit_only_numerical_frontier",
            "adapter_id": self.adapter_id,
            "request_sha256": self.request_sha256,
            "context_encoder_sha256": _tensor_sha256(
                self.context_encoder,
                "numerical_frontier.context_encoder",
            ),
            "fit_case_authentication_sha256": list(self.fit_case_authentication_sha256),
            "raw_candidate_count": len(self.raw_candidates),
            "raw_candidate_ids": [
                candidate.candidate_id for candidate in self.raw_candidates
            ],
            "raw_candidate_state_sha256": {
                candidate.candidate_id: candidate.state_sha256
                for candidate in self.raw_candidates
            },
            "raw_candidate_report_sha256": {
                candidate.candidate_id: candidate.report["report_sha256"]
                for candidate in self.raw_candidates
            },
            "raw_candidate_reports": candidate_reports,
            "source_channel_count": source_channels,
            "source_numeric_participant_indices": participants,
            "source_channels_selected": len(participants),
            "source_channels_dropped": source_channels - len(participants),
            "full_source_numeric_participation": (len(participants) == source_channels),
            "adapter_provenance": provenance,
            "adapter_provenance_sha256": _sha(provenance),
            "fit_only": True,
            "selection_cases_consumed": 0,
            "validation_cases_consumed": 0,
            "execution": dict(_ZERO_EXECUTION),
        }

    @property
    def full_source_numeric_participation(self) -> bool:
        return bool(self.report["full_source_numeric_participation"])

    @property
    def source_channels_selected(self) -> int:
        return int(self.report["source_channels_selected"])

    @property
    def source_channels_dropped(self) -> int:
        return int(self.report["source_channels_dropped"])

    def verify_authentication(self) -> bool:
        try:
            return bool(
                all(
                    candidate.verify_authentication()
                    for candidate in self.raw_candidates
                )
                and _deep_thaw(self.report) == _seal_report(self._report_body())
            )
        except (
            AttributeError,
            KeyError,
            OverflowError,
            TypeError,
            ValueError,
        ):
            return False

    def as_dict(self) -> dict[str, Any]:
        if not self.verify_authentication():
            raise Lfm2CpcfShortConvV4FoldError(
                "Numerical frontier changed after authentication."
            )
        return _deep_thaw(self.report)


class Lfm2CpcfShortConvNumericalFrontierAdapter(Protocol):
    """Generate one typed fit-only raw frontier for the existing compiler."""

    def generate(
        self,
        request: Lfm2CpcfShortConvNumericalFrontierRequest,
    ) -> Lfm2CpcfShortConvNumericalFrontier:
        """Return a sealed raw frontier bound to ``request``."""


def _validate_numerical_frontier(
    frontier: Lfm2CpcfShortConvNumericalFrontier,
    *,
    request: Lfm2CpcfShortConvNumericalFrontierRequest,
) -> None:
    if (
        not isinstance(frontier, Lfm2CpcfShortConvNumericalFrontier)
        or not frontier.verify_authentication()
        or not request.verify_authentication()
        or frontier.request_sha256 != request.request_sha256
        or tuple(frontier.fit_case_authentication_sha256)
        != tuple(request.report["fit_case_authentication_sha256"])
        or tuple(frontier.context_encoder.shape)
        != (
            int(request.report["source_hidden_size"]),
            int(request.report["target_hidden_size"]),
        )
        or frontier.report["source_channel_count"]
        != request.report["source_hidden_size"]
    ):
        raise Lfm2CpcfShortConvV4FoldError(
            "Numerical frontier is not authenticated to the complete fit "
            "request and context geometry."
        )
    for candidate in frontier.raw_candidates:
        state = candidate.state_dict()
        for role in _NUMERICAL_CANDIDATE_ROLES:
            candidate_value = state[role]
            template_value = request.target_templates[role]
            if (candidate_value is None) != (template_value is None) or (
                candidate_value is not None
                and tuple(candidate_value.shape) != tuple(template_value.shape)
            ):
                raise Lfm2CpcfShortConvV4FoldError(
                    f"Numerical raw candidate differs from target topology for {role}."
                )


def _candidate_state_sha256(
    candidate_states: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    return {
        candidate_id: _state_sha256(candidate_states[candidate_id])
        for candidate_id in sorted(candidate_states)
    }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfShortConvV5MacrocellCandidate:
    """One detached target-dtype operator state exposed for macrocell replay."""

    candidate_id: str
    in_proj_weight: Any
    in_proj_bias: Any | None
    fir_kernel: Any
    fir_bias: Any | None
    writer_weight: Any
    writer_bias: Any | None
    fit_provenance: Mapping[str, Any]
    operator_selection_report: Mapping[str, Any]
    report: Mapping[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        torch = _torch()
        candidate_id = str(self.candidate_id).strip()
        if not candidate_id:
            raise Lfm2CpcfShortConvV4FoldError(
                "Macrocell candidate_id must be nonempty text."
            )
        for name, dimensions in (
            ("in_proj_weight", 2),
            ("fir_kernel", 2),
            ("writer_weight", 2),
        ):
            object.__setattr__(
                self,
                name,
                _snapshot_float(
                    getattr(self, name),
                    f"macrocell candidate {candidate_id}.{name}",
                    dimensions=dimensions,
                ),
            )
        for name in ("in_proj_bias", "fir_bias", "writer_bias"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _snapshot_float(
                        value,
                        f"macrocell candidate {candidate_id}.{name}",
                        dimensions=1,
                    ),
                )
        state = self.state_dict()
        hidden = int(self.in_proj_weight.shape[1])
        dtype = self.in_proj_weight.dtype
        if (
            tuple(self.in_proj_weight.shape) != (3 * hidden, hidden)
            or int(self.fir_kernel.shape[0]) != hidden
            or int(self.fir_kernel.shape[1]) <= 0
            or tuple(self.writer_weight.shape) != (hidden, hidden)
            or (
                self.in_proj_bias is not None
                and tuple(self.in_proj_bias.shape) != (3 * hidden,)
            )
            or (self.fir_bias is not None and tuple(self.fir_bias.shape) != (hidden,))
            or (
                self.writer_bias is not None
                and tuple(self.writer_bias.shape) != (hidden,)
            )
            or any(
                value is not None and value.dtype != dtype for value in state.values()
            )
            or dtype
            not in {
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
            }
        ):
            raise Lfm2CpcfShortConvV4FoldError(
                "Macrocell candidate differs from a native target short-conv "
                "topology or dtype."
            )
        fit = _canonical_copy(
            _deep_thaw(self.fit_provenance),
            f"macrocell candidate {candidate_id} fit provenance",
        )
        selection = _canonical_copy(
            _deep_thaw(self.operator_selection_report),
            f"macrocell candidate {candidate_id} operator selection report",
        )
        for name, value in (("fit provenance", fit), ("selection report", selection)):
            sealed = dict(value)
            supplied = sealed.pop("report_sha256", None)
            if supplied != _sha(sealed):
                raise Lfm2CpcfShortConvV4FoldError(
                    f"Macrocell candidate {name} is not canonically sealed."
                )
        state_sha256 = _state_sha256(state)
        install_state_sha256 = _state_sha256(self.install_state_dict())
        if (
            fit.get("candidate_id") != candidate_id
            or fit.get("target_dtype_candidate_state_sha256") != state_sha256
            or selection.get("candidate_state_sha256") != state_sha256
        ):
            raise Lfm2CpcfShortConvV4FoldError(
                "Macrocell candidate state is not bound to its fit and "
                "selection evidence."
            )
        body = {
            "schema": LFM2_CPCF_SHORT_CONV_V5_MACROCELL_CANDIDATE_SCHEMA,
            "candidate_id": candidate_id,
            "target_dtype": _dtype_name(self.in_proj_weight),
            "target_hidden_size": hidden,
            "target_fir_length": int(self.fir_kernel.shape[1]),
            "bias_topology": {
                "in_proj": self.in_proj_bias is not None,
                "conv": self.fir_bias is not None,
                "out_proj": self.writer_bias is not None,
            },
            "candidate_state_sha256": state_sha256,
            "candidate_install_state_sha256": install_state_sha256,
            "candidate_tensor_sha256": _optional_tensor_inventory(
                state,
                prefix=f"macrocell_candidate.{candidate_id}",
            ),
            "fit_provenance_report_sha256": fit["report_sha256"],
            "operator_selection_report_sha256": selection["report_sha256"],
            "fit_evidence_exposed": True,
            "operator_selection_evidence_exposed": True,
            "validation_evidence_exposed": False,
            "validation_tensors_exposed": False,
            "execution": dict(_ZERO_EXECUTION),
        }
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "fit_provenance", _deep_freeze(fit))
        object.__setattr__(
            self,
            "operator_selection_report",
            _deep_freeze(selection),
        )
        object.__setattr__(
            self,
            "report",
            _deep_freeze(_seal_report(body)),
        )

    @property
    def state_sha256(self) -> str:
        return str(self.report["candidate_state_sha256"])

    def state_dict(self) -> Mapping[str, Any | None]:
        return MappingProxyType(
            {
                "in_proj_weight": self.in_proj_weight.clone(),
                "in_proj_bias": (
                    None if self.in_proj_bias is None else self.in_proj_bias.clone()
                ),
                "fir_kernel": self.fir_kernel.clone(),
                "fir_bias": (None if self.fir_bias is None else self.fir_bias.clone()),
                "writer_weight": self.writer_weight.clone(),
                "writer_bias": (
                    None if self.writer_bias is None else self.writer_bias.clone()
                ),
            }
        )

    def install_state_dict(self) -> Mapping[str, Any]:
        """Return the exact native short-conv parameter names and shapes."""

        state = self.state_dict()
        result = {
            "in_proj.weight": state["in_proj_weight"],
            "conv.weight": state["fir_kernel"].unsqueeze(1).contiguous(),
            "out_proj.weight": state["writer_weight"],
        }
        for role, name in (
            ("in_proj_bias", "in_proj.bias"),
            ("fir_bias", "conv.bias"),
            ("writer_bias", "out_proj.bias"),
        ):
            if state[role] is not None:
                result[name] = state[role]
        return MappingProxyType(result)

    def verify_authentication(self) -> bool:
        try:
            report = _deep_thaw(self.report)
            body = dict(report)
            supplied = body.pop("report_sha256", None)
            state = self.state_dict()
            return bool(
                supplied == _sha(body)
                and report.get("candidate_id") == self.candidate_id
                and report.get("candidate_state_sha256") == _state_sha256(state)
                and report.get("candidate_install_state_sha256")
                == _state_sha256(self.install_state_dict())
                and report.get("candidate_tensor_sha256")
                == _optional_tensor_inventory(
                    state,
                    prefix=f"macrocell_candidate.{self.candidate_id}",
                )
                and report.get("bias_topology")
                == {
                    "in_proj": self.in_proj_bias is not None,
                    "conv": self.fir_bias is not None,
                    "out_proj": self.writer_bias is not None,
                }
                and report.get("fit_provenance_report_sha256")
                == self.fit_provenance["report_sha256"]
                and report.get("operator_selection_report_sha256")
                == self.operator_selection_report["report_sha256"]
            )
        except (KeyError, TypeError, ValueError):
            return False

    def as_dict(self) -> dict[str, Any]:
        if not self.verify_authentication():
            raise Lfm2CpcfShortConvV4FoldError(
                "Macrocell candidate changed after authentication."
            )
        return _deep_thaw(self.report)


@dataclass(frozen=True, slots=True)
class Lfm2CpcfShortConvV5CandidateFrontier:
    """Tensor-bearing fit/selection frontier with no validation exposure."""

    candidates: tuple[Lfm2CpcfShortConvV5MacrocellCandidate, ...]
    solver_policy_sha256: str
    target_dtype: str
    fit_bank_sha256: str
    selection_bank_sha256: str
    default_ranking: Mapping[str, Any]
    report: Mapping[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if not candidates or any(
            not isinstance(
                candidate,
                Lfm2CpcfShortConvV5MacrocellCandidate,
            )
            or not candidate.verify_authentication()
            for candidate in candidates
        ):
            raise Lfm2CpcfShortConvV4FoldError(
                "Macrocell frontier requires authenticated candidates."
            )
        candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
        if candidate_ids != tuple(sorted(candidate_ids)) or len(
            set(candidate_ids)
        ) != len(candidate_ids):
            raise Lfm2CpcfShortConvV4FoldError(
                "Macrocell frontier candidate IDs must be unique and sorted."
            )
        policy_sha256 = str(self.solver_policy_sha256).strip().lower()
        if not _valid_sha256_text(policy_sha256):
            raise Lfm2CpcfShortConvV4FoldError(
                "Macrocell frontier solver policy digest is invalid."
            )
        target_dtype = str(self.target_dtype).strip()
        if target_dtype not in {"float16", "bfloat16", "float32", "float64"}:
            raise Lfm2CpcfShortConvV4FoldError(
                "Macrocell frontier target dtype is unsupported."
            )
        fit_bank_sha256 = str(self.fit_bank_sha256).strip().lower()
        selection_bank_sha256 = str(self.selection_bank_sha256).strip().lower()
        if (
            not _valid_sha256_text(fit_bank_sha256)
            or not _valid_sha256_text(selection_bank_sha256)
            or fit_bank_sha256 == selection_bank_sha256
        ):
            raise Lfm2CpcfShortConvV4FoldError(
                "Macrocell frontier requires distinct authenticated fit and "
                "selection state banks."
            )
        ranking = _canonical_copy(
            _deep_thaw(self.default_ranking),
            "macrocell frontier default ranking",
        )
        ranking_body = dict(ranking)
        ranking_seal = ranking_body.pop("report_sha256", None)
        if (
            ranking_seal != _sha(ranking_body)
            or not isinstance(ranking.get("ranking"), list)
            or not ranking["ranking"]
            or set(candidate_ids)
            != {
                str(value["candidate_id"])
                for value in ranking["ranking"]
                if isinstance(value, Mapping)
            }
            or ranking.get("selected_candidate_id") not in candidate_ids
        ):
            raise Lfm2CpcfShortConvV4FoldError(
                "Macrocell frontier default ranking is not complete and sealed."
            )
        if any(
            candidate.report["target_dtype"] != target_dtype for candidate in candidates
        ):
            raise Lfm2CpcfShortConvV4FoldError(
                "Macrocell frontier candidates differ in target dtype."
            )
        bias_topology = dict(candidates[0].report["bias_topology"])
        if any(
            dict(candidate.report["bias_topology"]) != bias_topology
            for candidate in candidates
        ):
            raise Lfm2CpcfShortConvV4FoldError(
                "Macrocell frontier candidates differ in native target bias topology."
            )
        body = {
            "schema": LFM2_CPCF_SHORT_CONV_V5_MACROCELL_FRONTIER_SCHEMA,
            "solver_policy_sha256": policy_sha256,
            "target_dtype": target_dtype,
            "target_bias_topology": bias_topology,
            "fit_bank_sha256": fit_bank_sha256,
            "selection_bank_sha256": selection_bank_sha256,
            "candidate_count": len(candidates),
            "candidate_ids": list(candidate_ids),
            "candidate_state_sha256": {
                candidate.candidate_id: candidate.state_sha256
                for candidate in candidates
            },
            "candidate_report_sha256": {
                candidate.candidate_id: candidate.report["report_sha256"]
                for candidate in candidates
            },
            "default_ranking_report_sha256": ranking["report_sha256"],
            "default_selected_candidate_id": ranking["selected_candidate_id"],
            "candidate_state_tensors_exposed": True,
            "fit_provenance_exposed": True,
            "operator_selection_reports_exposed": True,
            "validation_evidence_exposed": False,
            "validation_tensors_exposed": False,
            "validation_reports_exposed": False,
            "selection_scope": (
                "candidate_specific_operator_plus_fit_only_mlp_then_"
                "selection_causal_replay"
            ),
            "execution": dict(_ZERO_EXECUTION),
        }
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "solver_policy_sha256", policy_sha256)
        object.__setattr__(self, "target_dtype", target_dtype)
        object.__setattr__(self, "fit_bank_sha256", fit_bank_sha256)
        object.__setattr__(
            self,
            "selection_bank_sha256",
            selection_bank_sha256,
        )
        object.__setattr__(self, "default_ranking", _deep_freeze(ranking))
        object.__setattr__(
            self,
            "report",
            _deep_freeze(_seal_report(body)),
        )

    @property
    def frontier_sha256(self) -> str:
        return str(self.report["report_sha256"])

    def candidate(
        self,
        candidate_id: str,
    ) -> Lfm2CpcfShortConvV5MacrocellCandidate:
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise KeyError(candidate_id)

    def verify_authentication(self) -> bool:
        try:
            report = _deep_thaw(self.report)
            body = dict(report)
            supplied = body.pop("report_sha256", None)
            return bool(
                supplied == _sha(body)
                and all(
                    candidate.verify_authentication() for candidate in self.candidates
                )
                and report.get("candidate_ids")
                == [candidate.candidate_id for candidate in self.candidates]
                and report.get("candidate_state_sha256")
                == {
                    candidate.candidate_id: candidate.state_sha256
                    for candidate in self.candidates
                }
                and report.get("candidate_report_sha256")
                == {
                    candidate.candidate_id: candidate.report["report_sha256"]
                    for candidate in self.candidates
                }
                and report.get("default_ranking_report_sha256")
                == self.default_ranking["report_sha256"]
                and report.get("fit_bank_sha256") == self.fit_bank_sha256
                and report.get("selection_bank_sha256") == self.selection_bank_sha256
                and report.get("target_bias_topology")
                == dict(self.candidates[0].report["bias_topology"])
            )
        except (KeyError, TypeError, ValueError):
            return False

    def as_dict(self) -> dict[str, Any]:
        if not self.verify_authentication():
            raise Lfm2CpcfShortConvV4FoldError(
                "Macrocell candidate frontier changed after authentication."
            )
        return _deep_thaw(self.report)


def select_lfm2_cpcf_short_conv_v5_macrocell_candidate_ids(
    frontier: Lfm2CpcfShortConvV5CandidateFrontier,
    candidate_budget: int,
) -> tuple[str, ...]:
    """Choose a deterministic, structurally diverse macrocell inventory."""

    if (
        not isinstance(frontier, Lfm2CpcfShortConvV5CandidateFrontier)
        or not frontier.verify_authentication()
    ):
        raise Lfm2CpcfShortConvV4FoldError(
            "Macrocell inventory selection requires an authenticated frontier."
        )
    candidate_ids = tuple(candidate.candidate_id for candidate in frontier.candidates)
    if type(candidate_budget) is not int or not 1 <= candidate_budget <= len(
        candidate_ids
    ):
        raise Lfm2CpcfShortConvV4FoldError(
            "Macrocell candidate budget is outside its authenticated frontier."
        )
    endpoint_control = str(frontier.default_ranking["selected_candidate_id"])
    if endpoint_control not in candidate_ids:
        raise Lfm2CpcfShortConvV4FoldError(
            "Macrocell endpoint control is absent from its frontier."
        )

    selected = {endpoint_control}
    anchor_pool = tuple(
        candidate_id
        for candidate_id in candidate_ids
        if "--anchor-" in candidate_id and candidate_id != endpoint_control
    )
    anchor_slots = candidate_budget - len(selected)
    if anchor_pool:
        if anchor_slots >= len(anchor_pool):
            selected.update(anchor_pool)
        elif anchor_slots == 1:
            selected.add(anchor_pool[(len(anchor_pool) - 1) // 2])
        elif anchor_slots > 1:
            selected.update(
                anchor_pool[(ordinal * (len(anchor_pool) - 1)) // (anchor_slots - 1)]
                for ordinal in range(anchor_slots)
            )

    for entry in frontier.default_ranking["ranking"]:
        candidate_id = str(entry["candidate_id"])
        if len(selected) == candidate_budget:
            break
        selected.add(candidate_id)
    if len(selected) != candidate_budget or not selected.issubset(candidate_ids):
        raise Lfm2CpcfShortConvV4FoldError(
            "Macrocell deterministic inventory did not fill its candidate budget."
        )
    return tuple(sorted(selected))


def _recursive_canonical_report_seals_valid(value: Any) -> bool:
    if isinstance(value, Mapping):
        document = dict(value)
        if "report_sha256" in document:
            supplied = document.pop("report_sha256")
            if not _valid_sha256_text(supplied) or supplied != _sha(document):
                return False
        return all(
            _recursive_canonical_report_seals_valid(child) for child in value.values()
        )
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return all(_recursive_canonical_report_seals_valid(child) for child in value)
    return True


_MACROCELL_EVIDENCE_COMMON_FIELDS = frozenset(
    {
        "schema",
        "kind",
        "candidate_id",
        "actual_report",
        "actual_report_sha256",
        "execution",
        "report_sha256",
    }
)
_MACROCELL_EVIDENCE_FIELDS = {
    "actual_operator_install": _MACROCELL_EVIDENCE_COMMON_FIELDS
    | {
        "candidate_state_sha256",
        "installed_state_sha256",
    },
    "actual_fit_operator_replay": _MACROCELL_EVIDENCE_COMMON_FIELDS
    | {
        "candidate_state_sha256",
        "input_bank_sha256",
    },
    "actual_fit_mlp_artifact": _MACROCELL_EVIDENCE_COMMON_FIELDS
    | {
        "fit_bank_sha256",
        "compiled_state_sha256",
    },
    "actual_fit_mlp_install": _MACROCELL_EVIDENCE_COMMON_FIELDS
    | {
        "compiled_state_sha256",
        "installed_state_sha256",
    },
    "actual_selection_final_replay": _MACROCELL_EVIDENCE_COMMON_FIELDS
    | {
        "candidate_state_sha256",
        "input_bank_sha256",
        "mlp_compiled_state_sha256",
    },
}
_MACROCELL_METRICS_FIELDS = frozenset(
    {
        "schema",
        "kind",
        "candidate_id",
        "selection_bank_sha256",
        "split",
        "phase_count",
        "phase_sha256",
        "post_operator",
        "post_layer",
        "phase_metrics",
        "actual_report",
        "actual_report_sha256",
        "execution",
        "report_sha256",
    }
)
_MACROCELL_PHASE_METRICS_FIELDS = frozenset(
    {
        "phase",
        "phase_count",
        "phase_sha256",
        "post_operator",
        "post_layer",
        "report_sha256",
    }
)
_MACROCELL_ENDPOINT_METRIC_FIELDS = frozenset(
    {
        "relative_rmse",
        "cosine_similarity",
    }
)
_MACROCELL_ROLLBACK_FIELDS = frozenset(
    {
        "schema",
        "kind",
        "candidate_id",
        "target_state_sha256_before",
        "target_state_sha256_after",
        "target_tensor_bytes_restored",
        "target_tensor_identities_preserved",
        "target_gradient_state_preserved",
        "target_module_modes_restored",
        "exact_layer_snapshot_restored",
        "actual_report",
        "actual_report_sha256",
        "execution",
        "report_sha256",
    }
)
_MACROCELL_EVALUATION_FIELDS = frozenset(
    {
        "schema",
        "candidate_id",
        "candidate_state_sha256",
        "operator_selection_report_sha256",
        "fit_provenance_report_sha256",
        "fit_bank_sha256",
        "selection_bank_sha256",
        "operator_install_report",
        "operator_install_report_sha256",
        "fit_operator_replay_report",
        "fit_operator_replay_sha256",
        "fit_mlp_artifact_report",
        "fit_mlp_artifact_report_sha256",
        "fit_mlp_compiled_state_sha256",
        "fit_mlp_install_report",
        "fit_mlp_install_report_sha256",
        "selection_final_replay_report",
        "selection_final_replay_sha256",
        "selection_metrics",
        "selection_metrics_report_sha256",
        "fit_only_mlp_fold",
        "selection_only_candidate_evaluation",
        "validation_accessible",
        "validation_replays",
        "rollback_proof",
        "rollback_proof_sha256",
        "execution",
        "report_sha256",
    }
)
_MACROCELL_DECISION_FIELDS = frozenset(
    {
        "schema",
        "kind",
        "frontier_sha256",
        "candidate_count",
        "candidate_budget",
        "evaluated_candidate_ids",
        "candidate_evaluations",
        "fit_bank_sha256",
        "selection_bank_sha256",
        "macrocell_ranking",
        "ranking_policy",
        "operator_endpoint_first_control_included",
        "selected_candidate_id",
        "selected_candidate_state_sha256",
        "selection_split_only_for_ranking",
        "validation_accessible",
        "validation_reports_consumed",
        "validation_replays",
        "candidate_specific_fit_mlp_folds",
        "candidate_specific_selection_replays",
        "exact_candidate_rollbacks",
        "execution",
        "report_sha256",
    }
)
_ZERO_EXECUTION_COUNTER_FIELDS = frozenset(_ZERO_EXECUTION) | {
    "parent_forward_count",
    "child_model_forward_count",
    "target_layer_forward_count",
    "cache_generation_count",
    "cache_update_count",
    "training_step_count",
    "backward_call_count",
    "optimizer_step_count",
    "model_forward_calls",
}


def _exact_zero_execution(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and set(value) == set(_ZERO_EXECUTION)
        and all(type(value[name]) is int and value[name] == 0 for name in value)
    )


def _recursive_zero_execution_claims(value: Any) -> bool:
    if isinstance(value, Mapping):
        for name, child in value.items():
            if name in _ZERO_EXECUTION_COUNTER_FIELDS and (
                type(child) is not int or child != 0
            ):
                return False
            if not _recursive_zero_execution_claims(child):
                return False
        return True
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return all(_recursive_zero_execution_claims(child) for child in value)
    return True


def _verified_macrocell_evidence_document(
    value: Any,
    *,
    kind: str,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2CpcfShortConvV4FoldError(
            f"Macrocell {name} must be an embedded evidence document."
        )
    report = _canonical_copy(_deep_thaw(value), f"macrocell {name}")
    body = dict(report)
    supplied = body.pop("report_sha256", None)
    actual_value = report.get("actual_report")
    actual_report = actual_value if isinstance(actual_value, Mapping) else {}
    actual_body = dict(actual_report)
    actual_seal = actual_body.pop("report_sha256", None)
    if (
        set(report) != _MACROCELL_EVIDENCE_FIELDS[kind]
        or report.get("schema") != LFM2_CPCF_SHORT_CONV_V5_MACROCELL_EVIDENCE_SCHEMA
        or report.get("kind") != kind
        or not _valid_sha256_text(supplied)
        or supplied != _sha(body)
        or not _valid_sha256_text(actual_seal)
        or actual_seal != _sha(actual_body)
        or report.get("actual_report_sha256") != actual_seal
        or not _recursive_canonical_report_seals_valid(report)
        or not _recursive_zero_execution_claims(report)
        or not _exact_zero_execution(report.get("execution"))
    ):
        raise Lfm2CpcfShortConvV4FoldError(
            f"Macrocell {name} is not an exact, recursively sealed, "
            "zero-execution evidence document."
        )
    return report


def _verified_macrocell_rollback_document(
    value: Any,
    *,
    candidate_id: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2CpcfShortConvV4FoldError(
            "Macrocell rollback proof must be an embedded evidence document."
        )
    report = _canonical_copy(_deep_thaw(value), "macrocell rollback proof")
    body = dict(report)
    supplied = body.pop("report_sha256", None)
    actual_value = report.get("actual_report")
    actual_report = actual_value if isinstance(actual_value, Mapping) else {}
    actual_body = dict(actual_report)
    actual_seal = actual_body.pop("report_sha256", None)
    gradient_restored = actual_report.get(
        "target_gradient_state_preserved",
        actual_report.get(
            "target_parameter_gradient_identity_and_bytes_restored",
        ),
    )
    exact_restored = actual_report.get(
        "exact_layer_snapshot_restored",
        actual_report.get("exact_layer_rollback_proven"),
    )
    if (
        set(report) != _MACROCELL_ROLLBACK_FIELDS
        or report.get("schema") != LFM2_CPCF_SHORT_CONV_V5_MACROCELL_EVIDENCE_SCHEMA
        or report.get("kind") != "actual_candidate_rollback"
        or report.get("candidate_id") != candidate_id
        or not _valid_sha256_text(supplied)
        or supplied != _sha(body)
        or not _valid_sha256_text(actual_seal)
        or actual_seal != _sha(actual_body)
        or report.get("actual_report_sha256") != actual_seal
        or not _recursive_canonical_report_seals_valid(report)
        or not _recursive_zero_execution_claims(report)
        or not _exact_zero_execution(report.get("execution"))
        or report.get("exact_layer_snapshot_restored") is not True
        or report.get("target_tensor_bytes_restored") is not True
        or report.get("target_tensor_identities_preserved") is not True
        or report.get("target_gradient_state_preserved") is not True
        or report.get("target_module_modes_restored") is not True
        or report.get("target_state_sha256_before")
        != report.get("target_state_sha256_after")
        or not _valid_sha256_text(report.get("target_state_sha256_before"))
        or actual_report.get("target_state_sha256_before")
        != report["target_state_sha256_before"]
        or actual_report.get("target_state_sha256_after")
        != report["target_state_sha256_after"]
        or actual_report.get("target_tensor_bytes_restored") is not True
        or actual_report.get("target_tensor_identities_preserved") is not True
        or gradient_restored is not True
        or actual_report.get("target_module_modes_restored") is not True
        or exact_restored is not True
    ):
        raise Lfm2CpcfShortConvV4FoldError(
            "Macrocell rollback proof is not exact, recursively sealed, "
            "and state-preserving."
        )
    return report


def _macrocell_metric_summary(value: Any) -> dict[str, Any] | None:
    if (
        not isinstance(value, Mapping)
        or set(value) != _MACROCELL_ENDPOINT_METRIC_FIELDS
    ):
        return None
    return dict(value)


def _finite_macrocell_metric(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Lfm2CpcfShortConvV4FoldError(f"Macrocell {name} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise Lfm2CpcfShortConvV4FoldError(f"Macrocell {name} must be a finite number.")
    return result


def _macrocell_selection_score(
    metrics: Mapping[str, Any],
    *,
    candidate_id: str,
    selection_bank_sha256: str,
) -> tuple[float | str, ...]:
    """Recompute the causal post-layer score from a sealed selection report."""

    report = (
        _canonical_copy(_deep_thaw(metrics), "macrocell selection metrics")
        if isinstance(metrics, Mapping)
        else {}
    )
    body = dict(report)
    supplied = body.pop("report_sha256", None)
    actual_value = report.get("actual_report")
    actual_report = actual_value if isinstance(actual_value, Mapping) else {}
    actual_body = dict(actual_report)
    actual_seal = actual_body.pop("report_sha256", None)
    if (
        set(report) != _MACROCELL_METRICS_FIELDS
        or report.get("schema") != LFM2_CPCF_SHORT_CONV_V5_MACROCELL_EVIDENCE_SCHEMA
        or report.get("kind") != "actual_selection_metrics"
        or report.get("candidate_id") != candidate_id
        or report.get("selection_bank_sha256") != selection_bank_sha256
        or report.get("split") != "selection"
        or not _valid_sha256_text(supplied)
        or supplied != _sha(body)
        or not _valid_sha256_text(actual_seal)
        or actual_seal != _sha(actual_body)
        or report.get("actual_report_sha256") != actual_seal
        or not _recursive_canonical_report_seals_valid(report)
        or not _recursive_zero_execution_claims(report)
        or not _exact_zero_execution(report.get("execution"))
        or not isinstance(report.get("phase_metrics"), Mapping)
        or not report["phase_metrics"]
        or type(report.get("phase_count")) is not int
        or report["phase_count"] <= 0
        or not isinstance(report.get("phase_sha256"), list)
        or len(report["phase_sha256"]) != report["phase_count"]
        or not all(_valid_sha256_text(value) for value in report["phase_sha256"])
        or _macrocell_metric_summary(report.get("post_operator")) is None
        or _macrocell_metric_summary(report.get("post_layer")) is None
        or actual_report.get("split") != report["split"]
        or actual_report.get("phase_count") != report["phase_count"]
        or actual_report.get("phase_sha256") != report["phase_sha256"]
        or not isinstance(actual_report.get("phase_metrics"), Mapping)
        or set(actual_report["phase_metrics"]) != set(report["phase_metrics"])
        or any(
            not isinstance(actual_report.get(endpoint), Mapping)
            or any(
                actual_report[endpoint].get(name) != metric
                for name, metric in report[endpoint].items()
            )
            for endpoint in ("post_operator", "post_layer")
        )
    ):
        raise Lfm2CpcfShortConvV4FoldError(
            "Macrocell selection metrics are not a sealed selection-only "
            "causal endpoint report."
        )
    phase_relative_rmse = []
    phase_cosine = []
    for phase_name, phase_report in report["phase_metrics"].items():
        actual_phase = actual_report["phase_metrics"].get(phase_name)
        if (
            not isinstance(phase_name, str)
            or not phase_name
            or not isinstance(phase_report, Mapping)
            or set(phase_report) != _MACROCELL_PHASE_METRICS_FIELDS
            or phase_report.get("phase") != phase_name
            or type(phase_report.get("phase_count")) is not int
            or phase_report["phase_count"] <= 0
            or not isinstance(phase_report.get("phase_sha256"), list)
            or len(phase_report["phase_sha256"]) != phase_report["phase_count"]
            or not all(
                _valid_sha256_text(value) for value in phase_report["phase_sha256"]
            )
            or _macrocell_metric_summary(phase_report.get("post_operator")) is None
            or _macrocell_metric_summary(phase_report.get("post_layer")) is None
            or not _recursive_canonical_report_seals_valid(phase_report)
            or not isinstance(actual_phase, Mapping)
            or actual_phase.get("phase") != phase_report["phase"]
            or actual_phase.get("phase_count") != phase_report["phase_count"]
            or actual_phase.get("phase_sha256") != phase_report["phase_sha256"]
            or any(
                not isinstance(actual_phase.get(endpoint), Mapping)
                or any(
                    actual_phase[endpoint].get(name) != metric
                    for name, metric in phase_report[endpoint].items()
                )
                for endpoint in ("post_operator", "post_layer")
            )
        ):
            raise Lfm2CpcfShortConvV4FoldError(
                "Macrocell causal phase metrics are incomplete."
            )
        post_layer = phase_report["post_layer"]
        relative_rmse = _finite_macrocell_metric(
            post_layer.get("relative_rmse"),
            f"{phase_name} post-layer relative RMSE",
        )
        cosine = _finite_macrocell_metric(
            post_layer.get("cosine_similarity"),
            f"{phase_name} post-layer cosine",
        )
        if relative_rmse < 0.0 or not -1.0 <= cosine <= 1.0:
            raise Lfm2CpcfShortConvV4FoldError(
                "Macrocell causal phase metrics are outside their valid range."
            )
        phase_relative_rmse.append(relative_rmse)
        phase_cosine.append(cosine)
    aggregate = report["post_layer"]
    aggregate_relative_rmse = _finite_macrocell_metric(
        aggregate.get("relative_rmse"),
        "aggregate post-layer relative RMSE",
    )
    aggregate_cosine = _finite_macrocell_metric(
        aggregate.get("cosine_similarity"),
        "aggregate post-layer cosine",
    )
    if aggregate_relative_rmse < 0.0 or not -1.0 <= aggregate_cosine <= 1.0:
        raise Lfm2CpcfShortConvV4FoldError(
            "Macrocell aggregate metrics are outside their valid range."
        )
    return (
        max(phase_relative_rmse),
        -min(phase_cosine),
        aggregate_relative_rmse,
        -aggregate_cosine,
        candidate_id,
    )


def _verified_macrocell_selector_decision(
    frontier: Lfm2CpcfShortConvV5CandidateFrontier,
    decision_value: Any,
) -> tuple[str, dict[str, Any]]:
    """Authenticate candidate-specific MLP/replay evidence and ranking."""

    decision = _canonical_copy(
        _deep_thaw(decision_value),
        "macrocell selector decision",
    )
    candidate_by_id = {
        candidate.candidate_id: candidate for candidate in frontier.candidates
    }
    candidate_count = len(candidate_by_id)
    evaluations = decision.get("candidate_evaluations")
    evaluated_ids = decision.get("evaluated_candidate_ids")
    budget = decision.get("candidate_budget")
    execution = decision.get("execution")
    if (
        set(decision) != _MACROCELL_DECISION_FIELDS
        or not verify_lfm2_cpcf_short_conv_v5_report(decision)
        or not _recursive_canonical_report_seals_valid(decision)
        or not _recursive_zero_execution_claims(decision)
        or decision.get("schema") != LFM2_CPCF_SHORT_CONV_V5_SCHEMA
        or decision.get("kind") != "authenticated_fit_selection_whole_macrocell_ranking"
        or decision.get("frontier_sha256") != frontier.frontier_sha256
        or type(decision.get("candidate_count")) is not int
        or decision.get("candidate_count") != candidate_count
        or type(budget) is not int
        or not 1 <= budget <= candidate_count
        or not isinstance(evaluated_ids, list)
        or not evaluated_ids
        or not all(isinstance(value, str) and value for value in evaluated_ids)
        or len(evaluated_ids) != len(set(evaluated_ids))
        or evaluated_ids != sorted(evaluated_ids)
        or not isinstance(evaluations, Mapping)
        or sorted(evaluations) != evaluated_ids
        or decision.get("selection_split_only_for_ranking") is not True
        or decision.get("validation_accessible") is not False
        or type(decision.get("validation_reports_consumed")) is not int
        or decision["validation_reports_consumed"] != 0
        or type(decision.get("validation_replays")) is not int
        or decision["validation_replays"] != 0
        or decision.get("ranking_policy")
        != (
            "minimum_worst_causal_phase_post_layer_relative_rmse_then_"
            "maximum_minimum_causal_phase_post_layer_cosine_then_minimum_"
            "aggregate_post_layer_relative_rmse_then_maximum_aggregate_"
            "post_layer_cosine_then_candidate_id"
        )
        or decision.get("operator_endpoint_first_control_included") is not True
        or not _exact_zero_execution(execution)
    ):
        raise Lfm2CpcfShortConvV4FoldError(
            "Macrocell selector decision lacks authenticated fit/selection "
            "candidate evidence."
        )
    expected_evaluated = list(
        select_lfm2_cpcf_short_conv_v5_macrocell_candidate_ids(
            frontier,
            budget,
        )
    )
    if (
        evaluated_ids != expected_evaluated
        or frontier.default_ranking["selected_candidate_id"] not in evaluated_ids
    ):
        raise Lfm2CpcfShortConvV4FoldError(
            "Macrocell selector did not evaluate the deterministic structural "
            "frontier inventory and its endpoint control."
        )
    fit_bank_sha256 = decision.get("fit_bank_sha256")
    selection_bank_sha256 = decision.get("selection_bank_sha256")
    if (
        fit_bank_sha256 != frontier.fit_bank_sha256
        or selection_bank_sha256 != frontier.selection_bank_sha256
    ):
        raise Lfm2CpcfShortConvV4FoldError(
            "Macrocell selector fit and selection bank identities differ "
            "from its authenticated frontier."
        )

    recomputed = []
    for candidate_id in evaluated_ids:
        evaluation_value = evaluations[candidate_id]
        candidate = candidate_by_id.get(candidate_id)
        if not isinstance(evaluation_value, Mapping) or candidate is None:
            raise Lfm2CpcfShortConvV4FoldError(
                "Macrocell selector evaluation inventory is incomplete."
            )
        evaluation = _canonical_copy(
            _deep_thaw(evaluation_value),
            f"macrocell selector evaluation {candidate_id}",
        )
        body = dict(evaluation)
        seal = body.pop("report_sha256", None)
        operator_install = _verified_macrocell_evidence_document(
            evaluation.get("operator_install_report"),
            kind="actual_operator_install",
            name=f"{candidate_id} operator install",
        )
        fit_replay = _verified_macrocell_evidence_document(
            evaluation.get("fit_operator_replay_report"),
            kind="actual_fit_operator_replay",
            name=f"{candidate_id} fit operator replay",
        )
        mlp_artifact = _verified_macrocell_evidence_document(
            evaluation.get("fit_mlp_artifact_report"),
            kind="actual_fit_mlp_artifact",
            name=f"{candidate_id} fit MLP artifact",
        )
        mlp_install = _verified_macrocell_evidence_document(
            evaluation.get("fit_mlp_install_report"),
            kind="actual_fit_mlp_install",
            name=f"{candidate_id} fit MLP install",
        )
        selection_replay = _verified_macrocell_evidence_document(
            evaluation.get("selection_final_replay_report"),
            kind="actual_selection_final_replay",
            name=f"{candidate_id} selection final replay",
        )
        rollback = _verified_macrocell_rollback_document(
            evaluation.get("rollback_proof"),
            candidate_id=candidate_id,
        )
        mlp_state_sha256 = evaluation.get("fit_mlp_compiled_state_sha256")
        operator_actual = operator_install["actual_report"]
        fit_replay_actual = fit_replay["actual_report"]
        mlp_artifact_actual = mlp_artifact["actual_report"]
        mlp_install_actual = mlp_install["actual_report"]
        selection_replay_actual = selection_replay["actual_report"]
        candidate_install_state_sha256 = candidate.report[
            "candidate_install_state_sha256"
        ]
        selection_metrics = evaluation.get("selection_metrics")
        selection_metrics_sha256 = (
            selection_metrics.get("report_sha256")
            if isinstance(selection_metrics, Mapping)
            else None
        )
        if (
            set(evaluation) != _MACROCELL_EVALUATION_FIELDS
            or not _valid_sha256_text(seal)
            or seal != _sha(body)
            or not _recursive_canonical_report_seals_valid(evaluation)
            or not _recursive_zero_execution_claims(evaluation)
            or evaluation.get("schema")
            != LFM2_CPCF_SHORT_CONV_V5_MACROCELL_EVALUATION_SCHEMA
            or evaluation.get("candidate_id") != candidate_id
            or evaluation.get("candidate_state_sha256") != candidate.state_sha256
            or evaluation.get("operator_selection_report_sha256")
            != candidate.operator_selection_report["report_sha256"]
            or evaluation.get("fit_provenance_report_sha256")
            != candidate.fit_provenance["report_sha256"]
            or evaluation.get("fit_bank_sha256") != fit_bank_sha256
            or evaluation.get("selection_bank_sha256") != selection_bank_sha256
            or evaluation.get("operator_install_report_sha256")
            != operator_install["report_sha256"]
            or evaluation.get("fit_operator_replay_sha256")
            != fit_replay["report_sha256"]
            or evaluation.get("fit_mlp_artifact_report_sha256")
            != mlp_artifact["report_sha256"]
            or not _valid_sha256_text(mlp_state_sha256)
            or evaluation.get("fit_mlp_install_report_sha256")
            != mlp_install["report_sha256"]
            or evaluation.get("selection_final_replay_sha256")
            != selection_replay["report_sha256"]
            or evaluation.get("selection_metrics_report_sha256")
            != selection_metrics_sha256
            or evaluation.get("rollback_proof_sha256") != rollback["report_sha256"]
            or operator_install.get("candidate_id") != candidate_id
            or operator_install.get("candidate_state_sha256") != candidate.state_sha256
            or operator_install.get("installed_state_sha256")
            != candidate_install_state_sha256
            or operator_actual.get("compiled_state_sha256")
            != candidate_install_state_sha256
            or operator_actual.get("installed_state_sha256")
            != candidate_install_state_sha256
            or fit_replay.get("candidate_id") != candidate_id
            or fit_replay.get("candidate_state_sha256") != candidate.state_sha256
            or fit_replay.get("input_bank_sha256") != fit_bank_sha256
            or fit_replay_actual.get("input_bank_sha256") != fit_bank_sha256
            or mlp_artifact.get("candidate_id") != candidate_id
            or mlp_artifact.get("fit_bank_sha256") != fit_bank_sha256
            or mlp_artifact.get("compiled_state_sha256") != mlp_state_sha256
            or mlp_artifact_actual.get("compiled_state_sha256") != mlp_state_sha256
            or mlp_install.get("candidate_id") != candidate_id
            or mlp_install.get("compiled_state_sha256") != mlp_state_sha256
            or mlp_install.get("installed_state_sha256") != mlp_state_sha256
            or mlp_install_actual.get("compiled_state_sha256") != mlp_state_sha256
            or mlp_install_actual.get("installed_state_sha256") != mlp_state_sha256
            or selection_replay.get("candidate_id") != candidate_id
            or selection_replay.get("candidate_state_sha256") != candidate.state_sha256
            or selection_replay.get("input_bank_sha256") != selection_bank_sha256
            or selection_replay.get("mlp_compiled_state_sha256") != mlp_state_sha256
            or selection_replay_actual.get("input_bank_sha256") != selection_bank_sha256
            or evaluation.get("fit_only_mlp_fold") is not True
            or evaluation.get("selection_only_candidate_evaluation") is not True
            or evaluation.get("validation_accessible") is not False
            or type(evaluation.get("validation_replays")) is not int
            or evaluation["validation_replays"] != 0
            or not _exact_zero_execution(evaluation.get("execution"))
        ):
            raise Lfm2CpcfShortConvV4FoldError(
                "Macrocell selector candidate evaluation is not state-bound, "
                "fit/selection-only, and exactly rolled back."
            )
        score = _macrocell_selection_score(
            selection_metrics,
            candidate_id=candidate_id,
            selection_bank_sha256=selection_bank_sha256,
        )
        recomputed.append(
            {
                "candidate_id": candidate_id,
                "score": list(score),
                "evaluation_report_sha256": seal,
            }
        )
    recomputed.sort(
        key=lambda value: tuple(value["score"]),
    )
    expected_ranking = [
        {
            "rank": index + 1,
            **value,
        }
        for index, value in enumerate(recomputed)
    ]
    selected_id = expected_ranking[0]["candidate_id"]
    selected_state_sha256 = decision.get("selected_candidate_state_sha256")
    if (
        decision.get("macrocell_ranking") != expected_ranking
        or decision.get("selected_candidate_id") != selected_id
        or selected_state_sha256 != candidate_by_id[selected_id].state_sha256
        or type(decision.get("candidate_specific_fit_mlp_folds")) is not int
        or decision.get("candidate_specific_fit_mlp_folds") != len(evaluated_ids)
        or type(decision.get("candidate_specific_selection_replays")) is not int
        or decision.get("candidate_specific_selection_replays") != len(evaluated_ids)
        or type(decision.get("exact_candidate_rollbacks")) is not int
        or decision.get("exact_candidate_rollbacks") != len(evaluated_ids)
    ):
        raise Lfm2CpcfShortConvV4FoldError(
            "Macrocell selector ranking does not recompute from every "
            "candidate-specific MLP and causal replay."
        )
    return selected_id, decision


def _run_preinstall_macrocell_candidate_selector(
    selector: Lfm2CpcfShortConvV5MacrocellCandidateSelector,
    *,
    candidate_states: Mapping[str, Mapping[str, Any]],
    candidate_fit_reports: Mapping[str, Mapping[str, Any]],
    selection_reports: Mapping[str, Mapping[str, Any]],
    ranking: Mapping[str, Any],
    solver_policy: Lfm2CpcfShortConvV4SolverPolicy,
    fit_bank_sha256: str,
    selection_bank_sha256: str,
) -> tuple[str, Mapping[str, Any]]:
    """Expose detached states for fit/selection-only whole-macrocell replay."""

    state_sha256_before = _candidate_state_sha256(candidate_states)
    candidate_ids = sorted(candidate_states)
    if (
        candidate_ids != sorted(candidate_fit_reports)
        or candidate_ids != sorted(selection_reports)
        or any(
            selection_reports[candidate_id].get("candidate_state_sha256")
            != state_sha256_before[candidate_id]
            or candidate_fit_reports[candidate_id].get(
                "target_dtype_candidate_state_sha256"
            )
            != state_sha256_before[candidate_id]
            for candidate_id in candidate_ids
        )
    ):
        raise Lfm2CpcfShortConvV4FoldError(
            "Macrocell selector frontier is not bound to live candidates."
        )
    candidates = tuple(
        Lfm2CpcfShortConvV5MacrocellCandidate(
            candidate_id=candidate_id,
            in_proj_weight=candidate_states[candidate_id]["in_proj_weight"],
            in_proj_bias=candidate_states[candidate_id]["in_proj_bias"],
            fir_kernel=candidate_states[candidate_id]["fir_kernel"],
            fir_bias=candidate_states[candidate_id]["fir_bias"],
            writer_weight=candidate_states[candidate_id]["writer_weight"],
            writer_bias=candidate_states[candidate_id]["writer_bias"],
            fit_provenance=candidate_fit_reports[candidate_id],
            operator_selection_report=selection_reports[candidate_id],
        )
        for candidate_id in candidate_ids
    )
    frontier = Lfm2CpcfShortConvV5CandidateFrontier(
        candidates=candidates,
        solver_policy_sha256=solver_policy.authentication_sha256,
        target_dtype=solver_policy.target_dtype,
        fit_bank_sha256=fit_bank_sha256,
        selection_bank_sha256=selection_bank_sha256,
        default_ranking=ranking,
    )
    frontier_sha256 = frontier.frontier_sha256
    try:
        decision_value = selector.select(frontier)
    except BaseException as error:
        if _candidate_state_sha256(candidate_states) != state_sha256_before:
            raise Lfm2CpcfShortConvV4FoldError(
                "Macrocell selector changed a live retained candidate before failing."
            ) from error
        if not frontier.verify_authentication():
            raise Lfm2CpcfShortConvV4FoldError(
                "Macrocell selector changed its detached frontier before failing."
            ) from error
        raise
    if _candidate_state_sha256(candidate_states) != state_sha256_before:
        raise Lfm2CpcfShortConvV4FoldError(
            "Macrocell selector changed a live retained candidate."
        )
    if not frontier.verify_authentication() or (
        frontier.frontier_sha256 != frontier_sha256
    ):
        raise Lfm2CpcfShortConvV4FoldError(
            "Macrocell selector changed its detached frontier."
        )
    selected_id, decision = _verified_macrocell_selector_decision(
        frontier,
        decision_value,
    )
    selected_state_sha256 = decision.get("selected_candidate_state_sha256")
    default_selected_id = str(ranking["selected_candidate_id"])
    receipt_body = {
        "schema": LFM2_CPCF_SHORT_CONV_V5_MACROCELL_SELECTOR_RECEIPT_SCHEMA,
        "frontier": frontier.as_dict(),
        "frontier_sha256": frontier_sha256,
        "selector_decision": decision,
        "selector_decision_sha256": decision["report_sha256"],
        "candidate_state_inventory_sha256": _sha(state_sha256_before),
        "candidate_count": len(candidate_ids),
        "default_v4_selected_candidate_id": default_selected_id,
        "selected_candidate_id": selected_id,
        "selected_candidate_state_sha256": selected_state_sha256,
        "selection_changed_from_v4": selected_id != default_selected_id,
        "all_live_candidate_states_verified_before_callback": True,
        "all_live_candidate_states_verified_after_callback": True,
        "detached_frontier_verified_after_callback": True,
        "candidate_state_tensors_exposed_to_macrocell_selector": True,
        "fit_provenance_exposed_to_macrocell_selector": True,
        "operator_selection_reports_exposed_to_macrocell_selector": True,
        "validation_evidence_exposed_to_macrocell_selector": False,
        "validation_reports_consumed": 0,
        "post_install_rescoring": False,
        "execution": dict(_ZERO_EXECUTION),
    }
    return str(selected_id), MappingProxyType(_seal_report(receipt_body))


def _run_preinstall_candidate_selector(
    selector: Lfm2CpcfShortConvV5CandidateSelector,
    *,
    candidate_states: Mapping[str, Mapping[str, Any]],
    selection_reports: Mapping[str, Mapping[str, Any]],
    ranking: Mapping[str, Any],
    solver_policy: Lfm2CpcfShortConvV4SolverPolicy,
) -> tuple[str, Mapping[str, Any]]:
    """Run one tensor-blind selector while retained candidate states exist."""

    state_sha256_before = _candidate_state_sha256(candidate_states)
    candidate_ids = sorted(candidate_states)
    if candidate_ids != sorted(selection_reports) or any(
        selection_reports[candidate_id].get("candidate_state_sha256")
        != state_sha256_before[candidate_id]
        for candidate_id in candidate_ids
    ):
        raise Lfm2CpcfShortConvV4FoldError(
            "Pre-install selector frontier is not bound to live candidates."
        )
    selection_report_copies = {
        candidate_id: _canonical_copy(
            _deep_thaw(selection_reports[candidate_id]),
            f"selection report {candidate_id}",
        )
        for candidate_id in candidate_ids
    }
    default_selected_id = str(ranking["selected_candidate_id"])
    request_body = {
        "schema": LFM2_CPCF_SHORT_CONV_V5_SELECTOR_REQUEST_SCHEMA,
        "solver_policy_sha256": solver_policy.authentication_sha256,
        "target_dtype": solver_policy.target_dtype,
        "candidate_ids": candidate_ids,
        "candidate_state_sha256": state_sha256_before,
        "candidate_state_inventory_sha256": _sha(state_sha256_before),
        "candidate_selection_report_sha256": {
            candidate_id: selection_report_copies[candidate_id]["report_sha256"]
            for candidate_id in candidate_ids
        },
        "candidate_selection_reports": selection_report_copies,
        "default_v4_ranking_report_sha256": ranking["report_sha256"],
        "default_v4_selected_candidate_id": default_selected_id,
        "candidate_state_tensors_exposed_to_selector": False,
        "fit_reports_exposed_to_selector": False,
        "validation_reports_exposed_to_selector": False,
        "post_install_rescoring": False,
        "execution": {
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
        },
    }
    request = _seal_report(request_body)
    request_bytes = _canonical_json(request)
    try:
        selector_report_value = selector.select(request)
    except BaseException as error:
        if _candidate_state_sha256(candidate_states) != state_sha256_before:
            raise Lfm2CpcfShortConvV4FoldError(
                "Pre-install selector changed a retained candidate state "
                "before failing."
            ) from error
        if _canonical_json(request) != request_bytes:
            raise Lfm2CpcfShortConvV4FoldError(
                "Pre-install selector changed its authenticated request before failing."
            ) from error
        raise
    state_sha256_after = _candidate_state_sha256(candidate_states)
    if state_sha256_after != state_sha256_before:
        raise Lfm2CpcfShortConvV4FoldError(
            "Pre-install selector changed a retained candidate state."
        )
    if _canonical_json(request) != request_bytes:
        raise Lfm2CpcfShortConvV4FoldError(
            "Pre-install selector changed its authenticated request."
        )
    selector_report = _canonical_copy(
        _deep_thaw(selector_report_value),
        "pre-install selector report",
    )
    selected_id = selector_report.get("selected_candidate_id")
    selected_state_sha256 = selector_report.get("selected_candidate_state_sha256")
    execution = selector_report.get("execution")
    if (
        not verify_lfm2_cpcf_short_conv_v5_report(selector_report)
        or selector_report.get("schema") != LFM2_CPCF_SHORT_CONV_V5_SCHEMA
        or not isinstance(selector_report.get("kind"), str)
        or not selector_report["kind"]
        or not isinstance(selected_id, str)
        or selected_id not in state_sha256_before
        or selected_state_sha256 != state_sha256_before[selected_id]
        or selector_report.get("candidate_count") != len(candidate_ids)
        or selector_report.get("selection_reports_consumed") != len(candidate_ids)
        or selector_report.get("fit_reports_consumed") != 0
        or selector_report.get("validation_reports_consumed") != 0
        or selector_report.get("validation_accessible") is not False
        or not isinstance(execution, Mapping)
        or any(
            execution.get(name) != 0
            for name in (
                "training_steps",
                "backward_calls",
                "optimizer_steps",
                "parameters_pruned",
                "parameters_quantized",
            )
        )
    ):
        raise Lfm2CpcfShortConvV4FoldError(
            "Pre-install selector did not return a sealed, state-bound v5 "
            "selection report."
        )
    receipt_body = {
        "schema": LFM2_CPCF_SHORT_CONV_V5_SELECTOR_RECEIPT_SCHEMA,
        "request_sha256": request["report_sha256"],
        "selector_report": selector_report,
        "selector_report_sha256": selector_report["report_sha256"],
        "candidate_state_inventory_sha256": _sha(state_sha256_before),
        "candidate_count": len(candidate_ids),
        "default_v4_selected_candidate_id": default_selected_id,
        "selected_candidate_id": selected_id,
        "selected_candidate_state_sha256": selected_state_sha256,
        "selection_changed_from_v4": selected_id != default_selected_id,
        "all_live_candidate_states_verified_before_callback": True,
        "all_live_candidate_states_verified_after_callback": True,
        "live_candidate_state_bytes_unchanged": True,
        "candidate_state_tensors_exposed_to_selector": False,
        "fit_reports_exposed_to_selector": False,
        "validation_reports_exposed_to_selector": False,
        "post_install_rescoring": False,
        "execution": {
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
        },
    }
    return str(selected_id), MappingProxyType(_seal_report(receipt_body))


def _selection_feasibility_policy(
    gate_policy: Lfm2CpcfShortConvGatePolicy,
) -> dict[str, float]:
    return {
        "maximum_endpoint_relative_rmse": min(
            gate_policy.maximum_fit_endpoint_relative_rmse,
            gate_policy.maximum_validation_endpoint_relative_rmse,
        ),
        "maximum_context_relative_rmse": min(
            gate_policy.maximum_fit_endpoint_relative_rmse,
            gate_policy.maximum_validation_endpoint_relative_rmse,
        ),
        "minimum_endpoint_cosine_similarity": max(
            gate_policy.minimum_fit_endpoint_cosine_similarity,
            gate_policy.minimum_validation_endpoint_cosine_similarity,
        ),
        "minimum_context_cosine_similarity": max(
            gate_policy.minimum_fit_endpoint_cosine_similarity,
            gate_policy.minimum_validation_endpoint_cosine_similarity,
        ),
    }


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
            or report.get("schema")
            not in {
                LFM2_CPCF_SHORT_CONV_V4_FOLD_SCHEMA,
                LFM2_CPCF_SHORT_CONV_V5_FOLD_SCHEMA,
                LFM2_CPCF_SHORT_CONV_V5_SELECTOR_FOLD_SCHEMA,
                LFM2_CPCF_SHORT_CONV_V5_MACROCELL_SELECTOR_FOLD_SCHEMA,
                LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_FOLD_SCHEMA,
            }
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


def _generate_legacy_v4_raw_candidates(
    *,
    ordered: Sequence[Lfm2CpcfShortConvV4FoldCase],
    parent_contexts: Mapping[str, Any],
    source_in_proj_weight: Any,
    source_in_proj_bias: Any | None,
    source_kernel: Any,
    source_fir_bias: Any | None,
    source_out_proj_weight: Any,
    source_out_proj_bias: Any | None,
    output_encoder: Any,
    target_hidden: int,
    target_fir_length: int,
    in_bias_template: Any | None,
    conv_bias_template: Any | None,
    out_bias_template: Any | None,
    solver_policy: Lfm2CpcfShortConvV4SolverPolicy,
) -> tuple[
    Any,
    Mapping[str, Any],
    Any,
    Mapping[str, tuple[Lfm2CpcfV4ProjectionCase, ...]],
    Mapping[str, Any],
    list[dict[str, Any]],
]:
    """Run the byte-stable legacy basis, prior, and fit-path generator."""

    torch = _torch()
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
        target_fir_length=target_fir_length,
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
    ridge_results = {}
    raw_candidates = []
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
        ridge_id = f"ridge-alpha-{index:02d}-{alpha:.17g}"
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
        if not result.verify_authentication():
            raise Lfm2CpcfShortConvV4FoldError(
                f"{ridge_id} variable-projection result failed authentication."
            )
        ridge_results[ridge_id] = result
        for path_index, path_state in enumerate(result.path_states):
            raw_candidates.append(
                {
                    "candidate_id": (
                        f"{ridge_id}--anchor-{path_index:02d}-{path_state.stage_id}"
                    ),
                    "ridge_id": ridge_id,
                    "candidate": path_state,
                    "candidate_kind": "accepted_fit_path_anchor",
                }
            )
        prior_state = result.path_states[0]
        final_state = result.path_states[-1]
        first_writer_state = next(
            (
                state
                for state in result.path_states
                if state.stage_id.endswith("-writer")
            ),
            None,
        )
        for fraction_index, fraction in enumerate(
            fixed_projection_trust_path_fractions()
        ):
            interpolated = interpolate_short_conv_v4_candidate_state(
                left=prior_state,
                right=final_state,
                fraction=fraction,
                stage_id=(f"prior-to-final-{fraction_index:02d}-{fraction:.17g}"),
            )
            raw_candidates.append(
                {
                    "candidate_id": (
                        f"{ridge_id}--trust-{fraction_index:02d}-{fraction:.17g}"
                    ),
                    "ridge_id": ridge_id,
                    "candidate": interpolated,
                    "candidate_kind": "prior_to_final_trust_path",
                }
            )
            if first_writer_state is not None:
                writer_interpolated = interpolate_short_conv_v4_candidate_state(
                    left=prior_state,
                    right=first_writer_state,
                    fraction=fraction,
                    stage_id=(
                        f"prior-to-first-writer-{fraction_index:02d}-{fraction:.17g}"
                    ),
                )
                raw_candidates.append(
                    {
                        "candidate_id": (
                            f"{ridge_id}--writer-trust-"
                            f"{fraction_index:02d}-{fraction:.17g}"
                        ),
                        "ridge_id": ridge_id,
                        "candidate": writer_interpolated,
                        "candidate_kind": ("prior_to_first_writer_trust_path"),
                    }
                )
    return (
        basis,
        lift_report,
        priors,
        MappingProxyType(split_cases),
        MappingProxyType(ridge_results),
        raw_candidates,
    )


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
    candidate_replay_evaluator: Lfm2CpcfCandidateReplayEvaluator | None = None,
    candidate_selector: Lfm2CpcfShortConvV5CandidateSelector | None = None,
    macrocell_candidate_selector: (
        Lfm2CpcfShortConvV5MacrocellCandidateSelector | None
    ) = None,
    macrocell_fit_bank_sha256: str | None = None,
    macrocell_selection_bank_sha256: str | None = None,
    numerical_frontier_adapter: (
        Lfm2CpcfShortConvNumericalFrontierAdapter | None
    ) = None,
) -> Lfm2CompiledCpcfShortConvV4Fold:
    """Compile one complete v4 operator without executing either model."""

    torch = _torch()
    if not isinstance(solver_policy, Lfm2CpcfShortConvV4SolverPolicy):
        raise TypeError("solver_policy must be Lfm2CpcfShortConvV4SolverPolicy.")
    if not solver_policy.verify_authentication():
        raise Lfm2CpcfShortConvV4FoldError("V4 solver policy authentication failed.")
    if not isinstance(gate_policy, Lfm2CpcfShortConvGatePolicy):
        raise TypeError("gate_policy must be Lfm2CpcfShortConvGatePolicy.")
    if candidate_selector is not None and not callable(
        getattr(candidate_selector, "select", None)
    ):
        raise TypeError("candidate_selector must implement select(request).")
    if macrocell_candidate_selector is not None and not callable(
        getattr(macrocell_candidate_selector, "select", None)
    ):
        raise TypeError("macrocell_candidate_selector must implement select(frontier).")
    if candidate_selector is not None and macrocell_candidate_selector is not None:
        raise Lfm2CpcfShortConvV4FoldError(
            "Tensor-blind and whole-macrocell candidate selectors are "
            "mutually exclusive."
        )
    if macrocell_candidate_selector is None:
        if (
            macrocell_fit_bank_sha256 is not None
            or macrocell_selection_bank_sha256 is not None
        ):
            raise Lfm2CpcfShortConvV4FoldError(
                "Macrocell bank identities require a macrocell selector."
            )
    elif (
        not _valid_sha256_text(macrocell_fit_bank_sha256)
        or not _valid_sha256_text(macrocell_selection_bank_sha256)
        or macrocell_fit_bank_sha256 == macrocell_selection_bank_sha256
    ):
        raise Lfm2CpcfShortConvV4FoldError(
            "Macrocell selection requires distinct authenticated fit and "
            "selection state-bank digests."
        )
    if numerical_frontier_adapter is not None and not callable(
        getattr(numerical_frontier_adapter, "generate", None)
    ):
        raise TypeError("numerical_frontier_adapter must implement generate(request).")
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
    numerical_frontier_request = None
    numerical_frontier = None
    if numerical_frontier_adapter is None:
        (
            basis,
            lift_report,
            priors,
            split_cases,
            ridge_results,
            raw_candidates,
        ) = _generate_legacy_v4_raw_candidates(
            ordered=ordered,
            parent_contexts=parent_contexts,
            source_in_proj_weight=source_in_proj_weight,
            source_in_proj_bias=source_in_proj_bias,
            source_kernel=source_kernel,
            source_fir_bias=source_fir_bias,
            source_out_proj_weight=source_out_proj_weight,
            source_out_proj_bias=source_out_proj_bias,
            output_encoder=output_encoder,
            target_hidden=target_hidden,
            target_fir_length=int(conv_shape[2]),
            in_bias_template=in_bias_template,
            conv_bias_template=conv_bias_template,
            out_bias_template=out_bias_template,
            solver_policy=solver_policy,
        )
    else:
        numerical_frontier_request = Lfm2CpcfShortConvNumericalFrontierRequest(
            source_tensors={
                "source_in_proj_weight": source_in_proj_weight,
                "source_in_proj_bias": source_in_proj_bias,
                "source_fir_kernel": source_kernel,
                "source_fir_bias": source_fir_bias,
                "source_out_proj_weight": source_out_proj_weight,
                "source_out_proj_bias": source_out_proj_bias,
                "output_encoder": output_encoder,
            },
            cases=tuple(case for case in ordered if case.split == "fit"),
            parent_contexts={
                case.case_id: parent_contexts[case.case_id]
                for case in ordered
                if case.split == "fit"
            },
            target_templates={
                "in_proj_weight": target_in_proj_weight_template,
                "in_proj_bias": in_bias_template,
                "fir_kernel": (target_conv_weight_template[:, 0, :]),
                "fir_bias": conv_bias_template,
                "writer_weight": target_out_proj_weight_template,
                "writer_bias": out_bias_template,
            },
            policy=solver_policy,
        )
        try:
            generated_frontier = numerical_frontier_adapter.generate(
                numerical_frontier_request
            )
        except BaseException as error:
            if not numerical_frontier_request.verify_authentication():
                raise Lfm2CpcfShortConvV4FoldError(
                    "Numerical-frontier adapter changed its detached request "
                    "before failing."
                ) from error
            raise
        if not numerical_frontier_request.verify_authentication():
            raise Lfm2CpcfShortConvV4FoldError(
                "Numerical-frontier adapter changed its detached request."
            )
        if not isinstance(
            generated_frontier,
            Lfm2CpcfShortConvNumericalFrontier,
        ):
            raise Lfm2CpcfShortConvV4FoldError(
                "Numerical-frontier adapter returned the wrong frontier type."
            )
        numerical_frontier = generated_frontier
        _validate_numerical_frontier(
            numerical_frontier,
            request=numerical_frontier_request,
        )
        projections = _projection_cases(
            ordered,
            parent_contexts=parent_contexts,
            encoder=numerical_frontier.context_encoder,
        )
        split_cases = MappingProxyType(
            {
                split: tuple(case for case in projections if case.split == split)
                for split in ("fit", "selection", "validation")
            }
        )
        ridge_results = MappingProxyType({})
        raw_candidates = list(numerical_frontier.raw_candidates)

    deduplicated = {}
    for raw in raw_candidates:
        if numerical_frontier is None:
            candidate = raw["candidate"]
            materialized = {
                role: (
                    None
                    if getattr(candidate, role) is None
                    else getattr(candidate, role)
                    .detach()
                    .to(device="cpu", dtype=target_dtype)
                    .contiguous()
                )
                for role in _NUMERICAL_CANDIDATE_ROLES
            }
            source = {
                "candidate_id": raw["candidate_id"],
                "ridge_id": raw["ridge_id"],
                "candidate_kind": raw["candidate_kind"],
                "stage_id": candidate.stage_id,
                "fit_path_state_sha256": candidate.report["state_sha256"],
                "candidate_report": dict(candidate.report),
            }
            raw_candidate_id = raw["candidate_id"]
        else:
            candidate_state = raw.state_dict()
            materialized = {
                role: (
                    None
                    if candidate_state[role] is None
                    else candidate_state[role]
                    .detach()
                    .to(device="cpu", dtype=target_dtype)
                    .contiguous()
                )
                for role in _NUMERICAL_CANDIDATE_ROLES
            }
            source = {
                "candidate_id": raw.candidate_id,
                "candidate_kind": raw.candidate_kind,
                "numerical_raw_state_sha256": raw.state_sha256,
                "numerical_raw_candidate_report": raw.as_dict(),
                "numerical_raw_candidate_report_sha256": (raw.report["report_sha256"]),
            }
            raw_candidate_id = raw.candidate_id
        target_state_sha = _state_sha256(materialized)
        if target_state_sha in deduplicated:
            deduplicated[target_state_sha]["sources"].append(source)
            continue
        deduplicated[target_state_sha] = {
            "candidate_id": raw_candidate_id,
            "state": materialized,
            "sources": [source],
        }

    candidate_states = {}
    candidate_fit_reports = {}
    for target_state_sha, candidate_group in deduplicated.items():
        candidate_id = candidate_group["candidate_id"]
        state = candidate_group["state"]
        sources = candidate_group["sources"]
        if numerical_frontier is None:
            provenance_body = {
                "schema": LFM2_CPCF_SHORT_CONV_V4_FOLD_SCHEMA,
                "kind": "authenticated_fit_only_candidate_provenance",
                "candidate_id": candidate_id,
                "policy_sha256": solver_policy.authentication_sha256,
                "target_dtype": solver_policy.target_dtype,
                "target_dtype_candidate_state_sha256": target_state_sha,
                "source_alias_count": len(sources),
                "source_alias_candidate_ids": [
                    source["candidate_id"] for source in sources
                ],
                "sources": sources,
                "fit_only": True,
                "selection_cases_consumed": 0,
                "validation_cases_consumed": 0,
                "actual_target_dtype_materialization_before_deduplication": (True),
                "execution": {
                    "parent_model_forward_calls": 0,
                    "target_model_forward_calls": 0,
                    "generation_calls": 0,
                    "training_steps": 0,
                    "backward_calls": 0,
                    "optimizer_steps": 0,
                    "parameters_pruned": 0,
                    "parameters_quantized": 0,
                },
            }
        else:
            provenance_body = {
                "schema": (LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_PROVENANCE_SCHEMA),
                "kind": (
                    "authenticated_fit_only_numerical_frontier_candidate_provenance"
                ),
                "candidate_id": candidate_id,
                "policy_sha256": solver_policy.authentication_sha256,
                "target_dtype": solver_policy.target_dtype,
                "request_sha256": (numerical_frontier_request.request_sha256),
                "numerical_frontier_report_sha256": (
                    numerical_frontier.report["report_sha256"]
                ),
                "target_dtype_candidate_state_sha256": target_state_sha,
                "source_alias_count": len(sources),
                "source_alias_candidate_ids": [
                    source["candidate_id"] for source in sources
                ],
                "sources": sources,
                "fit_case_authentication_sha256": list(
                    numerical_frontier.fit_case_authentication_sha256
                ),
                "fit_only": True,
                "selection_cases_consumed": 0,
                "validation_cases_consumed": 0,
                "actual_target_dtype_materialization_before_deduplication": (True),
                "legacy_v4_candidate_provenance_claimed": False,
                "execution": dict(_ZERO_EXECUTION),
            }
        provenance = _seal_report(provenance_body)
        candidate_states[candidate_id] = state
        candidate_fit_reports[candidate_id] = provenance

    if numerical_frontier is not None:
        _validate_numerical_frontier(
            numerical_frontier,
            request=numerical_frontier_request,
        )

    candidate_replay_backend = None
    candidate_replay_work_items = ()
    if candidate_replay_evaluator is None:
        selection_reports = {
            candidate_id: evaluate_candidate_state_target_dtype(
                cases=split_cases["selection"],
                in_proj_weight=state["in_proj_weight"],
                in_proj_bias=state["in_proj_bias"],
                fir_kernel=state["fir_kernel"],
                fir_bias=state["fir_bias"],
                writer_weight=state["writer_weight"],
                writer_bias=state["writer_bias"],
                required_split="selection",
                policy=solver_policy,
            )
            for candidate_id, state in candidate_states.items()
        }
    else:
        (
            selection_reports,
            candidate_replay_backend,
            candidate_replay_work_items,
        ) = evaluate_lfm2_cpcf_candidate_states_with_backend(
            candidate_states,
            cases=split_cases["selection"],
            required_split="selection",
            policy=solver_policy,
            evaluator=candidate_replay_evaluator,
        )
    for candidate_id, selection in selection_reports.items():
        target_state_sha = _state_sha256(candidate_states[candidate_id])
        if selection["candidate_state_sha256"] != target_state_sha:
            raise Lfm2CpcfShortConvV4FoldError(
                "Target-dtype candidate deduplication and replay state differ."
            )

    selection_feasibility = _selection_feasibility_policy(gate_policy)
    ranking = rank_short_conv_v4_selection_candidates(
        selection_reports,
        policy_sha256=solver_policy.authentication_sha256,
        feasibility_policy=selection_feasibility,
    )
    default_selected_id = str(ranking["selected_candidate_id"])
    selector_receipt = None
    macrocell_selector_receipt = None
    if macrocell_candidate_selector is not None:
        selected_id, macrocell_selector_receipt = (
            _run_preinstall_macrocell_candidate_selector(
                macrocell_candidate_selector,
                candidate_states=candidate_states,
                candidate_fit_reports=candidate_fit_reports,
                selection_reports=selection_reports,
                ranking=ranking,
                solver_policy=solver_policy,
                fit_bank_sha256=macrocell_fit_bank_sha256,
                selection_bank_sha256=(macrocell_selection_bank_sha256),
            )
        )
    elif candidate_selector is None:
        selected_id = default_selected_id
    else:
        selected_id, selector_receipt = _run_preinstall_candidate_selector(
            candidate_selector,
            candidate_states=candidate_states,
            selection_reports=selection_reports,
            ranking=ranking,
            solver_policy=solver_policy,
        )
    selected = candidate_states[selected_id]
    in_weight = selected["in_proj_weight"].clone()
    fir_kernel = selected["fir_kernel"].clone()
    out_weight = selected["writer_weight"].clone()
    in_bias = _materialize_bias(
        selected["in_proj_bias"],
        in_bias_template,
        "in_proj_bias",
    )
    fir_bias = _materialize_bias(
        selected["fir_bias"],
        conv_bias_template,
        "conv_bias",
    )
    out_bias = _materialize_bias(
        selected["writer_bias"],
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
    if numerical_frontier is None:
        full_source_coverage = bool(
            basis.report["full_source_numeric_participation"]
            and priors.report["source_channels_dropped"] == 0
        )
    else:
        full_source_coverage = numerical_frontier.full_source_numeric_participation
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
    if macrocell_candidate_selector is not None:
        report_schema = LFM2_CPCF_SHORT_CONV_V5_MACROCELL_SELECTOR_FOLD_SCHEMA
    elif numerical_frontier is not None:
        report_schema = LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_FOLD_SCHEMA
    elif candidate_selector is not None:
        report_schema = LFM2_CPCF_SHORT_CONV_V5_SELECTOR_FOLD_SCHEMA
    elif candidate_replay_evaluator is not None:
        report_schema = LFM2_CPCF_SHORT_CONV_V5_FOLD_SCHEMA
    else:
        report_schema = LFM2_CPCF_SHORT_CONV_V4_FOLD_SCHEMA
    if numerical_frontier is None:
        candidate_pool_body = {
            "schema": report_schema,
            "kind": "authenticated_target_dtype_candidate_pool",
            "policy_sha256": solver_policy.authentication_sha256,
            "target_dtype": solver_policy.target_dtype,
            "ridge_candidate_count": len(ridge_results),
            "raw_candidate_count": len(raw_candidates),
            "unique_target_dtype_candidate_count": len(candidate_states),
            "target_dtype_duplicate_count": (
                len(raw_candidates) - len(candidate_states)
            ),
            "accepted_phase_anchors_included": True,
            "rejected_phase_states_included": False,
            "prior_to_final_trust_path_fractions": list(
                fixed_projection_trust_path_fractions()
            ),
            "prior_to_first_writer_trust_path_fractions": list(
                fixed_projection_trust_path_fractions()
            ),
            "selection_feasibility_policy": selection_feasibility,
            "selection_feasibility_policy_sha256": ranking[
                "selection_feasibility_policy_sha256"
            ],
            "fold_case_authentication_sha256": {
                split: [
                    case.authentication_sha256
                    for case in ordered
                    if case.split == split
                ]
                for split in ("fit", "selection", "validation")
            },
            "projection_case_authentication_sha256": {
                split: [case.authentication_sha256 for case in split_cases[split]]
                for split in ("fit", "selection", "validation")
            },
            "candidate_ids": sorted(candidate_states),
            "candidate_state_sha256": {
                candidate_id: selection_reports[candidate_id]["candidate_state_sha256"]
                for candidate_id in sorted(candidate_states)
            },
            "fit_only_candidate_generation": True,
            "selection_cases_consumed_during_generation": 0,
            "validation_cases_consumed_during_generation": 0,
        }
    else:
        candidate_pool_body = {
            "schema": LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_POOL_SCHEMA,
            "kind": ("authenticated_v5_numerical_target_dtype_candidate_pool"),
            "policy_sha256": solver_policy.authentication_sha256,
            "target_dtype": solver_policy.target_dtype,
            "request_sha256": numerical_frontier_request.request_sha256,
            "numerical_frontier_report_sha256": (
                numerical_frontier.report["report_sha256"]
            ),
            "raw_candidate_count": len(raw_candidates),
            "unique_target_dtype_candidate_count": len(candidate_states),
            "target_dtype_duplicate_count": (
                len(raw_candidates) - len(candidate_states)
            ),
            "selection_feasibility_policy": selection_feasibility,
            "selection_feasibility_policy_sha256": ranking[
                "selection_feasibility_policy_sha256"
            ],
            "fold_case_authentication_sha256": {
                split: [
                    case.authentication_sha256
                    for case in ordered
                    if case.split == split
                ]
                for split in ("fit", "selection", "validation")
            },
            "projection_case_authentication_sha256": {
                split: [case.authentication_sha256 for case in split_cases[split]]
                for split in ("fit", "selection", "validation")
            },
            "candidate_ids": sorted(candidate_states),
            "candidate_state_sha256": {
                candidate_id: selection_reports[candidate_id]["candidate_state_sha256"]
                for candidate_id in sorted(candidate_states)
            },
            "source_channels_selected": (numerical_frontier.source_channels_selected),
            "source_channels_dropped": (numerical_frontier.source_channels_dropped),
            "full_source_numeric_participation": (
                numerical_frontier.full_source_numeric_participation
            ),
            "fit_only_candidate_generation": True,
            "selection_cases_consumed_during_generation": 0,
            "validation_cases_consumed_during_generation": 0,
            "legacy_v4_candidate_pool_claimed": False,
        }
    if candidate_replay_backend is not None:
        candidate_pool_body.update(
            {
                "candidate_replay_backend_report_sha256": (
                    candidate_replay_backend["report_sha256"]
                ),
                "candidate_replay_work_items": [
                    item.as_dict() for item in candidate_replay_work_items
                ],
                "candidate_replay_work_item_count": len(candidate_replay_work_items),
                "canonical_candidate_replay_order": [
                    item.payload["candidate_id"] for item in candidate_replay_work_items
                ],
                "scheduled_replay_changes_closed_form_solve_math": False,
            }
        )
    if numerical_frontier is None:
        common = {
            "solver_schema": LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
            "solver_policy": solver_policy.as_dict(),
            "solver_policy_sha256": solver_policy.authentication_sha256,
            "gate_policy": gate_policy.as_dict(),
            "source_hidden_size": source_hidden,
            "target_hidden_size": target_hidden,
            "source_fir_length": int(source_kernel.shape[1]),
            "target_fir_length": int(fir_kernel.shape[1]),
            "hidden_width_compression_ratio": (source_hidden / target_hidden),
            "case_authentication_sha256": {
                split: [
                    case.authentication_sha256
                    for case in ordered
                    if case.split == split
                ]
                for split in ("fit", "selection", "validation")
            },
            "basis": dict(basis.report),
            "input_lift": dict(lift_report),
            "transported_priors": dict(priors.report),
            "ridge_fit_reports": {
                ridge_id: dict(result.report)
                for ridge_id, result in ridge_results.items()
            },
            "candidate_fit_reports": {
                candidate_id: dict(report)
                for candidate_id, report in candidate_fit_reports.items()
            },
            "candidate_selection_reports": {
                candidate_id: dict(report)
                for candidate_id, report in selection_reports.items()
            },
            "candidate_pool_report": _seal_report(candidate_pool_body),
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
    else:
        common = {
            "solver_schema": LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
            "solver_policy": solver_policy.as_dict(),
            "solver_policy_sha256": solver_policy.authentication_sha256,
            "gate_policy": gate_policy.as_dict(),
            "source_hidden_size": source_hidden,
            "target_hidden_size": target_hidden,
            "source_fir_length": int(source_kernel.shape[1]),
            "target_fir_length": int(fir_kernel.shape[1]),
            "hidden_width_compression_ratio": (source_hidden / target_hidden),
            "case_authentication_sha256": {
                split: [
                    case.authentication_sha256
                    for case in ordered
                    if case.split == split
                ]
                for split in ("fit", "selection", "validation")
            },
            "numerical_frontier_request": (numerical_frontier_request.as_dict()),
            "numerical_frontier_request_sha256": (
                numerical_frontier_request.request_sha256
            ),
            "numerical_frontier": numerical_frontier.as_dict(),
            "numerical_frontier_report_sha256": (
                numerical_frontier.report["report_sha256"]
            ),
            "candidate_fit_reports": {
                candidate_id: dict(report)
                for candidate_id, report in candidate_fit_reports.items()
            },
            "candidate_selection_reports": {
                candidate_id: dict(report)
                for candidate_id, report in selection_reports.items()
            },
            "candidate_pool_report": _seal_report(candidate_pool_body),
            "selection_ranking": dict(ranking),
            "selected_candidate_id": selected_id,
            "fit_evaluation": dict(evaluations["fit"]),
            "selection_evaluation": dict(evaluations["selection"]),
            "validation_evaluation": dict(evaluations["validation"]),
            "behavior_gate": gate,
            "compiled_state_sha256": compiled_state_sha,
            "source_channels_selected": (numerical_frontier.source_channels_selected),
            "source_channels_dropped": (numerical_frontier.source_channels_dropped),
            "source_channels_pruned": 0,
            "source_channels_quantized": 0,
            "fit_only_numerical_generation": True,
            "selection_used_only_for_candidate_ranking": True,
            "validation_used_only_after_candidate_freeze": True,
            "numerical_frontier_adapter_active": True,
            "legacy_v4_numerical_generation_claimed": False,
            "legacy_v4_report_claimed": False,
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
    if selector_receipt is not None:
        common.update(
            {
                "preinstall_candidate_selector_receipt": dict(selector_receipt),
                "preinstall_candidate_selector_receipt_sha256": (
                    selector_receipt["report_sha256"]
                ),
                "default_v4_selected_candidate_id": default_selected_id,
                "selection_changed_from_v4": (selected_id != default_selected_id),
                "preinstall_candidate_selector_active": True,
                "post_install_rescoring_performed": False,
                "legacy_v4_report_claimed": False,
            }
        )
    if macrocell_selector_receipt is not None:
        common.update(
            {
                "macrocell_candidate_selector_receipt": dict(
                    macrocell_selector_receipt
                ),
                "macrocell_candidate_selector_receipt_sha256": (
                    macrocell_selector_receipt["report_sha256"]
                ),
                "default_v4_selected_candidate_id": default_selected_id,
                "selection_changed_from_v4": (selected_id != default_selected_id),
                "macrocell_candidate_selector_active": True,
                "candidate_state_tensors_exposed_to_macrocell_selector": True,
                "fit_provenance_exposed_to_macrocell_selector": True,
                "operator_selection_reports_exposed_to_macrocell_selector": (True),
                "validation_evidence_exposed_to_macrocell_selector": False,
                "validation_used_for_macrocell_selection": False,
                "validation_frozen_until_after_macrocell_winner": True,
                "post_install_rescoring_performed": False,
                "legacy_v4_report_claimed": False,
            }
        )
    if candidate_replay_backend is not None:
        common.update(
            {
                "candidate_replay_backend": dict(candidate_replay_backend),
                "candidate_replay_backend_report_sha256": (
                    candidate_replay_backend["report_sha256"]
                ),
                "selection_candidate_replay_backend": (
                    "authenticated_injected_backend"
                ),
                "fit_and_validation_winner_replay_backend": "cpu_reference",
                "legacy_v4_report_claimed": False,
            }
        )
    if gate_policy.enforce and not gate["passed"]:
        failure_body = {
            "schema": (
                LFM2_CPCF_SHORT_CONV_V5_MACROCELL_SELECTOR_FAILURE_SCHEMA
                if macrocell_candidate_selector is not None
                else (
                    LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_FAILURE_SCHEMA
                    if numerical_frontier is not None
                    else (
                        LFM2_CPCF_SHORT_CONV_V5_SELECTOR_FAILURE_SCHEMA
                        if candidate_selector is not None
                        else (
                            LFM2_CPCF_SHORT_CONV_V4_FAILURE_SCHEMA
                            if candidate_replay_evaluator is None
                            else LFM2_CPCF_SHORT_CONV_V5_FAILURE_SCHEMA
                        )
                    )
                )
            ),
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
        "schema": report_schema,
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
    "LFM2_CPCF_SHORT_CONV_V4_FAILURE_V1_SCHEMA",
    "LFM2_CPCF_SHORT_CONV_V4_FAILURE_SCHEMA",
    "LFM2_CPCF_SHORT_CONV_V4_FOLD_SCHEMA",
    "LFM2_CPCF_SHORT_CONV_V5_FAILURE_SCHEMA",
    "LFM2_CPCF_SHORT_CONV_V5_FOLD_SCHEMA",
    "LFM2_CPCF_SHORT_CONV_V5_MACROCELL_CANDIDATE_SCHEMA",
    "LFM2_CPCF_SHORT_CONV_V5_MACROCELL_EVIDENCE_SCHEMA",
    "LFM2_CPCF_SHORT_CONV_V5_MACROCELL_EVALUATION_SCHEMA",
    "LFM2_CPCF_SHORT_CONV_V5_MACROCELL_FRONTIER_SCHEMA",
    "LFM2_CPCF_SHORT_CONV_V5_MACROCELL_SELECTOR_FAILURE_SCHEMA",
    "LFM2_CPCF_SHORT_CONV_V5_MACROCELL_SELECTOR_FOLD_SCHEMA",
    "LFM2_CPCF_SHORT_CONV_V5_MACROCELL_SELECTOR_RECEIPT_SCHEMA",
    "LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_FAILURE_SCHEMA",
    "LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_FOLD_SCHEMA",
    "LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_REQUEST_SCHEMA",
    "LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_SCHEMA",
    "LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_POOL_SCHEMA",
    "LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_PROVENANCE_SCHEMA",
    "LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_RAW_CANDIDATE_SCHEMA",
    "LFM2_CPCF_SHORT_CONV_V5_SELECTOR_FAILURE_SCHEMA",
    "LFM2_CPCF_SHORT_CONV_V5_SELECTOR_FOLD_SCHEMA",
    "LFM2_CPCF_SHORT_CONV_V5_SELECTOR_RECEIPT_SCHEMA",
    "LFM2_CPCF_SHORT_CONV_V5_SELECTOR_REQUEST_SCHEMA",
    "Lfm2CompiledCpcfShortConvV4Fold",
    "Lfm2CpcfShortConvNumericalFrontier",
    "Lfm2CpcfShortConvNumericalFrontierAdapter",
    "Lfm2CpcfShortConvNumericalFrontierRequest",
    "Lfm2CpcfShortConvNumericalRawCandidate",
    "Lfm2CpcfShortConvV5CandidateFrontier",
    "Lfm2CpcfShortConvV5MacrocellCandidate",
    "Lfm2CpcfShortConvV5MacrocellCandidateSelector",
    "Lfm2CpcfShortConvV4FoldCase",
    "Lfm2CpcfShortConvV4FoldError",
    "Lfm2CpcfShortConvV5CandidateSelector",
    "compile_lfm2_cpcf_short_conv_v4_fold",
    "select_lfm2_cpcf_short_conv_v5_macrocell_candidate_ids",
]
