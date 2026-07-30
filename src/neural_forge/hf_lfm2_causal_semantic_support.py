"""Causal, role-separated sparse semantic support selection for LFM2.

This module consumes the sealed four-lane teacher-forced trajectory audit and
turns *observed causal damage* into localized sparse-route repairs.  It does
not fit a global residual map and it does not use review prompts, raw token
frequency, gradients, a parent forward, pruning, or quantization.

Two searches are run from scratch:

``C_in``
    Uses the isolated ``folded_exact`` prompt intervention and positive
    increments of the ``exact_folded`` continuation intervention.
``C_out``
    Uses the exact-hidden-state difference between the exact-table and folded
    readouts, attributing lost margin to both the desired row and an erroneous
    competing row when one exists.

Each occurrence retains its prompt, phase, and position-bin identity.  Cell
aggregation is by maximum causal damage, not occurrence count, so repeating a
token cannot turn this into another frequency allocator.  Context signatures
participate in residual-OMP *support discovery*.  Runtime coefficients are
then reclosed locally against explicit channel feature transforms.  The
transforms are compile-time evidence bound to fit-only trajectory artifacts;
they are not runtime child state.  Every published FP32 C_in and C_out
coefficient is nevertheless counted independently as child state.

The current live teacher-forced audit was not available when this contract was
introduced.  The implementation is therefore deliberately artifact-driven:
it can be proved on sealed fixtures now and consume the live v2 audit without
changing its mathematical ABI once that audit is published.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .dynamic_circuit_ir import canonical_sha256
from .hf_lfm2_teacher_forced_trajectory_audit import (
    LFM2_TEACHER_FORCED_TRAJECTORY_AUDIT_SCHEMA,
    authenticate_prompt_result,
)


LFM2_CAUSAL_SUPPORT_FEATURES_SCHEMA = (
    "neural-forge-lfm2-causal-semantic-support-features-v1"
)
LFM2_CAUSAL_SUPPORT_REQUEST_SCHEMA = (
    "neural-forge-lfm2-causal-semantic-support-request-v1"
)
LFM2_CAUSAL_SUPPORT_SELECTION_SCHEMA = (
    "neural-forge-lfm2-causal-semantic-support-selection-v1"
)
LFM2_CAUSAL_SUPPORT_ALGORITHM = (
    "fit-only-position-context-causal-damage-independent-local-omp-v1"
)

_TENSOR_HASH_DOMAIN = b"neural-forge-lfm2-causal-semantic-support-tensor-v1\0"
_ZERO_OPERATION_AUDIT: Mapping[str, int] = {
    "training_steps": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "autograd_grad_calls": 0,
    "backward_calls": 0,
    "model_objects_constructed": 0,
    "model_forward_calls": 0,
    "model_generate_calls": 0,
    "parent_forward_calls": 0,
    "parent_generate_calls": 0,
    "parent_trace_recaptures": 0,
    "global_map_solve_calls": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
}


class Lfm2CausalSemanticSupportError(RuntimeError):
    """Raised when causal evidence or a localized support patch is invalid."""


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Lfm2CausalSemanticSupportError(
                "Receipts cannot contain non-finite numbers."
            )
        return value
    raise Lfm2CausalSemanticSupportError(
        f"Receipt contains unsupported {type(value).__name__}."
    )


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CausalSemanticSupportError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CausalSemanticSupportError(f"{name} must be positive.")
    return int(value)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2CausalSemanticSupportError(f"{name} must be nonnegative.")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise Lfm2CausalSemanticSupportError(f"{name} must be finite and positive.")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2CausalSemanticSupportError(f"{name} must be finite and nonnegative.")
    return result


def _dtype_name(value: Tensor | torch.dtype) -> str:
    dtype = value.dtype if isinstance(value, Tensor) else value
    return str(dtype).removeprefix("torch.")


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256(_TENSOR_HASH_DOMAIN)
    digest.update(_dtype_name(tensor).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validated_float_matrix(value: Tensor, name: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 2
        or value.dtype
        not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}
        or value.requires_grad
        or min(int(size) for size in value.shape) <= 0
        or not bool(torch.isfinite(value.detach()).all())
    ):
        raise Lfm2CausalSemanticSupportError(
            f"{name} must be one finite detached floating matrix."
        )
    return value.detach().to(device="cpu").contiguous()


def _validated_int64_vector(value: Tensor, name: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 1
        or value.dtype != torch.int64
        or value.device.type != "cpu"
        or value.requires_grad
    ):
        raise Lfm2CausalSemanticSupportError(
            f"{name} must be one detached CPU int64 vector."
        )
    return value.detach().clone().contiguous()


def _validated_fp32_vector(value: Tensor, name: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 1
        or value.dtype != torch.float32
        or value.device.type != "cpu"
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
    ):
        raise Lfm2CausalSemanticSupportError(
            f"{name} must be one finite detached CPU FP32 vector."
        )
    return value.detach().clone().contiguous()


def _validated_route(
    offsets: Tensor,
    target_ids: Tensor,
    coefficients: Tensor,
    *,
    donor_count: int,
    target_count: int,
    name: str,
) -> tuple[Tensor, Tensor, Tensor]:
    checked_offsets = _validated_int64_vector(offsets, f"{name} offsets")
    checked_ids = _validated_int64_vector(target_ids, f"{name} target IDs")
    checked_coefficients = _validated_fp32_vector(coefficients, f"{name} coefficients")
    lengths = checked_offsets[1:] - checked_offsets[:-1]
    if (
        int(checked_offsets.numel()) != donor_count + 1
        or int(checked_offsets[0].item()) != 0
        or int(checked_offsets[-1].item()) != int(checked_ids.numel())
        or int(checked_ids.numel()) != int(checked_coefficients.numel())
        or bool((lengths <= 0).any())
        or bool((checked_ids < 0).any())
        or bool((checked_ids >= target_count).any())
    ):
        raise Lfm2CausalSemanticSupportError(
            f"{name} is not one complete nonempty donor-to-target CSR route."
        )
    return checked_offsets, checked_ids, checked_coefficients


def _verify_direct_routes(
    route: tuple[Tensor, Tensor, Tensor], target_to_donor: Tensor, *, name: str
) -> None:
    offsets, ids, coefficients = route
    one_bits = torch.tensor(1.0, dtype=torch.float32).view(torch.int32).item()
    for target_id, donor_id in enumerate(target_to_donor.tolist()):
        start = int(offsets[donor_id].item())
        stop = int(offsets[donor_id + 1].item())
        if (
            stop - start != 1
            or int(ids[start].item()) != target_id
            or int(coefficients[start].view(torch.int32).item()) != one_bits
        ):
            raise Lfm2CausalSemanticSupportError(
                f"{name} changed an exact direct target/donor anchor."
            )


def _prompt_id_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise Lfm2CausalSemanticSupportError(f"{name} must be a sequence.")
    result = tuple(str(value) for value in values)
    if (
        not result
        or any(not value for value in result)
        or len(set(result)) != len(result)
    ):
        raise Lfm2CausalSemanticSupportError(
            f"{name} must contain unique nonempty prompt IDs."
        )
    return result


def _authenticated_audit(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    audit = _plain_json(value)
    supplied = audit.pop("audit_sha256", None)
    results_value = audit.get("prompt_results")
    operation = audit.get("operation_audit")
    if (
        supplied != canonical_sha256(audit)
        or audit.get("schema") != LFM2_TEACHER_FORCED_TRAJECTORY_AUDIT_SCHEMA
        or not isinstance(results_value, list)
        or not results_value
        or audit.get("prompt_count") != len(results_value)
        or not isinstance(operation, Mapping)
    ):
        raise Lfm2CausalSemanticSupportError(
            "Teacher-forced trajectory audit authentication failed."
        )
    for key in (
        "training_steps",
        "backward_calls",
        "optimizer_steps",
        "parent_forward_calls",
        "parent_trace_recapture_calls",
        "global_residual_maps_fit",
        "pruned_parameters",
        "quantized_parameters",
    ):
        if operation.get(key) != 0:
            raise Lfm2CausalSemanticSupportError(
                "Trajectory selection evidence violates fold-only provenance."
            )
    results: dict[str, dict[str, Any]] = {}
    for item in results_value:
        result = authenticate_prompt_result(item)
        prompt_id = str(result["prompt_id"])
        if not prompt_id or prompt_id in results:
            raise Lfm2CausalSemanticSupportError(
                "Trajectory audit prompt IDs must be nonempty and unique."
            )
        results[prompt_id] = result
    return {**audit, "audit_sha256": supplied}, results


@dataclass(frozen=True, slots=True)
class Lfm2CausalSupportFeaturePack:
    """Sealed fit-only feature transforms used for local route reclosure."""

    input_feature_transform: Tensor
    output_feature_transform: Tensor
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        input_transform = _validated_float_matrix(
            self.input_feature_transform, "input_feature_transform"
        )
        output_transform = _validated_float_matrix(
            self.output_feature_transform, "output_feature_transform"
        )
        value = _plain_json(self.receipt)
        supplied = value.pop("artifact_sha256", None)
        fit_ids = tuple(value.get("fit_prompt_ids", ()))
        review_ids = tuple(value.get("manual_review_prompt_ids", ()))
        if (
            supplied != canonical_sha256(value)
            or value.get("schema") != LFM2_CAUSAL_SUPPORT_FEATURES_SCHEMA
            or not fit_ids
            or not review_ids
            or set(fit_ids).intersection(review_ids)
            or value.get("selection_and_manual_review_prompt_ids_disjoint") is not True
            or value.get("input_feature_transform_sha256")
            != _tensor_sha256(input_transform)
            or value.get("output_feature_transform_sha256")
            != _tensor_sha256(output_transform)
            or value.get("input_feature_shape") != list(input_transform.shape)
            or value.get("output_feature_shape") != list(output_transform.shape)
            or value.get("input_feature_dtype") != _dtype_name(input_transform)
            or value.get("output_feature_dtype") != _dtype_name(output_transform)
            or int(input_transform.shape[1]) != int(output_transform.shape[1])
            or value.get("hidden_size") != int(input_transform.shape[1])
            or value.get("compile_time_features_are_runtime_child_state") is not False
            or value.get("operation_audit") != _ZERO_OPERATION_AUDIT
        ):
            raise Lfm2CausalSemanticSupportError(
                "Causal support feature-pack authentication failed."
            )
        _sha256(value.get("trajectory_audit_sha256"), "trajectory_audit_sha256")
        _sha256(value.get("topology_sha256"), "topology_sha256")
        _sha256(value.get("numeric_contract_sha256"), "numeric_contract_sha256")
        object.__setattr__(self, "input_feature_transform", input_transform.clone())
        object.__setattr__(self, "output_feature_transform", output_transform.clone())
        object.__setattr__(
            self, "receipt", _deep_freeze({**value, "artifact_sha256": supplied})
        )

    @property
    def artifact_sha256(self) -> str:
        return str(self.receipt["artifact_sha256"])

    def as_dict(self) -> dict[str, Any]:
        return _plain_json(self.receipt)


def seal_lfm2_causal_support_feature_pack(
    *,
    input_feature_transform: Tensor,
    output_feature_transform: Tensor,
    trajectory_audit_sha256: str,
    fit_prompt_ids: Sequence[str],
    manual_review_prompt_ids: Sequence[str],
    topology_sha256: str,
    numeric_contract_sha256: str,
    source_artifact_sha256s: Mapping[str, str],
    input_feature_kind: str = "cached-fit-only-causal-input-feature-transform",
    output_feature_kind: str = "cached-fit-only-parent-hidden-output-features",
) -> Lfm2CausalSupportFeaturePack:
    """Seal compile-time features without treating them as child parameters."""

    input_transform = _validated_float_matrix(
        input_feature_transform, "input_feature_transform"
    )
    output_transform = _validated_float_matrix(
        output_feature_transform, "output_feature_transform"
    )
    if int(input_transform.shape[1]) != int(output_transform.shape[1]):
        raise Lfm2CausalSemanticSupportError(
            "Input and output feature transforms must share hidden width."
        )
    fit_ids = _prompt_id_tuple(fit_prompt_ids, "fit_prompt_ids")
    review_ids = _prompt_id_tuple(manual_review_prompt_ids, "manual_review_prompt_ids")
    if set(fit_ids).intersection(review_ids):
        raise Lfm2CausalSemanticSupportError(
            "Selection evidence and manual review prompts must be disjoint."
        )
    sources = {
        str(key): _sha256(value, f"source {key}")
        for key, value in source_artifact_sha256s.items()
    }
    if not sources:
        raise Lfm2CausalSemanticSupportError(
            "At least one cached source artifact binding is required."
        )
    if not input_feature_kind or not output_feature_kind:
        raise Lfm2CausalSemanticSupportError("Feature kinds must be explicit.")
    body = {
        "schema": LFM2_CAUSAL_SUPPORT_FEATURES_SCHEMA,
        "trajectory_audit_sha256": _sha256(
            trajectory_audit_sha256, "trajectory_audit_sha256"
        ),
        "fit_prompt_ids": list(fit_ids),
        "fit_prompt_ids_sha256": canonical_sha256(list(fit_ids)),
        "manual_review_prompt_ids": list(review_ids),
        "manual_review_prompt_ids_sha256": canonical_sha256(list(review_ids)),
        "selection_and_manual_review_prompt_ids_disjoint": True,
        "topology_sha256": _sha256(topology_sha256, "topology_sha256"),
        "numeric_contract_sha256": _sha256(
            numeric_contract_sha256, "numeric_contract_sha256"
        ),
        "source_artifact_sha256s": sources,
        "input_feature_kind": input_feature_kind,
        "output_feature_kind": output_feature_kind,
        "input_feature_transform_sha256": _tensor_sha256(input_transform),
        "output_feature_transform_sha256": _tensor_sha256(output_transform),
        "input_feature_shape": list(input_transform.shape),
        "output_feature_shape": list(output_transform.shape),
        "input_feature_dtype": _dtype_name(input_transform),
        "output_feature_dtype": _dtype_name(output_transform),
        "hidden_size": int(input_transform.shape[1]),
        "compile_time_feature_scalar_count": int(
            input_transform.numel() + output_transform.numel()
        ),
        "compile_time_features_are_runtime_child_state": False,
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
    }
    return Lfm2CausalSupportFeaturePack(
        input_transform,
        output_transform,
        {**body, "artifact_sha256": canonical_sha256(body)},
    )


def _prompt_token_manifest(
    values: Mapping[str, Sequence[int]], fit_ids: Sequence[str], donor_count: int
) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for prompt_id in fit_ids:
        if prompt_id not in values:
            raise Lfm2CausalSemanticSupportError(
                f"Missing prompt-token manifest row for {prompt_id}."
            )
        row = values[prompt_id]
        if isinstance(row, (str, bytes)):
            raise Lfm2CausalSemanticSupportError("Prompt token rows must be sequences.")
        ids = []
        for token_id in row:
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                or token_id >= donor_count
            ):
                raise Lfm2CausalSemanticSupportError(
                    "Prompt token manifest contains an invalid donor token ID."
                )
            ids.append(int(token_id))
        if not ids:
            raise Lfm2CausalSemanticSupportError(
                "Prompt token manifest rows must be nonempty."
            )
        result[prompt_id] = ids
    return result


def _validated_request(value: Mapping[str, Any]) -> dict[str, Any]:
    request = _plain_json(value)
    supplied = request.pop("cache_key", None)
    if (
        supplied != canonical_sha256(request)
        or request.get("schema") != LFM2_CAUSAL_SUPPORT_REQUEST_SCHEMA
        or request.get("algorithm") != LFM2_CAUSAL_SUPPORT_ALGORITHM
        or request.get("input_output_support_searches_independent") is not True
        or request.get("output_support_ids_inherited_from_input") is not False
        or request.get("selection_evidence_is_frequency_only") is not False
        or request.get("selection_and_manual_review_prompt_ids_disjoint") is not True
        or request.get("operation_audit") != _ZERO_OPERATION_AUDIT
    ):
        raise Lfm2CausalSemanticSupportError(
            "Causal support request authentication failed."
        )
    return {**request, "cache_key": supplied}


def lfm2_causal_support_request(
    *,
    trajectory_audit: Mapping[str, Any],
    feature_pack: Lfm2CausalSupportFeaturePack,
    prompt_token_ids_by_prompt: Mapping[str, Sequence[int]],
    donor_table: Tensor,
    target_table: Tensor,
    target_to_donor: Tensor,
    base_input_offsets: Tensor,
    base_input_target_ids: Tensor,
    base_input_coefficients: Tensor,
    base_output_offsets: Tensor,
    base_output_target_ids: Tensor,
    base_output_coefficients: Tensor,
    physical_child_parameter_count: int,
    maximum_combined_coefficient_count: int,
    input_support_width: int,
    output_support_width: int,
    maximum_input_donor_repairs: int,
    maximum_output_donor_repairs: int,
    position_bin_count: int = 4,
    context_hash_bucket_count: int = 256,
    context_weight: float = 0.25,
    solve_ridge: float = 1e-8,
    solve_dtype: str = "float64",
) -> dict[str, Any]:
    """Build the exact content identity before any localized support search."""

    if not isinstance(feature_pack, Lfm2CausalSupportFeaturePack):
        raise TypeError("feature_pack must be an authenticated feature pack.")
    audit, results = _authenticated_audit(trajectory_audit)
    feature = feature_pack.as_dict()
    if audit["audit_sha256"] != feature["trajectory_audit_sha256"]:
        raise Lfm2CausalSemanticSupportError(
            "Feature pack belongs to a different trajectory audit."
        )
    donor = _validated_float_matrix(donor_table, "donor_table")
    target = _validated_float_matrix(target_table, "target_table")
    donor_count = int(donor.shape[0])
    target_count = int(target.shape[0])
    hidden_size = int(donor.shape[1])
    if int(target.shape[1]) != hidden_size or feature["hidden_size"] != hidden_size:
        raise Lfm2CausalSemanticSupportError(
            "Tables and feature pack have different hidden widths."
        )
    mapping = _validated_int64_vector(target_to_donor, "target_to_donor")
    if (
        int(mapping.numel()) != target_count
        or bool((mapping < 0).any())
        or bool((mapping >= donor_count).any())
        or int(torch.unique(mapping).numel()) != target_count
    ):
        raise Lfm2CausalSemanticSupportError(
            "target_to_donor must be one unique direct donor per target row."
        )
    input_route = _validated_route(
        base_input_offsets,
        base_input_target_ids,
        base_input_coefficients,
        donor_count=donor_count,
        target_count=target_count,
        name="base input route",
    )
    output_route = _validated_route(
        base_output_offsets,
        base_output_target_ids,
        base_output_coefficients,
        donor_count=donor_count,
        target_count=target_count,
        name="base output route",
    )
    _verify_direct_routes(input_route, mapping, name="base input route")
    _verify_direct_routes(output_route, mapping, name="base output route")
    fit_ids = tuple(feature["fit_prompt_ids"])
    review_ids = tuple(feature["manual_review_prompt_ids"])
    if any(prompt_id not in results for prompt_id in fit_ids):
        raise Lfm2CausalSemanticSupportError(
            "Feature-pack fit prompts are missing from the trajectory audit."
        )
    if set(fit_ids).intersection(review_ids):
        raise Lfm2CausalSemanticSupportError(
            "Selection and manual-review prompts overlap."
        )
    manifest = _prompt_token_manifest(prompt_token_ids_by_prompt, fit_ids, donor_count)
    for prompt_id in fit_ids:
        if len(manifest[prompt_id]) != int(results[prompt_id]["prompt_token_count"]):
            raise Lfm2CausalSemanticSupportError(
                "Prompt-token manifest length differs from the trajectory result."
            )
    input_width = _positive_int(input_support_width, "input_support_width")
    output_width = _positive_int(output_support_width, "output_support_width")
    if input_width > target_count or output_width > target_count:
        raise Lfm2CausalSemanticSupportError(
            "Support width cannot exceed the target dictionary."
        )
    input_repairs = _positive_int(
        maximum_input_donor_repairs, "maximum_input_donor_repairs"
    )
    output_repairs = _positive_int(
        maximum_output_donor_repairs, "maximum_output_donor_repairs"
    )
    physical = _positive_int(
        physical_child_parameter_count, "physical_child_parameter_count"
    )
    coefficient_budget = _positive_int(
        maximum_combined_coefficient_count,
        "maximum_combined_coefficient_count",
    )
    if coefficient_budget < int(input_route[2].numel() + output_route[2].numel()):
        raise Lfm2CausalSemanticSupportError(
            "Coefficient budget is already smaller than the base routes."
        )
    bins = _positive_int(position_bin_count, "position_bin_count")
    context_buckets = _positive_int(
        context_hash_bucket_count, "context_hash_bucket_count"
    )
    context = _nonnegative_float(context_weight, "context_weight")
    ridge = _positive_float(solve_ridge, "solve_ridge")
    if solve_dtype not in {"float32", "float64"}:
        raise Lfm2CausalSemanticSupportError(
            "solve_dtype must be explicit float32 or float64."
        )
    body = {
        "schema": LFM2_CAUSAL_SUPPORT_REQUEST_SCHEMA,
        "algorithm": LFM2_CAUSAL_SUPPORT_ALGORITHM,
        "trajectory_audit_sha256": audit["audit_sha256"],
        "feature_pack_artifact_sha256": feature_pack.artifact_sha256,
        "fit_prompt_ids": list(fit_ids),
        "fit_prompt_ids_sha256": canonical_sha256(list(fit_ids)),
        "manual_review_prompt_ids": list(review_ids),
        "manual_review_prompt_ids_sha256": canonical_sha256(list(review_ids)),
        "selection_and_manual_review_prompt_ids_disjoint": True,
        "selection_consumes_only_fit_prompt_ids": True,
        "prompt_token_manifest": manifest,
        "prompt_token_manifest_sha256": canonical_sha256(manifest),
        "topology_sha256": feature["topology_sha256"],
        "numeric_contract_sha256": feature["numeric_contract_sha256"],
        "donor_table_sha256": _tensor_sha256(donor),
        "target_table_sha256": _tensor_sha256(target),
        "target_to_donor_sha256": _tensor_sha256(mapping),
        "base_input_offsets_sha256": _tensor_sha256(input_route[0]),
        "base_input_target_ids_sha256": _tensor_sha256(input_route[1]),
        "base_input_coefficients_sha256": _tensor_sha256(input_route[2]),
        "base_output_offsets_sha256": _tensor_sha256(output_route[0]),
        "base_output_target_ids_sha256": _tensor_sha256(output_route[1]),
        "base_output_coefficients_sha256": _tensor_sha256(output_route[2]),
        "donor_row_count": donor_count,
        "target_row_count": target_count,
        "hidden_size": hidden_size,
        "physical_child_parameter_count": physical,
        "base_input_coefficient_count": int(input_route[2].numel()),
        "base_output_coefficient_count": int(output_route[2].numel()),
        "maximum_combined_coefficient_count": coefficient_budget,
        "input_support_width": input_width,
        "output_support_width": output_width,
        "maximum_input_donor_repairs": input_repairs,
        "maximum_output_donor_repairs": output_repairs,
        "position_bin_count": bins,
        "context_hash_bucket_count": context_buckets,
        "context_hash_algorithm": "sha256-prefix-u64-modulo-v1",
        "context_weight": context,
        "solve_ridge": ridge,
        "dtype_contract": {
            "donor_table_storage_dtype": _dtype_name(donor),
            "target_table_storage_dtype": _dtype_name(target),
            "input_feature_storage_dtype": feature["input_feature_dtype"],
            "output_feature_storage_dtype": feature["output_feature_dtype"],
            "solve_dtype": solve_dtype,
            "publication_dtype": "float32",
            "integer_index_dtype": "int64",
            "low_precision_inputs_promoted_before_linear_algebra": True,
        },
        "input_damage_attribution": (
            "prompt-set-intervention-plus-positive-continuation-prefix-increment"
        ),
        "output_damage_attribution": (
            "exact-hidden-folded-readout-margin-loss-winner-and-competitor"
        ),
        "context_cell_aggregation": "maximum-not-occurrence-count",
        "selection_evidence_is_frequency_only": False,
        "input_output_support_searches_independent": True,
        "output_support_ids_inherited_from_input": False,
        "global_residual_map_fit": False,
        "localized_reclosure_only": True,
        "compile_time_features_are_runtime_child_state": False,
        "continuous_runtime_coefficients_count_as_child_state": True,
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
    }
    return {**body, "cache_key": canonical_sha256(body)}


def _position_bin(index: int, count: int, bins: int) -> int:
    if count <= 1:
        return 0
    return min(bins - 1, (index * bins) // count)


def _finite_metric(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise Lfm2CausalSemanticSupportError(f"{name} must be finite.")
    return result


def _isolated_input_damage(
    exact_row: Mapping[str, Any], candidate_row: Mapping[str, Any]
) -> float:
    hidden = _nonnegative_float(
        candidate_row["hidden_vs_same_run_exact_exact"]["relative_l2"],
        "same-run relative L2",
    )
    reference = exact_row["readouts"]["exact_parent_table"]
    candidate = candidate_row["readouts"]["exact_parent_table"]
    reference_margin = _finite_metric(
        reference["parent_winner_margin"], "reference parent-winner margin"
    )
    candidate_margin = _finite_metric(
        candidate["parent_winner_margin"], "candidate parent-winner margin"
    )
    margin_loss = max(0.0, reference_margin - candidate_margin) / (
        1.0 + abs(reference_margin)
    )
    mismatch = float(
        bool(reference["top1_matches_parent"])
        and not bool(candidate["top1_matches_parent"])
    )
    return hidden * hidden + margin_loss + mismatch


def _isolated_output_damage(exact_row: Mapping[str, Any]) -> float:
    reference = exact_row["readouts"]["exact_parent_table"]
    candidate = exact_row["readouts"]["folded_c_out"]
    if not bool(reference["top1_matches_parent"]):
        return 0.0
    reference_margin = _finite_metric(
        reference["parent_winner_margin"], "reference output margin"
    )
    candidate_margin = _finite_metric(
        candidate["parent_winner_margin"], "folded output margin"
    )
    margin_loss = max(0.0, reference_margin - candidate_margin) / (
        1.0 + abs(reference_margin)
    )
    mismatch = float(not bool(candidate["top1_matches_parent"]))
    return margin_loss + mismatch


def _add_damage(
    cells: dict[int, dict[str, float]], donor_id: int, context: str, damage: float
) -> None:
    if damage <= 0.0:
        return
    donor_cells = cells.setdefault(donor_id, {})
    # Maximum makes repeated token occurrences unable to manufacture priority.
    donor_cells[context] = max(donor_cells.get(context, 0.0), damage)


def _collect_causal_damage(
    *,
    results: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Sequence[int]],
    fit_prompt_ids: Sequence[str],
    donor_count: int,
    position_bin_count: int,
) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]], dict[str, Any]]:
    input_cells: dict[int, dict[str, float]] = {}
    output_cells: dict[int, dict[str, float]] = {}
    prompt_set_interventions = 0
    continuation_increments = 0
    output_state_interventions = 0
    for prompt_id in fit_prompt_ids:
        result = results[prompt_id]
        prompt_ids = [int(value) for value in manifest[prompt_id]]
        state_count = int(result["state_count"])
        exact = result["lanes"]["exact_exact"]["tokens"]
        folded_prompt = result["lanes"]["folded_exact"]["tokens"]
        folded_continuation = result["lanes"]["exact_folded"]["tokens"]
        references = result["parent_reference_tokens"]
        if (
            len(exact) != state_count
            or len(folded_prompt) != state_count
            or len(folded_continuation) != state_count
            or len(references) != state_count
        ):
            raise Lfm2CausalSemanticSupportError(
                "Trajectory token rows do not match state_count."
            )
        forced_ids = [int(row["next_forced_token_id"]) for row in references[:-1]]
        if references[-1].get("next_forced_token_id") is not None:
            raise Lfm2CausalSemanticSupportError(
                "Final trajectory reference unexpectedly has a forced token."
            )
        if any(token_id < 0 or token_id >= donor_count for token_id in forced_ids):
            raise Lfm2CausalSemanticSupportError(
                "Trajectory contains an invalid forced donor token."
            )

        prompt_damage = (
            sum(
                _isolated_input_damage(reference, candidate)
                for reference, candidate in zip(exact, folded_prompt, strict=True)
            )
            / state_count
        )
        per_prompt_token = prompt_damage / len(prompt_ids)
        for position, donor_id in enumerate(prompt_ids):
            context = (
                f"{prompt_id}:prompt:"
                f"{_position_bin(position, len(prompt_ids), position_bin_count)}"
            )
            _add_damage(input_cells, donor_id, context, per_prompt_token)
        prompt_set_interventions += 1

        cumulative = [
            _isolated_input_damage(reference, candidate)
            for reference, candidate in zip(exact, folded_continuation, strict=True)
        ]
        for position, donor_id in enumerate(forced_ids):
            increment = max(0.0, cumulative[position + 1] - cumulative[position])
            context = (
                f"{prompt_id}:continuation:"
                f"{_position_bin(position, len(forced_ids), position_bin_count)}"
            )
            _add_damage(input_cells, donor_id, context, increment)
            continuation_increments += 1

        for position, row in enumerate(exact):
            damage = _isolated_output_damage(row)
            if damage <= 0.0:
                continue
            context = (
                f"{prompt_id}:output:"
                f"{_position_bin(position, state_count, position_bin_count)}"
            )
            winner = int(row["parent_winner_token_id"])
            competitor = int(row["readouts"]["folded_c_out"]["best_competing_token_id"])
            if not (0 <= winner < donor_count and 0 <= competitor < donor_count):
                raise Lfm2CausalSemanticSupportError(
                    "Output intervention references an invalid donor row."
                )
            _add_damage(output_cells, winner, context, damage)
            if competitor != winner:
                _add_damage(output_cells, competitor, context, damage)
            output_state_interventions += 1
    diagnostics = {
        "fit_prompt_count": len(fit_prompt_ids),
        "prompt_set_intervention_count": prompt_set_interventions,
        "continuation_prefix_increment_count": continuation_increments,
        "damaged_input_donor_count": len(input_cells),
        "output_state_intervention_count": output_state_interventions,
        "damaged_output_donor_count": len(output_cells),
        "context_cell_aggregation": "maximum-not-occurrence-count",
        "raw_occurrence_frequency_used_as_priority": False,
    }
    return input_cells, output_cells, diagnostics


def _hashed_context_rows(
    cells: Mapping[int, Mapping[str, float]],
    donor_ids: Sequence[int] | Tensor,
    bucket_count: int,
) -> Tensor:
    ids = (
        donor_ids.detach().cpu().reshape(-1).tolist()
        if isinstance(donor_ids, Tensor)
        else list(donor_ids)
    )
    if not cells:
        return torch.zeros((len(ids), 0), dtype=torch.float64)
    result = torch.zeros((len(ids), bucket_count), dtype=torch.float64)
    for row_index, donor_id in enumerate(ids):
        for key, value in cells.get(int(donor_id), {}).items():
            digest = hashlib.sha256(key.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big") % bucket_count
            # Collision aggregation remains a maximum; neither repeated tokens
            # nor hash collisions can manufacture frequency mass.
            result[row_index, bucket] = max(
                float(result[row_index, bucket].item()), float(value)
            )
    norms = torch.linalg.vector_norm(result, dim=1, keepdim=True)
    result = torch.where(norms > 0.0, result / norms.clamp_min(1e-300), result)
    return result


def _rank_damaged_donors(
    cells: Mapping[int, Mapping[str, float]],
    direct_donors: set[int],
    maximum_count: int,
) -> list[int]:
    scored = [
        (sum(row.values()), donor_id)
        for donor_id, row in cells.items()
        if donor_id not in direct_donors and sum(row.values()) > 0.0
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [donor_id for _, donor_id in scored[:maximum_count]]


def _local_ridge(
    atoms: Tensor, target: Tensor, ridge: float, *, solve_dtype: torch.dtype
) -> Tensor:
    design = atoms.to(dtype=solve_dtype)
    vector = target.to(dtype=solve_dtype)
    gram = design @ design.transpose(0, 1)
    gram.diagonal().add_(ridge)
    rhs = design @ vector
    try:
        return torch.linalg.solve(gram, rhs)
    except RuntimeError:
        return torch.linalg.lstsq(gram, rhs.unsqueeze(1)).solution[:, 0]


def _select_support(
    *,
    donor_id: int,
    donor_features: Tensor,
    target_features: Tensor,
    donor_context: Tensor,
    target_context: Tensor,
    support_width: int,
    context_weight: float,
    ridge: float,
    solve_dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    target_feature = donor_features[donor_id]
    feature_norm = float(torch.linalg.vector_norm(target_feature).item())
    context_scale = max(feature_norm, 1.0) * math.sqrt(context_weight)
    if donor_context.numel() == 0 or context_weight == 0.0:
        augmented_target = target_feature
        augmented_atoms = target_features
    else:
        augmented_target = torch.cat((target_feature, donor_context * context_scale))
        augmented_atoms = torch.cat(
            (target_features, target_context * context_scale), dim=1
        )
    residual = augmented_target.clone()
    selected: list[int] = []
    atom_norms = torch.linalg.vector_norm(augmented_atoms, dim=1)
    for _ in range(support_width):
        scores = torch.abs(augmented_atoms @ residual) / atom_norms.clamp_min(1e-300)
        if selected:
            scores[torch.tensor(selected, dtype=torch.int64)] = -1.0
        target_id = int(torch.argmax(scores).item())
        if float(scores[target_id].item()) < 0.0:
            break
        selected.append(target_id)
        augmented_coefficients = _local_ridge(
            augmented_atoms.index_select(0, torch.tensor(selected, dtype=torch.int64)),
            augmented_target,
            ridge,
            solve_dtype=solve_dtype,
        )
        residual = augmented_target - augmented_coefficients.to(
            dtype=augmented_target.dtype
        ) @ augmented_atoms.index_select(0, torch.tensor(selected, dtype=torch.int64))
    if not selected:
        raise Lfm2CausalSemanticSupportError(
            "Residual OMP failed to select a nonempty support."
        )
    support = torch.tensor(selected, dtype=torch.int64)
    # Context decides *which* local atoms to use.  Runtime coefficients are
    # reclosed only against channel features because context evidence is not a
    # runtime model input and must not become an uncounted sidecar.
    coefficients = _local_ridge(
        target_features.index_select(0, support),
        target_feature,
        ridge,
        solve_dtype=solve_dtype,
    ).to(dtype=torch.float32)
    if not bool(torch.isfinite(coefficients).all()):
        raise Lfm2CausalSemanticSupportError(
            "Localized support reclosure produced non-finite coefficients."
        )
    return support, coefficients


def _materialize_route(
    base: tuple[Tensor, Tensor, Tensor],
    replacements: Mapping[int, tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor]:
    base_offsets, base_ids, base_coefficients = base
    offsets = [0]
    ids: list[Tensor] = []
    coefficients: list[Tensor] = []
    for donor_id in range(int(base_offsets.numel()) - 1):
        if donor_id in replacements:
            row_ids, row_coefficients = replacements[donor_id]
        else:
            start = int(base_offsets[donor_id].item())
            stop = int(base_offsets[donor_id + 1].item())
            row_ids = base_ids[start:stop]
            row_coefficients = base_coefficients[start:stop]
        ids.append(row_ids.clone())
        coefficients.append(row_coefficients.clone())
        offsets.append(offsets[-1] + int(row_ids.numel()))
    return (
        torch.tensor(offsets, dtype=torch.int64),
        torch.cat(ids).to(dtype=torch.int64).contiguous(),
        torch.cat(coefficients).to(dtype=torch.float32).contiguous(),
    )


@dataclass(frozen=True, slots=True)
class Lfm2CausalSupportSelection:
    """Authenticated full C_in/C_out routes after localized causal reclosure."""

    input_offsets: Tensor
    input_target_ids: Tensor
    input_coefficients: Tensor
    output_offsets: Tensor
    output_target_ids: Tensor
    output_coefficients: Tensor
    input_reselected_donor_ids: Tensor
    output_reselected_donor_ids: Tensor
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        value = _plain_json(self.receipt)
        supplied = value.pop("result_sha256", None)
        request = _validated_request(value.get("request", {}))
        donor_count = int(request["donor_row_count"])
        target_count = int(request["target_row_count"])
        input_route = _validated_route(
            self.input_offsets,
            self.input_target_ids,
            self.input_coefficients,
            donor_count=donor_count,
            target_count=target_count,
            name="selected input route",
        )
        output_route = _validated_route(
            self.output_offsets,
            self.output_target_ids,
            self.output_coefficients,
            donor_count=donor_count,
            target_count=target_count,
            name="selected output route",
        )
        input_donors = _validated_int64_vector(
            self.input_reselected_donor_ids, "input_reselected_donor_ids"
        )
        output_donors = _validated_int64_vector(
            self.output_reselected_donor_ids, "output_reselected_donor_ids"
        )
        if (
            supplied != canonical_sha256(value)
            or value.get("schema") != LFM2_CAUSAL_SUPPORT_SELECTION_SCHEMA
            or value.get("cache_key") != request["cache_key"]
            or value.get("input_offsets_sha256") != _tensor_sha256(input_route[0])
            or value.get("input_target_ids_sha256") != _tensor_sha256(input_route[1])
            or value.get("input_coefficients_sha256") != _tensor_sha256(input_route[2])
            or value.get("output_offsets_sha256") != _tensor_sha256(output_route[0])
            or value.get("output_target_ids_sha256") != _tensor_sha256(output_route[1])
            or value.get("output_coefficients_sha256")
            != _tensor_sha256(output_route[2])
            or value.get("input_reselected_donor_ids_sha256")
            != _tensor_sha256(input_donors)
            or value.get("output_reselected_donor_ids_sha256")
            != _tensor_sha256(output_donors)
            or value.get("input_coefficient_count") != int(input_route[2].numel())
            or value.get("output_coefficient_count") != int(output_route[2].numel())
            or value.get("continuous_child_state_count")
            != int(input_route[2].numel() + output_route[2].numel())
            or value.get("continuous_child_state_bytes")
            != 4 * int(input_route[2].numel() + output_route[2].numel())
            or value.get("continuous_child_state_count")
            > request["maximum_combined_coefficient_count"]
            or value.get("input_output_coefficients_counted_independently") is not True
            or value.get("input_output_support_searches_independent") is not True
            or value.get("output_support_ids_inherited_from_input") is not False
            or value.get("localized_reclosure_only") is not True
            or value.get("selection_and_manual_review_prompt_ids_disjoint") is not True
            or value.get("operation_audit") != _ZERO_OPERATION_AUDIT
        ):
            raise Lfm2CausalSemanticSupportError(
                "Causal support selection tensor binding is invalid."
            )
        if (
            bool((input_donors < 0).any())
            or bool((input_donors >= donor_count).any())
            or bool((output_donors < 0).any())
            or bool((output_donors >= donor_count).any())
            or not torch.equal(input_donors, torch.unique(input_donors, sorted=True))
            or not torch.equal(output_donors, torch.unique(output_donors, sorted=True))
        ):
            raise Lfm2CausalSemanticSupportError(
                "Reselected donor inventories are invalid."
            )
        object.__setattr__(self, "input_offsets", input_route[0])
        object.__setattr__(self, "input_target_ids", input_route[1])
        object.__setattr__(self, "input_coefficients", input_route[2])
        object.__setattr__(self, "output_offsets", output_route[0])
        object.__setattr__(self, "output_target_ids", output_route[1])
        object.__setattr__(self, "output_coefficients", output_route[2])
        object.__setattr__(self, "input_reselected_donor_ids", input_donors)
        object.__setattr__(self, "output_reselected_donor_ids", output_donors)
        object.__setattr__(
            self, "receipt", _deep_freeze({**value, "result_sha256": supplied})
        )

    @property
    def cache_key(self) -> str:
        return str(self.receipt["cache_key"])

    @property
    def result_sha256(self) -> str:
        return str(self.receipt["result_sha256"])

    def as_dict(self) -> dict[str, Any]:
        return _plain_json(self.receipt)


def select_lfm2_causal_semantic_supports(
    *,
    trajectory_audit: Mapping[str, Any],
    feature_pack: Lfm2CausalSupportFeaturePack,
    prompt_token_ids_by_prompt: Mapping[str, Sequence[int]],
    donor_table: Tensor,
    target_table: Tensor,
    target_to_donor: Tensor,
    base_input_offsets: Tensor,
    base_input_target_ids: Tensor,
    base_input_coefficients: Tensor,
    base_output_offsets: Tensor,
    base_output_target_ids: Tensor,
    base_output_coefficients: Tensor,
    physical_child_parameter_count: int,
    maximum_combined_coefficient_count: int,
    input_support_width: int,
    output_support_width: int,
    maximum_input_donor_repairs: int,
    maximum_output_donor_repairs: int,
    position_bin_count: int = 4,
    context_hash_bucket_count: int = 256,
    context_weight: float = 0.25,
    solve_ridge: float = 1e-8,
    solve_dtype: str = "float64",
) -> Lfm2CausalSupportSelection:
    """Select independent supports and reclose only causally damaged rows."""

    arguments = dict(
        trajectory_audit=trajectory_audit,
        feature_pack=feature_pack,
        prompt_token_ids_by_prompt=prompt_token_ids_by_prompt,
        donor_table=donor_table,
        target_table=target_table,
        target_to_donor=target_to_donor,
        base_input_offsets=base_input_offsets,
        base_input_target_ids=base_input_target_ids,
        base_input_coefficients=base_input_coefficients,
        base_output_offsets=base_output_offsets,
        base_output_target_ids=base_output_target_ids,
        base_output_coefficients=base_output_coefficients,
        physical_child_parameter_count=physical_child_parameter_count,
        maximum_combined_coefficient_count=maximum_combined_coefficient_count,
        input_support_width=input_support_width,
        output_support_width=output_support_width,
        maximum_input_donor_repairs=maximum_input_donor_repairs,
        maximum_output_donor_repairs=maximum_output_donor_repairs,
        position_bin_count=position_bin_count,
        context_hash_bucket_count=context_hash_bucket_count,
        context_weight=context_weight,
        solve_ridge=solve_ridge,
        solve_dtype=solve_dtype,
    )
    request = lfm2_causal_support_request(**arguments)
    audit, results = _authenticated_audit(trajectory_audit)
    del audit
    donor = _validated_float_matrix(donor_table, "donor_table").to(torch.float64)
    target = _validated_float_matrix(target_table, "target_table").to(torch.float64)
    mapping = _validated_int64_vector(target_to_donor, "target_to_donor")
    donor_count = int(donor.shape[0])
    target_count = int(target.shape[0])
    input_base = _validated_route(
        base_input_offsets,
        base_input_target_ids,
        base_input_coefficients,
        donor_count=donor_count,
        target_count=target_count,
        name="base input route",
    )
    output_base = _validated_route(
        base_output_offsets,
        base_output_target_ids,
        base_output_coefficients,
        donor_count=donor_count,
        target_count=target_count,
        name="base output route",
    )
    input_cells, output_cells, damage_diagnostics = _collect_causal_damage(
        results=results,
        manifest=request["prompt_token_manifest"],
        fit_prompt_ids=request["fit_prompt_ids"],
        donor_count=donor_count,
        position_bin_count=request["position_bin_count"],
    )
    direct_donors = set(mapping.tolist())
    input_priority = _rank_damaged_donors(
        input_cells,
        direct_donors,
        request["maximum_input_donor_repairs"],
    )
    output_priority = _rank_damaged_donors(
        output_cells,
        direct_donors,
        request["maximum_output_donor_repairs"],
    )
    input_context_keys = tuple(
        sorted({key for row in input_cells.values() for key in row})
    )
    output_context_keys = tuple(
        sorted({key for row in output_cells.values() for key in row})
    )
    context_buckets = request["context_hash_bucket_count"]
    input_target_context = _hashed_context_rows(input_cells, mapping, context_buckets)
    output_target_context = _hashed_context_rows(output_cells, mapping, context_buckets)
    dtype = (
        torch.float64
        if request["dtype_contract"]["solve_dtype"] == "float64"
        else torch.float32
    )
    input_transform = feature_pack.input_feature_transform.to(dtype=dtype)
    output_transform = feature_pack.output_feature_transform.to(dtype=dtype)
    donor_input_features = donor.to(dtype=dtype) @ input_transform.transpose(0, 1)
    target_input_features = target.to(dtype=dtype) @ input_transform.transpose(0, 1)
    donor_output_features = donor.to(dtype=dtype) @ output_transform.transpose(0, 1)
    target_output_features = target.to(dtype=dtype) @ output_transform.transpose(0, 1)

    input_replacements: dict[int, tuple[Tensor, Tensor]] = {}
    for donor_id in input_priority:
        input_replacements[donor_id] = _select_support(
            donor_id=donor_id,
            donor_features=donor_input_features,
            target_features=target_input_features,
            donor_context=_hashed_context_rows(
                input_cells, [donor_id], context_buckets
            )[0],
            target_context=input_target_context,
            support_width=request["input_support_width"],
            context_weight=request["context_weight"],
            ridge=request["solve_ridge"],
            solve_dtype=dtype,
        )
    output_replacements: dict[int, tuple[Tensor, Tensor]] = {}
    for donor_id in output_priority:
        output_replacements[donor_id] = _select_support(
            donor_id=donor_id,
            donor_features=donor_output_features,
            target_features=target_output_features,
            donor_context=_hashed_context_rows(
                output_cells, [donor_id], context_buckets
            )[0],
            target_context=output_target_context,
            support_width=request["output_support_width"],
            context_weight=request["context_weight"],
            ridge=request["solve_ridge"],
            solve_dtype=dtype,
        )
    input_route = _materialize_route(input_base, input_replacements)
    output_route = _materialize_route(output_base, output_replacements)
    _verify_direct_routes(input_route, mapping, name="selected input route")
    _verify_direct_routes(output_route, mapping, name="selected output route")
    input_donors = torch.tensor(sorted(input_replacements), dtype=torch.int64)
    output_donors = torch.tensor(sorted(output_replacements), dtype=torch.int64)
    coefficient_count = int(input_route[2].numel() + output_route[2].numel())
    if coefficient_count > request["maximum_combined_coefficient_count"]:
        raise Lfm2CausalSemanticSupportError(
            "Localized route selection exceeds the authenticated coefficient budget."
        )
    coincident_support_rows = 0
    for donor_id in set(input_replacements).intersection(output_replacements):
        if torch.equal(
            input_replacements[donor_id][0], output_replacements[donor_id][0]
        ):
            coincident_support_rows += 1
    body = {
        "schema": LFM2_CAUSAL_SUPPORT_SELECTION_SCHEMA,
        "status": "complete-independent-causal-local-support-reclosure",
        "cache_key": request["cache_key"],
        "request": request,
        "input_offsets_sha256": _tensor_sha256(input_route[0]),
        "input_target_ids_sha256": _tensor_sha256(input_route[1]),
        "input_coefficients_sha256": _tensor_sha256(input_route[2]),
        "output_offsets_sha256": _tensor_sha256(output_route[0]),
        "output_target_ids_sha256": _tensor_sha256(output_route[1]),
        "output_coefficients_sha256": _tensor_sha256(output_route[2]),
        "input_reselected_donor_ids_sha256": _tensor_sha256(input_donors),
        "output_reselected_donor_ids_sha256": _tensor_sha256(output_donors),
        "input_reselected_donor_ids": input_donors.tolist(),
        "output_reselected_donor_ids": output_donors.tolist(),
        "input_reselected_donor_count": int(input_donors.numel()),
        "output_reselected_donor_count": int(output_donors.numel()),
        "input_support_search_call_count": len(input_replacements),
        "output_support_search_call_count": len(output_replacements),
        "input_output_support_searches_independent": True,
        "output_support_ids_inherited_from_input": False,
        "coincident_support_rows_after_independent_search": coincident_support_rows,
        "coincident_supports_are_allowed_but_never_copied": True,
        "input_context_dimension_count": (context_buckets if input_context_keys else 0),
        "output_context_dimension_count": (
            context_buckets if output_context_keys else 0
        ),
        "input_observed_context_key_count": len(input_context_keys),
        "output_observed_context_key_count": len(output_context_keys),
        "input_context_keys_sha256": canonical_sha256(list(input_context_keys)),
        "output_context_keys_sha256": canonical_sha256(list(output_context_keys)),
        "damage_diagnostics": damage_diagnostics,
        "selection_and_manual_review_prompt_ids_disjoint": True,
        "manual_review_prompts_consumed": 0,
        "selection_evidence_is_frequency_only": False,
        "localized_reclosure_only": True,
        "unselected_routes_preserved_from_base": True,
        "runtime_context_sidecar_required": False,
        "compile_time_features_are_runtime_child_state": False,
        "input_coefficient_count": int(input_route[2].numel()),
        "output_coefficient_count": int(output_route[2].numel()),
        "continuous_child_state_count": coefficient_count,
        "continuous_child_state_bytes": coefficient_count * 4,
        "input_output_coefficients_counted_independently": True,
        "physical_child_parameter_count": request["physical_child_parameter_count"],
        "effective_child_parameter_count": (
            request["physical_child_parameter_count"] + coefficient_count
        ),
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
    }
    return Lfm2CausalSupportSelection(
        input_route[0],
        input_route[1],
        input_route[2],
        output_route[0],
        output_route[1],
        output_route[2],
        input_donors,
        output_donors,
        {**body, "result_sha256": canonical_sha256(body)},
    )


def authenticate_lfm2_causal_support_selection(
    *,
    input_offsets: Tensor,
    input_target_ids: Tensor,
    input_coefficients: Tensor,
    output_offsets: Tensor,
    output_target_ids: Tensor,
    output_coefficients: Tensor,
    input_reselected_donor_ids: Tensor,
    output_reselected_donor_ids: Tensor,
    receipt: Mapping[str, Any],
    expected_cache_key: str,
) -> Lfm2CausalSupportSelection:
    """Authenticate a loaded content-addressed result before cache reuse."""

    expected = _sha256(expected_cache_key, "expected_cache_key")
    result = Lfm2CausalSupportSelection(
        input_offsets,
        input_target_ids,
        input_coefficients,
        output_offsets,
        output_target_ids,
        output_coefficients,
        input_reselected_donor_ids,
        output_reselected_donor_ids,
        receipt,
    )
    if result.cache_key != expected:
        raise Lfm2CausalSemanticSupportError(
            "Cached causal support selection belongs to different math."
        )
    return result


__all__ = [
    "LFM2_CAUSAL_SUPPORT_ALGORITHM",
    "LFM2_CAUSAL_SUPPORT_FEATURES_SCHEMA",
    "LFM2_CAUSAL_SUPPORT_REQUEST_SCHEMA",
    "LFM2_CAUSAL_SUPPORT_SELECTION_SCHEMA",
    "Lfm2CausalSemanticSupportError",
    "Lfm2CausalSupportFeaturePack",
    "Lfm2CausalSupportSelection",
    "authenticate_lfm2_causal_support_selection",
    "lfm2_causal_support_request",
    "seal_lfm2_causal_support_feature_pack",
    "select_lfm2_causal_semantic_supports",
]
