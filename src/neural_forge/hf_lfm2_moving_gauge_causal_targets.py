"""Authenticated state-conditioned targets for LFM2 residual folding.

The moving-gauge compiler does not assign a universal coordinate system to
operator and MLP branch deltas.  It assigns coordinate charts to residual
*endpoints*.  A physical writer target therefore has to be computed from an
encoded downstream endpoint and the actual upstream child state::

    writer = E_downstream(parent_endpoint) - actual_child_upstream

This module makes that contract explicit for two complementary diagnostics:

``joint``
    Faithfully reproduces the existing R12 rule by treating the operator and
    MLP writers as one additive group aimed at the next post-layer boundary.

``sequential``
    Adds an authenticated post-operator boundary chart.  The operator is aimed
    at that endpoint first; the MLP is then aimed at the post-layer endpoint
    from a freshly observed child post-operator state.  This is the concrete
    residual-attention folding contract.

Only authenticated :class:`Lfm2MovingGaugeBoundaryChart` objects are accepted.
A raw balanced/global basis object can therefore be used only after it has been
bound into an authenticated locked input or terminal chart; it can never be
used directly as a branch-delta projector.  All arithmetic is detached FP64 on
CPU.  There is no model execution, mutation, autograd, optimizer, or training.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import torch

from .dynamic_circuit_ir import canonical_sha256
from .hf_lfm2_moving_gauge_residual_reclosure import (
    Lfm2MovingGaugeBoundaryChart,
    Lfm2MovingGaugeBoundaryEvidence,
    fingerprint_lfm2_moving_gauge_tensor,
)


LFM2_MOVING_GAUGE_CAUSAL_TARGETS_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-causal-targets-v1"
)
LFM2_MOVING_GAUGE_CAUSAL_STATE_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-causal-state-receipt-v1"
)
LFM2_LAYER_TOPOLOGY_SCHEMA = "neural-forge-lfm2-layer-topology-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*\Z")
_ZERO_EXECUTION = {
    "parent_model_objects_constructed": 0,
    "parent_model_forward_calls": 0,
    "parent_trace_recaptures": 0,
    "child_model_forward_calls": 0,
    "model_mutations": 0,
    "backward_calls": 0,
    "autograd_grad_calls": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "training_steps": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
}
_STATE_RECEIPT_BODY_FIELDS = frozenset(
    {
        "schema",
        "kind",
        "plan_sha256",
        "layer_index",
        "operator_kind",
        "parent_cache_sha256",
        "base_child_state_sha256",
        "child_recapture_sha256",
        "row_alignment_sha256",
        "operator_solution_sha256",
        "tensor_sha256s",
        "outer_hmac_checkpoint_required",
        "outer_hmac_checkpoint_verified",
        "execution",
    }
)
_TARGET_REPORT_BODY_FIELDS_BY_KIND = {
    "state_conditioned_causal_boundary_plan": frozenset(
        {
            "schema",
            "kind",
            "layer_index",
            "layer_count",
            "is_final_layer",
            "operator_kind",
            "source_width",
            "target_width",
            "available_target_modes",
            "chart_ids",
            "chart_report_sha256s",
            "boundary_evidence_sha256s",
            "parent_cache_sha256",
            "child_state_sha256",
            "child_recapture_sha256",
            "row_alignment_sha256",
            "split_identity_sha256",
            "layer_topology_sha256",
            "base_encoder_sha256",
            "locked_global_map_allowed_only_for_residual_boundary_state",
            "raw_global_writer_projection_allowed",
            "joint_target_formula",
            "sequential_operator_target_formula",
            "sequential_mlp_target_formula",
            "fit_only",
            "numeric_quality_gate",
            "outer_hmac_checkpoint_required",
            "outer_hmac_checkpoint_verified",
            "execution",
        }
    ),
    "joint_writer_target": frozenset(
        {
            "schema",
            "kind",
            "plan_sha256",
            "layer_index",
            "operator_kind",
            "post_layer_chart_sha256",
            "state_receipt_sha256",
            "desired_total_formula",
            "fixed_other_formula",
            "physical_writer_formula",
            "tensor_sha256s",
            "solution_installed",
            "fit_only",
            "outer_hmac_checkpoint_required",
            "outer_hmac_checkpoint_verified",
            "execution",
        }
    ),
    "sequential_writer_target": frozenset(
        {
            "schema",
            "kind",
            "plan_sha256",
            "layer_index",
            "operator_kind",
            "post_operator_chart_sha256",
            "post_layer_chart_sha256",
            "operator_desired_total_formula",
            "operator_fixed_other_formula",
            "operator_physical_writer_formula",
            "mlp_desired_total_formula",
            "mlp_fixed_other_formula",
            "mlp_physical_writer_formula",
            "fresh_post_operator_state_required",
            "operator_source_receipt_sha256",
            "mlp_source_receipt_sha256",
            "fresh_recapture_kind",
            "operator_solution_sha256",
            "evidence_stage",
            "operator_solution_installed",
            "target_writer_solutions_installed",
            "full_post_layer_reclosure_verified",
            "fixed_other_writes_derived_from_observed_endpoint_equations",
            "raw_parent_branch_delta_projection_used",
            "tensor_sha256s",
            "fit_only",
            "outer_hmac_checkpoint_required",
            "outer_hmac_checkpoint_verified",
            "execution",
        }
    ),
}


class Lfm2MovingGaugeCausalTargetError(ValueError):
    """A causal chart plan or state-conditioned target is invalid."""


class Lfm2MovingGaugeCausalTargetAuthenticationError(Lfm2MovingGaugeCausalTargetError):
    """A causal chart, cache identity, or target receipt failed authentication."""


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2MovingGaugeCausalTargetAuthenticationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _identifier(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 256
        or _IDENTIFIER.fullmatch(value) is None
    ):
        raise Lfm2MovingGaugeCausalTargetError(
            f"{name} must be a nonempty bounded stable identifier."
        )
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2MovingGaugeCausalTargetError(f"{name} must be a nonnegative integer.")
    return value


def _seal(body: Mapping[str, Any]) -> Mapping[str, Any]:
    detached = dict(body)
    return MappingProxyType({**detached, "result_sha256": canonical_sha256(detached)})


def _plain_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2MovingGaugeCausalTargetAuthenticationError(
            f"{name} must be a canonical mapping."
        )
    try:
        encoded = json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        result = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise Lfm2MovingGaugeCausalTargetAuthenticationError(
            f"{name} is not canonical JSON."
        ) from exc
    if not isinstance(result, dict):  # pragma: no cover - Mapping encoded above
        raise Lfm2MovingGaugeCausalTargetAuthenticationError(
            f"{name} must encode a JSON object."
        )
    return result


def _layer_topology(
    value: Any,
    *,
    layer_index: int,
    layer_count: int,
    operator_kind: str,
) -> tuple[Mapping[str, Any], str]:
    topology = _plain_mapping(value, "layer_topology")
    claimed = topology.pop("layer_topology_sha256", None)
    topology_sha256 = _digest(claimed, "layer_topology_sha256")
    if (
        topology.get("schema") != LFM2_LAYER_TOPOLOGY_SCHEMA
        or topology.get("layer_count") != layer_count
        or canonical_sha256(topology) != topology_sha256
    ):
        raise Lfm2MovingGaugeCausalTargetAuthenticationError(
            "Layer topology seal or layer count does not verify."
        )
    layers = topology.get("layers")
    if not isinstance(layers, list) or len(layers) != layer_count:
        raise Lfm2MovingGaugeCausalTargetAuthenticationError(
            "Layer topology does not contain its declared layer sequence."
        )
    for index, entry in enumerate(layers):
        if not isinstance(entry, dict) or entry.get("container_index") != index:
            raise Lfm2MovingGaugeCausalTargetAuthenticationError(
                "Layer topology container order is noncanonical."
            )
        contract = entry.get("contract")
        if not isinstance(contract, dict) or contract.get("layer_index") != index:
            raise Lfm2MovingGaugeCausalTargetAuthenticationError(
                "Layer topology contract order is noncanonical."
            )
    if layers[layer_index]["contract"].get("kind") != operator_kind:
        raise Lfm2MovingGaugeCausalTargetAuthenticationError(
            "operator_kind differs from the sealed layer topology."
        )
    authenticated = MappingProxyType(
        {**topology, "layer_topology_sha256": topology_sha256}
    )
    return authenticated, topology_sha256


def _expected_boundary_ids(
    layer_index: int,
    layer_count: int,
    *,
    sequential: bool,
) -> dict[str, str]:
    result = {
        "pre": f"layer.{layer_index:03d}.residual.pre",
        "post_layer": (
            "stack.terminal.residual"
            if layer_index == layer_count - 1
            else f"layer.{layer_index:03d}.residual.post_layer"
        ),
        "terminal": "stack.terminal.residual",
    }
    if sequential:
        result["post_operator"] = f"layer.{layer_index:03d}.residual.post_operator"
    return result


def _report_digest_mapping(
    value: Any,
    *,
    exact_keys: set[str] | None = None,
) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    if exact_keys is not None and set(value) != exact_keys:
        return False
    return all(
        isinstance(name, str)
        and bool(name)
        and isinstance(digest, str)
        and _SHA256.fullmatch(digest) is not None
        for name, digest in value.items()
    )


def _verify_causal_report_semantics(body: Mapping[str, Any], kind: str) -> bool:
    operator_kind = body.get("operator_kind")
    layer_index = body.get("layer_index")
    common = (
        isinstance(operator_kind, str)
        and _IDENTIFIER.fullmatch(operator_kind) is not None
        and isinstance(layer_index, int)
        and not isinstance(layer_index, bool)
        and layer_index >= 0
    )
    if not common:
        return False
    if kind == "state_conditioned_causal_boundary_plan":
        layer_count = body.get("layer_count")
        source_width = body.get("source_width")
        target_width = body.get("target_width")
        modes = body.get("available_target_modes")
        if (
            not isinstance(layer_count, int)
            or isinstance(layer_count, bool)
            or layer_count <= 0
            or layer_index >= layer_count
            or not isinstance(source_width, int)
            or isinstance(source_width, bool)
            or source_width <= 0
            or not isinstance(target_width, int)
            or isinstance(target_width, bool)
            or target_width <= 0
            or modes not in (["joint"], ["joint", "sequential"])
        ):
            return False
        sequential = modes == ["joint", "sequential"]
        expected_ids = _expected_boundary_ids(
            layer_index,
            layer_count,
            sequential=sequential,
        )
        root_digests = (
            "parent_cache_sha256",
            "child_state_sha256",
            "child_recapture_sha256",
            "row_alignment_sha256",
            "split_identity_sha256",
            "layer_topology_sha256",
            "base_encoder_sha256",
        )
        return bool(
            body.get("is_final_layer") is (layer_index == layer_count - 1)
            and body.get("chart_ids") == expected_ids
            and _report_digest_mapping(
                body.get("chart_report_sha256s"),
                exact_keys=set(expected_ids),
            )
            and _report_digest_mapping(
                body.get("boundary_evidence_sha256s"),
                exact_keys=set(expected_ids),
            )
            and all(
                isinstance(body.get(name), str)
                and _SHA256.fullmatch(body[name]) is not None
                for name in root_digests
            )
            and body.get("joint_target_formula")
            == "E_post_layer(parent_post_layer)-actual_child_pre"
            and body.get("sequential_operator_target_formula")
            == (
                "E_post_operator(parent_post_operator)-actual_child_pre"
                if sequential
                else None
            )
            and body.get("sequential_mlp_target_formula")
            == (
                "E_post_layer(parent_post_layer)-fresh_child_post_operator"
                if sequential
                else None
            )
        )
    if kind == "joint_writer_target":
        tensor_names = {
            "parent_post_layer",
            "actual_child_pre",
            "actual_child_post_layer",
            "existing_combined_writer_output",
            "desired_endpoint",
            "desired_total_write",
            "fixed_other_write",
            "target_combined_writer_output",
        }
        return bool(
            all(
                isinstance(body.get(name), str)
                and _SHA256.fullmatch(body[name]) is not None
                for name in (
                    "plan_sha256",
                    "post_layer_chart_sha256",
                    "state_receipt_sha256",
                )
            )
            and body.get("desired_total_formula")
            == "E_post_layer(parent_post_layer)-actual_child_pre"
            and body.get("fixed_other_formula")
            == (
                "actual_child_post_layer-actual_child_pre-"
                "existing_combined_writer_output"
            )
            and body.get("physical_writer_formula")
            == "desired_total_write-fixed_other_write"
            and _report_digest_mapping(
                body.get("tensor_sha256s"),
                exact_keys=tensor_names,
            )
        )
    if kind == "sequential_writer_target":
        tensor_names = {
            "parent_post_operator",
            "actual_child_pre",
            "actual_child_post_operator",
            "existing_operator_writer_output",
            "parent_post_layer",
            "operator_endpoint",
            "desired_operator_total_write",
            "operator_fixed_other_write",
            "target_operator_writer_output",
            "fresh_child_post_operator",
            "fresh_child_post_layer",
            "existing_mlp_writer_output",
            "post_layer_endpoint",
            "desired_mlp_total_write",
            "mlp_fixed_other_write",
            "target_mlp_writer_output",
        }
        recapture_kind = body.get("fresh_recapture_kind")
        solution = body.get("operator_solution_sha256")
        expected_stage = {
            "baseline_current_child": "baseline_diagnostic",
            "operator_conditioned_temporary_intervention": (
                "temporary_operator_mlp_target_ready"
            ),
        }.get(recapture_kind)
        return bool(
            all(
                isinstance(body.get(name), str)
                and _SHA256.fullmatch(body[name]) is not None
                for name in (
                    "plan_sha256",
                    "post_operator_chart_sha256",
                    "post_layer_chart_sha256",
                    "operator_source_receipt_sha256",
                    "mlp_source_receipt_sha256",
                )
            )
            and expected_stage is not None
            and body.get("evidence_stage") == expected_stage
            and (
                (recapture_kind == "baseline_current_child" and solution is None)
                or (
                    recapture_kind == "operator_conditioned_temporary_intervention"
                    and isinstance(solution, str)
                    and _SHA256.fullmatch(solution) is not None
                )
            )
            and body.get("operator_desired_total_formula")
            == "E_post_operator(parent_post_operator)-actual_child_pre"
            and body.get("operator_fixed_other_formula")
            == (
                "actual_child_post_operator-actual_child_pre-"
                "existing_operator_writer_output"
            )
            and body.get("operator_physical_writer_formula")
            == "desired_operator_total_write-operator_fixed_other_write"
            and body.get("mlp_desired_total_formula")
            == "E_post_layer(parent_post_layer)-fresh_child_post_operator"
            and body.get("mlp_fixed_other_formula")
            == (
                "fresh_child_post_layer-fresh_child_post_operator-"
                "existing_mlp_writer_output"
            )
            and body.get("mlp_physical_writer_formula")
            == "desired_mlp_total_write-mlp_fixed_other_write"
            and _report_digest_mapping(
                body.get("tensor_sha256s"),
                exact_keys=tensor_names,
            )
        )
    return False


def verify_lfm2_moving_gauge_causal_target_report(
    value: Mapping[str, Any],
) -> bool:
    """Verify one canonical report emitted by this module."""

    if not isinstance(value, Mapping):
        return False
    try:
        body = dict(value)
        supplied = body.pop("result_sha256", None)
        kind = body.get("kind")
        expected_fields = _TARGET_REPORT_BODY_FIELDS_BY_KIND.get(kind)
        common_contract = (
            body.get("fit_only") is True
            and body.get("outer_hmac_checkpoint_required") is True
            and body.get("outer_hmac_checkpoint_verified") is False
            and body.get("execution") == _ZERO_EXECUTION
        )
        kind_contract = {
            "state_conditioned_causal_boundary_plan": (
                body.get("numeric_quality_gate") is False
                and body.get("raw_global_writer_projection_allowed") is False
                and body.get(
                    "locked_global_map_allowed_only_for_residual_boundary_state"
                )
                is True
            ),
            "joint_writer_target": body.get("solution_installed") is False,
            "sequential_writer_target": (
                body.get("fresh_post_operator_state_required") is True
                and body.get("operator_solution_installed") is False
                and body.get("target_writer_solutions_installed") is False
                and body.get("full_post_layer_reclosure_verified") is False
                and body.get(
                    "fixed_other_writes_derived_from_observed_endpoint_equations"
                )
                is True
                and body.get("raw_parent_branch_delta_projection_used") is False
            ),
        }.get(kind, False)
        return (
            body.get("schema") == LFM2_MOVING_GAUGE_CAUSAL_TARGETS_SCHEMA
            and expected_fields is not None
            and set(body) == expected_fields
            and common_contract
            and kind_contract
            and _verify_causal_report_semantics(body, kind)
            and isinstance(supplied, str)
            and supplied == canonical_sha256(body)
        )
    except (TypeError, ValueError):
        return False


def _chart(value: Any, name: str) -> Lfm2MovingGaugeBoundaryChart:
    if (
        not isinstance(value, Lfm2MovingGaugeBoundaryChart)
        or not value.verify_authentication()
    ):
        raise Lfm2MovingGaugeCausalTargetAuthenticationError(
            f"{name} must be an authenticated moving-gauge boundary chart; "
            "raw global coordinate maps are forbidden at writer endpoints."
        )
    return value


def _state(
    value: Any,
    name: str,
    *,
    width: int,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim < 2
        or int(value.shape[-1]) != width
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or value.is_complex()
        or value.requires_grad
        or value.grad_fn is not None
    ):
        raise Lfm2MovingGaugeCausalTargetError(
            f"{name} must be a detached dense real state with width {width}."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2MovingGaugeCausalTargetError(f"{name} contains NaN or infinity.")
    return result


def _is_canonical_state(value: Any, *, width: int) -> bool:
    return bool(
        isinstance(value, torch.Tensor)
        and value.layout == torch.strided
        and value.device.type == "cpu"
        and value.dtype == torch.float64
        and value.ndim >= 2
        and int(value.shape[-1]) == width
        and value.numel() > 0
        and value.is_contiguous()
        and not value.requires_grad
        and value.grad_fn is None
        and bool(torch.isfinite(value).all().item())
    )


def _roundoff_close(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether one FP64 subtract/add identity closes to roundoff."""

    epsilon = torch.finfo(torch.float64).eps
    return bool(
        torch.allclose(
            left,
            right,
            rtol=8.0 * epsilon,
            atol=8.0 * epsilon,
        )
    )


def _chart_role(chart: Lfm2MovingGaugeBoundaryChart, name: str) -> str:
    role = chart.report.get("interface_role")
    if role not in {"input", "intermediate", "terminal"}:
        raise Lfm2MovingGaugeCausalTargetAuthenticationError(
            f"{name} does not authenticate an interface role."
        )
    return str(role)


_STATE_RECEIPT_KINDS = {
    "baseline_current_child",
    "operator_conditioned_temporary_intervention",
}


def _receipt_tensor_sha256(value: Any, name: str) -> str:
    if not isinstance(value, torch.Tensor) or value.ndim < 2:
        raise Lfm2MovingGaugeCausalTargetError(
            f"State receipt tensor {name!r} must be a residual state tensor."
        )
    canonical = _state(value, name, width=int(value.shape[-1]))
    return fingerprint_lfm2_moving_gauge_tensor(canonical)


def build_lfm2_moving_gauge_causal_state_receipt(
    plan: Lfm2MovingGaugeCausalBoundaryPlan,
    *,
    kind: str,
    tensors: Mapping[str, Any],
    child_recapture_sha256: str | None = None,
    operator_solution_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Seal the exact source tensors for one cached or fresh child state.

    A v5 HMAC checkpoint remains the outer run authority.  This inner receipt
    prevents an unbound digest string from standing in for the tensors used by
    the state-conditioned algebra.
    """

    if not isinstance(plan, Lfm2MovingGaugeCausalBoundaryPlan) or not (
        plan.verify_authentication()
    ):
        raise Lfm2MovingGaugeCausalTargetAuthenticationError(
            "State receipts require an authenticated causal plan."
        )
    receipt_kind = _identifier(kind, "kind")
    if receipt_kind not in _STATE_RECEIPT_KINDS:
        raise Lfm2MovingGaugeCausalTargetError("State receipt kind is unsupported.")
    if not isinstance(tensors, Mapping) or not tensors:
        raise Lfm2MovingGaugeCausalTargetError(
            "State receipt tensors must be a nonempty named mapping."
        )
    raw_names = tuple(tensors)
    if any(not isinstance(name, str) or not name for name in raw_names):
        raise Lfm2MovingGaugeCausalTargetError(
            "State receipt tensor names must be nonempty strings."
        )
    names = tuple(sorted(raw_names))
    tensor_sha256s: dict[str, str] = {}
    for name in names:
        value = tensors[name]
        if not isinstance(value, torch.Tensor):
            raise Lfm2MovingGaugeCausalTargetError(
                f"State receipt tensor {name!r} is not a tensor."
            )
        tensor_sha256s[name] = _receipt_tensor_sha256(value, name)
    if receipt_kind == "baseline_current_child":
        if child_recapture_sha256 not in {None, plan.child_recapture_sha256}:
            raise Lfm2MovingGaugeCausalTargetAuthenticationError(
                "Baseline receipt differs from the chart plan child recapture."
            )
        if operator_solution_sha256 is not None:
            raise Lfm2MovingGaugeCausalTargetAuthenticationError(
                "Baseline receipt cannot claim an operator solution."
            )
        recapture = plan.child_recapture_sha256
        solution = None
    else:
        recapture = _digest(
            child_recapture_sha256,
            "child_recapture_sha256",
        )
        if recapture == plan.child_recapture_sha256:
            raise Lfm2MovingGaugeCausalTargetAuthenticationError(
                "Conditioned or installed evidence requires a fresh recapture."
            )
        solution = _digest(
            operator_solution_sha256,
            "operator_solution_sha256",
        )
    return _seal(
        {
            "schema": LFM2_MOVING_GAUGE_CAUSAL_STATE_RECEIPT_SCHEMA,
            "kind": receipt_kind,
            "plan_sha256": plan.report["result_sha256"],
            "layer_index": plan.layer_index,
            "operator_kind": plan.operator_kind,
            "parent_cache_sha256": plan.parent_cache_sha256,
            "base_child_state_sha256": plan.child_state_sha256,
            "child_recapture_sha256": recapture,
            "row_alignment_sha256": plan.row_alignment_sha256,
            "operator_solution_sha256": solution,
            "tensor_sha256s": tensor_sha256s,
            "outer_hmac_checkpoint_required": True,
            "outer_hmac_checkpoint_verified": False,
            "execution": dict(_ZERO_EXECUTION),
        }
    )


def authenticate_lfm2_moving_gauge_causal_state_receipt(
    value: Mapping[str, Any],
    *,
    plan: Lfm2MovingGaugeCausalBoundaryPlan,
    tensors: Mapping[str, torch.Tensor],
    allowed_kinds: set[str] | frozenset[str],
) -> Mapping[str, Any]:
    """Authenticate a sealed state receipt and all tensors it names."""

    if not isinstance(plan, Lfm2MovingGaugeCausalBoundaryPlan) or not (
        plan.verify_authentication()
    ):
        raise Lfm2MovingGaugeCausalTargetAuthenticationError(
            "State receipt authentication requires an authenticated causal plan."
        )
    if (
        not isinstance(allowed_kinds, (set, frozenset))
        or not allowed_kinds
        or not set(allowed_kinds).issubset(_STATE_RECEIPT_KINDS)
    ):
        raise Lfm2MovingGaugeCausalTargetAuthenticationError(
            "State receipt allowed_kinds exceed the closed receipt-kind inventory."
        )
    row = _plain_mapping(value, "state receipt")
    supplied = row.pop("result_sha256", None)
    if (
        _digest(supplied, "state receipt result_sha256") != canonical_sha256(row)
        or set(row) != _STATE_RECEIPT_BODY_FIELDS
        or row.get("schema") != LFM2_MOVING_GAUGE_CAUSAL_STATE_RECEIPT_SCHEMA
        or row.get("kind") not in allowed_kinds
        or row.get("plan_sha256") != plan.report["result_sha256"]
        or row.get("layer_index") != plan.layer_index
        or row.get("operator_kind") != plan.operator_kind
        or row.get("parent_cache_sha256") != plan.parent_cache_sha256
        or row.get("base_child_state_sha256") != plan.child_state_sha256
        or row.get("row_alignment_sha256") != plan.row_alignment_sha256
        or row.get("outer_hmac_checkpoint_required") is not True
        or row.get("outer_hmac_checkpoint_verified") is not False
        or row.get("execution") != _ZERO_EXECUTION
    ):
        raise Lfm2MovingGaugeCausalTargetAuthenticationError(
            "State receipt seal or plan binding does not verify."
        )
    kind = str(row["kind"])
    recapture = _digest(
        row.get("child_recapture_sha256"),
        "state receipt child_recapture_sha256",
    )
    solution = row.get("operator_solution_sha256")
    if kind == "baseline_current_child":
        if recapture != plan.child_recapture_sha256 or solution is not None:
            raise Lfm2MovingGaugeCausalTargetAuthenticationError(
                "Baseline state receipt has conditioned provenance."
            )
    elif (
        recapture == plan.child_recapture_sha256
        or _digest(solution, "state receipt operator_solution_sha256") != solution
    ):
        raise Lfm2MovingGaugeCausalTargetAuthenticationError(
            "Conditioned state receipt lacks a fresh recapture or solution."
        )
    expected_tensor_sha256s = {
        name: _receipt_tensor_sha256(tensors[name], name) for name in sorted(tensors)
    }
    if row.get("tensor_sha256s") != expected_tensor_sha256s:
        raise Lfm2MovingGaugeCausalTargetAuthenticationError(
            "State receipt does not bind the supplied tensor set."
        )
    return MappingProxyType({**row, "result_sha256": supplied})


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeCausalBoundaryPlan:
    """Authenticated endpoint charts and paired evidence for one child layer.

    ``post_operator_chart`` is optional so the faithful R12 joint baseline can
    be authenticated without inventing an intra-layer gauge.  Its presence
    enables the explicit sequential residual-attention diagnostic.
    """

    layer_index: int
    layer_count: int
    operator_kind: str
    pre_chart: Lfm2MovingGaugeBoundaryChart = field(repr=False, compare=False)
    post_layer_chart: Lfm2MovingGaugeBoundaryChart = field(
        repr=False,
        compare=False,
    )
    terminal_chart: Lfm2MovingGaugeBoundaryChart = field(
        repr=False,
        compare=False,
    )
    pre_evidence: Lfm2MovingGaugeBoundaryEvidence = field(
        repr=False,
        compare=False,
    )
    post_layer_evidence: Lfm2MovingGaugeBoundaryEvidence = field(
        repr=False,
        compare=False,
    )
    terminal_evidence: Lfm2MovingGaugeBoundaryEvidence = field(
        repr=False,
        compare=False,
    )
    parent_cache_sha256: str
    child_state_sha256: str
    child_recapture_sha256: str
    row_alignment_sha256: str
    split_identity_sha256: str
    layer_topology: Mapping[str, Any] = field(repr=False, compare=False)
    post_operator_chart: Lfm2MovingGaugeBoundaryChart | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    post_operator_evidence: Lfm2MovingGaugeBoundaryEvidence | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    report: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        layer_index = _nonnegative_int(self.layer_index, "layer_index")
        layer_count = _nonnegative_int(self.layer_count, "layer_count")
        if layer_count <= 0 or layer_index >= layer_count:
            raise Lfm2MovingGaugeCausalTargetError(
                "layer_index must identify one layer inside layer_count."
            )
        operator_kind = _identifier(self.operator_kind, "operator_kind")
        pre = _chart(self.pre_chart, "pre_chart")
        post_layer = _chart(self.post_layer_chart, "post_layer_chart")
        terminal = _chart(self.terminal_chart, "terminal_chart")
        post_operator = (
            None
            if self.post_operator_chart is None
            else _chart(self.post_operator_chart, "post_operator_chart")
        )
        if (post_operator is None) != (self.post_operator_evidence is None):
            raise Lfm2MovingGaugeCausalTargetAuthenticationError(
                "Post-operator chart and paired evidence must be present together."
            )
        evidence_by_endpoint: dict[str, Lfm2MovingGaugeBoundaryEvidence] = {
            "pre": self.pre_evidence,
            "post_layer": self.post_layer_evidence,
            "terminal": self.terminal_evidence,
        }
        chart_by_endpoint: dict[str, Lfm2MovingGaugeBoundaryChart] = {
            "pre": pre,
            "post_layer": post_layer,
            "terminal": terminal,
        }
        if post_operator is not None:
            assert self.post_operator_evidence is not None
            evidence_by_endpoint["post_operator"] = self.post_operator_evidence
            chart_by_endpoint["post_operator"] = post_operator
        if any(
            not isinstance(value, Lfm2MovingGaugeBoundaryEvidence)
            or not value.verify_authentication()
            for value in evidence_by_endpoint.values()
        ):
            raise Lfm2MovingGaugeCausalTargetAuthenticationError(
                "Every causal endpoint requires authenticated paired boundary evidence."
            )
        parent_cache = _digest(self.parent_cache_sha256, "parent_cache_sha256")
        child_state = _digest(self.child_state_sha256, "child_state_sha256")
        child_recapture = _digest(
            self.child_recapture_sha256,
            "child_recapture_sha256",
        )
        row_alignment = _digest(
            self.row_alignment_sha256,
            "row_alignment_sha256",
        )
        split_identity = _digest(
            self.split_identity_sha256,
            "split_identity_sha256",
        )
        layer_topology, layer_topology_sha256 = _layer_topology(
            self.layer_topology,
            layer_index=layer_index,
            layer_count=layer_count,
            operator_kind=operator_kind,
        )
        source_widths = {
            value.chart.source_width for value in chart_by_endpoint.values()
        }
        target_widths = {
            value.chart.target_width for value in chart_by_endpoint.values()
        }
        if len(source_widths) != 1 or len(target_widths) != 1:
            raise Lfm2MovingGaugeCausalTargetError(
                "Causal endpoint charts do not share parent/child residual geometry."
            )
        is_final_layer = layer_index == layer_count - 1
        pre_role = _chart_role(pre, "pre_chart")
        post_layer_role = _chart_role(post_layer, "post_layer_chart")
        terminal_role = _chart_role(terminal, "terminal_chart")
        if (
            (layer_index == 0 and pre_role != "input")
            or (layer_index > 0 and pre_role != "intermediate")
            or (layer_index == 0 and not bool(pre.report.get("interface_locked")))
            or (layer_index > 0 and bool(pre.report.get("interface_locked")))
            or terminal_role != "terminal"
            or not bool(terminal.report.get("interface_locked"))
            or (is_final_layer and post_layer_role != "terminal")
            or (not is_final_layer and post_layer_role != "intermediate")
            or (
                post_layer_role == "intermediate"
                and bool(post_layer.report.get("interface_locked"))
            )
            or (
                post_layer_role == "terminal"
                and not bool(post_layer.report.get("interface_locked"))
            )
        ):
            raise Lfm2MovingGaugeCausalTargetAuthenticationError(
                "Causal endpoint chart roles or lock states are inconsistent."
            )
        if post_operator is not None and (
            _chart_role(post_operator, "post_operator_chart") != "intermediate"
            or bool(post_operator.report.get("interface_locked"))
        ):
            raise Lfm2MovingGaugeCausalTargetAuthenticationError(
                "Post-operator chart must be an unlocked internal boundary."
            )
        if is_final_layer and (
            post_layer.report["report_sha256"] != terminal.report["report_sha256"]
            or post_layer.report.get("boundary_id")
            != terminal.report.get("boundary_id")
        ):
            raise Lfm2MovingGaugeCausalTargetAuthenticationError(
                "Final post-layer and terminal evaluation must use one chart."
            )
        for endpoint, chart in chart_by_endpoint.items():
            evidence = evidence_by_endpoint[endpoint]
            if (
                chart.evidence_sha256 != evidence.report["report_sha256"]
                or chart.report.get("boundary_id") != evidence.boundary_id
                or chart.report.get("interface_role") != evidence.interface_role
                or chart.report.get("interface_locked") != evidence.interface_locked
            ):
                raise Lfm2MovingGaugeCausalTargetAuthenticationError(
                    f"{endpoint} chart is not bound to its supplied boundary evidence."
                )
        if (
            {
                value.cached_parent_evidence_sha256
                for value in evidence_by_endpoint.values()
            }
            != {parent_cache}
            or {
                value.live_child_recapture_sha256
                for value in evidence_by_endpoint.values()
            }
            != {child_recapture}
            or {value.row_alignment_sha256 for value in evidence_by_endpoint.values()}
            != {row_alignment}
        ):
            raise Lfm2MovingGaugeCausalTargetAuthenticationError(
                "Causal boundary evidence mixes parent cache, child recapture, or "
                "row-alignment identities."
            )
        chart_ids = {
            endpoint: chart.report.get("boundary_id")
            for endpoint, chart in chart_by_endpoint.items()
        }
        if any(not isinstance(value, str) or not value for value in chart_ids.values()):
            raise Lfm2MovingGaugeCausalTargetAuthenticationError(
                "Every causal endpoint chart must authenticate its boundary id."
            )
        expected_chart_ids = _expected_boundary_ids(
            layer_index,
            layer_count,
            sequential=post_operator is not None,
        )
        if chart_ids != expected_chart_ids:
            raise Lfm2MovingGaugeCausalTargetAuthenticationError(
                "Causal boundary ids do not belong to the sealed layer index."
            )
        base_encoder_sha256s = {
            evidence.report.get("base_encoder_sha256")
            for evidence in evidence_by_endpoint.values()
        }
        if (
            len(base_encoder_sha256s) != 1
            or not isinstance(next(iter(base_encoder_sha256s), None), str)
            or _SHA256.fullmatch(next(iter(base_encoder_sha256s))) is None
        ):
            raise Lfm2MovingGaugeCausalTargetAuthenticationError(
                "Causal endpoint charts do not share one base encoder identity."
            )
        body = {
            "schema": LFM2_MOVING_GAUGE_CAUSAL_TARGETS_SCHEMA,
            "kind": "state_conditioned_causal_boundary_plan",
            "layer_index": layer_index,
            "layer_count": layer_count,
            "is_final_layer": is_final_layer,
            "operator_kind": operator_kind,
            "source_width": next(iter(source_widths)),
            "target_width": next(iter(target_widths)),
            "available_target_modes": (
                ["joint", "sequential"] if post_operator is not None else ["joint"]
            ),
            "chart_ids": chart_ids,
            "chart_report_sha256s": {
                endpoint: chart.report["report_sha256"]
                for endpoint, chart in chart_by_endpoint.items()
            },
            "boundary_evidence_sha256s": {
                endpoint: evidence.report["report_sha256"]
                for endpoint, evidence in evidence_by_endpoint.items()
            },
            "parent_cache_sha256": parent_cache,
            "child_state_sha256": child_state,
            "child_recapture_sha256": child_recapture,
            "row_alignment_sha256": row_alignment,
            "split_identity_sha256": split_identity,
            "layer_topology_sha256": layer_topology_sha256,
            "base_encoder_sha256": next(iter(base_encoder_sha256s)),
            "locked_global_map_allowed_only_for_residual_boundary_state": True,
            "raw_global_writer_projection_allowed": False,
            "joint_target_formula": (
                "E_post_layer(parent_post_layer)-actual_child_pre"
            ),
            "sequential_operator_target_formula": (
                None
                if post_operator is None
                else "E_post_operator(parent_post_operator)-actual_child_pre"
            ),
            "sequential_mlp_target_formula": (
                None
                if post_operator is None
                else "E_post_layer(parent_post_layer)-fresh_child_post_operator"
            ),
            "fit_only": True,
            "numeric_quality_gate": False,
            "outer_hmac_checkpoint_required": True,
            "outer_hmac_checkpoint_verified": False,
            "execution": dict(_ZERO_EXECUTION),
        }
        expected = _seal(body)
        supplied = dict(self.report)
        if supplied and supplied != dict(expected):
            raise Lfm2MovingGaugeCausalTargetAuthenticationError(
                "Causal boundary-plan report is stale or malformed."
            )
        object.__setattr__(self, "layer_index", layer_index)
        object.__setattr__(self, "layer_count", layer_count)
        object.__setattr__(self, "operator_kind", operator_kind)
        object.__setattr__(self, "parent_cache_sha256", parent_cache)
        object.__setattr__(self, "child_state_sha256", child_state)
        object.__setattr__(self, "child_recapture_sha256", child_recapture)
        object.__setattr__(self, "row_alignment_sha256", row_alignment)
        object.__setattr__(self, "split_identity_sha256", split_identity)
        object.__setattr__(self, "layer_topology", layer_topology)
        object.__setattr__(self, "report", expected)

    @property
    def source_width(self) -> int:
        return self.pre_chart.chart.source_width

    @property
    def target_width(self) -> int:
        return self.pre_chart.chart.target_width

    def verify_authentication(self) -> bool:
        try:
            rebuilt = replace(self, report={})
            authenticated_topology, topology_sha256 = _layer_topology(
                self.layer_topology,
                layer_index=self.layer_index,
                layer_count=self.layer_count,
                operator_kind=self.operator_kind,
            )
            chart_pairs = [
                ("pre", self.pre_chart, self.pre_evidence),
                ("post_layer", self.post_layer_chart, self.post_layer_evidence),
                ("terminal", self.terminal_chart, self.terminal_evidence),
            ]
            if self.post_operator_chart is not None:
                assert self.post_operator_evidence is not None
                chart_pairs.append(
                    (
                        "post_operator",
                        self.post_operator_chart,
                        self.post_operator_evidence,
                    )
                )
            return bool(
                verify_lfm2_moving_gauge_causal_target_report(self.report)
                and dict(self.report) == dict(rebuilt.report)
                and all(
                    chart.verify_authentication()
                    and evidence.verify_authentication()
                    and chart.evidence_sha256 == evidence.report["report_sha256"]
                    and self.report["chart_report_sha256s"][endpoint]
                    == chart.report["report_sha256"]
                    and self.report["boundary_evidence_sha256s"][endpoint]
                    == evidence.report["report_sha256"]
                    for endpoint, chart, evidence in chart_pairs
                )
                and self.report["layer_index"] == self.layer_index
                and self.report["layer_count"] == self.layer_count
                and self.report["operator_kind"] == self.operator_kind
                and self.report["parent_cache_sha256"] == self.parent_cache_sha256
                and self.report["child_state_sha256"] == self.child_state_sha256
                and self.report["child_recapture_sha256"] == self.child_recapture_sha256
                and self.report["row_alignment_sha256"] == self.row_alignment_sha256
                and self.report["split_identity_sha256"] == self.split_identity_sha256
                and self.report["layer_topology_sha256"]
                == authenticated_topology["layer_topology_sha256"]
                == topology_sha256
            )
        except (KeyError, TypeError, ValueError):
            return False


def _joint_writer_report(
    plan: Lfm2MovingGaugeCausalBoundaryPlan,
    state_receipt: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
) -> Mapping[str, Any]:
    return _seal(
        {
            "schema": LFM2_MOVING_GAUGE_CAUSAL_TARGETS_SCHEMA,
            "kind": "joint_writer_target",
            "plan_sha256": plan.report["result_sha256"],
            "layer_index": plan.layer_index,
            "operator_kind": plan.operator_kind,
            "post_layer_chart_sha256": plan.post_layer_chart.report["report_sha256"],
            "state_receipt_sha256": state_receipt["result_sha256"],
            "desired_total_formula": (
                "E_post_layer(parent_post_layer)-actual_child_pre"
            ),
            "fixed_other_formula": (
                "actual_child_post_layer-actual_child_pre-"
                "existing_combined_writer_output"
            ),
            "physical_writer_formula": "desired_total_write-fixed_other_write",
            "tensor_sha256s": {
                name: fingerprint_lfm2_moving_gauge_tensor(value)
                for name, value in tensors.items()
            },
            "solution_installed": False,
            "fit_only": True,
            "outer_hmac_checkpoint_required": True,
            "outer_hmac_checkpoint_verified": False,
            "execution": dict(_ZERO_EXECUTION),
        }
    )


def _sequential_writer_report(
    plan: Lfm2MovingGaugeCausalBoundaryPlan,
    *,
    operator_source_receipt: Mapping[str, Any],
    mlp_source_receipt: Mapping[str, Any],
    recapture_kind: str,
    operator_solution_sha256: str | None,
    evidence_stage: str,
    tensors: Mapping[str, torch.Tensor],
) -> Mapping[str, Any]:
    assert plan.post_operator_chart is not None
    return _seal(
        {
            "schema": LFM2_MOVING_GAUGE_CAUSAL_TARGETS_SCHEMA,
            "kind": "sequential_writer_target",
            "plan_sha256": plan.report["result_sha256"],
            "layer_index": plan.layer_index,
            "operator_kind": plan.operator_kind,
            "post_operator_chart_sha256": plan.post_operator_chart.report[
                "report_sha256"
            ],
            "post_layer_chart_sha256": plan.post_layer_chart.report["report_sha256"],
            "operator_desired_total_formula": (
                "E_post_operator(parent_post_operator)-actual_child_pre"
            ),
            "operator_fixed_other_formula": (
                "actual_child_post_operator-actual_child_pre-"
                "existing_operator_writer_output"
            ),
            "operator_physical_writer_formula": (
                "desired_operator_total_write-operator_fixed_other_write"
            ),
            "mlp_desired_total_formula": (
                "E_post_layer(parent_post_layer)-fresh_child_post_operator"
            ),
            "mlp_fixed_other_formula": (
                "fresh_child_post_layer-fresh_child_post_operator-"
                "existing_mlp_writer_output"
            ),
            "mlp_physical_writer_formula": (
                "desired_mlp_total_write-mlp_fixed_other_write"
            ),
            "fresh_post_operator_state_required": True,
            "operator_source_receipt_sha256": operator_source_receipt["result_sha256"],
            "mlp_source_receipt_sha256": mlp_source_receipt["result_sha256"],
            "fresh_recapture_kind": recapture_kind,
            "operator_solution_sha256": operator_solution_sha256,
            "evidence_stage": evidence_stage,
            "operator_solution_installed": False,
            "target_writer_solutions_installed": False,
            "full_post_layer_reclosure_verified": False,
            "fixed_other_writes_derived_from_observed_endpoint_equations": True,
            "raw_parent_branch_delta_projection_used": False,
            "tensor_sha256s": {
                name: fingerprint_lfm2_moving_gauge_tensor(value)
                for name, value in tensors.items()
            },
            "fit_only": True,
            "outer_hmac_checkpoint_required": True,
            "outer_hmac_checkpoint_verified": False,
            "execution": dict(_ZERO_EXECUTION),
        }
    )


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeJointWriterTarget:
    """One faithful R12 combined operator+MLP target."""

    plan: Lfm2MovingGaugeCausalBoundaryPlan = field(repr=False, compare=False)
    parent_post_layer: torch.Tensor = field(repr=False, compare=False)
    actual_child_pre: torch.Tensor = field(repr=False, compare=False)
    actual_child_post_layer: torch.Tensor = field(repr=False, compare=False)
    existing_combined_writer_output: torch.Tensor = field(
        repr=False,
        compare=False,
    )
    desired_endpoint: torch.Tensor = field(repr=False, compare=False)
    desired_total_write: torch.Tensor = field(repr=False, compare=False)
    fixed_other_write: torch.Tensor = field(repr=False, compare=False)
    target_combined_writer_output: torch.Tensor = field(repr=False, compare=False)
    state_receipt: Mapping[str, Any] = field(repr=False, compare=False)
    report: Mapping[str, Any]

    def verify_authentication(self) -> bool:
        try:
            tensors = {
                "parent_post_layer": self.parent_post_layer,
                "actual_child_pre": self.actual_child_pre,
                "actual_child_post_layer": self.actual_child_post_layer,
                "existing_combined_writer_output": (
                    self.existing_combined_writer_output
                ),
                "desired_endpoint": self.desired_endpoint,
                "desired_total_write": self.desired_total_write,
                "fixed_other_write": self.fixed_other_write,
                "target_combined_writer_output": self.target_combined_writer_output,
            }
            if (
                not _is_canonical_state(
                    self.parent_post_layer,
                    width=self.plan.source_width,
                )
                or any(
                    not _is_canonical_state(value, width=self.plan.target_width)
                    for name, value in tensors.items()
                    if name != "parent_post_layer"
                )
                or len({tuple(value.shape[:-1]) for value in tensors.values()}) != 1
            ):
                return False
            expected_endpoint = self.plan.post_layer_chart.chart.encode_shaped(
                self.parent_post_layer
            )
            source_receipt = authenticate_lfm2_moving_gauge_causal_state_receipt(
                self.state_receipt,
                plan=self.plan,
                tensors={
                    "actual_child_post_layer": self.actual_child_post_layer,
                    "actual_child_pre": self.actual_child_pre,
                    "existing_combined_writer_output": (
                        self.existing_combined_writer_output
                    ),
                    "parent_post_layer": self.parent_post_layer,
                },
                allowed_kinds={"baseline_current_child"},
            )
            expected_report = _joint_writer_report(
                self.plan,
                source_receipt,
                tensors,
            )
            return bool(
                self.plan.verify_authentication()
                and verify_lfm2_moving_gauge_causal_target_report(self.report)
                and dict(self.report) == dict(expected_report)
                and _roundoff_close(self.desired_endpoint, expected_endpoint)
                and _roundoff_close(
                    self.desired_total_write,
                    self.desired_endpoint - self.actual_child_pre,
                )
                and _roundoff_close(
                    self.fixed_other_write,
                    self.actual_child_post_layer
                    - self.actual_child_pre
                    - self.existing_combined_writer_output,
                )
                and _roundoff_close(
                    self.target_combined_writer_output,
                    self.desired_total_write - self.fixed_other_write,
                )
            )
        except (KeyError, RuntimeError, TypeError, ValueError):
            return False


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeSequentialWriterTarget:
    """One post-operator then post-layer residual-attention target."""

    plan: Lfm2MovingGaugeCausalBoundaryPlan = field(repr=False, compare=False)
    parent_post_operator: torch.Tensor = field(repr=False, compare=False)
    actual_child_pre: torch.Tensor = field(repr=False, compare=False)
    actual_child_post_operator: torch.Tensor = field(repr=False, compare=False)
    existing_operator_writer_output: torch.Tensor = field(
        repr=False,
        compare=False,
    )
    parent_post_layer: torch.Tensor = field(repr=False, compare=False)
    operator_endpoint: torch.Tensor = field(repr=False, compare=False)
    desired_operator_total_write: torch.Tensor = field(repr=False, compare=False)
    operator_fixed_other_write: torch.Tensor = field(repr=False, compare=False)
    target_operator_writer_output: torch.Tensor = field(repr=False, compare=False)
    fresh_child_post_operator: torch.Tensor = field(repr=False, compare=False)
    fresh_child_post_layer: torch.Tensor = field(repr=False, compare=False)
    existing_mlp_writer_output: torch.Tensor = field(repr=False, compare=False)
    post_layer_endpoint: torch.Tensor = field(repr=False, compare=False)
    desired_mlp_total_write: torch.Tensor = field(repr=False, compare=False)
    mlp_fixed_other_write: torch.Tensor = field(repr=False, compare=False)
    target_mlp_writer_output: torch.Tensor = field(repr=False, compare=False)
    operator_source_receipt: Mapping[str, Any] = field(repr=False, compare=False)
    mlp_source_receipt: Mapping[str, Any] = field(repr=False, compare=False)
    fresh_recapture_kind: str
    operator_solution_sha256: str | None
    report: Mapping[str, Any]

    @property
    def operator_write(self) -> torch.Tensor:
        """Compatibility name for the physical operator-writer target."""

        return self.target_operator_writer_output

    @property
    def mlp_write(self) -> torch.Tensor:
        """Compatibility name for the physical MLP-writer target."""

        return self.target_mlp_writer_output

    def verify_authentication(self) -> bool:
        try:
            tensors = {
                "parent_post_operator": self.parent_post_operator,
                "actual_child_pre": self.actual_child_pre,
                "actual_child_post_operator": self.actual_child_post_operator,
                "existing_operator_writer_output": (
                    self.existing_operator_writer_output
                ),
                "parent_post_layer": self.parent_post_layer,
                "operator_endpoint": self.operator_endpoint,
                "desired_operator_total_write": self.desired_operator_total_write,
                "operator_fixed_other_write": self.operator_fixed_other_write,
                "target_operator_writer_output": self.target_operator_writer_output,
                "fresh_child_post_operator": self.fresh_child_post_operator,
                "fresh_child_post_layer": self.fresh_child_post_layer,
                "existing_mlp_writer_output": self.existing_mlp_writer_output,
                "post_layer_endpoint": self.post_layer_endpoint,
                "desired_mlp_total_write": self.desired_mlp_total_write,
                "mlp_fixed_other_write": self.mlp_fixed_other_write,
                "target_mlp_writer_output": self.target_mlp_writer_output,
            }
            parent_names = {"parent_post_operator", "parent_post_layer"}
            if (
                any(
                    not _is_canonical_state(
                        value,
                        width=(
                            self.plan.source_width
                            if name in parent_names
                            else self.plan.target_width
                        ),
                    )
                    for name, value in tensors.items()
                )
                or len({tuple(value.shape[:-1]) for value in tensors.values()}) != 1
            ):
                return False
            if self.plan.post_operator_chart is None:
                return False
            expected_operator_endpoint = (
                self.plan.post_operator_chart.chart.encode_shaped(
                    self.parent_post_operator
                )
            )
            expected_post_layer_endpoint = (
                self.plan.post_layer_chart.chart.encode_shaped(self.parent_post_layer)
            )
            expected_stage = {
                "baseline_current_child": "baseline_diagnostic",
                "operator_conditioned_temporary_intervention": (
                    "temporary_operator_mlp_target_ready"
                ),
            }.get(self.fresh_recapture_kind)
            operator_receipt = authenticate_lfm2_moving_gauge_causal_state_receipt(
                self.operator_source_receipt,
                plan=self.plan,
                tensors={
                    "actual_child_post_operator": (self.actual_child_post_operator),
                    "actual_child_pre": self.actual_child_pre,
                    "existing_operator_writer_output": (
                        self.existing_operator_writer_output
                    ),
                    "parent_post_operator": self.parent_post_operator,
                },
                allowed_kinds={"baseline_current_child"},
            )
            mlp_receipt = authenticate_lfm2_moving_gauge_causal_state_receipt(
                self.mlp_source_receipt,
                plan=self.plan,
                tensors={
                    "existing_mlp_writer_output": self.existing_mlp_writer_output,
                    "fresh_child_post_layer": self.fresh_child_post_layer,
                    "fresh_child_post_operator": self.fresh_child_post_operator,
                    "parent_post_layer": self.parent_post_layer,
                },
                allowed_kinds={
                    "baseline_current_child",
                    "operator_conditioned_temporary_intervention",
                },
            )
            expected_report = _sequential_writer_report(
                self.plan,
                operator_source_receipt=operator_receipt,
                mlp_source_receipt=mlp_receipt,
                recapture_kind=self.fresh_recapture_kind,
                operator_solution_sha256=self.operator_solution_sha256,
                evidence_stage=expected_stage,
                tensors=tensors,
            )
            return bool(
                self.plan.verify_authentication()
                and verify_lfm2_moving_gauge_causal_target_report(self.report)
                and dict(self.report) == dict(expected_report)
                and self.fresh_recapture_kind == mlp_receipt["kind"]
                and self.operator_solution_sha256
                == mlp_receipt["operator_solution_sha256"]
                and (
                    (
                        self.fresh_recapture_kind == "baseline_current_child"
                        and self.operator_solution_sha256 is None
                    )
                    or (
                        self.fresh_recapture_kind
                        == "operator_conditioned_temporary_intervention"
                        and isinstance(self.operator_solution_sha256, str)
                        and _SHA256.fullmatch(self.operator_solution_sha256) is not None
                    )
                )
                and _roundoff_close(
                    self.operator_endpoint,
                    expected_operator_endpoint,
                )
                and _roundoff_close(
                    self.desired_operator_total_write,
                    self.operator_endpoint - self.actual_child_pre,
                )
                and _roundoff_close(
                    self.operator_fixed_other_write,
                    self.actual_child_post_operator
                    - self.actual_child_pre
                    - self.existing_operator_writer_output,
                )
                and _roundoff_close(
                    self.target_operator_writer_output,
                    self.desired_operator_total_write - self.operator_fixed_other_write,
                )
                and _roundoff_close(
                    self.mlp_fixed_other_write,
                    self.fresh_child_post_layer
                    - self.fresh_child_post_operator
                    - self.existing_mlp_writer_output,
                )
                and _roundoff_close(
                    self.post_layer_endpoint,
                    expected_post_layer_endpoint,
                )
                and _roundoff_close(
                    self.desired_mlp_total_write,
                    self.post_layer_endpoint - self.fresh_child_post_operator,
                )
                and _roundoff_close(
                    self.target_mlp_writer_output,
                    self.desired_mlp_total_write - self.mlp_fixed_other_write,
                )
            )
        except (KeyError, RuntimeError, TypeError, ValueError):
            return False


def project_lfm2_boundary_state(
    plan: Lfm2MovingGaugeCausalBoundaryPlan,
    *,
    boundary: str,
    parent_state: Any,
) -> torch.Tensor:
    """Encode C/pre or terminal state through its authenticated chart."""

    if not isinstance(plan, Lfm2MovingGaugeCausalBoundaryPlan) or not (
        plan.verify_authentication()
    ):
        raise Lfm2MovingGaugeCausalTargetAuthenticationError(
            "Boundary-state projection requires an authenticated causal plan."
        )
    if boundary == "pre":
        chart = plan.pre_chart
    elif boundary == "terminal":
        chart = plan.terminal_chart
        if not bool(chart.report.get("interface_locked")):
            raise Lfm2MovingGaugeCausalTargetAuthenticationError(
                "Terminal projection requires a locked terminal chart."
            )
    else:
        raise Lfm2MovingGaugeCausalTargetError("boundary must be pre or terminal.")
    parent = _state(
        parent_state,
        "parent_state",
        width=chart.chart.source_width,
    )
    return chart.chart.encode_shaped(parent).to(dtype=torch.float64).contiguous()


def build_lfm2_joint_writer_target(
    plan: Lfm2MovingGaugeCausalBoundaryPlan,
    *,
    parent_post_layer: Any,
    actual_child_pre: Any,
    actual_child_post_layer: Any,
    existing_combined_writer_output: Any,
    state_receipt: Mapping[str, Any],
) -> Lfm2MovingGaugeJointWriterTarget:
    """Build the faithful R12 combined writer target for one replay state."""

    if not isinstance(plan, Lfm2MovingGaugeCausalBoundaryPlan) or not (
        plan.verify_authentication()
    ):
        raise Lfm2MovingGaugeCausalTargetAuthenticationError(
            "Joint writer target requires an authenticated causal plan."
        )
    parent = _state(
        parent_post_layer,
        "parent_post_layer",
        width=plan.source_width,
    )
    child_pre = _state(actual_child_pre, "actual_child_pre", width=plan.target_width)
    child_post = _state(
        actual_child_post_layer,
        "actual_child_post_layer",
        width=plan.target_width,
    )
    existing = _state(
        existing_combined_writer_output,
        "existing_combined_writer_output",
        width=plan.target_width,
    )
    if not (
        tuple(parent.shape[:-1])
        == tuple(child_pre.shape[:-1])
        == tuple(child_post.shape[:-1])
        == tuple(existing.shape[:-1])
    ):
        raise Lfm2MovingGaugeCausalTargetError(
            "Joint writer states do not share aligned leading dimensions."
        )
    source_tensors = {
        "actual_child_post_layer": child_post,
        "actual_child_pre": child_pre,
        "existing_combined_writer_output": existing,
        "parent_post_layer": parent,
    }
    authenticated_state_receipt = authenticate_lfm2_moving_gauge_causal_state_receipt(
        state_receipt,
        plan=plan,
        tensors=source_tensors,
        allowed_kinds={"baseline_current_child"},
    )
    desired_endpoint, desired_total_write = (
        plan.post_layer_chart.chart.residual_write_target(
            actual_child_pre_state=child_pre,
            parent_endpoint=parent,
        )
    )
    live_branch_write = child_post - child_pre
    fixed_other_write = (live_branch_write - existing).contiguous()
    target_output = (desired_total_write - fixed_other_write).contiguous()
    tensors = {
        "parent_post_layer": parent,
        "actual_child_pre": child_pre,
        "actual_child_post_layer": child_post,
        "existing_combined_writer_output": existing,
        "desired_endpoint": desired_endpoint,
        "desired_total_write": desired_total_write,
        "fixed_other_write": fixed_other_write,
        "target_combined_writer_output": target_output,
    }
    report = _joint_writer_report(
        plan,
        authenticated_state_receipt,
        tensors,
    )
    result = Lfm2MovingGaugeJointWriterTarget(
        plan=plan,
        **tensors,
        state_receipt=authenticated_state_receipt,
        report=report,
    )
    if not result.verify_authentication():
        raise Lfm2MovingGaugeCausalTargetAuthenticationError(
            "Joint writer target failed self-authentication."
        )
    return result


def build_lfm2_sequential_writer_target(
    plan: Lfm2MovingGaugeCausalBoundaryPlan,
    *,
    parent_post_operator: Any,
    actual_child_pre: Any,
    actual_child_post_operator: Any,
    existing_operator_writer_output: Any,
    parent_post_layer: Any,
    fresh_child_post_operator: Any,
    fresh_child_post_layer: Any,
    existing_mlp_writer_output: Any,
    operator_source_receipt: Mapping[str, Any],
    mlp_source_receipt: Mapping[str, Any],
) -> Lfm2MovingGaugeSequentialWriterTarget:
    """Build state-conditioned operator and MLP targets for one replay state."""

    if not isinstance(plan, Lfm2MovingGaugeCausalBoundaryPlan) or not (
        plan.verify_authentication()
    ):
        raise Lfm2MovingGaugeCausalTargetAuthenticationError(
            "Sequential writer target requires an authenticated causal plan."
        )
    if plan.post_operator_chart is None:
        raise Lfm2MovingGaugeCausalTargetAuthenticationError(
            "Sequential targets require an authenticated post-operator chart."
        )
    parent_operator = _state(
        parent_post_operator,
        "parent_post_operator",
        width=plan.source_width,
    )
    parent_layer = _state(
        parent_post_layer,
        "parent_post_layer",
        width=plan.source_width,
    )
    child_pre = _state(actual_child_pre, "actual_child_pre", width=plan.target_width)
    child_post_operator = _state(
        actual_child_post_operator,
        "actual_child_post_operator",
        width=plan.target_width,
    )
    existing_operator = _state(
        existing_operator_writer_output,
        "existing_operator_writer_output",
        width=plan.target_width,
    )
    fresh_post_operator = _state(
        fresh_child_post_operator,
        "fresh_child_post_operator",
        width=plan.target_width,
    )
    fresh_post_layer = _state(
        fresh_child_post_layer,
        "fresh_child_post_layer",
        width=plan.target_width,
    )
    existing_mlp = _state(
        existing_mlp_writer_output,
        "existing_mlp_writer_output",
        width=plan.target_width,
    )
    if not (
        tuple(parent_operator.shape[:-1])
        == tuple(parent_layer.shape[:-1])
        == tuple(child_pre.shape[:-1])
        == tuple(child_post_operator.shape[:-1])
        == tuple(existing_operator.shape[:-1])
        == tuple(fresh_post_operator.shape[:-1])
        == tuple(fresh_post_layer.shape[:-1])
        == tuple(existing_mlp.shape[:-1])
    ):
        raise Lfm2MovingGaugeCausalTargetError(
            "Sequential writer states do not share aligned leading dimensions."
        )
    authenticated_operator_receipt = (
        authenticate_lfm2_moving_gauge_causal_state_receipt(
            operator_source_receipt,
            plan=plan,
            tensors={
                "actual_child_post_operator": child_post_operator,
                "actual_child_pre": child_pre,
                "existing_operator_writer_output": existing_operator,
                "parent_post_operator": parent_operator,
            },
            allowed_kinds={"baseline_current_child"},
        )
    )
    authenticated_mlp_receipt = authenticate_lfm2_moving_gauge_causal_state_receipt(
        mlp_source_receipt,
        plan=plan,
        tensors={
            "existing_mlp_writer_output": existing_mlp,
            "fresh_child_post_layer": fresh_post_layer,
            "fresh_child_post_operator": fresh_post_operator,
            "parent_post_layer": parent_layer,
        },
        allowed_kinds={
            "baseline_current_child",
            "operator_conditioned_temporary_intervention",
        },
    )
    recapture_kind = str(authenticated_mlp_receipt["kind"])
    operator_solution = authenticated_mlp_receipt["operator_solution_sha256"]
    evidence_stage = {
        "baseline_current_child": "baseline_diagnostic",
        "operator_conditioned_temporary_intervention": (
            "temporary_operator_mlp_target_ready"
        ),
    }[recapture_kind]
    operator_fixed = (child_post_operator - child_pre - existing_operator).contiguous()
    mlp_fixed = (fresh_post_layer - fresh_post_operator - existing_mlp).contiguous()
    operator_endpoint, desired_operator_total_write = (
        plan.post_operator_chart.chart.residual_write_target(
            actual_child_pre_state=child_pre,
            parent_endpoint=parent_operator,
        )
    )
    target_operator_writer_output = (
        desired_operator_total_write - operator_fixed
    ).contiguous()
    post_layer_endpoint, desired_mlp_total_write = (
        plan.post_layer_chart.chart.residual_write_target(
            actual_child_pre_state=fresh_post_operator,
            parent_endpoint=parent_layer,
        )
    )
    target_mlp_writer_output = (desired_mlp_total_write - mlp_fixed).contiguous()
    if not _roundoff_close(
        child_pre + operator_fixed + target_operator_writer_output,
        operator_endpoint,
    ):
        raise Lfm2MovingGaugeCausalTargetError(
            "Operator residual target does not close its post-operator endpoint."
        )
    if not _roundoff_close(
        fresh_post_operator + mlp_fixed + target_mlp_writer_output,
        post_layer_endpoint,
    ):
        raise Lfm2MovingGaugeCausalTargetError(
            "MLP residual target does not close its post-layer endpoint."
        )
    tensors = {
        "parent_post_operator": parent_operator,
        "actual_child_pre": child_pre,
        "actual_child_post_operator": child_post_operator,
        "existing_operator_writer_output": existing_operator,
        "parent_post_layer": parent_layer,
        "operator_endpoint": operator_endpoint,
        "desired_operator_total_write": desired_operator_total_write,
        "operator_fixed_other_write": operator_fixed,
        "target_operator_writer_output": target_operator_writer_output,
        "fresh_child_post_operator": fresh_post_operator,
        "fresh_child_post_layer": fresh_post_layer,
        "existing_mlp_writer_output": existing_mlp,
        "post_layer_endpoint": post_layer_endpoint,
        "desired_mlp_total_write": desired_mlp_total_write,
        "mlp_fixed_other_write": mlp_fixed,
        "target_mlp_writer_output": target_mlp_writer_output,
    }
    report = _sequential_writer_report(
        plan,
        operator_source_receipt=authenticated_operator_receipt,
        mlp_source_receipt=authenticated_mlp_receipt,
        recapture_kind=recapture_kind,
        operator_solution_sha256=operator_solution,
        evidence_stage=evidence_stage,
        tensors=tensors,
    )
    result = Lfm2MovingGaugeSequentialWriterTarget(
        plan=plan,
        **tensors,
        operator_source_receipt=authenticated_operator_receipt,
        mlp_source_receipt=authenticated_mlp_receipt,
        fresh_recapture_kind=recapture_kind,
        operator_solution_sha256=operator_solution,
        report=report,
    )
    if not result.verify_authentication():
        raise Lfm2MovingGaugeCausalTargetAuthenticationError(
            "Sequential writer target failed self-authentication."
        )
    return result


__all__ = [
    "LFM2_LAYER_TOPOLOGY_SCHEMA",
    "LFM2_MOVING_GAUGE_CAUSAL_STATE_RECEIPT_SCHEMA",
    "LFM2_MOVING_GAUGE_CAUSAL_TARGETS_SCHEMA",
    "Lfm2MovingGaugeCausalBoundaryPlan",
    "Lfm2MovingGaugeCausalTargetAuthenticationError",
    "Lfm2MovingGaugeCausalTargetError",
    "Lfm2MovingGaugeJointWriterTarget",
    "Lfm2MovingGaugeSequentialWriterTarget",
    "authenticate_lfm2_moving_gauge_causal_state_receipt",
    "build_lfm2_joint_writer_target",
    "build_lfm2_moving_gauge_causal_state_receipt",
    "build_lfm2_sequential_writer_target",
    "project_lfm2_boundary_state",
    "verify_lfm2_moving_gauge_causal_target_report",
]
