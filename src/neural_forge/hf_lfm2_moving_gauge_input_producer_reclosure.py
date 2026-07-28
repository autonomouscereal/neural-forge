"""Analytic input-producer reclosure after an LFM2 residual gauge change.

Residual-writer replacement changes the *actual* child state presented to the
next RMSNorm.  An arbitrary residual chart cannot be pushed through RMSNorm.
This primitive therefore consumes paired, row-aligned normalized inputs that
were observed before and after the writer change.  For every dense producer it
freezes the old producer output

``old_actual_normalized_rows @ seed_weight.T``

and solves one seed-anchored FP64 ridge system that reproduces that output from
the new actual normalized rows.  Only a dense FP32/BF16 replacement is
published.  There is no learned sidecar, parent execution, autograd, optimizer,
pruning, quantization, or numeric quality decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_fold_executor import run_fold_only_callback


LFM2_MOVING_GAUGE_INPUT_PRODUCER_RECLOSURE_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-input-producer-reclosure-v1"
)
_IMPLEMENTATION = (
    "neural_forge.hf_lfm2_moving_gauge_input_producer_reclosure."
    "paired-actual-normalized-input-anchored-fp64-ridge.v1"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SUPPORTED_TARGET_DTYPES = frozenset({"float32", "bfloat16"})
_ROLE_FAMILY = MappingProxyType(
    {
        "attention_q": "attention",
        "attention_k": "attention",
        "attention_v": "attention",
        "short_conv_in_proj": "short_conv",
        "mlp_w1": "mlp",
        "mlp_w3": "mlp",
    }
)
_ZERO_EXECUTION = {
    "parent_model_objects_accepted": 0,
    "parent_model_forward_calls": 0,
    "parent_model_generate_calls": 0,
    "parent_trace_recaptures": 0,
    "parent_cache_regenerations": 0,
    "target_model_objects_accepted": 0,
    "target_model_forward_calls": 0,
    "target_layer_forward_calls": 0,
    "generation_calls": 0,
    "backward_calls": 0,
    "autograd_grad_calls": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "training_steps": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
}


class Lfm2MovingGaugeInputProducerReclosureError(ValueError):
    """Input-producer evidence, geometry, or publication is invalid."""


class Lfm2MovingGaugeInputProducerAuthenticationError(
    Lfm2MovingGaugeInputProducerReclosureError
):
    """A sealed input-producer artifact failed authentication."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for moving-gauge producer reclosure."
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
        raise Lfm2MovingGaugeInputProducerReclosureError(
            "Input-producer metadata must be finite canonical JSON."
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _seal(body: Mapping[str, Any]) -> Mapping[str, Any]:
    detached = dict(body)
    return MappingProxyType({**detached, "report_sha256": _sha256(detached)})


def verify_lfm2_moving_gauge_input_producer_report(
    value: Mapping[str, Any],
) -> bool:
    """Verify a canonical report emitted by this module."""

    if not isinstance(value, Mapping):
        return False
    try:
        body = dict(value)
        supplied = body.pop("report_sha256", None)
        return isinstance(supplied, str) and supplied == _sha256(body)
    except (TypeError, ValueError):
        return False


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2MovingGaugeInputProducerAuthenticationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2MovingGaugeInputProducerReclosureError(
            f"{name} must be nonempty text."
        )
    return value.strip()


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Lfm2MovingGaugeInputProducerReclosureError(
            f"{name} must be a finite nonnegative number."
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2MovingGaugeInputProducerReclosureError(
            f"{name} must be a finite nonnegative number."
        )
    return result


def _finite_positive(value: Any, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result == 0.0:
        raise Lfm2MovingGaugeInputProducerReclosureError(f"{name} must be positive.")
    return result


def _dtype_name(value: Any) -> str:
    return str(value.dtype).removeprefix("torch.")


def fingerprint_lfm2_input_producer_tensor(value: Any) -> str:
    """Hash a tensor's dtype, shape, and exact dense bytes."""

    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2MovingGaugeInputProducerReclosureError(
            "Only materialized dense tensors can be authenticated."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": _dtype_name(tensor),
        "shape": [int(size) for size in tensor.shape],
    }
    digest = hashlib.sha256(_canonical_json(descriptor))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _matrix(value: Any, name: str, *, dtype: Any | None = None) -> Any:
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
    ):
        raise Lfm2MovingGaugeInputProducerReclosureError(
            f"{name} must be a detached nonempty dense real floating matrix."
        )
    result = value.detach().to(device="cpu", dtype=dtype).contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2MovingGaugeInputProducerReclosureError(
            f"{name} contains NaN or infinity."
        )
    return result


def _row_weights(value: Any | None, rows: int) -> Any:
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
        raise Lfm2MovingGaugeInputProducerReclosureError(
            "row_weights must contain one detached positive value per row."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()
    if not bool(torch.isfinite(result).all().item()) or bool(
        (result <= 0.0).any().item()
    ):
        raise Lfm2MovingGaugeInputProducerReclosureError(
            "row_weights must contain finite positive values."
        )
    return result


def _rmse(prediction: Any, target: Any, weights: Any) -> float:
    torch = _torch()
    energy = (weights.unsqueeze(1) * (prediction - target).square()).sum()
    denominator = weights.sum() * int(target.shape[1])
    return float(torch.sqrt(energy / denominator).item())


def _relative_rmse(prediction: Any, target: Any, weights: Any) -> float:
    torch = _torch()
    scale = torch.sqrt(
        (weights.unsqueeze(1) * target.square()).sum()
        / (weights.sum() * int(target.shape[1]))
    ).clamp_min(torch.finfo(torch.float64).tiny)
    return _rmse(prediction, target, weights) / float(scale.item())


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeInputProducerReclosurePolicy:
    """Fixed analytic ridge policy; diagnostics never select a child."""

    ridge_ratio: float = 1e-8
    absolute_ridge_floor: float = 1e-12
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ridge_ratio",
            _finite_nonnegative(self.ridge_ratio, "ridge_ratio"),
        )
        object.__setattr__(
            self,
            "absolute_ridge_floor",
            _finite_positive(
                self.absolute_ridge_floor,
                "absolute_ridge_floor",
            ),
        )
        expected = _sha256(self.authentication_body())
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2MovingGaugeInputProducerAuthenticationError(
                "Input-producer policy authentication_sha256 is incorrect."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_MOVING_GAUGE_INPUT_PRODUCER_RECLOSURE_SCHEMA,
            "kind": "input_producer_reclosure_policy",
            "ridge_ratio": self.ridge_ratio,
            "absolute_ridge_floor": self.absolute_ridge_floor,
            "solve_dtype": "float64",
            "target_dtypes": sorted(_SUPPORTED_TARGET_DTYPES),
            "closed_form_ridge_only": True,
            "numeric_quality_gate": False,
            "execution": dict(_ZERO_EXECUTION),
        }

    def verify_authentication(self) -> bool:
        return self.authentication_sha256 == _sha256(self.authentication_body())

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.authentication_body(),
            "authentication_sha256": self.authentication_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeInputProducerSiteEvidence:
    """Authenticated paired actual-normalized rows for one producer weight."""

    parameter_id: str
    producer_group_id: str
    producer_role: str
    seed_weight: Any = field(repr=False, compare=False)
    old_actual_normalized_input_rows: Any = field(repr=False, compare=False)
    new_actual_normalized_input_rows: Any = field(repr=False, compare=False)
    row_alignment_sha256: str
    old_child_state_sha256: str
    new_child_state_sha256: str
    row_weights: Any | None = field(default=None, repr=False, compare=False)
    report: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parameter_id = _text(self.parameter_id, "parameter_id")
        group_id = _text(self.producer_group_id, "producer_group_id")
        role = str(self.producer_role).strip().lower()
        if role not in _ROLE_FAMILY:
            raise Lfm2MovingGaugeInputProducerReclosureError(
                f"producer_role must be one of {sorted(_ROLE_FAMILY)}."
            )
        seed = _matrix(self.seed_weight, "seed_weight")
        if _dtype_name(seed) not in _SUPPORTED_TARGET_DTYPES:
            raise Lfm2MovingGaugeInputProducerReclosureError(
                "seed_weight target dtype must be FP32 or BF16."
            )
        old = _matrix(
            self.old_actual_normalized_input_rows,
            "old_actual_normalized_input_rows",
        )
        new = _matrix(
            self.new_actual_normalized_input_rows,
            "new_actual_normalized_input_rows",
        )
        if tuple(old.shape) != tuple(new.shape):
            raise Lfm2MovingGaugeInputProducerReclosureError(
                "Old and new actual normalized rows must have identical shape."
            )
        if int(old.shape[1]) != int(seed.shape[1]):
            raise Lfm2MovingGaugeInputProducerReclosureError(
                "Normalized input width and seed producer input width differ."
            )
        weights = _row_weights(self.row_weights, int(old.shape[0]))
        alignment = _digest(
            self.row_alignment_sha256,
            "row_alignment_sha256",
        )
        old_state = _digest(
            self.old_child_state_sha256,
            "old_child_state_sha256",
        )
        new_state = _digest(
            self.new_child_state_sha256,
            "new_child_state_sha256",
        )
        if (
            not bool(
                _torch().equal(
                    old.to(dtype=_torch().float64), new.to(dtype=_torch().float64)
                )
            )
            and old_state == new_state
        ):
            raise Lfm2MovingGaugeInputProducerAuthenticationError(
                "Changed normalized rows require distinct old/new child-state hashes."
            )
        object.__setattr__(self, "parameter_id", parameter_id)
        object.__setattr__(self, "producer_group_id", group_id)
        object.__setattr__(self, "producer_role", role)
        object.__setattr__(self, "seed_weight", seed)
        object.__setattr__(
            self,
            "old_actual_normalized_input_rows",
            old,
        )
        object.__setattr__(
            self,
            "new_actual_normalized_input_rows",
            new,
        )
        object.__setattr__(self, "row_weights", weights)
        object.__setattr__(self, "row_alignment_sha256", alignment)
        object.__setattr__(self, "old_child_state_sha256", old_state)
        object.__setattr__(self, "new_child_state_sha256", new_state)
        expected = _seal(self.authentication_body())
        supplied = dict(self.report)
        if supplied and supplied != dict(expected):
            raise Lfm2MovingGaugeInputProducerAuthenticationError(
                "Input-producer site evidence report is stale or malformed."
            )
        object.__setattr__(
            self, "report", expected if not supplied else MappingProxyType(supplied)
        )

    @property
    def producer_family(self) -> str:
        return _ROLE_FAMILY[self.producer_role]

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_MOVING_GAUGE_INPUT_PRODUCER_RECLOSURE_SCHEMA,
            "kind": "paired_actual_normalized_input_site_evidence",
            "parameter_id": self.parameter_id,
            "producer_group_id": self.producer_group_id,
            "producer_family": self.producer_family,
            "producer_role": self.producer_role,
            "row_count": int(self.old_actual_normalized_input_rows.shape[0]),
            "input_width": int(self.seed_weight.shape[1]),
            "output_width": int(self.seed_weight.shape[0]),
            "target_dtype": _dtype_name(self.seed_weight),
            "row_alignment_sha256": self.row_alignment_sha256,
            "old_child_state_sha256": self.old_child_state_sha256,
            "new_child_state_sha256": self.new_child_state_sha256,
            "seed_weight_sha256": fingerprint_lfm2_input_producer_tensor(
                self.seed_weight
            ),
            "old_actual_normalized_input_rows_sha256": (
                fingerprint_lfm2_input_producer_tensor(
                    self.old_actual_normalized_input_rows
                )
            ),
            "new_actual_normalized_input_rows_sha256": (
                fingerprint_lfm2_input_producer_tensor(
                    self.new_actual_normalized_input_rows
                )
            ),
            "row_weights_sha256": fingerprint_lfm2_input_producer_tensor(
                self.row_weights
            ),
            "normalization_evidence_kind": (
                "paired-pre-change-and-post-change-actual-normalized-child-rows"
            ),
            "arbitrary_residual_map_commuted_through_rmsnorm": False,
            "cached_parent_evidence_required": False,
            "parent_execution_required": False,
            "execution": dict(_ZERO_EXECUTION),
        }

    @property
    def evidence_sha256(self) -> str:
        return str(self.report["report_sha256"])

    def verify_authentication(self) -> bool:
        try:
            return dict(self.report) == dict(_seal(self.authentication_body()))
        except (KeyError, TypeError, ValueError):
            return False


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeInputProducerReplacement:
    """One target-dtype dense producer replacement and its sealed receipt."""

    parameter_id: str
    replacement_weight: Any = field(repr=False, compare=False)
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        parameter_id = _text(self.parameter_id, "parameter_id")
        replacement = _matrix(
            self.replacement_weight,
            "replacement_weight",
        )
        if _dtype_name(replacement) not in _SUPPORTED_TARGET_DTYPES:
            raise Lfm2MovingGaugeInputProducerReclosureError(
                "Replacement target dtype must be FP32 or BF16."
            )
        if not verify_lfm2_moving_gauge_input_producer_report(self.report):
            raise Lfm2MovingGaugeInputProducerAuthenticationError(
                "Input-producer replacement receipt seal is invalid."
            )
        if self.report.get("parameter_id") != parameter_id or self.report.get(
            "replacement_weight_sha256"
        ) != fingerprint_lfm2_input_producer_tensor(replacement):
            raise Lfm2MovingGaugeInputProducerAuthenticationError(
                "Input-producer replacement differs from its receipt."
            )
        object.__setattr__(self, "parameter_id", parameter_id)
        object.__setattr__(self, "replacement_weight", replacement)
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    def verify_authentication(self) -> bool:
        try:
            return verify_lfm2_moving_gauge_input_producer_report(
                self.report
            ) and self.report[
                "replacement_weight_sha256"
            ] == fingerprint_lfm2_input_producer_tensor(self.replacement_weight)
        except (KeyError, TypeError, ValueError):
            return False


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeInputProducerReclosureState:
    """Authenticated dense replacements; no runtime or parent sidecar."""

    evidence: tuple[Lfm2MovingGaugeInputProducerSiteEvidence, ...]
    replacements: tuple[Lfm2MovingGaugeInputProducerReplacement, ...]
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        evidence = tuple(self.evidence)
        replacements = tuple(self.replacements)
        if not evidence or len(evidence) != len(replacements):
            raise Lfm2MovingGaugeInputProducerReclosureError(
                "Reclosure state requires one replacement per evidence site."
            )
        if not all(value.verify_authentication() for value in evidence):
            raise Lfm2MovingGaugeInputProducerAuthenticationError(
                "Reclosure state evidence failed authentication."
            )
        if not all(value.verify_authentication() for value in replacements):
            raise Lfm2MovingGaugeInputProducerAuthenticationError(
                "Reclosure state replacements failed authentication."
            )
        evidence_ids = tuple(value.parameter_id for value in evidence)
        replacement_ids = tuple(value.parameter_id for value in replacements)
        if evidence_ids != replacement_ids or len(set(evidence_ids)) != len(
            evidence_ids
        ):
            raise Lfm2MovingGaugeInputProducerReclosureError(
                "Reclosure state parameter ids must be aligned and unique."
            )
        if not verify_lfm2_moving_gauge_input_producer_report(self.report):
            raise Lfm2MovingGaugeInputProducerAuthenticationError(
                "Reclosure state report seal is invalid."
            )
        if (
            self.report.get("parameter_ids") != list(evidence_ids)
            or self.report.get("evidence_sha256")
            != [value.evidence_sha256 for value in evidence]
            or self.report.get("replacement_receipt_sha256")
            != [value.report["report_sha256"] for value in replacements]
        ):
            raise Lfm2MovingGaugeInputProducerAuthenticationError(
                "Reclosure state contents differ from its report."
            )
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "replacements", replacements)
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    @property
    def replacements_by_parameter_id(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                replacement.parameter_id: replacement.replacement_weight
                for replacement in self.replacements
            }
        )

    def verify_authentication(self) -> bool:
        try:
            return (
                verify_lfm2_moving_gauge_input_producer_report(self.report)
                and all(value.verify_authentication() for value in self.evidence)
                and all(value.verify_authentication() for value in self.replacements)
                and self.report["evidence_sha256"]
                == [value.evidence_sha256 for value in self.evidence]
                and self.report["replacement_receipt_sha256"]
                == [value.report["report_sha256"] for value in self.replacements]
            )
        except (KeyError, TypeError, ValueError):
            return False


def _group_key(site: Lfm2MovingGaugeInputProducerSiteEvidence) -> tuple[str, str]:
    return (site.producer_group_id, site.producer_family)


def _validate_group(
    sites: Sequence[Lfm2MovingGaugeInputProducerSiteEvidence],
) -> None:
    if not sites:
        raise AssertionError("Internal input-producer group is empty.")
    first = sites[0]
    if len({site.producer_role for site in sites}) != len(sites):
        raise Lfm2MovingGaugeInputProducerReclosureError(
            f"Producer group {first.producer_group_id} repeats a semantic role."
        )
    shared = (
        first.producer_family,
        int(first.seed_weight.shape[1]),
        first.row_alignment_sha256,
        first.old_child_state_sha256,
        first.new_child_state_sha256,
        fingerprint_lfm2_input_producer_tensor(first.old_actual_normalized_input_rows),
        fingerprint_lfm2_input_producer_tensor(first.new_actual_normalized_input_rows),
        fingerprint_lfm2_input_producer_tensor(first.row_weights),
    )
    for site in sites[1:]:
        candidate = (
            site.producer_family,
            int(site.seed_weight.shape[1]),
            site.row_alignment_sha256,
            site.old_child_state_sha256,
            site.new_child_state_sha256,
            fingerprint_lfm2_input_producer_tensor(
                site.old_actual_normalized_input_rows
            ),
            fingerprint_lfm2_input_producer_tensor(
                site.new_actual_normalized_input_rows
            ),
            fingerprint_lfm2_input_producer_tensor(site.row_weights),
        )
        if candidate != shared:
            raise Lfm2MovingGaugeInputProducerReclosureError(
                f"Producer group {first.producer_group_id} does not share one "
                "authenticated normalized-input row contract."
            )


def _solve_group(
    sites: Sequence[Lfm2MovingGaugeInputProducerSiteEvidence],
    *,
    policy: Lfm2MovingGaugeInputProducerReclosurePolicy,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Solve one grouped normal equation and return unsplit FP64 columns."""

    torch = _torch()
    first = sites[0]
    old = first.old_actual_normalized_input_rows.to(dtype=torch.float64)
    new = first.new_actual_normalized_input_rows.to(dtype=torch.float64)
    weights = first.row_weights
    anchors = [site.seed_weight.to(dtype=torch.float64).T for site in sites]
    targets = [old @ anchor for anchor in anchors]
    anchor = torch.cat(anchors, dim=1)
    target = torch.cat(targets, dim=1)
    exact_normalized_input_noop = bool(torch.equal(old, new))
    weighted_new = new * weights.unsqueeze(1)
    gram = new.T @ weighted_new
    mean_gram_scale = float((torch.trace(gram) / int(gram.shape[0])).item())
    ridge = max(
        policy.absolute_ridge_floor,
        policy.ridge_ratio * max(mean_gram_scale, 0.0),
    )
    if exact_normalized_input_noop:
        solution = anchor.clone()
        solve_performed = False
        identity_solution = True
    else:
        lhs = gram + ridge * torch.eye(
            int(gram.shape[0]),
            dtype=torch.float64,
        )
        rhs = new.T @ (weights.unsqueeze(1) * target) + ridge * anchor
        factor, info = torch.linalg.cholesky_ex(lhs, check_errors=False)
        if bool((info != 0).any().item()):
            raise Lfm2MovingGaugeInputProducerReclosureError(
                f"Producer group {first.producer_group_id} ridge system is not "
                "positive definite."
            )
        solution = torch.cholesky_solve(rhs, factor)
        solve_performed = True
        identity_solution = False
    if not bool(torch.isfinite(solution).all().item()):
        raise Lfm2MovingGaugeInputProducerReclosureError(
            f"Producer group {first.producer_group_id} produced non-finite weights."
        )
    normal_residual = (
        gram @ solution
        - new.T @ (weights.unsqueeze(1) * target)
        + ridge * (solution - anchor)
    )
    singular_values = torch.linalg.svdvals(new * torch.sqrt(weights).unsqueeze(1))
    largest = float(singular_values.max().item())
    tolerance = (
        max(int(new.shape[0]), int(new.shape[1]))
        * torch.finfo(torch.float64).eps
        * largest
    )
    observed_rank = int((singular_values > tolerance).sum().item())
    body = {
        "schema": LFM2_MOVING_GAUGE_INPUT_PRODUCER_RECLOSURE_SCHEMA,
        "kind": "grouped_input_producer_fp64_ridge_solve",
        "producer_group_id": first.producer_group_id,
        "producer_family": first.producer_family,
        "parameter_ids": [site.parameter_id for site in sites],
        "producer_roles": [site.producer_role for site in sites],
        "row_count": int(new.shape[0]),
        "input_width": int(new.shape[1]),
        "combined_output_width": int(target.shape[1]),
        "observed_design_rank": observed_rank,
        "ridge": ridge,
        "ridge_ratio": policy.ridge_ratio,
        "absolute_ridge_floor": policy.absolute_ridge_floor,
        "solve_dtype": "float64",
        "closed_form_cholesky_solve_performed": solve_performed,
        "closed_form_identity_solution": identity_solution,
        "exact_normalized_input_noop": exact_normalized_input_noop,
        "normal_equation_residual_l2_norm": float(
            torch.linalg.vector_norm(normal_residual).item()
        ),
        "frozen_target_formula": "old_design @ seed_weight.T",
        "anchor_kind": "seed_target_dtype_dense_weight",
        "arbitrary_residual_map_commuted_through_rmsnorm": False,
        "all_numeric_metrics_diagnostic_only": True,
        "automatic_quality_accept_reject": False,
        "execution": dict(_ZERO_EXECUTION),
    }
    return MappingProxyType({"solution": solution, "target": target}), _seal(body)


def compile_lfm2_moving_gauge_input_producer_reclosure(
    site_evidence: Sequence[Lfm2MovingGaugeInputProducerSiteEvidence],
    *,
    policy: Lfm2MovingGaugeInputProducerReclosurePolicy | None = None,
) -> Lfm2MovingGaugeInputProducerReclosureState:
    """Reclose all supplied input producers without executing a model."""

    torch = _torch()
    active_policy = policy or Lfm2MovingGaugeInputProducerReclosurePolicy()
    sites = tuple(site_evidence)
    if (
        not isinstance(active_policy, Lfm2MovingGaugeInputProducerReclosurePolicy)
        or not active_policy.verify_authentication()
    ):
        raise Lfm2MovingGaugeInputProducerAuthenticationError(
            "Input-producer reclosure policy failed authentication."
        )
    if not sites or not all(
        isinstance(site, Lfm2MovingGaugeInputProducerSiteEvidence)
        and site.verify_authentication()
        for site in sites
    ):
        raise Lfm2MovingGaugeInputProducerAuthenticationError(
            "Input-producer site evidence failed authentication."
        )
    if len({site.parameter_id for site in sites}) != len(sites):
        raise Lfm2MovingGaugeInputProducerReclosureError(
            "Input-producer parameter ids must be globally unique."
        )
    groups: dict[tuple[str, str], list[Lfm2MovingGaugeInputProducerSiteEvidence]] = {}
    group_families: dict[str, str] = {}
    for site in sites:
        prior_family = group_families.setdefault(
            site.producer_group_id,
            site.producer_family,
        )
        if prior_family != site.producer_family:
            raise Lfm2MovingGaugeInputProducerReclosureError(
                f"Producer group {site.producer_group_id} mixes semantic families."
            )
        groups.setdefault(_group_key(site), []).append(site)
    for values in groups.values():
        values.sort(key=lambda value: (value.producer_role, value.parameter_id))
        _validate_group(values)

    def operation() -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
        replacements: dict[str, Any] = {}
        receipts: dict[str, Mapping[str, Any]] = {}
        for group_key in sorted(groups):
            group_sites = groups[group_key]
            solved, group_receipt = _solve_group(
                group_sites,
                policy=active_policy,
            )
            solution = solved["solution"]
            target = solved["target"]
            cursor = 0
            for site in group_sites:
                output_width = int(site.seed_weight.shape[0])
                columns = solution[:, cursor : cursor + output_width]
                target_columns = target[:, cursor : cursor + output_width]
                cursor += output_width
                replacement = (
                    columns.T.to(dtype=site.seed_weight.dtype).contiguous().clone()
                )
                if not bool(torch.isfinite(replacement).all().item()):
                    raise Lfm2MovingGaugeInputProducerReclosureError(
                        f"Replacement for {site.parameter_id} is non-finite after "
                        "the direct target-dtype cast."
                    )
                old = site.old_actual_normalized_input_rows.to(dtype=torch.float64)
                new = site.new_actual_normalized_input_rows.to(dtype=torch.float64)
                seed = site.seed_weight.to(dtype=torch.float64)
                cast = replacement.to(dtype=torch.float64)
                before = new @ seed.T
                after = new @ cast.T
                target_direct = old @ seed.T
                if not bool(torch.equal(target_columns, target_direct)):
                    raise AssertionError(
                        "Grouped target split changed the frozen producer output."
                    )
                changed = int(
                    torch.count_nonzero(replacement != site.seed_weight).item()
                )
                receipt_body = {
                    "schema": LFM2_MOVING_GAUGE_INPUT_PRODUCER_RECLOSURE_SCHEMA,
                    "kind": "target_dtype_input_producer_dense_replacement",
                    "implementation": _IMPLEMENTATION,
                    "parameter_id": site.parameter_id,
                    "producer_group_id": site.producer_group_id,
                    "producer_family": site.producer_family,
                    "producer_role": site.producer_role,
                    "site_evidence_sha256": site.evidence_sha256,
                    "group_solve_sha256": group_receipt["report_sha256"],
                    "group_solve_receipt": dict(group_receipt),
                    "seed_weight_sha256": (
                        fingerprint_lfm2_input_producer_tensor(site.seed_weight)
                    ),
                    "replacement_weight_sha256": (
                        fingerprint_lfm2_input_producer_tensor(replacement)
                    ),
                    "frozen_target_output_sha256": (
                        fingerprint_lfm2_input_producer_tensor(target_direct)
                    ),
                    "target_dtype": _dtype_name(replacement),
                    "shape": [int(size) for size in replacement.shape],
                    "changed_coordinate_count": changed,
                    "target_dtype_effectful": changed > 0,
                    "exact_normalized_input_noop": group_receipt[
                        "exact_normalized_input_noop"
                    ],
                    "pre_reclosure_rmse": _rmse(
                        before,
                        target_direct,
                        site.row_weights,
                    ),
                    "post_reclosure_rmse": _rmse(
                        after,
                        target_direct,
                        site.row_weights,
                    ),
                    "pre_reclosure_relative_rmse": _relative_rmse(
                        before,
                        target_direct,
                        site.row_weights,
                    ),
                    "post_reclosure_relative_rmse": _relative_rmse(
                        after,
                        target_direct,
                        site.row_weights,
                    ),
                    "maximum_absolute_output_error": float(
                        (after - target_direct).abs().max().item()
                    ),
                    "replacement_delta_l2_norm": float(
                        torch.linalg.vector_norm(cast - seed).item()
                    ),
                    "frozen_target_formula": "old_design @ seed_weight.T",
                    "direct_target_dtype_dense_replacement": True,
                    "ordinary_bfloat16_cast_is_quantization": False,
                    "seed_tensor_mutated": False,
                    "runtime_sidecar_required": False,
                    "parent_sidecar_required": False,
                    "arbitrary_residual_map_commuted_through_rmsnorm": False,
                    "all_numeric_metrics_diagnostic_only": True,
                    "automatic_quality_accept_reject": False,
                    "manual_whole_response_review_required": True,
                    "execution": dict(_ZERO_EXECUTION),
                }
                replacements[site.parameter_id] = replacement
                receipts[site.parameter_id] = _seal(receipt_body)
            if cursor != int(solution.shape[1]):
                raise AssertionError("Grouped producer split lost output columns.")
        return replacements, receipts

    (replacement_by_id, receipt_by_id), guard = run_fold_only_callback(
        operation,
        phase="lfm2-moving-gauge-input-producer-reclosure",
        protected_models=(),
    )
    ordered_sites = tuple(sorted(sites, key=lambda site: site.parameter_id))
    replacements = tuple(
        Lfm2MovingGaugeInputProducerReplacement(
            parameter_id=site.parameter_id,
            replacement_weight=replacement_by_id[site.parameter_id],
            report=receipt_by_id[site.parameter_id],
        )
        for site in ordered_sites
    )
    changed_by_parameter = {
        replacement.parameter_id: int(replacement.report["changed_coordinate_count"])
        for replacement in replacements
    }
    body = {
        "schema": LFM2_MOVING_GAUGE_INPUT_PRODUCER_RECLOSURE_SCHEMA,
        "kind": "moving_gauge_input_producer_reclosure_state",
        "implementation": _IMPLEMENTATION,
        "policy_sha256": active_policy.authentication_sha256,
        "parameter_ids": [site.parameter_id for site in ordered_sites],
        "producer_groups": [
            {
                "producer_group_id": group_id,
                "producer_family": family,
            }
            for group_id, family in sorted(groups)
        ],
        "evidence_sha256": [site.evidence_sha256 for site in ordered_sites],
        "replacement_receipt_sha256": [
            replacement.report["report_sha256"] for replacement in replacements
        ],
        "replacement_weight_sha256": [
            fingerprint_lfm2_input_producer_tensor(replacement.replacement_weight)
            for replacement in replacements
        ],
        "changed_coordinate_count_by_parameter": changed_by_parameter,
        "changed_coordinate_count": sum(changed_by_parameter.values()),
        "changed_parameter_count": sum(
            value > 0 for value in changed_by_parameter.values()
        ),
        "exact_noop_parameter_count": sum(
            replacement.report["exact_normalized_input_noop"]
            for replacement in replacements
        ),
        "direct_target_dtype_dense_replacements": True,
        "runtime_sidecar_required": False,
        "parent_sidecar_required": False,
        "parent_or_target_model_execution_surface": False,
        "arbitrary_residual_map_commuted_through_rmsnorm": False,
        "all_numeric_metrics_diagnostic_only": True,
        "automatic_quality_accept_reject": False,
        "manual_whole_response_review_required": True,
        "fold_only_guard": guard.as_dict(),
        "execution": {
            **_ZERO_EXECUTION,
            "optimizer_constructions": guard.optimizer_constructions,
            "optimizer_steps": guard.optimizer_steps,
            "backward_calls": guard.backward_calls,
            "autograd_grad_calls": guard.autograd_grad_calls,
            "parameters_pruned": guard.pruning_operations,
            "parameters_quantized": guard.quantization_operations,
            "guard_clean": guard.clean,
            "closed_form_linear_solves_only": True,
        },
    }
    return Lfm2MovingGaugeInputProducerReclosureState(
        evidence=ordered_sites,
        replacements=replacements,
        report=_seal(body),
    )


__all__ = [
    "LFM2_MOVING_GAUGE_INPUT_PRODUCER_RECLOSURE_SCHEMA",
    "Lfm2MovingGaugeInputProducerAuthenticationError",
    "Lfm2MovingGaugeInputProducerReclosureError",
    "Lfm2MovingGaugeInputProducerReclosurePolicy",
    "Lfm2MovingGaugeInputProducerReclosureState",
    "Lfm2MovingGaugeInputProducerReplacement",
    "Lfm2MovingGaugeInputProducerSiteEvidence",
    "compile_lfm2_moving_gauge_input_producer_reclosure",
    "fingerprint_lfm2_input_producer_tensor",
    "verify_lfm2_moving_gauge_input_producer_report",
]
