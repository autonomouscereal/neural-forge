"""Deterministic numerical policy for the LFM2 CPCF short-conv v5 lane.

This module is deliberately additive.  It does not import, wrap, or alter the
v4 compiler and therefore cannot change a v4 schema, report, fingerprint, or
selection result.  It provides four building blocks for a future v5 compiler:

* sealed endpoint-first rescoring of already sealed v4 selection receipts;
* condition-capped, equilibrated prior-residual affine ridge solves;
* a balanced all-source-channel basis that commutes with a depthwise FIR; and
* interpolation over every accepted edge of a fit-only projection path.

Every operation is closed form.  There are no gradients, optimizer steps,
training steps, pruning decisions, quantization operations, model forwards, or
validation-driven choices in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


LFM2_CPCF_SHORT_CONV_V5_SCHEMA = "neural-forge-lfm2-cpcf-short-conv-v5"
LFM2_CPCF_SHORT_CONV_V4_SCHEMA = "neural-forge-lfm2-cpcf-short-conv-v4"
V5_PRIOR_CONTROL_SUFFIX = "--anchor-00-prior"
V5_CONTEXT_EXCHANGE_BETA = 0.05
V5_CONDITION_CAP_CANDIDATES = (1_000.0, 10_000.0)
V5_ACCEPTED_EDGE_INTERPOLATION_FRACTIONS = (
    1.0 / 64.0,
    1.0 / 32.0,
    1.0 / 16.0,
    1.0 / 8.0,
    1.0 / 4.0,
    1.0 / 2.0,
)
V5_DIRECT_SEED_DISTANCE_ROW_CHUNK_SIZE = 128

_SELECTION_METRICS = (
    "maximum_case_relative_rmse",
    "minimum_case_cosine_similarity",
    "mean_case_relative_rmse",
    "mean_case_cosine_similarity",
    "maximum_case_context_relative_rmse",
    "minimum_case_context_cosine_similarity",
    "mean_case_context_relative_rmse",
    "mean_case_context_cosine_similarity",
)
_ZERO_TRAINING_EXECUTION = {
    "backward_calls": 0,
    "optimizer_steps": 0,
    "training_steps": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
}


class Lfm2CpcfShortConvV5Error(ValueError):
    """The v5 numerical policy or its authenticated evidence is invalid."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for LFM2 CPCF short-conv v5 math."
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
        raise Lfm2CpcfShortConvV5Error(
            "V5 metadata must be finite canonical JSON."
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _seal_report(body: Mapping[str, Any]) -> Mapping[str, Any]:
    detached = dict(body)
    return MappingProxyType({**detached, "report_sha256": _sha256(detached)})


def verify_lfm2_cpcf_short_conv_v5_report(report: Mapping[str, Any]) -> bool:
    """Verify one top-level v5 (or input v4) canonical report seal."""

    if not isinstance(report, Mapping):
        return False
    try:
        body = dict(report)
        supplied = body.pop("report_sha256", None)
        return isinstance(supplied, str) and supplied == _sha256(body)
    except (Lfm2CpcfShortConvV5Error, TypeError, ValueError):
        return False


def _valid_sha256(value: Any, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise Lfm2CpcfShortConvV5Error(f"{name} must be a lowercase SHA-256 digest.")
    return text


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Lfm2CpcfShortConvV5Error(f"{name} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise Lfm2CpcfShortConvV5Error(f"{name} must be a finite number.")
    return result


def _positive_float(value: Any, name: str, *, allow_zero: bool = False) -> float:
    result = _finite_float(value, name)
    if result < 0.0 or (not allow_zero and result == 0.0):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise Lfm2CpcfShortConvV5Error(f"{name} must be a finite {qualifier} number.")
    return result


def _snapshot_tensor(
    value: Any,
    name: str,
    *,
    dimensions: int | tuple[int, ...],
    dtype: Any | None = None,
) -> Any:
    torch = _torch()
    allowed_dimensions = (
        (dimensions,) if isinstance(dimensions, int) else tuple(dimensions)
    )
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim not in allowed_dimensions
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or value.is_complex()
    ):
        raise Lfm2CpcfShortConvV5Error(
            f"{name} must be a nonempty dense real floating tensor with rank "
            f"in {allowed_dimensions}."
        )
    target_dtype = torch.float64 if dtype is None else dtype
    result = value.detach().to(device="cpu", dtype=target_dtype).contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CpcfShortConvV5Error(f"{name} contains NaN or infinity.")
    return result


def _snapshot_state_tensor(value: Any, name: str) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or value.is_complex()
    ):
        raise Lfm2CpcfShortConvV5Error(
            f"{name} must be a nonempty dense real floating tensor."
        )
    result = value.detach().to(device="cpu").contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CpcfShortConvV5Error(f"{name} contains NaN or infinity.")
    return result


def _snapshot_mask(value: Any | None, name: str) -> Any | None:
    if value is None:
        return None
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 1
        or value.numel() == 0
    ):
        raise Lfm2CpcfShortConvV5Error(f"{name} must be a nonempty dense rank-1 mask.")
    return value.detach().to(device="cpu", dtype=torch.bool).contiguous().clone()


def _dtype_name(value: Any) -> str:
    return str(value.dtype).removeprefix("torch.")


def _tensor_sha256(value: Any, name: str) -> str:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2CpcfShortConvV5Error(
            f"{name} must be a materialized dense tensor for authentication."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": _dtype_name(tensor),
        "shape": [int(size) for size in tensor.shape],
        "layout": str(tensor.layout),
    }
    digest = hashlib.sha256(_canonical_json(descriptor))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _state_sha256(state: Mapping[str, Any], name: str) -> str:
    return _sha256(
        {
            role: _tensor_sha256(value, f"{name}.{role}")
            for role, value in sorted(state.items())
        }
    )


def _rms(value: Any) -> float:
    return float(value.square().mean().sqrt().item())


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5SelectionPolicy:
    """The frozen endpoint-first selection contract from the layer-0 audit."""

    endpoint_maximum_relative_rmse: float = 1.0
    context_maximum_relative_rmse: float = 1.0
    endpoint_minimum_cosine_similarity: float = 0.0
    context_minimum_cosine_similarity: float = 0.0
    context_exchange_beta: float = V5_CONTEXT_EXCHANGE_BETA
    prior_control_suffix: str = V5_PRIOR_CONTROL_SUFFIX
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        frozen_values = (
            (self.endpoint_maximum_relative_rmse, 1.0, "endpoint RMSE gate"),
            (self.context_maximum_relative_rmse, 1.0, "context RMSE gate"),
            (
                self.endpoint_minimum_cosine_similarity,
                0.0,
                "endpoint cosine gate",
            ),
            (
                self.context_minimum_cosine_similarity,
                0.0,
                "context cosine gate",
            ),
            (
                self.context_exchange_beta,
                V5_CONTEXT_EXCHANGE_BETA,
                "context exchange beta",
            ),
        )
        for supplied, expected, name in frozen_values:
            if _finite_float(supplied, name) != expected:
                raise Lfm2CpcfShortConvV5Error(
                    f"The v5 {name} is frozen at {expected!r}."
                )
        if self.prior_control_suffix != V5_PRIOR_CONTROL_SUFFIX:
            raise Lfm2CpcfShortConvV5Error(
                f"The v5 prior control suffix is frozen at {V5_PRIOR_CONTROL_SUFFIX!r}."
            )
        expected = _sha256(self.authentication_body())
        supplied_digest = str(self.authentication_sha256).strip().lower()
        if supplied_digest and supplied_digest != expected:
            raise Lfm2CpcfShortConvV5Error(
                "selection policy authentication_sha256 is incorrect."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CPCF_SHORT_CONV_V5_SCHEMA,
            "kind": "endpoint_first_selection_policy",
            "absolute_gates": {
                "maximum_case_relative_rmse": 1.0,
                "maximum_case_context_relative_rmse": 1.0,
                "minimum_case_cosine_similarity": 0.0,
                "minimum_case_context_cosine_similarity": 0.0,
            },
            "context_exchange_beta": V5_CONTEXT_EXCHANGE_BETA,
            "prior_control_suffix": V5_PRIOR_CONTROL_SUFFIX,
            "validation_accessible": False,
            "minimum_gate_slack_objective": False,
            "lexicographic_key": [
                "maximum_case_relative_rmse",
                "-minimum_case_cosine_similarity",
                "mean_case_relative_rmse",
                "-mean_case_cosine_similarity",
                "maximum_case_context_relative_rmse",
                "-minimum_case_context_cosine_similarity",
                "mean_case_context_relative_rmse",
                "-mean_case_context_cosine_similarity",
                "candidate_id",
            ],
        }

    def verify_authentication(self) -> bool:
        return self.authentication_sha256 == _sha256(self.authentication_body())

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.authentication_body(),
            "authentication_sha256": self.authentication_sha256,
        }


def _selection_evidence_contract(report: Mapping[str, Any]) -> tuple[Any, ...]:
    target_dtype = report.get("target_dtype")
    if target_dtype not in {"bfloat16", "float16", "float32", "float64"}:
        raise Lfm2CpcfShortConvV5Error(
            "V4 selection reports require a supported target dtype."
        )
    policy_sha256 = _valid_sha256(
        report.get("policy_sha256"),
        "policy_sha256",
    )
    evidence_sha256 = _valid_sha256(
        report.get("evidence_sha256"),
        "evidence_sha256",
    )
    case_reports = report.get("cases")
    if not isinstance(case_reports, list) or not case_reports:
        raise Lfm2CpcfShortConvV5Error(
            "V4 selection reports require nonempty case receipts."
        )
    contract_cases = []
    for case in case_reports:
        if not isinstance(case, Mapping):
            raise Lfm2CpcfShortConvV5Error(
                "V4 selection case receipts must be mappings."
            )
        projection_digest = _valid_sha256(
            case.get("projection_case_authentication_sha256"),
            "projection_case_authentication_sha256",
        )
        target_digest = _valid_sha256(
            case.get("target_sha256"),
            "target_sha256",
        )
        context_target_digest = _valid_sha256(
            case.get("context_target_sha256"),
            "context_target_sha256",
        )
        valid_rows_digest = case.get("valid_rows_sha256")
        if valid_rows_digest is not None:
            valid_rows_digest = _valid_sha256(
                valid_rows_digest,
                "valid_rows_sha256",
            )
        valid_context_rows_digest = case.get("valid_context_rows_sha256")
        if valid_context_rows_digest is not None:
            valid_context_rows_digest = _valid_sha256(
                valid_context_rows_digest,
                "valid_context_rows_sha256",
            )
        contract_cases.append(
            (
                case.get("case_id"),
                case.get("rows"),
                case.get("width"),
                target_digest,
                valid_rows_digest,
                projection_digest,
                case.get("context_rows"),
                case.get("context_width"),
                context_target_digest,
                valid_context_rows_digest,
            )
        )
    return (
        target_dtype,
        policy_sha256,
        evidence_sha256,
        tuple(contract_cases),
    )


def _normalize_sealed_v4_selection_report(
    candidate_id: str,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise Lfm2CpcfShortConvV5Error("Selection candidate ids must be nonempty text.")
    if not isinstance(report, Mapping) or not verify_lfm2_cpcf_short_conv_v5_report(
        report
    ):
        raise Lfm2CpcfShortConvV5Error(
            f"Selection report {candidate_id!r} is not canonically sealed."
        )
    if (
        report.get("schema") != LFM2_CPCF_SHORT_CONV_V4_SCHEMA
        or report.get("kind") != "target_dtype_evaluation"
        or report.get("split") != "selection"
        or report.get("context_evidence_present") is not True
        or report.get("actual_parameter_and_intermediate_target_dtype_replay")
        is not True
    ):
        raise Lfm2CpcfShortConvV5Error(
            "V5 rescoring accepts only sealed v4 selection reports with full "
            "context evidence and actual parameter/intermediate target-dtype replay."
        )
    if (
        report.get("selection_used_for_parameter_fitting") is not False
        or report.get("validation_used_for_parameter_fitting") is not False
    ):
        raise Lfm2CpcfShortConvV5Error(
            "A rescored report indicates fit/selection/validation leakage."
        )
    metrics = {
        name: _finite_float(report.get(name), f"{candidate_id}.{name}")
        for name in _SELECTION_METRICS
    }
    return {
        "candidate_id": candidate_id,
        "metrics": metrics,
        "report_sha256": report["report_sha256"],
        "candidate_state_sha256": _valid_sha256(
            report.get("candidate_state_sha256"),
            "candidate_state_sha256",
        ),
        "evidence_contract": _selection_evidence_contract(report),
    }


def _context_exchange_constraints(
    candidate: Mapping[str, float],
    prior: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    beta = V5_CONTEXT_EXCHANGE_BETA

    def rmse_constraint(
        *,
        endpoint_name: str,
        context_name: str,
    ) -> dict[str, Any]:
        endpoint_gain = max(0.0, prior[endpoint_name] - candidate[endpoint_name])
        context_damage = max(0.0, candidate[context_name] - prior[context_name])
        budget = beta * endpoint_gain
        return {
            "endpoint_gain": endpoint_gain,
            "context_damage": context_damage,
            "allowed_context_damage": budget,
            "passed": context_damage <= budget,
            "formula": ("max(0,Rh(c)-Rh(p)) <= beta*max(0,Re(p)-Re(c))"),
        }

    def cosine_constraint(
        *,
        endpoint_name: str,
        context_name: str,
    ) -> dict[str, Any]:
        endpoint_gain = max(0.0, candidate[endpoint_name] - prior[endpoint_name])
        context_damage = max(0.0, prior[context_name] - candidate[context_name])
        budget = beta * endpoint_gain
        return {
            "endpoint_gain": endpoint_gain,
            "context_damage": context_damage,
            "allowed_context_damage": budget,
            "passed": context_damage <= budget,
            "formula": ("max(0,Ch(p)-Ch(c)) <= beta*max(0,Ce(c)-Ce(p))"),
        }

    return {
        "maximum_relative_rmse": rmse_constraint(
            endpoint_name="maximum_case_relative_rmse",
            context_name="maximum_case_context_relative_rmse",
        ),
        "mean_relative_rmse": rmse_constraint(
            endpoint_name="mean_case_relative_rmse",
            context_name="mean_case_context_relative_rmse",
        ),
        "minimum_cosine_similarity": cosine_constraint(
            endpoint_name="minimum_case_cosine_similarity",
            context_name="minimum_case_context_cosine_similarity",
        ),
        "mean_cosine_similarity": cosine_constraint(
            endpoint_name="mean_case_cosine_similarity",
            context_name="mean_case_context_cosine_similarity",
        ),
    }


def rescore_sealed_v4_short_conv_selection_frontier(
    candidates: Mapping[str, Mapping[str, Any]],
    *,
    policy: Lfm2CpcfV5SelectionPolicy | None = None,
) -> Mapping[str, Any]:
    """Rescore a complete sealed v4 selection frontier under the v5 contract.

    The function has no validation argument and rejects every non-selection
    receipt.  It freezes the transported ``--anchor-00-prior`` control, applies
    the four absolute gates and four prior-relative context-exchange gates,
    rejects a frontier with zero feasible states, and then applies the exact
    endpoint-first lexicographic key recovered from the layer-0 audit.
    """

    if not isinstance(candidates, Mapping) or not candidates:
        raise Lfm2CpcfShortConvV5Error(
            "V5 selection rescoring requires at least one candidate."
        )
    selection_policy = policy or Lfm2CpcfV5SelectionPolicy()
    if not isinstance(selection_policy, Lfm2CpcfV5SelectionPolicy):
        raise TypeError("policy must be Lfm2CpcfV5SelectionPolicy.")
    if not selection_policy.verify_authentication():
        raise Lfm2CpcfShortConvV5Error("Selection policy failed authentication.")

    normalized = [
        _normalize_sealed_v4_selection_report(candidate_id, report)
        for candidate_id, report in sorted(candidates.items())
    ]
    evidence_contract = normalized[0]["evidence_contract"]
    if any(value["evidence_contract"] != evidence_contract for value in normalized):
        raise Lfm2CpcfShortConvV5Error(
            "Selection candidates do not share identical evidence and dtype."
        )
    prior_controls = [
        value
        for value in normalized
        if value["candidate_id"].endswith(V5_PRIOR_CONTROL_SUFFIX)
    ]
    if len(prior_controls) != 1:
        raise Lfm2CpcfShortConvV5Error(
            "The sealed frontier must contain exactly one transported "
            f"{V5_PRIOR_CONTROL_SUFFIX!r} control."
        )
    prior = prior_controls[0]
    prior_metrics = prior["metrics"]

    inventory = []
    feasible = []
    for value in normalized:
        metrics = value["metrics"]
        absolute_checks = {
            "endpoint_maximum_relative_rmse": (
                metrics["maximum_case_relative_rmse"] <= 1.0
            ),
            "context_maximum_relative_rmse": (
                metrics["maximum_case_context_relative_rmse"] <= 1.0
            ),
            "endpoint_minimum_cosine_similarity": (
                metrics["minimum_case_cosine_similarity"] >= 0.0
            ),
            "context_minimum_cosine_similarity": (
                metrics["minimum_case_context_cosine_similarity"] >= 0.0
            ),
        }
        exchange = _context_exchange_constraints(metrics, prior_metrics)
        exchange_checks = {
            name: bool(receipt["passed"]) for name, receipt in exchange.items()
        }
        passed = all(absolute_checks.values()) and all(exchange_checks.values())
        key = (
            metrics["maximum_case_relative_rmse"],
            -metrics["minimum_case_cosine_similarity"],
            metrics["mean_case_relative_rmse"],
            -metrics["mean_case_cosine_similarity"],
            metrics["maximum_case_context_relative_rmse"],
            -metrics["minimum_case_context_cosine_similarity"],
            metrics["mean_case_context_relative_rmse"],
            -metrics["mean_case_context_cosine_similarity"],
            value["candidate_id"],
        )
        record = {
            "candidate_id": value["candidate_id"],
            "candidate_state_sha256": value["candidate_state_sha256"],
            "v4_selection_report_sha256": value["report_sha256"],
            "absolute_gate_checks": absolute_checks,
            "prior_relative_context_exchange": exchange,
            "passed": passed,
            "failed_checks": [
                name
                for name, check in {
                    **absolute_checks,
                    **{
                        f"context_exchange_{name}": check
                        for name, check in exchange_checks.items()
                    },
                }.items()
                if not check
            ],
            "lexicographic_key": [
                float(item) if not isinstance(item, str) else item for item in key
            ],
        }
        inventory.append(record)
        if passed:
            feasible.append((key, record))
    if not feasible:
        raise Lfm2CpcfShortConvV5Error(
            "V5 selection rejected the frontier because zero candidates are feasible."
        )
    feasible.sort(key=lambda item: item[0])
    ranking = [
        {
            "rank": index + 1,
            "candidate_id": record["candidate_id"],
            "candidate_state_sha256": record["candidate_state_sha256"],
            "v4_selection_report_sha256": record["v4_selection_report_sha256"],
            "lexicographic_key": record["lexicographic_key"],
        }
        for index, (_key, record) in enumerate(feasible)
    ]
    selected = ranking[0]
    body = {
        "schema": LFM2_CPCF_SHORT_CONV_V5_SCHEMA,
        "kind": "sealed_v4_endpoint_first_selection_rescore",
        "policy_sha256": selection_policy.authentication_sha256,
        "prior_control_candidate_id": prior["candidate_id"],
        "prior_control_v4_selection_report_sha256": prior["report_sha256"],
        "prior_control_metrics": dict(prior_metrics),
        "candidate_count": len(normalized),
        "feasible_candidate_count": len(feasible),
        "rejected_candidate_count": len(normalized) - len(feasible),
        "candidate_inventory": inventory,
        "ranking": ranking,
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate_state_sha256": selected["candidate_state_sha256"],
        "identical_candidate_evidence_contract": True,
        "evidence_sha256": evidence_contract[2],
        "endpoint_metrics_ranked_before_context_tiebreakers": True,
        "minimum_gate_slack_ranked": False,
        "selection_reports_consumed": len(normalized),
        "fit_reports_consumed": 0,
        "validation_reports_consumed": 0,
        "validation_accessible": False,
        "execution": dict(_ZERO_TRAINING_EXECUTION),
    }
    return _seal_report(body)


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5RidgeCase:
    """One authenticated fit trajectory for an affine prior-residual solve."""

    case_id: str
    split: str
    design: Any
    target: Any
    valid_rows: Any | None = None
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        name = str(self.case_id).strip()
        if not name:
            raise Lfm2CpcfShortConvV5Error("case_id must be nonempty text.")
        if str(self.split).strip().lower() != "fit":
            raise Lfm2CpcfShortConvV5Error(
                "V5 coefficient solves accept fit evidence only."
            )
        design = _snapshot_tensor(self.design, f"{name}.design", dimensions=2)
        target = _snapshot_tensor(self.target, f"{name}.target", dimensions=(1, 2))
        if target.ndim == 1:
            target = target.unsqueeze(-1).contiguous()
        if int(design.shape[0]) != int(target.shape[0]):
            raise Lfm2CpcfShortConvV5Error(
                f"{name} design and target row counts differ."
            )
        mask = _snapshot_mask(self.valid_rows, f"{name}.valid_rows")
        if mask is not None and tuple(mask.shape) != (int(design.shape[0]),):
            raise Lfm2CpcfShortConvV5Error(f"{name}.valid_rows has the wrong shape.")
        object.__setattr__(self, "case_id", name)
        object.__setattr__(self, "split", "fit")
        object.__setattr__(self, "design", design)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "valid_rows", mask)
        expected = _sha256(self.authentication_body())
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2CpcfShortConvV5Error(
                "ridge case authentication_sha256 is incorrect."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        tensors = {
            "design": _tensor_sha256(self.design, f"{self.case_id}.design"),
            "target": _tensor_sha256(self.target, f"{self.case_id}.target"),
        }
        if self.valid_rows is not None:
            tensors["valid_rows"] = _tensor_sha256(
                self.valid_rows,
                f"{self.case_id}.valid_rows",
            )
        return {
            "schema": LFM2_CPCF_SHORT_CONV_V5_SCHEMA,
            "kind": "equilibrated_prior_residual_ridge_case",
            "case_id": self.case_id,
            "split": "fit",
            "tensors": tensors,
        }

    def verify_authentication(self) -> bool:
        return self.authentication_sha256 == _sha256(self.authentication_body())


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5RidgePolicy:
    """Authenticated conditioning and same-unit trust policy."""

    condition_caps: tuple[float, ...] = V5_CONDITION_CAP_CANDIDATES
    rank_relative_tolerance: float = 1e-12
    parameter_trust_radius: float = 4.0
    effect_trust_radius: float = 1.0
    parameter_reference_floor: float = 1e-6
    effect_reference_floor: float = 1e-6
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        caps = tuple(float(value) for value in self.condition_caps)
        if caps != V5_CONDITION_CAP_CANDIDATES:
            raise Lfm2CpcfShortConvV5Error(
                "V5 condition-cap candidates are frozen at (1e3, 1e4)."
            )
        tolerance = _positive_float(
            self.rank_relative_tolerance,
            "rank_relative_tolerance",
        )
        parameter_radius = _positive_float(
            self.parameter_trust_radius,
            "parameter_trust_radius",
        )
        effect_radius = _positive_float(
            self.effect_trust_radius,
            "effect_trust_radius",
        )
        parameter_floor = _positive_float(
            self.parameter_reference_floor,
            "parameter_reference_floor",
        )
        effect_floor = _positive_float(
            self.effect_reference_floor,
            "effect_reference_floor",
        )
        object.__setattr__(self, "condition_caps", caps)
        object.__setattr__(self, "rank_relative_tolerance", tolerance)
        object.__setattr__(self, "parameter_trust_radius", parameter_radius)
        object.__setattr__(self, "effect_trust_radius", effect_radius)
        object.__setattr__(self, "parameter_reference_floor", parameter_floor)
        object.__setattr__(self, "effect_reference_floor", effect_floor)
        expected = _sha256(self.authentication_body())
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2CpcfShortConvV5Error(
                "ridge policy authentication_sha256 is incorrect."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CPCF_SHORT_CONV_V5_SCHEMA,
            "kind": "equilibrated_prior_residual_ridge_policy",
            "condition_caps": list(self.condition_caps),
            "rank_relative_tolerance": self.rank_relative_tolerance,
            "parameter_trust_radius": self.parameter_trust_radius,
            "effect_trust_radius": self.effect_trust_radius,
            "parameter_reference_floor": self.parameter_reference_floor,
            "effect_reference_floor": self.effect_reference_floor,
            "fit_intercept_modes": ["separate_unpenalized", "absent_frozen_zero"],
            "case_balanced_centering": "separate_unpenalized_intercept_only",
            "no_intercept_uses_uncentered_system": True,
            "column_equilibration": True,
            "nullspace_preserves_prior": True,
            "execution": {
                **_ZERO_TRAINING_EXECUTION,
                "closed_form_only": True,
            },
        }

    def verify_authentication(self) -> bool:
        return self.authentication_sha256 == _sha256(self.authentication_body())

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.authentication_body(),
            "authentication_sha256": self.authentication_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5RidgeResult:
    """One condition-cap candidate and its fully bound numerical receipt."""

    weight: Any
    bias: Any | None
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "weight",
            _snapshot_tensor(self.weight, "ridge_result.weight", dimensions=2),
        )
        if self.bias is not None:
            object.__setattr__(
                self,
                "bias",
                _snapshot_tensor(self.bias, "ridge_result.bias", dimensions=1),
            )
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType(dict(self.diagnostics)),
        )

    def verify_authentication(self) -> bool:
        return (
            verify_lfm2_cpcf_short_conv_v5_report(self.diagnostics)
            and self.diagnostics.get("solution_weight_sha256")
            == _tensor_sha256(self.weight, "ridge_result.weight")
            and (
                (
                    self.bias is None
                    and self.diagnostics.get("solution_bias_sha256") is None
                    and self.diagnostics.get("bias_contract") == "absent_frozen_zero"
                    and self.diagnostics.get("bias_effect_is_frozen_zero") is True
                )
                or (
                    self.bias is not None
                    and self.diagnostics.get("solution_bias_sha256")
                    == _tensor_sha256(self.bias, "ridge_result.bias")
                    and self.diagnostics.get("bias_contract")
                    == "separate_unpenalized_intercept"
                )
            )
        )


def _canonical_ridge_rows(
    cases: Sequence[Lfm2CpcfV5RidgeCase],
) -> tuple[tuple[str, str, Any, Any], ...]:
    values = tuple(cases)
    if not values:
        raise Lfm2CpcfShortConvV5Error(
            "Equilibrated ridge requires at least one fit case."
        )
    if any(not isinstance(case, Lfm2CpcfV5RidgeCase) for case in values):
        raise Lfm2CpcfShortConvV5Error(
            "All ridge cases must be Lfm2CpcfV5RidgeCase values."
        )
    ordered = tuple(sorted(values, key=lambda case: case.case_id))
    if len({case.case_id for case in ordered}) != len(ordered):
        raise Lfm2CpcfShortConvV5Error("Ridge case ids must be unique.")
    rows = []
    width = None
    outputs = None
    for case in ordered:
        if not case.verify_authentication():
            raise Lfm2CpcfShortConvV5Error(
                f"Ridge case {case.case_id!r} changed after authentication."
            )
        design = case.design
        target = case.target
        if case.valid_rows is not None:
            design = design[case.valid_rows]
            target = target[case.valid_rows]
        if int(design.shape[0]) == 0:
            raise Lfm2CpcfShortConvV5Error(
                f"Ridge case {case.case_id!r} has no valid rows."
            )
        width = int(design.shape[1]) if width is None else width
        outputs = int(target.shape[1]) if outputs is None else outputs
        if int(design.shape[1]) != width or int(target.shape[1]) != outputs:
            raise Lfm2CpcfShortConvV5Error(
                "Ridge cases have inconsistent feature or output widths."
            )
        rows.append((case.case_id, case.authentication_sha256, design, target))
    return tuple(rows)


def _trust_scale(limit: float, observed: float) -> float:
    if observed == 0.0:
        return 1.0
    return min(1.0, limit / observed)


@dataclass(frozen=True, slots=True)
class _EquilibratedRidgeSystem:
    """Condition-cap-independent sufficient statistics for one ridge frontier."""

    rows: tuple[tuple[str, str, Any, Any], ...]
    case_count: int
    feature_width: int
    output_width: int
    design_mean: Any | None
    residual_mean: Any | None
    case_receipts: tuple[dict[str, Any], ...]
    solve_design: Any
    solve_residual: Any
    column_rms: Any
    active_columns: Any
    equilibration_scale: Any
    equilibrated_design: Any
    gram: Any
    right: Any
    eigenvalues: Any
    eigenvectors: Any
    maximum_eigenvalue: float
    minimum_eigenvalue: float
    identifiable: Any
    eigen_right: Any


def _prepare_equilibrated_ridge_system(
    *,
    rows: tuple[tuple[str, str, Any, Any], ...],
    prior_weight: Any,
    prior_bias: Any | None,
    fit_intercept: bool,
    policy: Lfm2CpcfV5RidgePolicy,
) -> _EquilibratedRidgeSystem:
    """Build the exact shared design, Gram, RHS, and eigensystem once."""

    torch = _torch()
    case_count = len(rows)
    feature_width = int(rows[0][2].shape[1])
    output_width = int(rows[0][3].shape[1])

    residual_rows = []
    for _case_id, _case_sha, design, target in rows:
        prior_effect = design @ prior_weight
        if fit_intercept:
            assert prior_bias is not None
            prior_effect = prior_effect + prior_bias
        residual_rows.append(target - prior_effect)
    design_mean = None
    residual_mean = None
    if fit_intercept:
        design_mean = (
            sum(
                (design.mean(dim=0) for _case_id, _case_sha, design, _target in rows),
                torch.zeros(feature_width, dtype=torch.float64),
            )
            / case_count
        )
        residual_mean = (
            sum(
                (residual.mean(dim=0) for residual in residual_rows),
                torch.zeros(output_width, dtype=torch.float64),
            )
            / case_count
        )

    balanced_design = []
    balanced_residual = []
    case_receipts = []
    for (case_id, case_sha, design, target), residual in zip(rows, residual_rows):
        row_count = int(design.shape[0])
        case_scale = 1.0 / math.sqrt(case_count * row_count)
        if fit_intercept:
            assert design_mean is not None and residual_mean is not None
            solve_design = design - design_mean
            solve_residual = residual - residual_mean
        else:
            solve_design = design
            solve_residual = residual
        balanced_design.append(solve_design * case_scale)
        balanced_residual.append(solve_residual * case_scale)
        case_receipts.append(
            {
                "case_id": case_id,
                "case_authentication_sha256": case_sha,
                "valid_rows": row_count,
                "case_balance_scale": case_scale,
                "design_sha256": _tensor_sha256(design, f"{case_id}.design"),
                "target_sha256": _tensor_sha256(target, f"{case_id}.target"),
                "prior_residual_sha256": _tensor_sha256(
                    residual,
                    f"{case_id}.prior_residual",
                ),
            }
        )
    solve_design = torch.cat(balanced_design, dim=0)
    solve_residual = torch.cat(balanced_residual, dim=0)
    column_rms = solve_design.square().sum(dim=0).sqrt()
    maximum_column_rms = float(column_rms.max().item())
    column_threshold = (
        maximum_column_rms * policy.rank_relative_tolerance
        if maximum_column_rms > 0.0
        else policy.rank_relative_tolerance
    )
    active_columns = column_rms > column_threshold
    equilibration_scale = torch.where(
        active_columns,
        column_rms,
        torch.ones_like(column_rms),
    )
    equilibrated_design = solve_design / equilibration_scale
    gram = equilibrated_design.T @ equilibrated_design
    gram = 0.5 * (gram + gram.T)
    right = equilibrated_design.T @ solve_residual
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.clamp(min=0.0)
    maximum_eigenvalue = float(eigenvalues.max().item())
    minimum_eigenvalue = float(eigenvalues.min().item())
    rank_threshold = maximum_eigenvalue * policy.rank_relative_tolerance
    identifiable = eigenvalues > rank_threshold
    eigen_right = eigenvectors.T @ right
    return _EquilibratedRidgeSystem(
        rows=rows,
        case_count=case_count,
        feature_width=feature_width,
        output_width=output_width,
        design_mean=design_mean,
        residual_mean=residual_mean,
        case_receipts=tuple(case_receipts),
        solve_design=solve_design,
        solve_residual=solve_residual,
        column_rms=column_rms,
        active_columns=active_columns,
        equilibration_scale=equilibration_scale,
        equilibrated_design=equilibrated_design,
        gram=gram,
        right=right,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        maximum_eigenvalue=maximum_eigenvalue,
        minimum_eigenvalue=minimum_eigenvalue,
        identifiable=identifiable,
        eigen_right=eigen_right,
    )


def _solve_equilibrated_candidate(
    *,
    rows: tuple[tuple[str, str, Any, Any], ...],
    prior_weight: Any,
    prior_bias: Any | None,
    fit_intercept: bool,
    condition_cap: float,
    policy: Lfm2CpcfV5RidgePolicy,
    shared_system: _EquilibratedRidgeSystem | None = None,
) -> Lfm2CpcfV5RidgeResult:
    torch = _torch()
    system = shared_system or _prepare_equilibrated_ridge_system(
        rows=rows,
        prior_weight=prior_weight,
        prior_bias=prior_bias,
        fit_intercept=fit_intercept,
        policy=policy,
    )
    case_count = system.case_count
    feature_width = system.feature_width
    output_width = system.output_width
    design_mean = system.design_mean
    residual_mean = system.residual_mean
    case_receipts = [dict(receipt) for receipt in system.case_receipts]
    solve_design = system.solve_design
    solve_residual = system.solve_residual
    column_rms = system.column_rms
    active_columns = system.active_columns
    equilibration_scale = system.equilibration_scale
    equilibrated_design = system.equilibrated_design
    gram = system.gram
    right = system.right
    eigenvalues = system.eigenvalues
    eigenvectors = system.eigenvectors
    maximum_eigenvalue = system.maximum_eigenvalue
    minimum_eigenvalue = system.minimum_eigenvalue
    if maximum_eigenvalue == 0.0:
        ridge = 0.0
    else:
        ridge = max(
            0.0,
            (maximum_eigenvalue - condition_cap * minimum_eigenvalue)
            / (condition_cap - 1.0),
        )
    identifiable = system.identifiable
    inverse = torch.zeros_like(eigenvalues)
    inverse[identifiable] = 1.0 / (eigenvalues[identifiable] + ridge)
    eigen_right = system.eigen_right
    raw_eigen_delta = inverse.unsqueeze(-1) * eigen_right
    raw_equilibrated_delta = eigenvectors @ raw_eigen_delta
    raw_weight_delta = raw_equilibrated_delta / equilibration_scale.unsqueeze(-1)
    raw_weight_delta[~active_columns, :] = 0.0
    raw_bias_delta = None
    if fit_intercept:
        assert design_mean is not None and residual_mean is not None
        raw_bias_delta = residual_mean - design_mean @ raw_weight_delta

    raw_parameter_parts = [raw_weight_delta.reshape(-1)]
    prior_parameter_parts = [prior_weight.reshape(-1)]
    if fit_intercept:
        assert raw_bias_delta is not None and prior_bias is not None
        raw_parameter_parts.append(raw_bias_delta.reshape(-1))
        prior_parameter_parts.append(prior_bias.reshape(-1))
    raw_parameter_delta = torch.cat(raw_parameter_parts)
    prior_parameters = torch.cat(prior_parameter_parts)
    raw_parameter_correction_rms = _rms(raw_parameter_delta)
    parameter_reference_rms = max(
        _rms(prior_parameters),
        policy.parameter_reference_floor,
    )
    parameter_limit = policy.parameter_trust_radius * parameter_reference_rms
    parameter_scale = _trust_scale(
        parameter_limit,
        raw_parameter_correction_rms,
    )

    raw_effects = []
    prior_effects = []
    targets = []
    for _case_id, _case_sha, design, target in rows:
        case_scale = 1.0 / math.sqrt(case_count * int(design.shape[0]))
        raw_effect = design @ raw_weight_delta
        prior_effect = design @ prior_weight
        if fit_intercept:
            assert raw_bias_delta is not None and prior_bias is not None
            raw_effect = raw_effect + raw_bias_delta
            prior_effect = prior_effect + prior_bias
        raw_effects.append(raw_effect * case_scale)
        prior_effects.append(prior_effect * case_scale)
        targets.append(target * case_scale)
    raw_effect = torch.cat(raw_effects, dim=0)
    prior_effect = torch.cat(prior_effects, dim=0)
    balanced_target = torch.cat(targets, dim=0)
    raw_effect_correction_rms = _rms(raw_effect)
    effect_reference_rms = max(
        _rms(prior_effect),
        _rms(balanced_target),
        policy.effect_reference_floor,
    )
    effect_limit = policy.effect_trust_radius * effect_reference_rms
    effect_scale = _trust_scale(effect_limit, raw_effect_correction_rms)
    accepted_scale = min(1.0, parameter_scale, effect_scale)

    accepted_weight_delta = raw_weight_delta * accepted_scale
    accepted_bias_delta = (
        None if raw_bias_delta is None else raw_bias_delta * accepted_scale
    )
    solution_weight = (prior_weight + accepted_weight_delta).contiguous()
    solution_bias = None
    if fit_intercept:
        assert prior_bias is not None and accepted_bias_delta is not None
        solution_bias = (prior_bias + accepted_bias_delta).contiguous()
    accepted_effect = raw_effect * accepted_scale
    accepted_parameter_rms = raw_parameter_correction_rms * accepted_scale
    accepted_effect_rms = raw_effect_correction_rms * accepted_scale

    if maximum_eigenvalue == 0.0:
        regularized_condition = 0.0
    else:
        regularized_condition = (maximum_eigenvalue + ridge) / (
            minimum_eigenvalue + ridge
        )
    accepted_eigen_delta = raw_eigen_delta * accepted_scale
    null_coefficients = accepted_eigen_delta[~identifiable]
    body = {
        "schema": LFM2_CPCF_SHORT_CONV_V5_SCHEMA,
        "kind": "condition_capped_equilibrated_prior_residual_ridge",
        "policy_sha256": policy.authentication_sha256,
        "required_split": "fit",
        "fit_only": True,
        "selection_cases_consumed": 0,
        "validation_cases_consumed": 0,
        "case_balanced": True,
        "row_duplication_invariant_objective": True,
        "case_count": case_count,
        "case_receipts": case_receipts,
        "feature_width": feature_width,
        "output_width": output_width,
        "fit_intercept": fit_intercept,
        "bias_contract": (
            "separate_unpenalized_intercept" if fit_intercept else "absent_frozen_zero"
        ),
        "separate_unpenalized_intercept": fit_intercept,
        "bias_value_is_absent": not fit_intercept,
        "bias_effect_is_frozen_zero": not fit_intercept,
        "centering_applied": fit_intercept,
        "no_intercept_uses_uncentered_system": not fit_intercept,
        "design_mean_sha256": (
            None
            if design_mean is None
            else _tensor_sha256(
                design_mean,
                "ridge.design_mean",
            )
        ),
        "prior_residual_mean_sha256": (
            None
            if residual_mean is None
            else _tensor_sha256(
                residual_mean,
                "ridge.prior_residual_mean",
            )
        ),
        "column_equilibrated": True,
        "column_rms": [float(value) for value in column_rms.tolist()],
        "active_column_indices": [
            index for index, value in enumerate(active_columns.tolist()) if value
        ],
        "inactive_column_indices": [
            index for index, value in enumerate(active_columns.tolist()) if not value
        ],
        "equilibration_scale_sha256": _tensor_sha256(
            equilibration_scale,
            "ridge.equilibration_scale",
        ),
        "centered_case_balanced_design_sha256": (
            _tensor_sha256(
                solve_design,
                "ridge.centered_case_balanced_design",
            )
            if fit_intercept
            else None
        ),
        "centered_case_balanced_prior_residual_sha256": (
            _tensor_sha256(
                solve_residual,
                "ridge.centered_case_balanced_prior_residual",
            )
            if fit_intercept
            else None
        ),
        "uncentered_case_balanced_design_sha256": (
            None
            if fit_intercept
            else _tensor_sha256(
                solve_design,
                "ridge.uncentered_case_balanced_design",
            )
        ),
        "uncentered_case_balanced_prior_residual_sha256": (
            None
            if fit_intercept
            else _tensor_sha256(
                solve_residual,
                "ridge.uncentered_case_balanced_prior_residual",
            )
        ),
        "equilibrated_design_sha256": _tensor_sha256(
            equilibrated_design,
            "ridge.equilibrated_design",
        ),
        "equilibrated_gram_sha256": _tensor_sha256(
            gram,
            "ridge.equilibrated_gram",
        ),
        "equilibrated_right_hand_side_sha256": _tensor_sha256(
            right,
            "ridge.equilibrated_right_hand_side",
        ),
        "symmetric_factorization": "torch.linalg.eigh",
        "eigenvalues": [float(value) for value in eigenvalues.tolist()],
        "eigenvalues_sha256": _tensor_sha256(
            eigenvalues,
            "ridge.eigenvalues",
        ),
        "eigenvectors_sha256": _tensor_sha256(
            eigenvectors,
            "ridge.eigenvectors",
        ),
        "effective_rank": int(identifiable.sum().item()),
        "nullspace_dimension": int((~identifiable).sum().item()),
        "nullspace_update_maximum_absolute": (
            0.0
            if int(null_coefficients.numel()) == 0
            else float(null_coefficients.abs().max().item())
        ),
        "nullspace_preserves_prior": (
            int(null_coefficients.numel()) == 0
            or float(null_coefficients.abs().max().item())
            <= 64.0 * torch.finfo(torch.float64).eps
        ),
        "condition_cap": condition_cap,
        "effective_ridge": ridge,
        "regularized_condition": regularized_condition,
        "condition_cap_passed": (
            regularized_condition <= condition_cap * (1.0 + 1e-12)
        ),
        "prior_weight_sha256": _tensor_sha256(
            prior_weight,
            "ridge.prior_weight",
        ),
        "prior_bias_sha256": (
            None
            if prior_bias is None
            else _tensor_sha256(prior_bias, "ridge.prior_bias")
        ),
        "raw_weight_delta_sha256": _tensor_sha256(
            raw_weight_delta,
            "ridge.raw_weight_delta",
        ),
        "raw_bias_delta_sha256": (
            None
            if raw_bias_delta is None
            else _tensor_sha256(
                raw_bias_delta,
                "ridge.raw_bias_delta",
            )
        ),
        "raw_eigen_delta_sha256": _tensor_sha256(
            raw_eigen_delta,
            "ridge.raw_eigen_delta",
        ),
        "accepted_eigen_delta_sha256": _tensor_sha256(
            accepted_eigen_delta,
            "ridge.accepted_eigen_delta",
        ),
        "solution_weight_sha256": _tensor_sha256(
            solution_weight,
            "ridge_result.weight",
        ),
        "solution_bias_sha256": (
            None
            if solution_bias is None
            else _tensor_sha256(
                solution_bias,
                "ridge_result.bias",
            )
        ),
        "parameter_space_trust": {
            "units": "coefficient_parameter_units",
            "reference": "transported_prior_parameters",
            "reference_rms": parameter_reference_rms,
            "relative_radius": policy.parameter_trust_radius,
            "absolute_limit_rms": parameter_limit,
            "raw_correction_rms": raw_parameter_correction_rms,
            "individual_scale": parameter_scale,
            "accepted_correction_rms": accepted_parameter_rms,
            "passed": accepted_parameter_rms <= parameter_limit * (1.0 + 1e-12),
        },
        "function_space_trust": {
            "units": "predicted_effect_units",
            "reference": "max(prior_effect,target_effect)",
            "reference_rms": effect_reference_rms,
            "relative_radius": policy.effect_trust_radius,
            "absolute_limit_rms": effect_limit,
            "raw_correction_rms": raw_effect_correction_rms,
            "individual_scale": effect_scale,
            "accepted_correction_rms": accepted_effect_rms,
            "accepted_effect_sha256": _tensor_sha256(
                accepted_effect,
                "ridge.accepted_effect",
            ),
            "passed": accepted_effect_rms <= effect_limit * (1.0 + 1e-12),
        },
        "joint_trust_scale": accepted_scale,
        "joint_trust_is_intersection": True,
        "parameter_trust_reference_uses_function_units": False,
        "function_trust_reference_uses_parameter_units": False,
        "execution": {
            **_ZERO_TRAINING_EXECUTION,
            "closed_form_symmetric_eigendecomposition": True,
        },
    }
    if not body["nullspace_preserves_prior"]:
        raise Lfm2CpcfShortConvV5Error(
            "Equilibrated ridge changed a numerically null prior direction."
        )
    if not body["condition_cap_passed"]:
        raise Lfm2CpcfShortConvV5Error(
            "Equilibrated ridge failed its declared condition cap."
        )
    return Lfm2CpcfV5RidgeResult(
        weight=solution_weight,
        bias=solution_bias,
        diagnostics=_seal_report(body),
    )


def condition_capped_equilibrated_prior_residual_ridge_frontier(
    cases: Sequence[Lfm2CpcfV5RidgeCase],
    *,
    prior_weight: Any,
    prior_bias: Any | None,
    fit_intercept: bool = True,
    policy: Lfm2CpcfV5RidgePolicy | None = None,
) -> Mapping[str, Lfm2CpcfV5RidgeResult]:
    """Return the frozen ``kappa=1e3`` and ``kappa=1e4`` fit candidates.

    ``fit_intercept=False`` is the physical-parameter mode for FIR and B/C/x
    designs that already encode every installable bias coordinate.  It solves
    the uncentered system and returns ``bias=None`` under a sealed
    absent/frozen-zero intercept contract.
    """

    ridge_policy = policy or Lfm2CpcfV5RidgePolicy()
    if not isinstance(ridge_policy, Lfm2CpcfV5RidgePolicy):
        raise TypeError("policy must be Lfm2CpcfV5RidgePolicy.")
    if not ridge_policy.verify_authentication():
        raise Lfm2CpcfShortConvV5Error("Ridge policy failed authentication.")
    if type(fit_intercept) is not bool:
        raise TypeError("fit_intercept must be boolean.")
    rows = _canonical_ridge_rows(cases)
    feature_width = int(rows[0][2].shape[1])
    output_width = int(rows[0][3].shape[1])
    prior_weight_value = _snapshot_tensor(
        prior_weight,
        "prior_weight",
        dimensions=2,
    )
    if tuple(prior_weight_value.shape) != (feature_width, output_width):
        raise Lfm2CpcfShortConvV5Error(
            "prior_weight must have shape [features, outputs]."
        )
    prior_bias_value = None
    if fit_intercept:
        if prior_bias is None:
            raise Lfm2CpcfShortConvV5Error(
                "fit_intercept=True requires prior_bias with shape [outputs]."
            )
        prior_bias_value = _snapshot_tensor(
            prior_bias,
            "prior_bias",
            dimensions=1,
        )
        if tuple(prior_bias_value.shape) != (output_width,):
            raise Lfm2CpcfShortConvV5Error("prior_bias must have shape [outputs].")
    elif prior_bias is not None:
        raise Lfm2CpcfShortConvV5Error(
            "fit_intercept=False requires prior_bias=None so the separate "
            "intercept is explicitly absent and frozen at zero."
        )
    shared_system = _prepare_equilibrated_ridge_system(
        rows=rows,
        prior_weight=prior_weight_value,
        prior_bias=prior_bias_value,
        fit_intercept=fit_intercept,
        policy=ridge_policy,
    )
    results = {
        f"kappa-{int(cap):05d}": _solve_equilibrated_candidate(
            rows=rows,
            prior_weight=prior_weight_value,
            prior_bias=prior_bias_value,
            fit_intercept=fit_intercept,
            condition_cap=cap,
            policy=ridge_policy,
            shared_system=shared_system,
        )
        for cap in ridge_policy.condition_caps
    }
    return MappingProxyType(results)


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5FirBasisCase:
    """Fit-only activation and endpoint descriptors over all source channels."""

    case_id: str
    split: str
    activation_descriptors: Any
    endpoint_descriptors: Any
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        name = str(self.case_id).strip()
        if not name:
            raise Lfm2CpcfShortConvV5Error("case_id must be nonempty text.")
        if str(self.split).strip().lower() != "fit":
            raise Lfm2CpcfShortConvV5Error(
                "FIR-commuting basis construction accepts fit evidence only."
            )
        activation = _snapshot_tensor(
            self.activation_descriptors,
            f"{name}.activation_descriptors",
            dimensions=2,
        )
        endpoint = _snapshot_tensor(
            self.endpoint_descriptors,
            f"{name}.endpoint_descriptors",
            dimensions=2,
        )
        if int(activation.shape[0]) != int(endpoint.shape[0]):
            raise Lfm2CpcfShortConvV5Error(
                f"{name} descriptor source-channel counts differ."
            )
        object.__setattr__(self, "case_id", name)
        object.__setattr__(self, "split", "fit")
        object.__setattr__(self, "activation_descriptors", activation)
        object.__setattr__(self, "endpoint_descriptors", endpoint)
        expected = _sha256(self.authentication_body())
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2CpcfShortConvV5Error(
                "FIR basis case authentication_sha256 is incorrect."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CPCF_SHORT_CONV_V5_SCHEMA,
            "kind": "fir_commuting_basis_case",
            "case_id": self.case_id,
            "split": "fit",
            "tensors": {
                "activation_descriptors": _tensor_sha256(
                    self.activation_descriptors,
                    f"{self.case_id}.activation_descriptors",
                ),
                "endpoint_descriptors": _tensor_sha256(
                    self.endpoint_descriptors,
                    f"{self.case_id}.endpoint_descriptors",
                ),
            },
        }

    def verify_authentication(self) -> bool:
        return self.authentication_sha256 == _sha256(self.authentication_body())


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5FirBasisPolicy:
    """Descriptor weighting and representability gate for the grouped basis."""

    fir_descriptor_weight: float = 1.0
    activation_descriptor_weight: float = 1.0
    endpoint_descriptor_weight: float = 1.0
    energy_floor_relative: float = 1e-12
    maximum_projected_fir_offdiagonal_relative_frobenius: float = 1e-12
    maximum_per_lag_commutator_relative_frobenius: float = 1e-12
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "fir_descriptor_weight",
            "activation_descriptor_weight",
            "endpoint_descriptor_weight",
            "energy_floor_relative",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive_float(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "maximum_projected_fir_offdiagonal_relative_frobenius",
            _positive_float(
                self.maximum_projected_fir_offdiagonal_relative_frobenius,
                "maximum_projected_fir_offdiagonal_relative_frobenius",
                allow_zero=True,
            ),
        )
        object.__setattr__(
            self,
            "maximum_per_lag_commutator_relative_frobenius",
            _positive_float(
                self.maximum_per_lag_commutator_relative_frobenius,
                "maximum_per_lag_commutator_relative_frobenius",
                allow_zero=True,
            ),
        )
        expected = _sha256(self.authentication_body())
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2CpcfShortConvV5Error(
                "FIR basis policy authentication_sha256 is incorrect."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CPCF_SHORT_CONV_V5_SCHEMA,
            "kind": "fir_commuting_balanced_basis_policy",
            "fir_descriptor_weight": self.fir_descriptor_weight,
            "activation_descriptor_weight": self.activation_descriptor_weight,
            "endpoint_descriptor_weight": self.endpoint_descriptor_weight,
            "energy_floor_relative": self.energy_floor_relative,
            "maximum_projected_fir_offdiagonal_relative_frobenius": (
                self.maximum_projected_fir_offdiagonal_relative_frobenius
            ),
            "maximum_per_lag_commutator_relative_frobenius": (
                self.maximum_per_lag_commutator_relative_frobenius
            ),
            "commuting_claim_requires_intertwining_residual_gate": True,
            "projected_offdiagonal_is_not_sufficient_for_commuting_claim": True,
            "seed_method": "deterministic_farthest_first",
            "assignment_method": "capacity_balanced_farthest_first",
            "energy_weighting": "sqrt_group_energy_share",
            "exact_dual": True,
            "execution": {
                **_ZERO_TRAINING_EXECUTION,
                "closed_form_only": True,
            },
        }

    def verify_authentication(self) -> bool:
        return self.authentication_sha256 == _sha256(self.authentication_body())


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5FirCommutingBasis:
    """A disjoint-support all-channel encoder and its exact dual."""

    encoder: Any
    decoder: Any
    target_fir_kernel: Any
    source_to_target_group: tuple[int, ...]
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "encoder",
            _snapshot_tensor(self.encoder, "basis.encoder", dimensions=2),
        )
        object.__setattr__(
            self,
            "decoder",
            _snapshot_tensor(self.decoder, "basis.decoder", dimensions=2),
        )
        object.__setattr__(
            self,
            "target_fir_kernel",
            _snapshot_tensor(
                self.target_fir_kernel,
                "basis.target_fir_kernel",
                dimensions=2,
            ),
        )
        object.__setattr__(
            self,
            "source_to_target_group",
            tuple(int(value) for value in self.source_to_target_group),
        )
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    def verify_authentication(self) -> bool:
        state = {
            "decoder": self.decoder,
            "encoder": self.encoder,
            "target_fir_kernel": self.target_fir_kernel,
        }
        return (
            verify_lfm2_cpcf_short_conv_v5_report(self.report)
            and self.report.get("state_sha256")
            == _state_sha256(state, "fir_commuting_basis")
            and self.report.get("source_to_target_group")
            == list(self.source_to_target_group)
        )


def _equilibrate_descriptor_block(value: Any) -> Any:
    torch = _torch()
    centered = value - value.mean(dim=0, keepdim=True)
    scale = centered.square().mean(dim=0).sqrt()
    maximum = float(scale.max().item())
    threshold = max(
        maximum * 1e-12,
        torch.finfo(torch.float64).eps,
    )
    scale = torch.where(scale > threshold, scale, torch.ones_like(scale))
    return centered / scale


def _squared_distance(left: Any, right: Any) -> float:
    return float((left - right).square().sum().item())


def _direct_seed_squared_distances(
    descriptors: Any,
    *,
    seed_index: int,
) -> Any:
    """Compute one bounded FP64 direct-distance column.

    The kernel keeps the direct ``(candidate - seed).square().sum()`` objective
    instead of the cancellation-prone Gram identity.  A fixed row chunk exposes
    enough independent work to Torch without materializing the complete
    source-by-descriptor temporary.  Batched row reductions may differ from the
    historical scalar-row reduction only by FP64 association.
    """

    torch = _torch()
    source_channels = int(descriptors.shape[0])
    seed = descriptors[seed_index]
    distances = torch.empty(source_channels, dtype=torch.float64)
    row_chunk_size = min(
        source_channels,
        V5_DIRECT_SEED_DISTANCE_ROW_CHUNK_SIZE,
    )
    differences = torch.empty(
        (row_chunk_size, int(descriptors.shape[1])),
        dtype=torch.float64,
    )
    for row_start in range(0, source_channels, row_chunk_size):
        row_stop = min(source_channels, row_start + row_chunk_size)
        row_count = row_stop - row_start
        chunk = differences[:row_count]
        torch.sub(
            descriptors[row_start:row_stop],
            seed.unsqueeze(0),
            out=chunk,
        )
        chunk.square_()
        distances[row_start:row_stop] = chunk.sum(dim=1)
    distances[seed_index] = 0.0
    return distances.contiguous()


def _seed_distance_table_complexity_receipt(
    *,
    source_channels: int,
    target_channels: int,
    descriptor_width: int,
    descriptor_block_count: int,
    capacity_assignment_distance_lookups: int,
) -> dict[str, Any]:
    """Describe the removed scalar-reduction work without timing claims."""

    seed_distance_evaluations = sum(
        (source_channels - seed_count) * seed_count
        for seed_count in range(1, target_channels)
    )
    remaining_count = source_channels - target_channels
    assignment_sort_distance_evaluations = remaining_count * target_channels
    legacy_evaluations = (
        seed_distance_evaluations
        + assignment_sort_distance_evaluations
        + capacity_assignment_distance_lookups
    )
    seed_table_entries = source_channels * target_channels
    row_chunk_size = min(
        source_channels,
        V5_DIRECT_SEED_DISTANCE_ROW_CHUNK_SIZE,
    )
    chunks_per_seed = math.ceil(source_channels / row_chunk_size)
    return {
        "kind": "cached_direct_selected_seed_squared_euclidean_table",
        "source_channels": source_channels,
        "target_channels": target_channels,
        "descriptor_width": descriptor_width,
        "descriptor_block_count": descriptor_block_count,
        "seed_distance_table_entries": seed_table_entries,
        "seed_distance_table_bytes": seed_table_entries * 8,
        "maximum_direct_seed_temporary_bytes": (row_chunk_size * descriptor_width * 8),
        "direct_seed_column_count": target_channels,
        "direct_seed_chunk_reduction_count": (target_channels * chunks_per_seed),
        "direct_seed_vector_reduction_count": seed_table_entries,
        "direct_seed_row_distance_count": seed_table_entries,
        "legacy_seed_vector_reduction_count": (seed_distance_evaluations),
        "legacy_assignment_sort_vector_reduction_count": (
            assignment_sort_distance_evaluations
        ),
        "legacy_capacity_assignment_vector_reduction_count": (
            capacity_assignment_distance_lookups
        ),
        "legacy_total_vector_reduction_count": legacy_evaluations,
        "legacy_descriptor_coordinate_reductions": (
            legacy_evaluations * descriptor_width
        ),
        "direct_seed_descriptor_coordinate_reductions": (
            seed_table_entries * descriptor_width
        ),
        "distance_metric": "direct_difference_squared_euclidean",
        "distance_kernel": (
            "differences=empty([min(source_rows,128),descriptor_width]);"
            "for row_chunk: sub(descriptors[row_chunk],descriptors[seed],"
            "out=differences[:row_count]);differences[:row_count].square_();"
            "distances[row_chunk]=differences[:row_count].sum(dim=1)"
        ),
        "distance_row_chunk_size": V5_DIRECT_SEED_DISTANCE_ROW_CHUNK_SIZE,
        "maximum_rows_per_distance_chunk": row_chunk_size,
        "distance_chunks_per_seed": chunks_per_seed,
        "direct_seed_temporary_reused_across_rows": True,
        "rowwise_direct_squared_euclidean_objective_preserved": True,
        "scalar_legacy_reduction_semantics_preserved": False,
        "finite_precision_reduction_association": ("fixed_128_row_batched_reduction"),
        "full_source_by_descriptor_temporary_materialized": False,
        "gram_identity_used": False,
        "negative_distance_clamping_used": False,
        "seed_tie_break": "lowest_source_index",
        "assignment_tie_break": "lowest_group_index",
        "folding_objective_changed": False,
    }


def _balanced_farthest_first_with_seed_distance_table(
    descriptors: Any,
    *,
    target_channels: int,
) -> tuple[list[int], list[list[int]], list[int], int, Any]:
    """Run legacy decisions while caching one direct column per selected seed."""

    torch = _torch()
    if (
        not isinstance(descriptors, torch.Tensor)
        or descriptors.device.type != "cpu"
        or descriptors.dtype != torch.float64
        or descriptors.ndim != 2
        or int(descriptors.shape[0]) <= 0
        or int(descriptors.shape[1]) <= 0
        or not bool(torch.isfinite(descriptors).all().item())
    ):
        raise Lfm2CpcfShortConvV5Error(
            "Farthest-first descriptors must be one finite nonempty CPU FP64 matrix."
        )
    source_channels = int(descriptors.shape[0])
    if (
        isinstance(target_channels, bool)
        or not isinstance(target_channels, int)
        or target_channels <= 0
        or target_channels > source_channels
    ):
        raise Lfm2CpcfShortConvV5Error(
            "target_channels must be in [1, source_channels]."
        )
    centroid = descriptors.mean(dim=0)
    first_seed = min(
        range(source_channels),
        key=lambda index: (
            -_squared_distance(descriptors[index], centroid),
            index,
        ),
    )
    seeds = [first_seed]
    eligible = torch.ones(source_channels, dtype=torch.bool)
    eligible[first_seed] = False
    seed_squared_distances = torch.empty(
        (source_channels, target_channels),
        dtype=torch.float64,
    )
    first_distances = _direct_seed_squared_distances(
        descriptors,
        seed_index=first_seed,
    )
    if not bool(torch.isfinite(first_distances).all().item()):
        raise Lfm2CpcfShortConvV5Error("Direct seed distances became nonfinite.")
    seed_squared_distances[:, 0].copy_(first_distances)
    minimum_seed_distance = first_distances.clone()
    while len(seeds) < target_channels:
        # torch.argmax returns the first maximum, preserving the legacy
        # lowest-source-index tie break without hundreds of thousands of
        # Python scalar extractions.
        seed_scores = torch.where(
            eligible,
            minimum_seed_distance,
            torch.full_like(minimum_seed_distance, -torch.inf),
        )
        next_seed = int(torch.argmax(seed_scores).item())
        seeds.append(next_seed)
        eligible[next_seed] = False
        next_distances = _direct_seed_squared_distances(
            descriptors,
            seed_index=next_seed,
        )
        if not bool(torch.isfinite(next_distances).all().item()):
            raise Lfm2CpcfShortConvV5Error("Direct seed distances became nonfinite.")
        seed_squared_distances[:, len(seeds) - 1].copy_(next_distances)
        minimum_seed_distance.copy_(
            torch.minimum(
                minimum_seed_distance,
                next_distances,
            )
        )

    minimum_group_size, larger_groups = divmod(
        source_channels,
        target_channels,
    )
    capacities = [
        minimum_group_size + int(group < larger_groups)
        for group in range(target_channels)
    ]
    groups = [[seed] for seed in seeds]
    assignment = [-1] * source_channels
    for group, seed in enumerate(seeds):
        assignment[seed] = group
    remaining = [index for index in range(source_channels) if assignment[index] < 0]
    remaining.sort(
        key=lambda index: (
            -float(minimum_seed_distance[index].item()),
            index,
        )
    )
    capacity_assignment_distance_lookups = 0
    for source in remaining:
        available = [
            group
            for group in range(target_channels)
            if len(groups[group]) < capacities[group]
        ]
        if not available:
            raise Lfm2CpcfShortConvV5Error(
                "Balanced FIR assignment exhausted all group capacities."
            )
        capacity_assignment_distance_lookups += len(available)
        group = min(
            available,
            key=lambda index: (
                float(
                    seed_squared_distances[
                        source,
                        index,
                    ].item()
                ),
                index,
            ),
        )
        groups[group].append(source)
        assignment[source] = group
    return (
        seeds,
        groups,
        assignment,
        capacity_assignment_distance_lookups,
        seed_squared_distances,
    )


def build_fir_commuting_balanced_all_channel_basis(
    cases: Sequence[Lfm2CpcfV5FirBasisCase],
    *,
    source_fir_kernel: Any,
    target_channels: int,
    policy: Lfm2CpcfV5FirBasisPolicy | None = None,
) -> Lfm2CpcfV5FirCommutingBasis:
    """Partition every source channel into a balanced FIR-commuting basis."""

    torch = _torch()
    basis_policy = policy or Lfm2CpcfV5FirBasisPolicy()
    if not isinstance(basis_policy, Lfm2CpcfV5FirBasisPolicy):
        raise TypeError("policy must be Lfm2CpcfV5FirBasisPolicy.")
    if not basis_policy.verify_authentication():
        raise Lfm2CpcfShortConvV5Error("FIR basis policy failed authentication.")
    if (
        isinstance(target_channels, bool)
        or not isinstance(target_channels, int)
        or target_channels <= 0
    ):
        raise Lfm2CpcfShortConvV5Error("target_channels must be a positive integer.")
    fir = _snapshot_tensor(
        source_fir_kernel,
        "source_fir_kernel",
        dimensions=2,
    )
    source_channels = int(fir.shape[0])
    if target_channels > source_channels:
        raise Lfm2CpcfShortConvV5Error("target_channels cannot exceed source channels.")
    values = tuple(cases)
    if not values or any(
        not isinstance(case, Lfm2CpcfV5FirBasisCase) for case in values
    ):
        raise Lfm2CpcfShortConvV5Error(
            "FIR basis construction requires authenticated fit cases."
        )
    ordered = tuple(sorted(values, key=lambda case: case.case_id))
    if len({case.case_id for case in ordered}) != len(ordered):
        raise Lfm2CpcfShortConvV5Error("FIR basis case ids must be unique.")
    activation_width = int(ordered[0].activation_descriptors.shape[1])
    endpoint_width = int(ordered[0].endpoint_descriptors.shape[1])
    for case in ordered:
        if not case.verify_authentication():
            raise Lfm2CpcfShortConvV5Error(
                f"FIR basis case {case.case_id!r} changed after authentication."
            )
        if (
            int(case.activation_descriptors.shape[0]) != source_channels
            or int(case.endpoint_descriptors.shape[0]) != source_channels
            or int(case.activation_descriptors.shape[1]) != activation_width
            or int(case.endpoint_descriptors.shape[1]) != endpoint_width
        ):
            raise Lfm2CpcfShortConvV5Error(
                "FIR basis descriptor dimensions are inconsistent."
            )

    descriptor_blocks = [
        _equilibrate_descriptor_block(fir) * basis_policy.fir_descriptor_weight
    ]
    for case in ordered:
        descriptor_blocks.extend(
            (
                _equilibrate_descriptor_block(case.activation_descriptors)
                * basis_policy.activation_descriptor_weight,
                _equilibrate_descriptor_block(case.endpoint_descriptors)
                * basis_policy.endpoint_descriptor_weight,
            )
        )
    descriptors = torch.cat(descriptor_blocks, dim=1).contiguous()
    (
        seeds,
        groups,
        assignment,
        capacity_assignment_distance_lookups,
        _seed_squared_distances,
    ) = _balanced_farthest_first_with_seed_distance_table(
        descriptors,
        target_channels=target_channels,
    )
    minimum_group_size, larger_groups = divmod(
        source_channels,
        target_channels,
    )
    capacities = [
        minimum_group_size + int(group < larger_groups)
        for group in range(target_channels)
    ]
    groups = [sorted(group) for group in groups]
    if sorted(source for group in groups for source in group) != list(
        range(source_channels)
    ) or any(len(group) != capacity for group, capacity in zip(groups, capacities)):
        raise Lfm2CpcfShortConvV5Error(
            "Balanced FIR grouping did not cover every source exactly once."
        )

    source_energy = fir.square().mean(dim=1)
    for case in ordered:
        source_energy = source_energy + case.activation_descriptors.square().mean(dim=1)
        source_energy = source_energy + case.endpoint_descriptors.square().mean(dim=1)
    maximum_energy = float(source_energy.max().item())
    energy_floor = max(
        maximum_energy * basis_policy.energy_floor_relative,
        torch.finfo(torch.float64).tiny,
    )
    source_energy = source_energy.clamp(min=energy_floor)
    encoder = torch.zeros(
        (source_channels, target_channels),
        dtype=torch.float64,
    )
    group_energy_shares = []
    for group, sources in enumerate(groups):
        group_energy = source_energy[sources]
        shares = group_energy / group_energy.sum()
        encoder[sources, group] = shares.sqrt()
        group_energy_shares.append([float(value) for value in shares.tolist()])
    decoder = encoder.T.contiguous()
    dual_identity = decoder @ encoder
    dual_error = float(
        (dual_identity - torch.eye(target_channels, dtype=torch.float64))
        .abs()
        .max()
        .item()
    )
    projected_fir = torch.stack(
        [
            decoder @ torch.diag(fir[:, lag]) @ encoder
            for lag in range(int(fir.shape[1]))
        ],
        dim=0,
    )
    projected_diagonal = torch.diagonal(
        projected_fir,
        dim1=1,
        dim2=2,
    )
    projected_offdiagonal = projected_fir - torch.diag_embed(projected_diagonal)
    projected_norm = float(torch.linalg.vector_norm(projected_fir).item())
    offdiagonal_norm = float(torch.linalg.vector_norm(projected_offdiagonal).item())
    offdiagonal_relative = offdiagonal_norm / max(
        projected_norm,
        torch.finfo(torch.float64).eps,
    )
    projected_offdiagonal_gate = (
        basis_policy.maximum_projected_fir_offdiagonal_relative_frobenius
    )
    target_fir_kernel = projected_diagonal.T.contiguous()
    commutator_residuals = []
    for lag in range(int(fir.shape[1])):
        residual = torch.diag(fir[:, lag]) @ encoder - encoder @ torch.diag(
            target_fir_kernel[:, lag]
        )
        denominator = max(
            float(torch.linalg.vector_norm(torch.diag(fir[:, lag]) @ encoder).item()),
            torch.finfo(torch.float64).eps,
        )
        commutator_residuals.append(
            float(torch.linalg.vector_norm(residual).item()) / denominator
        )
    maximum_commutator_residual = max(commutator_residuals)
    commutator_gate = basis_policy.maximum_per_lag_commutator_relative_frobenius
    projected_offdiagonal_gate_passed = (
        offdiagonal_relative <= projected_offdiagonal_gate
    )
    commutator_gate_passed = maximum_commutator_residual <= commutator_gate
    if not projected_offdiagonal_gate_passed:
        raise Lfm2CpcfShortConvV5Error(
            "Projected FIR off-diagonal energy exceeds the declared gate."
        )
    if not commutator_gate_passed:
        raise Lfm2CpcfShortConvV5Error(
            "Maximum per-lag FIR commutator residual exceeds the declared gate."
        )

    row_participation = encoder.abs().sum(dim=1)
    numeric_participants = [
        index
        for index, value in enumerate(row_participation.tolist())
        if float(value) > 0.0
    ]
    state = {
        "decoder": decoder,
        "encoder": encoder,
        "target_fir_kernel": target_fir_kernel,
    }
    body = {
        "schema": LFM2_CPCF_SHORT_CONV_V5_SCHEMA,
        "kind": "fir_commuting_balanced_all_channel_basis",
        "policy_sha256": basis_policy.authentication_sha256,
        "fit_only": True,
        "selection_cases_consumed": 0,
        "validation_cases_consumed": 0,
        "fit_case_authentication_sha256": [
            case.authentication_sha256 for case in ordered
        ],
        "source_fir_kernel_sha256": _tensor_sha256(
            fir,
            "source_fir_kernel",
        ),
        "source_channels": source_channels,
        "target_channels": target_channels,
        "fir_length": int(fir.shape[1]),
        "activation_descriptor_width": activation_width,
        "endpoint_descriptor_width": endpoint_width,
        "combined_descriptor_sha256": _tensor_sha256(
            descriptors,
            "fir_basis.combined_descriptors",
        ),
        "seed_distance_table": _seed_distance_table_complexity_receipt(
            source_channels=source_channels,
            target_channels=target_channels,
            descriptor_width=int(descriptors.shape[1]),
            descriptor_block_count=len(descriptor_blocks),
            capacity_assignment_distance_lookups=(capacity_assignment_distance_lookups),
        ),
        "seed_source_indices": seeds,
        "deterministic_farthest_first": True,
        "tie_break": "lowest_source_then_lowest_group_index",
        "balanced_group_capacities": capacities,
        "balanced_groups": groups,
        "source_to_target_group": assignment,
        "all_source_channels_consumed": list(range(source_channels)),
        "source_channels_consumed_count": source_channels,
        "source_channels_selected": len(numeric_participants),
        "source_channels_dropped": 0,
        "source_channels_pruned": 0,
        "full_source_coverage": len(numeric_participants) == source_channels,
        "full_source_numeric_participation": (
            len(numeric_participants) == source_channels
        ),
        "numeric_participating_source_indices": numeric_participants,
        "minimum_numeric_participation_weight": float(row_participation.min().item()),
        "source_energy_floor": energy_floor,
        "source_energy": [float(value) for value in source_energy.tolist()],
        "group_energy_shares": group_energy_shares,
        "energy_weighting": "sqrt_group_energy_share",
        "encoder_disjoint_support": True,
        "decoder": "exact_euclidean_dual",
        "maximum_dual_identity_error": dual_error,
        "exact_dual": dual_error <= 64.0 * torch.finfo(torch.float64).eps,
        "projected_fir_offdiagonal_relative_frobenius": offdiagonal_relative,
        "projected_fir_offdiagonal_gate": projected_offdiagonal_gate,
        "projected_fir_offdiagonal_gate_passed": (projected_offdiagonal_gate_passed),
        "projected_offdiagonal_is_not_sufficient_for_commuting_claim": True,
        "per_lag_commutator_relative_frobenius": commutator_residuals,
        "maximum_per_lag_commutator_relative_frobenius": (maximum_commutator_residual),
        "maximum_per_lag_commutator_relative_frobenius_gate": (commutator_gate),
        "maximum_per_lag_commutator_relative_frobenius_gate_passed": (
            commutator_gate_passed
        ),
        "commuting_claim_uses_actual_intertwining_residual": True,
        "depthwise_fir_representable_in_target_basis": (
            projected_offdiagonal_gate_passed and commutator_gate_passed
        ),
        "state_sha256": _state_sha256(state, "fir_commuting_basis"),
        "execution": {
            **_ZERO_TRAINING_EXECUTION,
            "deterministic_farthest_first_only": True,
            "all_channel_partition_not_pruning": True,
        },
    }
    if not body["full_source_numeric_participation"] or not body["exact_dual"]:
        raise Lfm2CpcfShortConvV5Error(
            "FIR-commuting basis failed participation or exact-dual invariants."
        )
    return Lfm2CpcfV5FirCommutingBasis(
        encoder=encoder,
        decoder=decoder,
        target_fir_kernel=target_fir_kernel,
        source_to_target_group=tuple(assignment),
        report=_seal_report(body),
    )


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5AcceptedPathState:
    """One fit-only accepted path state bound to its immediate predecessor."""

    stage_id: str
    state: Mapping[str, Any]
    acceptance_report_sha256: str
    predecessor_authentication_sha256: str | None = None
    authentication_sha256: str = ""
    _state_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        name = str(self.stage_id).strip()
        if not name:
            raise Lfm2CpcfShortConvV5Error("stage_id must be nonempty text.")
        if not isinstance(self.state, Mapping) or not self.state:
            raise Lfm2CpcfShortConvV5Error(
                "An accepted path state requires at least one tensor role."
            )
        snapshot = {
            str(role): _snapshot_state_tensor(value, f"{name}.{role}")
            for role, value in sorted(self.state.items())
        }
        if any(not role for role in snapshot):
            raise Lfm2CpcfShortConvV5Error(
                "Accepted path tensor role names must be nonempty."
            )
        acceptance = _valid_sha256(
            self.acceptance_report_sha256,
            "acceptance_report_sha256",
        )
        predecessor = self.predecessor_authentication_sha256
        if predecessor is not None:
            predecessor = _valid_sha256(
                predecessor,
                "predecessor_authentication_sha256",
            )
        object.__setattr__(self, "stage_id", name)
        object.__setattr__(self, "state", MappingProxyType(snapshot))
        object.__setattr__(self, "acceptance_report_sha256", acceptance)
        object.__setattr__(self, "predecessor_authentication_sha256", predecessor)
        state_digest = _state_sha256(snapshot, f"accepted_path.{name}")
        object.__setattr__(self, "_state_sha256", state_digest)
        expected = _sha256(self.authentication_body())
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2CpcfShortConvV5Error(
                "accepted path state authentication_sha256 is incorrect."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    @property
    def state_sha256(self) -> str:
        return self._state_sha256

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CPCF_SHORT_CONV_V5_SCHEMA,
            "kind": "accepted_fit_only_projection_path_state",
            "stage_id": self.stage_id,
            "state_sha256": self._state_sha256,
            "tensor_roles": list(self.state),
            "acceptance_report_sha256": self.acceptance_report_sha256,
            "predecessor_authentication_sha256": (
                self.predecessor_authentication_sha256
            ),
            "fit_only": True,
            "selection_cases_consumed": 0,
            "validation_cases_consumed": 0,
        }

    def verify_authentication(self) -> bool:
        return self._state_sha256 == _state_sha256(
            self.state, f"accepted_path.{self.stage_id}"
        ) and self.authentication_sha256 == _sha256(self.authentication_body())


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5InterpolatedEdgeState:
    """One materialized interior state on one authenticated accepted edge."""

    stage_id: str
    state: Mapping[str, Any]
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "state",
            MappingProxyType(
                {
                    str(role): _snapshot_state_tensor(
                        value,
                        f"{self.stage_id}.{role}",
                    )
                    for role, value in sorted(self.state.items())
                }
            ),
        )
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    def verify_authentication(self) -> bool:
        return verify_lfm2_cpcf_short_conv_v5_report(self.report) and self.report.get(
            "state_sha256"
        ) == _state_sha256(self.state, f"accepted_edge.{self.stage_id}")


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5AcceptedEdgeFrontier:
    candidates: tuple[Lfm2CpcfV5InterpolatedEdgeState, ...]
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    def verify_authentication(self) -> bool:
        return (
            verify_lfm2_cpcf_short_conv_v5_report(self.report)
            and all(candidate.verify_authentication() for candidate in self.candidates)
            and self.report.get("candidate_report_sha256")
            == [candidate.report["report_sha256"] for candidate in self.candidates]
        )


def fixed_v5_accepted_edge_interpolation_fractions() -> tuple[float, ...]:
    """Return the frozen interior grid applied to every accepted edge."""

    return V5_ACCEPTED_EDGE_INTERPOLATION_FRACTIONS


def interpolate_lfm2_cpcf_v5_accepted_edge(
    *,
    left: Lfm2CpcfV5AcceptedPathState,
    right: Lfm2CpcfV5AcceptedPathState,
    fraction: float,
    edge_index: int,
    fraction_index: int,
) -> Lfm2CpcfV5InterpolatedEdgeState:
    """Interpolate all tensor roles on one authenticated adjacent path edge."""

    if not isinstance(left, Lfm2CpcfV5AcceptedPathState) or not isinstance(
        right,
        Lfm2CpcfV5AcceptedPathState,
    ):
        raise TypeError("left and right must be accepted v5 path states.")
    if not left.verify_authentication() or not right.verify_authentication():
        raise Lfm2CpcfShortConvV5Error("Accepted-edge endpoints failed authentication.")
    if right.predecessor_authentication_sha256 != left.authentication_sha256:
        raise Lfm2CpcfShortConvV5Error(
            "Accepted-edge endpoints are not an authenticated adjacent pair."
        )
    amount = _finite_float(fraction, "fraction")
    if amount <= 0.0 or amount >= 1.0:
        raise Lfm2CpcfShortConvV5Error(
            "Accepted-edge interpolation fractions must be strictly inside (0, 1)."
        )
    if (
        isinstance(edge_index, bool)
        or not isinstance(edge_index, int)
        or edge_index < 0
        or isinstance(fraction_index, bool)
        or not isinstance(fraction_index, int)
        or fraction_index < 0
    ):
        raise Lfm2CpcfShortConvV5Error(
            "edge_index and fraction_index must be nonnegative integers."
        )
    if tuple(left.state) != tuple(right.state):
        raise Lfm2CpcfShortConvV5Error("Accepted-edge tensor role topology differs.")
    state = {}
    for role in left.state:
        left_value = left.state[role]
        right_value = right.state[role]
        if (
            tuple(left_value.shape) != tuple(right_value.shape)
            or left_value.dtype != right_value.dtype
        ):
            raise Lfm2CpcfShortConvV5Error(
                f"Accepted-edge tensor role {role!r} shape or dtype differs."
            )
        state[role] = left_value.add(
            right_value.sub(left_value),
            alpha=amount,
        ).contiguous()
    stage_id = (
        f"edge-{edge_index:02d}-{left.stage_id}-to-{right.stage_id}"
        f"--fraction-{fraction_index:02d}-{amount:.17g}"
    )
    body = {
        "schema": LFM2_CPCF_SHORT_CONV_V5_SCHEMA,
        "kind": "per_accepted_edge_interpolated_state",
        "stage_id": stage_id,
        "edge_index": edge_index,
        "fraction_index": fraction_index,
        "fraction": amount,
        "left_stage_id": left.stage_id,
        "left_authentication_sha256": left.authentication_sha256,
        "left_state_sha256": left.state_sha256,
        "right_stage_id": right.stage_id,
        "right_authentication_sha256": right.authentication_sha256,
        "right_state_sha256": right.state_sha256,
        "right_declares_left_as_predecessor": True,
        "all_tensor_roles_interpolated": list(state),
        "state_sha256": _state_sha256(state, f"accepted_edge.{stage_id}"),
        "fit_only": True,
        "selection_cases_consumed": 0,
        "validation_cases_consumed": 0,
        "execution": {
            **_ZERO_TRAINING_EXECUTION,
            "closed_form_tensor_interpolation": True,
        },
    }
    return Lfm2CpcfV5InterpolatedEdgeState(
        stage_id=stage_id,
        state=state,
        report=_seal_report(body),
    )


def interpolate_every_lfm2_cpcf_v5_accepted_edge(
    path: Sequence[Lfm2CpcfV5AcceptedPathState],
) -> Lfm2CpcfV5AcceptedEdgeFrontier:
    """Materialize the frozen interpolation grid on every adjacent accepted edge."""

    states = tuple(path)
    if len(states) < 2:
        raise Lfm2CpcfShortConvV5Error(
            "Per-edge interpolation requires at least two accepted states."
        )
    if any(not isinstance(state, Lfm2CpcfV5AcceptedPathState) for state in states):
        raise Lfm2CpcfShortConvV5Error("The accepted path contains an invalid state.")
    if len({state.stage_id for state in states}) != len(states):
        raise Lfm2CpcfShortConvV5Error("Accepted path stage ids must be unique.")
    if not all(state.verify_authentication() for state in states):
        raise Lfm2CpcfShortConvV5Error("An accepted path state failed authentication.")
    if states[0].predecessor_authentication_sha256 is not None:
        raise Lfm2CpcfShortConvV5Error(
            "The first accepted path state must not declare a predecessor."
        )
    candidates = []
    edge_inventory = []
    fractions = fixed_v5_accepted_edge_interpolation_fractions()
    for edge_index, (left, right) in enumerate(zip(states, states[1:])):
        edge_candidates = [
            interpolate_lfm2_cpcf_v5_accepted_edge(
                left=left,
                right=right,
                fraction=fraction,
                edge_index=edge_index,
                fraction_index=fraction_index,
            )
            for fraction_index, fraction in enumerate(fractions)
        ]
        candidates.extend(edge_candidates)
        edge_inventory.append(
            {
                "edge_index": edge_index,
                "left_stage_id": left.stage_id,
                "left_authentication_sha256": left.authentication_sha256,
                "right_stage_id": right.stage_id,
                "right_authentication_sha256": right.authentication_sha256,
                "candidate_count": len(edge_candidates),
                "candidate_report_sha256": [
                    candidate.report["report_sha256"] for candidate in edge_candidates
                ],
            }
        )
    body = {
        "schema": LFM2_CPCF_SHORT_CONV_V5_SCHEMA,
        "kind": "complete_per_accepted_edge_interpolation_frontier",
        "accepted_path_state_count": len(states),
        "accepted_edge_count": len(states) - 1,
        "covered_accepted_edge_count": len(edge_inventory),
        "skipped_accepted_edge_count": 0,
        "per_accepted_edge_invariant": True,
        "fractions": list(fractions),
        "candidates_per_edge": len(fractions),
        "candidate_count": len(candidates),
        "accepted_path_authentication_sha256": [
            state.authentication_sha256 for state in states
        ],
        "edge_inventory": edge_inventory,
        "candidate_report_sha256": [
            candidate.report["report_sha256"] for candidate in candidates
        ],
        "fit_only": True,
        "selection_cases_consumed": 0,
        "validation_cases_consumed": 0,
        "execution": {
            **_ZERO_TRAINING_EXECUTION,
            "closed_form_tensor_interpolation": True,
        },
    }
    return Lfm2CpcfV5AcceptedEdgeFrontier(
        candidates=tuple(candidates),
        report=_seal_report(body),
    )


__all__ = [
    "LFM2_CPCF_SHORT_CONV_V5_SCHEMA",
    "Lfm2CpcfShortConvV5Error",
    "Lfm2CpcfV5AcceptedEdgeFrontier",
    "Lfm2CpcfV5AcceptedPathState",
    "Lfm2CpcfV5FirBasisCase",
    "Lfm2CpcfV5FirBasisPolicy",
    "Lfm2CpcfV5FirCommutingBasis",
    "Lfm2CpcfV5InterpolatedEdgeState",
    "Lfm2CpcfV5RidgeCase",
    "Lfm2CpcfV5RidgePolicy",
    "Lfm2CpcfV5RidgeResult",
    "Lfm2CpcfV5SelectionPolicy",
    "V5_ACCEPTED_EDGE_INTERPOLATION_FRACTIONS",
    "V5_CONDITION_CAP_CANDIDATES",
    "V5_CONTEXT_EXCHANGE_BETA",
    "V5_PRIOR_CONTROL_SUFFIX",
    "build_fir_commuting_balanced_all_channel_basis",
    "condition_capped_equilibrated_prior_residual_ridge_frontier",
    "fixed_v5_accepted_edge_interpolation_fractions",
    "interpolate_every_lfm2_cpcf_v5_accepted_edge",
    "interpolate_lfm2_cpcf_v5_accepted_edge",
    "rescore_sealed_v4_short_conv_selection_frontier",
    "verify_lfm2_cpcf_short_conv_v5_report",
]
