"""Real fit-only v5 numerical frontier for the frozen short-conv seam.

The adapter consumes only the detached FIT evidence exposed by
``Lfm2CpcfShortConvNumericalFrontierRequest``.  It performs deterministic
closed-form folding in this audited order:

1. resize the source FIR to the target FIR length;
2. build case-balanced activation and endpoint descriptors;
3. build a commutator-gated, balanced, all-source-channel basis;
4. realize two authenticated future-Hankel Volterra matrix-free policy arms;
5. fit the two frozen condition-cap input lifts and transport parent priors;
6. construct target projection cases;
7. alternate affine writer, no-intercept physical FIR, and no-intercept
   physical B/C/x projection solves;
8. chain every FIT-accepted state and interpolate every accepted edge;
9. select the strongest accepted writer control on FIT evidence only and
   construct one fixed full-coverage writer hybrid per FHVR arm; and
10. return the complete typed raw frontier to the existing compiler.

There are no model forwards, generation calls, gradients, optimizer steps,
training steps, pruning decisions, quantization operations, selection cases,
or validation cases in this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .hf_lfm2_cpcf_short_conv_v4 import (
    Lfm2CpcfV4ProjectionCase,
    linearized_bcx_affine_design,
    replay_sampled_cp_volterra,
    sampled_cp_volterra_objective,
)
from .hf_lfm2_cpcf_short_conv_v4_fold import (
    Lfm2CpcfShortConvNumericalFrontier,
    Lfm2CpcfShortConvNumericalFrontierRequest,
    Lfm2CpcfShortConvNumericalRawCandidate,
)
from .hf_lfm2_cpcf_short_conv_v5 import (
    V5_CONDITION_CAP_CANDIDATES,
    V5_PRIOR_CONTROL_SUFFIX,
    Lfm2CpcfShortConvV5Error,
    Lfm2CpcfV5AcceptedPathState,
    Lfm2CpcfV5AllChannelFirRidgeCase,
    Lfm2CpcfV5FirBasisCase,
    Lfm2CpcfV5FirCommutingBasis,
    Lfm2CpcfV5FirBasisPolicy,
    Lfm2CpcfV5RidgeCase,
    Lfm2CpcfV5RidgePolicy,
    build_fir_commuting_balanced_all_channel_basis,
    condition_capped_equilibrated_prior_residual_ridge_frontier,
    interpolate_every_lfm2_cpcf_v5_accepted_edge,
    solve_condition_capped_batched_all_channel_fir_ridge_candidate,
    solve_condition_capped_equilibrated_prior_residual_ridge_candidate,
)
from .hf_lfm2_cpcf_short_conv_v5_matrix_free_bcx import (
    Lfm2CpcfV5MatrixFreeBcxPolicy,
    solve_lfm2_cpcf_v5_matrix_free_bcx,
)
from .hf_lfm2_cpcf_short_conv_v5_fhvr_mf import (
    FHVR_MF_CAP10K_SWEEPS1_CANDIDATE_ID,
    FHVR_MF_CAP10K_SWEEPS1_FULL_COVERAGE_HYBRID_ID,
    FHVR_MF_CAP1K_SWEEPS2_CANDIDATE_ID,
    FHVR_MF_CAP1K_SWEEPS2_FULL_COVERAGE_HYBRID_ID,
    FHVR_MF_CANDIDATE_ID,
    FHVR_MF_CANDIDATE_KIND,
    FHVR_MF_FULL_COVERAGE_FHVR_FRACTION,
    FHVR_MF_FULL_COVERAGE_HYBRID_KIND,
    FHVR_MF_FULL_COVERAGE_REFACTOR_SWEEPS,
    FHVR_MF_FULL_COVERAGE_WRITER_FRACTION,
    Lfm2CpcfFhvrMfPolicy,
    build_lfm2_cpcf_fhvr_mf_candidate,
    build_lfm2_cpcf_fhvr_writer_full_coverage_hybrid_candidate,
)
from .hf_lfm2_cpcf_short_conv_v5_residual_subspace_bcx import (
    causal_horizon_balanced_row_weights,
    solve_lfm2_cpcf_v5_residual_subspace_bcx,
)


LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_ADAPTER_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-v5-numerical-adapter"
)
_ADAPTER_IMPLEMENTATION = (
    "neural_forge.hf_lfm2_cpcf_short_conv_v5_numerical_adapter."
    "Lfm2CpcfShortConvV5NumericalFrontierAdapter.v6"
)
_STATE_ROLES = (
    "in_proj_weight",
    "in_proj_bias",
    "fir_kernel",
    "fir_bias",
    "writer_weight",
    "writer_bias",
)
_SCALABLE_BCX_MINIMUM_HIDDEN_SIZE = 64
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


class Lfm2CpcfShortConvV5NumericalAdapterError(ValueError):
    """The real v5 numerical adapter received inconsistent evidence."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = (
            None
            if diagnostics is None
            else json.loads(_canonical_json(dict(diagnostics)).decode("ascii"))
        )


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for the LFM2 CPCF v5 numerical adapter."
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
        raise Lfm2CpcfShortConvV5NumericalAdapterError(
            "Numerical-adapter metadata must be finite canonical JSON."
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _seal_report(body: Mapping[str, Any]) -> dict[str, Any]:
    detached = dict(body)
    return {**detached, "report_sha256": _sha256(detached)}


def _cap_id(condition_cap: float) -> str:
    return f"kappa-{int(condition_cap):05d}"


def _condition_cap_from_id(
    ridge_policy: Lfm2CpcfV5RidgePolicy,
    cap_id: str,
) -> float:
    return {
        _cap_id(condition_cap): condition_cap
        for condition_cap in ridge_policy.condition_caps
    }[cap_id]


def _factor_ridge_policy(
    base: Lfm2CpcfV5RidgePolicy,
    *,
    parameter_trust_radius: float,
) -> Lfm2CpcfV5RidgePolicy:
    """Preserve v5 conditioning while honoring the narrower factor radius."""

    return Lfm2CpcfV5RidgePolicy(
        condition_caps=base.condition_caps,
        rank_relative_tolerance=base.rank_relative_tolerance,
        parameter_trust_radius=min(
            base.parameter_trust_radius,
            parameter_trust_radius,
        ),
        effect_trust_radius=base.effect_trust_radius,
        parameter_reference_floor=base.parameter_reference_floor,
        effect_reference_floor=base.effect_reference_floor,
    )


def _fp64_compute_tensor(value: Any) -> Any:
    """Promote detached numerical evidence to the authoritative solve domain."""

    torch = _torch()
    return value.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()


def _resize_fir_rows(source_fir: Any, target_length: int) -> Any:
    """Resize every source FIR row without selecting or dropping channels."""

    torch = _torch()
    source_fir = _fp64_compute_tensor(source_fir)
    source_length = int(source_fir.shape[1])
    if source_length == target_length:
        return source_fir.clone().contiguous()
    if target_length == 1:
        return source_fir.mean(dim=1, keepdim=True).contiguous()
    if source_length == 1:
        return source_fir.expand(-1, target_length).clone().contiguous()
    return (
        torch.nn.functional.interpolate(
            source_fir.unsqueeze(0),
            size=target_length,
            mode="linear",
            align_corners=True,
        )
        .squeeze(0)
        .contiguous()
    )


def _lag_windows(value: Any, fir_length: int) -> Any:
    torch = _torch()
    padded = torch.nn.functional.pad(value, (fir_length - 1, 0))
    return padded.unfold(dimension=1, size=fir_length, step=1)


def _all_channel_fir_design(
    replay: Any,
    fir_length: int,
    *,
    include_bias: bool,
) -> Any:
    """Build every channel's FIR design in one tensor operation."""

    torch = _torch()
    product = (replay.b_history * replay.x_history).movedim(-1, 1)
    lagged = torch.nn.functional.pad(
        product,
        (fir_length - 1, 0),
    ).unfold(
        dimension=2,
        size=fir_length,
        step=1,
    )
    c_history = replay.c_history.movedim(-1, 1).unsqueeze(-1)
    design = lagged * c_history
    if include_bias:
        design = torch.cat((design, c_history), dim=-1)
    return design.movedim(1, 0).contiguous()


def _present_state(state: Mapping[str, Any | None]) -> dict[str, Any]:
    return {role: state[role] for role in _STATE_ROLES if state[role] is not None}


def _clone_state(state: Mapping[str, Any | None]) -> dict[str, Any | None]:
    return {
        role: None if state[role] is None else state[role].clone()
        for role in _STATE_ROLES
    }


def _masked_channel_statistics(value: Any, mask: Any) -> tuple[Any, Any]:
    rows = value.reshape(-1, int(value.shape[-1]))[mask.reshape(-1)]
    return rows.mean(dim=0), rows.square().mean(dim=0).sqrt()


def _basis_cases(
    request: Lfm2CpcfShortConvNumericalFrontierRequest,
) -> tuple[Lfm2CpcfV5FirBasisCase, ...]:
    """Build one equally represented descriptor block per FIT case."""

    torch = _torch()
    source = request.source_tensors
    endpoint_operator = (
        _fp64_compute_tensor(source["source_out_proj_weight"]).T
        @ _fp64_compute_tensor(source["output_encoder"])
    )
    result = []
    for case in request.cases:
        mask = case.valid_prefix_mask
        parent_context = _fp64_compute_tensor(
            request.parent_contexts[case.case_id]
        )
        activation_columns = []
        for value in (
            case.parent_operator_input_history,
            case.parent_b_history,
            case.parent_c_history,
            case.parent_x_history,
            parent_context,
        ):
            mean, rms = _masked_channel_statistics(
                _fp64_compute_tensor(value),
                mask,
            )
            activation_columns.extend((mean, rms))
        activation_descriptors = torch.stack(
            activation_columns,
            dim=1,
        ).contiguous()

        suffix_mask = mask[:, case.suffix_start :]
        context_rows = parent_context[:, case.suffix_start :, :].reshape(
            -1,
            int(parent_context.shape[-1]),
        )[suffix_mask.reshape(-1)]
        desired_write = (
            _fp64_compute_tensor(
                case.parent_post_operator_residual_target_space_suffix
            )
            - _fp64_compute_tensor(
                case.actual_child_pre_operator_residual_suffix
            )
        )
        write_rows = desired_write.reshape(
            -1,
            int(desired_write.shape[-1]),
        )[suffix_mask.reshape(-1)]
        cross_endpoint = context_rows.T @ write_rows / int(context_rows.shape[0])
        context_rms = context_rows.square().mean(dim=0).sqrt()
        transported_endpoint = endpoint_operator * context_rms.unsqueeze(1)
        endpoint_descriptors = torch.cat(
            (cross_endpoint, transported_endpoint),
            dim=1,
        ).contiguous()
        result.append(
            Lfm2CpcfV5FirBasisCase(
                case_id=case.case_id,
                split="fit",
                activation_descriptors=activation_descriptors,
                endpoint_descriptors=endpoint_descriptors,
            )
        )
    return tuple(result)


def _build_audited_basis(
    request: Lfm2CpcfShortConvNumericalFrontierRequest,
    *,
    descriptors: Sequence[Lfm2CpcfV5FirBasisCase],
    resized_fir: Any,
    target_hidden: int,
    requested_policy: Lfm2CpcfV5FirBasisPolicy,
) -> Any:
    """Measure the actual commutator before enforcing the preregistered gate."""

    audit_policy = Lfm2CpcfV5FirBasisPolicy(
        fir_descriptor_weight=requested_policy.fir_descriptor_weight,
        activation_descriptor_weight=(requested_policy.activation_descriptor_weight),
        endpoint_descriptor_weight=requested_policy.endpoint_descriptor_weight,
        energy_floor_relative=requested_policy.energy_floor_relative,
        maximum_projected_fir_offdiagonal_relative_frobenius=(
            requested_policy.maximum_projected_fir_offdiagonal_relative_frobenius
        ),
        maximum_per_lag_commutator_relative_frobenius=1e300,
    )
    audit_basis = build_fir_commuting_balanced_all_channel_basis(
        descriptors,
        source_fir_kernel=resized_fir,
        target_channels=target_hidden,
        policy=audit_policy,
    )
    observed = float(
        audit_basis.report["maximum_per_lag_commutator_relative_frobenius"]
    )
    requested_gate = requested_policy.maximum_per_lag_commutator_relative_frobenius
    if observed > requested_gate:
        diagnostic = _seal_report(
            {
                "schema": (LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_ADAPTER_SCHEMA),
                "kind": "fir_commutator_gate_failure_evidence",
                "request_sha256": request.request_sha256,
                "requested_basis_policy_sha256": (
                    requested_policy.authentication_sha256
                ),
                "audit_basis_policy_sha256": (audit_policy.authentication_sha256),
                "requested_maximum_per_lag_commutator_relative_frobenius": (
                    requested_gate
                ),
                "observed_maximum_per_lag_commutator_relative_frobenius": (observed),
                "per_lag_commutator_relative_frobenius": list(
                    audit_basis.report["per_lag_commutator_relative_frobenius"]
                ),
                "source_to_target_group": list(audit_basis.source_to_target_group),
                "balanced_groups": list(audit_basis.report["balanced_groups"]),
                "audit_basis_report_sha256": audit_basis.report["report_sha256"],
                "audit_basis_report": dict(audit_basis.report),
                "requested_gate_passed": False,
                "fit_only": True,
                "selection_cases_consumed": 0,
                "validation_cases_consumed": 0,
                "execution": dict(_ZERO_EXECUTION),
            }
        )
        raise Lfm2CpcfShortConvV5NumericalAdapterError(
            "Maximum per-lag FIR commutator residual "
            f"{observed:.17g} exceeds the preregistered gate "
            f"{requested_gate:.17g}.",
            diagnostics=diagnostic,
        )
    if not requested_policy.verify_authentication():
        raise Lfm2CpcfShortConvV5Error("FIR basis policy failed authentication.")
    requested_report = copy.deepcopy(dict(audit_basis.report))
    requested_report.pop("report_sha256")
    requested_report.update(
        {
            "policy_sha256": requested_policy.authentication_sha256,
            "maximum_per_lag_commutator_relative_frobenius_gate": (
                requested_gate
            ),
            "maximum_per_lag_commutator_relative_frobenius_gate_passed": True,
            "depthwise_fir_representable_in_target_basis": True,
        }
    )
    return Lfm2CpcfV5FirCommutingBasis(
        encoder=audit_basis.encoder,
        decoder=audit_basis.decoder,
        target_fir_kernel=audit_basis.target_fir_kernel,
        source_to_target_group=audit_basis.source_to_target_group,
        report=_seal_report(requested_report),
    )


def _input_lift_frontier(
    request: Lfm2CpcfShortConvNumericalFrontierRequest,
    *,
    prior_weight: Any,
    ridge_policy: Lfm2CpcfV5RidgePolicy,
) -> Mapping[str, Any]:
    """Fit child-to-parent input lifts for both frozen condition caps."""

    torch = _torch()
    fit_intercept = request.target_templates["in_proj_bias"] is not None
    cases = tuple(
        Lfm2CpcfV5RidgeCase(
            case_id=case.case_id,
            split="fit",
            design=_fp64_compute_tensor(
                case.actual_child_operator_input_history
            ).reshape(
                -1,
                int(case.actual_child_operator_input_history.shape[-1]),
            ),
            target=_fp64_compute_tensor(
                case.parent_operator_input_history
            ).reshape(
                -1,
                int(case.parent_operator_input_history.shape[-1]),
            ),
            valid_rows=case.valid_prefix_mask.reshape(-1),
        )
        for case in request.cases
    )
    return condition_capped_equilibrated_prior_residual_ridge_frontier(
        cases,
        prior_weight=prior_weight,
        prior_bias=(
            torch.zeros(
                (int(prior_weight.shape[1]),),
                dtype=torch.float64,
            )
            if fit_intercept
            else None
        ),
        fit_intercept=fit_intercept,
        policy=ridge_policy,
    )


def _transport_prior(
    request: Lfm2CpcfShortConvNumericalFrontierRequest,
    *,
    encoder: Any,
    decoder: Any,
    target_fir_kernel: Any,
    input_lift: Any,
    input_lift_bias: Any | None,
) -> dict[str, Any | None]:
    """Transport every installable parent tensor into one blank target state."""

    torch = _torch()
    source = request.source_tensors
    templates = request.target_templates
    source_hidden = int(encoder.shape[0])
    role_weights = _fp64_compute_tensor(
        source["source_in_proj_weight"]
    ).reshape(
        3,
        source_hidden,
        source_hidden,
    )
    lift_bias = (
        torch.zeros((source_hidden,), dtype=torch.float64)
        if input_lift_bias is None
        else input_lift_bias
    )
    in_proj_weight = torch.cat(
        tuple(encoder.T @ role_weight @ input_lift.T for role_weight in role_weights),
        dim=0,
    ).contiguous()
    in_proj_bias = None
    if templates["in_proj_bias"] is not None:
        source_in_bias = (
            torch.zeros(
                (3, source_hidden),
                dtype=torch.float64,
            )
            if source["source_in_proj_bias"] is None
            else _fp64_compute_tensor(
                source["source_in_proj_bias"]
            ).reshape(3, source_hidden)
        )
        in_proj_bias = torch.cat(
            tuple(
                encoder.T @ (role_bias + role_weight @ lift_bias)
                for role_weight, role_bias in zip(
                    role_weights,
                    source_in_bias,
                )
            )
        ).contiguous()

    fir_bias = None
    if templates["fir_bias"] is not None:
        source_fir_bias = (
            torch.zeros((source_hidden,), dtype=torch.float64)
            if source["source_fir_bias"] is None
            else _fp64_compute_tensor(source["source_fir_bias"])
        )
        diagonal_transport = decoder.T * encoder
        fir_bias = (diagonal_transport.T @ source_fir_bias).contiguous()

    output_encoder = _fp64_compute_tensor(source["output_encoder"])
    source_out_proj_weight = _fp64_compute_tensor(
        source["source_out_proj_weight"]
    )
    writer_weight = (
        output_encoder.T @ source_out_proj_weight @ decoder.T
    ).contiguous()
    writer_bias = None
    if templates["writer_bias"] is not None:
        source_writer_bias = (
            torch.zeros((source_hidden,), dtype=torch.float64)
            if source["source_out_proj_bias"] is None
            else _fp64_compute_tensor(source["source_out_proj_bias"])
        )
        writer_bias = (output_encoder.T @ source_writer_bias).contiguous()
    return {
        "in_proj_weight": in_proj_weight,
        "in_proj_bias": in_proj_bias,
        "fir_kernel": target_fir_kernel.clone().contiguous(),
        "fir_bias": fir_bias,
        "writer_weight": writer_weight,
        "writer_bias": writer_bias,
    }


def _projection_cases(
    request: Lfm2CpcfShortConvNumericalFrontierRequest,
    *,
    context_encoder: Any,
) -> tuple[Lfm2CpcfV4ProjectionCase, ...]:
    """Project the parent context while retaining the original child endpoint."""

    context_encoder = _fp64_compute_tensor(context_encoder)
    result = []
    for case in request.cases:
        desired_context = (
            _fp64_compute_tensor(request.parent_contexts[case.case_id])
            @ context_encoder
        )
        desired_write = (
            _fp64_compute_tensor(
                case.parent_post_operator_residual_target_space_suffix
            )
            - _fp64_compute_tensor(
                case.actual_child_pre_operator_residual_suffix
            )
        )
        result.append(
            Lfm2CpcfV4ProjectionCase(
                case_id=case.case_id,
                split="fit",
                child_input_history=_fp64_compute_tensor(
                    case.actual_child_operator_input_history
                ),
                desired_context_history=desired_context,
                desired_write_suffix=desired_write,
                suffix_start=case.suffix_start,
                valid_prefix_mask=case.valid_prefix_mask,
                valid_suffix_mask=(case.valid_prefix_mask[:, case.suffix_start :]),
            )
        )
    return tuple(result)


def _replays(
    cases: Sequence[Lfm2CpcfV4ProjectionCase],
    state: Mapping[str, Any | None],
) -> tuple[Any, ...]:
    return tuple(
        replay_sampled_cp_volterra(
            case.child_input_history,
            in_proj_weight=state["in_proj_weight"],
            in_proj_bias=state["in_proj_bias"],
            fir_kernel=state["fir_kernel"],
            fir_bias=state["fir_bias"],
            writer_weight=state["writer_weight"],
            writer_bias=state["writer_bias"],
            valid_prefix_mask=case.valid_prefix_mask,
        )
        for case in cases
    )


def _fit_objective(
    cases: Sequence[Lfm2CpcfV4ProjectionCase],
    state: Mapping[str, Any | None],
    *,
    context_objective_weight: float,
    replays: Sequence[Any] | None = None,
) -> Mapping[str, Any]:
    torch = _torch()
    current_replays = (
        _replays(cases, state) if replays is None else tuple(replays)
    )
    uniform_objective = sampled_cp_volterra_objective(
        cases,
        current_replays,
        required_split="fit",
        context_objective_weight=context_objective_weight,
    )
    reports = []
    for case, replay in sorted(
        zip(cases, current_replays),
        key=lambda value: value[0].case_id,
    ):
        suffix_prediction = replay.write_history[:, case.suffix_start :, :]
        suffix_target = case.desired_write_suffix
        suffix_mask = case.valid_suffix_mask
        prefix_prediction = replay.context_history
        prefix_target = case.desired_context_history
        prefix_mask = case.valid_prefix_mask
        suffix_weights = causal_horizon_balanced_row_weights(
            suffix_mask,
            dtype=torch.float64,
        )
        prefix_weights = causal_horizon_balanced_row_weights(
            prefix_mask,
            dtype=torch.float64,
        )

        def weighted_relative_rmse(
            prediction: Any,
            target: Any,
            weights: Any,
        ) -> float:
            difference_energy = (
                weights
                * (prediction - target).square().mean(dim=-1)
            ).sum()
            target_energy = (
                weights * target.square().mean(dim=-1)
            ).sum()
            floor = torch.finfo(torch.float64).eps
            return float(
                (
                    difference_energy
                    / target_energy.clamp(min=floor)
                )
                .clamp(min=0.0)
                .sqrt()
                .item()
            )

        reports.append(
            {
                "case_id": case.case_id,
                "causal_horizon_endpoint_relative_rmse": (
                    weighted_relative_rmse(
                        suffix_prediction,
                        suffix_target,
                        suffix_weights,
                    )
                ),
                "causal_horizon_context_relative_rmse": (
                    weighted_relative_rmse(
                        prefix_prediction,
                        prefix_target,
                        prefix_weights,
                    )
                ),
            }
        )
    endpoint = sum(
        value["causal_horizon_endpoint_relative_rmse"]
        for value in reports
    ) / len(reports)
    context = sum(
        value["causal_horizon_context_relative_rmse"]
        for value in reports
    ) / len(reports)
    return _seal_report(
        {
            "schema": LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_ADAPTER_SCHEMA,
            "kind": "causal_horizon_residual_fit_objective",
            "fit_only": True,
            "case_balanced": True,
            "case_count": len(reports),
            "cases": reports,
            "row_mass": {
                "uniform_fraction": 0.5,
                "exponential_tail_fraction": 0.5,
                "tail_tau": "max(4,sqrt(valid_rows))",
                "exact_phase_boundaries_available": False,
            },
            "mean_case_endpoint_relative_rmse": endpoint,
            "mean_case_context_relative_rmse": context,
            "context_objective_weight": context_objective_weight,
            "combined_fit_objective": (
                endpoint + context_objective_weight * context
            ),
            "uniform_reference_objective_report_sha256": (
                uniform_objective["report_sha256"]
            ),
            "uniform_reference_combined_fit_objective": (
                uniform_objective["combined_fit_objective"]
            ),
            "execution": dict(_ZERO_EXECUTION),
        }
    )


def _solve_writer(
    cases: Sequence[Lfm2CpcfV4ProjectionCase],
    state: Mapping[str, Any | None],
    *,
    cap_id: str,
    ridge_policy: Lfm2CpcfV5RidgePolicy,
    replays: Sequence[Any] | None = None,
) -> tuple[dict[str, Any | None], list[str]]:
    fit_intercept = state["writer_bias"] is not None
    condition_cap = _condition_cap_from_id(ridge_policy, cap_id)
    ridge_cases = []
    current_replays = _replays(cases, state) if replays is None else tuple(replays)
    for case, replay in zip(cases, current_replays):
        suffix_context = replay.context_history[:, case.suffix_start :, :]
        ridge_cases.append(
            Lfm2CpcfV5RidgeCase(
                case_id=case.case_id,
                split="fit",
                design=suffix_context.reshape(
                    -1,
                    int(suffix_context.shape[-1]),
                ),
                target=case.desired_write_suffix.reshape(
                    -1,
                    int(case.desired_write_suffix.shape[-1]),
                ),
                valid_rows=case.valid_suffix_mask.reshape(-1),
            )
        )
    result = solve_condition_capped_equilibrated_prior_residual_ridge_candidate(
        ridge_cases,
        condition_cap=condition_cap,
        prior_weight=state["writer_weight"].T,
        prior_bias=state["writer_bias"],
        fit_intercept=fit_intercept,
        policy=ridge_policy,
    )
    candidate = _clone_state(state)
    candidate["writer_weight"] = result.weight.T.contiguous()
    candidate["writer_bias"] = None if result.bias is None else result.bias.contiguous()
    return candidate, [str(result.diagnostics["report_sha256"])]


def _solve_fir(
    cases: Sequence[Lfm2CpcfV4ProjectionCase],
    state: Mapping[str, Any | None],
    *,
    cap_id: str,
    ridge_policy: Lfm2CpcfV5RidgePolicy,
    replays: Sequence[Any] | None = None,
) -> tuple[dict[str, Any | None], list[str]]:
    torch = _torch()
    current_replays = _replays(cases, state) if replays is None else tuple(replays)
    kernel = state["fir_kernel"].clone()
    fir_bias = None if state["fir_bias"] is None else state["fir_bias"].clone()
    target_hidden, fir_length = map(int, kernel.shape)
    condition_cap = _condition_cap_from_id(ridge_policy, cap_id)
    case_designs = tuple(
        _all_channel_fir_design(
            replay,
            fir_length,
            include_bias=fir_bias is not None,
        )
        for replay in current_replays
    )
    ridge_cases = tuple(
        Lfm2CpcfV5AllChannelFirRidgeCase(
            case_id=case.case_id,
            split="fit",
            design=base_design.reshape(
                target_hidden,
                -1,
                int(base_design.shape[-1]),
            ),
            target=case.desired_context_history.movedim(-1, 0).reshape(
                target_hidden,
                -1,
            ),
            valid_rows=case.valid_prefix_mask.reshape(-1),
        )
        for case, base_design in zip(cases, case_designs)
    )
    prior = (
        kernel
        if fir_bias is None
        else torch.cat((kernel, fir_bias.unsqueeze(-1)), dim=1)
    )
    result = solve_condition_capped_batched_all_channel_fir_ridge_candidate(
        ridge_cases,
        condition_cap=condition_cap,
        prior_parameters=prior,
        physical_bias_coordinate=fir_bias is not None,
        policy=ridge_policy,
    )
    if not result.verify_authentication():
        raise Lfm2CpcfShortConvV5NumericalAdapterError(
            "Batched all-channel FIR ridge result failed authentication."
        )
    kernel = result.parameters[:, :fir_length].contiguous()
    if fir_bias is not None:
        fir_bias = result.parameters[:, -1].contiguous()
    candidate = _clone_state(state)
    candidate["fir_kernel"] = kernel.contiguous()
    candidate["fir_bias"] = None if fir_bias is None else fir_bias.contiguous()
    return candidate, [str(result.diagnostics["report_sha256"])]


def _solve_bcx_legacy(
    cases: Sequence[Lfm2CpcfV4ProjectionCase],
    state: Mapping[str, Any | None],
    *,
    cap_id: str,
    ridge_policy: Lfm2CpcfV5RidgePolicy,
    replays: Sequence[Any] | None = None,
) -> tuple[dict[str, Any | None], list[str]]:
    torch = _torch()
    current_replays = _replays(cases, state) if replays is None else tuple(replays)
    in_weight = state["in_proj_weight"].clone()
    in_bias = None if state["in_proj_bias"] is None else state["in_proj_bias"].clone()
    target_hidden = int(in_weight.shape[1])
    fit_physical_bias = in_bias is not None
    condition_cap = _condition_cap_from_id(ridge_policy, cap_id)
    receipts = []
    for channel in range(target_hidden):
        row_indices = (
            channel,
            target_hidden + channel,
            2 * target_hidden + channel,
        )
        prior_parts = []
        for row in row_indices:
            prior_parts.append(in_weight[row])
            if in_bias is not None:
                prior_parts.append(in_bias[row].reshape(1))
        prior = torch.cat(prior_parts).reshape(-1, 1)
        ridge_cases = []
        for case, replay in zip(cases, current_replays):
            design = linearized_bcx_affine_design(
                case.child_input_history,
                replay.b_history,
                replay.c_history,
                replay.x_history,
                state["fir_kernel"],
                state["fir_bias"],
                channel=channel,
                fit_bias=fit_physical_bias,
            )
            context_residual = (
                case.desired_context_history[..., channel]
                - replay.context_history[..., channel]
            ).reshape(-1, 1)
            design_rows = design.reshape(-1, int(design.shape[-1]))
            ridge_cases.append(
                Lfm2CpcfV5RidgeCase(
                    case_id=case.case_id,
                    split="fit",
                    design=design_rows,
                    target=context_residual + design_rows @ prior,
                    valid_rows=case.valid_prefix_mask.reshape(-1),
                )
            )
        result = solve_condition_capped_equilibrated_prior_residual_ridge_candidate(
            ridge_cases,
            condition_cap=condition_cap,
            prior_weight=prior,
            prior_bias=None,
            fit_intercept=False,
            policy=ridge_policy,
        )
        feature_width = target_hidden + int(fit_physical_bias)
        solution = result.weight.reshape(3, feature_width)
        for role, row in enumerate(row_indices):
            in_weight[row] = solution[role, :target_hidden]
            if in_bias is not None:
                in_bias[row] = solution[role, -1]
        receipts.append(str(result.diagnostics["report_sha256"]))
    candidate = _clone_state(state)
    candidate["in_proj_weight"] = in_weight.contiguous()
    candidate["in_proj_bias"] = None if in_bias is None else in_bias.contiguous()
    return candidate, receipts


def _solve_bcx(
    cases: Sequence[Lfm2CpcfV4ProjectionCase],
    state: Mapping[str, Any | None],
    *,
    cap_id: str,
    ridge_policy: Lfm2CpcfV5RidgePolicy,
    replays: Sequence[Any] | None = None,
    context_objective_weight: float = 0.05,
) -> tuple[dict[str, Any | None], list[str]]:
    target_hidden = int(state["in_proj_weight"].shape[1])
    if target_hidden < _SCALABLE_BCX_MINIMUM_HIDDEN_SIZE:
        return _solve_bcx_legacy(
            cases,
            state,
            cap_id=cap_id,
            ridge_policy=ridge_policy,
            replays=replays,
        )
    current_replays = (
        _replays(cases, state) if replays is None else tuple(replays)
    )
    if cap_id == _cap_id(V5_CONDITION_CAP_CANDIDATES[0]):
        result = solve_lfm2_cpcf_v5_matrix_free_bcx(
            cases,
            current_replays,
            in_proj_weight=state["in_proj_weight"],
            in_proj_bias=state["in_proj_bias"],
            fir_kernel=state["fir_kernel"],
            fir_bias=state["fir_bias"],
            writer_weight=state["writer_weight"],
            policy=Lfm2CpcfV5MatrixFreeBcxPolicy(
                condition_cap=V5_CONDITION_CAP_CANDIDATES[0],
                context_objective_weight=context_objective_weight,
                parameter_trust_radius=ridge_policy.parameter_trust_radius,
                effect_trust_radius=ridge_policy.effect_trust_radius,
                parameter_reference_floor=(
                    ridge_policy.parameter_reference_floor
                ),
                effect_reference_floor=ridge_policy.effect_reference_floor,
            ),
        )
        if not result.verify_authentication():
            raise Lfm2CpcfShortConvV5NumericalAdapterError(
                "Global matrix-free residual BCX result failed authentication."
            )
        candidate = _clone_state(state)
        candidate["in_proj_weight"] = result.in_proj_weight
        candidate["in_proj_bias"] = result.in_proj_bias
        return candidate, [str(result.diagnostics["report_sha256"])]
    result = solve_lfm2_cpcf_v5_residual_subspace_bcx(
        cases=cases,
        replays=current_replays,
        state=state,
        cap_id=cap_id,
        ridge_policy=ridge_policy,
        context_objective_weight=context_objective_weight,
    )
    if not result.verify_authentication():
        raise Lfm2CpcfShortConvV5NumericalAdapterError(
            "Scalable residual-subspace BCX result failed authentication."
        )
    candidate = _clone_state(state)
    candidate["in_proj_weight"] = result.in_proj_weight
    candidate["in_proj_bias"] = result.in_proj_bias
    return candidate, list(result.receipt_sha256)


def _accepted_phase_report(
    *,
    request: Lfm2CpcfShortConvNumericalFrontierRequest,
    cap_id: str,
    pass_index: int,
    phase: str,
    solver_intercept_contract: str,
    solver_report_sha256: Sequence[str],
    solver_channel_count: int | None,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    accepted: bool,
) -> dict[str, Any]:
    return _seal_report(
        {
            "schema": LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_ADAPTER_SCHEMA,
            "kind": "fit_only_projection_phase_acceptance",
            "request_sha256": request.request_sha256,
            "condition_cap_id": cap_id,
            "pass_index": pass_index,
            "phase": phase,
            "solver_intercept_contract": solver_intercept_contract,
            "solver_report_sha256": list(solver_report_sha256),
            "solver_report_count": len(solver_report_sha256),
            "solver_report_scope": (
                "aggregate_all_target_fir_channels"
                if phase == "fir"
                else "solver_native"
            ),
            "solver_channel_count": solver_channel_count,
            "before_objective_report_sha256": before["report_sha256"],
            "after_objective_report_sha256": after["report_sha256"],
            "before_combined_fit_objective": before["combined_fit_objective"],
            "after_combined_fit_objective": after["combined_fit_objective"],
            "accepted": accepted,
            "fit_only_accept_revert": True,
            "fit_only": True,
            "selection_cases_consumed": 0,
            "validation_cases_consumed": 0,
            "execution": dict(_ZERO_EXECUTION),
        }
    )


def _objective_not_worse(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> bool:
    before_value = float(before["combined_fit_objective"])
    after_value = float(after["combined_fit_objective"])
    tolerance = 1e-12 * max(1.0, abs(before_value))
    return math.isfinite(after_value) and after_value <= before_value + tolerance


def _projection_path(
    request: Lfm2CpcfShortConvNumericalFrontierRequest,
    *,
    cap_id: str,
    initial_state: Mapping[str, Any | None],
    projection_cases: Sequence[Lfm2CpcfV4ProjectionCase],
    input_lift_report_sha256: str,
    basis_report_sha256: str,
    factor_ridge_policies: Mapping[str, Lfm2CpcfV5RidgePolicy],
) -> tuple[
    tuple[Lfm2CpcfV5AcceptedPathState, ...],
    tuple[dict[str, Any], ...],
]:
    state = _clone_state(initial_state)
    prior_id = (
        f"v5-{cap_id}{V5_PRIOR_CONTROL_SUFFIX}"
        if cap_id == _cap_id(V5_CONDITION_CAP_CANDIDATES[0])
        else f"v5-{cap_id}--anchor-00-transported"
    )
    prior_report = _seal_report(
        {
            "schema": LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_ADAPTER_SCHEMA,
            "kind": "cap_specific_transported_prior_acceptance",
            "request_sha256": request.request_sha256,
            "condition_cap_id": cap_id,
            "basis_report_sha256": basis_report_sha256,
            "input_lift_report_sha256": input_lift_report_sha256,
            "fit_only": True,
            "selection_cases_consumed": 0,
            "validation_cases_consumed": 0,
            "execution": dict(_ZERO_EXECUTION),
        }
    )
    path = [
        Lfm2CpcfV5AcceptedPathState(
            stage_id=prior_id,
            state=_present_state(state),
            acceptance_report_sha256=prior_report["report_sha256"],
        )
    ]
    phase_reports = [prior_report]
    current_replays = _replays(projection_cases, state)
    solvers = (
        ("writer", _solve_writer),
        ("fir", _solve_fir),
        ("bcx", _solve_bcx),
    )
    for pass_index in range(request.policy.variable_projection_passes):
        for phase, solver in solvers:
            before = _fit_objective(
                projection_cases,
                state,
                context_objective_weight=(request.policy.context_objective_weight),
                replays=current_replays,
            )
            solver_arguments = {
                "cap_id": cap_id,
                "ridge_policy": factor_ridge_policies[phase],
                "replays": current_replays,
            }
            if phase == "bcx":
                solver_arguments["context_objective_weight"] = (
                    request.policy.context_objective_weight
                )
            candidate, solver_receipts = solver(
                projection_cases,
                state,
                **solver_arguments,
            )
            candidate_replays = _replays(projection_cases, candidate)
            after = _fit_objective(
                projection_cases,
                candidate,
                context_objective_weight=(request.policy.context_objective_weight),
                replays=candidate_replays,
            )
            accepted = _objective_not_worse(before, after)
            solver_intercept_contract = {
                "writer": (
                    "separate_unpenalized_intercept"
                    if state["writer_bias"] is not None
                    else "absent_frozen_zero"
                ),
                "fir": (
                    "physical_bias_coordinate_in_design_with_"
                    "separate_intercept_absent_frozen_zero"
                ),
                "bcx": (
                    "physical_bias_coordinates_in_design_with_"
                    "separate_intercept_absent_frozen_zero"
                ),
            }[phase]
            report = _accepted_phase_report(
                request=request,
                cap_id=cap_id,
                pass_index=pass_index,
                phase=phase,
                solver_intercept_contract=solver_intercept_contract,
                solver_report_sha256=solver_receipts,
                solver_channel_count=(
                    int(state["fir_kernel"].shape[0])
                    if phase == "fir"
                    else None
                ),
                before=before,
                after=after,
                accepted=accepted,
            )
            phase_reports.append(report)
            if not accepted:
                continue
            state = candidate
            current_replays = candidate_replays
            stage_id = (
                f"v5-{cap_id}--anchor-{len(path):02d}-pass-{pass_index:02d}-{phase}"
            )
            path.append(
                Lfm2CpcfV5AcceptedPathState(
                    stage_id=stage_id,
                    state=_present_state(state),
                    acceptance_report_sha256=report["report_sha256"],
                    predecessor_authentication_sha256=(path[-1].authentication_sha256),
                )
            )
    if len(path) == 1:
        no_op_report = _seal_report(
            {
                "schema": LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_ADAPTER_SCHEMA,
                "kind": "fit_only_accepted_noop_terminal",
                "request_sha256": request.request_sha256,
                "condition_cap_id": cap_id,
                "reason": "all_projection_phase_candidates_reverted",
                "fit_state_unchanged": True,
                "fit_only": True,
                "selection_cases_consumed": 0,
                "validation_cases_consumed": 0,
                "execution": dict(_ZERO_EXECUTION),
            }
        )
        phase_reports.append(no_op_report)
        path.append(
            Lfm2CpcfV5AcceptedPathState(
                stage_id=f"v5-{cap_id}--anchor-01-accepted-noop",
                state=_present_state(state),
                acceptance_report_sha256=no_op_report["report_sha256"],
                predecessor_authentication_sha256=(path[-1].authentication_sha256),
            )
        )
    return tuple(path), tuple(phase_reports)


def _raw_candidate(
    request: Lfm2CpcfShortConvNumericalFrontierRequest,
    *,
    candidate_id: str,
    candidate_kind: str,
    state: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Lfm2CpcfShortConvNumericalRawCandidate:
    return Lfm2CpcfShortConvNumericalRawCandidate(
        request_sha256=request.request_sha256,
        candidate_id=candidate_id,
        candidate_kind=candidate_kind,
        in_proj_weight=state["in_proj_weight"],
        in_proj_bias=state.get("in_proj_bias"),
        fir_kernel=state["fir_kernel"],
        fir_bias=state.get("fir_bias"),
        writer_weight=state["writer_weight"],
        writer_bias=state.get("writer_bias"),
        method_provenance=provenance,
    )


@dataclass(frozen=True, slots=True)
class Lfm2CpcfShortConvV5NumericalFrontierAdapter:
    """Generate the real v5 raw numerical frontier for the frozen seam.

    The default FIR policy is deliberately strict.  A production caller may
    supply a looser *explicit authenticated* gate, but the adapter never
    estimates a tolerance from the same evidence and never silently bypasses a
    failed actual per-lag commutator residual.
    """

    basis_policy: Lfm2CpcfV5FirBasisPolicy = field(
        default_factory=Lfm2CpcfV5FirBasisPolicy
    )
    ridge_policy: Lfm2CpcfV5RidgePolicy = field(default_factory=Lfm2CpcfV5RidgePolicy)
    adapter_id: str = "neural-forge.lfm2-cpcf-short-conv-v5-real"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.basis_policy, Lfm2CpcfV5FirBasisPolicy)
            or not self.basis_policy.verify_authentication()
        ):
            raise TypeError(
                "basis_policy must be an authenticated Lfm2CpcfV5FirBasisPolicy."
            )
        if (
            not isinstance(self.ridge_policy, Lfm2CpcfV5RidgePolicy)
            or not self.ridge_policy.verify_authentication()
        ):
            raise TypeError(
                "ridge_policy must be an authenticated Lfm2CpcfV5RidgePolicy."
            )
        if not isinstance(self.adapter_id, str) or not self.adapter_id.strip():
            raise Lfm2CpcfShortConvV5NumericalAdapterError(
                "adapter_id must be nonempty text."
            )
        object.__setattr__(self, "adapter_id", self.adapter_id.strip())

    def generate(
        self,
        request: Lfm2CpcfShortConvNumericalFrontierRequest,
    ) -> Lfm2CpcfShortConvNumericalFrontier:
        """Generate a complete FIT-only frontier bound to ``request``."""

        if not isinstance(
            request,
            Lfm2CpcfShortConvNumericalFrontierRequest,
        ):
            raise TypeError(
                "request must be Lfm2CpcfShortConvNumericalFrontierRequest."
            )
        if not request.verify_authentication():
            raise Lfm2CpcfShortConvV5NumericalAdapterError(
                "Numerical-frontier request failed authentication."
            )
        source_hidden = int(request.report["source_hidden_size"])
        target_hidden = int(request.report["target_hidden_size"])
        if target_hidden > source_hidden:
            raise Lfm2CpcfShortConvV5NumericalAdapterError(
                "The FIR-commuting compression basis requires target hidden "
                "size not to exceed source hidden size."
            )
        resized_fir = _resize_fir_rows(
            request.source_tensors["source_fir_kernel"],
            int(request.report["target_fir_length"]),
        )
        descriptors = _basis_cases(request)
        basis = _build_audited_basis(
            request,
            descriptors=descriptors,
            resized_fir=resized_fir,
            target_hidden=target_hidden,
            requested_policy=self.basis_policy,
        )
        if (
            not basis.verify_authentication()
            or basis.report["full_source_numeric_participation"] is not True
        ):
            raise Lfm2CpcfShortConvV5NumericalAdapterError(
                "The FIR-commuting basis failed authentication or full "
                "source participation."
            )
        fhvr_arm_specs = (
            (
                FHVR_MF_CAP1K_SWEEPS2_CANDIDATE_ID,
                Lfm2CpcfFhvrMfPolicy(),
            ),
            (
                FHVR_MF_CAP10K_SWEEPS1_CANDIDATE_ID,
                Lfm2CpcfFhvrMfPolicy(
                    condition_cap=10_000.0,
                    factor_sweeps=1,
                ),
            ),
        )
        fhvr_candidates = tuple(
            build_lfm2_cpcf_fhvr_mf_candidate(
                request,
                context_encoder=basis.encoder,
                policy=policy,
                candidate_id=candidate_id,
            )
            for candidate_id, policy in fhvr_arm_specs
        )
        for (candidate_id, policy), candidate in zip(
            fhvr_arm_specs,
            fhvr_candidates,
            strict=True,
        ):
            if (
                not candidate.verify_authentication()
                or candidate.candidate_id != candidate_id
                or candidate.candidate_kind != FHVR_MF_CANDIDATE_KIND
                or dict(candidate.method_provenance.get("policy", {}))
                != policy.as_dict()
            ):
                raise Lfm2CpcfShortConvV5NumericalAdapterError(
                    "An FHVR-MF policy arm failed authentication, identity, "
                    "or policy binding."
                )
        legacy_fhvr_candidate = fhvr_candidates[0]
        if legacy_fhvr_candidate.candidate_id != FHVR_MF_CANDIDATE_ID:
            raise Lfm2CpcfShortConvV5NumericalAdapterError(
                "The legacy FHVR-MF candidate id changed policy-arm identity."
            )
        fhvr_candidate_inventory = [
            {
                "arm_index": arm_index,
                "candidate_id": candidate.candidate_id,
                "candidate_kind": candidate.candidate_kind,
                "condition_cap": policy.condition_cap,
                "factor_sweeps": policy.factor_sweeps,
                "pcg_iterations": policy.pcg_iterations,
                "policy_sha256": policy.authentication_sha256,
                "candidate_state_sha256": candidate.state_sha256,
                "candidate_report_sha256": candidate.report["report_sha256"],
            }
            for arm_index, ((_, policy), candidate) in enumerate(
                zip(fhvr_arm_specs, fhvr_candidates, strict=True)
            )
        ]
        lift_frontier = _input_lift_frontier(
            request,
            prior_weight=basis.decoder,
            ridge_policy=self.ridge_policy,
        )
        factor_ridge_policies = {
            "writer": _factor_ridge_policy(
                self.ridge_policy,
                parameter_trust_radius=request.policy.writer_trust_radius,
            ),
            "fir": _factor_ridge_policy(
                self.ridge_policy,
                parameter_trust_radius=request.policy.fir_trust_radius,
            ),
            "bcx": _factor_ridge_policy(
                self.ridge_policy,
                parameter_trust_radius=request.policy.bcx_trust_radius,
            ),
        }
        projection_cases = _projection_cases(
            request,
            context_encoder=basis.encoder,
        )
        raw_candidates = []
        cap_inventory = {}
        for condition_cap in V5_CONDITION_CAP_CANDIDATES:
            cap_id = _cap_id(condition_cap)
            lift = lift_frontier[cap_id]
            if not lift.verify_authentication():
                raise Lfm2CpcfShortConvV5NumericalAdapterError(
                    f"{cap_id} input lift failed authentication."
                )
            prior = _transport_prior(
                request,
                encoder=basis.encoder,
                decoder=basis.decoder,
                target_fir_kernel=basis.target_fir_kernel,
                input_lift=lift.weight,
                input_lift_bias=lift.bias,
            )
            path, phase_reports = _projection_path(
                request,
                cap_id=cap_id,
                initial_state=prior,
                projection_cases=projection_cases,
                input_lift_report_sha256=(lift.diagnostics["report_sha256"]),
                basis_report_sha256=basis.report["report_sha256"],
                factor_ridge_policies=factor_ridge_policies,
            )
            interpolation = interpolate_every_lfm2_cpcf_v5_accepted_edge(path)
            if not interpolation.verify_authentication():
                raise Lfm2CpcfShortConvV5NumericalAdapterError(
                    f"{cap_id} accepted-edge frontier failed authentication."
                )
            for path_index, path_state in enumerate(path):
                raw_candidates.append(
                    _raw_candidate(
                        request,
                        candidate_id=path_state.stage_id,
                        candidate_kind=(
                            "cap_specific_transported_prior_control"
                            if path_index == 0
                            else "accepted_fit_projection_path_anchor"
                        ),
                        state=path_state.state,
                        provenance={
                            "method": ("v5_condition_capped_accepted_fit_path"),
                            "condition_cap_id": cap_id,
                            "condition_cap": condition_cap,
                            "path_index": path_index,
                            "path_state_authentication_sha256": (
                                path_state.authentication_sha256
                            ),
                            "acceptance_report_sha256": (
                                path_state.acceptance_report_sha256
                            ),
                            "predecessor_authentication_sha256": (
                                path_state.predecessor_authentication_sha256
                            ),
                            "fit_only": True,
                            "selection_cases_consumed": 0,
                            "validation_cases_consumed": 0,
                        },
                    )
                )
            for interpolation_index, interpolated in enumerate(
                interpolation.candidates
            ):
                raw_candidates.append(
                    _raw_candidate(
                        request,
                        candidate_id=(
                            f"v5-{cap_id}--interior-"
                            f"{interpolation_index:04d}-"
                            f"{interpolated.stage_id}"
                        ),
                        candidate_kind=("every_accepted_edge_interpolation"),
                        state=interpolated.state,
                        provenance={
                            "method": ("v5_complete_per_accepted_edge_interpolation"),
                            "condition_cap_id": cap_id,
                            "condition_cap": condition_cap,
                            "interpolation_index": interpolation_index,
                            "interpolation_report_sha256": (
                                interpolated.report["report_sha256"]
                            ),
                            "accepted_edge_frontier_report_sha256": (
                                interpolation.report["report_sha256"]
                            ),
                            "fit_only": True,
                            "selection_cases_consumed": 0,
                            "validation_cases_consumed": 0,
                        },
                    )
                )
            phase_attempts = [
                report
                for report in phase_reports
                if report["kind"] == "fit_only_projection_phase_acceptance"
            ]
            cap_inventory[cap_id] = {
                "condition_cap": condition_cap,
                "input_lift_report_sha256": (lift.diagnostics["report_sha256"]),
                "accepted_path_state_count": len(path),
                "accepted_path_authentication_sha256": [
                    state.authentication_sha256 for state in path
                ],
                "phase_attempt_count": len(phase_attempts),
                "phase_accepted_count": len(
                    [report for report in phase_attempts if report["accepted"]]
                ),
                "phase_inventory": [
                    {
                        "pass_index": report["pass_index"],
                        "phase": report["phase"],
                        "solver_intercept_contract": report[
                            "solver_intercept_contract"
                        ],
                        "solver_report_count": report["solver_report_count"],
                        "solver_report_scope": report["solver_report_scope"],
                        "solver_channel_count": report["solver_channel_count"],
                        "before_combined_fit_objective": report[
                            "before_combined_fit_objective"
                        ],
                        "after_combined_fit_objective": report[
                            "after_combined_fit_objective"
                        ],
                        "accepted": report["accepted"],
                        "report_sha256": report["report_sha256"],
                    }
                    for report in phase_attempts
                ],
                "phase_acceptance_report_sha256": [
                    report["report_sha256"] for report in phase_reports
                ],
                "accepted_edge_count": interpolation.report["accepted_edge_count"],
                "interpolated_candidate_count": len(interpolation.candidates),
                "interpolation_frontier_report_sha256": (
                    interpolation.report["report_sha256"]
                ),
            }
        writer_controls = tuple(
            candidate
            for candidate in raw_candidates
            if candidate.candidate_kind == "accepted_fit_projection_path_anchor"
            and candidate.candidate_id.endswith("-writer")
        )
        if not writer_controls:
            raise Lfm2CpcfShortConvV5NumericalAdapterError(
                "The full-coverage FHVR hybrids require at least one "
                "FIT-accepted writer control."
            )
        writer_control_fit_reports = {
            candidate.candidate_id: _fit_objective(
                projection_cases,
                candidate.state_dict(),
                context_objective_weight=(
                    request.policy.context_objective_weight
                ),
            )
            for candidate in writer_controls
        }
        if any(
            not math.isfinite(
                float(report["mean_case_endpoint_relative_rmse"])
            )
            or not math.isfinite(float(report["combined_fit_objective"]))
            for report in writer_control_fit_reports.values()
        ):
            raise Lfm2CpcfShortConvV5NumericalAdapterError(
                "A FIT-only writer-control objective is non-finite."
            )
        writer_control_ranking = sorted(
            (
                {
                    "candidate_id": candidate.candidate_id,
                    "candidate_kind": candidate.candidate_kind,
                    "candidate_state_sha256": candidate.state_sha256,
                    "candidate_report_sha256": (
                        candidate.report["report_sha256"]
                    ),
                    "fit_objective_report_sha256": (
                        writer_control_fit_reports[
                            candidate.candidate_id
                        ]["report_sha256"]
                    ),
                    "mean_case_endpoint_relative_rmse": (
                        writer_control_fit_reports[
                            candidate.candidate_id
                        ]["mean_case_endpoint_relative_rmse"]
                    ),
                    "combined_fit_objective": (
                        writer_control_fit_reports[
                            candidate.candidate_id
                        ]["combined_fit_objective"]
                    ),
                }
                for candidate in writer_controls
            ),
            key=lambda value: (
                value["mean_case_endpoint_relative_rmse"],
                value["combined_fit_objective"],
                value["candidate_id"],
            ),
        )
        strongest_writer_control_id = str(
            writer_control_ranking[0]["candidate_id"]
        )
        strongest_writer_control = next(
            candidate
            for candidate in writer_controls
            if candidate.candidate_id == strongest_writer_control_id
        )
        writer_control_fit_selection = _seal_report(
            {
                "schema": (
                    LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_ADAPTER_SCHEMA
                ),
                "kind": "fit_only_strongest_accepted_writer_control",
                "request_sha256": request.request_sha256,
                "eligible_candidate_count": len(writer_controls),
                "ranking": writer_control_ranking,
                "ranking_key": [
                    "mean_case_endpoint_relative_rmse",
                    "combined_fit_objective",
                    "candidate_id",
                ],
                "selected_candidate_id": strongest_writer_control_id,
                "selected_candidate_state_sha256": (
                    strongest_writer_control.state_sha256
                ),
                "selected_candidate_report_sha256": (
                    strongest_writer_control.report["report_sha256"]
                ),
                "fit_only": True,
                "selection_cases_consumed": 0,
                "validation_cases_consumed": 0,
                "execution": dict(_ZERO_EXECUTION),
            }
        )
        hybrid_arm_specs = (
            (
                fhvr_candidates[0],
                fhvr_arm_specs[0][1],
                FHVR_MF_CAP1K_SWEEPS2_FULL_COVERAGE_HYBRID_ID,
            ),
            (
                fhvr_candidates[1],
                fhvr_arm_specs[1][1],
                FHVR_MF_CAP10K_SWEEPS1_FULL_COVERAGE_HYBRID_ID,
            ),
        )
        hybrid_candidates = tuple(
            build_lfm2_cpcf_fhvr_writer_full_coverage_hybrid_candidate(
                request,
                context_encoder=basis.encoder,
                writer_candidate=strongest_writer_control,
                fhvr_candidate=fhvr_candidate,
                policy=policy,
                candidate_id=hybrid_id,
            )
            for fhvr_candidate, policy, hybrid_id in hybrid_arm_specs
        )
        fhvr_hybrid_inventory = []
        for (
            source_fhvr,
            policy,
            hybrid_id,
        ), hybrid_candidate in zip(
            hybrid_arm_specs,
            hybrid_candidates,
            strict=True,
        ):
            hybrid_provenance = hybrid_candidate.method_provenance
            coverage = hybrid_provenance.get("full_coverage", {})
            if (
                not hybrid_candidate.verify_authentication()
                or hybrid_candidate.request_sha256 != request.request_sha256
                or hybrid_candidate.candidate_id != hybrid_id
                or hybrid_candidate.candidate_kind
                != FHVR_MF_FULL_COVERAGE_HYBRID_KIND
                or hybrid_provenance.get("source_writer_candidate_id")
                != strongest_writer_control_id
                or hybrid_provenance.get("source_fhvr_candidate_id")
                != source_fhvr.candidate_id
                or dict(hybrid_provenance.get("policy", {}))
                != policy.as_dict()
                or hybrid_provenance.get("fixed_fhvr_fraction")
                != FHVR_MF_FULL_COVERAGE_FHVR_FRACTION
                or hybrid_provenance.get("fixed_writer_fraction")
                != FHVR_MF_FULL_COVERAGE_WRITER_FRACTION
                or hybrid_provenance.get("fixed_refactor_sweeps")
                != FHVR_MF_FULL_COVERAGE_REFACTOR_SWEEPS
                or coverage.get("writer_channels_covered") != target_hidden
                or coverage.get("fhvr_channels_covered") != target_hidden
                or coverage.get("source_channel_tuples_assigned")
                != 2 * target_hidden
                or coverage.get(
                    "each_source_channel_assigned_exactly_once"
                )
                is not True
                or coverage.get("both_source_masses_strictly_positive")
                is not True
                or coverage.get("source_channels_dropped") != 0
                or coverage.get("source_channels_pruned") != 0
                or hybrid_provenance.get("fit_only") is not True
                or hybrid_provenance.get("selection_cases_consumed") != 0
                or hybrid_provenance.get("validation_cases_consumed") != 0
            ):
                raise Lfm2CpcfShortConvV5NumericalAdapterError(
                    "An FHVR full-coverage hybrid failed authentication, "
                    "source binding, fixed-policy binding, or coverage."
                )
            synthetic_evidence = hybrid_provenance.get(
                "synthetic_evidence",
                {},
            )
            if (
                synthetic_evidence.get("total_arm_seed_losses") != 2
                or synthetic_evidence.get("total_arm_seed_trials") != 10
                or synthetic_evidence.get("losses_disclosed") is not True
            ):
                raise Lfm2CpcfShortConvV5NumericalAdapterError(
                    "FHVR hybrid synthetic evidence must disclose all losses."
                )
            fhvr_hybrid_inventory.append(
                {
                    "candidate_id": hybrid_candidate.candidate_id,
                    "candidate_kind": hybrid_candidate.candidate_kind,
                    "candidate_state_sha256": (
                        hybrid_candidate.state_sha256
                    ),
                    "candidate_report_sha256": (
                        hybrid_candidate.report["report_sha256"]
                    ),
                    "source_fhvr_candidate_id": source_fhvr.candidate_id,
                    "source_fhvr_candidate_state_sha256": (
                        source_fhvr.state_sha256
                    ),
                    "source_writer_candidate_id": (
                        strongest_writer_control_id
                    ),
                    "source_writer_candidate_state_sha256": (
                        strongest_writer_control.state_sha256
                    ),
                    "policy_sha256": policy.authentication_sha256,
                    "writer_channels_covered": coverage[
                        "writer_channels_covered"
                    ],
                    "fhvr_channels_covered": coverage[
                        "fhvr_channels_covered"
                    ],
                    "source_channels_dropped": 0,
                    "source_channels_pruned": 0,
                    "fixed_fhvr_fraction": (
                        FHVR_MF_FULL_COVERAGE_FHVR_FRACTION
                    ),
                    "fixed_writer_fraction": (
                        FHVR_MF_FULL_COVERAGE_WRITER_FRACTION
                    ),
                    "fixed_refactor_sweeps": (
                        FHVR_MF_FULL_COVERAGE_REFACTOR_SWEEPS
                    ),
                    "total_arm_seed_losses": 2,
                    "total_arm_seed_trials": 10,
                    "losses_disclosed": True,
                }
            )
        raw_candidates.extend(fhvr_candidates)
        raw_candidates.extend(hybrid_candidates)
        prior_controls = [
            candidate.candidate_id
            for candidate in raw_candidates
            if candidate.candidate_id.endswith(V5_PRIOR_CONTROL_SUFFIX)
        ]
        if len(prior_controls) != 1:
            raise Lfm2CpcfShortConvV5NumericalAdapterError(
                "The raw frontier must contain exactly one v5 prior-control "
                "candidate id."
            )
        frontier = Lfm2CpcfShortConvNumericalFrontier(
            adapter_id=self.adapter_id,
            request_sha256=request.request_sha256,
            context_encoder=basis.encoder,
            raw_candidates=tuple(raw_candidates),
            fit_case_authentication_sha256=tuple(
                request.report["fit_case_authentication_sha256"]
            ),
            adapter_provenance={
                "implementation": _ADAPTER_IMPLEMENTATION,
                "real_v5_numerical_algorithm": True,
                "dynamic_source_target_topology": True,
                "dependency_order": [
                    "resize_source_fir",
                    "case_balanced_activation_endpoint_descriptors",
                    "commutator_gated_all_channel_basis",
                    "future_hankel_volterra_matrix_free_two_policy_arm_frontier",
                    "condition_capped_input_lift",
                    "transported_parent_priors",
                    "fit_only_projection_cases",
                    "condition_capped_writer_batched_all_channel_fir_"
                    "coupled_residual_bcx_solves",
                    "accepted_path_chaining",
                    "every_accepted_edge_interpolation",
                    "fit_only_strongest_accepted_writer_control",
                    "full_coverage_fhvr_writer_response_refactor",
                    "typed_raw_frontier",
                ],
                "source_hidden_size": source_hidden,
                "target_hidden_size": target_hidden,
                "source_fir_length": request.report["source_fir_length"],
                "target_fir_length": request.report["target_fir_length"],
                "source_fir_resized_before_basis": True,
                "basis_policy_sha256": (self.basis_policy.authentication_sha256),
                "ridge_policy_sha256": (self.ridge_policy.authentication_sha256),
                "factor_ridge_policy_sha256": {
                    phase: policy.authentication_sha256
                    for phase, policy in factor_ridge_policies.items()
                },
                "factor_parameter_trust_radius": {
                    phase: policy.parameter_trust_radius
                    for phase, policy in factor_ridge_policies.items()
                },
                "fir_solver_backend": (
                    "authenticated_cpu_fp64_batched_all_channel_ridge"
                ),
                "fir_solver_receipt_scope": "aggregate_all_target_fir_channels",
                "fir_scalar_oracle_policy": (
                    "deterministic_first_middle_last_plus_"
                    "threshold_ambiguous_fallback"
                ),
                "bcx_solver_backend_inventory": (
                    {
                        _cap_id(V5_CONDITION_CAP_CANDIDATES[0]): (
                            "global_full_coordinate_matrix_free_residual"
                        ),
                        _cap_id(V5_CONDITION_CAP_CANDIDATES[1]): (
                            "coupled_endpoint_residual_subspace_coordinate"
                        ),
                    }
                    if target_hidden >= _SCALABLE_BCX_MINIMUM_HIDDEN_SIZE
                    else {
                        _cap_id(cap): "legacy_full_width_context_only_oracle"
                        for cap in V5_CONDITION_CAP_CANDIDATES
                    }
                ),
                "bcx_scalable_hidden_threshold": (
                    _SCALABLE_BCX_MINIMUM_HIDDEN_SIZE
                ),
                "bcx_dense_transported_prior_preserved": True,
                "bcx_endpoint_residual_aware": (
                    target_hidden >= _SCALABLE_BCX_MINIMUM_HIDDEN_SIZE
                ),
                "bcx_writer_cross_channel_coupling": (
                    [
                        "global_implicit_writer_jacobian",
                        "deterministic_coordinate_residual_refresh",
                    ]
                    if target_hidden >= _SCALABLE_BCX_MINIMUM_HIDDEN_SIZE
                    else ["not_applicable_to_legacy_small_width_oracle"]
                ),
                "causal_horizon_phase_acceptance": {
                    "uniform_mass": 0.5,
                    "exponential_tail_mass": 0.5,
                    "tail_tau": "max(4,sqrt(valid_rows))",
                    "uniform_reference_metric_retained": True,
                },
                "basis_report_sha256": basis.report["report_sha256"],
                "basis_actual_commutator_gate": (
                    basis.report["maximum_per_lag_commutator_relative_frobenius_gate"]
                ),
                "basis_actual_commutator_residual": (
                    basis.report["maximum_per_lag_commutator_relative_frobenius"]
                ),
                "basis_actual_commutator_gate_passed": (
                    basis.report[
                        "maximum_per_lag_commutator_relative_frobenius_gate_passed"
                    ]
                ),
                "basis_full_source_numeric_participation": (
                    basis.report["full_source_numeric_participation"]
                ),
                "source_channels_dropped": 0,
                "source_channels_pruned": 0,
                "source_channels_quantized": 0,
                "fit_descriptor_case_count": len(descriptors),
                "fit_projection_case_count": len(projection_cases),
                "fhvr_mf_candidate_included": True,
                "fhvr_mf_candidate_id": legacy_fhvr_candidate.candidate_id,
                "fhvr_mf_candidate_kind": legacy_fhvr_candidate.candidate_kind,
                "fhvr_mf_candidate_report_sha256": (
                    legacy_fhvr_candidate.report["report_sha256"]
                ),
                "fhvr_mf_candidate_count": len(fhvr_candidates),
                "fhvr_mf_candidate_ids": [
                    candidate.candidate_id for candidate in fhvr_candidates
                ],
                "fhvr_mf_candidate_inventory": fhvr_candidate_inventory,
                "fhvr_mf_selector_diversity_axis": "condition_cap",
                "fhvr_mf_full_coverage_hybrid_count": len(
                    hybrid_candidates
                ),
                "fhvr_mf_full_coverage_hybrid_ids": [
                    candidate.candidate_id
                    for candidate in hybrid_candidates
                ],
                "fhvr_mf_full_coverage_hybrid_kind": (
                    FHVR_MF_FULL_COVERAGE_HYBRID_KIND
                ),
                "fhvr_mf_full_coverage_hybrid_inventory": (
                    fhvr_hybrid_inventory
                ),
                "fhvr_mf_full_coverage_fhvr_fraction": (
                    FHVR_MF_FULL_COVERAGE_FHVR_FRACTION
                ),
                "fhvr_mf_full_coverage_writer_fraction": (
                    FHVR_MF_FULL_COVERAGE_WRITER_FRACTION
                ),
                "fhvr_mf_full_coverage_refactor_sweeps": (
                    FHVR_MF_FULL_COVERAGE_REFACTOR_SWEEPS
                ),
                "fhvr_mf_hybrid_writer_control_fit_selection": (
                    writer_control_fit_selection
                ),
                "fhvr_mf_hybrid_real_selection_deferred_to_compiler": True,
                "fhvr_mf_hybrid_candidate_specific_fit_mlp_deferred_to_compiler": (
                    True
                ),
                "fhvr_mf_hybrid_synthetic_evidence": {
                    "fhvr_fractions_tested": [0.25, 0.5, 0.75],
                    "retained_fhvr_fraction": (
                        FHVR_MF_FULL_COVERAGE_FHVR_FRACTION
                    ),
                    "rejected_fhvr_fractions": [0.25, 0.5],
                    "total_arm_seed_losses": 2,
                    "total_arm_seed_trials": 10,
                    "losses_disclosed": True,
                },
                "condition_cap_inventory": cap_inventory,
                "prior_control_candidate_id": prior_controls[0],
                "prior_control_candidate_count": 1,
                "raw_candidate_count": len(raw_candidates),
                "fit_only": True,
                "selection_cases_consumed": 0,
                "validation_cases_consumed": 0,
                "selection_performed_by_adapter": False,
                "validation_performed_by_adapter": False,
                "model_forward_calls": 0,
                "training": False,
                "quantization": False,
                "pruning": False,
                "execution": dict(_ZERO_EXECUTION),
            },
        )
        if (
            not request.verify_authentication()
            or not frontier.verify_authentication()
            or not frontier.full_source_numeric_participation
        ):
            raise Lfm2CpcfShortConvV5NumericalAdapterError(
                "The generated frontier or its detached request failed final "
                "authentication."
            )
        return frontier


__all__ = [
    "LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_ADAPTER_SCHEMA",
    "Lfm2CpcfShortConvV5NumericalAdapterError",
    "Lfm2CpcfShortConvV5NumericalFrontierAdapter",
]
