"""Authenticated decoder-only bridge for normalized LFM2 MLP inputs.

The compiler in this module stops before any source-neuron signature,
clustering, medoid selection, or producer/consumer materialization.  It
extracts the shared normalized-input decoder construction used by the older
real-medoid compiler so nonlinear compilers such as Q-DEIM can reuse the
bridge without paying for an unrelated neuron plan.

Numeric quantities in the receipt describe structural construction and
authentication.  They have no authority to select or reject decoded
behavior.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    Lfm2CachedNormalizedInputMlpFoldCase,
    lfm2_authenticated_tensor_sha256,
    lfm2_normalized_input_mlp_fit_evidence_sha256,
)


LFM2_NORMALIZED_INPUT_DECODER_SCHEMA = (
    "neural-forge-lfm2-normalized-input-decoder-only-v1"
)


class Lfm2NormalizedInputDecoderError(ValueError):
    """Authenticated evidence or a structural decoder invariant is invalid."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "PyTorch is required for normalized-input decoder compilation."
        ) from exc
    return torch


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    if isinstance(value, list):
        return [_deep_thaw(item) for item in value]
    return value


def _valid_sha256_text(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2NormalizedInputDecoderError(
            f"{name} must be finite and non-negative."
        )
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2NormalizedInputDecoderError(
            f"{name} must be finite and non-negative."
        ) from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise Lfm2NormalizedInputDecoderError(
            f"{name} must be finite and non-negative."
        )
    return parsed


def _float_tensor(value: Any, name: str, *, ndim: int) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != ndim
        or any(int(size) <= 0 for size in value.shape)
        or not value.is_floating_point()
        or value.is_complex()
    ):
        raise Lfm2NormalizedInputDecoderError(
            f"{name} must be a non-empty materialized real rank-{ndim} tensor."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2NormalizedInputDecoderError(f"{name} contains NaN or infinity.")
    return result


def _relative_rmse(prediction: Any, target: Any) -> float:
    torch = _torch()
    error = float(torch.sqrt(torch.mean((prediction - target).square())).item())
    scale = float(torch.sqrt(torch.mean(target.square())).item())
    return error / max(scale, torch.finfo(torch.float64).eps)


def _parse_case_sample_weights(
    cases: Sequence[Lfm2CachedNormalizedInputMlpFoldCase],
    case_sample_weights: Sequence[float] | None,
) -> tuple[float, ...]:
    if case_sample_weights is None:
        return tuple(1.0 for _ in cases)
    if isinstance(case_sample_weights, (str, bytes)) or len(case_sample_weights) != len(
        cases
    ):
        raise Lfm2NormalizedInputDecoderError(
            "case_sample_weights must contain one value per FIT case."
        )
    parsed_weights = []
    for case_index, value in enumerate(case_sample_weights):
        if isinstance(value, bool):
            raise Lfm2NormalizedInputDecoderError(
                "case_sample_weights must contain positive finite values."
            )
        try:
            weight = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise Lfm2NormalizedInputDecoderError(
                f"case_sample_weights[{case_index}] is not finite."
            ) from exc
        if not math.isfinite(weight) or weight <= 0.0:
            raise Lfm2NormalizedInputDecoderError(
                "case_sample_weights must contain positive finite values."
            )
        parsed_weights.append(weight)
    return tuple(parsed_weights)


@dataclass(frozen=True, slots=True)
class Lfm2CompiledNormalizedInputDecoder:
    """Detached CPU-FP64 target-to-parent normalized-input decoder."""

    normalized_input_decoder: Any = field(repr=False, compare=False)
    normalized_input_decoder_sha256: str
    source_hidden_size: int
    target_hidden_size: int
    normalized_input_decoder_rank: int
    receipt_sha256: str
    report: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return _deep_thaw(self.report)

    def verify(self) -> Mapping[str, Any]:
        """Re-authenticate the materialized decoder and self-sealed receipt."""

        report = self.as_dict()
        claimed_receipt = report.pop("receipt_sha256", None)
        actual_decoder_sha256 = lfm2_authenticated_tensor_sha256(
            self.normalized_input_decoder
        )
        actual_receipt_sha256 = canonical_sha256(report)
        checks = {
            "decoder_sha256_matches": (
                actual_decoder_sha256 == self.normalized_input_decoder_sha256
            ),
            "report_decoder_sha256_matches": (
                report.get("normalized_input_decoder_sha256")
                == self.normalized_input_decoder_sha256
            ),
            "receipt_sha256_matches": (
                claimed_receipt == self.receipt_sha256 == actual_receipt_sha256
            ),
            "detached_cpu_fp64": (
                self.normalized_input_decoder.device.type == "cpu"
                and str(self.normalized_input_decoder.dtype) == "torch.float64"
                and not self.normalized_input_decoder.requires_grad
                and self.normalized_input_decoder.grad_fn is None
            ),
        }
        return MappingProxyType(
            {
                **checks,
                "verified": all(checks.values()),
                "semantic_acceptance_authority": False,
            }
        )


def compile_lfm2_normalized_input_decoder(
    *,
    residual_chart: Any,
    residual_chart_sha256: str,
    parent_ffn_norm_weight: Any,
    parent_ffn_norm_weight_sha256: str,
    target_ffn_norm_weight: Any,
    target_ffn_norm_weight_sha256: str,
    cases: Sequence[Lfm2CachedNormalizedInputMlpFoldCase],
    case_sample_weights: Sequence[float] | None = None,
    normalized_decoder_ridge: float = 1e-8,
) -> Lfm2CompiledNormalizedInputDecoder:
    """Compile only the gamma-aware normalized-input target-to-parent decoder.

    ``residual_chart`` has shape ``[source_hidden, target_hidden]``.  The
    returned decoder has shape ``[target_hidden, source_hidden]`` and is the
    exact same structural inverse, positive-gain fit, FIT-rowspace correction,
    and bounded null completion used by the normalized real-medoid compiler.
    No source MLP weight, target template value, cluster, or medoid is read.
    """

    torch = _torch()
    if not cases or not all(
        isinstance(value, Lfm2CachedNormalizedInputMlpFoldCase) for value in cases
    ):
        raise Lfm2NormalizedInputDecoderError(
            "cases must contain at least one Lfm2CachedNormalizedInputMlpFoldCase."
        )
    normalized_decoder_ridge = _nonnegative_float(
        normalized_decoder_ridge,
        "normalized_decoder_ridge",
    )
    normalized_case_weights = _parse_case_sample_weights(
        cases,
        case_sample_weights,
    )

    authenticated_inputs = (
        (residual_chart, residual_chart_sha256, "residual_chart"),
        (
            parent_ffn_norm_weight,
            parent_ffn_norm_weight_sha256,
            "parent_ffn_norm_weight",
        ),
        (
            target_ffn_norm_weight,
            target_ffn_norm_weight_sha256,
            "target_ffn_norm_weight",
        ),
    )
    for tensor, expected_sha256, name in authenticated_inputs:
        if not _valid_sha256_text(expected_sha256):
            raise Lfm2NormalizedInputDecoderError(
                f"{name}_sha256 must be a lowercase SHA-256 digest."
            )
        try:
            actual_sha256 = lfm2_authenticated_tensor_sha256(tensor)
        except (TypeError, RuntimeError, ValueError) as exc:
            raise Lfm2NormalizedInputDecoderError(
                f"{name} must be an authenticated materialized tensor."
            ) from exc
        if actual_sha256 != expected_sha256:
            raise Lfm2NormalizedInputDecoderError(
                f"{name} tensor authentication failed."
            )

    chart = _float_tensor(residual_chart, "residual_chart", ndim=2)
    parent_gamma = _float_tensor(
        parent_ffn_norm_weight,
        "parent_ffn_norm_weight",
        ndim=1,
    )
    target_gamma = _float_tensor(
        target_ffn_norm_weight,
        "target_ffn_norm_weight",
        ndim=1,
    )
    source_hidden, target_hidden = map(int, chart.shape)
    if tuple(parent_gamma.shape) != (source_hidden,) or tuple(target_gamma.shape) != (
        target_hidden,
    ):
        raise Lfm2NormalizedInputDecoderError(
            "Authenticated parent/target ffn_norm weights do not match the "
            "source/target hidden widths."
        )
    if target_hidden > source_hidden:
        raise Lfm2NormalizedInputDecoderError(
            "The normalized-input decoder cannot expand target hidden width "
            "beyond the parent hidden width."
        )
    if bool((parent_gamma == 0.0).any().item()) or bool(
        (target_gamma == 0.0).any().item()
    ):
        raise Lfm2NormalizedInputDecoderError(
            "Normalized-input chart construction requires nonzero ffn_norm weights."
        )

    with torch.no_grad():
        chart_identity = torch.eye(target_hidden, dtype=torch.float64)
        chart_gram = (chart.T @ chart).contiguous()
        chart_orthogonality_error = float(
            (chart_gram - chart_identity).abs().max().item()
        )
        chart_is_orthonormal = bool(
            torch.allclose(
                chart_gram,
                chart_identity,
                rtol=1e-8,
                atol=1e-8,
            )
        )
        if chart_is_orthonormal:
            chart_rank = target_hidden
            raw_chart_decoder = chart.T.contiguous()
            raw_chart_decoder_method = "authenticated_orthonormal_chart_transpose"
            raw_chart_svd_calls = 0
            raw_chart_pseudoinverse_calls = 0
        else:
            raw_chart_u, raw_chart_singular, raw_chart_vh = torch.linalg.svd(
                chart,
                full_matrices=False,
            )
            singular_threshold = (
                max(source_hidden, target_hidden)
                * torch.finfo(torch.float64).eps
                * raw_chart_singular.max()
            )
            supported_singular = raw_chart_singular > singular_threshold
            chart_rank = int(supported_singular.sum().item())
            raw_chart_decoder = (
                (
                    raw_chart_vh.T[:, supported_singular]
                    * raw_chart_singular[supported_singular].reciprocal().unsqueeze(0)
                )
                @ raw_chart_u[:, supported_singular].T
            ).contiguous()
            raw_chart_decoder_method = "single_authenticated_thin_svd_pseudoinverse"
            raw_chart_svd_calls = 1
            raw_chart_pseudoinverse_calls = 0
        if chart_rank != target_hidden:
            raise Lfm2NormalizedInputDecoderError(
                "residual_chart must have full target-column rank."
            )

        normalized_chart = (
            chart * parent_gamma.reciprocal().unsqueeze(1) * target_gamma.unsqueeze(0)
        ).contiguous()
        if not bool(torch.isfinite(normalized_chart).all().item()):
            raise Lfm2NormalizedInputDecoderError(
                "Normalized-input chart contains non-finite coefficients."
            )
        normalized_chart_rank = int(torch.linalg.matrix_rank(normalized_chart).item())
        if normalized_chart_rank != target_hidden:
            raise Lfm2NormalizedInputDecoderError(
                "Normalized-input chart must have full target-column rank."
            )
        polar_u, _, polar_vh = torch.linalg.svd(
            normalized_chart,
            full_matrices=False,
        )
        normalized_polar = (polar_u @ polar_vh).contiguous()
        polar_orthogonality_error = float(
            torch.linalg.matrix_norm(
                normalized_polar.T @ normalized_polar
                - torch.eye(target_hidden, dtype=torch.float64),
                ord=2,
            ).item()
        )
        gamma_aware_structural_decoder = (
            target_gamma.reciprocal().unsqueeze(1)
            * raw_chart_decoder
            * parent_gamma.unsqueeze(0)
        ).contiguous()
        structural_decoder_prior_rank = int(
            torch.linalg.matrix_rank(gamma_aware_structural_decoder).item()
        )
        structural_decoder_prior_operator_norm = float(
            torch.linalg.matrix_norm(
                gamma_aware_structural_decoder,
                ord=2,
            ).item()
        )
        structural_decoder_left_inverse_error = float(
            torch.linalg.matrix_norm(
                gamma_aware_structural_decoder @ normalized_chart
                - torch.eye(target_hidden, dtype=torch.float64),
                ord=2,
            ).item()
        )
        if (
            structural_decoder_prior_rank != target_hidden
            or not math.isfinite(structural_decoder_prior_operator_norm)
            or structural_decoder_prior_operator_norm <= torch.finfo(torch.float64).tiny
            or not math.isfinite(structural_decoder_left_inverse_error)
            or structural_decoder_left_inverse_error > 1e-8
        ):
            raise Lfm2NormalizedInputDecoderError(
                "Gamma-aware normalized-input structural decoder is not a "
                "finite full-rank left inverse of the normalized chart."
            )
        unit_structural_decoder = (
            gamma_aware_structural_decoder / structural_decoder_prior_operator_norm
        ).contiguous()

        normalized_cases: list[dict[str, Any]] = []
        for case_index, case in enumerate(cases):
            prefix = f"cases[{case_index}]"
            target_input = _float_tensor(
                case.target_input,
                f"{prefix}.target_input",
                ndim=2,
            )
            parent_input = _float_tensor(
                case.parent_normalized_input,
                f"{prefix}.parent_normalized_input",
                ndim=2,
            )
            rows = int(target_input.shape[0])
            if tuple(target_input.shape) != (rows, target_hidden) or tuple(
                parent_input.shape
            ) != (rows, source_hidden):
                raise Lfm2NormalizedInputDecoderError(
                    f"{prefix} normalized input shapes do not match "
                    "source/target topology."
                )
            normalized_cases.append(
                {
                    "target_input": target_input,
                    "parent_input": parent_input,
                    "sample_weight": normalized_case_weights[case_index],
                }
            )

        row_weights = torch.cat(
            [
                torch.full(
                    (int(value["target_input"].shape[0]),),
                    float(value["sample_weight"]),
                    dtype=torch.float64,
                )
                for value in normalized_cases
            ],
            dim=0,
        )
        sqrt_row_weights = torch.sqrt(row_weights).unsqueeze(1)
        target_input_rows = torch.cat(
            [value["target_input"] for value in normalized_cases],
            dim=0,
        )
        parent_input_rows = torch.cat(
            [value["parent_input"] for value in normalized_cases],
            dim=0,
        )
        weighted_target_input = target_input_rows * sqrt_row_weights
        weighted_parent_input = parent_input_rows * sqrt_row_weights

        unit_structural_prediction = weighted_target_input @ unit_structural_decoder
        gain_denominator = float(unit_structural_prediction.square().sum().item())
        gain_numerator = float(
            (unit_structural_prediction * weighted_parent_input).sum().item()
        )
        if (
            not math.isfinite(gain_numerator)
            or not math.isfinite(gain_denominator)
            or gain_denominator <= torch.finfo(torch.float64).tiny
            or gain_numerator <= 0.0
        ):
            raise Lfm2NormalizedInputDecoderError(
                "FIT rows do not identify one strictly positive normalized-input "
                "structural gain."
            )
        global_gain = gain_numerator / gain_denominator
        structural_decoder = (unit_structural_decoder * global_gain).contiguous()

        design_u, design_s, design_vh = torch.linalg.svd(
            weighted_target_input,
            full_matrices=False,
        )
        if int(design_s.numel()) == 0 or float(design_s.max().item()) <= 0.0:
            raise Lfm2NormalizedInputDecoderError(
                "FIT normalized child inputs have zero rank."
            )
        design_tolerance = (
            max(weighted_target_input.shape)
            * torch.finfo(torch.float64).eps
            * float(design_s.max().item())
        )
        fit_rank = int((design_s > design_tolerance).sum().item())
        if fit_rank <= 0:
            raise Lfm2NormalizedInputDecoderError(
                "FIT normalized child inputs have zero numerical rank."
            )
        supported_u = design_u[:, :fit_rank]
        supported_s = design_s[:fit_rank]
        supported_vh = design_vh[:fit_rank]
        rowspace_projector = (supported_vh.T @ supported_vh).contiguous()
        design_energy_scale = float((supported_s.square().sum() / fit_rank).item())
        scaled_decoder_ridge = normalized_decoder_ridge * design_energy_scale
        residual = weighted_parent_input - weighted_target_input @ structural_decoder
        spectral_filter = supported_s / (supported_s.square() + scaled_decoder_ridge)
        correction = supported_vh.T @ (
            spectral_filter.unsqueeze(1) * (supported_u.T @ residual)
        )

        supported_relative_cap = 1.0
        supported_prior = rowspace_projector @ structural_decoder
        supported_prior_norm = float(torch.linalg.vector_norm(supported_prior).item())
        correction_norm = float(torch.linalg.vector_norm(correction).item())
        allowed_correction_norm = supported_relative_cap * supported_prior_norm
        correction_scale = 1.0
        if correction_norm > allowed_correction_norm:
            correction_scale = allowed_correction_norm / max(
                correction_norm,
                torch.finfo(torch.float64).tiny,
            )
            correction = correction * correction_scale
        correction = correction.contiguous()
        normalized_input_decoder = (structural_decoder + correction).contiguous()

        identity = torch.eye(target_hidden, dtype=torch.float64)
        null_projector = identity - rowspace_projector
        correction_rowspace_leakage = float(
            torch.linalg.vector_norm(null_projector @ correction).item()
        )
        null_completion = null_projector @ normalized_input_decoder
        expected_null_completion = null_projector @ structural_decoder
        null_completion_error = float(
            torch.linalg.vector_norm(null_completion - expected_null_completion).item()
        )
        null_completion_operator_norm = float(
            torch.linalg.matrix_norm(null_completion, ord=2).item()
        )
        decoder_rank = int(torch.linalg.matrix_rank(normalized_input_decoder).item())
        decoder_prediction = target_input_rows @ normalized_input_decoder
        normalized_input_fit_error = _relative_rmse(
            decoder_prediction * sqrt_row_weights,
            parent_input_rows * sqrt_row_weights,
        )

        numerical_scale = max(
            float(torch.linalg.vector_norm(normalized_input_decoder).item()),
            global_gain,
            1.0,
        )
        numerical_tolerance = 1e-10 * numerical_scale
        if (
            decoder_rank != target_hidden
            or polar_orthogonality_error > 1e-10 * max(1.0, math.sqrt(target_hidden))
            or correction_rowspace_leakage > numerical_tolerance
            or null_completion_error > numerical_tolerance
            or null_completion_operator_norm > global_gain + numerical_tolerance
        ):
            raise Lfm2NormalizedInputDecoderError(
                "Normalized-input decoder failed rank, polar, rowspace, or "
                "bounded-null-completion invariants."
            )

    normalized_input_decoder = (
        normalized_input_decoder.detach().cpu().contiguous().clone()
    )
    decoder_sha256 = lfm2_authenticated_tensor_sha256(normalized_input_decoder)
    fit_evidence_sha256 = lfm2_normalized_input_mlp_fit_evidence_sha256(
        cases,
        normalized_case_weights,
    )
    normalized_chart_sha256 = lfm2_authenticated_tensor_sha256(normalized_chart)
    normalized_polar_sha256 = lfm2_authenticated_tensor_sha256(normalized_polar)
    structural_decoder_sha256 = lfm2_authenticated_tensor_sha256(
        gamma_aware_structural_decoder
    )
    correction_sha256 = lfm2_authenticated_tensor_sha256(correction)
    rowspace_projector_sha256 = lfm2_authenticated_tensor_sha256(rowspace_projector)
    row_weights_sha256 = lfm2_authenticated_tensor_sha256(row_weights)
    producer_bridge = {
        "role": "normalized_mlp_input_target_to_parent",
        "construction": (
            "positive_gain_gamma_aware_raw_chart_left_inverse_plus_"
            "fit_rowspace_correction_with_bounded_structural_null_completion"
        ),
        "normalized_chart_sha256": normalized_chart_sha256,
        "thin_polar_diagnostic_sha256": normalized_polar_sha256,
        "structural_decoder_prior_sha256": structural_decoder_sha256,
        "decoder_sha256": decoder_sha256,
        "correction_sha256": correction_sha256,
        "rowspace_projector_sha256": rowspace_projector_sha256,
        "normalized_chart_rank": normalized_chart_rank,
        "fit_rank": fit_rank,
        "decoder_rank": decoder_rank,
        "structural_decoder_prior_rank": structural_decoder_prior_rank,
        "structural_decoder_prior_operator_norm": (
            structural_decoder_prior_operator_norm
        ),
        "structural_decoder_left_inverse_error": (
            structural_decoder_left_inverse_error
        ),
        "positive_global_gain": global_gain,
        "ridge_policy_value": normalized_decoder_ridge,
        "scaled_ridge": scaled_decoder_ridge,
        "supported_relative_cap_rho": supported_relative_cap,
        "correction_scale": correction_scale,
        "fit_relative_rmse": normalized_input_fit_error,
        "polar_orthogonality_error": polar_orthogonality_error,
        "correction_rowspace_leakage": correction_rowspace_leakage,
        "null_completion_error": null_completion_error,
        "null_completion_operator_norm": null_completion_operator_norm,
        "null_completion_operator_norm_bound": global_gain,
    }
    operation_audit = {
        "source_mlp_weight_tensor_reads": 0,
        "target_template_value_reads": 0,
        "weight_complete_signature_builds": 0,
        "activation_cluster_plan_builds": 0,
        "multi_anchor_cluster_plan_builds": 0,
        "medoid_selection_operations": 0,
        "source_neuron_assignment_operations": 0,
        "producer_materializations": 0,
        "consumer_materializations": 0,
        "model_forward_calls": 0,
        "heldout_rows_used_by_solver": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
    }
    body = {
        "schema": LFM2_NORMALIZED_INPUT_DECODER_SCHEMA,
        "source_hidden_size": source_hidden,
        "target_hidden_size": target_hidden,
        "residual_chart_sha256": residual_chart_sha256,
        "parent_ffn_norm_weight_sha256": parent_ffn_norm_weight_sha256,
        "target_ffn_norm_weight_sha256": target_ffn_norm_weight_sha256,
        "normalized_chart_sha256": normalized_chart_sha256,
        "normalized_input_decoder_sha256": decoder_sha256,
        "fit_evidence_sha256": fit_evidence_sha256,
        "fit_weighting": "sqrt_case_weight_repeated_per_case_row",
        "fit_case_count": len(cases),
        "fit_case_weight_sum": math.fsum(normalized_case_weights),
        "fit_row_count": int(row_weights.numel()),
        "fit_row_weight_sum": float(row_weights.sum().item()),
        "fit_row_weight_minimum": float(row_weights.min().item()),
        "fit_row_weight_maximum": float(row_weights.max().item()),
        "fit_row_weights_sha256": row_weights_sha256,
        "producer_bridge": producer_bridge,
        "residual_chart_decoder_method": raw_chart_decoder_method,
        "residual_chart_rank": chart_rank,
        "residual_chart_orthonormal_columns": chart_is_orthonormal,
        "residual_chart_orthogonality_maximum_absolute_error": (
            chart_orthogonality_error
        ),
        "residual_chart_svd_calls": raw_chart_svd_calls,
        "residual_chart_pseudoinverse_calls": raw_chart_pseudoinverse_calls,
        "operation_audit": operation_audit,
        "decoder_only_scope": True,
        "cluster_or_medoid_state_constructed": False,
        "random_operations": 0,
        "numeric_diagnostics_used_for_semantic_acceptance_or_veto": False,
        "whole_decoded_responses_are_the_semantic_judgment_surface": True,
    }
    receipt_sha256 = canonical_sha256(body)
    report = MappingProxyType({**body, "receipt_sha256": receipt_sha256})
    compiled = Lfm2CompiledNormalizedInputDecoder(
        normalized_input_decoder=normalized_input_decoder,
        normalized_input_decoder_sha256=decoder_sha256,
        source_hidden_size=source_hidden,
        target_hidden_size=target_hidden,
        normalized_input_decoder_rank=decoder_rank,
        receipt_sha256=receipt_sha256,
        report=report,
    )
    if compiled.verify()["verified"] is not True:
        raise Lfm2NormalizedInputDecoderError(
            "The normalized-input decoder receipt failed self-verification."
        )
    return compiled


__all__ = [
    "LFM2_NORMALIZED_INPUT_DECODER_SCHEMA",
    "Lfm2CompiledNormalizedInputDecoder",
    "Lfm2NormalizedInputDecoderError",
    "compile_lfm2_normalized_input_decoder",
]
