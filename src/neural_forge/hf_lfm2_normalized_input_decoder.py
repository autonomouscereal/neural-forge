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

import hashlib
import json
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


def _coordinate_completion_tensor_sha256(value: Any) -> str:
    """Return the dense-tensor digest bound by completion candidate receipts."""

    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise Lfm2NormalizedInputDecoderError(
            "raw_structural_decoder must be an authenticated materialized tensor."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    if tensor.is_conj():
        tensor = tensor.resolve_conj()
    if tensor.is_neg():
        tensor = tensor.resolve_neg()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(
        json.dumps(
            [int(size) for size in tensor.shape],
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _authenticated_completion_provenance(
    *,
    raw_structural_decoder: Any,
    raw_structural_decoder_sha256: str,
    coordinate_completion_receipt: Mapping[str, Any],
    coordinate_completion_receipt_sha256: str,
    target_hidden: int,
    source_hidden: int,
) -> dict[str, Any]:
    """Fail closed unless a generic completion receipt seals the exact map."""

    if not _valid_sha256_text(raw_structural_decoder_sha256):
        raise Lfm2NormalizedInputDecoderError(
            "raw_structural_decoder_sha256 must be a lowercase SHA-256 digest."
        )
    if not _valid_sha256_text(coordinate_completion_receipt_sha256):
        raise Lfm2NormalizedInputDecoderError(
            "coordinate_completion_receipt_sha256 must be a lowercase SHA-256 digest."
        )
    try:
        actual_decoder_sha256 = lfm2_authenticated_tensor_sha256(raw_structural_decoder)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise Lfm2NormalizedInputDecoderError(
            "raw_structural_decoder must be an authenticated materialized tensor."
        ) from exc
    if actual_decoder_sha256 != raw_structural_decoder_sha256:
        raise Lfm2NormalizedInputDecoderError(
            "raw_structural_decoder tensor authentication failed."
        )
    if not isinstance(coordinate_completion_receipt, Mapping):
        raise Lfm2NormalizedInputDecoderError(
            "coordinate_completion_receipt must be a self-sealed mapping."
        )
    receipt = _deep_thaw(coordinate_completion_receipt)
    if not isinstance(receipt, dict):  # pragma: no cover - Mapping thaws to dict
        raise Lfm2NormalizedInputDecoderError(
            "coordinate_completion_receipt must be a self-sealed mapping."
        )
    seal_fields = tuple(
        name for name in ("result_sha256", "receipt_sha256") if name in receipt
    )
    if len(seal_fields) != 1:
        raise Lfm2NormalizedInputDecoderError(
            "coordinate_completion_receipt must contain exactly one stable "
            "result_sha256 or receipt_sha256 seal."
        )
    seal_field = seal_fields[0]
    claimed_receipt_sha256 = receipt.get(seal_field)
    if not _valid_sha256_text(claimed_receipt_sha256):
        raise Lfm2NormalizedInputDecoderError(
            "coordinate_completion_receipt contains an invalid stable seal."
        )
    receipt_body = dict(receipt)
    receipt_body.pop(seal_field)
    actual_receipt_sha256 = canonical_sha256(receipt_body)
    if not (
        actual_receipt_sha256
        == claimed_receipt_sha256
        == coordinate_completion_receipt_sha256
    ):
        raise Lfm2NormalizedInputDecoderError(
            "coordinate_completion_receipt authentication failed."
        )
    schema = receipt_body.get("schema")
    if not isinstance(schema, str) or not schema.strip():
        raise Lfm2NormalizedInputDecoderError(
            "coordinate_completion_receipt must identify a non-empty schema."
        )
    coordinate_map_sha256 = receipt_body.get("coordinate_map_sha256")
    if not _valid_sha256_text(coordinate_map_sha256):
        raise Lfm2NormalizedInputDecoderError(
            "coordinate_completion_receipt must bind coordinate_map_sha256."
        )
    actual_coordinate_map_sha256 = _coordinate_completion_tensor_sha256(
        raw_structural_decoder
    )
    if coordinate_map_sha256 != actual_coordinate_map_sha256:
        raise Lfm2NormalizedInputDecoderError(
            "coordinate_completion_receipt does not bind the supplied "
            "raw_structural_decoder."
        )
    coordinate_map_shape = receipt_body.get("coordinate_map_shape")
    expected_shape = [target_hidden, source_hidden]
    if coordinate_map_shape != expected_shape:
        raise Lfm2NormalizedInputDecoderError(
            "coordinate_completion_receipt coordinate_map_shape does not match "
            "the target-to-parent decoder topology."
        )
    evidence_sha256 = receipt_body.get("evidence_sha256")
    if evidence_sha256 is not None and not _valid_sha256_text(evidence_sha256):
        raise Lfm2NormalizedInputDecoderError(
            "coordinate_completion_receipt contains an invalid evidence_sha256."
        )
    return {
        "source": "authenticated_coordinate_completion_candidate",
        "schema": schema.strip(),
        "stable_seal_field": seal_field,
        "receipt_sha256": actual_receipt_sha256,
        "raw_structural_decoder_sha256": actual_decoder_sha256,
        "coordinate_map_sha256": actual_coordinate_map_sha256,
        "evidence_sha256": evidence_sha256,
        "receipt": receipt,
    }


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
    normalized_input_correction_rho: float = 0.25,
    raw_structural_decoder: Any | None = None,
    raw_structural_decoder_sha256: str | None = None,
    coordinate_completion_receipt: Mapping[str, Any] | None = None,
    coordinate_completion_receipt_sha256: str | None = None,
) -> Lfm2CompiledNormalizedInputDecoder:
    """Compile only the gamma-aware normalized-input target-to-parent decoder.

    ``residual_chart`` has shape ``[source_hidden, target_hidden]``.  The
    returned decoder has shape ``[target_hidden, source_hidden]`` and is the
    exact same structural inverse, FIT-rowspace correction, and bounded null
    completion used by the normalized real-medoid compiler.  An optional
    authenticated ``raw_structural_decoder`` may replace the chart
    pseudoinverse as that structural prior.  Its completion receipt must
    self-seal and bind the exact dense target-to-parent map.  Unlike the
    legacy chart-inverse lane, an authenticated completion retains its own
    scale and delegates FIT-identified action to an analytic rowspace
    correction whose Frobenius norm is capped at
    ``normalized_input_correction_rho * ||P_fit D0||_F``.  The default rho of
    0.25 preserves the strongest historical held-out normalized-input result;
    it is a construction policy, not a semantic quality gate.  No
    positive-correlation assumption is imposed across those different
    evidence spaces.  No source MLP weight, target template value, cluster,
    or medoid is read.
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
    normalized_input_correction_rho = _nonnegative_float(
        normalized_input_correction_rho,
        "normalized_input_correction_rho",
    )
    normalized_case_weights = _parse_case_sample_weights(
        cases,
        case_sample_weights,
    )
    completion_arguments = (
        raw_structural_decoder,
        raw_structural_decoder_sha256,
        coordinate_completion_receipt,
        coordinate_completion_receipt_sha256,
    )
    completion_supplied = any(value is not None for value in completion_arguments)
    if completion_supplied and not all(
        value is not None for value in completion_arguments
    ):
        raise Lfm2NormalizedInputDecoderError(
            "raw_structural_decoder, raw_structural_decoder_sha256, "
            "coordinate_completion_receipt, and "
            "coordinate_completion_receipt_sha256 must be supplied together."
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

    completion_provenance: dict[str, Any] | None = None
    external_raw_decoder = None
    if completion_supplied:
        completion_provenance = _authenticated_completion_provenance(
            raw_structural_decoder=raw_structural_decoder,
            raw_structural_decoder_sha256=raw_structural_decoder_sha256,
            coordinate_completion_receipt=coordinate_completion_receipt,
            coordinate_completion_receipt_sha256=(coordinate_completion_receipt_sha256),
            target_hidden=target_hidden,
            source_hidden=source_hidden,
        )
        external_raw_decoder = _float_tensor(
            raw_structural_decoder,
            "raw_structural_decoder",
            ndim=2,
        )
        if tuple(external_raw_decoder.shape) != (target_hidden, source_hidden):
            raise Lfm2NormalizedInputDecoderError(
                "raw_structural_decoder must have target_hidden by source_hidden shape."
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
            if external_raw_decoder is None:
                raw_chart_decoder = chart.T.contiguous()
                raw_chart_decoder_method = "authenticated_orthonormal_chart_transpose"
            else:
                raw_chart_decoder = external_raw_decoder
                raw_chart_decoder_method = (
                    "authenticated_coordinate_completion_structural_decoder"
                )
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
            if external_raw_decoder is None:
                raw_chart_decoder = (
                    (
                        raw_chart_vh.T[:, supported_singular]
                        * raw_chart_singular[supported_singular]
                        .reciprocal()
                        .unsqueeze(0)
                    )
                    @ raw_chart_u[:, supported_singular].T
                ).contiguous()
                raw_chart_decoder_method = "single_authenticated_thin_svd_pseudoinverse"
            else:
                raw_chart_decoder = external_raw_decoder
                raw_chart_decoder_method = (
                    "authenticated_coordinate_completion_structural_decoder"
                )
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
        structural_decoder_left_inverse_required = external_raw_decoder is None
        if (
            structural_decoder_prior_rank != target_hidden
            or not math.isfinite(structural_decoder_prior_operator_norm)
            or structural_decoder_prior_operator_norm <= torch.finfo(torch.float64).tiny
            or not math.isfinite(structural_decoder_left_inverse_error)
            or (
                structural_decoder_left_inverse_required
                and structural_decoder_left_inverse_error > 1e-8
            )
        ):
            raise Lfm2NormalizedInputDecoderError(
                "Gamma-aware normalized-input structural decoder is not a finite "
                "full-rank prior or required left inverse of the normalized chart."
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
        ):
            raise Lfm2NormalizedInputDecoderError(
                "FIT rows do not identify one strictly positive normalized-input "
                "structural gain."
            )
        fit_signed_scalar_gain = gain_numerator / gain_denominator
        if completion_provenance is None:
            if gain_numerator <= 0.0:
                raise Lfm2NormalizedInputDecoderError(
                    "FIT rows do not identify one strictly positive "
                    "normalized-input structural gain."
                )
            global_gain = fit_signed_scalar_gain
            structural_decoder = (unit_structural_decoder * global_gain).contiguous()
            structural_scale_policy = "positive-fit-ray-least-squares"
        else:
            # The body-global coordinate completion is already a fitted,
            # authenticated, full-rank target-to-parent map.  Normalizing it
            # and fitting a new positive scalar here discards that meaningful
            # scale and incorrectly assumes that residual FIT rows and this
            # layer's normalized MLP-input rows share one positive ray.  Keep
            # D0 intact.  The closed-form correction below owns the FIT row
            # space, while (I-P_X)D0 remains exactly structural.
            global_gain = structural_decoder_prior_operator_norm
            structural_decoder = gamma_aware_structural_decoder.clone().contiguous()
            structural_scale_policy = (
                "authenticated-coordinate-completion-prior-scale-preserved"
            )

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

        supported_relative_cap = (
            1.0 if completion_provenance is None else normalized_input_correction_rho
        )
        supported_prior = rowspace_projector @ structural_decoder
        supported_prior_norm = float(torch.linalg.vector_norm(supported_prior).item())
        correction_norm = float(torch.linalg.vector_norm(correction).item())
        correction_norm_cap_scale = 1.0
        allowed_correction_norm = supported_relative_cap * supported_prior_norm
        if correction_norm > allowed_correction_norm:
            correction_norm_cap_scale = allowed_correction_norm / max(
                correction_norm,
                torch.finfo(torch.float64).tiny,
            )
            correction = correction * correction_norm_cap_scale
        correction = correction.contiguous()

        # Replacing the FIT-identified action can cross an isolated singular
        # point even though D0 is full rank.  For authenticated completions,
        # first try the full correction, then a Weyl-safe half-minimum-
        # singular-value margin that retains structural rank.  This is a
        # topology invariant, never a behavior or fit-error ranking; alpha=0
        # is the authenticated full-rank fallback.
        rank_safe_correction_scale = 1.0
        normalized_input_decoder = (structural_decoder + correction).contiguous()
        decoder_rank = int(torch.linalg.matrix_rank(normalized_input_decoder).item())
        rank_safe_frontier_evaluations = 1
        rank_safe_prior_minimum_singular_value = None
        rank_safe_correction_operator_norm = None
        if completion_provenance is not None and decoder_rank != target_hidden:
            rank_safe_prior_minimum_singular_value = float(
                torch.linalg.matrix_norm(structural_decoder, ord=-2).item()
            )
            rank_safe_correction_operator_norm = float(
                torch.linalg.matrix_norm(correction, ord=2).item()
            )
            if (
                math.isfinite(rank_safe_prior_minimum_singular_value)
                and rank_safe_prior_minimum_singular_value > 0.0
                and math.isfinite(rank_safe_correction_operator_norm)
                and rank_safe_correction_operator_norm > 0.0
            ):
                rank_safe_correction_scale = min(
                    1.0,
                    0.5
                    * rank_safe_prior_minimum_singular_value
                    / rank_safe_correction_operator_norm,
                )
                normalized_input_decoder = (
                    structural_decoder + correction * rank_safe_correction_scale
                ).contiguous()
                decoder_rank = int(
                    torch.linalg.matrix_rank(normalized_input_decoder).item()
                )
                rank_safe_frontier_evaluations += 1
            if decoder_rank != target_hidden:
                rank_safe_correction_scale = 0.0
                normalized_input_decoder = structural_decoder.clone().contiguous()
                decoder_rank = structural_decoder_prior_rank
                rank_safe_frontier_evaluations += 1
            correction = (correction * rank_safe_correction_scale).contiguous()
        correction_scale = correction_norm_cap_scale * rank_safe_correction_scale

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
        structural_null_prior_operator_norm = float(
            torch.linalg.matrix_norm(expected_null_completion, ord=2).item()
        )
        null_completion_operator_norm_bound = (
            global_gain
            if completion_provenance is None
            else structural_null_prior_operator_norm
        )
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
            or null_completion_operator_norm
            > null_completion_operator_norm_bound + numerical_tolerance
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
            (
                "scale_preserving_gamma_aware_authenticated_coordinate_"
                "completion_prior_plus_fit_rowspace_correction_with_exact_"
                "structural_null_completion"
            )
            if completion_provenance is not None
            else (
                "positive_gain_gamma_aware_raw_chart_left_inverse_plus_"
                "fit_rowspace_correction_with_bounded_structural_null_completion"
            )
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
        "null_completion_operator_norm_bound": (null_completion_operator_norm_bound),
    }
    if completion_provenance is not None:
        producer_bridge.update(
            {
                "structural_prior_source": completion_provenance["source"],
                "structural_decoder_left_inverse_required": False,
                "structural_scale_policy": structural_scale_policy,
                "fit_signed_scalar_gain_advisory": fit_signed_scalar_gain,
                "fit_structural_gain_numerator_advisory": gain_numerator,
                "fit_structural_gain_denominator_advisory": gain_denominator,
                "fit_structural_gain_used_for_acceptance_or_veto": False,
                "identified_action_policy": (
                    "rho-capped-fit-rowspace-tikhonov-residual-correction"
                ),
                "normalized_input_correction_rho": (normalized_input_correction_rho),
                "fit_supported_structural_prior_frobenius_norm": (supported_prior_norm),
                "uncapped_fit_rowspace_correction_frobenius_norm": (correction_norm),
                "allowed_fit_rowspace_correction_frobenius_norm": (
                    allowed_correction_norm
                ),
                "correction_norm_cap_scale": correction_norm_cap_scale,
                "rank_safe_correction_scale": rank_safe_correction_scale,
                "rank_safe_frontier_evaluations": (rank_safe_frontier_evaluations),
                "rank_safe_homotopy_policy": (
                    "full-correction-then-weyl-half-minimum-singular-margin-"
                    "then-authenticated-prior"
                ),
                "rank_safe_prior_minimum_singular_value": (
                    rank_safe_prior_minimum_singular_value
                ),
                "rank_safe_correction_operator_norm": (
                    rank_safe_correction_operator_norm
                ),
                "rank_safe_selection_is_structural_not_semantic": True,
                "structural_null_prior_sha256": (
                    lfm2_authenticated_tensor_sha256(expected_null_completion)
                ),
                "structural_null_prior_operator_norm": (
                    structural_null_prior_operator_norm
                ),
                "structural_null_prior_preserved_exactly": True,
                "coordinate_completion_schema": completion_provenance["schema"],
                "coordinate_completion_receipt_sha256": (
                    completion_provenance["receipt_sha256"]
                ),
                "coordinate_completion_coordinate_map_sha256": (
                    completion_provenance["coordinate_map_sha256"]
                ),
                "coordinate_completion_evidence_sha256": (
                    completion_provenance["evidence_sha256"]
                ),
            }
        )
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
    if completion_provenance is not None:
        operation_audit.update(
            {
                "authenticated_raw_structural_decoder_tensor_reads": 1,
                "coordinate_completion_receipt_reads": 1,
                "residual_chart_decoder_recomputed": False,
            }
        )
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
        "normalized_input_correction_rho": (
            supported_relative_cap if completion_provenance is not None else None
        ),
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
    if completion_provenance is not None:
        body["coordinate_completion_provenance"] = completion_provenance
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
