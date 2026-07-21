"""Closed-form tied input/residual/output congruence candidates.

The causal-interface compiler historically optimized the tied table as an
output factor and then whitened its producer map.  That preserves the output
factorization, but it does not require the same table to be the image of the
mapped parent table under the resulting residual basis.  A tied child uses
that table for input embeddings too, so the missing invariant is::

    T_child = S_mapped @ B

where ``B`` is the orthonormal parent-to-child residual basis.  This module
builds that candidate directly in FP64 and authenticates both the incumbent
self-dual control and the congruent candidate.  It performs no model forwards,
optimization, backward pass, pruning, quantization, or model mutation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.hf_tied_causal_interface_fold import (
    TIED_CAUSAL_INTERFACE_SELF_DUAL_SCHEMA,
    TiedCausalInterfaceFold,
    TiedCausalInterfaceFoldError,
    whiten_tied_causal_interface,
)
from neural_forge.hf_tied_causal_interface_anchor_fold import (
    TIED_CAUSAL_ANCHOR_RESIDUAL_SCHEMA,
)


TIED_RESIDUAL_CONGRUENCE_SCHEMA = "neural-forge-tied-residual-congruence-candidate-v1"
TIED_RESIDUAL_CURRENT_SELF_DUAL_MODE = "current-selfdual"
TIED_RESIDUAL_STRONG_CONGRUENCE_MODE = "strong-congruence"
TIED_RESIDUAL_CONGRUENCE_MODES = (
    TIED_RESIDUAL_CURRENT_SELF_DUAL_MODE,
    TIED_RESIDUAL_STRONG_CONGRUENCE_MODE,
)
_TIED_CAUSAL_INTERFACE_FOLD_SCHEMA = "neural-forge-tied-causal-interface-fold-v1"
_TIED_CAUSAL_INTERFACE_RESUME_SCHEMA = "neural-forge-tied-causal-interface-resume-v1"
_SUPPORTED_SOURCE_INTERFACE_SCHEMAS = frozenset(
    {
        _TIED_CAUSAL_INTERFACE_FOLD_SCHEMA,
        _TIED_CAUSAL_INTERFACE_RESUME_SCHEMA,
        TIED_CAUSAL_ANCHOR_RESIDUAL_SCHEMA,
        TIED_CAUSAL_INTERFACE_SELF_DUAL_SCHEMA,
    }
)

_ZERO_OPERATIONS = MappingProxyType(
    {
        "parent_model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
        "parent_model_generate_calls": 0,
        "parent_trace_recaptures": 0,
        "child_model_forward_calls": 0,
        "child_model_generate_calls": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
)


class TiedResidualCongruenceError(ValueError):
    """A tied residual-congruence candidate failed authentication."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for tied residual congruence.") from exc
    return torch


def _matrix(value: Any, name: str) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 2
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
        or value.requires_grad
        or value.grad_fn is not None
    ):
        raise TiedResidualCongruenceError(
            f"{name} must be a nonempty detached dense real floating matrix."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise TiedResidualCongruenceError(f"{name} contains NaN or infinity.")
    return result


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            [int(size) for size in tensor.shape],
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _anchor_tensor_sha256(value: Any) -> str:
    """Match the causal-anchor residual interface tensor seal."""

    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
    }
    digest = hashlib.sha256(
        json.dumps(
            descriptor,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _authenticated_report(report: Mapping[str, Any], name: str) -> None:
    body = dict(report)
    seal = body.pop("report_sha256", None)
    if not isinstance(seal, str) or _canonical_sha256(body) != seal:
        raise TiedResidualCongruenceError(
            f"{name} report is not canonically authenticated."
        )


def _authenticate_source_interface(interface: TiedCausalInterfaceFold) -> None:
    """Bind a supported source report to the tensors it actually supplies."""

    if not isinstance(interface, TiedCausalInterfaceFold):
        raise TypeError("interface must be TiedCausalInterfaceFold.")
    report = dict(interface.report)
    _authenticated_report(report, "source interface")
    schema = report.get("schema")
    if schema not in _SUPPORTED_SOURCE_INTERFACE_SCHEMAS:
        raise TiedResidualCongruenceError(
            "Source interface report uses an unsupported schema."
        )
    tensor_sha256 = (
        _anchor_tensor_sha256
        if schema == TIED_CAUSAL_ANCHOR_RESIDUAL_SCHEMA
        else _tensor_sha256
    )
    if report.get("producer_encoder_sha256") != tensor_sha256(
        interface.producer_encoder
    ) or report.get("target_weight_sha256") != tensor_sha256(interface.target_weight):
        raise TiedResidualCongruenceError(
            "Source interface tensor lineage is not authentic."
        )


def _relative_error(actual: Any, expected: Any) -> float:
    torch = _torch()
    denominator = max(
        float(torch.linalg.vector_norm(expected).item()),
        torch.finfo(torch.float64).tiny,
    )
    return float(torch.linalg.vector_norm(actual - expected).item() / denominator)


def _cosine_summary(actual: Any, expected: Any, *, dimension: int) -> dict[str, Any]:
    torch = _torch()
    first = actual if dimension == 1 else actual.T
    second = expected if dimension == 1 else expected.T
    first_norm = torch.linalg.vector_norm(first, dim=1)
    second_norm = torch.linalg.vector_norm(second, dim=1)
    denominator = first_norm * second_norm
    nonzero = denominator > torch.finfo(torch.float64).tiny
    cosine = torch.ones_like(denominator)
    cosine[nonzero] = (
        (first[nonzero] * second[nonzero]).sum(dim=1) / denominator[nonzero]
    ).clamp(min=-1.0, max=1.0)
    both_zero = (first_norm <= torch.finfo(torch.float64).tiny) & (
        second_norm <= torch.finfo(torch.float64).tiny
    )
    cosine[~nonzero & ~both_zero] = 0.0
    return {
        "count": int(cosine.numel()),
        "mean": float(cosine.mean().item()),
        "minimum": float(cosine.min().item()),
        "maximum": float(cosine.max().item()),
        "zero_norm_pair_count": int(both_zero.sum().item()),
        "one_sided_zero_norm_count": int((~nonzero & ~both_zero).sum().item()),
    }


def _self_dual_control(interface: TiedCausalInterfaceFold) -> TiedCausalInterfaceFold:
    _authenticate_source_interface(interface)
    if interface.report.get("schema") == TIED_CAUSAL_INTERFACE_SELF_DUAL_SCHEMA:
        return interface
    try:
        self_dual = whiten_tied_causal_interface(interface)
    except TiedCausalInterfaceFoldError as exc:
        raise TiedResidualCongruenceError(
            "The source interface cannot produce a self-dual residual basis."
        ) from exc
    _authenticate_source_interface(self_dual)
    return self_dual


@dataclass(frozen=True, slots=True)
class TiedResidualCongruenceCandidate:
    """One selected tied interface plus its exact BF16 materialization proof."""

    interface: TiedCausalInterfaceFold
    materialized_target_weight: Any
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        torch = _torch()
        if not isinstance(self.interface, TiedCausalInterfaceFold):
            raise TypeError("interface must be TiedCausalInterfaceFold.")
        materialized = self.materialized_target_weight
        if (
            not isinstance(materialized, torch.Tensor)
            or materialized.layout != torch.strided
            or materialized.device.type != "cpu"
            or materialized.dtype != torch.bfloat16
            or tuple(materialized.shape) != tuple(self.interface.target_weight.shape)
            or materialized.requires_grad
            or materialized.grad_fn is not None
            or not bool(torch.isfinite(materialized).all().item())
        ):
            raise TiedResidualCongruenceError(
                "materialized_target_weight must be finite detached CPU BF16."
            )
        _authenticated_report(self.report, "congruence candidate")
        object.__setattr__(
            self,
            "materialized_target_weight",
            materialized.detach().contiguous().clone(),
        )
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    @property
    def mode(self) -> str:
        return str(self.report["mode"])


def authenticate_tied_residual_congruence_interface(
    interface: TiedCausalInterfaceFold,
    *,
    mapped_parent_tied_table: Any | None = None,
) -> Mapping[str, Any]:
    """Authenticate a congruence-schema interface and optionally recompute it.

    Layer-stack consumers can authenticate the sealed tensor lineage without
    retaining another copy of the mapped parent table.  Terminal consumers,
    which already receive that table, additionally recompute ``S @ B`` and
    require exact FP64 equality for the strong candidate.
    """

    torch = _torch()
    if not isinstance(interface, TiedCausalInterfaceFold):
        raise TypeError("interface must be TiedCausalInterfaceFold.")
    report = dict(interface.report)
    _authenticated_report(report, "tied residual congruence interface")
    if report.get("schema") != TIED_RESIDUAL_CONGRUENCE_SCHEMA:
        raise TiedResidualCongruenceError(
            "Interface does not use the tied residual congruence schema."
        )
    mode = report.get("mode")
    if mode not in TIED_RESIDUAL_CONGRUENCE_MODES:
        raise TiedResidualCongruenceError(
            "Tied residual congruence report has an unsupported mode."
        )
    target = _matrix(interface.target_weight, "interface.target_weight")
    basis = _matrix(interface.producer_encoder, "interface.producer_encoder")
    identity = torch.eye(int(basis.shape[1]), dtype=torch.float64)
    orthogonality_error = float(
        torch.linalg.matrix_norm(basis.T @ basis - identity, ord=2).item()
    )
    reported_tolerance = report.get("producer_basis_orthogonality_tolerance")
    if (
        not isinstance(reported_tolerance, (int, float))
        or isinstance(reported_tolerance, bool)
        or not math.isfinite(float(reported_tolerance))
        or float(reported_tolerance) <= 0.0
        or orthogonality_error > float(reported_tolerance)
        or report.get("producer_encoder_sha256") != _tensor_sha256(basis)
        or report.get("target_weight_sha256") != _tensor_sha256(target)
        or report.get("producer_basis_orthogonality_authenticated") is not True
        or report.get("global_residual_basis_unchanged") is not True
        or report.get("native_tied_parameter_topology_change_required") is not False
        or report.get("operation_audit") != dict(_ZERO_OPERATIONS)
    ):
        raise TiedResidualCongruenceError(
            "Tied residual congruence tensor lineage is not authentic."
        )
    strong = mode == TIED_RESIDUAL_STRONG_CONGRUENCE_MODE
    if (
        report.get("selected_target_is_strong_congruence") is not strong
        or report.get("selected_target_is_current_selfdual_control") is strong
        or (strong and report.get("fp64_congruence_exact") is not True)
        or (strong and report.get("bf16_congruence_exact") is not True)
    ):
        raise TiedResidualCongruenceError(
            "Tied residual congruence mode claims are inconsistent."
        )

    source_recomputed = mapped_parent_tied_table is not None
    exact_recomputed = False
    recomputed_relative_error: float | None = None
    if source_recomputed:
        source = _matrix(mapped_parent_tied_table, "mapped_parent_tied_table")
        if tuple(source.shape) != (int(target.shape[0]), int(basis.shape[0])):
            raise TiedResidualCongruenceError(
                "Mapped parent table is incompatible with the congruence interface."
            )
        expected = (source @ basis).contiguous()
        exact_recomputed = bool(torch.equal(target, expected))
        recomputed_relative_error = _relative_error(target, expected)
        if (
            report.get("mapped_parent_tied_table_sha256") != _tensor_sha256(source)
            or (strong and not exact_recomputed)
            or (strong and recomputed_relative_error != 0.0)
        ):
            raise TiedResidualCongruenceError(
                "Strong tied residual congruence does not equal mapped_parent @ B."
            )

    body = {
        "schema": "neural-forge-tied-residual-congruence-authentication-v1",
        "interface_report_sha256": report["report_sha256"],
        "mode": mode,
        "producer_encoder_sha256": report["producer_encoder_sha256"],
        "target_weight_sha256": report["target_weight_sha256"],
        "producer_basis_orthogonality_spectral_error_recomputed": (orthogonality_error),
        "producer_basis_orthogonality_authenticated": True,
        "mapped_parent_tied_table_supplied": source_recomputed,
        "strong_congruence_recomputed_exact": exact_recomputed if strong else False,
        "recomputed_congruence_relative_error": recomputed_relative_error,
        "strong_congruence_claimed": strong,
        "current_selfdual_control_claimed": not strong,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    return MappingProxyType({**body, "result_sha256": _canonical_sha256(body)})


def compile_tied_residual_congruence_candidate(
    interface: TiedCausalInterfaceFold,
    mapped_parent_tied_table: Any,
    *,
    mode: str = TIED_RESIDUAL_STRONG_CONGRUENCE_MODE,
) -> TiedResidualCongruenceCandidate:
    """Compile the incumbent control or ``S_mapped @ B`` tied candidate.

    ``B`` is always the authenticated self-dual producer basis.  The control
    retains the incumbent output-factor table.  The strong candidate replaces
    only that table with the exact mapped-parent image under ``B``.
    """

    torch = _torch()
    if mode not in TIED_RESIDUAL_CONGRUENCE_MODES:
        raise TiedResidualCongruenceError(
            f"mode must be one of {TIED_RESIDUAL_CONGRUENCE_MODES!r}."
        )
    current = _self_dual_control(interface)
    _authenticated_report(current.report, "self-dual control")
    source = _matrix(mapped_parent_tied_table, "mapped_parent_tied_table")
    current_target = _matrix(current.target_weight, "current target_weight")
    basis = _matrix(current.producer_encoder, "self-dual producer_encoder")
    if tuple(source.shape) != (
        int(current_target.shape[0]),
        int(basis.shape[0]),
    ):
        raise TiedResidualCongruenceError(
            "Mapped parent tied table must have vocabulary rows matching the "
            "child and source width matching the producer basis."
        )

    with torch.no_grad():
        identity = torch.eye(int(basis.shape[1]), dtype=torch.float64)
        orthogonality_residual = basis.T @ basis - identity
        orthogonality_spectral_error = float(
            torch.linalg.matrix_norm(orthogonality_residual, ord=2).item()
        )
        orthogonality_maximum_error = float(orthogonality_residual.abs().max().item())
        orthogonality_tolerance = max(
            1e-10,
            100.0
            * torch.finfo(torch.float64).eps
            * max(int(basis.shape[0]), int(basis.shape[1])),
        )
        if (
            not math.isfinite(orthogonality_spectral_error)
            or orthogonality_spectral_error > orthogonality_tolerance
        ):
            raise TiedResidualCongruenceError(
                "The self-dual producer basis is not numerically orthonormal."
            )

        congruent_target = (source @ basis).contiguous()
        congruence_reference = (source @ basis).contiguous()
        pre_materialization_exact = bool(
            torch.equal(congruent_target, congruence_reference)
        )
        pre_materialization_relative_error = _relative_error(
            congruent_target,
            congruence_reference,
        )
        pre_materialization_maximum_error = float(
            (congruent_target - congruence_reference).abs().max().item()
        )
        if (
            not pre_materialization_exact
            or pre_materialization_relative_error != 0.0
            or pre_materialization_maximum_error != 0.0
        ):
            raise TiedResidualCongruenceError(
                "FP64 tied residual congruence was not exact."
            )

        selected_target = (
            current_target
            if mode == TIED_RESIDUAL_CURRENT_SELF_DUAL_MODE
            else congruent_target
        ).contiguous()
        materialized = selected_target.to(torch.bfloat16).contiguous()
        expected_materialized = selected_target.to(torch.bfloat16).contiguous()
        congruent_materialized = congruence_reference.to(torch.bfloat16).contiguous()
        post_materialization_exact = bool(
            torch.equal(materialized, expected_materialized)
        )
        post_materialization_congruence_exact = bool(
            torch.equal(materialized, congruent_materialized)
        )
        materialized64 = materialized.to(torch.float64)

        old_operator = (basis @ current_target.T).contiguous()
        congruent_operator = (basis @ congruent_target.T).contiguous()
        selected_operator = (basis @ selected_target.T).contiguous()
        current_congruence_relative_error = _relative_error(
            current_target,
            congruent_target,
        )
        current_congruence_residual = current_target - congruent_target
        old_factor_relative_error = _relative_error(
            congruent_operator,
            old_operator,
        )
        selected_old_factor_relative_error = _relative_error(
            selected_operator,
            old_operator,
        )

    selected_body = {
        "schema": TIED_RESIDUAL_CONGRUENCE_SCHEMA,
        "mode": mode,
        "method": (
            "incumbent_self_dual_output_factor_control"
            if mode == TIED_RESIDUAL_CURRENT_SELF_DUAL_MODE
            else "mapped_parent_tied_table_times_global_orthonormal_residual_basis"
        ),
        "source_interface_report_sha256": interface.report["report_sha256"],
        "current_self_dual_interface_report_sha256": current.report["report_sha256"],
        "mapped_parent_tied_table_sha256": _tensor_sha256(source),
        "current_target_weight_sha256": _tensor_sha256(current_target),
        "congruent_target_weight_sha256": _tensor_sha256(congruent_target),
        "target_weight_sha256": _tensor_sha256(selected_target),
        "producer_encoder_sha256": _tensor_sha256(basis),
        "materialized_target_weight_sha256": _tensor_sha256(materialized),
        "source_width": int(basis.shape[0]),
        "target_width": int(basis.shape[1]),
        "full_vocabulary_rows_used": int(source.shape[0]),
        "producer_basis_orthogonality_spectral_error": (orthogonality_spectral_error),
        "producer_basis_orthogonality_maximum_absolute_error": (
            orthogonality_maximum_error
        ),
        "producer_basis_orthogonality_tolerance": orthogonality_tolerance,
        "producer_basis_orthogonality_authenticated": True,
        "fp64_congruence_exact": pre_materialization_exact,
        "fp64_congruence_relative_error": pre_materialization_relative_error,
        "fp64_congruence_maximum_absolute_error": (pre_materialization_maximum_error),
        "bf16_materialization_exact": post_materialization_exact,
        "bf16_congruence_exact": post_materialization_congruence_exact,
        "bf16_to_fp64_congruence_relative_error": _relative_error(
            materialized64,
            congruent_target,
        ),
        "bf16_to_fp64_congruence_maximum_absolute_error": float(
            (materialized64 - congruent_target).abs().max().item()
        ),
        "bf16_roundtrip_relative_error": _relative_error(
            materialized64,
            selected_target,
        ),
        "bf16_roundtrip_maximum_absolute_error": float(
            (materialized64 - selected_target).abs().max().item()
        ),
        "current_target_to_congruent_relative_error": (
            current_congruence_relative_error
        ),
        "current_target_to_congruent_maximum_absolute_error": float(
            current_congruence_residual.abs().max().item()
        ),
        "current_target_to_congruent_row_cosine": _cosine_summary(
            current_target,
            congruent_target,
            dimension=1,
        ),
        "congruent_vs_old_output_factor_relative_error": (old_factor_relative_error),
        "congruent_vs_old_output_factor_row_cosine": _cosine_summary(
            congruent_operator,
            old_operator,
            dimension=1,
        ),
        "congruent_vs_old_output_factor_column_cosine": _cosine_summary(
            congruent_operator,
            old_operator,
            dimension=0,
        ),
        "selected_vs_old_output_factor_relative_error": (
            selected_old_factor_relative_error
        ),
        "selected_target_is_current_selfdual_control": (
            mode == TIED_RESIDUAL_CURRENT_SELF_DUAL_MODE
        ),
        "selected_target_is_strong_congruence": (
            mode == TIED_RESIDUAL_STRONG_CONGRUENCE_MODE
        ),
        "global_residual_basis_unchanged": True,
        "native_tied_parameter_topology_change_required": False,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    selected_report = {
        **selected_body,
        "report_sha256": _canonical_sha256(selected_body),
    }
    selected_interface = TiedCausalInterfaceFold(
        target_weight=selected_target,
        producer_encoder=basis,
        report=selected_report,
    )
    return TiedResidualCongruenceCandidate(
        interface=selected_interface,
        materialized_target_weight=materialized,
        report=selected_report,
    )


__all__ = [
    "TIED_RESIDUAL_CONGRUENCE_MODES",
    "TIED_RESIDUAL_CONGRUENCE_SCHEMA",
    "TIED_RESIDUAL_CURRENT_SELF_DUAL_MODE",
    "TIED_RESIDUAL_STRONG_CONGRUENCE_MODE",
    "TiedResidualCongruenceCandidate",
    "TiedResidualCongruenceError",
    "authenticate_tied_residual_congruence_interface",
    "compile_tied_residual_congruence_candidate",
]
