"""Fold-only absorption of an output-action projector into one LFM2 writer.

This diagnostic parent-shell primitive consumes an authenticated compiled
output-action subspace and one dense FP32 ShortConv ``out_proj.weight`` with
shape ``[parent_hidden, internal_channels]``.  It computes exactly::

    W_star = P.T @ W_parent

in CPU FP64 and materializes one independent ordinary dense FP32 writer.  The
returned runtime state contains only ``out_proj.weight``: no Q/P factor,
projector, bridge, cache, parent sidecar, or shared parameter is retained.

The writer shape and parameter count do not change.  This is intentionally a
component-isolation diagnostic, not a parameter-reduction result.  The API
accepts no model and performs no model call, trace capture, training, autograd,
optimizer step, pruning, quantization, filesystem access, or network access.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_output_action_subspace import (
    Lfm2CompiledOutputActionSubspace,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    lfm2_authenticated_tensor_sha256,
)


LFM2_OUTPUT_ACTION_WRITER_FOLD_SCHEMA = "neural-forge-lfm2-output-action-writer-fold-v1"
LFM2_OUTPUT_ACTION_WRITER_STATE_SCHEMA = (
    "neural-forge-lfm2-output-action-writer-state-v1"
)
LFM2_OUTPUT_ACTION_WRITER_LINEAGE_SCHEMA = (
    "neural-forge-lfm2-output-action-writer-lineage-v1"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_STATE_KEY = "out_proj.weight"
_ZERO_OPERATION_AUDIT = MappingProxyType(
    {
        "model_objects_accepted": 0,
        "parent_model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
        "candidate_model_forward_calls": 0,
        "model_generate_calls": 0,
        "parent_trace_recaptures": 0,
        "new_parent_trace_rows": 0,
        "training_steps": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "parameters_sparsified": 0,
        "runtime_q_factors": 0,
        "runtime_p_factors": 0,
        "runtime_bridge_modules": 0,
        "runtime_parameter_aliases_created": 0,
        "runtime_parameter_sharing_used": False,
        "parent_sidecars_required": 0,
        "filesystem_reads": 0,
        "filesystem_writes": 0,
        "network_access": False,
    }
)


class Lfm2OutputActionWriterFoldError(ValueError):
    """The subspace, writer, pins, lineage, or materialization is invalid."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional native dependency
        raise RuntimeError("Output-action writer folding requires PyTorch.") from exc
    return torch


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_deep_thaw(item) for item in value]
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise Lfm2OutputActionWriterFoldError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _tensor_record(name: str, value: Any) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": str(value.dtype).removeprefix("torch."),
        "shape": [int(size) for size in value.shape],
        "sha256": lfm2_authenticated_tensor_sha256(value),
    }


def _state_sha256(value: Any) -> str:
    return canonical_sha256(
        {
            "schema": LFM2_OUTPUT_ACTION_WRITER_STATE_SCHEMA,
            "tensors": [_tensor_record(_STATE_KEY, value)],
        }
    )


def _source_writer(value: Any) -> Any:
    torch = _torch()
    allowed_types = (torch.Tensor, torch.nn.Parameter)
    if (
        type(value) not in allowed_types
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 2
        or any(int(size) <= 0 for size in value.shape)
        or value.dtype != torch.float32
        or value.is_complex()
        or bool(getattr(value, "is_quantized", False))
    ):
        raise Lfm2OutputActionWriterFoldError(
            "parent_out_proj_weight must be one exact dense materialized FP32 "
            "rank-2 Tensor or Parameter."
        )
    result = value.detach().to(device="cpu").contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2OutputActionWriterFoldError(
            "parent_out_proj_weight contains NaN or infinity."
        )
    return result


def _authenticated_subspace(
    value: Any,
    *,
    expected_parent_state_sha256: str,
    expected_cache_sha256: str,
    expected_subspace_result_sha256: str,
    expected_basis_receipt_sha256: str,
    expected_projector_sha256: str,
) -> tuple[Lfm2CompiledOutputActionSubspace, dict[str, Any]]:
    if not isinstance(value, Lfm2CompiledOutputActionSubspace):
        raise Lfm2OutputActionWriterFoldError(
            "subspace must be one authenticated compiled output-action subspace."
        )
    # as_dict() reauthenticates Q, P, Q Q.T, tensor seals, and the result seal.
    try:
        report = value.as_dict()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2OutputActionWriterFoldError(
            "Compiled output-action subspace authentication failed."
        ) from exc
    pins = {
        "parent_state_sha256": _digest(
            expected_parent_state_sha256,
            "expected_parent_state_sha256",
        ),
        "cache_sha256": _digest(
            expected_cache_sha256,
            "expected_cache_sha256",
        ),
        "result_sha256": _digest(
            expected_subspace_result_sha256,
            "expected_subspace_result_sha256",
        ),
        "basis_receipt_sha256": _digest(
            expected_basis_receipt_sha256,
            "expected_basis_receipt_sha256",
        ),
        "projector_sha256": _digest(
            expected_projector_sha256,
            "expected_projector_sha256",
        ),
    }
    observed = {
        "parent_state_sha256": report.get("parent_state_sha256"),
        "cache_sha256": report.get("cache_sha256"),
        "result_sha256": report.get("result_sha256"),
        "basis_receipt_sha256": report.get("basis_receipt_sha256"),
        "projector_sha256": (
            report.get("projector", {}).get("sha256")
            if isinstance(report.get("projector"), Mapping)
            else None
        ),
    }
    if observed != pins:
        mismatches = tuple(name for name in pins if observed.get(name) != pins[name])
        raise Lfm2OutputActionWriterFoldError(
            f"Independent output-action subspace pins do not match: {mismatches!r}."
        )
    return value, report


def _lineage_body(
    *,
    parent_state_sha256: str,
    cache_sha256: str,
    subspace_result_sha256: str,
    basis_receipt_sha256: str,
    projector_sha256: str,
    parent_out_proj_sha256: str,
    folded_state_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": LFM2_OUTPUT_ACTION_WRITER_LINEAGE_SCHEMA,
        "parent_state_sha256": parent_state_sha256,
        "cache_sha256": cache_sha256,
        "subspace_result_sha256": subspace_result_sha256,
        "basis_receipt_sha256": basis_receipt_sha256,
        "projector_sha256": projector_sha256,
        "parent_out_proj_sha256": parent_out_proj_sha256,
        "folded_state_sha256": folded_state_sha256,
        "operation": "W_star=P.T@W_parent",
        "compute_device": "cpu",
        "compute_dtype": "float64",
        "materialization_dtype": "float32",
    }


@dataclass(frozen=True, slots=True)
class Lfm2OutputActionWriterFoldResult:
    """One sealed independent dense FP32 ShortConv output writer."""

    out_proj_weight: Any = field(repr=False, compare=False)
    report: Mapping[str, Any]

    @property
    def state_sha256(self) -> str:
        return str(self.report["state_sha256"])

    @property
    def lineage_sha256(self) -> str:
        return str(self.report["lineage_sha256"])

    @property
    def result_sha256(self) -> str:
        return str(self.report["result_sha256"])

    def state_dict(self) -> Mapping[str, Any]:
        """Return a verified independent copy of the sole runtime tensor."""

        verify_lfm2_output_action_writer_fold(self)
        return MappingProxyType(
            {
                _STATE_KEY: self.out_proj_weight.detach()
                .to(device="cpu")
                .contiguous()
                .clone()
            }
        )

    def verify(
        self,
        *,
        expected_parent_state_sha256: str | None = None,
        expected_cache_sha256: str | None = None,
        expected_subspace_result_sha256: str | None = None,
        expected_basis_receipt_sha256: str | None = None,
        expected_projector_sha256: str | None = None,
        expected_parent_out_proj_sha256: str | None = None,
        expected_state_sha256: str | None = None,
        expected_lineage_sha256: str | None = None,
        expected_result_sha256: str | None = None,
    ) -> bool:
        """Reauthenticate state/lineage and any independently supplied pins."""

        return verify_lfm2_output_action_writer_fold(
            self,
            expected_parent_state_sha256=expected_parent_state_sha256,
            expected_cache_sha256=expected_cache_sha256,
            expected_subspace_result_sha256=expected_subspace_result_sha256,
            expected_basis_receipt_sha256=expected_basis_receipt_sha256,
            expected_projector_sha256=expected_projector_sha256,
            expected_parent_out_proj_sha256=expected_parent_out_proj_sha256,
            expected_state_sha256=expected_state_sha256,
            expected_lineage_sha256=expected_lineage_sha256,
            expected_result_sha256=expected_result_sha256,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a verified mutable JSON-compatible result receipt."""

        verify_lfm2_output_action_writer_fold(self)
        return _deep_thaw(self.report)


def _optional_pin(
    value: str | None,
    *,
    label: str,
    observed: Any,
) -> None:
    if value is None:
        return
    expected = _digest(value, label)
    if observed != expected:
        raise Lfm2OutputActionWriterFoldError(
            f"{label} does not match the sealed writer-fold result."
        )


def verify_lfm2_output_action_writer_fold(
    result: Lfm2OutputActionWriterFoldResult,
    *,
    expected_parent_state_sha256: str | None = None,
    expected_cache_sha256: str | None = None,
    expected_subspace_result_sha256: str | None = None,
    expected_basis_receipt_sha256: str | None = None,
    expected_projector_sha256: str | None = None,
    expected_parent_out_proj_sha256: str | None = None,
    expected_state_sha256: str | None = None,
    expected_lineage_sha256: str | None = None,
    expected_result_sha256: str | None = None,
) -> bool:
    """Verify the sole tensor, immutable receipt, accounting, and pins."""

    torch = _torch()
    if not isinstance(result, Lfm2OutputActionWriterFoldResult):
        raise Lfm2OutputActionWriterFoldError(
            "result must be one output-action writer-fold result."
        )
    weight = result.out_proj_weight
    if (
        type(weight) is not torch.Tensor
        or weight.layout != torch.strided
        or weight.device.type != "cpu"
        or weight.ndim != 2
        or any(int(size) <= 0 for size in weight.shape)
        or weight.dtype != torch.float32
        or not weight.is_contiguous()
        or weight.requires_grad
        or weight.grad_fn is not None
        or bool(getattr(weight, "is_quantized", False))
        or not bool(torch.isfinite(weight).all().item())
    ):
        raise Lfm2OutputActionWriterFoldError(
            "Folded out_proj.weight was mutated or malformed."
        )
    report = _deep_thaw(result.report)
    if report.get("schema") != LFM2_OUTPUT_ACTION_WRITER_FOLD_SCHEMA:
        raise Lfm2OutputActionWriterFoldError("Writer-fold result schema is invalid.")
    if (
        report.get("method") != "dense-output-action-projector-seam-absorption"
        or report.get("operation") != "W_star=P.T@W_parent"
        or report.get("compute_device") != "cpu"
        or report.get("compute_dtype") != "float64"
        or report.get("materialization_dtype") != "float32"
    ):
        raise Lfm2OutputActionWriterFoldError(
            "Writer-fold arithmetic contract is invalid."
        )
    if report.get("runtime_state_keys") != [_STATE_KEY]:
        raise Lfm2OutputActionWriterFoldError(
            "Writer-fold runtime state contains an unexpected sidecar."
        )
    if report.get("out_proj_weight") != _tensor_record(_STATE_KEY, weight):
        raise Lfm2OutputActionWriterFoldError(
            "Folded out_proj.weight tensor seal is invalid."
        )
    state_sha256 = _state_sha256(weight)
    if report.get("state_sha256") != state_sha256:
        raise Lfm2OutputActionWriterFoldError("Writer-fold state seal is invalid.")
    source_record = report.get("source_parent_out_proj_weight")
    if not isinstance(source_record, Mapping) or source_record != {
        "name": "parent_out_proj.weight",
        "dtype": "float32",
        "shape": [int(size) for size in weight.shape],
        "sha256": report.get("parent_out_proj_sha256"),
    }:
        raise Lfm2OutputActionWriterFoldError("Parent writer source record is invalid.")
    parameter_count = int(weight.numel())
    expected_accounting = {
        "parent_parameter_count": parameter_count,
        "folded_parameter_count": parameter_count,
        "parameter_delta": 0,
        "parameter_ratio_parent_over_folded": 1.0,
        "parameter_reduction_claimed": False,
        "diagnostic_parent_shell_only": True,
    }
    if report.get("parameter_accounting") != expected_accounting:
        raise Lfm2OutputActionWriterFoldError(
            "Writer-fold parameter accounting is invalid."
        )
    if (
        report.get("runtime_q_or_p_factors_retained") is not False
        or report.get("parent_sidecar_required") is not False
        or report.get("training_or_optimization_used") is not False
        or report.get("operation_audit") != dict(_ZERO_OPERATION_AUDIT)
    ):
        raise Lfm2OutputActionWriterFoldError(
            "Writer-fold runtime/training contract is invalid."
        )
    lineage_body = _lineage_body(
        parent_state_sha256=_digest(
            report.get("parent_state_sha256"), "parent_state_sha256"
        ),
        cache_sha256=_digest(report.get("cache_sha256"), "cache_sha256"),
        subspace_result_sha256=_digest(
            report.get("subspace_result_sha256"), "subspace_result_sha256"
        ),
        basis_receipt_sha256=_digest(
            report.get("basis_receipt_sha256"), "basis_receipt_sha256"
        ),
        projector_sha256=_digest(report.get("projector_sha256"), "projector_sha256"),
        parent_out_proj_sha256=_digest(
            report.get("parent_out_proj_sha256"), "parent_out_proj_sha256"
        ),
        folded_state_sha256=state_sha256,
    )
    lineage_sha256 = canonical_sha256(lineage_body)
    if (
        report.get("lineage") != lineage_body
        or report.get("lineage_sha256") != lineage_sha256
    ):
        raise Lfm2OutputActionWriterFoldError(
            "Writer-fold lineage authentication failed."
        )
    sealed_result_sha256 = report.pop("result_sha256", None)
    if sealed_result_sha256 != canonical_sha256(report):
        raise Lfm2OutputActionWriterFoldError(
            "Writer-fold result receipt authentication failed."
        )

    for value, label, observed in (
        (
            expected_parent_state_sha256,
            "expected_parent_state_sha256",
            lineage_body["parent_state_sha256"],
        ),
        (
            expected_cache_sha256,
            "expected_cache_sha256",
            lineage_body["cache_sha256"],
        ),
        (
            expected_subspace_result_sha256,
            "expected_subspace_result_sha256",
            lineage_body["subspace_result_sha256"],
        ),
        (
            expected_basis_receipt_sha256,
            "expected_basis_receipt_sha256",
            lineage_body["basis_receipt_sha256"],
        ),
        (
            expected_projector_sha256,
            "expected_projector_sha256",
            lineage_body["projector_sha256"],
        ),
        (
            expected_parent_out_proj_sha256,
            "expected_parent_out_proj_sha256",
            lineage_body["parent_out_proj_sha256"],
        ),
        (expected_state_sha256, "expected_state_sha256", state_sha256),
        (
            expected_lineage_sha256,
            "expected_lineage_sha256",
            lineage_sha256,
        ),
        (
            expected_result_sha256,
            "expected_result_sha256",
            sealed_result_sha256,
        ),
    ):
        _optional_pin(value, label=label, observed=observed)
    return True


def compile_lfm2_output_action_writer_fold(
    subspace: Lfm2CompiledOutputActionSubspace,
    parent_out_proj_weight: Any,
    *,
    expected_parent_state_sha256: str,
    expected_cache_sha256: str,
    expected_subspace_result_sha256: str,
    expected_basis_receipt_sha256: str,
    expected_projector_sha256: str,
    expected_parent_out_proj_sha256: str,
) -> Lfm2OutputActionWriterFoldResult:
    """Absorb ``P`` into one dense parent writer after independent pin checks."""

    torch = _torch()
    subspace, subspace_report = _authenticated_subspace(
        subspace,
        expected_parent_state_sha256=expected_parent_state_sha256,
        expected_cache_sha256=expected_cache_sha256,
        expected_subspace_result_sha256=expected_subspace_result_sha256,
        expected_basis_receipt_sha256=expected_basis_receipt_sha256,
        expected_projector_sha256=expected_projector_sha256,
    )
    parent_weight = _source_writer(parent_out_proj_weight)
    parent_weight_sha256 = lfm2_authenticated_tensor_sha256(parent_weight)
    if parent_weight_sha256 != _digest(
        expected_parent_out_proj_sha256,
        "expected_parent_out_proj_sha256",
    ):
        raise Lfm2OutputActionWriterFoldError(
            "Independent parent out_proj.weight pin does not match."
        )
    if int(parent_weight.shape[0]) != subspace.source_width:
        raise Lfm2OutputActionWriterFoldError(
            "Parent out_proj output width does not match the output-action projector."
        )

    with torch.no_grad():
        source_fp64 = parent_weight.to(dtype=torch.float64).contiguous()
        projected_fp64 = (subspace.projector.T @ source_fp64).contiguous()
        if not bool(torch.isfinite(projected_fp64).all().item()):
            raise Lfm2OutputActionWriterFoldError(
                "CPU FP64 projected writer contains NaN or infinity."
            )
        folded = projected_fp64.to(dtype=torch.float32).contiguous().clone()
    if not bool(torch.isfinite(folded).all().item()):
        raise Lfm2OutputActionWriterFoldError(
            "FP32 writer materialization contains NaN or infinity."
        )
    if (
        folded.untyped_storage().data_ptr()
        == parent_weight.untyped_storage().data_ptr()
    ):
        raise Lfm2OutputActionWriterFoldError(
            "Folded writer unexpectedly aliases parent writer storage."
        )

    state_sha256 = _state_sha256(folded)
    projector_sha256 = str(subspace_report["projector"]["sha256"])
    lineage = _lineage_body(
        parent_state_sha256=expected_parent_state_sha256,
        cache_sha256=expected_cache_sha256,
        subspace_result_sha256=expected_subspace_result_sha256,
        basis_receipt_sha256=expected_basis_receipt_sha256,
        projector_sha256=projector_sha256,
        parent_out_proj_sha256=parent_weight_sha256,
        folded_state_sha256=state_sha256,
    )
    lineage_sha256 = canonical_sha256(lineage)
    parameter_count = int(parent_weight.numel())
    body = {
        "schema": LFM2_OUTPUT_ACTION_WRITER_FOLD_SCHEMA,
        "method": "dense-output-action-projector-seam-absorption",
        "operation": "W_star=P.T@W_parent",
        "parent_state_sha256": expected_parent_state_sha256,
        "cache_sha256": expected_cache_sha256,
        "subspace_result_sha256": expected_subspace_result_sha256,
        "basis_receipt_sha256": expected_basis_receipt_sha256,
        "projector_sha256": projector_sha256,
        "parent_out_proj_sha256": parent_weight_sha256,
        "source_parent_out_proj_weight": _tensor_record(
            "parent_out_proj.weight", parent_weight
        ),
        "out_proj_weight": _tensor_record(_STATE_KEY, folded),
        "state_sha256": state_sha256,
        "lineage": lineage,
        "lineage_sha256": lineage_sha256,
        "compute_device": "cpu",
        "compute_dtype": "float64",
        "materialization_dtype": "float32",
        "runtime_state_keys": [_STATE_KEY],
        "runtime_q_or_p_factors_retained": False,
        "parent_sidecar_required": False,
        "parameter_accounting": {
            "parent_parameter_count": parameter_count,
            "folded_parameter_count": parameter_count,
            "parameter_delta": 0,
            "parameter_ratio_parent_over_folded": 1.0,
            "parameter_reduction_claimed": False,
            "diagnostic_parent_shell_only": True,
        },
        "training_or_optimization_used": False,
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
    }
    result = Lfm2OutputActionWriterFoldResult(
        out_proj_weight=folded,
        report=_deep_freeze({**body, "result_sha256": canonical_sha256(body)}),
    )
    verify_lfm2_output_action_writer_fold(
        result,
        expected_parent_state_sha256=expected_parent_state_sha256,
        expected_cache_sha256=expected_cache_sha256,
        expected_subspace_result_sha256=expected_subspace_result_sha256,
        expected_basis_receipt_sha256=expected_basis_receipt_sha256,
        expected_projector_sha256=expected_projector_sha256,
        expected_parent_out_proj_sha256=expected_parent_out_proj_sha256,
    )
    return result


def fold_lfm2_output_action_writer(
    subspace: Lfm2CompiledOutputActionSubspace,
    parent_out_proj_weight: Any,
    **expected_pins: str,
) -> Lfm2OutputActionWriterFoldResult:
    """Concise alias for :func:`compile_lfm2_output_action_writer_fold`."""

    return compile_lfm2_output_action_writer_fold(
        subspace,
        parent_out_proj_weight,
        **expected_pins,
    )


__all__ = [
    "LFM2_OUTPUT_ACTION_WRITER_FOLD_SCHEMA",
    "LFM2_OUTPUT_ACTION_WRITER_LINEAGE_SCHEMA",
    "LFM2_OUTPUT_ACTION_WRITER_STATE_SCHEMA",
    "Lfm2OutputActionWriterFoldError",
    "Lfm2OutputActionWriterFoldResult",
    "compile_lfm2_output_action_writer_fold",
    "fold_lfm2_output_action_writer",
    "verify_lfm2_output_action_writer_fold",
]
