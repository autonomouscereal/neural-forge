"""Sequential target-native CPCF compilation for one materialized LFM2 layer.

This module is the bridge between sealed parent circuit evidence and the
already-created dense blank child.  It deliberately compiles one layer in two
causal stages:

1. replay the current child full prefixes and compile/install the attention or
   short-convolution operator against those *actual* child coordinates;
2. replay the installed operator, then compile/install SwiGLU against the
   resulting actual post-operator residual and normalized MLP input.

Only ``fit`` phases enter attention and MLP closed-form solvers.  The
suffix-aware short-convolution compiler receives disjoint ``selection`` cases
for bounded solver choice and ``validation`` cases solely for its final
post-materialization evaluator; its own sealed report proves that neither
split enters any fit.

No model forward method, generation, cache update, backward pass, optimizer,
training, pruning, quantization, or fallback path is used here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.hf_lfm2_cpcf_boundary_bundle import (
    Lfm2CpcfLayerBoundaryCharts,
)
from neural_forge.hf_lfm2_cpcf_candidate_replay import (
    Lfm2CpcfCandidateReplayEvaluator,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import (
    Lfm2CpcfLayerReplayBank,
    Lfm2CpcfPhaseStateBank,
    Lfm2CpcfTargetPhase,
    replay_lfm2_cpcf_phase_bank,
)
from neural_forge.hf_lfm2_cpcf_short_conv_fold import (
    LFM2_CPCF_SHORT_CONV_FOLD_SCHEMA,
    Lfm2CompiledCpcfShortConvFold,
    Lfm2CpcfShortConvFoldCase,
    Lfm2CpcfShortConvGatePolicy,
    Lfm2CpcfShortConvValidationError,
    compile_lfm2_cpcf_short_conv_fold,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (
    LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
    Lfm2CpcfShortConvV4SolverPolicy,
    fixed_projection_trust_path_fractions,
    rank_short_conv_v4_selection_candidates,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v4_fold import (
    LFM2_CPCF_SHORT_CONV_V4_FAILURE_V1_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V4_FAILURE_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V4_FOLD_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_FAILURE_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_REQUEST_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_POOL_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_PROVENANCE_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_RAW_CANDIDATE_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_SELECTOR_FAILURE_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_SELECTOR_RECEIPT_SCHEMA,
    Lfm2CompiledCpcfShortConvV4Fold,
    Lfm2CpcfShortConvNumericalFrontierAdapter,
    Lfm2CpcfShortConvV4FoldCase,
    Lfm2CpcfShortConvV5CandidateSelector,
    compile_lfm2_cpcf_short_conv_v4_fold,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v4_policy import (
    parse_lfm2_cpcf_short_conv_v4_solver_policy,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    Lfm2CachedAttentionFoldCase,
    Lfm2CachedMlpFoldCase,
    Lfm2CompiledStructuralAttentionFold,
    Lfm2CompiledStructuralMlpFold,
    compile_lfm2_structural_residual_attention_fold,
    compile_lfm2_structural_residual_mlp_fold,
    install_lfm2_structural_attention_fold,
    install_lfm2_structural_mlp_fold,
)


LFM2_CPCF_LAYER_COMPILER_SCHEMA = "neural-forge-lfm2-cpcf-layer-compiler-v1"
LFM2_CPCF_LAYER_FAILURE_SCHEMA = "neural-forge-lfm2-cpcf-layer-failure-v3"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EVALUATION_SPLITS = ("fit", "validation")
_SHORT_CONV_CASE_SPLITS = ("selection", "fit", "validation")
_LAYER_TRANSACTION_STAGES = (
    "preinstall_replay",
    "operator_compile",
    "operator_install",
    "operator_replay",
    "mlp_compile",
    "mlp_install",
    "final_replay",
    "endpoint_evaluation",
    "immutability_audit",
    "result_seal",
)
_LAYER_ROLLBACK_PROOF_FIELDS = frozenset(
    {
        "target_state_sha256_before",
        "target_state_sha256_after",
        "target_full_inventory_sha256_before",
        "target_full_inventory_sha256_after",
        "target_tensor_bytes_restored",
        "target_tensor_identities_preserved",
        "target_tensor_aliases_preserved",
        "target_tensor_shapes_preserved",
        "target_tensor_dtypes_preserved",
        "target_all_named_parameters_and_buffers_restored",
        "target_parameter_and_buffer_registries_restored",
        "target_nonpersistent_buffer_membership_restored",
        "target_parameter_gradient_identity_and_bytes_restored",
        "target_parameter_gradient_storage_aliases_restored",
        "target_module_modes_restored",
        "source_state_sha256_before",
        "source_state_sha256_after",
        "source_full_inventory_sha256_before",
        "source_full_inventory_sha256_after",
        "source_state_preserved",
        "source_full_inventory_preserved",
        "evidence_sha256_before",
        "evidence_sha256_after",
        "evidence_preserved",
        "input_tensor_sha256_before",
        "input_tensor_sha256_after",
        "input_tensors_preserved",
        "boundary_charts_sha256_before",
        "boundary_charts_sha256_after",
        "boundary_charts_payload_sha256_before",
        "boundary_charts_payload_sha256_after",
        "boundary_charts_preserved",
        "exact_layer_rollback_proven",
        "proof_sha256",
    }
)
_LAYER_FAILURE_REPORT_FIELDS = frozenset(
    {
        "schema",
        "stage",
        "layer_index",
        "operator_kind",
        "input_bank_sha256",
        "input_tensor_sha256",
        "boundary_charts_sha256",
        "boundary_charts_payload_sha256",
        "cause",
        "nested_short_conv_validation_report",
        "nested_short_conv_validation_report_sha256",
        "nested_short_conv_validation_report_canonical_bytes_sha256",
        "nested_short_conv_validation_report_preserved_byte_for_byte",
        "rollback_proof",
        "layer_result_issued",
        "promotion_claimed",
        "report_sha256",
    }
)
_SHORT_CONV_VALIDATION_REPORT_FIELDS = frozenset(
    {
        "schema",
        "source_hidden_size",
        "target_hidden_size",
        "hidden_width_compression_ratio",
        "source_fir_length",
        "target_fir_length",
        "target_logical_parameters",
        "fit_case_count",
        "validation_case_count",
        "fit_metrics",
        "validation_metrics",
        "fit_case_reports",
        "validation_case_reports",
        "output_writer_scope",
        "producer_and_fir_scope",
        "fir_state_policy",
        "writer_fit_rows",
        "fit_prefix_rows",
        "fit_fir_warmup_rows",
        "provenance",
        "behavior_gate",
        "audit",
        "execution",
        "compiled_state_sha256",
        "report_sha256",
    }
)
_SHORT_CONV_GATE_POLICY_FIELDS = frozenset(
    {
        "maximum_fit_endpoint_relative_rmse",
        "maximum_validation_endpoint_relative_rmse",
        "minimum_fit_endpoint_cosine_similarity",
        "minimum_validation_endpoint_cosine_similarity",
        "enforce",
    }
)
_SHORT_CONV_GATE_CHECK_FIELDS = frozenset(
    {
        "fit_endpoint_relative_rmse",
        "validation_endpoint_relative_rmse",
        "fit_endpoint_cosine_similarity",
        "validation_endpoint_cosine_similarity",
    }
)
_SHORT_CONV_AUDIT_FIELDS = frozenset(
    {
        "source_inputs_sha256_before",
        "source_inputs_sha256_after",
        "source_inputs_unchanged",
        "target_templates_sha256_before",
        "target_templates_sha256_after",
        "target_templates_unchanged",
        "evidence_sha256_before",
        "evidence_sha256_after",
        "evidence_unchanged",
        "case_authentication_verified",
        "blank_target_values_used_by_compiler",
        "target_templates_used_for_shape_and_dtype_only",
        "returned_tensors_detached",
    }
)
_SHORT_CONV_EXECUTION_FIELDS = frozenset(
    {
        "closed_form_linear_solve",
        "deterministic_barycentric_channel_clustering",
        "parent_model_forward_calls",
        "target_model_forward_calls",
        "backward_calls",
        "optimizer_steps",
        "training_steps",
        "parameters_pruned",
        "parameters_quantized",
    }
)
_SHORT_CONV_V4_FAILURE_REPORT_FIELDS = frozenset(
    {
        "schema",
        "solver_schema",
        "solver_policy",
        "solver_policy_sha256",
        "gate_policy",
        "source_hidden_size",
        "target_hidden_size",
        "source_fir_length",
        "target_fir_length",
        "hidden_width_compression_ratio",
        "case_authentication_sha256",
        "basis",
        "input_lift",
        "transported_priors",
        "ridge_fit_reports",
        "candidate_fit_reports",
        "candidate_selection_reports",
        "candidate_pool_report",
        "selection_ranking",
        "selected_candidate_id",
        "fit_evaluation",
        "selection_evaluation",
        "validation_evaluation",
        "behavior_gate",
        "compiled_state_sha256",
        "source_channels_selected",
        "source_channels_dropped",
        "source_channels_pruned",
        "source_channels_quantized",
        "fit_only_parameter_solves",
        "selection_used_only_for_candidate_ranking",
        "validation_used_only_after_candidate_freeze",
        "execution",
        "compiled_short_conv_fold_issued",
        "promotion_claimed",
        "promotable_candidate",
        "failure_evidence_only",
        "report_sha256",
    }
)
_SHORT_CONV_V4_FAILURE_V1_REPORT_FIELDS = (
    _SHORT_CONV_V4_FAILURE_REPORT_FIELDS
    - {"ridge_fit_reports", "candidate_pool_report"}
)
_SHORT_CONV_V4_GATE_CHECK_FIELDS = frozenset(
    {
        "fit_endpoint_relative_rmse",
        "validation_endpoint_relative_rmse",
        "fit_endpoint_cosine_similarity",
        "validation_endpoint_cosine_similarity",
        "fit_context_relative_rmse",
        "validation_context_relative_rmse",
        "fit_context_cosine_similarity",
        "validation_context_cosine_similarity",
        "full_source_numeric_participation",
        "actual_fit_target_dtype_replay",
        "actual_validation_target_dtype_replay",
    }
)
_SHORT_CONV_V4_EXECUTION_FIELDS = frozenset(
    {
        "parent_model_forward_calls",
        "target_model_forward_calls",
        "generation_calls",
        "training_steps",
        "backward_calls",
        "optimizer_steps",
        "parameters_pruned",
        "parameters_quantized",
        "fallback_paths",
    }
)
_NO_SUCCESS_FIELDS = frozenset(
    {
        "accepted_artifact",
        "behavior_success_proven",
        "compaction_success_proven",
        "compiled_model_fold_issued",
        "layer_result_issued",
        "parameter_compaction_success_receipt_issued",
        "promotable_candidate",
        "promotion_authorized",
        "promotion_claimed",
        "success_claimed",
    }
)


class Lfm2CpcfLayerCompilerError(ValueError):
    """One target layer cannot be compiled from the supplied sealed evidence."""


class Lfm2CpcfLayerInstallError(RuntimeError):
    """A target layer installation failed and was rolled back."""


class Lfm2CpcfLayerCompilationFailure(Lfm2CpcfLayerInstallError):
    """A typed layer failure emitted only after exact rollback is proven."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        layer_index: int,
        operator_kind: str,
        input_bank_sha256: str,
        input_tensor_sha256: str,
        boundary_charts_sha256: str,
        boundary_charts_payload_sha256: str,
        rollback_proof: Mapping[str, Any],
        report: Mapping[str, Any],
        short_conv_validation_report: Mapping[str, Any] | None = None,
        short_conv_validation_report_bytes: bytes | None = None,
    ) -> None:
        super().__init__(message)
        if stage not in _LAYER_TRANSACTION_STAGES:
            raise Lfm2CpcfLayerCompilerError(
                "Layer failure stage is not a strict transaction stage."
            )
        index = _nonnegative_int(layer_index, "layer_index")
        if operator_kind not in {"full_attention", "short_conv"}:
            raise Lfm2CpcfLayerCompilerError(
                "Layer failure operator_kind is not canonical LFM2."
            )
        input_sha = _digest(input_bank_sha256, "input_bank_sha256")
        input_tensor_sha = _digest(
            input_tensor_sha256,
            "input_tensor_sha256",
        )
        charts_sha = _digest(
            boundary_charts_sha256,
            "boundary_charts_sha256",
        )
        charts_payload_sha = _digest(
            boundary_charts_payload_sha256,
            "boundary_charts_payload_sha256",
        )
        proof = dict(rollback_proof)
        proof_sha = _digest(
            proof.pop("proof_sha256", None),
            "rollback_proof.proof_sha256",
        )
        proof_with_seal = {**proof, "proof_sha256": proof_sha}
        for name in (
            "target_state_sha256_before",
            "target_state_sha256_after",
            "target_full_inventory_sha256_before",
            "target_full_inventory_sha256_after",
            "source_state_sha256_before",
            "source_state_sha256_after",
            "source_full_inventory_sha256_before",
            "source_full_inventory_sha256_after",
            "evidence_sha256_before",
            "evidence_sha256_after",
            "input_tensor_sha256_before",
            "input_tensor_sha256_after",
            "boundary_charts_sha256_before",
            "boundary_charts_sha256_after",
            "boundary_charts_payload_sha256_before",
            "boundary_charts_payload_sha256_after",
        ):
            _digest(proof.get(name), f"rollback_proof.{name}")
        true_proof_fields = (
            "target_tensor_bytes_restored",
            "target_tensor_identities_preserved",
            "target_tensor_aliases_preserved",
            "target_tensor_shapes_preserved",
            "target_tensor_dtypes_preserved",
            "target_all_named_parameters_and_buffers_restored",
            "target_parameter_and_buffer_registries_restored",
            "target_nonpersistent_buffer_membership_restored",
            "target_parameter_gradient_identity_and_bytes_restored",
            "target_parameter_gradient_storage_aliases_restored",
            "target_module_modes_restored",
            "source_state_preserved",
            "source_full_inventory_preserved",
            "evidence_preserved",
            "input_tensors_preserved",
            "boundary_charts_preserved",
            "exact_layer_rollback_proven",
        )
        if (
            set(proof_with_seal) != _LAYER_ROLLBACK_PROOF_FIELDS
            or _sha(proof) != proof_sha
            or any(proof.get(name) is not True for name in true_proof_fields)
            or proof.get("target_state_sha256_before")
            != proof.get("target_state_sha256_after")
            or proof.get("target_full_inventory_sha256_before")
            != proof.get("target_full_inventory_sha256_after")
            or proof.get("source_state_sha256_before")
            != proof.get("source_state_sha256_after")
            or proof.get("source_full_inventory_sha256_before")
            != proof.get("source_full_inventory_sha256_after")
            or proof.get("evidence_sha256_before") != proof.get("evidence_sha256_after")
            or proof.get("input_tensor_sha256_before")
            != proof.get("input_tensor_sha256_after")
            or proof.get("boundary_charts_sha256_before")
            != proof.get("boundary_charts_sha256_after")
            or proof.get("boundary_charts_payload_sha256_before")
            != proof.get("boundary_charts_payload_sha256_after")
            or proof.get("input_tensor_sha256_before") != input_tensor_sha
            or proof.get("boundary_charts_sha256_before") != charts_sha
            or proof.get("boundary_charts_payload_sha256_before") != charts_payload_sha
        ):
            raise Lfm2CpcfLayerInstallError(
                "Layer failure lacks an exact rollback proof."
            )
        failure_report = dict(report)
        report_sha = _digest(
            failure_report.pop("report_sha256", None),
            "failure_report.report_sha256",
        )
        report_with_seal = {
            **failure_report,
            "report_sha256": report_sha,
        }
        if (
            set(report_with_seal) != _LAYER_FAILURE_REPORT_FIELDS
            or _sha(failure_report) != report_sha
        ):
            raise Lfm2CpcfLayerCompilerError("Layer failure report seal is invalid.")
        if (
            failure_report.get("schema") != LFM2_CPCF_LAYER_FAILURE_SCHEMA
            or failure_report.get("stage") != stage
            or failure_report.get("layer_index") != index
            or failure_report.get("operator_kind") != operator_kind
            or failure_report.get("input_bank_sha256") != input_sha
            or failure_report.get("input_tensor_sha256") != input_tensor_sha
            or failure_report.get("boundary_charts_sha256") != charts_sha
            or failure_report.get("boundary_charts_payload_sha256")
            != charts_payload_sha
            or failure_report.get("rollback_proof") != proof_with_seal
            or failure_report.get("layer_result_issued") is not False
            or failure_report.get("promotion_claimed") is not False
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Layer failure report differs from its typed context."
            )
        cause = failure_report.get("cause")
        if (
            not isinstance(cause, Mapping)
            or set(cause) != {"module", "type", "message"}
            or any(
                not isinstance(cause.get(name), str)
                for name in ("module", "type", "message")
            )
        ):
            raise Lfm2CpcfLayerCompilerError("Layer failure cause is not canonical.")
        nested = short_conv_validation_report
        nested_bytes = short_conv_validation_report_bytes
        if (nested is None) != (nested_bytes is None):
            raise Lfm2CpcfLayerCompilerError(
                "Short-conv validation report and bytes must travel together."
            )
        if nested is not None:
            observed_bytes = _canonical_json(dict(nested))
            if observed_bytes != nested_bytes:
                raise Lfm2CpcfLayerCompilerError(
                    "Nested short-conv validation report bytes changed."
                )
            nested_body = dict(nested)
            nested_sha = _digest(
                nested_body.pop("report_sha256", None),
                "short_conv_validation_report.report_sha256",
            )
            if (
                operator_kind != "short_conv"
                or stage != "operator_compile"
                or _sha(nested_body) != nested_sha
                or not _valid_short_conv_validation_report(nested)
                or failure_report.get("nested_short_conv_validation_report")
                != dict(nested)
                or failure_report.get("nested_short_conv_validation_report_sha256")
                != nested_sha
                or failure_report.get(
                    "nested_short_conv_validation_report_canonical_bytes_sha256"
                )
                != hashlib.sha256(nested_bytes).hexdigest()
                or failure_report.get(
                    "nested_short_conv_validation_report_preserved_byte_for_byte"
                )
                is not True
            ):
                raise Lfm2CpcfLayerCompilerError(
                    "Nested short-conv validation evidence is not preserved "
                    "and self-sealed."
                )
        elif (
            failure_report.get("nested_short_conv_validation_report") is not None
            or failure_report.get("nested_short_conv_validation_report_sha256")
            is not None
            or failure_report.get(
                "nested_short_conv_validation_report_canonical_bytes_sha256"
            )
            is not None
            or failure_report.get(
                "nested_short_conv_validation_report_preserved_byte_for_byte"
            )
            is not False
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Absent short-conv validation evidence is inconsistent."
            )
        self.stage = stage
        self.layer_index = index
        self.operator_kind = operator_kind
        self.input_bank_sha256 = input_sha
        self.input_tensor_sha256 = input_tensor_sha
        self.boundary_charts_sha256 = charts_sha
        self.boundary_charts_payload_sha256 = charts_payload_sha
        self._rollback_proof_bytes = _canonical_json(proof_with_seal)
        self._short_conv_validation_report_bytes = nested_bytes
        self._report_bytes = _canonical_json(report_with_seal)

    @property
    def rollback_proof(self) -> Mapping[str, Any]:
        """Return an independent canonical copy of the rollback proof."""

        return json.loads(self._rollback_proof_bytes.decode("ascii"))

    @property
    def short_conv_validation_report(self) -> Mapping[str, Any] | None:
        """Return an independent canonical copy of nested gate evidence."""

        if self._short_conv_validation_report_bytes is None:
            return None
        return json.loads(self._short_conv_validation_report_bytes.decode("ascii"))

    @property
    def short_conv_validation_report_bytes(self) -> bytes | None:
        return self._short_conv_validation_report_bytes

    @property
    def report(self) -> Mapping[str, Any]:
        """Return an independent canonical copy of the failure report."""

        return json.loads(self._report_bytes.decode("ascii"))


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError("PyTorch is required for LFM2 CPCF compilation.") from exc
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
        raise Lfm2CpcfLayerCompilerError(
            "Layer-compiler metadata must be finite canonical JSON."
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2CpcfLayerCompilerError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2CpcfLayerCompilerError(f"{name} must be nonnegative.")
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2CpcfLayerCompilerError(f"{name} must be finite and nonnegative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CpcfLayerCompilerError(
            f"{name} must be finite and nonnegative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2CpcfLayerCompilerError(f"{name} must be finite and nonnegative.")
    return result


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError("Only tensors can be content authenticated.")
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
    }
    digest = hashlib.sha256(_canonical_json(descriptor))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tensor_bundle_sha256(values: Mapping[str, Any]) -> str:
    return _sha({name: _tensor_sha256(value) for name, value in sorted(values.items())})


def _short_conv_state_sha256(values: Mapping[str, Any]) -> str:
    """Match the suffix-aware short-conv compiler's sealed state digest."""

    torch = _torch()
    inventory = {}
    for name, value in sorted(values.items()):
        tensor = value.detach().to(device="cpu").contiguous()
        descriptor = {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": [int(size) for size in tensor.shape],
            "layout": str(torch.strided),
        }
        digest = hashlib.sha256(_canonical_json(descriptor))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
        inventory[name] = digest.hexdigest()
    return _sha(inventory)


def _model_core(model: Any, name: str) -> Any:
    core = getattr(model, "model", None)
    layers = getattr(core, "layers", None)
    if core is None or layers is None:
        raise Lfm2CpcfLayerCompilerError(
            f"{name} must expose an LFM2-compatible model.layers sequence."
        )
    model_type = getattr(getattr(model, "config", None), "model_type", "lfm2")
    if model_type != "lfm2":
        raise Lfm2CpcfLayerCompilerError(f"{name} must have model_type='lfm2'.")
    return core


def _layer(model: Any, index: int, name: str) -> Any:
    layers = tuple(_model_core(model, name).layers)
    if index >= len(layers):
        raise Lfm2CpcfLayerCompilerError(f"layer_index is outside {name}.model.layers.")
    return layers[index]


def _named_values_all(module: Any, method_name: str) -> tuple[tuple[str, Any], ...]:
    method = getattr(module, method_name, None)
    if not callable(method):
        raise Lfm2CpcfLayerCompilerError(f"Layer must expose {method_name}().")
    try:
        values = tuple(method(remove_duplicate=False))
    except TypeError:  # pragma: no cover - older supported Torch boundary
        values = tuple(method())
    names = [str(name) for name, _value in values]
    if len(set(names)) != len(names):
        raise Lfm2CpcfLayerCompilerError(
            f"Layer {method_name}() contains duplicate names."
        )
    return tuple((str(name), value) for name, value in values)


def _named_modules_all(module: Any) -> tuple[tuple[str, Any], ...]:
    return _named_values_all(module, "named_modules")


def _module_owner(
    module: Any,
    qualified_name: str,
) -> tuple[Any, str]:
    owner_name, _separator, local_name = qualified_name.rpartition(".")
    modules = dict(_named_modules_all(module))
    try:
        return modules[owner_name], local_name
    except KeyError as exc:
        raise Lfm2CpcfLayerCompilerError(
            f"Tensor owner for {qualified_name!r} is absent."
        ) from exc


def _module_tensors(module: Any) -> dict[str, Any]:
    torch = _torch()
    values = _named_values_all(module, "named_parameters") + _named_values_all(
        module, "named_buffers"
    )
    if not values:
        raise Lfm2CpcfLayerCompilerError("Layer has no named parameters or buffers.")
    if len({name for name, _value in values}) != len(values):
        raise Lfm2CpcfLayerCompilerError("Layer parameter and buffer names collide.")
    if any(not isinstance(value, torch.Tensor) for _name, value in values):
        raise Lfm2CpcfLayerCompilerError(
            "Layer parameters and buffers must be tensors."
        )
    return dict(values)


def _tensor_identity(value: Any) -> tuple[Any, ...]:
    storage_pointer = (
        0 if value.numel() == 0 else int(value.untyped_storage().data_ptr())
    )
    return (
        id(value),
        int(value.data_ptr()),
        storage_pointer,
        int(value.storage_offset()),
        tuple(int(item) for item in value.stride()),
        tuple(int(item) for item in value.shape),
        value.dtype,
        value.device.type,
        value.device.index,
    )


def _tensor_identity_record(value: Any) -> dict[str, Any]:
    identity = _tensor_identity(value)
    return {
        "object_id": identity[0],
        "data_pointer": identity[1],
        "storage_pointer": identity[2],
        "storage_offset": identity[3],
        "stride": list(identity[4]),
        "shape": list(identity[5]),
        "dtype": str(identity[6]).removeprefix("torch."),
        "device_type": identity[7],
        "device_index": identity[8],
    }


def _alias_groups(
    values: Mapping[str, Any],
    *,
    storage: bool,
) -> list[list[str]]:
    groups: dict[Any, list[str]] = {}
    for name, value in values.items():
        if storage and value.numel() != 0:
            key = (
                value.device.type,
                value.device.index,
                int(value.untyped_storage().data_ptr()),
            )
        elif storage:
            key = ("empty", id(value))
        else:
            key = id(value)
        groups.setdefault(key, []).append(name)
    return sorted(
        (sorted(names) for names in groups.values()),
        key=lambda names: names,
    )


def _module_registry_inventory(module: Any) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "object_id": id(value),
            "parameters": [
                [local_name, None if child is None else id(child)]
                for local_name, child in value._parameters.items()
            ],
            "buffers": [
                [local_name, None if child is None else id(child)]
                for local_name, child in value._buffers.items()
            ],
            "children": [
                [local_name, None if child is None else id(child)]
                for local_name, child in value._modules.items()
            ],
            "nonpersistent_buffers": sorted(
                str(item) for item in value._non_persistent_buffers_set
            ),
        }
        for name, value in _named_modules_all(module)
    ]


def _layer_full_inventory(layer: Any) -> dict[str, Any]:
    parameters = dict(_named_values_all(layer, "named_parameters"))
    buffers = dict(_named_values_all(layer, "named_buffers"))
    tensors = {**parameters, **buffers}
    if len(tensors) != len(parameters) + len(buffers):
        raise Lfm2CpcfLayerCompilerError("Layer parameter and buffer names collide.")
    tensor_inventory = {}
    for name, value in sorted(tensors.items()):
        owner, local_name = _module_owner(layer, name)
        tensor_inventory[name] = {
            **_tensor_identity_record(value),
            "role": "parameter" if name in parameters else "buffer",
            "persistent": (
                True
                if name in parameters
                else local_name not in owner._non_persistent_buffers_set
            ),
            "tensor_sha256": _tensor_sha256(value),
        }
    gradients = {}
    gradient_values = {}
    for name, parameter in sorted(parameters.items()):
        gradient = parameter.grad
        if gradient is not None:
            gradient_values[name] = gradient
        gradients[name] = {
            "requires_grad": bool(parameter.requires_grad),
            "present": gradient is not None,
            "identity": (
                None if gradient is None else _tensor_identity_record(gradient)
            ),
            "tensor_sha256": (None if gradient is None else _tensor_sha256(gradient)),
        }
    return {
        "tensor_inventory": tensor_inventory,
        "tensor_object_alias_groups": _alias_groups(
            tensors,
            storage=False,
        ),
        "tensor_storage_alias_groups": _alias_groups(
            tensors,
            storage=True,
        ),
        "parameter_gradients": gradients,
        "gradient_object_alias_groups": _alias_groups(
            gradient_values,
            storage=False,
        ),
        "gradient_storage_alias_groups": _alias_groups(
            gradient_values,
            storage=True,
        ),
        "module_registries": _module_registry_inventory(layer),
        "module_modes": [
            [name, id(value), bool(value.training)]
            for name, value in _named_modules_all(layer)
        ],
    }


@dataclass(frozen=True, slots=True)
class _GradientSnapshot:
    requires_grad: bool
    gradient: Any | None
    storage_anchor: Any | None
    value: Any | None
    identity: tuple[Any, ...] | None


@dataclass(frozen=True, slots=True)
class _ModuleRegistrySnapshot:
    module: Any
    parameters: tuple[tuple[str, Any], ...]
    buffers: tuple[tuple[str, Any], ...]
    children: tuple[tuple[str, Any], ...]
    nonpersistent_buffers: frozenset[str]


def _capture_module_registries(
    module: Any,
) -> tuple[_ModuleRegistrySnapshot, ...]:
    seen = set()
    result = []
    for _name, value in _named_modules_all(module):
        if id(value) in seen:
            continue
        seen.add(id(value))
        result.append(
            _ModuleRegistrySnapshot(
                module=value,
                parameters=tuple(value._parameters.items()),
                buffers=tuple(value._buffers.items()),
                children=tuple(value._modules.items()),
                nonpersistent_buffers=frozenset(value._non_persistent_buffers_set),
            )
        )
    return tuple(result)


def _restore_module_registries(
    snapshots: Sequence[_ModuleRegistrySnapshot],
) -> None:
    for snapshot in snapshots:
        module = snapshot.module
        module._parameters.clear()
        module._parameters.update(snapshot.parameters)
        module._buffers.clear()
        module._buffers.update(snapshot.buffers)
        module._modules.clear()
        module._modules.update(snapshot.children)
        module._non_persistent_buffers_set.clear()
        module._non_persistent_buffers_set.update(snapshot.nonpersistent_buffers)


@dataclass(frozen=True, slots=True)
class _LayerSnapshot:
    tensors: Mapping[str, Any]
    tensor_storage_anchors: Mapping[str, Any]
    tensor_identity: Mapping[str, tuple[Any, ...]]
    tensor_bytes_sha256: str
    full_inventory_sha256: str
    registry_inventory_sha256: str
    parameter_gradients: Mapping[str, _GradientSnapshot]
    modes: tuple[tuple[str, int, bool], ...]
    module_registries: tuple[_ModuleRegistrySnapshot, ...]


def _layer_snapshot(layer: Any) -> _LayerSnapshot:
    tensors = _module_tensors(layer)
    named_parameters = dict(_named_values_all(layer, "named_parameters"))
    gradients = {}
    for name, parameter in named_parameters.items():
        gradient = parameter.grad
        gradients[name] = _GradientSnapshot(
            requires_grad=bool(parameter.requires_grad),
            gradient=gradient,
            storage_anchor=(None if gradient is None else gradient.detach()),
            value=(
                None
                if gradient is None
                else gradient.detach().to(device="cpu").contiguous().clone()
            ),
            identity=(None if gradient is None else _tensor_identity(gradient)),
        )
    modes = tuple(
        (name, id(module), bool(module.training))
        for name, module in _named_modules_all(layer)
    )
    return _LayerSnapshot(
        tensors=MappingProxyType(
            {
                name: value.detach().to(device="cpu").contiguous().clone()
                for name, value in tensors.items()
            }
        ),
        tensor_storage_anchors=MappingProxyType(
            {name: value.detach() for name, value in tensors.items()}
        ),
        tensor_identity=MappingProxyType(
            {name: _tensor_identity(value) for name, value in tensors.items()}
        ),
        tensor_bytes_sha256=_tensor_bundle_sha256(tensors),
        full_inventory_sha256=_sha(_layer_full_inventory(layer)),
        registry_inventory_sha256=_sha(_module_registry_inventory(layer)),
        parameter_gradients=MappingProxyType(gradients),
        modes=modes,
        module_registries=_capture_module_registries(layer),
    )


def _restore_layer(layer: Any, snapshot: _LayerSnapshot) -> None:
    torch = _torch()
    _restore_module_registries(snapshot.module_registries)
    current = _module_tensors(layer)
    if set(current) != set(snapshot.tensors):
        raise Lfm2CpcfLayerInstallError(
            "Cannot roll back a target layer whose state keys changed."
        )
    with torch.no_grad():
        for name, tensor in current.items():
            if _tensor_identity(tensor) != snapshot.tensor_identity[name]:
                tensor.data = snapshot.tensor_storage_anchors[name]
            if _tensor_identity(tensor) != snapshot.tensor_identity[name]:
                raise Lfm2CpcfLayerInstallError(
                    "Cannot restore target tensor storage identity."
                )
            tensor.copy_(
                snapshot.tensors[name].to(
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
            )
    named_parameters = dict(_named_values_all(layer, "named_parameters"))
    if set(named_parameters) != set(snapshot.parameter_gradients):
        raise Lfm2CpcfLayerInstallError(
            "Cannot roll back a target layer whose Parameter set changed."
        )
    for name, parameter in named_parameters.items():
        gradient_snapshot = snapshot.parameter_gradients[name]
        parameter.requires_grad_(gradient_snapshot.requires_grad)
        parameter.grad = gradient_snapshot.gradient
        if gradient_snapshot.gradient is not None:
            gradient = gradient_snapshot.gradient
            with torch.no_grad():
                if _tensor_identity(gradient) != gradient_snapshot.identity:
                    gradient.data = gradient_snapshot.storage_anchor
                gradient.copy_(
                    gradient_snapshot.value.to(
                        device=gradient.device,
                        dtype=gradient.dtype,
                    )
                )
    modules = dict(_named_modules_all(layer))
    if set(modules) != {name for name, _id, _mode in snapshot.modes}:
        raise Lfm2CpcfLayerInstallError(
            "Cannot roll back a target layer whose module tree changed."
        )
    for name, identity, mode in snapshot.modes:
        if id(modules[name]) != identity:
            raise Lfm2CpcfLayerInstallError(
                "Cannot roll back a target layer whose module aliases changed."
            )
        modules[name].training = mode


def _assert_snapshot_identity(layer: Any, snapshot: _LayerSnapshot) -> None:
    current = _module_tensors(layer)
    if set(current) != set(snapshot.tensor_identity) or any(
        _tensor_identity(current[name]) != snapshot.tensor_identity[name]
        for name in current
    ):
        raise Lfm2CpcfLayerInstallError(
            "Target layer tensor identity, shape, or dtype changed during install."
        )
    named_parameters = dict(_named_values_all(layer, "named_parameters"))
    if set(named_parameters) != set(snapshot.parameter_gradients):
        raise Lfm2CpcfLayerInstallError(
            "Target layer Parameter set changed during install."
        )
    for name, parameter in _named_values_all(
        layer,
        "named_parameters",
    ):
        gradient = parameter.grad
        expected = snapshot.parameter_gradients[name]
        if (
            bool(parameter.requires_grad) != expected.requires_grad
            or gradient is not expected.gradient
            or (
                gradient is not None
                and (
                    _tensor_identity(gradient) != expected.identity
                    or _tensor_sha256(gradient) != _tensor_sha256(expected.value)
                )
            )
        ):
            raise Lfm2CpcfLayerInstallError(
                "Target layer gradient state changed during install."
            )
    modes = tuple(
        (name, id(module), bool(module.training))
        for name, module in _named_modules_all(layer)
    )
    if modes != snapshot.modes:
        raise Lfm2CpcfLayerInstallError(
            "Target layer training modes changed during install."
        )
    if _sha(_module_registry_inventory(layer)) != snapshot.registry_inventory_sha256:
        raise Lfm2CpcfLayerInstallError(
            "Target layer registries or buffer persistence changed."
        )


def _boundary_charts_payload_sha256(
    boundary_charts: Lfm2CpcfLayerBoundaryCharts,
) -> str:
    values = {}
    for boundary_name in ("post_operator", "post_layer"):
        chart = getattr(boundary_charts, boundary_name)
        mapping = chart.mapping
        values[f"{boundary_name}.encoder"] = mapping.encoder
        values[f"{boundary_name}.decoder"] = mapping.decoder
        for name, tensor in sorted(mapping.feature_predictors.items()):
            values[f"{boundary_name}.feature_predictors.{name}"] = tensor
    return _sha(
        {
            "layer_charts": boundary_charts.as_dict(),
            "post_operator_report": dict(boundary_charts.post_operator.report),
            "post_layer_report": dict(boundary_charts.post_layer.report),
            "tensor_payload": {
                name: {
                    **_tensor_identity_record(value),
                    "tensor_sha256": _tensor_sha256(value),
                }
                for name, value in sorted(values.items())
            },
        }
    )


def _live_layer_evidence_sha256(
    phases: Sequence[Lfm2CpcfTargetPhase],
    layer_index: int,
) -> str:
    inventory = []
    for phase in phases:
        evidence = phase.parent_evidence.layers[layer_index]
        inventory.append(
            {
                "identity": [phase.batch_index, phase.phase],
                "phase_sha256": phase.phase_sha256,
                "parent_evidence_sha256": (phase.parent_evidence.evidence_sha256),
                "layer_evidence": evidence.as_dict(),
                "layer_tensor_identity": {
                    name: {
                        **_tensor_identity_record(value),
                        "tensor_sha256": _tensor_sha256(value),
                    }
                    for name, value in sorted(evidence.tensors.items())
                },
            }
        )
    return _sha(inventory)


def _live_input_bank_sha256(
    input_bank: Lfm2CpcfPhaseStateBank,
) -> str:
    phases = []
    for phase in input_bank.phases:
        phases.append(
            {
                "phase": phase.as_dict(),
                "input_ids": {
                    **_tensor_identity_record(phase.input_ids),
                    "tensor_sha256": _tensor_sha256(phase.input_ids),
                },
                "attention_mask": {
                    **_tensor_identity_record(phase.attention_mask),
                    "tensor_sha256": _tensor_sha256(phase.attention_mask),
                },
                "position_ids": {
                    **_tensor_identity_record(phase.position_ids),
                    "tensor_sha256": _tensor_sha256(phase.position_ids),
                },
                "parent_evidence_binding": {
                    "cache_key": phase.parent_evidence.cache_key,
                    "batch_index": phase.parent_evidence.batch_index,
                    "input_sha256": phase.parent_evidence.input_sha256,
                    "phase": phase.parent_evidence.phase,
                    "evidence_sha256": (phase.parent_evidence.evidence_sha256),
                },
                "state": {
                    **_tensor_identity_record(input_bank.states[phase.identity]),
                    "tensor_sha256": _tensor_sha256(input_bank.states[phase.identity]),
                },
            }
        )
    return _sha(
        {
            "boundary": input_bank.boundary,
            "layer_index": input_bank.layer_index,
            "state_bank_sha256": input_bank.state_bank_sha256,
            "phases": phases,
        }
    )


def _prove_exact_layer_rollback(
    layer: Any,
    snapshot: _LayerSnapshot,
    *,
    source_layer: Any,
    source_snapshot: _LayerSnapshot,
    input_bank: Lfm2CpcfPhaseStateBank,
    layer_index: int,
    evidence_sha256_before: str,
    input_sha256_before: str,
    boundary_charts: Lfm2CpcfLayerBoundaryCharts,
    boundary_charts_sha256_before: str,
    boundary_charts_payload_sha256_before: str,
) -> Mapping[str, Any]:
    """Verify target rollback and all immutable transaction inputs."""

    _assert_snapshot_identity(layer, snapshot)
    state_after = _tensor_bundle_sha256(_module_tensors(layer))
    target_full_inventory_after = _sha(_layer_full_inventory(layer))
    if (
        state_after != snapshot.tensor_bytes_sha256
        or target_full_inventory_after != snapshot.full_inventory_sha256
    ):
        raise Lfm2CpcfLayerInstallError(
            "Target layer tensor bytes failed exact rollback."
        )
    _assert_snapshot_identity(source_layer, source_snapshot)
    source_state_after = _tensor_bundle_sha256(_module_tensors(source_layer))
    source_full_inventory_after = _sha(_layer_full_inventory(source_layer))
    evidence_after = _live_layer_evidence_sha256(
        input_bank.phases,
        layer_index,
    )
    input_after = _live_input_bank_sha256(input_bank)
    boundary_payload_after = _boundary_charts_payload_sha256(boundary_charts)
    if (
        source_state_after != source_snapshot.tensor_bytes_sha256
        or source_full_inventory_after != source_snapshot.full_inventory_sha256
        or evidence_after != evidence_sha256_before
        or input_after != input_sha256_before
        or boundary_charts.layer_charts_sha256 != boundary_charts_sha256_before
        or boundary_payload_after != boundary_charts_payload_sha256_before
    ):
        raise Lfm2CpcfLayerInstallError(
            "Layer failure changed source, evidence, incoming states, or "
            "boundary charts."
        )
    body = {
        "target_state_sha256_before": snapshot.tensor_bytes_sha256,
        "target_state_sha256_after": state_after,
        "target_full_inventory_sha256_before": (snapshot.full_inventory_sha256),
        "target_full_inventory_sha256_after": (target_full_inventory_after),
        "target_tensor_bytes_restored": True,
        "target_tensor_identities_preserved": True,
        "target_tensor_aliases_preserved": True,
        "target_tensor_shapes_preserved": True,
        "target_tensor_dtypes_preserved": True,
        "target_all_named_parameters_and_buffers_restored": True,
        "target_parameter_and_buffer_registries_restored": True,
        "target_nonpersistent_buffer_membership_restored": True,
        "target_parameter_gradient_identity_and_bytes_restored": True,
        "target_parameter_gradient_storage_aliases_restored": True,
        "target_module_modes_restored": True,
        "source_state_sha256_before": source_snapshot.tensor_bytes_sha256,
        "source_state_sha256_after": source_state_after,
        "source_full_inventory_sha256_before": (source_snapshot.full_inventory_sha256),
        "source_full_inventory_sha256_after": (source_full_inventory_after),
        "source_state_preserved": True,
        "source_full_inventory_preserved": True,
        "evidence_sha256_before": evidence_sha256_before,
        "evidence_sha256_after": evidence_after,
        "evidence_preserved": True,
        "input_tensor_sha256_before": input_sha256_before,
        "input_tensor_sha256_after": input_after,
        "input_tensors_preserved": True,
        "boundary_charts_sha256_before": (boundary_charts_sha256_before),
        "boundary_charts_sha256_after": (boundary_charts.layer_charts_sha256),
        "boundary_charts_payload_sha256_before": (
            boundary_charts_payload_sha256_before
        ),
        "boundary_charts_payload_sha256_after": (boundary_payload_after),
        "boundary_charts_preserved": True,
        "exact_layer_rollback_proven": True,
    }
    return MappingProxyType({**body, "proof_sha256": _sha(body)})


def _evidence_tensors(
    phases: Sequence[Lfm2CpcfTargetPhase],
    layer_index: int,
) -> dict[str, Any]:
    result = {}
    for phase in phases:
        evidence = phase.parent_evidence.layers[layer_index]
        for name, value in evidence.tensors.items():
            result[f"{phase.batch_index}:{phase.phase}:layer.{layer_index}:{name}"] = (
                value
            )
    return result


def _phase_evidence(
    phase: Lfm2CpcfTargetPhase,
    layer_index: int,
    operator_kind: str,
) -> Mapping[str, Any]:
    try:
        evidence = phase.parent_evidence.layers[layer_index]
    except IndexError as exc:
        raise Lfm2CpcfLayerCompilerError(
            f"Phase {phase.identity!r} lacks layer {layer_index} evidence."
        ) from exc
    if evidence.layer_index != layer_index or evidence.operator_kind != operator_kind:
        raise Lfm2CpcfLayerCompilerError(
            f"Phase {phase.identity!r} has the wrong layer evidence."
        )
    return evidence.tensors


def _finite_tensor(
    value: Any,
    name: str,
    *,
    dimensions: int,
) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != dimensions
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
    ):
        raise Lfm2CpcfLayerCompilerError(
            f"{name} must be a nonempty dense floating rank-{dimensions} tensor."
        )
    result = value.detach().to(device="cpu").contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CpcfLayerCompilerError(f"{name} contains NaN or infinity.")
    return result


def _weight(module: Any, name: str, *, dimensions: int) -> Any:
    return _finite_tensor(
        getattr(module, "weight", None),
        f"{name}.weight",
        dimensions=dimensions,
    )


def _variance_epsilon(module: Any, name: str) -> float:
    value = getattr(module, "variance_epsilon", None)
    epsilon = _nonnegative_float(value, f"{name}.variance_epsilon")
    if epsilon == 0.0:
        raise Lfm2CpcfLayerCompilerError(f"{name}.variance_epsilon must be positive.")
    return epsilon


def _rotate_half(value: Any) -> Any:
    half = int(value.shape[-1]) // 2
    return _torch().cat((-value[..., half:], value[..., :half]), dim=-1)


def _rope_terms(
    model: Any,
    position_ids: Any,
    *,
    dtype: Any,
) -> tuple[Any, Any]:
    torch = _torch()
    core = _model_core(model, "model")
    rotary = getattr(core, "rotary_emb", None)
    inv_freq = getattr(rotary, "inv_freq", None)
    scaling = getattr(rotary, "attention_scaling", None)
    if (
        not isinstance(inv_freq, torch.Tensor)
        or inv_freq.ndim != 1
        or inv_freq.numel() == 0
        or isinstance(scaling, bool)
    ):
        raise Lfm2CpcfLayerCompilerError(
            "LFM2 rotary embedding lacks static inv_freq/attention_scaling."
        )
    try:
        scale = float(scaling)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CpcfLayerCompilerError(
            "LFM2 RoPE attention scaling is invalid."
        ) from exc
    if not math.isfinite(scale) or scale <= 0.0:
        raise Lfm2CpcfLayerCompilerError(
            "LFM2 RoPE attention scaling must be positive and finite."
        )
    positions = position_ids.detach().to(device="cpu", dtype=torch.float32)
    frequencies = positions.unsqueeze(-1) * inv_freq.detach().to(
        device="cpu",
        dtype=torch.float32,
    ).reshape(1, 1, -1)
    angles = torch.cat((frequencies, frequencies), dim=-1)
    return (
        (angles.cos() * scale).to(dtype=dtype).contiguous(),
        (angles.sin() * scale).to(dtype=dtype).contiguous(),
    )


def invert_lfm2_rope(
    rotated: Any,
    cosine: Any,
    sine: Any,
) -> Any:
    """Invert duplicated-half LFM2 RoPE, including a finite non-unit radius."""

    torch = _torch()
    value = _finite_tensor(rotated, "rotated", dimensions=4)
    cos = _finite_tensor(cosine, "cosine", dimensions=3)
    sin = _finite_tensor(sine, "sine", dimensions=3)
    if (
        int(value.shape[-1]) % 2
        or tuple(cos.shape)
        != (int(value.shape[0]), int(value.shape[2]), int(value.shape[3]))
        or tuple(sin.shape) != tuple(cos.shape)
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Rotated rows and RoPE terms have incompatible shapes."
        )
    half = int(cos.shape[-1]) // 2
    if not torch.equal(cos[..., :half], cos[..., half:]) or not torch.equal(
        sin[..., :half],
        sin[..., half:],
    ):
        raise Lfm2CpcfLayerCompilerError(
            "LFM2 RoPE terms do not have duplicated-half pair structure."
        )
    radius = cos.square() + sin.square()
    floor = torch.finfo(torch.float32).tiny
    if bool((radius.to(torch.float32) <= floor).any().item()):
        raise Lfm2CpcfLayerCompilerError("Cannot invert a zero-radius RoPE row.")
    numerator = value * cos.unsqueeze(1) - _rotate_half(value) * sin.unsqueeze(1)
    result = numerator / radius.unsqueeze(1)
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CpcfLayerCompilerError("Inverse RoPE produced a non-finite row.")
    return result.to(dtype=value.dtype).contiguous()


def _runtime_rms_norm(value: Any, weight: Any, epsilon: float) -> Any:
    torch = _torch()
    input_dtype = value.dtype
    direction = value.to(torch.float32)
    variance = direction.square().mean(dim=-1, keepdim=True)
    direction = direction * torch.rsqrt(variance + epsilon)
    return weight.to(dtype=input_dtype) * direction.to(dtype=input_dtype)


def _rmsnorm_preimage(
    normalized: Any,
    weight: Any,
    *,
    epsilon: float,
) -> Any:
    """Choose a deterministic finite-precision preimage of an RMSNorm row."""

    torch = _torch()
    values = _finite_tensor(normalized, "normalized", dimensions=4)
    norm_weight = _finite_tensor(weight, "weight", dimensions=1)
    width = int(values.shape[-1])
    if tuple(norm_weight.shape) != (width,):
        raise Lfm2CpcfLayerCompilerError(
            "RMSNorm weight width differs from normalized rows."
        )
    weight64 = norm_weight.to(torch.float64)
    if bool((weight64.abs() <= torch.finfo(torch.float64).tiny).any().item()):
        raise Lfm2CpcfLayerCompilerError(
            "Cannot recover an RMSNorm preimage through a zero learned weight."
        )
    rows = values.reshape(-1, width)
    direction = rows.to(torch.float64) / weight64.reshape(1, width)
    mean_square = direction.square().mean(dim=-1, keepdim=True)
    denominator = 1.0 - mean_square
    ideal = torch.sqrt(
        torch.as_tensor(epsilon, dtype=torch.float64)
        / denominator.clamp_min(torch.finfo(torch.float64).eps)
    )
    ideal = torch.where(
        denominator > torch.finfo(torch.float32).eps,
        ideal,
        torch.full_like(ideal, 128.0),
    ).clamp(min=2.0**-8, max=2.0**16)
    scales = torch.stack(
        (
            ideal,
            torch.ones_like(ideal),
            torch.full_like(ideal, 16.0),
            torch.full_like(ideal, 128.0),
            torch.full_like(ideal, 1024.0),
        ),
        dim=0,
    )
    candidates = (direction.unsqueeze(0) * scales).to(dtype=values.dtype)
    predictions = _runtime_rms_norm(
        candidates,
        norm_weight.reshape(1, 1, width),
        epsilon,
    )
    targets = rows.unsqueeze(0)
    errors = (
        (predictions.to(torch.float64) - targets.to(torch.float64))
        .square()
        .mean(dim=-1)
    )
    choice = errors.argmin(dim=0)
    row_indices = torch.arange(int(rows.shape[0]))
    selected = candidates[choice, row_indices, :]
    return selected.reshape_as(values).contiguous()


def recover_lfm2_full_key_linear_history(
    *,
    parent_key_history: Any,
    parent_key_norm_weight: Any,
    full_cosine: Any,
    full_sine: Any,
    norm_epsilon: float,
    cached_current_key_linear: Any | None = None,
    cached_current_normalized_pre_rope: Any | None = None,
) -> Any:
    """Recover full-prefix pre-norm K rows from a sealed post-RoPE history.

    Decode traces contain exact post-RoPE K history but only current-step
    projection rows.  RoPE is inverted analytically and RMSNorm is assigned a
    deterministic finite-precision preimage.  Exact cached current rows then
    replace the reconstructed suffix.
    """

    torch = _torch()
    epsilon = _nonnegative_float(norm_epsilon, "norm_epsilon")
    if epsilon == 0.0:
        raise Lfm2CpcfLayerCompilerError("norm_epsilon must be positive.")
    history = _finite_tensor(
        parent_key_history,
        "parent_key_history",
        dimensions=4,
    )
    normalized = invert_lfm2_rope(history, full_cosine, full_sine)
    if cached_current_normalized_pre_rope is not None:
        cached_normalized = _finite_tensor(
            cached_current_normalized_pre_rope,
            "cached_current_normalized_pre_rope",
            dimensions=4,
        )
        current = int(cached_normalized.shape[2])
        if (
            tuple(cached_normalized.shape[:2]) != tuple(normalized.shape[:2])
            or int(cached_normalized.shape[3]) != int(normalized.shape[3])
            or current > int(normalized.shape[2])
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Cached current normalized K rows do not align to K history."
            )
        tolerance = (
            4.0 * torch.finfo(history.dtype).eps
            if history.dtype in {torch.float16, torch.bfloat16}
            else 2e-5
        )
        if not bool(
            torch.allclose(
                normalized[:, :, -current:, :].to(torch.float32),
                cached_normalized.to(torch.float32),
                rtol=tolerance,
                atol=tolerance,
            )
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Inverse RoPE disagrees with cached current pre-RoPE K rows."
            )
    linear = _rmsnorm_preimage(
        normalized,
        parent_key_norm_weight,
        epsilon=epsilon,
    )
    if cached_current_key_linear is not None:
        cached_linear = _finite_tensor(
            cached_current_key_linear,
            "cached_current_key_linear",
            dimensions=4,
        )
        current = int(cached_linear.shape[2])
        if (
            tuple(cached_linear.shape[:2]) != tuple(linear.shape[:2])
            or int(cached_linear.shape[3]) != int(linear.shape[3])
            or current > int(linear.shape[2])
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Cached current linear K rows do not align to K history."
            )
        linear = linear.clone()
        linear[:, :, -current:, :] = cached_linear.to(dtype=linear.dtype)
    reconstructed_normalized = _runtime_rms_norm(
        linear,
        _finite_tensor(
            parent_key_norm_weight,
            "parent_key_norm_weight",
            dimensions=1,
        ).reshape(1, 1, 1, -1),
        epsilon,
    )
    reconstructed = reconstructed_normalized * full_cosine.unsqueeze(1) + _rotate_half(
        reconstructed_normalized
    ) * full_sine.unsqueeze(1)
    tolerance = (
        8.0 * torch.finfo(history.dtype).eps
        if history.dtype in {torch.float16, torch.bfloat16}
        else 5e-5
    )
    if not bool(
        torch.allclose(
            reconstructed.to(torch.float32),
            history.to(torch.float32),
            rtol=tolerance,
            atol=tolerance,
        )
    ):
        maximum = float(
            (reconstructed.to(torch.float32) - history.to(torch.float32))
            .abs()
            .max()
            .item()
        )
        raise Lfm2CpcfLayerCompilerError(
            "Recovered full K-linear history does not reproduce the sealed "
            f"post-RoPE history (maximum absolute error {maximum:.6g})."
        )
    return linear


def _same_rope(
    source_model: Any,
    target_model: Any,
    phase: Lfm2CpcfTargetPhase,
    *,
    dtype: Any,
) -> tuple[Any, Any]:
    torch = _torch()
    source_cos, source_sin = _rope_terms(
        source_model,
        phase.position_ids,
        dtype=dtype,
    )
    target_cos, target_sin = _rope_terms(
        target_model,
        phase.position_ids,
        dtype=dtype,
    )
    tolerance = (
        2.0 * torch.finfo(dtype).eps
        if dtype in {torch.float16, torch.bfloat16}
        else 1e-6
    )
    if not (
        torch.allclose(source_cos, target_cos, rtol=tolerance, atol=tolerance)
        and torch.allclose(source_sin, target_sin, rtol=tolerance, atol=tolerance)
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Source and target LFM2 RoPE differ; this preserved-head-dimension "
            "compiler requires an explicit rotary bridge."
        )
    return source_cos, source_sin


def _copy_attention_norms(
    source_attention: Any, target_attention: Any
) -> Mapping[str, Any]:
    torch = _torch()
    source = {
        "q_layernorm.weight": _weight(
            source_attention.q_layernorm,
            "source_attention.q_layernorm",
            dimensions=1,
        ),
        "k_layernorm.weight": _weight(
            source_attention.k_layernorm,
            "source_attention.k_layernorm",
            dimensions=1,
        ),
    }
    target = {
        "q_layernorm.weight": getattr(
            getattr(target_attention, "q_layernorm", None),
            "weight",
            None,
        ),
        "k_layernorm.weight": getattr(
            getattr(target_attention, "k_layernorm", None),
            "weight",
            None,
        ),
    }
    if any(not isinstance(value, torch.Tensor) for value in target.values()):
        raise Lfm2CpcfLayerCompilerError("Target attention lacks Q/K RMSNorm weights.")
    if any(tuple(target[name].shape) != tuple(source[name].shape) for name in source):
        raise Lfm2CpcfLayerCompilerError(
            "Source/target attention norm widths differ; route this layer "
            "through the attention-dimension compiler."
        )
    identities = {name: id(value) for name, value in target.items()}
    gradients = {
        name: (
            bool(value.requires_grad),
            None if value.grad is None else id(value.grad),
            None if value.grad is None else _tensor_sha256(value.grad),
        )
        for name, value in target.items()
    }
    with torch.no_grad():
        for name, value in target.items():
            value.copy_(source[name].to(device=value.device, dtype=value.dtype))
    body = {
        "scope": "attention_qk_norm_transport",
        "weights": {
            name: _tensor_sha256(value) for name, value in sorted(target.items())
        },
        "parameter_identities_preserved": all(
            id(target[name]) == identities[name] for name in target
        ),
        "gradient_state_preserved": all(
            (
                bool(target[name].requires_grad),
                None if target[name].grad is None else id(target[name].grad),
                None
                if target[name].grad is None
                else _tensor_sha256(target[name].grad),
            )
            == gradients[name]
            for name in target
        ),
        "source_derived": True,
        "training_steps": 0,
    }
    if (
        not body["parameter_identities_preserved"]
        or not body["gradient_state_preserved"]
    ):
        raise Lfm2CpcfLayerInstallError(
            "Attention Q/K norm transport changed parameter or gradient identity."
        )
    return MappingProxyType({**body, "report_sha256": _sha(body)})


def _attention_key_mask(additive_mask: Any) -> Any:
    torch = _torch()
    value = _finite_tensor(additive_mask, "attention.additive_mask", dimensions=4)
    if bool((value > 0).any().item()):
        raise Lfm2CpcfLayerCompilerError(
            "Parent additive attention mask contains a positive score."
        )
    allowed = value == 0
    if int(allowed.shape[1]) > 1:
        first = allowed[:, :1, :, :]
        if not bool(torch.equal(allowed, first.expand_as(allowed))):
            raise Lfm2CpcfLayerCompilerError(
                "Head-specific parent masks cannot be reduced to one GQA mask."
            )
    return allowed[:, 0, :, :].contiguous()


def _attention_cases(
    *,
    source_model: Any,
    target_model: Any,
    source_layer: Any,
    layer_index: int,
    input_bank: Lfm2CpcfPhaseStateBank,
    preinstall_replay: Lfm2CpcfLayerReplayBank,
    chart: Lfm2CpcfLayerBoundaryCharts,
) -> tuple[Lfm2CachedAttentionFoldCase, ...]:
    """Assemble one complete causal attention trajectory per fit record.

    The sealed prefill contains every raw K-linear row.  Each decode phase
    contains the one newly executed K-linear suffix plus the complete
    post-RoPE K/V history.  Concatenating those exact raw rows is both more
    faithful and more stable than assigning an artificial RMSNorm preimage to
    every historical key.  It also avoids fitting the same prompt prefix once
    per decode phase.
    """

    torch = _torch()
    source_attention = getattr(source_layer, "self_attn", None)
    if source_attention is None:
        raise Lfm2CpcfLayerCompilerError("Source attention layer lacks self_attn.")
    q_epsilon = _variance_epsilon(
        source_attention.q_layernorm,
        "source_attention.q_layernorm",
    )
    k_epsilon = _variance_epsilon(
        source_attention.k_layernorm,
        "source_attention.k_layernorm",
    )
    if not math.isclose(q_epsilon, k_epsilon, rel_tol=0.0, abs_tol=0.0):
        raise Lfm2CpcfLayerCompilerError("Source Q/K normalization epsilon differs.")
    grouped: dict[
        tuple[int, str],
        list[Lfm2CpcfTargetPhase],
    ] = {}
    for phase in input_bank.phases:
        if phase.split == "fit":
            grouped.setdefault(
                (phase.batch_index, phase.record_id),
                [],
            ).append(phase)

    result = []
    for (batch_index, record_id), trajectory in grouped.items():
        query_linear_parts = []
        query_parts = []
        context_parts = []
        endpoint_parts = []
        key_linear_history = None
        key_history = None
        value_history = None
        key_mask_history = None
        previous_phase: Lfm2CpcfTargetPhase | None = None
        previous_child_input = None
        previous_child_state = None
        previous_input_ids = None
        previous_attention_mask = None
        previous_position_ids = None
        final_phase: Lfm2CpcfTargetPhase | None = None
        final_child_input = None
        final_child_state = None
        final_cos = None
        final_sin = None

        for phase_index, phase in enumerate(trajectory):
            expected_phase = (
                "prefill" if phase_index == 0 else f"decode.{phase_index - 1:03d}"
            )
            if phase.phase != expected_phase:
                raise Lfm2CpcfLayerCompilerError(
                    f"Attention trajectory {batch_index}:{record_id} skips "
                    f"or reorders causal phase {expected_phase!r}."
                )
            if previous_phase is not None and (
                phase.split != previous_phase.split
                or phase.full_prefix_tokens
                != (previous_phase.full_prefix_tokens + phase.sampled_suffix_tokens)
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Attention trajectory {batch_index}:{record_id} changed "
                    "split or target prefix geometry."
                )

            tensors = _phase_evidence(
                phase,
                layer_index,
                "full_attention",
            )
            replay = preinstall_replay.replays[phase.identity]
            parent_q_linear = _finite_tensor(
                tensors["attention.q_linear"],
                f"phase {phase.identity!r} attention.q_linear",
                dimensions=4,
            )
            parent_k_linear = _finite_tensor(
                tensors["attention.k_linear"],
                f"phase {phase.identity!r} attention.k_linear",
                dimensions=4,
            )
            parent_query = _finite_tensor(
                tensors["attention.q"],
                f"phase {phase.identity!r} attention.q",
                dimensions=4,
            )
            parent_context = _finite_tensor(
                tensors["attention.context"],
                f"phase {phase.identity!r} attention.context",
                dimensions=4,
            )
            current_key = _finite_tensor(
                tensors["attention.k"],
                f"phase {phase.identity!r} attention.k",
                dimensions=4,
            )
            current_value = _finite_tensor(
                tensors["attention.v"],
                f"phase {phase.identity!r} attention.v",
                dimensions=4,
            )
            observed_key_history = _finite_tensor(
                tensors["attention.key_history"],
                f"phase {phase.identity!r} attention.key_history",
                dimensions=4,
            )
            observed_value_history = _finite_tensor(
                tensors["attention.value_history"],
                f"phase {phase.identity!r} attention.value_history",
                dimensions=4,
            )
            current = phase.sampled_suffix_tokens
            if (
                int(parent_q_linear.shape[2]) != current
                or int(parent_k_linear.shape[2]) != current
                or int(parent_query.shape[2]) != current
                or int(parent_context.shape[2]) != current
                or int(current_key.shape[2]) != current
                or int(current_value.shape[2]) != current
                or int(observed_key_history.shape[2]) != phase.full_prefix_tokens
                or tuple(observed_value_history.shape)
                != tuple(observed_key_history.shape)
                or tuple(parent_q_linear.shape) != tuple(parent_query.shape)
                or tuple(parent_k_linear.shape) != tuple(current_key.shape)
                or tuple(current_value.shape) != tuple(current_key.shape)
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} attention rows do not describe "
                    "one complete history plus one aligned terminal suffix."
                )

            full_cos, full_sin = _same_rope(
                source_model,
                target_model,
                phase,
                dtype=parent_q_linear.dtype,
            )
            cached_cos = _finite_tensor(
                tensors["attention.rope_cos"],
                f"phase {phase.identity!r} attention.rope_cos",
                dimensions=3,
            )
            cached_sin = _finite_tensor(
                tensors["attention.rope_sin"],
                f"phase {phase.identity!r} attention.rope_sin",
                dimensions=3,
            )
            if tuple(cached_cos.shape) != (
                phase.batch_size,
                current,
                int(parent_q_linear.shape[3]),
            ) or tuple(cached_sin.shape) != tuple(cached_cos.shape):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} cached RoPE is not its exact "
                    "terminal query/key suffix."
                )
            tolerance = (
                4.0 * torch.finfo(parent_q_linear.dtype).eps
                if parent_q_linear.dtype in {torch.float16, torch.bfloat16}
                else 2e-5
            )
            if not (
                torch.allclose(
                    full_cos[:, -current:, :].to(torch.float32),
                    cached_cos.to(torch.float32),
                    rtol=tolerance,
                    atol=tolerance,
                )
                and torch.allclose(
                    full_sin[:, -current:, :].to(torch.float32),
                    cached_sin.to(torch.float32),
                    rtol=tolerance,
                    atol=tolerance,
                )
            ):
                raise Lfm2CpcfLayerCompilerError(
                    "Static source RoPE differs from sealed parent cache "
                    f"evidence for phase {phase.identity!r}."
                )
            if not (
                torch.equal(
                    observed_key_history[:, :, -current:, :],
                    current_key,
                )
                and torch.equal(
                    observed_value_history[:, :, -current:, :],
                    current_value,
                )
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} current K/V rows are not the "
                    "exact terminal suffix of their sealed histories."
                )

            observed_mask = _attention_key_mask(tensors["attention.additive_mask"])
            if tuple(observed_mask.shape) != (
                phase.batch_size,
                current,
                phase.full_prefix_tokens,
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} attention mask does not cover "
                    "its terminal queries over the complete key history."
                )
            child_input = _finite_tensor(
                replay.operator_input,
                f"phase {phase.identity!r} actual child operator input",
                dimensions=3,
            )
            child_state = _finite_tensor(
                input_bank.state(phase),
                f"phase {phase.identity!r} actual child pre-operator state",
                dimensions=3,
            )
            if tuple(child_input.shape[:2]) != (
                phase.batch_size,
                phase.full_prefix_tokens,
            ) or tuple(child_state.shape[:2]) != (
                phase.batch_size,
                phase.full_prefix_tokens,
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} actual child states do not "
                    "cover their complete target prefix."
                )

            if previous_phase is None:
                if current != phase.full_prefix_tokens:
                    raise Lfm2CpcfLayerCompilerError(
                        f"Phase {phase.identity!r} must seal the complete "
                        "prefill Q/K/V trajectory, not a sampled tail."
                    )
                key_linear_history = parent_k_linear
                key_history = observed_key_history
                value_history = observed_value_history
                key_mask_history = observed_mask
            else:
                previous_tokens = previous_phase.full_prefix_tokens
                if not (
                    _floating_causal_prefix_matches(
                        previous_child_input,
                        child_input,
                    )
                    and _floating_causal_prefix_matches(
                        previous_child_state,
                        child_state,
                    )
                    and _integer_causal_prefix_matches(
                        previous_input_ids,
                        phase.input_ids,
                    )
                    and _integer_causal_prefix_matches(
                        previous_attention_mask,
                        phase.attention_mask,
                    )
                    and _integer_causal_prefix_matches(
                        previous_position_ids,
                        phase.position_ids,
                    )
                ):
                    raise Lfm2CpcfLayerCompilerError(
                        f"Phase {phase.identity!r} changed an authenticated "
                        "causal child prefix."
                    )
                if (
                    key_linear_history is None
                    or key_history is None
                    or value_history is None
                    or key_mask_history is None
                    or int(key_linear_history.shape[2]) != previous_tokens
                    or not torch.equal(
                        observed_key_history[:, :, :previous_tokens, :],
                        key_history,
                    )
                    or not torch.equal(
                        observed_value_history[:, :, :previous_tokens, :],
                        value_history,
                    )
                ):
                    raise Lfm2CpcfLayerCompilerError(
                        f"Phase {phase.identity!r} changed its exact sealed "
                        "parent K/V causal prefix."
                    )
                key_linear_history = torch.cat(
                    (key_linear_history, parent_k_linear),
                    dim=2,
                ).contiguous()
                key_history = observed_key_history
                value_history = observed_value_history
                padded_previous_mask = torch.nn.functional.pad(
                    key_mask_history,
                    (0, current),
                    value=False,
                )
                key_mask_history = torch.cat(
                    (padded_previous_mask, observed_mask),
                    dim=1,
                ).contiguous()

            endpoint = chart.post_operator.encode_shaped(
                tensors["residual.post_operator"]
            )
            if (
                int(endpoint.shape[1]) != current
                or int(key_linear_history.shape[2]) != phase.full_prefix_tokens
                or tuple(key_mask_history.shape)
                != (
                    phase.batch_size,
                    phase.full_prefix_tokens,
                    phase.full_prefix_tokens,
                )
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} did not extend one contiguous "
                    "attention trajectory."
                )
            query_linear_parts.append(parent_q_linear)
            query_parts.append(parent_query)
            context_parts.append(parent_context)
            endpoint_parts.append(endpoint)
            previous_phase = phase
            previous_child_input = child_input
            previous_child_state = child_state
            previous_input_ids = phase.input_ids
            previous_attention_mask = phase.attention_mask
            previous_position_ids = phase.position_ids
            final_phase = phase
            final_child_input = child_input
            final_child_state = child_state
            final_cos = full_cos
            final_sin = full_sin

        if (
            final_phase is None
            or final_child_input is None
            or final_child_state is None
            or final_cos is None
            or final_sin is None
            or key_linear_history is None
            or key_history is None
            or value_history is None
            or key_mask_history is None
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Attention trajectory did not produce one complete causal case."
            )
        parent_query_linear = torch.cat(
            query_linear_parts,
            dim=2,
        ).contiguous()
        parent_query = torch.cat(query_parts, dim=2).contiguous()
        parent_context = torch.cat(context_parts, dim=2).contiguous()
        parent_endpoint = torch.cat(endpoint_parts, dim=1).contiguous()
        if not (
            int(parent_query_linear.shape[2])
            == int(parent_query.shape[2])
            == int(parent_context.shape[2])
            == int(parent_endpoint.shape[1])
            == final_phase.full_prefix_tokens
        ):
            raise Lfm2CpcfLayerCompilerError(
                f"Attention trajectory {batch_index}:{record_id} did not "
                "cover every causal query row exactly once."
            )
        result.append(
            Lfm2CachedAttentionFoldCase(
                target_query_input=final_child_input,
                target_key_value_input=final_child_input,
                parent_query_linear=parent_query_linear,
                parent_key_linear=key_linear_history,
                parent_value=value_history,
                parent_query=parent_query,
                parent_key=key_history,
                parent_context=parent_context,
                parent_post_attention_residual_target_space=parent_endpoint,
                target_pre_attention_residual=final_child_state,
                query_cosine=final_cos,
                query_sine=final_sin,
                key_cosine=final_cos,
                key_sine=final_sin,
                key_mask=key_mask_history,
            )
        )
    if not result:
        raise Lfm2CpcfLayerCompilerError(
            "Attention compilation requires at least one fit trajectory."
        )
    return tuple(result)


_SHORT_CONV_HISTORY_FIELDS = (
    "short_conv.b_history",
    "short_conv.c_history",
    "short_conv.x_history",
)
_SHORT_CONV_V4_HISTORY_FIELDS = (
    "operator.input",
    *_SHORT_CONV_HISTORY_FIELDS,
)


def _extend_short_conv_parent_histories(
    phase: Lfm2CpcfTargetPhase,
    tensors: Mapping[str, Any],
    *,
    layer_index: int,
    histories: dict[tuple[int, str], Any],
    include_operator_input: bool = False,
) -> dict[str, Any]:
    """Rebuild full causal parent histories from cached decode suffixes.

    The strict tokenizer trace adapter maps one target row to one donor row
    without changing batch or sequence geometry.  A prefill therefore seals
    the complete prompt history, while each cached decode phase contributes
    only its newly executed terminal suffix.  Both short-convolution solvers
    consume authenticated B/C/x histories.  V4 additionally consumes the
    complete parent operator-input history for its child-to-parent affine
    lift; it is never reconstructed from projected B/C/x values.
    """

    torch = _torch()
    history_fields = (
        _SHORT_CONV_V4_HISTORY_FIELDS
        if include_operator_input
        else _SHORT_CONV_HISTORY_FIELDS
    )
    result = {}
    previous_fields = {
        field: histories.get((phase.batch_index, field)) for field in history_fields
    }
    for field in history_fields:
        current = _finite_tensor(
            tensors[field],
            f"phase {phase.identity!r} layer {layer_index} {field}",
            dimensions=3,
        )
        if int(current.shape[0]) != phase.batch_size:
            raise Lfm2CpcfLayerCompilerError(
                f"Phase {phase.identity!r} {field} changed its batch boundary."
            )
        key = (phase.batch_index, field)
        if phase.phase == "prefill":
            if key in histories or int(current.shape[1]) != phase.full_prefix_tokens:
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} {field} is not one complete "
                    "causal prefill history."
                )
            history = current
        else:
            previous = histories.get(key)
            previous_tokens = phase.full_prefix_tokens - phase.sampled_suffix_tokens
            if (
                previous is None
                or tuple(previous.shape[:1]) != tuple(current.shape[:1])
                or int(previous.shape[1]) != previous_tokens
                or int(previous.shape[2]) != int(current.shape[2])
                or previous.dtype != current.dtype
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} {field} is not a causal "
                    "continuation of its authenticated parent history."
                )
            current_tokens = int(current.shape[1])
            if current_tokens == phase.sampled_suffix_tokens:
                history = torch.cat((previous, current), dim=1).contiguous()
            elif current_tokens == phase.full_prefix_tokens:
                if not torch.equal(current[:, :previous_tokens, :], previous):
                    raise Lfm2CpcfLayerCompilerError(
                        f"Phase {phase.identity!r} full {field} changed its "
                        "authenticated causal prefix."
                    )
                history = current
            else:
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} {field} contains neither one "
                    "terminal suffix nor one complete causal history."
                )
        if int(history.shape[1]) != phase.full_prefix_tokens:
            raise Lfm2CpcfLayerCompilerError(
                f"Phase {phase.identity!r} {field} did not reconstruct its "
                "complete target-aligned prefix."
            )
        histories[key] = history
        result[field] = history
    if phase.phase != "prefill":
        previous_b = previous_fields["short_conv.b_history"]
        previous_x = previous_fields["short_conv.x_history"]
        if previous_b is None or previous_x is None:
            raise Lfm2CpcfLayerCompilerError(
                f"Phase {phase.identity!r} has no authenticated short-conv "
                "warm-state prefix."
            )
        pre_bx_state = _finite_tensor(
            tensors["short_conv.pre_bx_state"],
            (f"phase {phase.identity!r} layer {layer_index} short_conv.pre_bx_state"),
            dimensions=3,
        )
        fir_length = int(pre_bx_state.shape[-1])
        previous_bx = (previous_b * previous_x).transpose(1, 2).contiguous()
        if int(previous_bx.shape[-1]) < fir_length:
            previous_bx = torch.nn.functional.pad(
                previous_bx,
                (fir_length - int(previous_bx.shape[-1]), 0),
            )
        expected_pre_state = previous_bx[..., -fir_length:].contiguous()
        if (
            tuple(pre_bx_state.shape) != tuple(expected_pre_state.shape)
            or pre_bx_state.dtype != expected_pre_state.dtype
            or not torch.equal(pre_bx_state, expected_pre_state)
        ):
            raise Lfm2CpcfLayerCompilerError(
                f"Phase {phase.identity!r} short-conv warm state is not the "
                "exact authenticated tail of its reconstructed causal prefix."
            )
    return result


def _floating_causal_prefix_matches(previous: Any, current: Any) -> bool:
    torch = _torch()
    if (
        not isinstance(previous, torch.Tensor)
        or not isinstance(current, torch.Tensor)
        or previous.ndim != current.ndim
        or previous.ndim < 2
        or previous.dtype != current.dtype
        or tuple(previous.shape[:1]) != tuple(current.shape[:1])
        or tuple(previous.shape[2:]) != tuple(current.shape[2:])
        or int(previous.shape[1]) > int(current.shape[1])
    ):
        return False
    tolerance = 8.0 * torch.finfo(previous.dtype).eps
    return bool(
        torch.allclose(
            current[:, : int(previous.shape[1]), ...].to(torch.float64),
            previous.to(torch.float64),
            rtol=tolerance,
            atol=tolerance,
        )
    )


def _integer_causal_prefix_matches(previous: Any, current: Any) -> bool:
    torch = _torch()
    return bool(
        isinstance(previous, torch.Tensor)
        and isinstance(current, torch.Tensor)
        and previous.ndim == current.ndim
        and previous.ndim >= 2
        and previous.dtype == current.dtype
        and tuple(previous.shape[:1]) == tuple(current.shape[:1])
        and tuple(previous.shape[2:]) == tuple(current.shape[2:])
        and int(previous.shape[1]) <= int(current.shape[1])
        and torch.equal(
            current[:, : int(previous.shape[1]), ...],
            previous,
        )
    )


def _short_conv_cases(
    *,
    layer_index: int,
    input_bank: Lfm2CpcfPhaseStateBank,
    preinstall_replay: Lfm2CpcfLayerReplayBank,
    chart: Lfm2CpcfLayerBoundaryCharts,
    use_v4: bool = False,
) -> tuple[
    Lfm2CpcfShortConvFoldCase | Lfm2CpcfShortConvV4FoldCase,
    ...,
]:
    grouped: dict[tuple[int, str], list[Lfm2CpcfTargetPhase]] = {}
    for phase in input_bank.phases:
        if phase.split in _SHORT_CONV_CASE_SPLITS:
            grouped.setdefault(
                (phase.batch_index, phase.record_id),
                [],
            ).append(phase)

    result = []
    for (_batch_index, _record_id), trajectory in grouped.items():
        histories: dict[tuple[int, str], Any] = {}
        parent_histories: dict[str, Any] | None = None
        endpoint_parts = []
        endpoint_start: int | None = None
        endpoint_end: int | None = None
        previous_phase: Lfm2CpcfTargetPhase | None = None
        previous_child_input = None
        previous_child_state = None
        previous_input_ids = None
        previous_attention_mask = None
        previous_position_ids = None
        final_phase: Lfm2CpcfTargetPhase | None = None
        final_child_input = None
        final_child_state = None
        for phase_index, phase in enumerate(trajectory):
            expected_phase = (
                "prefill" if phase_index == 0 else f"decode.{phase_index - 1:03d}"
            )
            if phase.phase != expected_phase:
                raise Lfm2CpcfLayerCompilerError(
                    f"Short-conv trajectory {phase.batch_index}:"
                    f"{phase.record_id} skips or reorders causal phase "
                    f"{expected_phase!r}."
                )
            if previous_phase is not None and (
                phase.split != previous_phase.split
                or phase.full_prefix_tokens
                != (previous_phase.full_prefix_tokens + phase.sampled_suffix_tokens)
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Short-conv trajectory {phase.batch_index}:"
                    f"{phase.record_id} changed split or target prefix "
                    "geometry."
                )
            tensors = _phase_evidence(phase, layer_index, "short_conv")
            parent_histories = _extend_short_conv_parent_histories(
                phase,
                tensors,
                layer_index=layer_index,
                histories=histories,
                include_operator_input=use_v4,
            )
            replay = preinstall_replay.replays[phase.identity]
            child_input = _finite_tensor(
                replay.operator_input,
                f"phase {phase.identity!r} actual child operator input",
                dimensions=3,
            )
            child_state = _finite_tensor(
                input_bank.state(phase),
                f"phase {phase.identity!r} actual child pre-operator state",
                dimensions=3,
            )
            if tuple(child_input.shape[:2]) != (
                phase.batch_size,
                phase.full_prefix_tokens,
            ) or tuple(child_state.shape[:2]) != (
                phase.batch_size,
                phase.full_prefix_tokens,
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} actual child states do not "
                    "cover their complete target prefix."
                )
            if previous_phase is not None and not (
                _floating_causal_prefix_matches(
                    previous_child_input,
                    child_input,
                )
                and _floating_causal_prefix_matches(
                    previous_child_state,
                    child_state,
                )
                and _integer_causal_prefix_matches(
                    previous_input_ids,
                    phase.input_ids,
                )
                and _integer_causal_prefix_matches(
                    previous_attention_mask,
                    phase.attention_mask,
                )
                and _integer_causal_prefix_matches(
                    previous_position_ids,
                    phase.position_ids,
                )
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} changed an authenticated causal "
                    "child prefix."
                )
            endpoint = chart.post_operator.encode_shaped(
                tensors["residual.post_operator"]
            )
            current_start = phase.full_prefix_tokens - phase.sampled_suffix_tokens
            if int(endpoint.shape[1]) != phase.sampled_suffix_tokens or (
                endpoint_end is not None and current_start != endpoint_end
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} residual endpoint is not the "
                    "next contiguous causal suffix."
                )
            if endpoint_start is None:
                endpoint_start = current_start
            endpoint_end = current_start + int(endpoint.shape[1])
            endpoint_parts.append(endpoint)
            previous_phase = phase
            previous_child_input = child_input
            previous_child_state = child_state
            previous_input_ids = phase.input_ids
            previous_attention_mask = phase.attention_mask
            previous_position_ids = phase.position_ids
            final_phase = phase
            final_child_input = child_input
            final_child_state = child_state
        if (
            final_phase is None
            or final_child_input is None
            or final_child_state is None
            or parent_histories is None
            or endpoint_start is None
            or endpoint_end != final_phase.full_prefix_tokens
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Short-conv trajectory did not produce one complete causal case."
            )
        endpoint = _torch().cat(endpoint_parts, dim=1).contiguous()
        actual_pre = final_child_state[:, endpoint_start:, :].contiguous()
        case_arguments = {
            "case_id": (
                f"{final_phase.batch_index}:{final_phase.record_id}:trajectory:"
                f"layer.{layer_index}"
            ),
            "split": final_phase.split,
            "suffix_start": endpoint_start,
            "actual_child_operator_input_history": final_child_input,
            "parent_b_history": parent_histories["short_conv.b_history"],
            "parent_c_history": parent_histories["short_conv.c_history"],
            "parent_x_history": parent_histories["short_conv.x_history"],
            "parent_post_operator_residual_target_space_suffix": endpoint,
            "actual_child_pre_operator_residual_suffix": actual_pre,
            "valid_prefix_mask": final_phase.attention_mask,
        }
        if use_v4:
            result.append(
                Lfm2CpcfShortConvV4FoldCase(
                    **case_arguments,
                    parent_operator_input_history=parent_histories["operator.input"],
                )
            )
        else:
            result.append(Lfm2CpcfShortConvFoldCase(**case_arguments))
    if not any(value.split == "fit" for value in result) or not any(
        value.split == "validation" for value in result
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Short-conv compilation requires fit and validation phases."
        )
    if use_v4 and not any(value.split == "selection" for value in result):
        raise Lfm2CpcfLayerCompilerError(
            "Short-conv v4 compilation requires selection phases."
        )
    return tuple(result)


def _mlp_cases(
    *,
    layer_index: int,
    input_bank: Lfm2CpcfPhaseStateBank,
    operator_replay: Lfm2CpcfLayerReplayBank,
    chart: Lfm2CpcfLayerBoundaryCharts,
) -> tuple[Lfm2CachedMlpFoldCase, ...]:
    result = []
    for phase in input_bank.phases:
        if phase.split != "fit":
            continue
        tensors = _phase_evidence(phase, layer_index, chart.operator_kind)
        replay = operator_replay.replays[phase.identity]
        target_input = phase.suffix(replay.mlp_input)
        target_pre = phase.suffix(replay.post_operator_residual)
        parent_endpoint = chart.post_layer.encode_shaped(tensors["residual.post_layer"])
        result.append(
            Lfm2CachedMlpFoldCase(
                target_input=target_input.reshape(-1, int(target_input.shape[-1])),
                parent_gate=tensors["mlp.gate"].reshape(
                    -1,
                    int(tensors["mlp.gate"].shape[-1]),
                ),
                parent_up=tensors["mlp.up"].reshape(
                    -1,
                    int(tensors["mlp.up"].shape[-1]),
                ),
                parent_post_swiglu=tensors["mlp.post_swiglu"].reshape(
                    -1,
                    int(tensors["mlp.post_swiglu"].shape[-1]),
                ),
                parent_post_mlp_residual_target_space=parent_endpoint.reshape(
                    -1,
                    int(parent_endpoint.shape[-1]),
                ),
                target_pre_mlp_residual=target_pre.reshape(
                    -1,
                    int(target_pre.shape[-1]),
                ),
            )
        )
    if not result:
        raise Lfm2CpcfLayerCompilerError(
            "SwiGLU compilation requires at least one fit phase."
        )
    return tuple(result)


def _install_short_conv(
    target_short_conv: Any,
    compiled: Lfm2CompiledCpcfShortConvFold | Lfm2CompiledCpcfShortConvV4Fold,
) -> Mapping[str, Any]:
    torch = _torch()
    if not isinstance(
        compiled,
        (Lfm2CompiledCpcfShortConvFold, Lfm2CompiledCpcfShortConvV4Fold),
    ):
        raise TypeError("compiled must be an authenticated LFM2 CPCF short-conv fold.")
    if (
        isinstance(
            compiled,
            Lfm2CompiledCpcfShortConvV4Fold,
        )
        and not compiled.verify_authentication()
    ):
        raise Lfm2CpcfLayerInstallError(
            "Compiled short-conv v4 authentication failed before install."
        )
    available = {
        "in_proj.weight": getattr(
            getattr(target_short_conv, "in_proj", None),
            "weight",
            None,
        ),
        "in_proj.bias": getattr(
            getattr(target_short_conv, "in_proj", None),
            "bias",
            None,
        ),
        "conv.weight": getattr(
            getattr(target_short_conv, "conv", None),
            "weight",
            None,
        ),
        "conv.bias": getattr(
            getattr(target_short_conv, "conv", None),
            "bias",
            None,
        ),
        "out_proj.weight": getattr(
            getattr(target_short_conv, "out_proj", None),
            "weight",
            None,
        ),
        "out_proj.bias": getattr(
            getattr(target_short_conv, "out_proj", None),
            "bias",
            None,
        ),
    }
    expected = compiled.state_dict()
    if set(expected) != {
        name for name, value in available.items() if isinstance(value, torch.Tensor)
    }:
        raise Lfm2CpcfLayerInstallError(
            "Target short-conv bias topology differs from compiled state."
        )
    for name, expected_value in expected.items():
        observed = available[name]
        if (
            not isinstance(observed, torch.Tensor)
            or tuple(observed.shape) != tuple(expected_value.shape)
            or observed.dtype != expected_value.dtype
        ):
            raise Lfm2CpcfLayerInstallError(
                f"Target short-conv {name} differs in shape or dtype."
            )
    identities = {name: id(available[name]) for name in expected}
    gradients = {
        name: (
            bool(available[name].requires_grad),
            None if available[name].grad is None else id(available[name].grad),
            None
            if available[name].grad is None
            else _tensor_sha256(available[name].grad),
        )
        for name in expected
    }
    backup = {
        name: available[name].detach().to(device="cpu").clone() for name in expected
    }
    try:
        with torch.no_grad():
            for name, expected_value in expected.items():
                available[name].copy_(
                    expected_value.to(
                        device=available[name].device,
                        dtype=available[name].dtype,
                    )
                )
        installed_sha = _short_conv_state_sha256(
            {name: available[name] for name in expected}
        )
        body = {
            "scope": (
                "authenticated_residual_endpoint_cpcf_short_conv_v4_install"
                if isinstance(compiled, Lfm2CompiledCpcfShortConvV4Fold)
                else "suffix_aware_cpcf_short_conv_install"
            ),
            "parameter_identities_preserved": all(
                id(available[name]) == identities[name] for name in expected
            ),
            "parameter_shapes_preserved": True,
            "gradient_state_preserved": all(
                (
                    bool(available[name].requires_grad),
                    None if available[name].grad is None else id(available[name].grad),
                    None
                    if available[name].grad is None
                    else _tensor_sha256(available[name].grad),
                )
                == gradients[name]
                for name in expected
            ),
            "compiled_state_sha256": compiled.report["compiled_state_sha256"],
            "installed_state_sha256": installed_sha,
            "target_logical_parameters": sum(
                int(available[name].numel()) for name in expected
            ),
            "training_steps": 0,
        }
        if (
            not body["parameter_identities_preserved"]
            or not body["gradient_state_preserved"]
            or body["compiled_state_sha256"] != installed_sha
        ):
            raise Lfm2CpcfLayerInstallError(
                "Installed short-conv state failed identity or content checks."
            )
        return MappingProxyType({**body, "report_sha256": _sha(body)})
    except Exception:
        with torch.no_grad():
            for name, backup_value in backup.items():
                available[name].copy_(
                    backup_value.to(
                        device=available[name].device,
                        dtype=available[name].dtype,
                    )
                )
        raise


def _metrics(prediction: Any, target: Any) -> Mapping[str, Any]:
    torch = _torch()
    left = prediction.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    right = target.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if tuple(left.shape) != tuple(right.shape) or left.numel() == 0:
        raise Lfm2CpcfLayerCompilerError(
            "Endpoint metric tensors are empty or misaligned."
        )
    difference = left - right
    rmse = torch.sqrt(torch.mean(difference.square()))
    target_scale = torch.sqrt(torch.mean(right.square())).clamp_min(
        torch.finfo(torch.float64).eps
    )
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    cosine = (
        1.0
        if float(left_norm.item()) == 0.0 and float(right_norm.item()) == 0.0
        else (
            0.0
            if float(left_norm.item()) == 0.0 or float(right_norm.item()) == 0.0
            else float(torch.dot(left, right).div(left_norm * right_norm).item())
        )
    )
    return MappingProxyType(
        {
            "rows": int(prediction.numel() // prediction.shape[-1]),
            "width": int(prediction.shape[-1]),
            "rmse": float(rmse.item()),
            "relative_rmse": float((rmse / target_scale).item()),
            "cosine_similarity": cosine,
            "maximum_absolute_error": float(difference.abs().max().item()),
            "prediction_sha256": _tensor_sha256(prediction),
            "target_sha256": _tensor_sha256(target),
        }
    )


def _split_endpoint_metrics(
    *,
    layer_index: int,
    input_bank: Lfm2CpcfPhaseStateBank,
    operator_replay: Lfm2CpcfLayerReplayBank,
    final_replay: Lfm2CpcfLayerReplayBank,
    chart: Lfm2CpcfLayerBoundaryCharts,
) -> Mapping[str, Any]:
    torch = _torch()
    result = {}
    for split in _EVALUATION_SPLITS:
        operator_predictions = []
        operator_targets = []
        layer_predictions = []
        layer_targets = []
        phase_seals = []
        for phase in input_bank.phases:
            if phase.split != split:
                continue
            tensors = _phase_evidence(phase, layer_index, chart.operator_kind)
            operator_predictions.append(
                phase.suffix(
                    operator_replay.replays[phase.identity].post_operator_residual
                ).to(device="cpu")
            )
            operator_targets.append(
                chart.post_operator.encode_shaped(tensors["residual.post_operator"]).to(
                    device="cpu"
                )
            )
            layer_predictions.append(
                phase.suffix(
                    final_replay.replays[phase.identity].post_layer_residual
                ).to(device="cpu")
            )
            layer_targets.append(
                chart.post_layer.encode_shaped(tensors["residual.post_layer"]).to(
                    device="cpu"
                )
            )
            phase_seals.append(phase.phase_sha256)
        if not operator_predictions:
            raise Lfm2CpcfLayerCompilerError(
                f"Layer evaluation requires at least one {split} phase."
            )
        operator_prediction = torch.cat(
            [value.reshape(-1, int(value.shape[-1])) for value in operator_predictions],
            dim=0,
        )
        operator_target = torch.cat(
            [value.reshape(-1, int(value.shape[-1])) for value in operator_targets],
            dim=0,
        )
        layer_prediction = torch.cat(
            [value.reshape(-1, int(value.shape[-1])) for value in layer_predictions],
            dim=0,
        )
        layer_target = torch.cat(
            [value.reshape(-1, int(value.shape[-1])) for value in layer_targets],
            dim=0,
        )
        body = {
            "split": split,
            "phase_count": len(phase_seals),
            "phase_sha256": phase_seals,
            "post_operator": dict(_metrics(operator_prediction, operator_target)),
            "post_layer": dict(_metrics(layer_prediction, layer_target)),
        }
        result[split] = {**body, "report_sha256": _sha(body)}
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class Lfm2CpcfCompiledLayerResult:
    """One installed layer and its actual-child next-boundary state bank."""

    layer_index: int
    operator_kind: str
    input_bank_sha256: str
    operator_artifact: Any
    mlp_artifact: Lfm2CompiledStructuralMlpFold
    operator_install_report: Mapping[str, Any]
    mlp_install_report: Mapping[str, Any]
    preinstall_replay: Lfm2CpcfLayerReplayBank
    operator_replay: Lfm2CpcfLayerReplayBank
    final_replay: Lfm2CpcfLayerReplayBank
    split_metrics: Mapping[str, Any]
    report: Mapping[str, Any]
    result_sha256: str

    def __post_init__(self) -> None:
        index = _nonnegative_int(self.layer_index, "layer_index")
        if self.operator_kind not in {"full_attention", "short_conv"}:
            raise Lfm2CpcfLayerCompilerError("operator_kind is not canonical LFM2.")
        _digest(self.input_bank_sha256, "input_bank_sha256")
        if self.operator_kind == "full_attention":
            if not isinstance(
                self.operator_artifact,
                Lfm2CompiledStructuralAttentionFold,
            ):
                raise TypeError("Attention result has the wrong operator artifact.")
        elif not isinstance(
            self.operator_artifact,
            (Lfm2CompiledCpcfShortConvFold, Lfm2CompiledCpcfShortConvV4Fold),
        ):
            raise TypeError("Short-conv result has the wrong operator artifact.")
        if not isinstance(
            self.mlp_artifact,
            Lfm2CompiledStructuralMlpFold,
        ):
            raise TypeError("mlp_artifact has the wrong type.")
        if (
            self.preinstall_replay.layer_index != index
            or self.operator_replay.layer_index != index
            or self.final_replay.layer_index != index
            or self.final_replay.post_layer_bank.layer_index != index + 1
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Layer replay sequence is incomplete or out of order."
            )
        if set(self.split_metrics) != set(_EVALUATION_SPLITS):
            raise Lfm2CpcfLayerCompilerError(
                "split_metrics must contain fit and validation."
            )
        result_sha = _digest(self.result_sha256, "result_sha256")
        report = dict(self.report)
        if (
            report.pop("result_sha256", None) != result_sha
            or _sha(report) != result_sha
        ):
            raise Lfm2CpcfLayerCompilerError("Layer compiler result seal is invalid.")
        object.__setattr__(
            self,
            "operator_install_report",
            MappingProxyType(dict(self.operator_install_report)),
        )
        object.__setattr__(
            self,
            "mlp_install_report",
            MappingProxyType(dict(self.mlp_install_report)),
        )
        object.__setattr__(
            self,
            "split_metrics",
            MappingProxyType(dict(self.split_metrics)),
        )
        object.__setattr__(
            self,
            "report",
            MappingProxyType({**report, "result_sha256": result_sha}),
        )

    @property
    def next_bank(self) -> Lfm2CpcfPhaseStateBank:
        return self.final_replay.post_layer_bank

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def _no_recursive_success_claims(value: Any) -> bool:
    if isinstance(value, Mapping):
        for name, child in value.items():
            if name in _NO_SUCCESS_FIELDS and child is not False:
                return False
            if not _no_recursive_success_claims(child):
                return False
        return True
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return all(_no_recursive_success_claims(child) for child in value)
    return True


def _valid_short_conv_v1_validation_report(
    report: Mapping[str, Any],
) -> bool:
    behavior_gate = report.get("behavior_gate")
    policy = behavior_gate.get("policy") if isinstance(behavior_gate, Mapping) else None
    checks = behavior_gate.get("checks") if isinstance(behavior_gate, Mapping) else None
    audit = report.get("audit")
    execution = report.get("execution")
    audit_hashes_match = bool(
        isinstance(audit, Mapping)
        and all(
            isinstance(audit.get(name), str)
            and _SHA256.fullmatch(audit[name]) is not None
            for name in (
                "source_inputs_sha256_before",
                "source_inputs_sha256_after",
                "target_templates_sha256_before",
                "target_templates_sha256_after",
                "evidence_sha256_before",
                "evidence_sha256_after",
            )
        )
        and audit.get("source_inputs_sha256_before")
        == audit.get("source_inputs_sha256_after")
        and audit.get("target_templates_sha256_before")
        == audit.get("target_templates_sha256_after")
        and audit.get("evidence_sha256_before") == audit.get("evidence_sha256_after")
    )
    return bool(
        set(report) == _SHORT_CONV_VALIDATION_REPORT_FIELDS
        and report.get("schema") == LFM2_CPCF_SHORT_CONV_FOLD_SCHEMA
        and isinstance(report.get("compiled_state_sha256"), str)
        and _SHA256.fullmatch(report["compiled_state_sha256"]) is not None
        and isinstance(behavior_gate, Mapping)
        and set(behavior_gate)
        == {
            "policy",
            "checks",
            "passed",
            "heldout_validation_used_for_fitting",
        }
        and behavior_gate.get("passed") is False
        and behavior_gate.get("heldout_validation_used_for_fitting") is False
        and isinstance(policy, Mapping)
        and set(policy) == _SHORT_CONV_GATE_POLICY_FIELDS
        and policy.get("enforce") is True
        and all(
            not isinstance(policy.get(name), bool)
            and isinstance(policy.get(name), (int, float))
            and math.isfinite(float(policy[name]))
            for name in _SHORT_CONV_GATE_POLICY_FIELDS - {"enforce"}
        )
        and isinstance(checks, Mapping)
        and set(checks) == _SHORT_CONV_GATE_CHECK_FIELDS
        and all(isinstance(value, bool) for value in checks.values())
        and not all(checks.values())
        and isinstance(audit, Mapping)
        and set(audit) == _SHORT_CONV_AUDIT_FIELDS
        and audit_hashes_match
        and audit.get("source_inputs_unchanged") is True
        and audit.get("target_templates_unchanged") is True
        and audit.get("evidence_unchanged") is True
        and audit.get("case_authentication_verified") is True
        and audit.get("blank_target_values_used_by_compiler") is False
        and audit.get("target_templates_used_for_shape_and_dtype_only") is True
        and audit.get("returned_tensors_detached") is True
        and isinstance(execution, Mapping)
        and set(execution) == _SHORT_CONV_EXECUTION_FIELDS
        and execution.get("closed_form_linear_solve") is True
        and execution.get("deterministic_barycentric_channel_clustering") is True
        and all(
            type(execution.get(name)) is int and execution[name] == 0
            for name in (
                "parent_model_forward_calls",
                "target_model_forward_calls",
                "backward_calls",
                "optimizer_steps",
                "training_steps",
                "parameters_pruned",
                "parameters_quantized",
            )
        )
        and _no_recursive_success_claims(report)
    )


def _recursive_report_seals_valid(value: Any) -> bool:
    if isinstance(value, Mapping):
        document = dict(value)
        if "report_sha256" in document:
            seal = document.pop("report_sha256")
            if (
                not isinstance(seal, str)
                or _SHA256.fullmatch(seal) is None
                or _sha(document) != seal
            ):
                return False
        return all(_recursive_report_seals_valid(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return all(_recursive_report_seals_valid(child) for child in value)
    return True


def _valid_short_conv_v4_candidate_provenance(
    *,
    candidate_id: str,
    provenance: Any,
    selection: Any,
    ridge_fit: Mapping[str, Any],
    solver_policy: Lfm2CpcfShortConvV4SolverPolicy,
) -> bool:
    try:
        if not isinstance(provenance, Mapping) or not isinstance(
            selection,
            Mapping,
        ):
            return False
        expected_fields = {
            "schema",
            "kind",
            "candidate_id",
            "policy_sha256",
            "target_dtype",
            "target_dtype_candidate_state_sha256",
            "source_alias_count",
            "source_alias_candidate_ids",
            "sources",
            "fit_only",
            "selection_cases_consumed",
            "validation_cases_consumed",
            "actual_target_dtype_materialization_before_deduplication",
            "execution",
            "report_sha256",
        }
        if set(provenance) != expected_fields:
            return False
        sources = provenance.get("sources")
        aliases = provenance.get("source_alias_candidate_ids")
        alias_count = provenance.get("source_alias_count")
        target_state_sha = provenance.get(
            "target_dtype_candidate_state_sha256"
        )
        if not (
            provenance.get("schema") == LFM2_CPCF_SHORT_CONV_V4_FOLD_SCHEMA
            and provenance.get("kind")
            == "authenticated_fit_only_candidate_provenance"
            and provenance.get("candidate_id") == candidate_id
            and provenance.get("policy_sha256")
            == solver_policy.authentication_sha256
            and provenance.get("target_dtype") == solver_policy.target_dtype
            and isinstance(target_state_sha, str)
            and _SHA256.fullmatch(target_state_sha) is not None
            and selection.get("candidate_state_sha256") == target_state_sha
            and type(alias_count) is int
            and alias_count > 0
            and isinstance(aliases, list)
            and len(aliases) == alias_count
            and len(set(aliases)) == alias_count
            and aliases[0] == candidate_id
            and isinstance(sources, list)
            and len(sources) == alias_count
            and aliases
            == [
                source.get("candidate_id")
                for source in sources
                if isinstance(source, Mapping)
            ]
            and provenance.get("fit_only") is True
            and provenance.get("selection_cases_consumed") == 0
            and provenance.get("validation_cases_consumed") == 0
            and provenance.get(
                "actual_target_dtype_materialization_before_deduplication"
            )
            is True
        ):
            return False
        execution = provenance.get("execution")
        if not (
            isinstance(execution, Mapping)
            and set(execution)
            == {
                "parent_model_forward_calls",
                "target_model_forward_calls",
                "generation_calls",
                "training_steps",
                "backward_calls",
                "optimizer_steps",
                "parameters_pruned",
                "parameters_quantized",
            }
            and all(type(value) is int and value == 0 for value in execution.values())
        ):
            return False
        for source in sources:
            if not isinstance(source, Mapping) or set(source) != {
                "candidate_id",
                "ridge_id",
                "candidate_kind",
                "stage_id",
                "fit_path_state_sha256",
                "candidate_report",
            }:
                return False
            ridge_id = source.get("ridge_id")
            candidate_report = source.get("candidate_report")
            if not (
                isinstance(ridge_id, str)
                and ridge_id in ridge_fit
                and isinstance(ridge_fit[ridge_id], Mapping)
                and isinstance(candidate_report, Mapping)
                and source.get("stage_id") == candidate_report.get("stage_id")
                and source.get("fit_path_state_sha256")
                == candidate_report.get("state_sha256")
                and candidate_report.get("policy_sha256")
                == solver_policy.authentication_sha256
                and candidate_report.get("alpha")
                == ridge_fit[ridge_id].get("alpha")
                and candidate_report.get("fit_case_authentication_sha256")
                == ridge_fit[ridge_id].get("fit_case_authentication_sha256")
                and candidate_report.get("fit_only") is True
                and candidate_report.get("selection_cases_consumed") == 0
                and candidate_report.get("validation_cases_consumed") == 0
            ):
                return False
            inventory = ridge_fit[ridge_id].get("path_state_inventory")
            if not isinstance(inventory, list) or not inventory:
                return False
            candidate_kind = source.get("candidate_kind")
            if candidate_kind == "accepted_fit_path_anchor":
                inventory_record = {
                    (
                        value.get("stage_id"),
                        value.get("state_sha256"),
                        value.get("candidate_report_sha256"),
                    )
                    for value in inventory
                    if isinstance(value, Mapping)
                }
                if not (
                    candidate_report.get("kind")
                    == "authenticated_fit_only_projection_path_state"
                    and (
                        candidate_report.get("stage_id"),
                        candidate_report.get("state_sha256"),
                        candidate_report.get("report_sha256"),
                    )
                    in inventory_record
                ):
                    return False
            elif candidate_kind == "prior_to_final_trust_path":
                if not (
                    candidate_report.get("kind")
                    == "authenticated_fit_only_projection_trust_path_state"
                    and candidate_report.get("left_stage_id")
                    == inventory[0].get("stage_id")
                    and candidate_report.get("left_state_sha256")
                    == inventory[0].get("state_sha256")
                    and candidate_report.get("left_candidate_report_sha256")
                    == inventory[0].get("candidate_report_sha256")
                    and candidate_report.get("right_stage_id")
                    == inventory[-1].get("stage_id")
                    and candidate_report.get("right_state_sha256")
                    == inventory[-1].get("state_sha256")
                    and candidate_report.get("right_candidate_report_sha256")
                    == inventory[-1].get("candidate_report_sha256")
                    and candidate_report.get("fraction")
                    in fixed_projection_trust_path_fractions()
                ):
                    return False
            elif candidate_kind == "prior_to_first_writer_trust_path":
                first_writer_record = next(
                    (
                        value
                        for value in inventory
                        if value.get("stage_id", "").endswith("-writer")
                    ),
                    None,
                )
                if not (
                    isinstance(first_writer_record, Mapping)
                    and candidate_report.get("kind")
                    == "authenticated_fit_only_projection_trust_path_state"
                    and candidate_report.get("left_stage_id")
                    == inventory[0].get("stage_id")
                    and candidate_report.get("left_state_sha256")
                    == inventory[0].get("state_sha256")
                    and candidate_report.get("left_candidate_report_sha256")
                    == inventory[0].get("candidate_report_sha256")
                    and candidate_report.get("right_stage_id")
                    == first_writer_record.get("stage_id")
                    and candidate_report.get("right_state_sha256")
                    == first_writer_record.get("state_sha256")
                    and candidate_report.get("right_candidate_report_sha256")
                    == first_writer_record.get("candidate_report_sha256")
                    and candidate_report.get("fraction")
                    in fixed_projection_trust_path_fractions()
                ):
                    return False
            else:
                return False
        return True
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
        return False


def _valid_short_conv_v4_raw_frontier_coverage(
    *,
    candidate_fit: Mapping[str, Any],
    ridge_fit: Mapping[str, Any],
    solver_policy: Lfm2CpcfShortConvV4SolverPolicy,
) -> bool:
    """Prove every configured pre-dedup path source appears exactly once."""

    try:
        expected_sources: dict[str, tuple[str, str, str]] = {}
        fractions = fixed_projection_trust_path_fractions()
        for ridge_index, alpha in enumerate(solver_policy.ridge_alphas):
            ridge_id = f"ridge-alpha-{ridge_index:02d}-{alpha:.17g}"
            ridge = ridge_fit.get(ridge_id)
            inventory = (
                ridge.get("path_state_inventory")
                if isinstance(ridge, Mapping)
                else None
            )
            if not (
                isinstance(inventory, list)
                and inventory
                and all(isinstance(record, Mapping) for record in inventory)
            ):
                return False
            for path_index, record in enumerate(inventory):
                stage_id = record.get("stage_id")
                if not isinstance(stage_id, str) or not stage_id:
                    return False
                candidate_id = (
                    f"{ridge_id}--anchor-{path_index:02d}-{stage_id}"
                )
                expected_sources[candidate_id] = (
                    ridge_id,
                    "accepted_fit_path_anchor",
                    stage_id,
                )

            for fraction_index, fraction in enumerate(fractions):
                stage_id = (
                    f"prior-to-final-{fraction_index:02d}-{fraction:.17g}"
                )
                candidate_id = (
                    f"{ridge_id}--trust-{fraction_index:02d}-"
                    f"{fraction:.17g}"
                )
                expected_sources[candidate_id] = (
                    ridge_id,
                    "prior_to_final_trust_path",
                    stage_id,
                )

            first_writer_record = next(
                (
                    record
                    for record in inventory
                    if record.get("stage_id", "").endswith("-writer")
                ),
                None,
            )
            if first_writer_record is not None:
                for fraction_index, fraction in enumerate(fractions):
                    stage_id = (
                        f"prior-to-first-writer-{fraction_index:02d}-"
                        f"{fraction:.17g}"
                    )
                    candidate_id = (
                        f"{ridge_id}--writer-trust-{fraction_index:02d}-"
                        f"{fraction:.17g}"
                    )
                    expected_sources[candidate_id] = (
                        ridge_id,
                        "prior_to_first_writer_trust_path",
                        stage_id,
                    )

        actual_sources: dict[str, tuple[Any, Any, Any]] = {}
        for candidate_id, provenance in candidate_fit.items():
            if not isinstance(provenance, Mapping):
                return False
            sources = provenance.get("sources")
            aliases = provenance.get("source_alias_candidate_ids")
            if not (
                isinstance(sources, list)
                and sources
                and isinstance(aliases, list)
                and aliases
                == [
                    source.get("candidate_id")
                    if isinstance(source, Mapping)
                    else None
                    for source in sources
                ]
                and aliases[0] == candidate_id
            ):
                return False
            for source in sources:
                if not isinstance(source, Mapping):
                    return False
                source_id = source.get("candidate_id")
                if not isinstance(source_id, str) or source_id in actual_sources:
                    return False
                actual_sources[source_id] = (
                    source.get("ridge_id"),
                    source.get("candidate_kind"),
                    source.get("stage_id"),
                )
        return actual_sources == expected_sources
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
        return False


def _valid_short_conv_v4_candidate_pool(
    *,
    pool: Any,
    candidate_fit: Mapping[str, Any],
    candidate_selection: Mapping[str, Any],
    ridge_fit: Mapping[str, Any],
    solver_policy: Lfm2CpcfShortConvV4SolverPolicy,
    gate_policy: Mapping[str, Any],
    fold_case_seals: Mapping[str, Any],
    split_evaluations: Mapping[str, Any],
) -> bool:
    try:
        if not isinstance(pool, Mapping):
            return False
        expected_fields = {
            "schema",
            "kind",
            "policy_sha256",
            "target_dtype",
            "ridge_candidate_count",
            "raw_candidate_count",
            "unique_target_dtype_candidate_count",
            "target_dtype_duplicate_count",
            "accepted_phase_anchors_included",
            "rejected_phase_states_included",
            "prior_to_final_trust_path_fractions",
            "prior_to_first_writer_trust_path_fractions",
            "selection_feasibility_policy",
            "selection_feasibility_policy_sha256",
            "fold_case_authentication_sha256",
            "projection_case_authentication_sha256",
            "candidate_ids",
            "candidate_state_sha256",
            "fit_only_candidate_generation",
            "selection_cases_consumed_during_generation",
            "validation_cases_consumed_during_generation",
            "report_sha256",
        }
        if set(pool) != expected_fields:
            return False
        counts = {
            name: pool.get(name)
            for name in (
                "ridge_candidate_count",
                "raw_candidate_count",
                "unique_target_dtype_candidate_count",
                "target_dtype_duplicate_count",
            )
        }
        if not all(type(value) is int and value >= 0 for value in counts.values()):
            return False
        candidate_ids = pool.get("candidate_ids")
        state_map = pool.get("candidate_state_sha256")
        projection_case_seals = pool.get(
            "projection_case_authentication_sha256"
        )
        expected_feasibility = {
            "maximum_endpoint_relative_rmse": min(
                float(gate_policy["maximum_fit_endpoint_relative_rmse"]),
                float(gate_policy["maximum_validation_endpoint_relative_rmse"]),
            ),
            "maximum_context_relative_rmse": min(
                float(gate_policy["maximum_fit_endpoint_relative_rmse"]),
                float(gate_policy["maximum_validation_endpoint_relative_rmse"]),
            ),
            "minimum_endpoint_cosine_similarity": max(
                float(gate_policy["minimum_fit_endpoint_cosine_similarity"]),
                float(gate_policy["minimum_validation_endpoint_cosine_similarity"]),
            ),
            "minimum_context_cosine_similarity": max(
                float(gate_policy["minimum_fit_endpoint_cosine_similarity"]),
                float(gate_policy["minimum_validation_endpoint_cosine_similarity"]),
            ),
        }
        expected_feasibility_sha = _sha(
            {
                "schema": LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
                "kind": "selection_feasibility_policy",
                **expected_feasibility,
            }
        )
        raw_alias_count = sum(
            int(value.get("source_alias_count", -1))
            for value in candidate_fit.values()
            if isinstance(value, Mapping)
        )
        split_names = {"fit", "selection", "validation"}
        if not (
            isinstance(fold_case_seals, Mapping)
            and set(fold_case_seals) == split_names
            and pool.get("fold_case_authentication_sha256")
            == dict(fold_case_seals)
            and isinstance(projection_case_seals, Mapping)
            and set(projection_case_seals) == split_names
            and all(
                isinstance(projection_case_seals[split], list)
                and projection_case_seals[split]
                and all(
                    isinstance(value, str)
                    and _SHA256.fullmatch(value) is not None
                    for value in projection_case_seals[split]
                )
                for split in split_names
            )
            and isinstance(split_evaluations, Mapping)
            and set(split_evaluations) == split_names
        ):
            return False

        def evaluation_projection_seals(value: Any) -> list[Any] | None:
            if not isinstance(value, Mapping) or not isinstance(
                value.get("cases"),
                list,
            ):
                return None
            return [
                case.get("projection_case_authentication_sha256")
                if isinstance(case, Mapping)
                else None
                for case in value["cases"]
            ]

        return bool(
            pool.get("schema") == LFM2_CPCF_SHORT_CONV_V4_FOLD_SCHEMA
            and pool.get("kind") == "authenticated_target_dtype_candidate_pool"
            and pool.get("policy_sha256")
            == solver_policy.authentication_sha256
            and pool.get("target_dtype") == solver_policy.target_dtype
            and counts["ridge_candidate_count"] == len(ridge_fit)
            and counts["unique_target_dtype_candidate_count"] == len(candidate_fit)
            and counts["raw_candidate_count"] == raw_alias_count
            and counts["raw_candidate_count"]
            == counts["unique_target_dtype_candidate_count"]
            + counts["target_dtype_duplicate_count"]
            and _valid_short_conv_v4_raw_frontier_coverage(
                candidate_fit=candidate_fit,
                ridge_fit=ridge_fit,
                solver_policy=solver_policy,
            )
            and isinstance(candidate_ids, list)
            and candidate_ids == sorted(candidate_fit)
            and len(candidate_ids) == len(set(candidate_ids))
            and isinstance(state_map, Mapping)
            and set(state_map) == set(candidate_fit)
            and all(
                isinstance(state_map[candidate_id], str)
                and _SHA256.fullmatch(state_map[candidate_id]) is not None
                and isinstance(candidate_selection[candidate_id], Mapping)
                and state_map[candidate_id]
                == candidate_selection[candidate_id].get("candidate_state_sha256")
                for candidate_id in candidate_fit
            )
            and pool.get("accepted_phase_anchors_included") is True
            and pool.get("rejected_phase_states_included") is False
            and pool.get("prior_to_final_trust_path_fractions")
            == list(fixed_projection_trust_path_fractions())
            and pool.get("prior_to_first_writer_trust_path_fractions")
            == list(fixed_projection_trust_path_fractions())
            and pool.get("selection_feasibility_policy") == expected_feasibility
            and pool.get("selection_feasibility_policy_sha256")
            == expected_feasibility_sha
            and all(
                ridge.get("fit_case_authentication_sha256")
                == projection_case_seals["fit"]
                for ridge in ridge_fit.values()
                if isinstance(ridge, Mapping)
            )
            and all(
                evaluation_projection_seals(selection)
                == projection_case_seals["selection"]
                for selection in candidate_selection.values()
            )
            and all(
                evaluation_projection_seals(split_evaluations[split])
                == projection_case_seals[split]
                for split in split_names
            )
            and pool.get("fit_only_candidate_generation") is True
            and pool.get("selection_cases_consumed_during_generation") == 0
            and pool.get("validation_cases_consumed_during_generation") == 0
        )
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
        return False


def _valid_short_conv_v4_ridge_fit_reports(
    ridge_fit: Any,
    *,
    solver_policy: Lfm2CpcfShortConvV4SolverPolicy,
) -> bool:
    try:
        if not isinstance(ridge_fit, Mapping):
            return False
        expected_ids = [
            f"ridge-alpha-{index:02d}-{alpha:.17g}"
            for index, alpha in enumerate(solver_policy.ridge_alphas)
        ]
        if set(ridge_fit) != set(expected_ids):
            return False
        for ridge_id, alpha in zip(expected_ids, solver_policy.ridge_alphas):
            value = ridge_fit[ridge_id]
            if not isinstance(value, Mapping):
                return False
            inventory = value.get("path_state_inventory")
            fit_case_seals = value.get("fit_case_authentication_sha256")
            if not (
                value.get("schema") == LFM2_CPCF_SHORT_CONV_V4_SCHEMA
                and value.get("kind")
                == "fixed_bounded_sampled_cp_variable_projection"
                and value.get("policy_sha256")
                == solver_policy.authentication_sha256
                and value.get("alpha") == alpha
                and value.get("fit_only") is True
                and value.get("selection_cases_consumed") == 0
                and value.get("validation_cases_consumed") == 0
                and isinstance(inventory, list)
                and inventory
                and all(
                    isinstance(item, Mapping)
                    and set(item)
                    == {
                        "stage_id",
                        "state_sha256",
                        "candidate_report_sha256",
                    }
                    and isinstance(item.get("stage_id"), str)
                    and isinstance(item.get("state_sha256"), str)
                    and _SHA256.fullmatch(item["state_sha256"]) is not None
                    and isinstance(item.get("candidate_report_sha256"), str)
                    and _SHA256.fullmatch(item["candidate_report_sha256"])
                    is not None
                    for item in inventory
                )
                and value.get("path_state_count") == len(inventory)
                and value.get("path_state_inventory_sha256") == _sha(inventory)
                and value.get("rejected_phases_excluded_from_path") is True
                and value.get("final_state_is_last_path_state") is True
                and inventory[-1].get("state_sha256")
                == value.get("state_sha256")
                and isinstance(fit_case_seals, list)
                and fit_case_seals
                and all(
                    isinstance(item, str)
                    and _SHA256.fullmatch(item) is not None
                    for item in fit_case_seals
                )
            ):
                return False
        return True
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
        return False


def _valid_short_conv_v4_selection_ranking(
    ranking: Any,
    *,
    candidate_selection: Mapping[str, Any],
    solver_policy: Lfm2CpcfShortConvV4SolverPolicy,
    feasibility_policy: Mapping[str, Any],
) -> bool:
    try:
        if not isinstance(ranking, Mapping):
            return False
        recomputed = rank_short_conv_v4_selection_candidates(
            candidate_selection,
            policy_sha256=solver_policy.authentication_sha256,
            feasibility_policy=feasibility_policy,
        )
        return dict(ranking) == dict(recomputed)
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
        return False


def _valid_short_conv_v4_behavior_gate(
    gate: Any,
    *,
    gate_policy: Mapping[str, Any],
    fit_evaluation: Mapping[str, Any],
    validation_evaluation: Mapping[str, Any],
    full_source_coverage: bool,
) -> bool:
    try:
        expected_checks = {
            "fit_endpoint_relative_rmse": (
                float(fit_evaluation["maximum_case_relative_rmse"])
                <= float(gate_policy["maximum_fit_endpoint_relative_rmse"])
            ),
            "validation_endpoint_relative_rmse": (
                float(validation_evaluation["maximum_case_relative_rmse"])
                <= float(gate_policy["maximum_validation_endpoint_relative_rmse"])
            ),
            "fit_endpoint_cosine_similarity": (
                float(fit_evaluation["minimum_case_cosine_similarity"])
                >= float(gate_policy["minimum_fit_endpoint_cosine_similarity"])
            ),
            "validation_endpoint_cosine_similarity": (
                float(validation_evaluation["minimum_case_cosine_similarity"])
                >= float(
                    gate_policy["minimum_validation_endpoint_cosine_similarity"]
                )
            ),
            "fit_context_relative_rmse": (
                fit_evaluation.get("context_evidence_present") is True
                and float(fit_evaluation["maximum_case_context_relative_rmse"])
                <= float(gate_policy["maximum_fit_endpoint_relative_rmse"])
            ),
            "validation_context_relative_rmse": (
                validation_evaluation.get("context_evidence_present") is True
                and float(
                    validation_evaluation["maximum_case_context_relative_rmse"]
                )
                <= float(gate_policy["maximum_validation_endpoint_relative_rmse"])
            ),
            "fit_context_cosine_similarity": (
                fit_evaluation.get("context_evidence_present") is True
                and float(fit_evaluation["minimum_case_context_cosine_similarity"])
                >= float(gate_policy["minimum_fit_endpoint_cosine_similarity"])
            ),
            "validation_context_cosine_similarity": (
                validation_evaluation.get("context_evidence_present") is True
                and float(
                    validation_evaluation["minimum_case_context_cosine_similarity"]
                )
                >= float(
                    gate_policy["minimum_validation_endpoint_cosine_similarity"]
                )
            ),
            "full_source_numeric_participation": full_source_coverage,
            "actual_fit_target_dtype_replay": (
                fit_evaluation.get(
                    "actual_parameter_and_intermediate_target_dtype_replay"
                )
                is True
            ),
            "actual_validation_target_dtype_replay": (
                validation_evaluation.get(
                    "actual_parameter_and_intermediate_target_dtype_replay"
                )
                is True
            ),
        }
        expected_failed = [
            name for name, passed in expected_checks.items() if not passed
        ]
        return bool(
            isinstance(gate, Mapping)
            and set(gate)
            == {"policy", "checks", "failed_checks", "passed", "enforced"}
            and gate.get("policy") == dict(gate_policy)
            and gate.get("checks") == expected_checks
            and gate.get("failed_checks") == expected_failed
            and gate.get("passed") is all(expected_checks.values())
            and gate.get("enforced") is gate_policy.get("enforce")
        )
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
        return False


def _valid_short_conv_v4_legacy_validation_report(
    report: Mapping[str, Any],
) -> bool:
    """Validate immutable v4 failure-v1 receipts emitted before path selection."""

    try:
        solver_policy = parse_lfm2_cpcf_short_conv_v4_solver_policy(
            report.get("solver_policy")
        )
        gate_policy = report.get("gate_policy")
        gate = report.get("behavior_gate")
        checks = gate.get("checks") if isinstance(gate, Mapping) else None
        failed_checks = (
            gate.get("failed_checks") if isinstance(gate, Mapping) else None
        )
        case_seals = report.get("case_authentication_sha256")
        execution = report.get("execution")
        candidate_fit = report.get("candidate_fit_reports")
        candidate_selection = report.get("candidate_selection_reports")
        ranking = report.get("selection_ranking")
        selected = report.get("selected_candidate_id")
        fit_evaluation = report.get("fit_evaluation")
        selection_evaluation = report.get("selection_evaluation")
        validation_evaluation = report.get("validation_evaluation")
        selected_state_sha = (
            selection_evaluation.get("candidate_state_sha256")
            if isinstance(selection_evaluation, Mapping)
            else None
        )
        dimensions = (
            report.get("source_hidden_size"),
            report.get("target_hidden_size"),
            report.get("source_fir_length"),
            report.get("target_fir_length"),
        )
        evaluations = (
            ("fit", fit_evaluation),
            ("selection", selection_evaluation),
            ("validation", validation_evaluation),
        )
        return bool(
            set(report) == _SHORT_CONV_V4_FAILURE_V1_REPORT_FIELDS
            and report.get("schema") == LFM2_CPCF_SHORT_CONV_V4_FAILURE_V1_SCHEMA
            and report.get("solver_schema") == LFM2_CPCF_SHORT_CONV_V4_SCHEMA
            and report.get("solver_policy_sha256")
            == solver_policy.authentication_sha256
            and solver_policy.verify_authentication()
            and isinstance(gate_policy, Mapping)
            and set(gate_policy) == _SHORT_CONV_GATE_POLICY_FIELDS
            and gate_policy.get("enforce") is True
            and all(
                not isinstance(gate_policy.get(name), bool)
                and isinstance(gate_policy.get(name), (int, float))
                and math.isfinite(float(gate_policy[name]))
                for name in _SHORT_CONV_GATE_POLICY_FIELDS - {"enforce"}
            )
            and isinstance(gate, Mapping)
            and set(gate)
            == {"policy", "checks", "failed_checks", "passed", "enforced"}
            and gate.get("policy") == dict(gate_policy)
            and gate.get("passed") is False
            and gate.get("enforced") is True
            and isinstance(checks, Mapping)
            and set(checks) == _SHORT_CONV_V4_GATE_CHECK_FIELDS
            and all(isinstance(value, bool) for value in checks.values())
            and isinstance(failed_checks, list)
            and failed_checks
            and len(failed_checks) == len(set(failed_checks))
            and set(failed_checks)
            == {name for name, passed in checks.items() if not passed}
            and isinstance(case_seals, Mapping)
            and set(case_seals) == {"fit", "selection", "validation"}
            and all(
                isinstance(values, list)
                and values
                and all(
                    isinstance(value, str)
                    and _SHA256.fullmatch(value) is not None
                    for value in values
                )
                for values in case_seals.values()
            )
            and all(type(value) is int and value > 0 for value in dimensions)
            and report["source_hidden_size"] >= report["target_hidden_size"]
            and isinstance(
                report.get("hidden_width_compression_ratio"),
                (int, float),
            )
            and not isinstance(
                report.get("hidden_width_compression_ratio"),
                bool,
            )
            and math.isfinite(float(report["hidden_width_compression_ratio"]))
            and float(report["hidden_width_compression_ratio"]) >= 1.0
            and isinstance(report.get("basis"), Mapping)
            and isinstance(report.get("input_lift"), Mapping)
            and isinstance(report.get("transported_priors"), Mapping)
            and isinstance(candidate_fit, Mapping)
            and isinstance(candidate_selection, Mapping)
            and candidate_fit
            and set(candidate_fit) == set(candidate_selection)
            and isinstance(selected, str)
            and selected in candidate_fit
            and isinstance(selected_state_sha, str)
            and _SHA256.fullmatch(selected_state_sha) is not None
            and isinstance(candidate_selection[selected], Mapping)
            and candidate_selection[selected].get("split") == "selection"
            and candidate_selection[selected].get("candidate_state_sha256")
            == selected_state_sha
            and isinstance(ranking, Mapping)
            and ranking.get("kind") == "selection_only_candidate_ranking"
            and ranking.get("selected_candidate_id") == selected
            and ranking.get("validation_isolation") is True
            and isinstance(ranking.get("ranking"), list)
            and ranking["ranking"]
            and ranking["ranking"][0].get("candidate_id") == selected
            and ranking["ranking"][0].get("candidate_state_sha256")
            == selected_state_sha
            and all(
                isinstance(evaluation, Mapping)
                and evaluation.get("schema") == LFM2_CPCF_SHORT_CONV_V4_SCHEMA
                and evaluation.get("kind") == "target_dtype_evaluation"
                and evaluation.get("split") == split
                and evaluation.get("policy_sha256")
                == solver_policy.authentication_sha256
                and evaluation.get("candidate_state_sha256")
                == selected_state_sha
                and evaluation.get(
                    "actual_parameter_and_intermediate_target_dtype_replay"
                )
                is True
                for split, evaluation in evaluations
            )
            and isinstance(report.get("compiled_state_sha256"), str)
            and _SHA256.fullmatch(report["compiled_state_sha256"]) is not None
            and all(
                type(report.get(name)) is int and report[name] >= 0
                for name in (
                    "source_channels_selected",
                    "source_channels_dropped",
                    "source_channels_pruned",
                    "source_channels_quantized",
                )
            )
            and report.get("source_channels_pruned") == 0
            and report.get("source_channels_quantized") == 0
            and report.get("fit_only_parameter_solves") is True
            and report.get("selection_used_only_for_candidate_ranking") is True
            and report.get("validation_used_only_after_candidate_freeze") is True
            and isinstance(execution, Mapping)
            and set(execution) == _SHORT_CONV_V4_EXECUTION_FIELDS
            and all(type(value) is int and value == 0 for value in execution.values())
            and report.get("compiled_short_conv_fold_issued") is False
            and report.get("promotion_claimed") is False
            and report.get("promotable_candidate") is False
            and report.get("failure_evidence_only") is True
            and _recursive_report_seals_valid(report)
            and _no_recursive_success_claims(report)
        )
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
        return False


def _valid_short_conv_v4_validation_report_unsafe(
    report: Mapping[str, Any],
) -> bool:
    if report.get("schema") == LFM2_CPCF_SHORT_CONV_V4_FAILURE_V1_SCHEMA:
        return _valid_short_conv_v4_legacy_validation_report(report)
    try:
        solver_policy = parse_lfm2_cpcf_short_conv_v4_solver_policy(
            report.get("solver_policy")
        )
    except Exception:
        return False
    gate_policy = report.get("gate_policy")
    gate = report.get("behavior_gate")
    checks = gate.get("checks") if isinstance(gate, Mapping) else None
    failed_checks = gate.get("failed_checks") if isinstance(gate, Mapping) else None
    case_seals = report.get("case_authentication_sha256")
    execution = report.get("execution")
    fit_evaluation = report.get("fit_evaluation")
    selection_evaluation = report.get("selection_evaluation")
    validation_evaluation = report.get("validation_evaluation")
    ranking = report.get("selection_ranking")
    ridge_fit = report.get("ridge_fit_reports")
    candidate_fit = report.get("candidate_fit_reports")
    candidate_selection = report.get("candidate_selection_reports")
    candidate_pool = report.get("candidate_pool_report")
    selected_candidate = report.get("selected_candidate_id")
    compiled_state_sha = report.get("compiled_state_sha256")
    basis = report.get("basis")
    transported_priors = report.get("transported_priors")
    full_source_coverage = bool(
        isinstance(basis, Mapping)
        and basis.get("full_source_numeric_participation") is True
        and isinstance(transported_priors, Mapping)
        and transported_priors.get("source_channels_dropped") == 0
    )
    selected_candidate_state_sha = (
        selection_evaluation.get("candidate_state_sha256")
        if isinstance(selection_evaluation, Mapping)
        else None
    )
    dimensions = (
        report.get("source_hidden_size"),
        report.get("target_hidden_size"),
        report.get("source_fir_length"),
        report.get("target_fir_length"),
    )
    evaluations = (
        ("fit", fit_evaluation),
        ("selection", selection_evaluation),
        ("validation", validation_evaluation),
    )
    return bool(
        set(report) == _SHORT_CONV_V4_FAILURE_REPORT_FIELDS
        and report.get("schema") == LFM2_CPCF_SHORT_CONV_V4_FAILURE_SCHEMA
        and report.get("solver_schema") == LFM2_CPCF_SHORT_CONV_V4_SCHEMA
        and report.get("solver_policy_sha256") == solver_policy.authentication_sha256
        and solver_policy.verify_authentication()
        and isinstance(gate_policy, Mapping)
        and set(gate_policy) == _SHORT_CONV_GATE_POLICY_FIELDS
        and gate_policy.get("enforce") is True
        and all(
            not isinstance(gate_policy.get(name), bool)
            and isinstance(gate_policy.get(name), (int, float))
            and math.isfinite(float(gate_policy[name]))
            for name in _SHORT_CONV_GATE_POLICY_FIELDS - {"enforce"}
        )
        and isinstance(gate, Mapping)
        and set(gate) == {"policy", "checks", "failed_checks", "passed", "enforced"}
        and gate.get("policy") == dict(gate_policy)
        and gate.get("passed") is False
        and gate.get("enforced") is True
        and isinstance(checks, Mapping)
        and set(checks) == _SHORT_CONV_V4_GATE_CHECK_FIELDS
        and all(isinstance(value, bool) for value in checks.values())
        and isinstance(failed_checks, list)
        and failed_checks
        and len(failed_checks) == len(set(failed_checks))
        and set(failed_checks)
        == {name for name, passed in checks.items() if not passed}
        and _valid_short_conv_v4_behavior_gate(
            gate,
            gate_policy=gate_policy,
            fit_evaluation=fit_evaluation,
            validation_evaluation=validation_evaluation,
            full_source_coverage=full_source_coverage,
        )
        and isinstance(case_seals, Mapping)
        and set(case_seals) == {"fit", "selection", "validation"}
        and all(
            isinstance(values, list)
            and values
            and all(
                isinstance(value, str) and _SHA256.fullmatch(value) is not None
                for value in values
            )
            for values in case_seals.values()
        )
        and all(type(value) is int and value > 0 for value in dimensions)
        and report["source_hidden_size"] >= report["target_hidden_size"]
        and isinstance(report.get("hidden_width_compression_ratio"), (int, float))
        and not isinstance(report.get("hidden_width_compression_ratio"), bool)
        and math.isfinite(float(report["hidden_width_compression_ratio"]))
        and float(report["hidden_width_compression_ratio"]) >= 1.0
        and isinstance(basis, Mapping)
        and isinstance(report.get("input_lift"), Mapping)
        and isinstance(transported_priors, Mapping)
        and _valid_short_conv_v4_ridge_fit_reports(
            ridge_fit,
            solver_policy=solver_policy,
        )
        and isinstance(candidate_fit, Mapping)
        and isinstance(candidate_selection, Mapping)
        and candidate_fit
        and set(candidate_fit) == set(candidate_selection)
        and all(
            _valid_short_conv_v4_candidate_provenance(
                candidate_id=candidate_id,
                provenance=candidate_fit[candidate_id],
                selection=candidate_selection[candidate_id],
                ridge_fit=ridge_fit,
                solver_policy=solver_policy,
            )
            for candidate_id in candidate_fit
        )
        and _valid_short_conv_v4_candidate_pool(
            pool=candidate_pool,
            candidate_fit=candidate_fit,
            candidate_selection=candidate_selection,
            ridge_fit=ridge_fit,
            solver_policy=solver_policy,
            gate_policy=gate_policy,
            fold_case_seals=case_seals,
            split_evaluations={
                "fit": fit_evaluation,
                "selection": selection_evaluation,
                "validation": validation_evaluation,
            },
        )
        and isinstance(selected_candidate, str)
        and bool(selected_candidate)
        and selected_candidate in candidate_fit
        and isinstance(selected_candidate_state_sha, str)
        and _SHA256.fullmatch(selected_candidate_state_sha) is not None
        and isinstance(candidate_selection[selected_candidate], Mapping)
        and candidate_selection[selected_candidate].get("split") == "selection"
        and candidate_selection[selected_candidate].get("candidate_state_sha256")
        == selected_candidate_state_sha
        and _valid_short_conv_v4_selection_ranking(
            ranking,
            candidate_selection=candidate_selection,
            solver_policy=solver_policy,
            feasibility_policy=candidate_pool["selection_feasibility_policy"],
        )
        and ranking.get("selected_candidate_id") == selected_candidate
        and ranking["ranking"][0].get("candidate_state_sha256")
        == selected_candidate_state_sha
        and all(
            isinstance(evaluation, Mapping)
            and evaluation.get("schema") == LFM2_CPCF_SHORT_CONV_V4_SCHEMA
            and evaluation.get("kind") == "target_dtype_evaluation"
            and evaluation.get("split") == split
            and evaluation.get("policy_sha256") == solver_policy.authentication_sha256
            and evaluation.get("candidate_state_sha256") == selected_candidate_state_sha
            and evaluation.get("actual_parameter_and_intermediate_target_dtype_replay")
            is True
            for split, evaluation in evaluations
        )
        and isinstance(compiled_state_sha, str)
        and _SHA256.fullmatch(compiled_state_sha) is not None
        and all(
            type(report.get(name)) is int and report[name] >= 0
            for name in (
                "source_channels_selected",
                "source_channels_dropped",
                "source_channels_pruned",
                "source_channels_quantized",
            )
        )
        and report.get("source_channels_pruned") == 0
        and report.get("source_channels_quantized") == 0
        and report.get("fit_only_parameter_solves") is True
        and report.get("selection_used_only_for_candidate_ranking") is True
        and report.get("validation_used_only_after_candidate_freeze") is True
        and isinstance(execution, Mapping)
        and set(execution) == _SHORT_CONV_V4_EXECUTION_FIELDS
        and all(type(value) is int and value == 0 for value in execution.values())
        and report.get("compiled_short_conv_fold_issued") is False
        and report.get("promotion_claimed") is False
        and report.get("promotable_candidate") is False
        and report.get("failure_evidence_only") is True
        and _recursive_report_seals_valid(report)
        and _no_recursive_success_claims(report)
    )


def _valid_short_conv_v4_validation_report(
    report: Mapping[str, Any],
) -> bool:
    try:
        return _valid_short_conv_v4_validation_report_unsafe(report)
    except (
        AttributeError,
        KeyError,
        OverflowError,
        TypeError,
        ValueError,
    ):
        return False


def _valid_short_conv_v5_selector_validation_report(
    report: Mapping[str, Any],
) -> bool:
    """Validate a selector-only failure without pretending it is v4."""

    try:
        receipt = report.get("preinstall_candidate_selector_receipt")
        selector_report = (
            receipt.get("selector_report")
            if isinstance(receipt, Mapping)
            else None
        )
        selection = report.get("selection_evaluation")
        candidate_reports = report.get("candidate_selection_reports")
        gate = report.get("behavior_gate")
        execution = report.get("execution")
        selected_id = report.get("selected_candidate_id")
        selected_sha = (
            selection.get("candidate_state_sha256")
            if isinstance(selection, Mapping)
            else None
        )
        receipt_body = dict(receipt) if isinstance(receipt, Mapping) else {}
        receipt_seal = receipt_body.pop("report_sha256", None)
        selector_body = (
            dict(selector_report)
            if isinstance(selector_report, Mapping)
            else {}
        )
        selector_seal = selector_body.pop("report_sha256", None)
        return bool(
            report.get("schema")
            == LFM2_CPCF_SHORT_CONV_V5_SELECTOR_FAILURE_SCHEMA
            and isinstance(receipt, Mapping)
            and receipt.get("schema")
            == LFM2_CPCF_SHORT_CONV_V5_SELECTOR_RECEIPT_SCHEMA
            and isinstance(receipt_seal, str)
            and _SHA256.fullmatch(receipt_seal) is not None
            and _sha(receipt_body) == receipt_seal
            and report.get("preinstall_candidate_selector_receipt_sha256")
            == receipt_seal
            and isinstance(selector_report, Mapping)
            and isinstance(selector_seal, str)
            and _SHA256.fullmatch(selector_seal) is not None
            and _sha(selector_body) == selector_seal
            and receipt.get("selector_report_sha256") == selector_seal
            and receipt.get(
                "all_live_candidate_states_verified_before_callback"
            )
            is True
            and receipt.get(
                "all_live_candidate_states_verified_after_callback"
            )
            is True
            and receipt.get("live_candidate_state_bytes_unchanged") is True
            and receipt.get("candidate_state_tensors_exposed_to_selector")
            is False
            and receipt.get("fit_reports_exposed_to_selector") is False
            and receipt.get("validation_reports_exposed_to_selector") is False
            and receipt.get("post_install_rescoring") is False
            and report.get("post_install_rescoring_performed") is False
            and report.get("preinstall_candidate_selector_active") is True
            and report.get("legacy_v4_report_claimed") is False
            and isinstance(selected_id, str)
            and bool(selected_id)
            and receipt.get("selected_candidate_id") == selected_id
            and selector_report.get("selected_candidate_id") == selected_id
            and isinstance(selected_sha, str)
            and _SHA256.fullmatch(selected_sha) is not None
            and receipt.get("selected_candidate_state_sha256")
            == selected_sha
            and selector_report.get("selected_candidate_state_sha256")
            == selected_sha
            and isinstance(candidate_reports, Mapping)
            and isinstance(candidate_reports.get(selected_id), Mapping)
            and candidate_reports[selected_id].get(
                "candidate_state_sha256"
            )
            == selected_sha
            and isinstance(gate, Mapping)
            and gate.get("passed") is False
            and gate.get("enforced") is True
            and report.get("compiled_short_conv_fold_issued") is False
            and report.get("promotion_claimed") is False
            and report.get("promotable_candidate") is False
            and report.get("failure_evidence_only") is True
            and isinstance(execution, Mapping)
            and all(type(value) is int and value == 0 for value in execution.values())
            and _recursive_report_seals_valid(report)
            and _no_recursive_success_claims(report)
        )
    except (
        AttributeError,
        KeyError,
        OverflowError,
        TypeError,
        ValueError,
    ):
        return False


def _valid_short_conv_v5_selector_receipt_only(
    report: Mapping[str, Any],
) -> bool:
    receipt = report.get("preinstall_candidate_selector_receipt")
    if not isinstance(receipt, Mapping):
        return False
    selector_report = receipt.get("selector_report")
    candidate_reports = report.get("candidate_selection_reports")
    selected_id = report.get("selected_candidate_id")
    selection = report.get("selection_evaluation")
    selected_sha = (
        selection.get("candidate_state_sha256")
        if isinstance(selection, Mapping)
        else None
    )
    receipt_body = dict(receipt)
    receipt_seal = receipt_body.pop("report_sha256", None)
    selector_body = (
        dict(selector_report)
        if isinstance(selector_report, Mapping)
        else {}
    )
    selector_seal = selector_body.pop("report_sha256", None)
    return bool(
        receipt.get("schema")
        == LFM2_CPCF_SHORT_CONV_V5_SELECTOR_RECEIPT_SCHEMA
        and isinstance(receipt_seal, str)
        and _SHA256.fullmatch(receipt_seal) is not None
        and _sha(receipt_body) == receipt_seal
        and report.get("preinstall_candidate_selector_receipt_sha256")
        == receipt_seal
        and isinstance(selector_report, Mapping)
        and isinstance(selector_seal, str)
        and _SHA256.fullmatch(selector_seal) is not None
        and _sha(selector_body) == selector_seal
        and receipt.get("selector_report_sha256") == selector_seal
        and receipt.get(
            "all_live_candidate_states_verified_before_callback"
        )
        is True
        and receipt.get(
            "all_live_candidate_states_verified_after_callback"
        )
        is True
        and receipt.get("live_candidate_state_bytes_unchanged") is True
        and receipt.get("candidate_state_tensors_exposed_to_selector")
        is False
        and receipt.get("fit_reports_exposed_to_selector") is False
        and receipt.get("validation_reports_exposed_to_selector") is False
        and receipt.get("post_install_rescoring") is False
        and report.get("post_install_rescoring_performed") is False
        and report.get("preinstall_candidate_selector_active") is True
        and receipt.get("selected_candidate_id") == selected_id
        and selector_report.get("selected_candidate_id") == selected_id
        and receipt.get("selected_candidate_state_sha256")
        == selected_sha
        and selector_report.get("selected_candidate_state_sha256")
        == selected_sha
        and isinstance(candidate_reports, Mapping)
        and isinstance(candidate_reports.get(selected_id), Mapping)
        and candidate_reports[selected_id].get(
            "candidate_state_sha256"
        )
        == selected_sha
    )


def _valid_short_conv_v5_numerical_frontier_validation_report(
    report: Mapping[str, Any],
) -> bool:
    """Validate explicit v5 numerical evidence without weakening v4."""

    try:
        request = report.get("numerical_frontier_request")
        frontier = report.get("numerical_frontier")
        candidate_fit = report.get("candidate_fit_reports")
        candidate_selection = report.get("candidate_selection_reports")
        pool = report.get("candidate_pool_report")
        ranking = report.get("selection_ranking")
        fit_evaluation = report.get("fit_evaluation")
        selection_evaluation = report.get("selection_evaluation")
        validation_evaluation = report.get("validation_evaluation")
        gate = report.get("behavior_gate")
        gate_policy = report.get("gate_policy")
        case_seals = report.get("case_authentication_sha256")
        execution = report.get("execution")
        solver_policy = parse_lfm2_cpcf_short_conv_v4_solver_policy(
            report.get("solver_policy")
        )

        def sealed(
            value: Any,
            schema: str,
        ) -> bool:
            if not isinstance(value, Mapping) or value.get("schema") != schema:
                return False
            body = dict(value)
            seal = body.pop("report_sha256", None)
            return bool(
                isinstance(seal, str)
                and _SHA256.fullmatch(seal) is not None
                and _sha(body) == seal
            )

        if not (
            report.get("schema")
            == LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_FAILURE_SCHEMA
            and solver_policy.verify_authentication()
            and report.get("solver_policy_sha256")
            == solver_policy.authentication_sha256
            and report.get("solver_schema")
            == LFM2_CPCF_SHORT_CONV_V4_SCHEMA
            and sealed(
                request,
                LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_REQUEST_SCHEMA,
            )
            and sealed(
                frontier,
                LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_SCHEMA,
            )
            and sealed(
                pool,
                LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_POOL_SCHEMA,
            )
        ):
            return False
        request_sha = request["report_sha256"]
        frontier_sha = frontier["report_sha256"]
        if not (
            report.get("numerical_frontier_request_sha256") == request_sha
            and report.get("numerical_frontier_report_sha256")
            == frontier_sha
            and frontier.get("request_sha256") == request_sha
            and pool.get("request_sha256") == request_sha
            and pool.get("numerical_frontier_report_sha256")
            == frontier_sha
            and request.get("fit_only") is True
            and request.get("selection_cases_exposed") is False
            and request.get("validation_cases_exposed") is False
            and request.get("policy_sha256")
            == solver_policy.authentication_sha256
            and request.get("source_tensor_inventory_sha256")
            == _sha(request.get("source_tensor_sha256"))
            and request.get("parent_context_inventory_sha256")
            == _sha(request.get("parent_context_sha256"))
            and request.get("target_template_inventory_sha256")
            == _sha(request.get("target_template_sha256"))
            and frontier.get("fit_only") is True
            and frontier.get("selection_cases_consumed") == 0
            and frontier.get("validation_cases_consumed") == 0
            and frontier.get("fit_case_authentication_sha256")
            == request.get("fit_case_authentication_sha256")
            and isinstance(case_seals, Mapping)
            and set(case_seals) == {"fit", "selection", "validation"}
            and all(
                isinstance(values, list)
                and values
                and all(
                    isinstance(value, str)
                    and _SHA256.fullmatch(value) is not None
                    for value in values
                )
                for values in case_seals.values()
            )
            and case_seals.get("fit")
            == request.get("fit_case_authentication_sha256")
        ):
            return False
        raw_reports = frontier.get("raw_candidate_reports")
        raw_ids = frontier.get("raw_candidate_ids")
        raw_state_sha = frontier.get("raw_candidate_state_sha256")
        raw_report_sha = frontier.get("raw_candidate_report_sha256")
        if not (
            isinstance(raw_reports, Mapping)
            and isinstance(raw_ids, list)
            and raw_ids
            and len(raw_ids) == len(set(raw_ids))
            and set(raw_ids) == set(raw_reports)
            and isinstance(raw_state_sha, Mapping)
            and set(raw_state_sha) == set(raw_ids)
            and isinstance(raw_report_sha, Mapping)
            and set(raw_report_sha) == set(raw_ids)
            and frontier.get("raw_candidate_count") == len(raw_ids)
            and all(
                sealed(
                    raw_reports[candidate_id],
                    LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_RAW_CANDIDATE_SCHEMA,
                )
                and raw_reports[candidate_id].get("candidate_id")
                == candidate_id
                and raw_reports[candidate_id].get("request_sha256")
                == request_sha
                and raw_reports[candidate_id].get("state_sha256")
                == raw_state_sha[candidate_id]
                and raw_reports[candidate_id].get("report_sha256")
                == raw_report_sha[candidate_id]
                and raw_reports[candidate_id].get("fit_only") is True
                and raw_reports[candidate_id].get(
                    "selection_cases_consumed"
                )
                == 0
                and raw_reports[candidate_id].get(
                    "validation_cases_consumed"
                )
                == 0
                for candidate_id in raw_ids
            )
        ):
            return False
        participants = frontier.get("source_numeric_participant_indices")
        source_count = frontier.get("source_channel_count")
        selected_count = frontier.get("source_channels_selected")
        dropped_count = frontier.get("source_channels_dropped")
        full_coverage = frontier.get("full_source_numeric_participation")
        if not (
            type(source_count) is int
            and source_count > 0
            and isinstance(participants, list)
            and participants == sorted(set(participants))
            and all(
                type(index) is int and 0 <= index < source_count
                for index in participants
            )
            and selected_count == len(participants)
            and dropped_count == source_count - len(participants)
            and full_coverage is (len(participants) == source_count)
            and pool.get("source_channels_selected") == selected_count
            and pool.get("source_channels_dropped") == dropped_count
            and pool.get("full_source_numeric_participation")
            is full_coverage
            and report.get("source_channels_selected") == selected_count
            and report.get("source_channels_dropped") == dropped_count
        ):
            return False
        if not (
            isinstance(candidate_fit, Mapping)
            and candidate_fit
            and isinstance(candidate_selection, Mapping)
            and set(candidate_fit) == set(candidate_selection)
            and pool.get("candidate_ids") == sorted(candidate_fit)
            and pool.get("unique_target_dtype_candidate_count")
            == len(candidate_fit)
            and pool.get("raw_candidate_count") == len(raw_ids)
            and pool.get("target_dtype_duplicate_count")
            == len(raw_ids) - len(candidate_fit)
            and pool.get("fit_only_candidate_generation") is True
            and pool.get("selection_cases_consumed_during_generation") == 0
            and pool.get("validation_cases_consumed_during_generation") == 0
            and pool.get("legacy_v4_candidate_pool_claimed") is False
        ):
            return False
        for candidate_id, provenance in candidate_fit.items():
            selection = candidate_selection[candidate_id]
            if not (
                sealed(
                    provenance,
                    LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_PROVENANCE_SCHEMA,
                )
                and provenance.get("candidate_id") == candidate_id
                and provenance.get("request_sha256") == request_sha
                and provenance.get("numerical_frontier_report_sha256")
                == frontier_sha
                and provenance.get("fit_case_authentication_sha256")
                == case_seals["fit"]
                and provenance.get("fit_only") is True
                and provenance.get("selection_cases_consumed") == 0
                and provenance.get("validation_cases_consumed") == 0
                and provenance.get("legacy_v4_candidate_provenance_claimed")
                is False
                and isinstance(selection, Mapping)
                and selection.get("split") == "selection"
                and selection.get("candidate_state_sha256")
                == provenance.get("target_dtype_candidate_state_sha256")
                and pool["candidate_state_sha256"].get(candidate_id)
                == selection.get("candidate_state_sha256")
            ):
                return False
        selected_id = report.get("selected_candidate_id")
        selected_sha = selection_evaluation.get(
            "candidate_state_sha256"
        )
        evaluations = (
            ("fit", fit_evaluation),
            ("selection", selection_evaluation),
            ("validation", validation_evaluation),
        )
        if not (
            isinstance(selected_id, str)
            and selected_id in candidate_fit
            and candidate_selection[selected_id].get(
                "candidate_state_sha256"
            )
            == selected_sha
            and all(
                isinstance(evaluation, Mapping)
                and evaluation.get("split") == split
                and evaluation.get("candidate_state_sha256")
                == selected_sha
                and evaluation.get(
                    "actual_parameter_and_intermediate_target_dtype_replay"
                )
                is True
                for split, evaluation in evaluations
            )
            and _valid_short_conv_v4_selection_ranking(
                ranking,
                candidate_selection=candidate_selection,
                solver_policy=solver_policy,
                feasibility_policy=pool[
                    "selection_feasibility_policy"
                ],
            )
            and (
                (
                    "preinstall_candidate_selector_receipt" not in report
                    and ranking.get("selected_candidate_id")
                    == selected_id
                )
                or (
                    "preinstall_candidate_selector_receipt" in report
                    and report.get("default_v4_selected_candidate_id")
                    == ranking.get("selected_candidate_id")
                )
            )
            and isinstance(gate_policy, Mapping)
            and isinstance(gate, Mapping)
            and _valid_short_conv_v4_behavior_gate(
                gate,
                gate_policy=gate_policy,
                fit_evaluation=fit_evaluation,
                validation_evaluation=validation_evaluation,
                full_source_coverage=full_coverage,
            )
            and gate.get("passed") is False
            and gate.get("enforced") is True
        ):
            return False
        selector_present = (
            "preinstall_candidate_selector_receipt" in report
        )
        return bool(
            (not selector_present)
            or _valid_short_conv_v5_selector_receipt_only(report)
        ) and bool(
            report.get("numerical_frontier_adapter_active") is True
            and report.get("legacy_v4_numerical_generation_claimed")
            is False
            and report.get("legacy_v4_report_claimed") is False
            and report.get("fit_only_numerical_generation") is True
            and report.get("selection_used_only_for_candidate_ranking")
            is True
            and report.get("validation_used_only_after_candidate_freeze")
            is True
            and report.get("source_channels_pruned") == 0
            and report.get("source_channels_quantized") == 0
            and isinstance(execution, Mapping)
            and set(execution) == _SHORT_CONV_V4_EXECUTION_FIELDS
            and all(
                type(value) is int and value == 0
                for value in execution.values()
            )
            and report.get("compiled_short_conv_fold_issued") is False
            and report.get("promotion_claimed") is False
            and report.get("promotable_candidate") is False
            and report.get("failure_evidence_only") is True
            and _recursive_report_seals_valid(report)
            and _no_recursive_success_claims(report)
        )
    except (
        AttributeError,
        KeyError,
        OverflowError,
        TypeError,
        ValueError,
    ):
        return False


def _valid_short_conv_validation_report(
    report: Mapping[str, Any],
) -> bool:
    try:
        schema = report.get("schema")
        if schema == LFM2_CPCF_SHORT_CONV_FOLD_SCHEMA:
            return _valid_short_conv_v1_validation_report(report)
        if schema in {
            LFM2_CPCF_SHORT_CONV_V4_FAILURE_V1_SCHEMA,
            LFM2_CPCF_SHORT_CONV_V4_FAILURE_SCHEMA,
        }:
            return _valid_short_conv_v4_validation_report(report)
        if schema == LFM2_CPCF_SHORT_CONV_V5_SELECTOR_FAILURE_SCHEMA:
            return _valid_short_conv_v5_selector_validation_report(report)
        if (
            schema
            == LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_FAILURE_SCHEMA
        ):
            return _valid_short_conv_v5_numerical_frontier_validation_report(
                report
            )
        return False
    except (
        AttributeError,
        KeyError,
        OverflowError,
        TypeError,
        ValueError,
    ):
        return False


def _verified_short_conv_validation_failure(
    error: Exception,
) -> tuple[Mapping[str, Any] | None, bytes | None, str | None]:
    if not isinstance(error, Lfm2CpcfShortConvValidationError):
        return None, None, None
    report = error.report
    if not isinstance(report, Mapping):
        raise Lfm2CpcfLayerInstallError(
            "Short-conv validation failure lacks a structured report."
        )
    report_body = dict(report)
    report_sha = _digest(
        report_body.pop("report_sha256", None),
        "short_conv_validation_report.report_sha256",
    )
    if _sha(report_body) != report_sha:
        raise Lfm2CpcfLayerInstallError(
            "Short-conv validation failure report is not self-sealed."
        )
    if not _valid_short_conv_validation_report(report):
        raise Lfm2CpcfLayerInstallError(
            "Short-conv validation failure report has invalid schema, "
            "gate semantics, or a nested success claim."
        )
    report_bytes = _canonical_json(dict(report))
    return (
        report,
        report_bytes,
        hashlib.sha256(report_bytes).hexdigest(),
    )


def _wrap_layer_failure(
    error: Exception,
    *,
    stage: str,
    layer_index: int,
    operator_kind: str,
    input_bank_sha256: str,
    input_tensor_sha256: str,
    boundary_charts_sha256: str,
    boundary_charts_payload_sha256: str,
    rollback_proof: Mapping[str, Any],
) -> Lfm2CpcfLayerCompilationFailure:
    nested_report, nested_bytes, nested_bytes_sha = (
        _verified_short_conv_validation_failure(error)
    )
    body = {
        "schema": LFM2_CPCF_LAYER_FAILURE_SCHEMA,
        "stage": stage,
        "layer_index": layer_index,
        "operator_kind": operator_kind,
        "input_bank_sha256": input_bank_sha256,
        "input_tensor_sha256": input_tensor_sha256,
        "boundary_charts_sha256": boundary_charts_sha256,
        "boundary_charts_payload_sha256": (boundary_charts_payload_sha256),
        "cause": {
            "module": type(error).__module__,
            "type": type(error).__qualname__,
            "message": str(error),
        },
        "nested_short_conv_validation_report": (
            None if nested_report is None else dict(nested_report)
        ),
        "nested_short_conv_validation_report_sha256": (
            None if nested_report is None else nested_report["report_sha256"]
        ),
        "nested_short_conv_validation_report_canonical_bytes_sha256": (
            nested_bytes_sha
        ),
        "nested_short_conv_validation_report_preserved_byte_for_byte": (
            nested_report is not None
        ),
        "rollback_proof": dict(rollback_proof),
        "layer_result_issued": False,
        "promotion_claimed": False,
    }
    report = {**body, "report_sha256": _sha(body)}
    message = (
        f"LFM2 CPCF layer {layer_index} ({operator_kind}) failed at "
        f"{stage} after exact rollback: {error}"
    )
    return Lfm2CpcfLayerCompilationFailure(
        message,
        stage=stage,
        layer_index=layer_index,
        operator_kind=operator_kind,
        input_bank_sha256=input_bank_sha256,
        input_tensor_sha256=input_tensor_sha256,
        boundary_charts_sha256=boundary_charts_sha256,
        boundary_charts_payload_sha256=(boundary_charts_payload_sha256),
        rollback_proof=rollback_proof,
        report=report,
        short_conv_validation_report=nested_report,
        short_conv_validation_report_bytes=nested_bytes,
    )


def compile_and_install_lfm2_cpcf_layer(
    source_model: Any,
    target_model: Any,
    *,
    layer_index: int,
    input_bank: Lfm2CpcfPhaseStateBank,
    boundary_charts: Lfm2CpcfLayerBoundaryCharts,
    ridge: float = 1e-8,
    short_conv_gate_policy: Lfm2CpcfShortConvGatePolicy | None = None,
    short_conv_v4_solver_policy: Lfm2CpcfShortConvV4SolverPolicy | None = None,
    candidate_replay_evaluator: Lfm2CpcfCandidateReplayEvaluator | None = None,
    candidate_selector: Lfm2CpcfShortConvV5CandidateSelector | None = None,
    numerical_frontier_adapter: (
        Lfm2CpcfShortConvNumericalFrontierAdapter | None
    ) = None,
) -> Lfm2CpcfCompiledLayerResult:
    """Compile, install, and replay one next sequential LFM2 target layer."""

    index = _nonnegative_int(layer_index, "layer_index")
    ridge_value = _nonnegative_float(ridge, "ridge")
    if not isinstance(input_bank, Lfm2CpcfPhaseStateBank):
        raise TypeError("input_bank must be Lfm2CpcfPhaseStateBank.")
    if not isinstance(boundary_charts, Lfm2CpcfLayerBoundaryCharts):
        raise TypeError("boundary_charts must be Lfm2CpcfLayerBoundaryCharts.")
    if short_conv_v4_solver_policy is not None and not isinstance(
        short_conv_v4_solver_policy,
        Lfm2CpcfShortConvV4SolverPolicy,
    ):
        raise TypeError(
            "short_conv_v4_solver_policy must be Lfm2CpcfShortConvV4SolverPolicy."
        )
    if (
        short_conv_v4_solver_policy is not None
        and not short_conv_v4_solver_policy.verify_authentication()
    ):
        raise Lfm2CpcfLayerCompilerError(
            "short_conv_v4_solver_policy authentication failed."
        )
    if candidate_replay_evaluator is not None and not callable(
        getattr(candidate_replay_evaluator, "evaluate", None)
    ):
        raise TypeError(
            "candidate_replay_evaluator must implement the authenticated "
            "candidate replay evaluator protocol."
        )
    if (
        candidate_replay_evaluator is not None
        and short_conv_v4_solver_policy is None
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Injected candidate replay is defined only for the authenticated "
            "short-conv v4 solver lane."
        )
    if candidate_selector is not None and not callable(
        getattr(candidate_selector, "select", None)
    ):
        raise TypeError(
            "candidate_selector must implement select(request)."
        )
    if (
        candidate_selector is not None
        and short_conv_v4_solver_policy is None
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Pre-install candidate selection is defined only for the "
            "authenticated short-conv v4 solver lane."
        )
    if numerical_frontier_adapter is not None and not callable(
        getattr(numerical_frontier_adapter, "generate", None)
    ):
        raise TypeError(
            "numerical_frontier_adapter must implement generate(request)."
        )
    if (
        numerical_frontier_adapter is not None
        and short_conv_v4_solver_policy is None
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Numerical-frontier generation is defined only for the "
            "authenticated short-conv v4 solver lane."
        )
    if input_bank.layer_index != index:
        raise Lfm2CpcfLayerCompilerError(
            "Input state bank is not the next sequential target layer."
        )
    if boundary_charts.layer_index != index:
        raise Lfm2CpcfLayerCompilerError(
            "Boundary charts do not describe the requested layer."
        )
    if input_bank.hidden_size != boundary_charts.target_width:
        raise Lfm2CpcfLayerCompilerError(
            "Actual child state width differs from boundary-chart width."
        )
    split_counts = {
        split: sum(phase.split == split for phase in input_bank.phases)
        for split in _SHORT_CONV_CASE_SPLITS
    }
    if split_counts["fit"] == 0 or split_counts["validation"] == 0:
        raise Lfm2CpcfLayerCompilerError(
            "Layer compilation requires nonempty fit and validation phases."
        )
    source_layer = _layer(source_model, index, "source_model")
    target_layer = _layer(target_model, index, "target_model")
    source_is_attention = bool(getattr(source_layer, "is_attention_layer", False))
    target_is_attention = bool(getattr(target_layer, "is_attention_layer", False))
    operator_kind = "full_attention" if source_is_attention else "short_conv"
    if (
        source_is_attention != target_is_attention
        or operator_kind != boundary_charts.operator_kind
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Source, target, and boundary chart operator kinds differ."
        )
    for phase in input_bank.phases:
        evidence = phase.parent_evidence.layers[index]
        if (
            evidence.source_width != boundary_charts.source_width
            or evidence.operator_kind != operator_kind
        ):
            raise Lfm2CpcfLayerCompilerError(
                f"Phase {phase.identity!r} evidence differs from layer charts."
            )

    source_before = _layer_snapshot(source_layer)
    target_before = _layer_snapshot(target_layer)
    evidence_before = _live_layer_evidence_sha256(
        input_bank.phases,
        index,
    )
    input_before = _live_input_bank_sha256(input_bank)
    chart_seal_before = boundary_charts.layer_charts_sha256
    chart_payload_before = _boundary_charts_payload_sha256(boundary_charts)
    stage = "preinstall_replay"

    try:
        preinstall_replay = replay_lfm2_cpcf_phase_bank(
            target_model,
            layer_index=index,
            input_bank=input_bank,
        )
        stage = "operator_compile"
        norm_report: Mapping[str, Any] | None = None
        if operator_kind == "full_attention":
            source_attention = source_layer.self_attn
            target_attention = target_layer.self_attn
            source_q_norm = _weight(
                source_attention.q_layernorm,
                "source_attention.q_layernorm",
                dimensions=1,
            )
            source_k_norm = _weight(
                source_attention.k_layernorm,
                "source_attention.k_layernorm",
                dimensions=1,
            )
            source_q_epsilon = _variance_epsilon(
                source_attention.q_layernorm,
                "source_attention.q_layernorm",
            )
            source_k_epsilon = _variance_epsilon(
                source_attention.k_layernorm,
                "source_attention.k_layernorm",
            )
            target_q_epsilon = _variance_epsilon(
                target_attention.q_layernorm,
                "target_attention.q_layernorm",
            )
            target_k_epsilon = _variance_epsilon(
                target_attention.k_layernorm,
                "target_attention.k_layernorm",
            )
            if not (
                source_q_epsilon
                == source_k_epsilon
                == target_q_epsilon
                == target_k_epsilon
            ):
                raise Lfm2CpcfLayerCompilerError(
                    "Source/target Q/K RMSNorm epsilon differs."
                )
            cases = _attention_cases(
                source_model=source_model,
                target_model=target_model,
                source_layer=source_layer,
                layer_index=index,
                input_bank=input_bank,
                preinstall_replay=preinstall_replay,
                chart=boundary_charts,
            )
            operator_artifact = compile_lfm2_structural_residual_attention_fold(
                source_query_weight=source_attention.q_proj.weight,
                source_key_weight=source_attention.k_proj.weight,
                source_value_weight=source_attention.v_proj.weight,
                source_output_weight=source_attention.out_proj.weight,
                target_query_weight_template=target_attention.q_proj.weight,
                target_key_weight_template=target_attention.k_proj.weight,
                target_value_weight_template=target_attention.v_proj.weight,
                target_output_weight_template=target_attention.out_proj.weight,
                cases=cases,
                head_dim=int(source_attention.head_dim),
                query_norm_weight=source_q_norm,
                key_norm_weight=source_k_norm,
                residual_assignment_sha256=(
                    boundary_charts.post_operator.report["report_sha256"]
                ),
                norm_epsilon=source_q_epsilon,
                ridge=ridge_value,
                max_prior_delta_ratio=None,
            )
            stage = "operator_install"
            norm_report = _copy_attention_norms(
                source_attention,
                target_attention,
            )
            raw_operator_install = install_lfm2_structural_attention_fold(
                target_attention,
                operator_artifact,
            ).as_dict()
            operator_install_report = MappingProxyType(
                {
                    "operator": raw_operator_install,
                    "attention_norms": dict(norm_report),
                }
            )
        else:
            source_conv = source_layer.conv
            target_conv = target_layer.conv
            cases = _short_conv_cases(
                layer_index=index,
                input_bank=input_bank,
                preinstall_replay=preinstall_replay,
                chart=boundary_charts,
                use_v4=short_conv_v4_solver_policy is not None,
            )
            policy = (
                Lfm2CpcfShortConvGatePolicy(enforce=False)
                if short_conv_gate_policy is None
                else short_conv_gate_policy
            )
            if short_conv_v4_solver_policy is None:
                operator_artifact = compile_lfm2_cpcf_short_conv_fold(
                    source_fir_weight=source_conv.conv.weight,
                    source_fir_bias=source_conv.conv.bias,
                    target_in_proj_weight_template=target_conv.in_proj.weight,
                    target_in_proj_bias_template=target_conv.in_proj.bias,
                    target_conv_weight_template=target_conv.conv.weight,
                    target_conv_bias_template=target_conv.conv.bias,
                    target_out_proj_weight_template=target_conv.out_proj.weight,
                    target_out_proj_bias_template=target_conv.out_proj.bias,
                    cases=cases,
                    ridge=ridge_value,
                    gate_policy=policy,
                )
            else:
                operator_artifact = compile_lfm2_cpcf_short_conv_v4_fold(
                    source_in_proj_weight=source_conv.in_proj.weight,
                    source_in_proj_bias=source_conv.in_proj.bias,
                    source_fir_weight=source_conv.conv.weight,
                    source_fir_bias=source_conv.conv.bias,
                    source_out_proj_weight=source_conv.out_proj.weight,
                    source_out_proj_bias=source_conv.out_proj.bias,
                    target_in_proj_weight_template=target_conv.in_proj.weight,
                    target_in_proj_bias_template=target_conv.in_proj.bias,
                    target_conv_weight_template=target_conv.conv.weight,
                    target_conv_bias_template=target_conv.conv.bias,
                    target_out_proj_weight_template=target_conv.out_proj.weight,
                    target_out_proj_bias_template=target_conv.out_proj.bias,
                    output_encoder=(boundary_charts.post_operator.mapping.encoder),
                    cases=cases,
                    solver_policy=short_conv_v4_solver_policy,
                    gate_policy=policy,
                    candidate_replay_evaluator=candidate_replay_evaluator,
                    candidate_selector=candidate_selector,
                    numerical_frontier_adapter=(
                        numerical_frontier_adapter
                    ),
                )
            stage = "operator_install"
            operator_install_report = _install_short_conv(
                target_conv,
                operator_artifact,
            )

        stage = "operator_replay"
        operator_replay = replay_lfm2_cpcf_phase_bank(
            target_model,
            layer_index=index,
            input_bank=input_bank,
        )
        stage = "mlp_compile"
        source_mlp = source_layer.feed_forward
        target_mlp = target_layer.feed_forward
        mlp_cases = _mlp_cases(
            layer_index=index,
            input_bank=input_bank,
            operator_replay=operator_replay,
            chart=boundary_charts,
        )
        mlp_artifact = compile_lfm2_structural_residual_mlp_fold(
            source_gate_weight=source_mlp.w1.weight,
            source_up_weight=source_mlp.w3.weight,
            source_down_weight=source_mlp.w2.weight,
            target_gate_weight_template=target_mlp.w1.weight,
            target_up_weight_template=target_mlp.w3.weight,
            target_down_weight_template=target_mlp.w2.weight,
            cases=mlp_cases,
            residual_assignment_sha256=(
                boundary_charts.post_layer.report["report_sha256"]
            ),
            ridge=ridge_value,
            max_prior_delta_ratio=None,
        )
        stage = "mlp_install"
        mlp_install_report = install_lfm2_structural_mlp_fold(
            target_mlp,
            mlp_artifact,
        ).as_dict()
        stage = "final_replay"
        final_replay = replay_lfm2_cpcf_phase_bank(
            target_model,
            layer_index=index,
            input_bank=input_bank,
        )
        stage = "endpoint_evaluation"
        split_metrics = _split_endpoint_metrics(
            layer_index=index,
            input_bank=input_bank,
            operator_replay=operator_replay,
            final_replay=final_replay,
            chart=boundary_charts,
        )

        stage = "immutability_audit"
        source_after = _layer_snapshot(source_layer)
        evidence_after = _live_layer_evidence_sha256(
            input_bank.phases,
            index,
        )
        input_after = _live_input_bank_sha256(input_bank)
        _assert_snapshot_identity(source_layer, source_before)
        _assert_snapshot_identity(target_layer, target_before)
        if (
            source_after.tensor_bytes_sha256 != source_before.tensor_bytes_sha256
            or source_after.full_inventory_sha256 != source_before.full_inventory_sha256
            or source_after.tensor_identity != source_before.tensor_identity
            or source_after.modes != source_before.modes
            or evidence_after != evidence_before
            or input_after != input_before
            or boundary_charts.layer_charts_sha256 != chart_seal_before
            or _boundary_charts_payload_sha256(boundary_charts) != chart_payload_before
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Source, evidence, charts, or actual input bank changed during "
                "closed-form layer compilation."
            )
        stage = "result_seal"
        body = {
            "schema": LFM2_CPCF_LAYER_COMPILER_SCHEMA,
            "layer_index": index,
            "operator_kind": operator_kind,
            "source_width": boundary_charts.source_width,
            "target_width": boundary_charts.target_width,
            "input_bank_sha256": input_bank.state_bank_sha256,
            "next_bank_sha256": final_replay.post_layer_bank.state_bank_sha256,
            "boundary_charts_sha256": boundary_charts.layer_charts_sha256,
            "preinstall_replay_sha256": preinstall_replay.report["report_sha256"],
            "operator_replay_sha256": operator_replay.report["report_sha256"],
            "final_replay_sha256": final_replay.report["report_sha256"],
            "operator_artifact": operator_artifact.as_dict(),
            "mlp_artifact": mlp_artifact.as_dict(),
            "operator_install_report": dict(operator_install_report),
            "mlp_install_report": dict(mlp_install_report),
            "split_metrics": {
                split: dict(split_metrics[split]) for split in _EVALUATION_SPLITS
            },
            "split_phase_counts": split_counts,
            "solver_evidence_policy": {
                "operator_case_count": len(cases),
                "selection_used_for_operator_fit": False,
                "selection_used_for_mlp_fit": False,
                "validation_used_for_attention_fit": False,
                "validation_used_for_mlp_fit": False,
                "validation_used_for_short_conv_fit": False,
                "attention_fit_case_granularity": (
                    "one_complete_causal_trajectory_per_fit_record"
                    if operator_kind == "full_attention"
                    else "not_applicable"
                ),
                "attention_key_linear_history": (
                    "exact_prefill_plus_ordered_decode_suffixes"
                    if operator_kind == "full_attention"
                    else "not_applicable"
                ),
                "attention_inverse_rope_preimage_used": (
                    False if operator_kind == "full_attention" else "not_applicable"
                ),
                "attention_repeated_prefix_fit_rows": (
                    False if operator_kind == "full_attention" else "not_applicable"
                ),
                "short_conv_validation_role": (
                    "post_materialization_heldout_evaluator"
                    if operator_kind == "short_conv"
                    else "not_applicable"
                ),
            },
            "actual_child_state_policy": {
                "operator_fit_input": (
                    "causally_stitched_full_prefix_preinstall_target_native_replay"
                ),
                "mlp_fit_input": ("suffix_of_full_prefix_installed_operator_replay"),
                "next_layer_input": (
                    "complete_prefill_plus_ordered_decode_suffix_layer_replay"
                ),
                "recomputed_historical_prefix_propagated": False,
                "independent_parent_projection_substitution": False,
            },
            "audit": {
                "source_state_sha256_before": (source_before.tensor_bytes_sha256),
                "source_state_sha256_after": (source_after.tensor_bytes_sha256),
                "source_unchanged": True,
                "evidence_sha256_before": evidence_before,
                "evidence_sha256_after": evidence_after,
                "evidence_unchanged": True,
                "input_bank_sha256_before": input_before,
                "input_bank_sha256_after": input_after,
                "input_bank_unchanged": True,
                "chart_seal_unchanged": True,
                "target_tensor_identities_preserved": True,
                "target_gradient_state_preserved": True,
                "target_training_modes_preserved": True,
                "layer_transaction_rollback_available": True,
            },
            "execution": {
                "closed_form_compilers": True,
                "parent_model_forward_calls": 0,
                "target_model_forward_calls": 0,
                "target_layer_forward_calls": 0,
                "generation_calls": 0,
                "cache_generation_calls": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
                "training_steps": 0,
                "parameters_pruned": 0,
                "parameters_quantized": 0,
                "fallback_paths": 0,
            },
        }
        result_sha = _sha(body)
        return Lfm2CpcfCompiledLayerResult(
            layer_index=index,
            operator_kind=operator_kind,
            input_bank_sha256=input_bank.state_bank_sha256,
            operator_artifact=operator_artifact,
            mlp_artifact=mlp_artifact,
            operator_install_report=operator_install_report,
            mlp_install_report=mlp_install_report,
            preinstall_replay=preinstall_replay,
            operator_replay=operator_replay,
            final_replay=final_replay,
            split_metrics=split_metrics,
            report={**body, "result_sha256": result_sha},
            result_sha256=result_sha,
        )
    except BaseException as error:
        try:
            _restore_layer(target_layer, target_before)
            rollback_proof = _prove_exact_layer_rollback(
                target_layer,
                target_before,
                source_layer=source_layer,
                source_snapshot=source_before,
                input_bank=input_bank,
                layer_index=index,
                evidence_sha256_before=evidence_before,
                input_sha256_before=input_before,
                boundary_charts=boundary_charts,
                boundary_charts_sha256_before=chart_seal_before,
                boundary_charts_payload_sha256_before=(chart_payload_before),
            )
        except BaseException as rollback_error:
            raise Lfm2CpcfLayerInstallError(
                f"LFM2 CPCF layer {index} ({operator_kind}) failed at "
                f"{stage}, and exact layer rollback could not be proven."
            ) from rollback_error
        if not isinstance(error, Exception):
            try:
                error.add_note(
                    "LFM2 CPCF layer target was exactly restored before "
                    "cancellation propagated."
                )
            except AttributeError:  # pragma: no cover - Python boundary
                pass
            raise
        failure = _wrap_layer_failure(
            error,
            stage=stage,
            layer_index=index,
            operator_kind=operator_kind,
            input_bank_sha256=input_bank.state_bank_sha256,
            input_tensor_sha256=input_before,
            boundary_charts_sha256=chart_seal_before,
            boundary_charts_payload_sha256=chart_payload_before,
            rollback_proof=rollback_proof,
        )
        raise failure from error


__all__ = [
    "LFM2_CPCF_LAYER_COMPILER_SCHEMA",
    "LFM2_CPCF_LAYER_FAILURE_SCHEMA",
    "Lfm2CpcfCompiledLayerResult",
    "Lfm2CpcfLayerCompilationFailure",
    "Lfm2CpcfLayerCompilerError",
    "Lfm2CpcfLayerInstallError",
    "compile_and_install_lfm2_cpcf_layer",
    "invert_lfm2_rope",
    "recover_lfm2_full_key_linear_history",
]
