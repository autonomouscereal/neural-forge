"""No-training semantic-endpoint SwiGLU refolding for a dense LFM2 child.

This adapter consumes one parent-exact semantic endpoint per cached phase and
actual-child macrocell replay captured from the current child state.  FIT is
the only partition used by either analytic solve.  FIT orders structurally
eligible frozen candidates deterministically; FIT-CHECK, SELECTION, and
VALIDATION remain trajectory-isolated diagnostics for behavioral review.

Two candidates compete with the incumbent dense SwiGLU:

* a conservative producer/writer candidate keeps incumbent W1/W3 and solves
  only W2 from actual child post-SwiGLU contexts to mapped parent residual
  writes; and
* the full candidate plants an exact-zero child from coupled real parent
  gate/up medoids, represents every parent neuron through a sparse multi-anchor
  consumer plan, transports that plan through the authenticated residual chart,
  and applies FIT-only closed-form W1/W3/W2 corrections.  Nonzero incumbents
  retain the established full-source structural compiler.  A requested
  maximum delta ratio is retained only as a review diagnostic; it never clips
  either analytic proposal.

The authenticated incumbent is a valid no-op result only when every analytic
proposal is structurally invalid or tensor-identical to it.  Finite, dense,
target-shaped, target-dtype, effectful candidates are never rejected or rolled
back because a local numeric metric regressed.  Whole-response intelligence is
reviewed downstream rather than impersonated by a hard component threshold.

There are no model executions, gradients, optimizers, training steps, pruning,
quantization, sparsity, shared-parameter aliases, or numerical parent/child
weight-equality objectives in this module.  The final layer can explicitly
defer W2 to the fused semantic terminal/lexical writer while exporting four
authenticated ``Lfm2CpcfTerminalActionEvidence`` banks.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (
    Lfm2CpcfTerminalActionDownWeightPrior,
    Lfm2CpcfTerminalActionEvidence,
    lfm2_cpcf_terminal_action_tensor_sha256,
)
from neural_forge.hf_lfm2_mlp_qdeim import (
    LFM2_MLP_QDEIM_FIT_CORRECTED_CANDIDATE,
    LFM2_MLP_QDEIM_STRUCTURAL_PRIOR_CANDIDATE,
    Lfm2MlpQdeimFitCase,
    Lfm2MlpQdeimFunctionalSnapshotBlock,
    compile_lfm2_mlp_nonlinear_qdeim,
    lfm2_mlp_qdeim_fit_evidence_sha256,
    lfm2_mlp_qdeim_functional_snapshot_evidence_sha256,
    lfm2_mlp_qdeim_source_state_sha256,
    lfm2_mlp_qdeim_target_template_sha256,
)
from neural_forge.hf_lfm2_normalized_input_decoder import (
    LFM2_NORMALIZED_INPUT_DECODER_SCHEMA,
    Lfm2CompiledNormalizedInputDecoder,
    compile_lfm2_normalized_input_decoder,
)
from neural_forge.hf_lfm2_semantic_body_recipe import (
    Lfm2SemanticBodyRecipe,
    resolve_lfm2_semantic_body_recipe,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    Lfm2CachedMlpFoldCase,
    Lfm2CachedNormalizedInputMlpFoldCase,
    Lfm2CompiledNormalizedInputRealMedoidStructuralMlpFold,
    Lfm2CompiledStructuralMlpFold,
    compile_lfm2_normalized_input_real_medoid_structural_mlp_fold,
    compile_lfm2_structural_residual_mlp_fold,
    lfm2_authenticated_tensor_sha256,
    lfm2_normalized_input_mlp_fit_evidence_sha256,
)
from neural_forge.hf_lfm2_swiglu_tangent_bank import (
    build_lfm2_swiglu_tangent_bank,
    fingerprint_lfm2_swiglu_tangent_tensor,
)


LFM2_SEMANTIC_MLP_SOURCE_SCHEMA = (
    "neural-forge-lfm2-direct-local-semantic-mlp-source-v3"
)
LFM2_SEMANTIC_MLP_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-semantic-mlp-endpoint-evidence-v3"
)
LFM2_SEMANTIC_MLP_CANDIDATE_SCHEMA = (
    "neural-forge-lfm2-semantic-mlp-frozen-candidate-v1"
)
LFM2_SEMANTIC_MLP_PREPARATION_SCHEMA = "neural-forge-lfm2-semantic-mlp-preparation-v1"
LFM2_SEMANTIC_MLP_INSTALLATION_SCHEMA = "neural-forge-lfm2-semantic-mlp-installation-v1"
LFM2_SEMANTIC_MLP_COLD_START_REJECTION_SCHEMA = (
    "neural-forge-lfm2-semantic-mlp-cold-start-rejection-v1"
)

_PARTITIONS = ("fit", "fit_check", "selection", "validation")
_NORMALIZED_DECODER_RIDGE = 1e-8
_QDEIM_TANGENT_ANCHOR_COUNT = 2
_QDEIM_MLP_LANE_TO_CANDIDATE = MappingProxyType(
    {
        "nonlinear_qdeim_structural_priority": (
            LFM2_MLP_QDEIM_STRUCTURAL_PRIOR_CANDIDATE
        ),
        "nonlinear_qdeim_fit_corrected_priority": (
            LFM2_MLP_QDEIM_FIT_CORRECTED_CANDIDATE
        ),
    }
)
_CANDIDATE_ELIGIBILITY_BASIS = (
    "finite_dense_shape_storage_coverage_and_nonincumbent_effect"
)
_STRUCTURAL_NOOP_STATUS = "structurally-identical-or-invalid-incumbent-noop"
_MAX_FINITE_FLOAT = float.fromhex("0x1.fffffffffffffp+1023")
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
        "parameters_sparsified": 0,
        "parameter_aliases_created": 0,
    }
)


class Lfm2SemanticMlpRefoldError(RuntimeError):
    """Semantic MLP evidence, analytic selection, or installation failed."""


class Lfm2SemanticMlpColdStartRejectionError(Lfm2SemanticMlpRefoldError):
    """A blank dense MLP had no complete candidate accepted by held-out gates."""

    def __init__(self, message: str, report: Mapping[str, Any]) -> None:
        frozen = MappingProxyType(dict(report))
        self.report = frozen
        super().__init__(
            f"{message} rejection_report="
            + json.dumps(
                dict(frozen),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError("Semantic MLP refolding requires PyTorch.") from exc
    return torch


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2SemanticMlpRefoldError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _finite_tensor(
    value: Any,
    name: str,
    *,
    ndim: int,
    exact_dtype: Any | None = None,
) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != ndim
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
        or value.requires_grad
        or (exact_dtype is not None and value.dtype != exact_dtype)
        or not bool(torch.isfinite(value).all().item())
    ):
        dtype = "" if exact_dtype is None else f" {exact_dtype}"
        raise Lfm2SemanticMlpRefoldError(
            f"{name} must be a detached finite rank-{ndim}{dtype} tensor."
        )
    return value.detach().to(device="cpu").contiguous().clone()


def _reconstruct_parent_rmsnorm_bf16(
    residual: Any,
    *,
    norm_weight: Any,
    norm_epsilon: float,
) -> Any:
    """Reconstruct cached LFM2 FFN RMSNorm bytes without a model forward."""

    torch = _torch()
    residual_bf16 = residual.to(device="cpu", dtype=torch.bfloat16)
    norm_bf16 = norm_weight.to(device="cpu", dtype=torch.bfloat16)
    with torch.no_grad():
        residual_float = residual_bf16.to(torch.float32)
        variance = residual_float.square().mean(dim=-1, keepdim=True)
        normalized = norm_bf16 * (
            residual_float * torch.rsqrt(variance + float(norm_epsilon))
        ).to(torch.bfloat16)
    return normalized.detach().to(device="cpu").contiguous()


def _bf16_reconstruction_metrics(replayed: Any, cached: Any) -> dict[str, Any]:
    """Measure legacy BF16 drift without granting it admission authority."""

    torch = _torch()
    replayed_float = replayed.to(torch.float64)
    cached_float = cached.to(torch.float64)
    replayed_finite = torch.isfinite(replayed_float)
    replayed_all_finite = bool(replayed_finite.all().item())
    finite_replayed = torch.nan_to_num(
        replayed_float,
        nan=0.0,
        posinf=float(torch.finfo(torch.bfloat16).max),
        neginf=-float(torch.finfo(torch.bfloat16).max),
    )
    residual = finite_replayed - cached_float
    error_rms = float(torch.sqrt(residual.square().mean()).item())
    cached_rms = float(torch.sqrt(cached_float.square().mean()).item())
    machine_epsilon = float(torch.finfo(torch.bfloat16).eps)
    unit_roundoff = machine_epsilon / 2.0
    relative_rmse = min(
        error_rms / max(cached_rms, float(torch.finfo(torch.float64).tiny)),
        _MAX_FINITE_FLOAT,
    )
    elementwise_close = bool(
        replayed_all_finite
        and torch.allclose(
            finite_replayed,
            cached_float,
            rtol=machine_epsilon,
            atol=unit_roundoff,
        )
    )
    maximum_absolute_error = float(residual.abs().max().item())
    diagnostics_finite = all(
        math.isfinite(value)
        for value in (
            maximum_absolute_error,
            error_rms,
            relative_rmse,
        )
    )
    if not diagnostics_finite:
        raise Lfm2SemanticMlpRefoldError(
            "Parent RMSNorm replay diagnostics must remain finite."
        )
    legacy_envelope_passed = bool(
        elementwise_close and relative_rmse <= machine_epsilon
    )
    return {
        "policy_schema": "neural-forge-bf16-cross-kernel-equivalence-policy-v1",
        "maximum_absolute_error": maximum_absolute_error,
        "rmse": error_rms,
        "relative_rmse": relative_rmse,
        "elementwise_close": elementwise_close,
        "replayed_all_finite": replayed_all_finite,
        "nonfinite_replayed_element_count": int((~replayed_finite).sum().item()),
        "within_legacy_bf16_envelope": legacy_envelope_passed,
        "diagnostics_finite": True,
        "diagnostic_only": True,
        "numeric_admission_authority": False,
    }


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    return canonical_sha256(
        {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": [int(size) for size in tensor.shape],
            "data_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )


def _state_sha256(state: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {name: _tensor_sha256(value) for name, value in sorted(state.items())}
    )


def _parameter_count(model: Any) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += int(parameter.numel())
    return total


def _topology_sha256(model: Any) -> str:
    aliases: dict[int, int] = {}
    records = []
    for name, parameter in model.named_parameters(remove_duplicate=False):
        alias = aliases.setdefault(id(parameter), len(aliases))
        records.append(
            {
                "name": name,
                "alias_group": alias,
                "shape": [int(size) for size in parameter.shape],
                "dtype": str(parameter.dtype).removeprefix("torch."),
                "requires_grad": bool(parameter.requires_grad),
            }
        )
    if not records:
        raise Lfm2SemanticMlpRefoldError("Target model has no Parameters.")
    return canonical_sha256(records)


@dataclass(frozen=True, slots=True)
class Lfm2DirectLocalMlpSourceInventory:
    """Authenticated direct-local parent RMSNorm/W1/W3/W2 bundle for one layer."""

    layer_index: int
    parent_model_identity_sha256: str
    parent_checkpoint_manifest_sha256: str
    direct_local_loader_receipt_sha256: str
    residual_assignment_sha256: str
    compiler_lineage_sha256: str
    residual_chart: Any = field(repr=False, compare=False)
    ffn_norm_weight: Any = field(repr=False, compare=False)
    norm_epsilon: float
    gate_weight: Any = field(repr=False, compare=False)
    up_weight: Any = field(repr=False, compare=False)
    down_weight: Any = field(repr=False, compare=False)
    normalization_lineage_sha256: str = field(init=False)
    inventory_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
        ):
            raise Lfm2SemanticMlpRefoldError("Source layer index is invalid.")
        for value, name in (
            (self.parent_model_identity_sha256, "parent model identity"),
            (self.parent_checkpoint_manifest_sha256, "checkpoint manifest"),
            (self.direct_local_loader_receipt_sha256, "local loader receipt"),
            (self.residual_assignment_sha256, "residual assignment"),
            (self.compiler_lineage_sha256, "compiler lineage"),
        ):
            _digest(value, name)
        gate = _finite_tensor(self.gate_weight, "parent W1", ndim=2)
        up = _finite_tensor(self.up_weight, "parent W3", ndim=2)
        down = _finite_tensor(self.down_weight, "parent W2", ndim=2)
        norm = _finite_tensor(
            self.ffn_norm_weight,
            "parent FFN RMSNorm weight",
            ndim=1,
        )
        chart = _finite_tensor(
            self.residual_chart,
            "source-to-target residual chart",
            ndim=2,
        )
        epsilon = self.norm_epsilon
        source_intermediate, source_hidden = map(int, gate.shape)
        if (
            tuple(up.shape) != tuple(gate.shape)
            or tuple(down.shape) != (source_hidden, source_intermediate)
            or tuple(norm.shape) != (source_hidden,)
            or bool((norm == 0).any().item())
            or int(chart.shape[0]) != source_hidden
            or int(chart.shape[1]) <= 0
            or isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float))
            or not math.isfinite(float(epsilon))
            or float(epsilon) <= 0.0
        ):
            raise Lfm2SemanticMlpRefoldError(
                "Direct-local source tensors, positive norm epsilon, and residual "
                "chart do not form one dense source-hidden-to-target-hidden "
                "RMSNorm/SwiGLU transport bundle."
            )
        object.__setattr__(self, "residual_chart", chart)
        object.__setattr__(self, "ffn_norm_weight", norm)
        object.__setattr__(self, "norm_epsilon", float(epsilon))
        object.__setattr__(self, "gate_weight", gate)
        object.__setattr__(self, "up_weight", up)
        object.__setattr__(self, "down_weight", down)
        normalization_lineage = canonical_sha256(
            {
                "schema": "neural-forge-lfm2-direct-parent-mlp-normalization-v1",
                "layer_index": self.layer_index,
                "parent_model_identity_sha256": self.parent_model_identity_sha256,
                "parent_checkpoint_manifest_sha256": (
                    self.parent_checkpoint_manifest_sha256
                ),
                "direct_local_loader_receipt_sha256": (
                    self.direct_local_loader_receipt_sha256
                ),
                "compiler_lineage_sha256": self.compiler_lineage_sha256,
                "ffn_norm_weight_sha256": _tensor_sha256(norm),
                "norm_epsilon": float(epsilon),
            }
        )
        object.__setattr__(
            self,
            "normalization_lineage_sha256",
            normalization_lineage,
        )
        object.__setattr__(
            self,
            "inventory_sha256",
            canonical_sha256(self.authentication_body()),
        )

    @property
    def source_hidden_size(self) -> int:
        return int(self.gate_weight.shape[1])

    @property
    def source_intermediate_size(self) -> int:
        return int(self.gate_weight.shape[0])

    @property
    def target_hidden_size(self) -> int:
        return int(self.residual_chart.shape[1])

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_SEMANTIC_MLP_SOURCE_SCHEMA,
            "layer_index": self.layer_index,
            "parent_model_identity_sha256": self.parent_model_identity_sha256,
            "parent_checkpoint_manifest_sha256": (
                self.parent_checkpoint_manifest_sha256
            ),
            "direct_local_loader_receipt_sha256": (
                self.direct_local_loader_receipt_sha256
            ),
            "residual_assignment_sha256": self.residual_assignment_sha256,
            "residual_chart_receipt_sha256": self.residual_assignment_sha256,
            "compiler_lineage_sha256": self.compiler_lineage_sha256,
            "normalization_lineage_sha256": self.normalization_lineage_sha256,
            "source_hidden_size": self.source_hidden_size,
            "source_intermediate_size": self.source_intermediate_size,
            "target_hidden_size": self.target_hidden_size,
            "residual_chart_sha256": _tensor_sha256(self.residual_chart),
            "residual_chart_shape": [
                self.source_hidden_size,
                self.target_hidden_size,
            ],
            "ffn_norm_weight_sha256": _tensor_sha256(self.ffn_norm_weight),
            "ffn_norm_weight_shape": [self.source_hidden_size],
            "norm_epsilon": self.norm_epsilon,
            "weight_sha256": {
                "ffn_norm.weight": _tensor_sha256(self.ffn_norm_weight),
                "w1.weight": _tensor_sha256(self.gate_weight),
                "w3.weight": _tensor_sha256(self.up_weight),
                "w2.weight": _tensor_sha256(self.down_weight),
            },
            "weight_bundle_sha256": _state_sha256(
                {
                    "ffn_norm.weight": self.ffn_norm_weight,
                    "w1.weight": self.gate_weight,
                    "w3.weight": self.up_weight,
                    "w2.weight": self.down_weight,
                }
            ),
            "transport_bundle_sha256": _state_sha256(
                {
                    "residual_chart": self.residual_chart,
                    "ffn_norm.weight": self.ffn_norm_weight,
                    "w1.weight": self.gate_weight,
                    "w3.weight": self.up_weight,
                    "w2.weight": self.down_weight,
                }
            ),
            "normalization_bundle_sha256": canonical_sha256(
                {
                    "ffn_norm_weight_sha256": _tensor_sha256(self.ffn_norm_weight),
                    "norm_epsilon": self.norm_epsilon,
                    "normalization_lineage_sha256": (self.normalization_lineage_sha256),
                }
            ),
            "residual_chart_and_receipt_jointly_hashed": True,
            "normalization_and_loader_lineage_jointly_hashed": True,
            "loader_mode": "direct-local-safetensors",
            "auto_model_loader_used": False,
            "from_pretrained_used": False,
            "hub_access_used": False,
            "dense_floating_source": True,
        }

    def verify(self) -> bool:
        return self.inventory_sha256 == canonical_sha256(self.authentication_body())

    def __getstate__(self) -> Any:
        raise Lfm2SemanticMlpRefoldError(
            "Direct-local MLP source inventories are authenticated in-memory "
            "objects and cannot be serialized with Python pickle."
        )

    def __setstate__(self, state: Any) -> None:
        del state
        raise Lfm2SemanticMlpRefoldError(
            "Legacy pickled MLP source inventories are rejected fail-closed."
        )


@dataclass(frozen=True, slots=True)
class Lfm2SemanticMlpEndpointEvidence:
    """One exact cached parent phase plus replay-fresh child MLP endpoint."""

    batch_index: int
    trajectory_sha256: str
    phase_index: int
    phase_sha256: str
    layer_index: int
    alignment_sha256: str
    cache_key: str
    cache_manifest_sha256: str
    bridge_artifact_sha256: str
    captured_child_model_state_sha256: str
    source_inventory: Lfm2DirectLocalMlpSourceInventory = field(
        repr=False,
        compare=False,
    )
    parent_gate: Any = field(repr=False, compare=False)
    parent_up: Any = field(repr=False, compare=False)
    parent_post_swiglu: Any = field(repr=False, compare=False)
    parent_pre_mlp_residual: Any = field(repr=False, compare=False)
    parent_normalized_mlp_input: Any = field(repr=False, compare=False)
    actual_child_mlp_input: Any = field(repr=False, compare=False)
    actual_child_pre_mlp_residual: Any = field(repr=False, compare=False)
    parent_post_mlp_residual_target_space: Any = field(repr=False, compare=False)
    parent_final_norm_output_target_space: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    sample_weight: float = 1.0
    parent_exact_semantic_endpoint: bool = True
    inserted_target_row: bool = False
    normalization_reconstruction_sha256: str = field(init=False)
    normalization_reconstruction_within_legacy_envelope: bool = field(init=False)
    evidence_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        torch = _torch()
        if (
            isinstance(self.batch_index, bool)
            or not isinstance(self.batch_index, int)
            or self.batch_index < 0
            or isinstance(self.phase_index, bool)
            or not isinstance(self.phase_index, int)
            or self.phase_index < 0
            or isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
            or not isinstance(self.source_inventory, Lfm2DirectLocalMlpSourceInventory)
            or not self.source_inventory.verify()
            or self.source_inventory.layer_index != self.layer_index
            or self.parent_exact_semantic_endpoint is not True
            or self.inserted_target_row is not False
        ):
            raise Lfm2SemanticMlpRefoldError(
                "MLP evidence is not one parent-exact, noninserted semantic endpoint."
            )
        for value, name in (
            (self.trajectory_sha256, "trajectory"),
            (self.phase_sha256, "phase"),
            (self.alignment_sha256, "alignment"),
            (self.cache_key, "cache key"),
            (self.cache_manifest_sha256, "cache manifest"),
            (self.bridge_artifact_sha256, "bridge artifact"),
            (self.captured_child_model_state_sha256, "captured child state"),
        ):
            _digest(value, name)
        parent_gate = _finite_tensor(
            self.parent_gate,
            "parent gate endpoint",
            ndim=2,
            exact_dtype=torch.bfloat16,
        )
        parent_up = _finite_tensor(
            self.parent_up,
            "parent up endpoint",
            ndim=2,
            exact_dtype=torch.bfloat16,
        )
        parent_post = _finite_tensor(
            self.parent_post_swiglu,
            "parent post-SwiGLU endpoint",
            ndim=2,
            exact_dtype=torch.bfloat16,
        )
        parent_pre = _finite_tensor(
            self.parent_pre_mlp_residual,
            "parent pre-MLP residual",
            ndim=2,
            exact_dtype=torch.bfloat16,
        )
        parent_normalized = _finite_tensor(
            self.parent_normalized_mlp_input,
            "parent normalized MLP input",
            ndim=2,
            exact_dtype=torch.bfloat16,
        )
        child_dtype = getattr(self.actual_child_mlp_input, "dtype", None)
        if child_dtype not in {torch.bfloat16, torch.float32}:
            raise Lfm2SemanticMlpRefoldError(
                "Actual child MLP evidence must use BF16 or FP32 target coordinates."
            )
        child_input = _finite_tensor(
            self.actual_child_mlp_input,
            "actual child MLP input",
            ndim=2,
            exact_dtype=child_dtype,
        )
        child_pre = _finite_tensor(
            self.actual_child_pre_mlp_residual,
            "actual child pre-MLP residual",
            ndim=2,
            exact_dtype=child_dtype,
        )
        parent_endpoint = _finite_tensor(
            self.parent_post_mlp_residual_target_space,
            "mapped parent post-MLP residual",
            ndim=2,
            exact_dtype=child_dtype,
        )
        terminal = (
            None
            if self.parent_final_norm_output_target_space is None
            else _finite_tensor(
                self.parent_final_norm_output_target_space,
                "mapped parent final-norm output",
                ndim=2,
                exact_dtype=child_dtype,
            )
        )
        source_intermediate = self.source_inventory.source_intermediate_size
        rows, target_hidden = map(int, child_input.shape)
        if (
            rows != 1
            or tuple(parent_gate.shape) != (1, source_intermediate)
            or tuple(parent_up.shape) != tuple(parent_gate.shape)
            or tuple(parent_post.shape) != tuple(parent_gate.shape)
            or tuple(parent_pre.shape) != (1, self.source_inventory.source_hidden_size)
            or tuple(parent_normalized.shape)
            != (1, self.source_inventory.source_hidden_size)
            or tuple(child_pre.shape) != (1, target_hidden)
            or tuple(parent_endpoint.shape) != tuple(child_pre.shape)
            or self.source_inventory.target_hidden_size != target_hidden
            or (
                terminal is not None and tuple(terminal.shape) != tuple(child_pre.shape)
            )
            or not isinstance(self.sample_weight, (int, float))
            or isinstance(self.sample_weight, bool)
            or not math.isfinite(float(self.sample_weight))
            or float(self.sample_weight) <= 0.0
        ):
            raise Lfm2SemanticMlpRefoldError(
                "Semantic MLP evidence must contain exactly one compatible endpoint row."
            )
        reconstructed_normalized = _reconstruct_parent_rmsnorm_bf16(
            parent_pre,
            norm_weight=self.source_inventory.ffn_norm_weight,
            norm_epsilon=self.source_inventory.norm_epsilon,
        )
        reconstruction_metrics = _bf16_reconstruction_metrics(
            reconstructed_normalized,
            parent_normalized,
        )
        object.__setattr__(self, "parent_gate", parent_gate)
        object.__setattr__(self, "parent_up", parent_up)
        object.__setattr__(self, "parent_post_swiglu", parent_post)
        object.__setattr__(self, "parent_pre_mlp_residual", parent_pre)
        object.__setattr__(
            self,
            "parent_normalized_mlp_input",
            parent_normalized,
        )
        object.__setattr__(self, "actual_child_mlp_input", child_input)
        object.__setattr__(self, "actual_child_pre_mlp_residual", child_pre)
        object.__setattr__(
            self,
            "parent_post_mlp_residual_target_space",
            parent_endpoint,
        )
        object.__setattr__(self, "parent_final_norm_output_target_space", terminal)
        object.__setattr__(self, "sample_weight", float(self.sample_weight))
        object.__setattr__(
            self,
            "normalization_reconstruction_sha256",
            canonical_sha256(
                {
                    **reconstruction_metrics,
                    "cached_sha256": _tensor_sha256(parent_normalized),
                    "reconstructed_sha256": _tensor_sha256(reconstructed_normalized),
                }
            ),
        )
        object.__setattr__(
            self,
            "normalization_reconstruction_within_legacy_envelope",
            bool(reconstruction_metrics["within_legacy_bf16_envelope"]),
        )
        object.__setattr__(
            self,
            "evidence_sha256",
            canonical_sha256(self.authentication_body()),
        )

    @property
    def target_hidden_size(self) -> int:
        return int(self.actual_child_mlp_input.shape[1])

    def authentication_body(self) -> dict[str, Any]:
        tensors = {
            "parent_gate": self.parent_gate,
            "parent_up": self.parent_up,
            "parent_post_swiglu": self.parent_post_swiglu,
            "parent_pre_mlp_residual": self.parent_pre_mlp_residual,
            "parent_normalized_mlp_input": self.parent_normalized_mlp_input,
            "actual_child_mlp_input": self.actual_child_mlp_input,
            "actual_child_pre_mlp_residual": self.actual_child_pre_mlp_residual,
            "parent_post_mlp_residual_target_space": (
                self.parent_post_mlp_residual_target_space
            ),
        }
        return {
            "schema": LFM2_SEMANTIC_MLP_EVIDENCE_SCHEMA,
            "batch_index": self.batch_index,
            "trajectory_sha256": self.trajectory_sha256,
            "phase_index": self.phase_index,
            "phase_sha256": self.phase_sha256,
            "layer_index": self.layer_index,
            "alignment_sha256": self.alignment_sha256,
            "cache_key": self.cache_key,
            "cache_manifest_sha256": self.cache_manifest_sha256,
            "bridge_artifact_sha256": self.bridge_artifact_sha256,
            "captured_child_model_state_sha256": (
                self.captured_child_model_state_sha256
            ),
            "source_inventory_sha256": self.source_inventory.inventory_sha256,
            "target_coordinate_dtype": str(
                self.actual_child_mlp_input.dtype
            ).removeprefix("torch."),
            "compiler_lineage_sha256": (self.source_inventory.compiler_lineage_sha256),
            "normalization_lineage_sha256": (
                self.source_inventory.normalization_lineage_sha256
            ),
            "tensor_sha256": {
                name: _tensor_sha256(value) for name, value in sorted(tensors.items())
            },
            "parent_final_norm_output_target_space_sha256": (
                None
                if self.parent_final_norm_output_target_space is None
                else _tensor_sha256(self.parent_final_norm_output_target_space)
            ),
            "sample_weight": self.sample_weight,
            "parent_exact_semantic_endpoint": True,
            "inserted_target_row": False,
            "complete_causal_cache_endpoint": True,
            "parent_normalized_mlp_input_is_exact_cached_mlp_input": True,
            "parent_normalized_mlp_input_reconstruction_independently_replayed": (True),
            "normalization_reconstruction_within_legacy_bf16_envelope": (
                self.normalization_reconstruction_within_legacy_envelope
            ),
            "normalization_reconstruction_diagnostic_only": True,
            "normalization_reconstruction_numeric_admission_authority": False,
            "normalization_reconstruction_sha256": (
                self.normalization_reconstruction_sha256
            ),
            "parent_model_forward_calls_for_normalized_mlp_input": 0,
            "actual_child_replay_freshness_bound": True,
        }

    def verify(self) -> bool:
        return self.evidence_sha256 == canonical_sha256(self.authentication_body())

    def __getstate__(self) -> Any:
        raise Lfm2SemanticMlpRefoldError(
            "Semantic MLP evidence is authenticated in memory and cannot be "
            "serialized with Python pickle."
        )

    def __setstate__(self, state: Any) -> None:
        del state
        raise Lfm2SemanticMlpRefoldError(
            "Legacy pickled semantic MLP evidence is rejected fail-closed."
        )


@dataclass(frozen=True, slots=True)
class Lfm2SemanticMlpEndpointMetrics:
    endpoint_rows: int
    endpoint_rmse: float
    endpoint_relative_rmse: float
    endpoint_maximum_absolute_error: float
    endpoint_target_maximum_absolute_value: float
    mean_endpoint_cosine: float
    minimum_endpoint_cosine: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "endpoint_rows": self.endpoint_rows,
            "endpoint_rmse": self.endpoint_rmse,
            "endpoint_relative_rmse": self.endpoint_relative_rmse,
            "endpoint_maximum_absolute_error": (self.endpoint_maximum_absolute_error),
            "endpoint_target_maximum_absolute_value": (
                self.endpoint_target_maximum_absolute_value
            ),
            "mean_endpoint_cosine": self.mean_endpoint_cosine,
            "minimum_endpoint_cosine": self.minimum_endpoint_cosine,
        }


CandidateRole = Literal["incumbent", "incumbent-producer-w2", "full-w1-w3-w2"]


@dataclass(frozen=True, slots=True)
class Lfm2FrozenSemanticMlpCandidate:
    """One immutable dense target-shaped candidate and its split metrics."""

    role: CandidateRole
    gate_weight: Any = field(repr=False, compare=False)
    up_weight: Any = field(repr=False, compare=False)
    down_weight: Any = field(repr=False, compare=False)
    metrics: Mapping[str, Lfm2SemanticMlpEndpointMetrics]
    solver_receipt: Mapping[str, Any]
    candidate_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.role not in {
            "incumbent",
            "incumbent-producer-w2",
            "full-w1-w3-w2",
        }:
            raise Lfm2SemanticMlpRefoldError("Unknown semantic MLP candidate role.")
        gate = _finite_tensor(self.gate_weight, "candidate W1", ndim=2)
        up = _finite_tensor(self.up_weight, "candidate W3", ndim=2)
        down = _finite_tensor(self.down_weight, "candidate W2", ndim=2)
        intermediate, hidden = map(int, gate.shape)
        if (
            tuple(up.shape) != tuple(gate.shape)
            or tuple(down.shape) != (hidden, intermediate)
            or set(self.metrics) != set(_PARTITIONS)
            or not all(
                isinstance(value, Lfm2SemanticMlpEndpointMetrics)
                for value in self.metrics.values()
            )
        ):
            raise Lfm2SemanticMlpRefoldError(
                "Frozen candidate state or partition metrics are incomplete."
            )
        object.__setattr__(self, "gate_weight", gate)
        object.__setattr__(self, "up_weight", up)
        object.__setattr__(self, "down_weight", down)
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(
            self,
            "solver_receipt",
            MappingProxyType(dict(self.solver_receipt)),
        )
        object.__setattr__(
            self,
            "candidate_sha256",
            canonical_sha256(self.authentication_body()),
        )

    def state_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "w1.weight": self.gate_weight.detach().cpu().clone(),
                "w3.weight": self.up_weight.detach().cpu().clone(),
                "w2.weight": self.down_weight.detach().cpu().clone(),
            }
        )

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_SEMANTIC_MLP_CANDIDATE_SCHEMA,
            "role": self.role,
            "state_sha256": _state_sha256(self.state_dict()),
            "metrics": {name: self.metrics[name].as_dict() for name in _PARTITIONS},
            "solver_receipt": dict(self.solver_receipt),
        }

    def verify(self) -> bool:
        return self.candidate_sha256 == canonical_sha256(self.authentication_body())


@dataclass(frozen=True, slots=True)
class Lfm2PreparedSemanticMlpRefold:
    """Frozen competing candidates, split gates, and optional terminal banks."""

    layer_index: int
    captured_child_model_state_sha256: str
    target_topology_sha256: str
    target_parameter_count: int
    source_inventory_sha256: str
    compiler_lineage_sha256: str
    cache_key: str
    cache_manifest_sha256: str
    bridge_artifact_sha256: str
    evidence_sha256: tuple[str, ...]
    partition_batch_indices: Mapping[str, tuple[int, ...]]
    incumbent: Lfm2FrozenSemanticMlpCandidate
    writer_candidate: Lfm2FrozenSemanticMlpCandidate
    full_candidate: Lfm2FrozenSemanticMlpCandidate
    body_recipe: Lfm2SemanticBodyRecipe
    selected_role: CandidateRole
    selection_receipt: Mapping[str, Any]
    terminal_action_evidence: Mapping[str, Lfm2CpcfTerminalActionEvidence]
    preparation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        recipe_binding_valid = bool(
            isinstance(self.body_recipe, Lfm2SemanticBodyRecipe)
            and self.body_recipe.verify()
            and isinstance(self.selection_receipt, Mapping)
            and self.selection_receipt.get("body_recipe") == self.body_recipe.as_dict()
            and self.selection_receipt.get("body_recipe_sha256")
            == self.body_recipe.recipe_sha256
            and self.selection_receipt.get("body_recipe_id")
            == self.body_recipe.recipe_id
            and self.selection_receipt.get("recipe_component_lane")
            == self.body_recipe.component_lanes["mlp"]
        )
        selected_role_binding_valid = bool(
            isinstance(self.selection_receipt, Mapping)
            and self.selection_receipt.get("selected_role") == self.selected_role
            and self.selection_receipt.get("prevalidation_selected_role")
            == self.selected_role
            and self.selection_receipt.get("validation_rejected_role") is None
        )
        candidate_reports = (
            self.selection_receipt.get("candidate_reports")
            if isinstance(self.selection_receipt, Mapping)
            else None
        )
        candidate_binding_valid = bool(
            self.incumbent.role == "incumbent"
            and self.writer_candidate.role == "incumbent-producer-w2"
            and self.full_candidate.role == "full-w1-w3-w2"
            and isinstance(candidate_reports, Mapping)
            and all(
                isinstance(candidate_reports.get(candidate.role), Mapping)
                and candidate_reports[candidate.role].get("candidate_sha256")
                == candidate.candidate_sha256
                for candidate in (
                    self.incumbent,
                    self.writer_candidate,
                    self.full_candidate,
                )
            )
        )
        for value, name in (
            (self.captured_child_model_state_sha256, "captured child state"),
            (self.target_topology_sha256, "target topology"),
            (self.source_inventory_sha256, "source inventory"),
            (self.compiler_lineage_sha256, "compiler lineage"),
            (self.cache_key, "cache key"),
            (self.cache_manifest_sha256, "cache manifest"),
            (self.bridge_artifact_sha256, "bridge artifact"),
        ):
            _digest(value, name)
        if (
            self.selected_role
            not in {"incumbent", "incumbent-producer-w2", "full-w1-w3-w2"}
            or not all(
                candidate.verify()
                for candidate in (
                    self.incumbent,
                    self.writer_candidate,
                    self.full_candidate,
                )
            )
            or self.selected_role
            not in {
                self.incumbent.role,
                self.writer_candidate.role,
                self.full_candidate.role,
            }
            or set(self.partition_batch_indices) != set(_PARTITIONS)
            or any(not self.partition_batch_indices[name] for name in _PARTITIONS)
            or any(
                not _digest(value, "endpoint evidence")
                for value in self.evidence_sha256
            )
            or set(self.terminal_action_evidence).difference(_PARTITIONS)
            or not recipe_binding_valid
            or not selected_role_binding_valid
            or not candidate_binding_valid
        ):
            raise Lfm2SemanticMlpRefoldError(
                "Prepared semantic MLP candidate set or recipe binding is incomplete."
            )
        object.__setattr__(
            self,
            "partition_batch_indices",
            MappingProxyType(
                {
                    name: tuple(self.partition_batch_indices[name])
                    for name in _PARTITIONS
                }
            ),
        )
        object.__setattr__(
            self,
            "selection_receipt",
            MappingProxyType(dict(self.selection_receipt)),
        )
        object.__setattr__(
            self,
            "terminal_action_evidence",
            MappingProxyType(dict(self.terminal_action_evidence)),
        )
        object.__setattr__(
            self,
            "preparation_sha256",
            canonical_sha256(self.authentication_body()),
        )

    @property
    def selected_candidate(self) -> Lfm2FrozenSemanticMlpCandidate:
        if self.selected_role == self.incumbent.role:
            return self.incumbent
        if self.selected_role == self.writer_candidate.role:
            return self.writer_candidate
        return self.full_candidate

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_SEMANTIC_MLP_PREPARATION_SCHEMA,
            "layer_index": self.layer_index,
            "captured_child_model_state_sha256": (
                self.captured_child_model_state_sha256
            ),
            "target_topology_sha256": self.target_topology_sha256,
            "target_parameter_count": self.target_parameter_count,
            "source_inventory_sha256": self.source_inventory_sha256,
            "compiler_lineage_sha256": self.compiler_lineage_sha256,
            "cache_key": self.cache_key,
            "cache_manifest_sha256": self.cache_manifest_sha256,
            "bridge_artifact_sha256": self.bridge_artifact_sha256,
            "evidence_sha256": list(self.evidence_sha256),
            "partition_batch_indices": {
                name: list(self.partition_batch_indices[name]) for name in _PARTITIONS
            },
            "candidate_sha256": {
                "incumbent": self.incumbent.candidate_sha256,
                "incumbent-producer-w2": self.writer_candidate.candidate_sha256,
                "full-w1-w3-w2": self.full_candidate.candidate_sha256,
            },
            "body_recipe": self.body_recipe.as_dict(),
            "body_recipe_sha256": self.body_recipe.recipe_sha256,
            "selected_role": self.selected_role,
            "selection_receipt": dict(self.selection_receipt),
            "terminal_action_evidence_sha256": {
                name: value.authentication_sha256
                for name, value in sorted(self.terminal_action_evidence.items())
            },
            "candidate_frozen_before_all_heldout_gates": True,
            "validation_triggered_refit": False,
            "operation_audit": dict(_ZERO_OPERATIONS),
        }

    def verify(self) -> bool:
        return self.preparation_sha256 == canonical_sha256(self.authentication_body())


@dataclass(frozen=True, slots=True)
class Lfm2SemanticMlpPostfoldResult:
    """Installation receipt plus final-layer evidence for the shared W2 solve."""

    receipt: Mapping[str, Any]
    prepared: Lfm2PreparedSemanticMlpRefold
    final_layer_mlp_evidence: Mapping[str, Lfm2CpcfTerminalActionEvidence]
    deferred_selected_w2_prior: Lfm2CpcfTerminalActionDownWeightPrior | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.prepared.verify():
            raise Lfm2SemanticMlpRefoldError("Prepared MLP result lost authentication.")
        object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))
        object.__setattr__(
            self,
            "final_layer_mlp_evidence",
            MappingProxyType(dict(self.final_layer_mlp_evidence)),
        )
        qdeim_deferred = bool(
            self.final_layer_mlp_evidence
            and self.prepared.body_recipe.component_lanes["mlp"]
            in _QDEIM_MLP_LANE_TO_CANDIDATE
        )
        prior = self.deferred_selected_w2_prior
        if qdeim_deferred:
            selected = self.prepared.selected_candidate.down_weight
            if (
                not isinstance(prior, Lfm2CpcfTerminalActionDownWeightPrior)
                or not prior.verify_authentication()
                or not _torch().equal(prior.down_weight, selected)
                or self.receipt.get("deferred_selected_w2_prior_sha256")
                != _tensor_sha256(prior.down_weight)
                or self.receipt.get("deferred_selected_w2_prior_candidate_sha256")
                != self.prepared.selected_candidate.candidate_sha256
                or self.receipt.get("deferred_selected_w2_prior_carried_to_terminal")
                is not True
                or prior.body_recipe_id != self.prepared.body_recipe.recipe_id
                or prior.body_recipe_sha256 != self.prepared.body_recipe.recipe_sha256
                or prior.mlp_lane != self.prepared.body_recipe.component_lanes["mlp"]
                or prior.selected_role != self.prepared.selected_role
                or prior.selected_candidate_sha256
                != self.prepared.selected_candidate.candidate_sha256
                or prior.preparation_sha256 != self.prepared.preparation_sha256
                or prior.mlp_installation_core_sha256
                != self.receipt.get(
                    "deferred_selected_w2_prior_mlp_installation_core_sha256"
                )
                or self.receipt.get("deferred_selected_w2_prior_authentication_sha256")
                != prior.authentication_sha256
            ):
                raise Lfm2SemanticMlpRefoldError(
                    "Deferred Q-DEIM W2 prior is not bound to the selected candidate."
                )
            object.__setattr__(
                self,
                "deferred_selected_w2_prior",
                prior,
            )
        elif prior is not None:
            raise Lfm2SemanticMlpRefoldError(
                "Only a deferred Q-DEIM final layer may carry a terminal W2 prior."
            )


def _mlp_module(target_model: Any, layer_index: int) -> Any:
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    if layer_index < 0 or layer_index >= len(layers):
        raise Lfm2SemanticMlpRefoldError(f"Target model has no layer {layer_index}.")
    mlp = getattr(layers[layer_index], "feed_forward", None)
    if mlp is None or any(
        getattr(mlp, name, None) is None for name in ("w1", "w3", "w2")
    ):
        raise Lfm2SemanticMlpRefoldError(
            f"Layer {layer_index} has no ordinary dense LFM2 SwiGLU."
        )
    return mlp


def _ffn_norm_weight(target_model: Any, layer_index: int) -> Any:
    """Return one authenticated live child FFN RMSNorm weight."""

    torch = _torch()
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    if layer_index < 0 or layer_index >= len(layers):
        raise Lfm2SemanticMlpRefoldError(f"Target model has no layer {layer_index}.")
    norm = getattr(layers[layer_index], "ffn_norm", None)
    weight = getattr(norm, "weight", None)
    name = f"model.layers.{layer_index}.ffn_norm.weight"
    named = dict(target_model.named_parameters())
    if (
        not isinstance(weight, torch.nn.Parameter)
        or named.get(name) is not weight
        or weight.ndim != 1
        or weight.numel() == 0
        or not torch.is_floating_point(weight)
        or weight.dtype not in {torch.bfloat16, torch.float32}
        or weight.layout != torch.strided
        or weight.device.type == "meta"
        or not bool(torch.isfinite(weight).all().item())
        or bool((weight == 0).any().item())
    ):
        raise Lfm2SemanticMlpRefoldError(
            f"Target FFN RMSNorm Parameter {name!r} is missing or invalid."
        )
    return weight.detach().to(device="cpu").contiguous().clone()


def _parameter_bundle(target_model: Any, layer_index: int) -> Mapping[str, Any]:
    torch = _torch()
    mlp = _mlp_module(target_model, layer_index)
    prefix = f"model.layers.{layer_index}.feed_forward."
    values = {
        prefix + name + ".weight": getattr(getattr(mlp, name), "weight", None)
        for name in ("w1", "w3", "w2")
    }
    named = dict(target_model.named_parameters())
    dtypes = {value.dtype for value in values.values() if value is not None}
    if (
        any(not isinstance(value, torch.nn.Parameter) for value in values.values())
        or any(named.get(name) is not value for name, value in values.items())
        or len({id(value) for value in values.values()}) != 3
        or any(
            not torch.is_floating_point(value)
            or value.layout != torch.strided
            or value.device.type == "meta"
            for value in values.values()
        )
        or len(dtypes) != 1
        or not dtypes.issubset({torch.bfloat16, torch.float32})
    ):
        raise Lfm2SemanticMlpRefoldError(
            "Target W1/W3/W2 Parameters must be independent dense Parameters that "
            "share one BF16 or FP32 dtype."
        )
    return MappingProxyType(values)


def _suffix_state(parameters: Mapping[str, Any], layer_index: int) -> Mapping[str, Any]:
    prefix = f"model.layers.{layer_index}.feed_forward."
    return MappingProxyType(
        {
            name.removeprefix(prefix): value.detach()
            .to(device="cpu")
            .contiguous()
            .clone()
            for name, value in parameters.items()
        }
    )


def _split_evidence(
    evidence: Sequence[Lfm2SemanticMlpEndpointEvidence],
) -> Mapping[str, tuple[Lfm2SemanticMlpEndpointEvidence, ...]]:
    grouped: dict[int, list[Lfm2SemanticMlpEndpointEvidence]] = {}
    for item in evidence:
        grouped.setdefault(item.batch_index, []).append(item)
    batches = tuple(sorted(grouped))
    if len(batches) < 4:
        raise Lfm2SemanticMlpRefoldError(
            "At least four unique batches are required for isolated MLP gates."
        )
    for batch_index, values in grouped.items():
        if (
            len({value.trajectory_sha256 for value in values}) != 1
            or len({value.phase_index for value in values}) != len(values)
            or len({value.phase_sha256 for value in values}) != len(values)
        ):
            raise Lfm2SemanticMlpRefoldError(
                f"Batch {batch_index} is not one whole trajectory of unique phases."
            )
    count = len(batches)
    counts = [max(1, count // 2), max(1, (3 * count) // 16), max(1, count // 8), 1]
    while sum(counts) > count:
        for index in (0, 1, 2):
            if counts[index] > 1 and sum(counts) > count:
                counts[index] -= 1
    counts[3] += count - sum(counts)
    cursor = 0
    result: dict[str, tuple[Lfm2SemanticMlpEndpointEvidence, ...]] = {}
    for partition, size in zip(_PARTITIONS, counts, strict=True):
        selected = batches[cursor : cursor + size]
        result[partition] = tuple(
            item
            for batch_index in selected
            for item in sorted(
                grouped[batch_index], key=lambda value: value.phase_index
            )
        )
        cursor += size
    if cursor != count or any(not result[name] for name in _PARTITIONS):
        raise AssertionError("Internal semantic MLP split construction failed.")
    return MappingProxyType(result)


def _prediction(state: Mapping[str, Any], item: Lfm2SemanticMlpEndpointEvidence) -> Any:
    torch = _torch()
    inputs = item.actual_child_mlp_input.to(torch.float64)
    gate = state["w1.weight"].to(torch.float64)
    up = state["w3.weight"].to(torch.float64)
    down = state["w2.weight"].to(torch.float64)
    post = torch.nn.functional.silu(inputs @ gate.T) * (inputs @ up.T)
    return item.actual_child_pre_mlp_residual.to(torch.float64) + post @ down.T


def _metrics(
    state: Mapping[str, Any],
    evidence: Sequence[Lfm2SemanticMlpEndpointEvidence],
) -> Lfm2SemanticMlpEndpointMetrics:
    torch = _torch()
    predictions = torch.cat(tuple(_prediction(state, item) for item in evidence), dim=0)
    targets = torch.cat(
        tuple(
            item.parent_post_mlp_residual_target_space.to(torch.float64)
            for item in evidence
        ),
        dim=0,
    )
    weights = torch.tensor(
        [item.sample_weight for item in evidence],
        dtype=torch.float64,
    )
    weights = weights / weights.sum()
    row_mse = (predictions - targets).square().mean(dim=1)
    rmse = math.sqrt(float((row_mse * weights).sum().item()))
    target_rms = math.sqrt(float((targets.square().mean(dim=1) * weights).sum().item()))
    dot = (predictions * targets).sum(dim=1)
    denominator = torch.linalg.vector_norm(
        predictions, dim=1
    ) * torch.linalg.vector_norm(
        targets,
        dim=1,
    )
    cosine = torch.where(
        denominator > 0.0,
        dot / denominator.clamp_min(1e-300),
        torch.where(
            torch.linalg.vector_norm(predictions - targets, dim=1) == 0.0,
            torch.ones_like(denominator),
            torch.zeros_like(denominator),
        ),
    ).clamp(-1.0, 1.0)
    return Lfm2SemanticMlpEndpointMetrics(
        endpoint_rows=len(evidence),
        endpoint_rmse=rmse,
        endpoint_relative_rmse=rmse / max(target_rms, 1e-300),
        endpoint_maximum_absolute_error=float(
            (predictions - targets).abs().max().item()
        ),
        endpoint_target_maximum_absolute_value=float(targets.abs().max().item()),
        mean_endpoint_cosine=float((cosine * weights).sum().item()),
        minimum_endpoint_cosine=float(cosine.min().item()),
    )


def _all_metrics(
    state: Mapping[str, Any],
    partitions: Mapping[str, Sequence[Lfm2SemanticMlpEndpointEvidence]],
) -> Mapping[str, Lfm2SemanticMlpEndpointMetrics]:
    return MappingProxyType(
        {name: _metrics(state, partitions[name]) for name in _PARTITIONS}
    )


def _diagnostic_all_metrics(
    state: Mapping[str, Any],
    partitions: Mapping[str, Sequence[Lfm2SemanticMlpEndpointEvidence]],
) -> tuple[
    Mapping[str, Lfm2SemanticMlpEndpointMetrics],
    Mapping[str, str | None],
]:
    """Evaluate diagnostics fail-soft so metrics cannot become a safety veto."""

    result: dict[str, Lfm2SemanticMlpEndpointMetrics] = {}
    errors: dict[str, str | None] = {}
    for name in _PARTITIONS:
        try:
            metrics = _metrics(state, partitions[name])
            values = (
                metrics.endpoint_rmse,
                metrics.endpoint_relative_rmse,
                metrics.endpoint_maximum_absolute_error,
                metrics.endpoint_target_maximum_absolute_value,
                metrics.mean_endpoint_cosine,
                metrics.minimum_endpoint_cosine,
            )
            if not all(math.isfinite(value) for value in values):
                raise ArithmeticError("nonfinite-derived-endpoint-metrics")
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            metrics = Lfm2SemanticMlpEndpointMetrics(
                endpoint_rows=len(partitions[name]),
                endpoint_rmse=_MAX_FINITE_FLOAT,
                endpoint_relative_rmse=_MAX_FINITE_FLOAT,
                endpoint_maximum_absolute_error=_MAX_FINITE_FLOAT,
                endpoint_target_maximum_absolute_value=_MAX_FINITE_FLOAT,
                mean_endpoint_cosine=-1.0,
                minimum_endpoint_cosine=-1.0,
            )
            errors[name] = f"{type(exc).__name__}: {exc}"
        else:
            errors[name] = None
        result[name] = metrics
    return MappingProxyType(result), MappingProxyType(errors)


def _nonregressing(
    candidate: Lfm2SemanticMlpEndpointMetrics,
    baseline: Lfm2SemanticMlpEndpointMetrics,
    *,
    rmse_relative_tolerance: float,
    cosine_tolerance: float,
) -> bool:
    rmse_limit = baseline.endpoint_rmse + max(
        1e-12,
        rmse_relative_tolerance * max(baseline.endpoint_rmse, 1e-12),
    )
    maximum_absolute_error_limit = baseline.endpoint_maximum_absolute_error + max(
        1e-12,
        rmse_relative_tolerance
        * max(baseline.endpoint_target_maximum_absolute_value, 1e-12),
    )
    return (
        candidate.endpoint_rmse <= rmse_limit
        and candidate.endpoint_maximum_absolute_error <= maximum_absolute_error_limit
        and candidate.mean_endpoint_cosine + cosine_tolerance
        >= baseline.mean_endpoint_cosine
        and candidate.minimum_endpoint_cosine + cosine_tolerance
        >= baseline.minimum_endpoint_cosine
    )


def _strictly_improving(
    candidate: Lfm2SemanticMlpEndpointMetrics,
    baseline: Lfm2SemanticMlpEndpointMetrics,
    *,
    rmse_relative_tolerance: float,
    cosine_tolerance: float,
) -> bool:
    rmse_margin = max(
        1e-12,
        rmse_relative_tolerance * max(baseline.endpoint_rmse, 1e-12),
    )
    maximum_absolute_error_margin = max(
        1e-12,
        rmse_relative_tolerance
        * max(baseline.endpoint_target_maximum_absolute_value, 1e-12),
    )
    return _nonregressing(
        candidate,
        baseline,
        rmse_relative_tolerance=rmse_relative_tolerance,
        cosine_tolerance=cosine_tolerance,
    ) and (
        candidate.endpoint_rmse < baseline.endpoint_rmse - rmse_margin
        or candidate.endpoint_maximum_absolute_error
        < baseline.endpoint_maximum_absolute_error - maximum_absolute_error_margin
        or candidate.mean_endpoint_cosine
        > baseline.mean_endpoint_cosine + cosine_tolerance
        or candidate.minimum_endpoint_cosine
        > baseline.minimum_endpoint_cosine + cosine_tolerance
    )


def _solve_writer(
    fit: Sequence[Lfm2SemanticMlpEndpointEvidence],
    incumbent: Mapping[str, Any],
    *,
    ridge: float,
    maximum_delta_ratio: float | None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    torch = _torch()
    gate = incumbent["w1.weight"].to(torch.float64)
    up = incumbent["w3.weight"].to(torch.float64)
    prior = incumbent["w2.weight"].to(torch.float64)
    inputs = torch.cat(
        tuple(item.actual_child_mlp_input.to(torch.float64) for item in fit),
        dim=0,
    )
    features = torch.nn.functional.silu(inputs @ gate.T) * (inputs @ up.T)
    target = torch.cat(
        tuple(
            (
                item.parent_post_mlp_residual_target_space
                - item.actual_child_pre_mlp_residual
            ).to(torch.float64)
            for item in fit
        ),
        dim=0,
    )
    weights = (
        torch.tensor(
            [item.sample_weight for item in fit],
            dtype=torch.float64,
        )
        .sqrt()
        .unsqueeze(1)
    )
    weighted_features = features * weights
    residual = (target - features @ prior.T) * weights
    gram = weighted_features @ weighted_features.T
    scale = float(gram.diagonal().mean().clamp_min(1e-30).item())
    regularizer = ridge * scale
    system = gram + regularizer * torch.eye(int(gram.shape[0]), dtype=torch.float64)
    solved = torch.linalg.solve(system, residual)
    delta = (weighted_features.T @ solved).T
    unclipped_ratio = float(
        (
            torch.linalg.vector_norm(delta)
            / torch.linalg.vector_norm(prior).clamp_min(1e-30)
        ).item()
    )
    candidate_down = (prior + delta).to(incumbent["w2.weight"].dtype)
    state = MappingProxyType(
        {
            "w1.weight": incumbent["w1.weight"].detach().cpu().clone(),
            "w3.weight": incumbent["w3.weight"].detach().cpu().clone(),
            "w2.weight": candidate_down.detach().cpu().contiguous().clone(),
        }
    )
    receipt = {
        "solver": "fit-only-weighted-dual-ridge-with-incumbent-w2-prior",
        "producer": "actual-child-incumbent-w1-w3-post-swiglu",
        "consumer": "dense-existing-w2-residual-writer",
        "fit_rows": len(fit),
        "ridge": ridge,
        "effective_ridge": regularizer,
        "unclipped_delta_ratio": unclipped_ratio,
        "observed_solver_delta_ratio": unclipped_ratio,
        "maximum_delta_ratio": maximum_delta_ratio,
        "requested_maximum_delta_ratio_goal": maximum_delta_ratio,
        "requested_delta_ratio_goal_met": (
            None
            if maximum_delta_ratio is None
            else unclipped_ratio <= maximum_delta_ratio
        ),
        "effective_maximum_delta_ratio_cap": None,
        "delta_ratio_cap_applied": False,
        "relative_incumbent_trust_cap_applied": False,
        "applied_delta_scale": 1.0,
        "numeric_delta_ratio_used_for_candidate_eligibility": False,
        "numeric_delta_ratio_used_for_candidate_ordering": False,
        "numeric_delta_ratio_used_for_publication_or_rollback": False,
        "numeric_delta_ratio_has_final_intelligence_authority": False,
        "analytic_linear_solve": True,
        "cluster_solve": False,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    return state, MappingProxyType(receipt)


def _qdeim_child_coordinate_rms(
    fit: Sequence[Lfm2SemanticMlpEndpointEvidence],
) -> tuple[Any, Mapping[str, Any]]:
    """Return authenticated weighted FIT RMS with a structural positive floor."""

    torch = _torch()
    child_inputs = torch.cat(
        tuple(item.actual_child_mlp_input.to(torch.float64) for item in fit),
        dim=0,
    ).contiguous()
    sample_weights = torch.tensor(
        [item.sample_weight for item in fit],
        dtype=torch.float64,
    )
    structural_floor = math.sqrt(torch.finfo(torch.float64).eps)
    with torch.no_grad():
        weighted_second_moment = (
            sample_weights.unsqueeze(1) * child_inputs.square()
        ).sum(dim=0) / sample_weights.sum()
        empirical_rms = torch.sqrt(weighted_second_moment).contiguous()
        coordinate_rms = empirical_rms.clamp_min(structural_floor).contiguous()
    if (
        coordinate_rms.ndim != 1
        or not bool(torch.isfinite(coordinate_rms).all().item())
        or not bool((coordinate_rms > 0.0).all().item())
    ):
        raise Lfm2SemanticMlpRefoldError(
            "Weighted FIT child-coordinate RMS is not finite and strictly positive."
        )
    body = {
        "schema": "neural-forge-lfm2-qdeim-child-coordinate-rms-v1",
        "method": "sqrt_weighted_fit_second_moment_with_structural_floor",
        "fit_child_inputs_sha256": lfm2_authenticated_tensor_sha256(child_inputs),
        "fit_sample_weights_sha256": lfm2_authenticated_tensor_sha256(sample_weights),
        "empirical_rms_sha256": lfm2_authenticated_tensor_sha256(empirical_rms),
        "coordinate_rms_sha256": lfm2_authenticated_tensor_sha256(coordinate_rms),
        "tangent_tensor_coordinate_rms_sha256": (
            fingerprint_lfm2_swiglu_tangent_tensor(coordinate_rms)
        ),
        "fit_row_count": int(child_inputs.shape[0]),
        "child_width": int(child_inputs.shape[1]),
        "sample_weight_sum": float(sample_weights.sum().item()),
        "structural_floor": structural_floor,
        "structural_floor_policy": "sqrt_float64_machine_epsilon",
        "structural_floor_applied_coordinate_count": int(
            (empirical_rms < structural_floor).sum().item()
        ),
        "floor_prevents_zero_authenticated_tangent_directions": True,
        "numeric_values_used_for_semantic_acceptance_or_veto": False,
        "whole_decoded_responses_are_primary_semantic_evidence": True,
        "binary_manual_gate_created": False,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    return coordinate_rms, MappingProxyType(
        {**body, "receipt_sha256": canonical_sha256(body)}
    )


def _compile_qdeim_full_candidate(
    fit: Sequence[Lfm2SemanticMlpEndpointEvidence],
    inventory: Lfm2DirectLocalMlpSourceInventory,
    incumbent: Mapping[str, Any],
    *,
    normalized_control: Lfm2CompiledNormalizedInputDecoder,
    mlp_lane: str,
    ridge: float,
    fold_compute_device: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Compile the lane-selected Q-DEIM state from one sealed decoder control."""

    torch = _torch()
    candidate_id = _QDEIM_MLP_LANE_TO_CANDIDATE.get(mlp_lane)
    if candidate_id is None:
        raise Lfm2SemanticMlpRefoldError(
            f"MLP lane {mlp_lane!r} is not a nonlinear Q-DEIM lane."
        )
    fit_gate = torch.cat(
        tuple(item.parent_gate for item in fit),
        dim=0,
    ).contiguous()
    fit_up = torch.cat(
        tuple(item.parent_up for item in fit),
        dim=0,
    ).contiguous()
    fit_post = torch.cat(
        tuple(item.parent_post_swiglu for item in fit),
        dim=0,
    ).contiguous()
    fit_child_inputs = torch.cat(
        tuple(item.actual_child_mlp_input for item in fit),
        dim=0,
    ).contiguous()
    fit_weights = torch.tensor(
        [item.sample_weight for item in fit],
        dtype=torch.float64,
    )
    coordinate_rms, coordinate_rms_receipt = _qdeim_child_coordinate_rms(fit)
    tangent_bank = build_lfm2_swiglu_tangent_bank(
        site_id=(
            f"semantic-mlp-layer-{inventory.layer_index}-"
            "fixed-two-anchor-child-lifted-tangent"
        ),
        parent_w1=inventory.gate_weight,
        parent_w1_sha256=fingerprint_lfm2_swiglu_tangent_tensor(inventory.gate_weight),
        parent_w3=inventory.up_weight,
        parent_w3_sha256=fingerprint_lfm2_swiglu_tangent_tensor(inventory.up_weight),
        cached_gate_rows=fit_gate,
        cached_gate_rows_sha256=fingerprint_lfm2_swiglu_tangent_tensor(fit_gate),
        cached_up_rows=fit_up,
        cached_up_rows_sha256=fingerprint_lfm2_swiglu_tangent_tensor(fit_up),
        cached_post_rows=fit_post,
        cached_post_rows_sha256=fingerprint_lfm2_swiglu_tangent_tensor(fit_post),
        target_to_parent_decoder=normalized_control.normalized_input_decoder,
        target_to_parent_decoder_sha256=(
            fingerprint_lfm2_swiglu_tangent_tensor(
                normalized_control.normalized_input_decoder
            )
        ),
        sample_weights=fit_weights,
        sample_weights_sha256=fingerprint_lfm2_swiglu_tangent_tensor(fit_weights),
        child_input_coordinate_rms=coordinate_rms,
        child_input_coordinate_rms_sha256=(
            fingerprint_lfm2_swiglu_tangent_tensor(coordinate_rms)
        ),
        anchor_count=_QDEIM_TANGENT_ANCHOR_COUNT,
        compute_device=fold_compute_device,
    )
    if len(tangent_bank.anchor_indices) != _QDEIM_TANGENT_ANCHOR_COUNT:
        raise Lfm2SemanticMlpRefoldError(
            "Nonlinear Q-DEIM requires exactly two structurally selected FIT anchors."
        )
    block_kwargs = tangent_bank.qdeim_functional_snapshot_kwargs(
        block_id=(
            f"semantic-mlp-layer-{inventory.layer_index}-"
            "two-anchor-child-lifted-swiglu-tangent"
        )
    )
    snapshot_block = Lfm2MlpQdeimFunctionalSnapshotBlock(**block_kwargs)
    snapshot_blocks = (snapshot_block,)
    qdeim_cases = tuple(
        Lfm2MlpQdeimFitCase(
            target_normalized_input=item.actual_child_mlp_input,
            parent_post_swiglu=item.parent_post_swiglu,
            parent_post_mlp_residual_target_space=(
                item.parent_post_mlp_residual_target_space
            ),
            target_pre_mlp_residual=item.actual_child_pre_mlp_residual,
        )
        for item in fit
    )
    case_weights = tuple(item.sample_weight for item in fit)
    source_state_sha256 = lfm2_mlp_qdeim_source_state_sha256(
        inventory.gate_weight,
        inventory.up_weight,
        inventory.down_weight,
    )
    target_template_sha256 = lfm2_mlp_qdeim_target_template_sha256(
        incumbent["w1.weight"],
        incumbent["w3.weight"],
        incumbent["w2.weight"],
    )
    fit_evidence_sha256 = lfm2_mlp_qdeim_fit_evidence_sha256(
        qdeim_cases,
        case_weights,
    )
    snapshot_evidence_sha256 = lfm2_mlp_qdeim_functional_snapshot_evidence_sha256(
        snapshot_blocks
    )
    compiled = compile_lfm2_mlp_nonlinear_qdeim(
        source_gate_weight=inventory.gate_weight,
        source_up_weight=inventory.up_weight,
        source_down_weight=inventory.down_weight,
        expected_source_state_sha256=source_state_sha256,
        target_gate_weight_template=incumbent["w1.weight"],
        target_up_weight_template=incumbent["w3.weight"],
        target_down_weight_template=incumbent["w2.weight"],
        expected_target_template_sha256=target_template_sha256,
        normalized_input_decoder=normalized_control.normalized_input_decoder,
        expected_normalized_input_decoder_sha256=(
            normalized_control.normalized_input_decoder_sha256
        ),
        residual_output_chart=inventory.residual_chart,
        expected_residual_output_chart_sha256=lfm2_authenticated_tensor_sha256(
            inventory.residual_chart
        ),
        fit_cases=qdeim_cases,
        expected_fit_evidence_sha256=fit_evidence_sha256,
        case_sample_weights=case_weights,
        functional_snapshot_blocks=snapshot_blocks,
        expected_functional_snapshot_evidence_sha256=(snapshot_evidence_sha256),
        compute_fit_residual_correction=True,
        fit_correction_ridge=ridge,
        compute_device=fold_compute_device,
    )
    if candidate_id not in compiled.candidate_ids:
        raise Lfm2SemanticMlpRefoldError(
            "Q-DEIM compiler did not materialize the body-recipe candidate."
        )
    state = compiled.state_dict(candidate_id)
    control_receipt = normalized_control.as_dict()
    tangent_receipt = tangent_bank.as_dict()
    qdeim_receipt = compiled.as_dict()
    target_materialization_dtype = qdeim_receipt.get("target_materialization_dtype")
    integration_body = {
        "schema": "neural-forge-lfm2-semantic-mlp-qdeim-integration-v1",
        "selected_qdeim_candidate_id": candidate_id,
        "qdeim_candidate_ids": list(compiled.candidate_ids),
        "source_inventory_sha256": inventory.inventory_sha256,
        "source_state_sha256": source_state_sha256,
        "target_template_sha256": target_template_sha256,
        "target_materialization_dtype": target_materialization_dtype,
        "normalized_decoder_control_receipt_sha256": (
            normalized_control.receipt_sha256
        ),
        "normalized_input_decoder_sha256": (
            normalized_control.normalized_input_decoder_sha256
        ),
        "normalized_decoder_control_candidate_weights_used": False,
        "normalized_decoder_control_used_only_for_decoder_and_receipt": True,
        "normalized_decoder_control_medoid_or_cluster_plan_compiled": False,
        "normalized_decoder_control_legacy_residual_writer_compiled": False,
        "child_coordinate_rms_receipt_sha256": coordinate_rms_receipt["receipt_sha256"],
        "tangent_bank_sha256": tangent_bank.bank_sha256,
        "tangent_bank_compute_execution": dict(tangent_receipt["compute_execution"]),
        "tangent_decoder_sha256": tangent_bank.source_hashes[
            "target_to_parent_decoder"
        ],
        "qdeim_fit_evidence_sha256": fit_evidence_sha256,
        "qdeim_functional_snapshot_evidence_sha256": (snapshot_evidence_sha256),
        "qdeim_result_sha256": qdeim_receipt["result_sha256"],
        "qdeim_compute_execution": dict(qdeim_receipt["compute_execution"]),
        "fold_compute_device": fold_compute_device,
        "selected_state_sha256": _state_sha256(state),
        "selected_state_tensor_sha256": {
            name: lfm2_authenticated_tensor_sha256(value)
            for name, value in sorted(state.items())
        },
        "fit_gate_rows_sha256": lfm2_authenticated_tensor_sha256(fit_gate),
        "fit_up_rows_sha256": lfm2_authenticated_tensor_sha256(fit_up),
        "fit_post_swiglu_rows_sha256": lfm2_authenticated_tensor_sha256(fit_post),
        "fit_child_input_rows_sha256": lfm2_authenticated_tensor_sha256(
            fit_child_inputs
        ),
        "fixed_tangent_anchor_count": _QDEIM_TANGENT_ANCHOR_COUNT,
        "heldout_rows_used_for_pod_or_tangent": 0,
        "tangent_rows_used_for_endpoint_w2_correction": 0,
        "numeric_metrics_used_for_semantic_acceptance_or_veto": False,
        "whole_decoded_responses_are_primary_semantic_evidence": True,
        "binary_manual_gate_created": False,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    receipt = {
        **qdeim_receipt,
        "solver": "authenticated-two-anchor-tangent-nonlinear-qdeim",
        "selected_qdeim_candidate_id": candidate_id,
        "qdeim_compiler_receipt": qdeim_receipt,
        "normalized_decoder_control": control_receipt,
        "normalized_decoder_control_receipt_sha256": integration_body[
            "normalized_decoder_control_receipt_sha256"
        ],
        "child_coordinate_rms": dict(coordinate_rms_receipt),
        "tangent_bank": tangent_receipt,
        "tangent_bank_sha256": tangent_bank.bank_sha256,
        "qdeim_integration": integration_body,
        "qdeim_integration_sha256": canonical_sha256(integration_body),
        "fit_rows": len(fit),
        "heldout_rows_used_by_solver": 0,
        "heldout_rows_used_for_plan": 0,
        "analytic_linear_solve": True,
        "cluster_solve": False,
        "decoder_control_cluster_solve": False,
        "decoder_control_only_solve": True,
        "full_source_coverage": True,
        "full_source_assignment_fraction": 1.0,
        "source_neurons_dropped": 0,
        "all_source_writer_columns_consumed": True,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    return state, MappingProxyType(receipt)


def _authenticate_qdeim_full_solver(
    full_state: Mapping[str, Any],
    full_solver: Mapping[str, Any],
    inventory: Lfm2DirectLocalMlpSourceInventory,
    *,
    mlp_lane: str,
    requested_maximum_delta_ratio: float | None,
) -> Mapping[str, Any]:
    """Fail closed on structural/provenance errors, never semantic diagnostics."""

    torch = _torch()
    full_state_dtypes = {value.dtype for value in full_state.values()}
    target_dtype = (
        next(iter(full_state_dtypes)) if len(full_state_dtypes) == 1 else None
    )
    target_dtype_name = (
        None if target_dtype is None else str(target_dtype).removeprefix("torch.")
    )
    expected_candidate = _QDEIM_MLP_LANE_TO_CANDIDATE.get(mlp_lane)
    qdeim = full_solver.get("qdeim_compiler_receipt")
    control = full_solver.get("normalized_decoder_control")
    coordinate_rms = full_solver.get("child_coordinate_rms")
    tangent = full_solver.get("tangent_bank")
    integration = full_solver.get("qdeim_integration")
    if not all(
        isinstance(value, Mapping)
        for value in (qdeim, control, coordinate_rms, tangent, integration)
    ):
        raise Lfm2SemanticMlpRefoldError(
            "Q-DEIM full proposal is missing an authenticated compiler receipt."
        )
    assert isinstance(qdeim, Mapping)
    assert isinstance(control, Mapping)
    assert isinstance(coordinate_rms, Mapping)
    assert isinstance(tangent, Mapping)
    assert isinstance(integration, Mapping)
    qdeim_body = dict(qdeim)
    qdeim_result_sha256 = qdeim_body.pop("result_sha256", None)
    control_body = dict(control)
    control_receipt_sha256 = control_body.pop("receipt_sha256", None)
    rms_body = dict(coordinate_rms)
    rms_receipt_sha256 = rms_body.pop("receipt_sha256", None)
    operation_audit = full_solver.get("operation_audit")
    qdeim_compute = qdeim.get("compute_execution")
    qdeim_compute_body = (
        dict(qdeim_compute) if isinstance(qdeim_compute, Mapping) else {}
    )
    qdeim_compute_sha256 = qdeim_compute_body.pop("receipt_sha256", None)
    tangent_compute = tangent.get("compute_execution")
    tangent_compute_body = (
        dict(tangent_compute) if isinstance(tangent_compute, Mapping) else {}
    )
    tangent_compute_sha256 = tangent_compute_body.pop("receipt_sha256", None)
    requested_fold_compute_device = integration.get("fold_compute_device")
    candidate_report = (
        qdeim.get("materialized_candidates", {}).get(expected_candidate)
        if isinstance(qdeim.get("materialized_candidates"), Mapping)
        else None
    )
    if (
        expected_candidate is None
        or full_solver.get("selected_qdeim_candidate_id") != expected_candidate
        or integration.get("selected_qdeim_candidate_id") != expected_candidate
        or qdeim.get("schema") != "neural-forge-lfm2-mlp-nonlinear-qdeim-v1"
        or target_dtype not in {torch.bfloat16, torch.float32}
        or qdeim.get("target_materialization_dtype") != target_dtype_name
        or qdeim.get("dense_supported_target_dtype_state") is not True
        or qdeim.get("dense_bfloat16_target_state")
        is not (target_dtype == torch.bfloat16)
        or integration.get("target_materialization_dtype") != target_dtype_name
        or qdeim_result_sha256 != canonical_sha256(qdeim_body)
        or full_solver.get("result_sha256") != qdeim_result_sha256
        or tuple(qdeim.get("candidate_ids", ()))
        != (
            LFM2_MLP_QDEIM_STRUCTURAL_PRIOR_CANDIDATE,
            LFM2_MLP_QDEIM_FIT_CORRECTED_CANDIDATE,
        )
        or not isinstance(candidate_report, Mapping)
        or qdeim.get("source_state_immutable") is not True
        or qdeim.get("source_state_sha256_before")
        != qdeim.get("source_state_sha256_after")
        or qdeim.get("writer_transport", {}).get("all_source_writer_columns_consumed")
        is not True
        or qdeim.get("producer_transport", {}).get("producer_averaging") is not False
        or qdeim.get("selected_source_neuron_count")
        != qdeim.get("target_intermediate_size")
        or qdeim.get("automatic_candidate_selection_performed") is not False
        or qdeim.get("automatic_candidate_acceptance_performed") is not False
        or qdeim.get("numeric_diagnostics_are_advisory_only") is not True
        or qdeim.get("rank_completion", {}).get("random_completion_modes") != 0
        or qdeim.get("rank_completion", {}).get("silent_completion_performed")
        is not False
        or qdeim.get("functional_snapshot_blocks_enter_fit_w2_correction") is not False
        or not isinstance(qdeim_compute, Mapping)
        or qdeim_compute.get("schema") != "neural-forge-lfm2-mlp-qdeim-compute-v1"
        or qdeim_compute_sha256 != canonical_sha256(qdeim_compute_body)
        or not isinstance(requested_fold_compute_device, str)
        or not requested_fold_compute_device
        or qdeim_compute.get("requested_device") != requested_fold_compute_device
        or qdeim_compute.get("resolved_device") != requested_fold_compute_device
        or qdeim_compute.get("backend")
        != ("cpu" if requested_fold_compute_device == "cpu" else "cuda")
        or qdeim_compute.get("compute_dtype") != "float64"
        or qdeim_compute.get("authenticated_source_archive_device") != "cpu"
        or qdeim_compute.get("materialized_candidate_device") != "cpu"
        or qdeim_compute.get("silent_device_fallback_performed") is not False
        or integration.get("qdeim_compute_execution") != dict(qdeim_compute)
        or control.get("schema") != LFM2_NORMALIZED_INPUT_DECODER_SCHEMA
        or control_receipt_sha256 != canonical_sha256(control_body)
        or control.get("decoder_only_scope") is not True
        or control.get("cluster_or_medoid_state_constructed") is not False
        or control.get("operation_audit", {}).get("activation_cluster_plan_builds") != 0
        or control.get("operation_audit", {}).get("medoid_selection_operations") != 0
        or control.get("operation_audit", {}).get("producer_materializations") != 0
        or control.get("operation_audit", {}).get("consumer_materializations") != 0
        or full_solver.get("normalized_decoder_control_receipt_sha256")
        != control_receipt_sha256
        or integration.get("normalized_decoder_control_receipt_sha256")
        != full_solver.get("normalized_decoder_control_receipt_sha256")
        or integration.get("normalized_decoder_control_candidate_weights_used")
        is not False
        or integration.get(
            "normalized_decoder_control_used_only_for_decoder_and_receipt"
        )
        is not True
        or integration.get("normalized_decoder_control_medoid_or_cluster_plan_compiled")
        is not False
        or integration.get("normalized_decoder_control_legacy_residual_writer_compiled")
        is not False
        or rms_receipt_sha256 != canonical_sha256(rms_body)
        or coordinate_rms.get("structural_floor_policy")
        != "sqrt_float64_machine_epsilon"
        or coordinate_rms.get("floor_prevents_zero_authenticated_tangent_directions")
        is not True
        or coordinate_rms.get("numeric_values_used_for_semantic_acceptance_or_veto")
        is not False
        or tangent.get("bank_sha256") != full_solver.get("tangent_bank_sha256")
        or tangent.get("geometry", {}).get("anchor_count")
        != _QDEIM_TANGENT_ANCHOR_COUNT
        or tangent.get("anchor_selection_policy")
        != "fixed_evenly_spaced_canonical_cache_row_positions"
        or tangent.get("anchor_values_or_numeric_metrics_used_for_selection")
        is not False
        or tangent.get("sample_weights_used_for_anchor_selection") is not False
        or not isinstance(tangent_compute, Mapping)
        or tangent_compute.get("schema")
        != "neural-forge-lfm2-swiglu-tangent-compute-v1"
        or tangent_compute_sha256 != canonical_sha256(tangent_compute_body)
        or tangent_compute.get("requested_device") != requested_fold_compute_device
        or tangent_compute.get("resolved_device") != requested_fold_compute_device
        or tangent_compute.get("backend")
        != ("cpu" if requested_fold_compute_device == "cpu" else "cuda")
        or tangent_compute.get("compute_dtype") != "float64"
        or tangent_compute.get("authenticated_source_archive_device") != "cpu"
        or tangent_compute.get("materialized_tangent_bank_device") != "cpu"
        or tangent_compute.get("silent_device_fallback_performed") is not False
        or integration.get("tangent_bank_compute_execution") != dict(tangent_compute)
        or tangent.get("operation_audit", {}).get("training_steps") != 0
        or tangent.get("operation_audit", {}).get(
            "numeric_diagnostics_used_for_semantic_decision"
        )
        is not False
        or full_solver.get("qdeim_integration_sha256") != canonical_sha256(integration)
        or integration.get("fixed_tangent_anchor_count") != _QDEIM_TANGENT_ANCHOR_COUNT
        or integration.get("heldout_rows_used_for_pod_or_tangent") != 0
        or integration.get("tangent_rows_used_for_endpoint_w2_correction") != 0
        or integration.get("numeric_metrics_used_for_semantic_acceptance_or_veto")
        is not False
        or integration.get("whole_decoded_responses_are_primary_semantic_evidence")
        is not True
        or integration.get("binary_manual_gate_created") is not False
        or integration.get("source_inventory_sha256") != inventory.inventory_sha256
        or integration.get("selected_state_sha256") != _state_sha256(full_state)
        or not isinstance(operation_audit, Mapping)
        or any(
            operation_audit.get(name) != value
            for name, value in _ZERO_OPERATIONS.items()
        )
        or full_solver.get("heldout_rows_used_by_solver") != 0
        or full_solver.get("heldout_rows_used_for_plan") != 0
        or full_solver.get("all_source_writer_columns_consumed") is not True
        or full_solver.get("full_source_coverage") is not True
    ):
        raise Lfm2SemanticMlpRefoldError(
            "Q-DEIM cold-start proposal failed authenticated structural, tangent, "
            "decoder-control, or no-training invariants."
        )
    planted_roles = _nonzero_swiglu_roles(full_state)
    return MappingProxyType(
        {
            **dict(full_solver),
            "cold_start_escape": {
                "incumbent_w1_w3_w2_exactly_zero": True,
                "requested_maximum_delta_ratio": requested_maximum_delta_ratio,
                "effective_full_structural_maximum_delta_ratio": None,
                "relative_incumbent_trust_cap_applied": False,
                "authenticated_full_source_structural_solve_used": True,
                "authenticated_normalized_input_bridge_used_for_w1_w3": True,
                "authenticated_raw_residual_chart_used_only_for_w2": True,
                "producer_consumer_bridge_fields_separate": True,
                "coupled_real_qdeim_gate_up_used": True,
                "shared_normalized_input_decoder_used": True,
                "per_neuron_producer_interpolation_used": False,
                "dense_all_source_qdeim_consumer_used": True,
                "materialized_target_dtype_producers_used_for_w2": True,
                "random_initialization_used": False,
                "gradient_training_used": False,
            },
            "source_derived_planting": {
                "source_inventory_sha256": inventory.inventory_sha256,
                "compiler_lineage_sha256": inventory.compiler_lineage_sha256,
                "residual_assignment_sha256": inventory.residual_assignment_sha256,
                "normalization_lineage_sha256": inventory.normalization_lineage_sha256,
                "residual_chart_sha256": lfm2_authenticated_tensor_sha256(
                    inventory.residual_chart
                ),
                "normalized_input_decoder_sha256": integration[
                    "normalized_input_decoder_sha256"
                ],
                "normalized_decoder_control_receipt_sha256": full_solver[
                    "normalized_decoder_control_receipt_sha256"
                ],
                "fold_compute_device": requested_fold_compute_device,
                "normalized_decoder_control_compute_device": "cpu",
                "normalized_decoder_control_medoid_or_cluster_plan_compiled": False,
                "normalized_decoder_control_legacy_residual_writer_compiled": False,
                "qdeim_pod_and_pivot_compute_device": (requested_fold_compute_device),
                "qdeim_compute_execution": dict(qdeim_compute),
                "child_coordinate_rms_receipt_sha256": integration[
                    "child_coordinate_rms_receipt_sha256"
                ],
                "tangent_bank_sha256": full_solver["tangent_bank_sha256"],
                "tangent_bank_compute_execution": dict(tangent_compute),
                "tangent_bank_compute_device": requested_fold_compute_device,
                "qdeim_result_sha256": qdeim_result_sha256,
                "qdeim_integration_sha256": full_solver["qdeim_integration_sha256"],
                "selected_qdeim_candidate_id": expected_candidate,
                "full_source_coverage": True,
                "producer_averaging": False,
                "producer_per_neuron_interpolation": False,
                "producer_shared_decoder": True,
                "source_neurons_dropped": 0,
                "heldout_rows_used_by_solver": 0,
                "tangent_rows_used_for_endpoint_w2_correction": 0,
                "planted_roles": list(planted_roles),
                "all_dense_w1_w3_w2_roles_planted": (
                    planted_roles == ("w1.weight", "w3.weight", "w2.weight")
                ),
                "fit_partition_only_used_for_endpoint_solve": True,
                "structural_tangent_rows_used_for_pod_only": True,
            },
        }
    )


def _full_candidate_state(
    fit: Sequence[Lfm2SemanticMlpEndpointEvidence],
    inventory: Lfm2DirectLocalMlpSourceInventory,
    incumbent: Mapping[str, Any],
    *,
    target_ffn_norm_weight: Any,
    exact_zero_incumbent: bool,
    ridge: float,
    maximum_delta_ratio: float | None,
    cluster_ridge: float,
    cluster_max_iterations: int,
    cluster_distance_chunk_size: int,
    fold_compute_device: str,
    mlp_lane: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    qdeim_lane = mlp_lane in _QDEIM_MLP_LANE_TO_CANDIDATE
    if qdeim_lane and not exact_zero_incumbent:
        raise Lfm2SemanticMlpRefoldError(
            "Nonlinear Q-DEIM body recipes require an exact-zero blank target MLP."
        )
    cases = tuple(
        Lfm2CachedMlpFoldCase(
            target_input=item.actual_child_mlp_input,
            parent_gate=item.parent_gate,
            parent_up=item.parent_up,
            parent_post_swiglu=item.parent_post_swiglu,
            parent_post_mlp_residual_target_space=(
                item.parent_post_mlp_residual_target_space
            ),
            target_pre_mlp_residual=item.actual_child_pre_mlp_residual,
        )
        for item in fit
    )
    if exact_zero_incumbent:
        normalized_cases = tuple(
            Lfm2CachedNormalizedInputMlpFoldCase(
                target_input=item.actual_child_mlp_input,
                parent_normalized_input=item.parent_normalized_mlp_input,
                parent_gate=item.parent_gate,
                parent_up=item.parent_up,
                parent_post_swiglu=item.parent_post_swiglu,
                parent_post_mlp_residual_target_space=(
                    item.parent_post_mlp_residual_target_space
                ),
                target_pre_mlp_residual=item.actual_child_pre_mlp_residual,
            )
            for item in fit
        )
        if qdeim_lane:
            normalized_control = compile_lfm2_normalized_input_decoder(
                residual_chart=inventory.residual_chart,
                residual_chart_sha256=lfm2_authenticated_tensor_sha256(
                    inventory.residual_chart
                ),
                parent_ffn_norm_weight=inventory.ffn_norm_weight,
                parent_ffn_norm_weight_sha256=lfm2_authenticated_tensor_sha256(
                    inventory.ffn_norm_weight
                ),
                target_ffn_norm_weight=target_ffn_norm_weight,
                target_ffn_norm_weight_sha256=lfm2_authenticated_tensor_sha256(
                    target_ffn_norm_weight
                ),
                cases=normalized_cases,
                case_sample_weights=tuple(item.sample_weight for item in fit),
                normalized_decoder_ridge=_NORMALIZED_DECODER_RIDGE,
            )
            if (
                not isinstance(
                    normalized_control,
                    Lfm2CompiledNormalizedInputDecoder,
                )
                or normalized_control.verify().get("verified") is not True
                or normalized_control.normalized_input_decoder_rank
                != normalized_control.target_hidden_size
                or normalized_control.normalized_input_decoder_sha256
                != lfm2_authenticated_tensor_sha256(
                    normalized_control.normalized_input_decoder
                )
            ):
                raise Lfm2SemanticMlpRefoldError(
                    "Decoder-only control did not authenticate one full-rank "
                    "Q-DEIM input decoder."
                )
            return _compile_qdeim_full_candidate(
                fit,
                inventory,
                incumbent,
                normalized_control=normalized_control,
                mlp_lane=mlp_lane,
                ridge=ridge,
                fold_compute_device=fold_compute_device,
            )
        compiled = compile_lfm2_normalized_input_real_medoid_structural_mlp_fold(
            source_gate_weight=inventory.gate_weight,
            source_up_weight=inventory.up_weight,
            source_down_weight=inventory.down_weight,
            target_gate_weight_template=incumbent["w1.weight"],
            target_up_weight_template=incumbent["w3.weight"],
            target_down_weight_template=incumbent["w2.weight"],
            residual_chart=inventory.residual_chart,
            residual_chart_sha256=lfm2_authenticated_tensor_sha256(
                inventory.residual_chart
            ),
            parent_ffn_norm_weight=inventory.ffn_norm_weight,
            parent_ffn_norm_weight_sha256=lfm2_authenticated_tensor_sha256(
                inventory.ffn_norm_weight
            ),
            target_ffn_norm_weight=target_ffn_norm_weight,
            target_ffn_norm_weight_sha256=lfm2_authenticated_tensor_sha256(
                target_ffn_norm_weight
            ),
            cases=normalized_cases,
            case_sample_weights=tuple(item.sample_weight for item in fit),
            normalized_decoder_ridge=_NORMALIZED_DECODER_RIDGE,
            ridge=ridge,
            max_prior_delta_ratio=None,
            cluster_ridge=cluster_ridge,
            cluster_max_iterations=cluster_max_iterations,
            cluster_distance_chunk_size=cluster_distance_chunk_size,
            fold_compute_device=fold_compute_device,
        )
        if not isinstance(
            compiled,
            Lfm2CompiledNormalizedInputRealMedoidStructuralMlpFold,
        ):
            raise Lfm2SemanticMlpRefoldError(
                "Exact-zero MLP dispatch did not return the normalized-input "
                "multi-anchor compiler result."
            )
        solver = (
            "fit-only-authenticated-normalized-input-real-medoid-multi-anchor-"
            "plus-shared-decoder-and-w2-closed-form-solve"
        )
        incumbent_priors_used = False
    else:
        compiled = compile_lfm2_structural_residual_mlp_fold(
            source_gate_weight=inventory.gate_weight,
            source_up_weight=inventory.up_weight,
            source_down_weight=inventory.down_weight,
            target_gate_weight_template=incumbent["w1.weight"],
            target_up_weight_template=incumbent["w3.weight"],
            target_down_weight_template=incumbent["w2.weight"],
            cases=cases,
            gate_weight_prior=incumbent["w1.weight"],
            up_weight_prior=incumbent["w3.weight"],
            down_weight_prior=incumbent["w2.weight"],
            residual_assignment_sha256=inventory.residual_assignment_sha256,
            ridge=ridge,
            max_prior_delta_ratio=None,
            cluster_ridge=cluster_ridge,
            cluster_max_iterations=cluster_max_iterations,
            cluster_distance_chunk_size=cluster_distance_chunk_size,
        )
        if not isinstance(compiled, Lfm2CompiledStructuralMlpFold):
            raise Lfm2SemanticMlpRefoldError(
                "Nonzero MLP dispatch did not return the legacy structural "
                "compiler result."
            )
        solver = "fit-only-full-source-cluster-plus-three-closed-form-linear-solves"
        incumbent_priors_used = True
    if (
        not compiled.full_source_coverage
        or compiled.source_state_sha256_before != compiled.source_state_sha256_after
    ):
        raise Lfm2SemanticMlpRefoldError(
            "Full structural MLP compiler did not preserve complete source coverage."
        )
    receipt = {
        **compiled.as_dict(),
        "solver": solver,
        "fit_rows": len(fit),
        "heldout_rows_used_by_solver": 0,
        "analytic_linear_solve": True,
        "cluster_solve": True,
        "requested_maximum_delta_ratio_goal": maximum_delta_ratio,
        "effective_maximum_delta_ratio_cap": None,
        "delta_ratio_cap_applied": False,
        "numeric_delta_ratio_used_for_candidate_eligibility": False,
        "numeric_delta_ratio_used_for_candidate_ordering": False,
        "numeric_delta_ratio_used_for_publication_or_rollback": False,
        "numeric_delta_ratio_has_final_intelligence_authority": False,
        "current_dense_c61_w1_w3_w2_used_as_priors": incumbent_priors_used,
        **(
            {
                "authenticated_normalized_input_multi_anchor_source_priors_used": (
                    True
                ),
                "authenticated_raw_residual_consumer_chart_used": True,
            }
            if exact_zero_incumbent
            else {}
        ),
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    return compiled.state_dict(), MappingProxyType(receipt)


def _changed_roles(
    incumbent: Mapping[str, Any], candidate: Mapping[str, Any]
) -> tuple[str, ...]:
    torch = _torch()
    return tuple(
        name for name in incumbent if not torch.equal(incumbent[name], candidate[name])
    )


def _planting_strength_diagnostics(
    candidate: Mapping[str, Any],
    incumbent: Mapping[str, Any],
    *,
    requested_maximum_delta_ratio_goal: float | None,
) -> dict[str, Any]:
    """Describe materialized planting strength without controlling the plant."""

    torch = _torch()
    by_role: dict[str, dict[str, float]] = {}
    for role in ("w1.weight", "w3.weight", "w2.weight"):
        prior = incumbent[role].to(torch.float64)
        delta = candidate[role].to(torch.float64) - prior
        incumbent_norm = float(torch.linalg.vector_norm(prior).item())
        delta_norm = float(torch.linalg.vector_norm(delta).item())
        observed_ratio = delta_norm / max(incumbent_norm, 1e-30)
        if not math.isfinite(observed_ratio):
            observed_ratio = _MAX_FINITE_FLOAT
        by_role[role] = {
            "incumbent_norm": incumbent_norm,
            "delta_norm": delta_norm,
            "observed_delta_ratio": observed_ratio,
        }
    observed_maximum = max(value["observed_delta_ratio"] for value in by_role.values())
    return {
        "requested_maximum_delta_ratio_goal": (requested_maximum_delta_ratio_goal),
        "observed_maximum_delta_ratio": observed_maximum,
        "observed_delta_ratio_by_role": by_role,
        "requested_delta_ratio_goal_met": (
            None
            if requested_maximum_delta_ratio_goal is None
            else observed_maximum <= requested_maximum_delta_ratio_goal
        ),
        "effective_maximum_delta_ratio_cap": None,
        "delta_ratio_cap_applied": False,
        "delta_ratio_used_to_scale_candidate": False,
        "delta_ratio_diagnostic_only": True,
        "numeric_delta_ratio_used_for_candidate_eligibility": False,
        "numeric_delta_ratio_used_for_candidate_ordering": False,
        "numeric_delta_ratio_used_for_publication_or_rollback": False,
        "numeric_delta_ratio_has_final_intelligence_authority": False,
    }


def _candidate_state_structural_report(
    state: Mapping[str, Any],
    incumbent: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Authenticate candidate safety without applying a numeric-quality veto."""

    torch = _torch()
    exact_roles = isinstance(state, Mapping) and set(state) == set(incumbent)
    role_reports: dict[str, dict[str, Any]] = {}
    tensor_hashes: dict[str, str] = {}
    normalized: dict[str, Any] = {}
    for name in sorted(
        set(incumbent) | (set(state) if isinstance(state, Mapping) else set())
    ):
        value = state.get(name) if isinstance(state, Mapping) else None
        baseline = incumbent.get(name)
        tensor = isinstance(value, torch.Tensor)
        baseline_tensor = isinstance(baseline, torch.Tensor)
        dense = bool(tensor and value.layout == torch.strided)
        materialized = bool(tensor and value.device.type != "meta")
        shape_matches = bool(
            tensor and baseline_tensor and tuple(value.shape) == tuple(baseline.shape)
        )
        dtype_matches = bool(
            tensor and baseline_tensor and value.dtype == baseline.dtype
        )
        finite = bool(
            tensor
            and dense
            and materialized
            and bool(torch.isfinite(value).all().item())
        )
        if tensor and dense and materialized:
            private = value.detach().to(device="cpu").contiguous().clone()
            normalized[name] = private
            tensor_hashes[name] = _tensor_sha256(private)
        role_reports[name] = {
            "present_in_candidate": bool(isinstance(state, Mapping) and name in state),
            "present_in_incumbent": name in incumbent,
            "tensor": tensor,
            "dense_strided": dense,
            "materialized_nonmeta": materialized,
            "shape_matches_incumbent": shape_matches,
            "dtype_matches_incumbent": dtype_matches,
            "all_elements_finite": finite,
            "private_detached_cpu_copy_available": name in normalized,
            "shape": list(value.shape) if tensor else None,
            "dtype": str(value.dtype).removeprefix("torch.") if tensor else None,
            "element_count": int(value.numel()) if tensor else None,
        }
    structural_safety = bool(
        exact_roles
        and role_reports
        and all(
            report[flag] is True
            for report in role_reports.values()
            for flag in (
                "tensor",
                "dense_strided",
                "materialized_nonmeta",
                "shape_matches_incumbent",
                "dtype_matches_incumbent",
                "all_elements_finite",
                "private_detached_cpu_copy_available",
            )
        )
    )
    changed_roles = (
        tuple(
            name
            for name in sorted(incumbent)
            if not torch.equal(normalized[name], incumbent[name])
        )
        if structural_safety
        else ()
    )
    body = {
        "candidate_eligibility_basis": _CANDIDATE_ELIGIBILITY_BASIS,
        "exact_dense_topology_roles_covered": exact_roles,
        "role_reports": role_reports,
        "state_tensor_bundle_sha256": (
            canonical_sha256(tensor_hashes)
            if exact_roles and len(tensor_hashes) == len(incumbent)
            else None
        ),
        "structural_safety_passed": structural_safety,
        "changed_roles": list(changed_roles),
        "effectful_nonincumbent": bool(structural_safety and changed_roles),
        "structurally_eligible": bool(structural_safety and changed_roles),
        "numeric_metrics_used_for_structural_eligibility": False,
        "numeric_thresholds_used_for_structural_eligibility": False,
        "relative_magnitude_used_for_structural_eligibility": False,
    }
    return MappingProxyType(
        {
            **body,
            "structural_report_sha256": canonical_sha256(body),
        }
    )


def _freeze_candidate_proposal_state(
    state: Mapping[str, Any],
    incumbent: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return a private valid state plus the unmodified proposal's safety seal.

    The typed prepared artifact always carries both analytic roles.  An invalid
    proposal is represented by an incumbent-shaped placeholder, while its
    sealed structural report remains in the solver and selection receipts and
    keeps it out of the analytic frontier.
    """

    report = _candidate_state_structural_report(state, incumbent)
    source = state if report["structural_safety_passed"] is True else incumbent
    return (
        MappingProxyType(
            {
                name: source[name].detach().to(device="cpu").contiguous().clone()
                for name in incumbent
            }
        ),
        report,
    )


def _fit_metric_ordering_key(
    candidate: Lfm2FrozenSemanticMlpCandidate,
) -> tuple[Any, ...]:
    """Order safe analytic proposals without converting metrics into gates."""

    metrics = candidate.metrics["fit"]
    complexity = {
        "incumbent-producer-w2": 0,
        "full-w1-w3-w2": 1,
    }
    if candidate.role not in complexity:
        raise Lfm2SemanticMlpRefoldError(
            "Only analytic MLP candidates have deterministic FIT ordering keys."
        )
    return (
        metrics.endpoint_relative_rmse,
        metrics.endpoint_rmse,
        metrics.endpoint_maximum_absolute_error,
        -metrics.mean_endpoint_cosine,
        -metrics.minimum_endpoint_cosine,
        complexity[candidate.role],
        _state_sha256(candidate.state_dict()),
    )


def _nonzero_swiglu_roles(state: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the nonzero roles of one complete dense W1/W3/W2 state."""

    torch = _torch()
    roles = ("w1.weight", "w3.weight", "w2.weight")
    if set(state) != set(roles):
        raise Lfm2SemanticMlpRefoldError(
            "Cold-start detection requires one complete dense W1/W3/W2 state."
        )
    return tuple(
        role for role in roles if int(torch.count_nonzero(state[role]).item()) > 0
    )


def _exact_zero_swiglu_state(state: Mapping[str, Any]) -> bool:
    """Return whether the complete dense W1/W3/W2 incumbent is exactly zero."""

    return not _nonzero_swiglu_roles(state)


def _terminal_banks(
    partitions: Mapping[str, Sequence[Lfm2SemanticMlpEndpointEvidence]],
    *,
    required: bool,
) -> Mapping[str, Lfm2CpcfTerminalActionEvidence]:
    torch = _torch()
    if not required:
        return MappingProxyType({})
    result = {}
    for partition in _PARTITIONS:
        values = partitions[partition]
        if any(item.parent_final_norm_output_target_space is None for item in values):
            raise Lfm2SemanticMlpRefoldError(
                "Final-layer W2 deferral requires mapped target final-norm evidence "
                f"for every {partition} endpoint."
            )
        result[partition] = Lfm2CpcfTerminalActionEvidence(
            split=partition,
            row_ids=tuple(
                f"semantic-mlp:b{item.batch_index}:p{item.phase_index}:"
                f"{item.phase_sha256}"
                for item in values
            ),
            mlp_input=torch.cat(
                tuple(item.actual_child_mlp_input for item in values), dim=0
            ),
            pre_mlp_residual=torch.cat(
                tuple(item.actual_child_pre_mlp_residual for item in values), dim=0
            ),
            parent_final_norm_output=torch.cat(
                tuple(item.parent_final_norm_output_target_space for item in values),
                dim=0,
            ),
            sample_weights=torch.tensor(
                [item.sample_weight for item in values], dtype=torch.float64
            ),
        )
    return MappingProxyType(result)


def prepare_lfm2_semantic_mlp_postfold(
    *,
    target_model: Any,
    layer_index: int,
    evidence: Sequence[Lfm2SemanticMlpEndpointEvidence],
    cache_key: str,
    cache_manifest_sha256: str,
    bridge_artifact_sha256: str,
    ridge: float = 1e-6,
    maximum_delta_ratio: float | None = 0.5,
    cluster_ridge: float = 1e-8,
    cluster_max_iterations: int = 16,
    cluster_distance_chunk_size: int = 512,
    rmse_relative_tolerance: float = 1e-6,
    cosine_tolerance: float = 1e-8,
    require_terminal_action_evidence: bool = False,
    fold_compute_device: str = "cpu",
    body_recipe: str | Lfm2SemanticBodyRecipe = "fit_guided",
) -> Lfm2PreparedSemanticMlpRefold:
    """Compile and freeze both FIT-only candidates without mutating the child."""

    torch = _torch()
    cache = _digest(cache_key, "cache key")
    manifest = _digest(cache_manifest_sha256, "cache manifest")
    bridge = _digest(bridge_artifact_sha256, "bridge artifact")
    recipe = resolve_lfm2_semantic_body_recipe(body_recipe)
    if (
        not isinstance(ridge, (int, float))
        or isinstance(ridge, bool)
        or not math.isfinite(float(ridge))
        or float(ridge) <= 0.0
        or maximum_delta_ratio is not None
        and (
            not isinstance(maximum_delta_ratio, (int, float))
            or isinstance(maximum_delta_ratio, bool)
            or not math.isfinite(float(maximum_delta_ratio))
            or float(maximum_delta_ratio) <= 0.0
        )
        or not math.isfinite(float(rmse_relative_tolerance))
        or float(rmse_relative_tolerance) < 0.0
        or not math.isfinite(float(cosine_tolerance))
        or float(cosine_tolerance) < 0.0
    ):
        raise Lfm2SemanticMlpRefoldError("Semantic MLP solver policy is invalid.")
    raw = tuple(evidence)
    if not raw or not all(
        isinstance(item, Lfm2SemanticMlpEndpointEvidence)
        and item.verify()
        and item.source_inventory.verify()
        for item in raw
    ):
        raise Lfm2SemanticMlpRefoldError(
            "Semantic MLP refolding requires authenticated typed endpoint evidence."
        )
    current_state = fingerprint_lfm2_parent(target_model).identity_sha256
    topology = _topology_sha256(target_model)
    parameter_count = _parameter_count(target_model)
    inventory = raw[0].source_inventory
    if any(
        item.layer_index != layer_index
        or item.cache_key != cache
        or item.cache_manifest_sha256 != manifest
        or item.bridge_artifact_sha256 != bridge
        or item.captured_child_model_state_sha256 != current_state
        or item.source_inventory.inventory_sha256 != inventory.inventory_sha256
        or item.source_inventory.compiler_lineage_sha256
        != inventory.compiler_lineage_sha256
        for item in raw
    ):
        raise Lfm2SemanticMlpRefoldError(
            "Layer/cache/bridge/source lineage diverged or child replay is stale."
        )
    parameters = _parameter_bundle(target_model, layer_index)
    incumbent_state = _suffix_state(parameters, layer_index)
    target_ffn_norm_weight = _ffn_norm_weight(target_model, layer_index)
    target_intermediate, target_hidden = map(int, incumbent_state["w1.weight"].shape)
    target_dtype = incumbent_state["w1.weight"].dtype
    if (
        tuple(incumbent_state["w3.weight"].shape)
        != (target_intermediate, target_hidden)
        or tuple(incumbent_state["w2.weight"].shape)
        != (target_hidden, target_intermediate)
        or tuple(target_ffn_norm_weight.shape) != (target_hidden,)
        or target_dtype not in {torch.bfloat16, torch.float32}
        or target_ffn_norm_weight.dtype != target_dtype
        or inventory.target_hidden_size != target_hidden
        or any(item.target_hidden_size != target_hidden for item in raw)
        or any(item.actual_child_mlp_input.dtype != target_dtype for item in raw)
    ):
        raise Lfm2SemanticMlpRefoldError(
            "Evidence and current child SwiGLU topology differ."
        )
    partitions = _split_evidence(raw)
    source_before = inventory.inventory_sha256
    exact_zero_incumbent = _exact_zero_swiglu_state(incumbent_state)
    mlp_lane = recipe.component_lanes["mlp"]
    qdeim_lane = mlp_lane in _QDEIM_MLP_LANE_TO_CANDIDATE
    requested_maximum_delta_ratio = (
        None if maximum_delta_ratio is None else float(maximum_delta_ratio)
    )
    full_state, full_solver = _full_candidate_state(
        partitions["fit"],
        inventory,
        incumbent_state,
        target_ffn_norm_weight=target_ffn_norm_weight,
        exact_zero_incumbent=exact_zero_incumbent,
        ridge=float(ridge),
        maximum_delta_ratio=requested_maximum_delta_ratio,
        cluster_ridge=float(cluster_ridge),
        cluster_max_iterations=int(cluster_max_iterations),
        cluster_distance_chunk_size=int(cluster_distance_chunk_size),
        fold_compute_device=fold_compute_device,
        mlp_lane=mlp_lane,
    )
    if exact_zero_incumbent and qdeim_lane:
        full_solver = _authenticate_qdeim_full_solver(
            full_state,
            full_solver,
            inventory,
            mlp_lane=mlp_lane,
            requested_maximum_delta_ratio=requested_maximum_delta_ratio,
        )
    elif exact_zero_incumbent:
        torch = _torch()
        consumer_correction = full_solver.get("consumer_correction_report")
        producer_bridge = full_solver.get("producer_bridge")
        consumer_bridge = full_solver.get("consumer_bridge")
        multi_anchor_plan = full_solver.get("multi_anchor_plan")
        weight_complete_signature = full_solver.get("weight_complete_signature")
        weight_signature_compute_execution = (
            weight_complete_signature.get("compute_execution")
            if isinstance(weight_complete_signature, Mapping)
            else None
        )
        if fold_compute_device == "cpu":
            weight_signature_compute_execution_valid = (
                weight_signature_compute_execution is None
            )
        elif isinstance(weight_signature_compute_execution, Mapping):
            compute_execution_body = dict(weight_signature_compute_execution)
            compute_execution_sha256 = compute_execution_body.pop(
                "execution_sha256",
                None,
            )
            weight_signature_compute_execution_valid = bool(
                weight_signature_compute_execution.get("backend") == "cuda"
                and weight_signature_compute_execution.get("requested_device")
                == fold_compute_device
                and weight_signature_compute_execution.get("compute_dtype") == "float64"
                and isinstance(compute_execution_sha256, str)
                and canonical_sha256(compute_execution_body) == compute_execution_sha256
            )
        else:
            weight_signature_compute_execution_valid = False
        fit_weights = tuple(item.sample_weight for item in partitions["fit"])
        fit_row_weights = torch.tensor(fit_weights, dtype=torch.float64)
        expected_chart_sha256 = lfm2_authenticated_tensor_sha256(
            inventory.residual_chart
        )
        expected_parent_norm_sha256 = lfm2_authenticated_tensor_sha256(
            inventory.ffn_norm_weight
        )
        expected_target_norm_sha256 = lfm2_authenticated_tensor_sha256(
            target_ffn_norm_weight
        )
        normalized_chart = (
            inventory.residual_chart.to(torch.float64)
            * inventory.ffn_norm_weight.to(torch.float64).reciprocal().unsqueeze(1)
            * target_ffn_norm_weight.to(torch.float64).unsqueeze(0)
        ).contiguous()
        expected_normalized_chart_sha256 = lfm2_authenticated_tensor_sha256(
            normalized_chart
        )
        normalized_fit_cases = tuple(
            Lfm2CachedNormalizedInputMlpFoldCase(
                target_input=item.actual_child_mlp_input,
                parent_normalized_input=item.parent_normalized_mlp_input,
                parent_gate=item.parent_gate,
                parent_up=item.parent_up,
                parent_post_swiglu=item.parent_post_swiglu,
                parent_post_mlp_residual_target_space=(
                    item.parent_post_mlp_residual_target_space
                ),
                target_pre_mlp_residual=item.actual_child_pre_mlp_residual,
            )
            for item in partitions["fit"]
        )
        expected_fit_evidence_sha256 = lfm2_normalized_input_mlp_fit_evidence_sha256(
            normalized_fit_cases,
            fit_weights,
        )
        expected_weight_sha256 = lfm2_authenticated_tensor_sha256(fit_row_weights)
        weight_signature_report_body = (
            dict(weight_complete_signature)
            if isinstance(weight_complete_signature, Mapping)
            else {}
        )
        weight_signature_report_sha256 = weight_signature_report_body.pop(
            "report_sha256",
            None,
        )
        with torch.no_grad():
            chart64 = inventory.residual_chart.to(torch.float64)
            chart_identity = torch.eye(target_hidden, dtype=torch.float64)
            if torch.allclose(
                chart64.T @ chart64,
                chart_identity,
                rtol=1e-8,
                atol=1e-8,
            ):
                expected_signature_basis = chart64.contiguous().clone()
            else:
                chart_u, _, chart_vh = torch.linalg.svd(
                    chart64,
                    full_matrices=False,
                )
                expected_signature_basis = (chart_u @ chart_vh).contiguous()
        expected_signature_input_sha256 = {
            "w1.weight": lfm2_authenticated_tensor_sha256(
                inventory.gate_weight.to(torch.float64)
            ),
            "w3.weight": lfm2_authenticated_tensor_sha256(
                inventory.up_weight.to(torch.float64)
            ),
            "w2.weight": lfm2_authenticated_tensor_sha256(
                inventory.down_weight.to(torch.float64)
            ),
            "residual_basis": lfm2_authenticated_tensor_sha256(
                expected_signature_basis
            ),
        }
        planted_roles = _nonzero_swiglu_roles(full_state)
        if (
            not isinstance(consumer_correction, Mapping)
            or consumer_correction.get("role")
            != "mlp_normalized_bridge_multi_anchor_down"
            or consumer_correction.get("prior_provided") is not True
            or not isinstance(consumer_correction.get("prior_norm"), (int, float))
            or not math.isfinite(float(consumer_correction["prior_norm"]))
            or float(consumer_correction["prior_norm"]) <= 0.0
            or consumer_correction.get("update_scale") != 1.0
            or full_solver.get("schema")
            != "neural-forge-lfm2-normalized-input-real-medoid-mlp-fold-v1"
            or full_solver.get("parent_ffn_norm_weight_sha256")
            != expected_parent_norm_sha256
            or full_solver.get("target_ffn_norm_weight_sha256")
            != expected_target_norm_sha256
            or not isinstance(producer_bridge, Mapping)
            or producer_bridge.get("role") != "normalized_mlp_input_target_to_parent"
            or isinstance(producer_bridge.get("fit_rank"), bool)
            or not isinstance(producer_bridge.get("fit_rank"), int)
            or not 0 < producer_bridge["fit_rank"] <= target_hidden
            or isinstance(producer_bridge.get("decoder_rank"), bool)
            or not isinstance(producer_bridge.get("decoder_rank"), int)
            or producer_bridge.get("decoder_rank") != target_hidden
            or isinstance(producer_bridge.get("structural_decoder_prior_rank"), bool)
            or not isinstance(producer_bridge.get("structural_decoder_prior_rank"), int)
            or producer_bridge.get("structural_decoder_prior_rank") != target_hidden
            or not isinstance(
                producer_bridge.get("structural_decoder_prior_operator_norm"),
                (int, float),
            )
            or not math.isfinite(
                float(producer_bridge["structural_decoder_prior_operator_norm"])
            )
            or float(producer_bridge["structural_decoder_prior_operator_norm"]) <= 0.0
            or not isinstance(
                producer_bridge.get("structural_decoder_left_inverse_error"),
                (int, float),
            )
            or not math.isfinite(
                float(producer_bridge["structural_decoder_left_inverse_error"])
            )
            or float(producer_bridge["structural_decoder_left_inverse_error"]) > 1e-8
            or not isinstance(
                producer_bridge.get("positive_global_gain"),
                (int, float),
            )
            or not math.isfinite(float(producer_bridge["positive_global_gain"]))
            or float(producer_bridge["positive_global_gain"]) <= 0.0
            or producer_bridge.get("ridge_policy_value") != _NORMALIZED_DECODER_RIDGE
            or producer_bridge.get("supported_relative_cap_rho") != 1.0
            or not isinstance(
                producer_bridge.get("correction_rowspace_leakage"),
                (int, float),
            )
            or not math.isfinite(float(producer_bridge["correction_rowspace_leakage"]))
            or float(producer_bridge["correction_rowspace_leakage"]) > 1e-8
            or not isinstance(
                producer_bridge.get("null_completion_error"), (int, float)
            )
            or not math.isfinite(float(producer_bridge["null_completion_error"]))
            or float(producer_bridge["null_completion_error"]) > 1e-8
            or not isinstance(
                producer_bridge.get("null_completion_operator_norm"),
                (int, float),
            )
            or not isinstance(
                producer_bridge.get("null_completion_operator_norm_bound"),
                (int, float),
            )
            or not math.isfinite(
                float(producer_bridge["null_completion_operator_norm"])
            )
            or not math.isfinite(
                float(producer_bridge["null_completion_operator_norm_bound"])
            )
            or float(producer_bridge["null_completion_operator_norm"]) < 0.0
            or float(producer_bridge["null_completion_operator_norm_bound"]) < 0.0
            or float(producer_bridge["null_completion_operator_norm"])
            > float(producer_bridge["null_completion_operator_norm_bound"]) + 1e-8
            or full_solver.get("normalized_chart_sha256")
            != expected_normalized_chart_sha256
            or producer_bridge.get("normalized_chart_sha256")
            != full_solver.get("normalized_chart_sha256")
            or producer_bridge.get("decoder_sha256")
            != full_solver.get("normalized_input_decoder_sha256")
            or not isinstance(consumer_bridge, Mapping)
            or consumer_bridge.get("role") != "raw_residual_source_to_target"
            or consumer_bridge.get("residual_chart_sha256") != expected_chart_sha256
            or consumer_bridge.get("consumer_prior_sha256")
            != full_solver.get("consumer_prior_sha256")
            or full_solver.get("producer_consumer_bridge_fields_separate") is not True
            or full_solver.get("producer_averaging") is not False
            or full_solver.get("producer_per_neuron_interpolation") is not False
            or full_solver.get("producer_shared_decoder") is not True
            or full_solver.get("full_source_coverage") is not True
            or full_solver.get("full_source_assignment_fraction") != 1.0
            or full_solver.get("source_neurons_dropped") != 0
            or full_solver.get("plan_evidence_scope")
            != "all_source_w1_w3_w2_weights_in_global_residual_basis"
            or full_solver.get("fit_cases_used_for_plan_diagnostics_only") is not True
            or full_solver.get("heldout_rows_used_for_plan") != 0
            or full_solver.get("heldout_rows_used_by_solver") != 0
            or full_solver.get("endpoint_feature_source")
            != "materialized_target_dtype_w1_w3_cast_back_to_fp64"
            or full_solver.get("current_dense_c61_w1_w3_w2_used_as_priors") is not False
            or full_solver.get(
                "authenticated_normalized_input_multi_anchor_source_priors_used"
            )
            is not True
            or full_solver.get("authenticated_raw_residual_consumer_chart_used")
            is not True
            or not isinstance(weight_complete_signature, Mapping)
            or weight_complete_signature.get("schema")
            != "neural-forge-gated-mlp-weight-complete-signature-v1"
            or weight_complete_signature.get("source_neurons")
            != inventory.source_intermediate_size
            or weight_complete_signature.get("source_hidden_size")
            != inventory.source_hidden_size
            or weight_complete_signature.get("target_hidden_size")
            != inventory.target_hidden_size
            or weight_complete_signature.get("input_sha256")
            != expected_signature_input_sha256
            or not isinstance(weight_complete_signature.get("invariants"), Mapping)
            or weight_complete_signature["invariants"].get("all_hold") is not True
            or not isinstance(weight_signature_report_sha256, str)
            or weight_signature_report_sha256
            != canonical_sha256(weight_signature_report_body)
            or full_solver.get("weight_complete_signature_report_sha256")
            != weight_signature_report_sha256
            or weight_signature_compute_execution_valid is not True
            or full_solver.get("weight_complete_signature_sha256")
            != weight_complete_signature.get("signature_sha256")
            or full_solver.get(
                "weight_complete_signature_used_for_structural_selection"
            )
            is not True
            or full_solver.get("cached_activation_rows_used_for_structural_selection")
            is not False
            or full_solver.get("cached_activation_rows_used_for_sparse_anchor_fit")
            is not False
            or not isinstance(multi_anchor_plan, Mapping)
            or multi_anchor_plan.get("calibration_provenance")
            != "cached_fit_rows_diagnostics_only"
            or multi_anchor_plan.get("planning_signature_source")
            != "full-weight-complete-structural-signature"
            or multi_anchor_plan.get("planning_signature_sha256")
            != weight_complete_signature.get("signature_sha256")
            or multi_anchor_plan.get("planning_signature_provenance")
            != weight_signature_report_sha256
            or multi_anchor_plan.get("cached_activation_rows_used_for_medoid_selection")
            is not False
            or multi_anchor_plan.get(
                "cached_activation_rows_used_for_sparse_anchor_fit"
            )
            is not False
            or not isinstance(multi_anchor_plan.get("invariants"), Mapping)
            or multi_anchor_plan["invariants"].get("all_hold") is not True
            or full_solver.get("multi_anchor_plan_sha256")
            != canonical_sha256(multi_anchor_plan)
            or full_solver.get("fit_case_weight_sum") != math.fsum(fit_weights)
            or not math.isclose(
                float(full_solver.get("fit_row_weight_sum", float("nan"))),
                float(fit_row_weights.sum().item()),
                rel_tol=1e-15,
                abs_tol=1e-15,
            )
            or full_solver.get("fit_row_weight_minimum")
            != float(fit_row_weights.min().item())
            or full_solver.get("fit_row_weight_maximum")
            != float(fit_row_weights.max().item())
            or full_solver.get("fit_row_weights_sha256") != expected_weight_sha256
            or full_solver.get("fit_evidence_sha256") != expected_fit_evidence_sha256
            or not all(
                isinstance(full_solver.get(name), str)
                and len(full_solver[name]) == 64
                and all(
                    character in "0123456789abcdef" for character in full_solver[name]
                )
                for name in ("producer_prior_sha256", "consumer_prior_sha256")
            )
            or not all(
                isinstance(producer_bridge.get(name), str)
                and len(producer_bridge[name]) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in producer_bridge[name]
                )
                for name in (
                    "normalized_chart_sha256",
                    "thin_polar_diagnostic_sha256",
                    "structural_decoder_prior_sha256",
                    "decoder_sha256",
                    "correction_sha256",
                    "rowspace_projector_sha256",
                )
            )
        ):
            raise Lfm2SemanticMlpRefoldError(
                "Exact-zero cold-start structural solve did not prove an "
                "authenticated normalized-input W1/W3 bridge, separate raw "
                "residual W2 bridge, and weighted full-source projection."
            )
        full_solver = MappingProxyType(
            {
                **dict(full_solver),
                "cold_start_escape": {
                    "incumbent_w1_w3_w2_exactly_zero": True,
                    "requested_maximum_delta_ratio": (requested_maximum_delta_ratio),
                    "effective_full_structural_maximum_delta_ratio": None,
                    "relative_incumbent_trust_cap_applied": False,
                    "authenticated_full_source_structural_solve_used": True,
                    "authenticated_normalized_input_bridge_used_for_w1_w3": True,
                    "authenticated_raw_residual_chart_used_only_for_w2": True,
                    "producer_consumer_bridge_fields_separate": True,
                    "coupled_real_medoid_gate_up_used": True,
                    "shared_normalized_input_decoder_used": True,
                    "per_neuron_producer_interpolation_used": False,
                    "every_source_multi_anchor_consumer_used": True,
                    "materialized_target_dtype_producers_used_for_w2": True,
                    "random_initialization_used": False,
                    "gradient_training_used": False,
                },
                "source_derived_planting": {
                    "source_inventory_sha256": inventory.inventory_sha256,
                    "compiler_lineage_sha256": inventory.compiler_lineage_sha256,
                    "residual_assignment_sha256": (
                        inventory.residual_assignment_sha256
                    ),
                    "normalization_lineage_sha256": (
                        inventory.normalization_lineage_sha256
                    ),
                    "residual_chart_sha256": expected_chart_sha256,
                    "parent_ffn_norm_weight_sha256": (expected_parent_norm_sha256),
                    "target_ffn_norm_weight_sha256": (expected_target_norm_sha256),
                    "normalized_input_decoder_sha256": (
                        producer_bridge["decoder_sha256"]
                    ),
                    "multi_anchor_plan_sha256": (
                        full_solver["multi_anchor_plan_sha256"]
                    ),
                    "weight_complete_signature_sha256": (
                        full_solver["weight_complete_signature_sha256"]
                    ),
                    "weight_complete_signature_report_sha256": (
                        full_solver["weight_complete_signature_report_sha256"]
                    ),
                    "fold_compute_device": fold_compute_device,
                    "weight_complete_signature_compute_execution": (
                        None
                        if weight_signature_compute_execution is None
                        else dict(weight_signature_compute_execution)
                    ),
                    "weight_complete_signature_used_for_structural_selection": (True),
                    "cached_activation_rows_used_for_structural_selection": False,
                    "cached_activation_rows_used_for_sparse_anchor_fit": False,
                    "producer_prior_sha256": full_solver["producer_prior_sha256"],
                    "consumer_prior_sha256": full_solver["consumer_prior_sha256"],
                    "full_source_coverage": (
                        full_solver.get("full_source_coverage") is True
                    ),
                    "producer_averaging": False,
                    "producer_per_neuron_interpolation": False,
                    "producer_shared_decoder": True,
                    "source_neurons_dropped": 0,
                    "fit_case_weight_sum": full_solver["fit_case_weight_sum"],
                    "fit_row_weight_sum": full_solver["fit_row_weight_sum"],
                    "fit_row_weights_sha256": full_solver["fit_row_weights_sha256"],
                    "heldout_rows_used_by_solver": 0,
                    "planted_roles": list(planted_roles),
                    "all_dense_w1_w3_w2_roles_planted": (
                        planted_roles == ("w1.weight", "w3.weight", "w2.weight")
                    ),
                    "fit_partition_only_used_for_solve": True,
                },
            }
        )
    if qdeim_lane:
        writer_state = MappingProxyType(
            {
                name: value.detach().to(device="cpu").contiguous().clone()
                for name, value in incumbent_state.items()
            }
        )
        writer_solver = MappingProxyType(
            {
                "solver": "not-compiled-unselected-qdeim-lane-placeholder",
                "residual_writer_solve_calls": 0,
                "candidate_selected_or_competed": False,
                "placeholder_state_is_exact_incumbent": True,
                "qdeim_full_candidate_is_only_analytic_lane_candidate": True,
                "heldout_rows_used_by_solver": 0,
                "operation_audit": dict(_ZERO_OPERATIONS),
            }
        )
    else:
        writer_fixed_producer_source = (
            "full-source-structural-w1-w3"
            if exact_zero_incumbent
            else "live-incumbent-w1-w3"
        )
        writer_fixed_producer_state = (
            full_state if exact_zero_incumbent else incumbent_state
        )
        writer_state, writer_solver = _solve_writer(
            partitions["fit"],
            writer_fixed_producer_state,
            ridge=float(ridge),
            maximum_delta_ratio=requested_maximum_delta_ratio,
        )
        writer_solver = MappingProxyType(
            {
                **dict(writer_solver),
                "fixed_producer_state_source": writer_fixed_producer_source,
                "fixed_producer_state_sha256": _state_sha256(
                    writer_fixed_producer_state
                ),
                "blank_compatible_full_source_producer_plant": (exact_zero_incumbent),
                "legacy_candidate_role_name_describes_nonblank_lane": (
                    exact_zero_incumbent
                ),
            }
        )
    if inventory.inventory_sha256 != source_before or not inventory.verify():
        raise Lfm2SemanticMlpRefoldError(
            "Direct-local parent W1/W3/W2 bundle changed during compilation."
        )
    writer_state, writer_structural = _freeze_candidate_proposal_state(
        writer_state,
        incumbent_state,
    )
    full_state, full_structural = _freeze_candidate_proposal_state(
        full_state,
        incumbent_state,
    )
    writer_planting_strength = _planting_strength_diagnostics(
        writer_state,
        incumbent_state,
        requested_maximum_delta_ratio_goal=requested_maximum_delta_ratio,
    )
    full_planting_strength = _planting_strength_diagnostics(
        full_state,
        incumbent_state,
        requested_maximum_delta_ratio_goal=requested_maximum_delta_ratio,
    )
    writer_solver = MappingProxyType(
        {
            **dict(writer_solver),
            "planting_strength_diagnostics": writer_planting_strength,
            "requested_maximum_delta_ratio_goal": requested_maximum_delta_ratio,
            "observed_maximum_delta_ratio": writer_planting_strength[
                "observed_maximum_delta_ratio"
            ],
            "requested_delta_ratio_goal_met": writer_planting_strength[
                "requested_delta_ratio_goal_met"
            ],
            "effective_maximum_delta_ratio_cap": None,
            "delta_ratio_cap_applied": False,
            "numeric_delta_ratio_used_for_candidate_eligibility": False,
            "numeric_delta_ratio_used_for_candidate_ordering": False,
            "numeric_delta_ratio_used_for_publication_or_rollback": False,
            "numeric_delta_ratio_has_final_intelligence_authority": False,
            "candidate_structural_report": dict(writer_structural),
            "candidate_structural_report_sha256": writer_structural[
                "structural_report_sha256"
            ],
            "invalid_proposal_materialized_as_incumbent_placeholder": (
                writer_structural["structural_safety_passed"] is not True
            ),
        }
    )
    full_solver = MappingProxyType(
        {
            **dict(full_solver),
            "planting_strength_diagnostics": full_planting_strength,
            "requested_maximum_delta_ratio_goal": requested_maximum_delta_ratio,
            "observed_maximum_delta_ratio": full_planting_strength[
                "observed_maximum_delta_ratio"
            ],
            "requested_delta_ratio_goal_met": full_planting_strength[
                "requested_delta_ratio_goal_met"
            ],
            "effective_maximum_delta_ratio_cap": None,
            "delta_ratio_cap_applied": False,
            "numeric_delta_ratio_used_for_candidate_eligibility": False,
            "numeric_delta_ratio_used_for_candidate_ordering": False,
            "numeric_delta_ratio_used_for_publication_or_rollback": False,
            "numeric_delta_ratio_has_final_intelligence_authority": False,
            "candidate_structural_report": dict(full_structural),
            "candidate_structural_report_sha256": full_structural[
                "structural_report_sha256"
            ],
            "invalid_proposal_materialized_as_incumbent_placeholder": (
                full_structural["structural_safety_passed"] is not True
            ),
        }
    )
    incumbent_metrics, incumbent_metric_errors = _diagnostic_all_metrics(
        incumbent_state,
        partitions,
    )
    writer_metrics, writer_metric_errors = _diagnostic_all_metrics(
        writer_state,
        partitions,
    )
    full_metrics, full_metric_errors = _diagnostic_all_metrics(
        full_state,
        partitions,
    )
    metric_errors = {
        "incumbent": incumbent_metric_errors,
        "incumbent-producer-w2": writer_metric_errors,
        "full-w1-w3-w2": full_metric_errors,
    }
    incumbent = Lfm2FrozenSemanticMlpCandidate(
        role="incumbent",
        gate_weight=incumbent_state["w1.weight"],
        up_weight=incumbent_state["w3.weight"],
        down_weight=incumbent_state["w2.weight"],
        metrics=incumbent_metrics,
        solver_receipt={
            "solver": "none-live-dense-c61-incumbent",
            "operation_audit": dict(_ZERO_OPERATIONS),
        },
    )
    writer = Lfm2FrozenSemanticMlpCandidate(
        role="incumbent-producer-w2",
        gate_weight=writer_state["w1.weight"],
        up_weight=writer_state["w3.weight"],
        down_weight=writer_state["w2.weight"],
        metrics=writer_metrics,
        solver_receipt=writer_solver,
    )
    full = Lfm2FrozenSemanticMlpCandidate(
        role="full-w1-w3-w2",
        gate_weight=full_state["w1.weight"],
        up_weight=full_state["w3.weight"],
        down_weight=full_state["w2.weight"],
        metrics=full_metrics,
        solver_receipt=full_solver,
    )
    heldout = ("fit_check", "selection", "validation")
    heldout_nonregression = {
        candidate.role: {
            name: _nonregressing(
                candidate.metrics[name],
                incumbent.metrics[name],
                rmse_relative_tolerance=float(rmse_relative_tolerance),
                cosine_tolerance=float(cosine_tolerance),
            )
            for name in heldout
        }
        for candidate in (incumbent, writer, full)
    }
    selection_improvement = {
        candidate.role: _strictly_improving(
            candidate.metrics["selection"],
            incumbent.metrics["selection"],
            rmse_relative_tolerance=float(rmse_relative_tolerance),
            cosine_tolerance=float(cosine_tolerance),
        )
        for candidate in (writer, full)
    }
    writer_selection_vs_full = _nonregressing(
        writer.metrics["selection"],
        full.metrics["selection"],
        rmse_relative_tolerance=float(rmse_relative_tolerance),
        cosine_tolerance=float(cosine_tolerance),
    )
    structural_reports = {
        "incumbent": _candidate_state_structural_report(
            incumbent_state,
            incumbent_state,
        ),
        writer.role: writer_structural,
        full.role: full_structural,
    }
    planting_strength_diagnostics = {
        "incumbent": _planting_strength_diagnostics(
            incumbent_state,
            incumbent_state,
            requested_maximum_delta_ratio_goal=requested_maximum_delta_ratio,
        ),
        writer.role: writer_planting_strength,
        full.role: full_planting_strength,
    }
    fit_ordering_keys = {
        candidate.role: (
            list(_fit_metric_ordering_key(candidate))
            if structural_reports[candidate.role]["structurally_eligible"] is True
            else None
        )
        for candidate in (writer, full)
    }
    analytic_frontier = tuple(
        candidate
        for candidate in (writer, full)
        if structural_reports[candidate.role]["structurally_eligible"] is True
    )
    if exact_zero_incumbent:
        if not any(candidate is full for candidate in analytic_frontier):
            raise Lfm2SemanticMlpRefoldError(
                "The compulsory cold-start full W1/W3/W2 proposal is not finite, "
                "dense, target-shaped, target-dtype, and effectful."
            )
        if mlp_lane in _QDEIM_MLP_LANE_TO_CANDIDATE:
            selected = full
            rationale = (
                "blank-target-authenticated-two-anchor-tangent-qdeim-lane-"
                "continues-to-whole-response-evaluation"
            )
        elif mlp_lane == "uncapped_w2_residual_writer_priority" and any(
            candidate is writer for candidate in analytic_frontier
        ):
            selected = writer
            rationale = (
                "blank-target-full-source-producers-plus-w2-residual-writer-"
                "continues-to-whole-response-evaluation"
            )
        else:
            selected = full
            rationale = (
                "blank-target-complete-structural-plant-continues-to-whole-"
                "response-evaluation"
            )
    elif recipe.fit_metrics_order_candidates and analytic_frontier:
        selected = min(
            analytic_frontier,
            key=_fit_metric_ordering_key,
        )
        rationale = (
            "deterministic-fit-order-among-structurally-eligible-effectful-"
            "analytic-proposals"
        )
    elif mlp_lane == "full_source_w1_w3_w2_structural_priority" and analytic_frontier:
        selected = (
            full
            if any(candidate is full for candidate in analytic_frontier)
            else writer
        )
        rationale = (
            "recipe-full-source-w1-w3-w2-structural-priority-without-numeric-ordering"
        )
    elif mlp_lane == "uncapped_w2_residual_writer_priority" and analytic_frontier:
        selected = (
            writer
            if any(candidate is writer for candidate in analytic_frontier)
            else full
        )
        rationale = (
            "recipe-uncapped-w2-residual-writer-priority-without-numeric-ordering"
        )
    else:
        selected = incumbent
        rationale = "all-analytic-proposals-structurally-invalid-or-identical"
    prevalidation_selected_role = selected.role
    validation_rejected_role: CandidateRole | None = None
    candidate_reports = {
        candidate.role: {
            "analytic_candidate": candidate.role != "incumbent",
            "incumbent_noop_control": candidate.role == "incumbent",
            "candidate_sha256": candidate.candidate_sha256,
            "structural_report": dict(structural_reports[candidate.role]),
            "structural_report_sha256": structural_reports[candidate.role][
                "structural_report_sha256"
            ],
            "structural_safety_passed": structural_reports[candidate.role][
                "structural_safety_passed"
            ],
            "effectful_nonincumbent": structural_reports[candidate.role][
                "effectful_nonincumbent"
            ],
            "structurally_eligible": (
                candidate.role != "incumbent"
                and structural_reports[candidate.role]["structurally_eligible"] is True
            ),
            "frozen_candidate_uses_incumbent_placeholder": (
                candidate.role != "incumbent"
                and structural_reports[candidate.role]["structural_safety_passed"]
                is not True
            ),
            "partition_metrics_represent_original_analytic_proposal": (
                candidate.role == "incumbent"
                or structural_reports[candidate.role]["structural_safety_passed"]
                is True
            ),
            "fit_ordering_key": fit_ordering_keys.get(candidate.role),
            "planting_strength_diagnostics": planting_strength_diagnostics[
                candidate.role
            ],
            "partition_metric_errors": dict(metric_errors[candidate.role]),
            "metric_evaluation_failure_can_exclude_candidate": False,
            "partition_nonregression_diagnostics": heldout_nonregression[
                candidate.role
            ],
            "numeric_diagnostic_miss": any(
                value is not True
                for value in heldout_nonregression[candidate.role].values()
            ),
            "numeric_diagnostic_miss_can_exclude_candidate": False,
            "numeric_metrics_have_final_intelligence_authority": False,
        }
        for candidate in (incumbent, writer, full)
    }
    selection_body = {
        "body_recipe": recipe.as_dict(),
        "body_recipe_sha256": recipe.recipe_sha256,
        "body_recipe_id": recipe.recipe_id,
        "recipe_component_lane": recipe.component_lanes["mlp"],
        "selected_role": selected.role,
        "prevalidation_selected_role": prevalidation_selected_role,
        "validation_rejected_role": validation_rejected_role,
        "rationale": rationale,
        "writer_changed_roles": list(_changed_roles(incumbent_state, writer_state)),
        "full_changed_roles": list(_changed_roles(incumbent_state, full_state)),
        "heldout_nonregression": heldout_nonregression,
        "heldout_selection_improvement": selection_improvement,
        "writer_selection_nonregressing_versus_full": writer_selection_vs_full,
        "candidate_reports": candidate_reports,
        "fit_ordering_keys": fit_ordering_keys,
        "planting_strength_diagnostics": planting_strength_diagnostics,
        "requested_maximum_delta_ratio_goal": requested_maximum_delta_ratio,
        "effective_maximum_delta_ratio_cap": None,
        "delta_ratio_cap_applied": False,
        "delta_ratio_used_to_scale_candidate": False,
        "delta_ratio_diagnostic_only": True,
        "numeric_delta_ratio_used_for_candidate_eligibility": False,
        "numeric_delta_ratio_used_for_candidate_ordering": False,
        "numeric_delta_ratio_used_for_publication_or_rollback": False,
        "numeric_delta_ratio_has_final_intelligence_authority": False,
        "partition_metric_errors": {
            role: dict(errors) for role, errors in metric_errors.items()
        },
        "invalid_proposal_metric_placeholder_roles": [
            candidate.role
            for candidate in (writer, full)
            if structural_reports[candidate.role]["structural_safety_passed"]
            is not True
        ],
        "incumbent_noop_candidate_competed": False,
        "incumbent_noop_fallback_used": selected.role == "incumbent",
        "incumbent_noop_fallback_reason": (
            "all-analytic-proposals-structurally-invalid-or-identical"
            if selected.role == "incumbent"
            else None
        ),
        "incumbent_selected_by_heldout_gates": False,
        "component_quality_gates_block_installation": False,
        "blank_target_requires_complete_structural_plant": exact_zero_incumbent,
        "blank_target_continues_to_whole_response_evaluation": (exact_zero_incumbent),
        "blank_target_selected_state_uses_full_source_structural_producers": (
            exact_zero_incumbent
            and selected.role != "incumbent"
            and mlp_lane not in _QDEIM_MLP_LANE_TO_CANDIDATE
        ),
        "blank_target_residual_writer_component_equivalent_to_full_structural": (
            exact_zero_incumbent
            and not qdeim_lane
            and structural_reports[writer.role]["structurally_eligible"] is not True
        ),
        "candidate_frozen_before_fit_check": True,
        "candidate_frozen_before_selection": True,
        "candidate_frozen_before_validation": True,
        "validation_triggered_refit": False,
        "validation_used_for_candidate_choice": False,
        "validation_used_only_for_acceptance_or_incumbent_fallback": False,
        "validation_publication_veto_applied": False,
        "rollback_authorized": False,
        "heldout_quality_retained_as_diagnostics": True,
        "heldout_metrics_are_diagnostic_only": True,
        "fit_check_metrics_diagnostic_only": True,
        "selection_metrics_diagnostic_only": True,
        "validation_metrics_diagnostic_only": True,
        "fit_partition_only_used_for_solves": True,
        "heldout_partitions_used_for_solves": False,
        "candidate_selection_evidence_scope": (
            "finite-dense-structural-eligibility-plus-fit-ordering"
            if recipe.fit_metrics_order_candidates
            else "finite-dense-structural-eligibility-plus-recipe-priority"
        ),
        "candidate_eligibility_basis": _CANDIDATE_ELIGIBILITY_BASIS,
        "structural_safety_is_only_candidate_veto": True,
        "structurally_eligible_candidate_count": len(analytic_frontier),
        "fit_metrics_role": (
            "ordering-among-structurally-eligible-not-threshold-gating"
            if recipe.fit_metrics_order_candidates
            else "diagnostic-only"
        ),
        "fit_metrics_used_for_candidate_ordering": (
            recipe.fit_metrics_order_candidates
        ),
        "fit_metrics_used_for_pre_review_candidate_selection": (
            recipe.fit_metrics_order_candidates
        ),
        "fit_metrics_used_for_structural_eligibility": False,
        "fit_thresholds_used_for_candidate_eligibility": False,
        "fit_check_thresholds_used_for_candidate_eligibility": False,
        "selection_thresholds_used_for_candidate_eligibility": False,
        "validation_thresholds_used_for_publication": False,
        "numeric_metrics_used_for_candidate_eligibility": False,
        "numeric_thresholds_used_for_candidate_eligibility": False,
        "numeric_thresholds_used_for_publication": False,
        "numeric_metric_used_for_publication_or_rollback": False,
        "numeric_diagnostic_miss_can_exclude_structural_candidate": False,
        "metric_evaluation_failure_can_exclude_structural_candidate": False,
        "numeric_metrics_have_final_intelligence_authority": False,
        "behavioral_intelligence_review_outside_mlp_compiler": True,
        "whole_response_behavioral_intelligence_review_required": True,
        "intelligence_retention_target_is_review_goal_not_numeric_gate": True,
        "endpoint_metrics": {
            candidate.role: {
                name: candidate.metrics[name].as_dict() for name in _PARTITIONS
            }
            for candidate in (incumbent, writer, full)
        },
    }
    if mlp_lane in _QDEIM_MLP_LANE_TO_CANDIDATE:
        selection_body.update(
            {
                "blank_target_selected_state_uses_real_qdeim_producers": (
                    exact_zero_incumbent and selected.role == "full-w1-w3-w2"
                ),
                "whole_decoded_responses_are_primary_semantic_evidence": True,
                "binary_manual_gate_created": False,
                "legacy_residual_writer_solve_performed": False,
                "legacy_residual_writer_candidate_competed": False,
                "qdeim_full_candidate_is_only_analytic_lane_candidate": True,
            }
        )
    if exact_zero_incumbent:
        selection_body["cold_start_escape"] = dict(
            full.solver_receipt["cold_start_escape"]
        )
        selection_body["source_derived_planting"] = dict(
            full.solver_receipt["source_derived_planting"]
        )
    if fingerprint_lfm2_parent(target_model).identity_sha256 != current_state:
        raise Lfm2SemanticMlpRefoldError(
            "Candidate preparation mutated the live child."
        )
    if exact_zero_incumbent and (
        selected.role not in {"incumbent-producer-w2", "full-w1-w3-w2"}
        or _nonzero_swiglu_roles(selected.state_dict())
        != ("w1.weight", "w3.weight", "w2.weight")
    ):
        raise Lfm2SemanticMlpRefoldError(
            "Exact-zero cold start did not produce one complete nonzero "
            "structural W1/W3/W2 plant."
        )
    if not heldout_nonregression["incumbent"]["validation"]:
        raise Lfm2SemanticMlpRefoldError(
            "The authenticated incumbent did not exactly replay VALIDATION."
        )
    terminal = _terminal_banks(
        partitions,
        required=require_terminal_action_evidence,
    )
    selection = MappingProxyType(selection_body)
    return Lfm2PreparedSemanticMlpRefold(
        layer_index=layer_index,
        captured_child_model_state_sha256=current_state,
        target_topology_sha256=topology,
        target_parameter_count=parameter_count,
        source_inventory_sha256=inventory.inventory_sha256,
        compiler_lineage_sha256=inventory.compiler_lineage_sha256,
        cache_key=cache,
        cache_manifest_sha256=manifest,
        bridge_artifact_sha256=bridge,
        evidence_sha256=tuple(item.evidence_sha256 for item in raw),
        partition_batch_indices={
            name: tuple(sorted({item.batch_index for item in partitions[name]}))
            for name in _PARTITIONS
        },
        incumbent=incumbent,
        writer_candidate=writer,
        full_candidate=full,
        body_recipe=recipe,
        selected_role=selected.role,
        selection_receipt=selection,
        terminal_action_evidence=terminal,
    )


def _copy_candidate(parameter: Any, value: Any) -> None:
    """Single testable mutation point; rollback deliberately bypasses it."""

    parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def _metric_guidance_receipt_fields(
    prepared: Lfm2PreparedSemanticMlpRefold,
) -> dict[str, Any]:
    """Project the authenticated structural-only selection policy into install."""

    selection = prepared.selection_receipt
    selected_nonregression = selection["heldout_nonregression"][prepared.selected_role]
    return {
        "body_recipe": prepared.body_recipe.as_dict(),
        "body_recipe_sha256": prepared.body_recipe.recipe_sha256,
        "body_recipe_id": prepared.body_recipe.recipe_id,
        "recipe_component_lane": prepared.body_recipe.component_lanes["mlp"],
        "incumbent_noop_fallback_used": prepared.selected_role == "incumbent",
        "incumbent_noop_fallback_reason": selection["incumbent_noop_fallback_reason"],
        "incumbent_selected_by_heldout_gates": False,
        "fit_check_nonregression_passed": bool(selected_nonregression["fit_check"]),
        "selection_nonregression_passed": bool(selected_nonregression["selection"]),
        "validation_nonregression_passed": bool(selected_nonregression["validation"]),
        "validation_diagnostic_would_have_failed_legacy_gate": (
            selected_nonregression["validation"] is not True
        ),
        "candidate_selection_evidence_scope": selection[
            "candidate_selection_evidence_scope"
        ],
        "candidate_eligibility_basis": selection["candidate_eligibility_basis"],
        "structural_safety_is_only_candidate_veto": True,
        "structurally_eligible_candidate_count": selection[
            "structurally_eligible_candidate_count"
        ],
        "fit_metrics_role": selection["fit_metrics_role"],
        "fit_metrics_used_for_candidate_ordering": selection[
            "fit_metrics_used_for_candidate_ordering"
        ],
        "fit_metrics_used_for_pre_review_candidate_selection": selection[
            "fit_metrics_used_for_pre_review_candidate_selection"
        ],
        "fit_metrics_used_for_structural_eligibility": False,
        "fit_thresholds_used_for_candidate_eligibility": False,
        "fit_check_thresholds_used_for_candidate_eligibility": False,
        "selection_thresholds_used_for_candidate_eligibility": False,
        "validation_thresholds_used_for_publication": False,
        "fit_check_metrics_diagnostic_only": True,
        "selection_metrics_diagnostic_only": True,
        "validation_metrics_diagnostic_only": True,
        "heldout_metrics_are_diagnostic_only": True,
        "numeric_metrics_used_for_candidate_eligibility": False,
        "numeric_thresholds_used_for_candidate_eligibility": False,
        "numeric_thresholds_used_for_publication": False,
        "numeric_metric_used_for_publication_or_rollback": False,
        "planting_strength_diagnostics": dict(
            selection["planting_strength_diagnostics"]
        ),
        "selected_candidate_planting_strength_diagnostics": dict(
            selection["planting_strength_diagnostics"][prepared.selected_role]
        ),
        "requested_maximum_delta_ratio_goal": selection[
            "requested_maximum_delta_ratio_goal"
        ],
        "effective_maximum_delta_ratio_cap": None,
        "delta_ratio_cap_applied": False,
        "delta_ratio_used_to_scale_candidate": False,
        "delta_ratio_diagnostic_only": True,
        "numeric_delta_ratio_used_for_candidate_eligibility": False,
        "numeric_delta_ratio_used_for_candidate_ordering": False,
        "numeric_delta_ratio_used_for_publication_or_rollback": False,
        "numeric_delta_ratio_has_final_intelligence_authority": False,
        "numeric_diagnostic_miss_can_exclude_structural_candidate": False,
        "metric_evaluation_failure_can_exclude_structural_candidate": False,
        "validation_publication_veto_applied": False,
        "validation_rejected_candidate": None,
        "rollback_authorized": False,
        "numeric_metrics_have_final_intelligence_authority": False,
        "behavioral_intelligence_review_outside_mlp_compiler": True,
        "whole_response_behavioral_intelligence_review_required": True,
        "intelligence_retention_target_is_review_goal_not_numeric_gate": True,
    }


def _analytic_construction_receipt_fields(
    prepared: Lfm2PreparedSemanticMlpRefold,
) -> dict[str, Any]:
    """Describe the lane's closed-form construction without changing old receipts."""

    if prepared.body_recipe.component_lanes["mlp"] in _QDEIM_MLP_LANE_TO_CANDIDATE:
        planting = prepared.selection_receipt.get("source_derived_planting", {})
        compute_device = (
            planting.get("qdeim_pod_and_pivot_compute_device")
            if isinstance(planting, Mapping)
            else None
        )
        if not isinstance(compute_device, str) or not compute_device:
            raise Lfm2SemanticMlpRefoldError(
                "Q-DEIM installation lacks its authenticated algebra device."
            )
        return {
            "analytic_linear_and_cluster_solves_only": False,
            "analytic_linear_cluster_tangent_and_qdeim_solves_only": True,
            "qdeim_pod_and_pivot_compute_device": compute_device,
            "qdeim_numeric_diagnostics_are_advisory_only": True,
            "qdeim_numeric_thresholds_can_accept_veto_or_block_progression": False,
        }
    return {"analytic_linear_and_cluster_solves_only": True}


def _deferred_qdeim_w2_prior_receipt_fields(
    prepared: Lfm2PreparedSemanticMlpRefold,
    *,
    defer_down_weight: bool,
) -> dict[str, Any]:
    """Bind the selected Q-DEIM W2 without installing it before the terminal."""

    if (
        not defer_down_weight
        or prepared.body_recipe.component_lanes["mlp"]
        not in _QDEIM_MLP_LANE_TO_CANDIDATE
    ):
        return {}
    prior = prepared.selected_candidate.down_weight
    return {
        "deferred_selected_w2_prior_sha256": _tensor_sha256(prior),
        "deferred_selected_w2_prior_terminal_action_sha256": (
            lfm2_cpcf_terminal_action_tensor_sha256(prior)
        ),
        "deferred_selected_w2_prior_candidate_sha256": (
            prepared.selected_candidate.candidate_sha256
        ),
        "deferred_selected_w2_prior_selected_role": prepared.selected_role,
        "deferred_selected_w2_prior_qdeim_candidate_id": (
            prepared.selected_candidate.solver_receipt["selected_qdeim_candidate_id"]
        ),
        "deferred_selected_w2_prior_carried_to_terminal": True,
        "deferred_selected_w2_prior_installed_before_terminal": False,
        "deferred_selected_w2_prior_terminal_frontier_role": (
            "authenticated-terminal-seed-prior"
        ),
    }


def _build_deferred_qdeim_w2_prior(
    prepared: Lfm2PreparedSemanticMlpRefold,
    *,
    defer_down_weight: bool,
    receipt_body: Mapping[str, Any],
    live_deferred_w2: Any,
) -> Lfm2CpcfTerminalActionDownWeightPrior | None:
    """Own and authenticate the selected W2 against the unsealed install core."""

    if (
        not defer_down_weight
        or prepared.body_recipe.component_lanes["mlp"]
        not in _QDEIM_MLP_LANE_TO_CANDIDATE
    ):
        return None
    return Lfm2CpcfTerminalActionDownWeightPrior(
        down_weight=prepared.selected_candidate.down_weight,
        body_recipe_id=prepared.body_recipe.recipe_id,
        body_recipe_sha256=prepared.body_recipe.recipe_sha256,
        mlp_lane=prepared.body_recipe.component_lanes["mlp"],
        selected_role=prepared.selected_role,
        selected_candidate_sha256=(prepared.selected_candidate.candidate_sha256),
        preparation_sha256=prepared.preparation_sha256,
        mlp_installation_core_sha256=canonical_sha256(dict(receipt_body)),
        live_deferred_w2_sha256=lfm2_cpcf_terminal_action_tensor_sha256(
            live_deferred_w2
        ),
    )


def install_lfm2_prepared_semantic_mlp_postfold(
    *,
    target_model: Any,
    prepared: Lfm2PreparedSemanticMlpRefold,
    defer_down_weight: bool = False,
) -> Lfm2SemanticMlpPostfoldResult:
    """Transactionally install a frozen candidate or defer the shared final W2."""

    torch = _torch()
    if not isinstance(prepared, Lfm2PreparedSemanticMlpRefold) or not prepared.verify():
        raise Lfm2SemanticMlpRefoldError("Prepared semantic MLP refold is invalid.")
    current = fingerprint_lfm2_parent(target_model).identity_sha256
    if (
        current != prepared.captured_child_model_state_sha256
        or _topology_sha256(target_model) != prepared.target_topology_sha256
        or _parameter_count(target_model) != prepared.target_parameter_count
    ):
        raise Lfm2SemanticMlpRefoldError(
            "Prepared candidate is stale; sequential child replay must be refreshed."
        )
    parameters = _parameter_bundle(target_model, prepared.layer_index)
    prefix = f"model.layers.{prepared.layer_index}.feed_forward."
    incumbent = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in parameters.items()
    }
    identities = {name: id(value) for name, value in parameters.items()}
    pointers = {name: int(value.data_ptr()) for name, value in parameters.items()}
    shapes = {name: tuple(value.shape) for name, value in parameters.items()}
    dtypes = {name: value.dtype for name, value in parameters.items()}
    requires_grad = {
        name: bool(value.requires_grad) for name, value in parameters.items()
    }
    gradient_ids = {
        name: None if value.grad is None else id(value.grad)
        for name, value in parameters.items()
    }
    gradient_hashes = {
        name: None if value.grad is None else _tensor_sha256(value.grad)
        for name, value in parameters.items()
    }
    state = prepared.selected_candidate.state_dict()
    install_roles = (
        ("w1.weight", "w3.weight")
        if defer_down_weight
        else (
            "w1.weight",
            "w3.weight",
            "w2.weight",
        )
    )
    selected_state_identical_to_incumbent = all(
        torch.equal(state[role], incumbent[prefix + role])
        for role in ("w1.weight", "w3.weight", "w2.weight")
    )
    # The layer-stack transaction computes live state diffs from a lexically
    # canonicalized state dict.  Match that ordering here so a full blank
    # W1/W3/W2 plant cannot publish a semantically identical but differently
    # ordered mutation inventory.
    expected_changed = tuple(
        sorted(
            role
            for role in install_roles
            if not torch.equal(
                state[role],
                incumbent[prefix + role],
            )
        )
    )
    if not expected_changed:
        if defer_down_weight and set(prepared.terminal_action_evidence) != set(
            _PARTITIONS
        ):
            raise Lfm2SemanticMlpRefoldError(
                "A fully deferred W2 requires all four terminal-action banks."
            )
        if (
            fingerprint_lfm2_parent(target_model).identity_sha256 != current
            or _topology_sha256(target_model) != prepared.target_topology_sha256
            or _parameter_count(target_model) != prepared.target_parameter_count
            or any(
                not torch.equal(parameter.detach().cpu(), incumbent[name])
                for name, parameter in parameters.items()
            )
            or any(id(parameters[name]) != identities[name] for name in parameters)
            or any(
                int(parameters[name].data_ptr()) != pointers[name]
                for name in parameters
            )
            or any(tuple(parameters[name].shape) != shapes[name] for name in parameters)
            or any(parameters[name].dtype != dtypes[name] for name in parameters)
            or any(
                bool(parameters[name].requires_grad) != requires_grad[name]
                for name in parameters
            )
            or any(
                (None if parameters[name].grad is None else id(parameters[name].grad))
                != gradient_ids[name]
                for name in parameters
            )
            or any(
                parameters[name].grad is not None
                and _tensor_sha256(parameters[name].grad) != gradient_hashes[name]
                for name in parameters
            )
        ):
            raise Lfm2SemanticMlpRefoldError(
                "Semantic MLP no-op changed state, topology, identity, or gradients."
            )
        receipt_body = {
            "schema": LFM2_SEMANTIC_MLP_INSTALLATION_SCHEMA,
            "status": (
                "prepared-final-w2-deferred"
                if defer_down_weight
                else _STRUCTURAL_NOOP_STATUS
            ),
            "installed": False,
            "layer_index": prepared.layer_index,
            "preparation_sha256": prepared.preparation_sha256,
            "selected_role": prepared.selected_role,
            "selected_candidate_sha256": (prepared.selected_candidate.candidate_sha256),
            "model_state_before_sha256": current,
            "model_state_after_sha256": current,
            "target_topology_sha256": prepared.target_topology_sha256,
            "parameter_count_before": prepared.target_parameter_count,
            "parameter_count_after": prepared.target_parameter_count,
            "parameter_count_changed": False,
            "topology_changed": False,
            "parameter_identities_preserved": True,
            "parameter_storage_preserved": True,
            "gradient_state_preserved": True,
            "exact_noop_state_proven": True,
            "install_count": 0,
            "changed_parameter_roles": [],
            "selected_install_roles_identical_to_live_state": True,
            "selected_candidate": (
                "incumbent-noop"
                if prepared.selected_role == "incumbent"
                else prepared.selected_role
            ),
            "selected_candidate_state_identical_to_incumbent": (
                selected_state_identical_to_incumbent
            ),
            **_metric_guidance_receipt_fields(prepared),
            "final_layer_mlp_w2_deferred_to_fused_terminal": defer_down_weight,
            "downstream_terminal_replay_and_validation_required": defer_down_weight,
            "terminal_action_evidence_sha256": {
                name: value.authentication_sha256
                for name, value in prepared.terminal_action_evidence.items()
            },
            "source_inventory_sha256": prepared.source_inventory_sha256,
            "compiler_lineage_sha256": prepared.compiler_lineage_sha256,
            "cache_key": prepared.cache_key,
            "cache_manifest_sha256": prepared.cache_manifest_sha256,
            "bridge_artifact_sha256": prepared.bridge_artifact_sha256,
            "evidence_sha256": list(prepared.evidence_sha256),
            "evidence_endpoint_count": len(prepared.evidence_sha256),
            "parent_exact_endpoint_count": len(prepared.evidence_sha256),
            "partition_batch_indices": {
                name: list(prepared.partition_batch_indices[name])
                for name in _PARTITIONS
            },
            "candidate_frozen_before_fit_check": True,
            "candidate_frozen_before_selection": True,
            "candidate_frozen_before_validation": True,
            "validation_triggered_refit": False,
            "inserted_target_rows_used": 0,
            "query_scope": "one-parent-exact-semantic-endpoint-per-phase",
            "numeric_parent_child_weight_match_objective": False,
            **_analytic_construction_receipt_fields(prepared),
            **_deferred_qdeim_w2_prior_receipt_fields(
                prepared,
                defer_down_weight=defer_down_weight,
            ),
            "sequential_replay_freshness_enforced": True,
            **(
                {
                    "cold_start_escape": dict(
                        prepared.selection_receipt["cold_start_escape"]
                    ),
                    "source_derived_planting": dict(
                        prepared.selection_receipt["source_derived_planting"]
                    ),
                }
                if "cold_start_escape" in prepared.selection_receipt
                else {}
            ),
            "operation_audit": dict(_ZERO_OPERATIONS),
        }
        deferred_prior = _build_deferred_qdeim_w2_prior(
            prepared,
            defer_down_weight=defer_down_weight,
            receipt_body=receipt_body,
            live_deferred_w2=incumbent[prefix + "w2.weight"],
        )
        if deferred_prior is not None:
            receipt_body.update(
                {
                    "deferred_selected_w2_prior_authentication_sha256": (
                        deferred_prior.authentication_sha256
                    ),
                    "deferred_selected_w2_prior_mlp_installation_core_sha256": (
                        deferred_prior.mlp_installation_core_sha256
                    ),
                }
            )
        receipt = MappingProxyType(
            {
                **receipt_body,
                "result_sha256": canonical_sha256(receipt_body),
            }
        )
        return Lfm2SemanticMlpPostfoldResult(
            receipt=receipt,
            prepared=prepared,
            final_layer_mlp_evidence=(
                prepared.terminal_action_evidence if defer_down_weight else {}
            ),
            deferred_selected_w2_prior=deferred_prior,
        )
    try:
        with torch.no_grad():
            for role in install_roles:
                parameter = parameters[prefix + role]
                candidate = state[role]
                if (
                    tuple(candidate.shape) != tuple(parameter.shape)
                    or candidate.dtype != parameter.dtype
                ):
                    raise Lfm2SemanticMlpRefoldError(
                        f"Frozen candidate changed the topology of {role}."
                    )
                _copy_candidate(parameter, candidate)
        after = fingerprint_lfm2_parent(target_model).identity_sha256
        changed = tuple(
            sorted(
                name.removeprefix(prefix)
                for name, parameter in parameters.items()
                if not torch.equal(parameter.detach().cpu(), incumbent[name])
            )
        )
        if (
            tuple(changed) != expected_changed
            or after == current
            or _parameter_count(target_model) != prepared.target_parameter_count
            or _topology_sha256(target_model) != prepared.target_topology_sha256
            or any(id(parameters[name]) != identities[name] for name in parameters)
            or any(
                int(parameters[name].data_ptr()) != pointers[name]
                for name in parameters
            )
            or any(tuple(parameters[name].shape) != shapes[name] for name in parameters)
            or any(parameters[name].dtype != dtypes[name] for name in parameters)
            or any(
                bool(parameters[name].requires_grad) != requires_grad[name]
                for name in parameters
            )
            or any(
                (None if parameters[name].grad is None else id(parameters[name].grad))
                != gradient_ids[name]
                for name in parameters
            )
            or any(
                parameters[name].grad is not None
                and _tensor_sha256(parameters[name].grad) != gradient_hashes[name]
                for name in parameters
            )
        ):
            raise Lfm2SemanticMlpRefoldError(
                "Semantic MLP installation changed topology, identity, or gradient state."
            )
        if defer_down_weight and set(prepared.terminal_action_evidence) != set(
            _PARTITIONS
        ):
            raise Lfm2SemanticMlpRefoldError(
                "Deferred final W2 has no complete terminal-action evidence bank."
            )
        receipt_body = {
            "schema": LFM2_SEMANTIC_MLP_INSTALLATION_SCHEMA,
            "status": (
                "installed-w1-w3-final-w2-deferred"
                if defer_down_weight
                else "installed-frozen-semantic-mlp-candidate"
            ),
            "installed": True,
            "layer_index": prepared.layer_index,
            "preparation_sha256": prepared.preparation_sha256,
            "selected_role": prepared.selected_role,
            "selected_candidate_sha256": prepared.selected_candidate.candidate_sha256,
            "model_state_before_sha256": current,
            "model_state_after_sha256": after,
            "target_topology_sha256": prepared.target_topology_sha256,
            "parameter_count_before": prepared.target_parameter_count,
            "parameter_count_after": prepared.target_parameter_count,
            "parameter_count_changed": False,
            "topology_changed": False,
            "parameter_identities_preserved": True,
            "parameter_storage_preserved": True,
            "gradient_state_preserved": True,
            "changed_parameter_roles": list(changed),
            "install_count": 1,
            "selected_candidate": prepared.selected_role,
            "selected_candidate_state_identical_to_incumbent": False,
            **_metric_guidance_receipt_fields(prepared),
            "final_layer_mlp_w2_deferred_to_fused_terminal": defer_down_weight,
            "downstream_terminal_replay_and_validation_required": defer_down_weight,
            "terminal_action_evidence_sha256": {
                name: value.authentication_sha256
                for name, value in prepared.terminal_action_evidence.items()
            },
            "source_inventory_sha256": prepared.source_inventory_sha256,
            "compiler_lineage_sha256": prepared.compiler_lineage_sha256,
            "cache_key": prepared.cache_key,
            "cache_manifest_sha256": prepared.cache_manifest_sha256,
            "bridge_artifact_sha256": prepared.bridge_artifact_sha256,
            "evidence_sha256": list(prepared.evidence_sha256),
            "evidence_endpoint_count": len(prepared.evidence_sha256),
            "parent_exact_endpoint_count": len(prepared.evidence_sha256),
            "partition_batch_indices": {
                name: list(prepared.partition_batch_indices[name])
                for name in _PARTITIONS
            },
            "candidate_frozen_before_fit_check": True,
            "candidate_frozen_before_selection": True,
            "candidate_frozen_before_validation": True,
            "validation_triggered_refit": False,
            "inserted_target_rows_used": 0,
            "query_scope": "one-parent-exact-semantic-endpoint-per-phase",
            "candidate_scope": (
                "authenticated-incumbent-noop-and-incumbent-producer-dense-w2-"
                "residual-writer-and-two-anchor-tangent-dense-qdeim-w1-w3-w2"
                if prepared.body_recipe.component_lanes["mlp"]
                in _QDEIM_MLP_LANE_TO_CANDIDATE
                else "authenticated-incumbent-noop-and-incumbent-producer-dense-"
                "w2-residual-writer-and-full-source-clustered-dense-w1-w3-w2"
            ),
            "numeric_parent_child_weight_match_objective": False,
            **_analytic_construction_receipt_fields(prepared),
            **_deferred_qdeim_w2_prior_receipt_fields(
                prepared,
                defer_down_weight=defer_down_weight,
            ),
            "sequential_replay_freshness_enforced": True,
            **(
                {
                    "cold_start_escape": dict(
                        prepared.selection_receipt["cold_start_escape"]
                    ),
                    "source_derived_planting": dict(
                        prepared.selection_receipt["source_derived_planting"]
                    ),
                }
                if "cold_start_escape" in prepared.selection_receipt
                else {}
            ),
            "operation_audit": dict(_ZERO_OPERATIONS),
        }
        deferred_prior = _build_deferred_qdeim_w2_prior(
            prepared,
            defer_down_weight=defer_down_weight,
            receipt_body=receipt_body,
            live_deferred_w2=incumbent[prefix + "w2.weight"],
        )
        if deferred_prior is not None:
            receipt_body.update(
                {
                    "deferred_selected_w2_prior_authentication_sha256": (
                        deferred_prior.authentication_sha256
                    ),
                    "deferred_selected_w2_prior_mlp_installation_core_sha256": (
                        deferred_prior.mlp_installation_core_sha256
                    ),
                }
            )
        receipt = MappingProxyType(
            {
                **receipt_body,
                "result_sha256": canonical_sha256(receipt_body),
            }
        )
        return Lfm2SemanticMlpPostfoldResult(
            receipt=receipt,
            prepared=prepared,
            final_layer_mlp_evidence=(
                prepared.terminal_action_evidence if defer_down_weight else {}
            ),
            deferred_selected_w2_prior=deferred_prior,
        )
    except BaseException:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(
                    incumbent[name].to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )
        if (
            fingerprint_lfm2_parent(target_model).identity_sha256 != current
            or _parameter_count(target_model) != prepared.target_parameter_count
            or _topology_sha256(target_model) != prepared.target_topology_sha256
        ):
            raise Lfm2SemanticMlpRefoldError(
                "Semantic MLP installation failed and exact rollback was not proven."
            )
        raise


def compile_install_lfm2_semantic_mlp_postfold(
    *,
    target_model: Any,
    layer_index: int,
    evidence: Sequence[Lfm2SemanticMlpEndpointEvidence],
    cache_key: str,
    cache_manifest_sha256: str,
    bridge_artifact_sha256: str,
    defer_down_weight: bool = False,
    **policy: Any,
) -> Lfm2SemanticMlpPostfoldResult:
    """Prepare, freeze, gate, and transactionally install one semantic MLP lane."""

    prepared = prepare_lfm2_semantic_mlp_postfold(
        target_model=target_model,
        layer_index=layer_index,
        evidence=evidence,
        cache_key=cache_key,
        cache_manifest_sha256=cache_manifest_sha256,
        bridge_artifact_sha256=bridge_artifact_sha256,
        require_terminal_action_evidence=defer_down_weight,
        **policy,
    )
    return install_lfm2_prepared_semantic_mlp_postfold(
        target_model=target_model,
        prepared=prepared,
        defer_down_weight=defer_down_weight,
    )


__all__ = [
    "LFM2_SEMANTIC_MLP_CANDIDATE_SCHEMA",
    "LFM2_SEMANTIC_MLP_COLD_START_REJECTION_SCHEMA",
    "LFM2_SEMANTIC_MLP_EVIDENCE_SCHEMA",
    "LFM2_SEMANTIC_MLP_INSTALLATION_SCHEMA",
    "LFM2_SEMANTIC_MLP_PREPARATION_SCHEMA",
    "LFM2_SEMANTIC_MLP_SOURCE_SCHEMA",
    "Lfm2DirectLocalMlpSourceInventory",
    "Lfm2FrozenSemanticMlpCandidate",
    "Lfm2PreparedSemanticMlpRefold",
    "Lfm2SemanticMlpEndpointEvidence",
    "Lfm2SemanticMlpEndpointMetrics",
    "Lfm2SemanticMlpColdStartRejectionError",
    "Lfm2SemanticMlpPostfoldResult",
    "Lfm2SemanticMlpRefoldError",
    "compile_install_lfm2_semantic_mlp_postfold",
    "install_lfm2_prepared_semantic_mlp_postfold",
    "prepare_lfm2_semantic_mlp_postfold",
]
