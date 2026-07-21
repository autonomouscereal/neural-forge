"""No-training semantic-endpoint SwiGLU refolding for a dense LFM2 child.

This adapter consumes one parent-exact semantic endpoint per cached phase and
actual-child macrocell replay captured from the current child state.  FIT is
the only partition used by either analytic solve.  FIT-CHECK, SELECTION, and
VALIDATION are trajectory-isolated gates over already-frozen candidates.

Two candidates compete with the incumbent dense SwiGLU:

* a conservative producer/writer candidate keeps incumbent W1/W3 and solves
  only W2 from actual child post-SwiGLU contexts to mapped parent residual
  writes; and
* the existing structural residual MLP compiler clusters every parent neuron
  and solves dense W1/W3/W2 with the incumbent child state as its prior.

There are no model executions, gradients, optimizers, training steps, pruning,
quantization, sparsity, shared-parameter aliases, or numerical parent/child
weight-equality objectives in this module.  The final layer can explicitly
defer W2 to the fused semantic terminal/lexical writer while exporting four
authenticated ``Lfm2CpcfTerminalActionEvidence`` banks.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (
    Lfm2CpcfTerminalActionEvidence,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    Lfm2CachedMlpFoldCase,
    Lfm2CompiledStructuralMlpFold,
    compile_lfm2_structural_residual_mlp_fold,
)


LFM2_SEMANTIC_MLP_SOURCE_SCHEMA = (
    "neural-forge-lfm2-direct-local-semantic-mlp-source-v1"
)
LFM2_SEMANTIC_MLP_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-semantic-mlp-endpoint-evidence-v1"
)
LFM2_SEMANTIC_MLP_CANDIDATE_SCHEMA = (
    "neural-forge-lfm2-semantic-mlp-frozen-candidate-v1"
)
LFM2_SEMANTIC_MLP_PREPARATION_SCHEMA = "neural-forge-lfm2-semantic-mlp-preparation-v1"
LFM2_SEMANTIC_MLP_INSTALLATION_SCHEMA = "neural-forge-lfm2-semantic-mlp-installation-v1"

_PARTITIONS = ("fit", "fit_check", "selection", "validation")
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
    """Authenticated direct-local parent W1/W3/W2 bundle for one layer."""

    layer_index: int
    parent_model_identity_sha256: str
    parent_checkpoint_manifest_sha256: str
    direct_local_loader_receipt_sha256: str
    residual_assignment_sha256: str
    compiler_lineage_sha256: str
    gate_weight: Any = field(repr=False, compare=False)
    up_weight: Any = field(repr=False, compare=False)
    down_weight: Any = field(repr=False, compare=False)
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
        source_intermediate, source_hidden = map(int, gate.shape)
        if tuple(up.shape) != tuple(gate.shape) or tuple(down.shape) != (
            source_hidden,
            source_intermediate,
        ):
            raise Lfm2SemanticMlpRefoldError(
                "Direct-local source tensors are not one dense SwiGLU bundle."
            )
        object.__setattr__(self, "gate_weight", gate)
        object.__setattr__(self, "up_weight", up)
        object.__setattr__(self, "down_weight", down)
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
            "compiler_lineage_sha256": self.compiler_lineage_sha256,
            "source_hidden_size": self.source_hidden_size,
            "source_intermediate_size": self.source_intermediate_size,
            "weight_sha256": {
                "w1.weight": _tensor_sha256(self.gate_weight),
                "w3.weight": _tensor_sha256(self.up_weight),
                "w2.weight": _tensor_sha256(self.down_weight),
            },
            "weight_bundle_sha256": _state_sha256(
                {
                    "w1.weight": self.gate_weight,
                    "w3.weight": self.up_weight,
                    "w2.weight": self.down_weight,
                }
            ),
            "loader_mode": "direct-local-safetensors",
            "auto_model_loader_used": False,
            "from_pretrained_used": False,
            "hub_access_used": False,
            "dense_floating_source": True,
        }

    def verify(self) -> bool:
        return self.inventory_sha256 == canonical_sha256(self.authentication_body())


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
        child_input = _finite_tensor(
            self.actual_child_mlp_input,
            "actual child MLP input",
            ndim=2,
            exact_dtype=torch.bfloat16,
        )
        child_pre = _finite_tensor(
            self.actual_child_pre_mlp_residual,
            "actual child pre-MLP residual",
            ndim=2,
            exact_dtype=torch.bfloat16,
        )
        parent_endpoint = _finite_tensor(
            self.parent_post_mlp_residual_target_space,
            "mapped parent post-MLP residual",
            ndim=2,
            exact_dtype=torch.bfloat16,
        )
        terminal = (
            None
            if self.parent_final_norm_output_target_space is None
            else _finite_tensor(
                self.parent_final_norm_output_target_space,
                "mapped parent final-norm output",
                ndim=2,
                exact_dtype=torch.bfloat16,
            )
        )
        source_intermediate = self.source_inventory.source_intermediate_size
        rows, target_hidden = map(int, child_input.shape)
        if (
            rows != 1
            or tuple(parent_gate.shape) != (1, source_intermediate)
            or tuple(parent_up.shape) != tuple(parent_gate.shape)
            or tuple(parent_post.shape) != tuple(parent_gate.shape)
            or tuple(child_pre.shape) != (1, target_hidden)
            or tuple(parent_endpoint.shape) != tuple(child_pre.shape)
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
        object.__setattr__(self, "parent_gate", parent_gate)
        object.__setattr__(self, "parent_up", parent_up)
        object.__setattr__(self, "parent_post_swiglu", parent_post)
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
            "compiler_lineage_sha256": (self.source_inventory.compiler_lineage_sha256),
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
            "actual_child_replay_freshness_bound": True,
        }

    def verify(self) -> bool:
        return self.evidence_sha256 == canonical_sha256(self.authentication_body())


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
    selected_role: CandidateRole
    selection_receipt: Mapping[str, Any]
    terminal_action_evidence: Mapping[str, Lfm2CpcfTerminalActionEvidence]
    preparation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
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
            self.selected_role not in {"incumbent-producer-w2", "full-w1-w3-w2"}
            or not all(
                candidate.verify()
                for candidate in (
                    self.incumbent,
                    self.writer_candidate,
                    self.full_candidate,
                )
            )
            or self.selected_role
            not in {self.writer_candidate.role, self.full_candidate.role}
            or set(self.partition_batch_indices) != set(_PARTITIONS)
            or any(not self.partition_batch_indices[name] for name in _PARTITIONS)
            or any(
                not _digest(value, "endpoint evidence")
                for value in self.evidence_sha256
            )
            or set(self.terminal_action_evidence).difference(_PARTITIONS)
        ):
            raise Lfm2SemanticMlpRefoldError(
                "Prepared semantic MLP candidate set is incomplete."
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

    def __post_init__(self) -> None:
        if not self.prepared.verify():
            raise Lfm2SemanticMlpRefoldError("Prepared MLP result lost authentication.")
        object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))
        object.__setattr__(
            self,
            "final_layer_mlp_evidence",
            MappingProxyType(dict(self.final_layer_mlp_evidence)),
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


def _parameter_bundle(target_model: Any, layer_index: int) -> Mapping[str, Any]:
    torch = _torch()
    mlp = _mlp_module(target_model, layer_index)
    prefix = f"model.layers.{layer_index}.feed_forward."
    values = {
        prefix + name + ".weight": getattr(getattr(mlp, name), "weight", None)
        for name in ("w1", "w3", "w2")
    }
    named = dict(target_model.named_parameters())
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
    ):
        raise Lfm2SemanticMlpRefoldError(
            "Target W1/W3/W2 Parameters are missing, aliased, renamed, or nondense."
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
    applied_scale = 1.0
    if maximum_delta_ratio is not None and unclipped_ratio > maximum_delta_ratio:
        applied_scale = maximum_delta_ratio / unclipped_ratio
        delta = delta * applied_scale
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
        "maximum_delta_ratio": maximum_delta_ratio,
        "applied_delta_scale": applied_scale,
        "analytic_linear_solve": True,
        "cluster_solve": False,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    return state, MappingProxyType(receipt)


def _full_candidate_state(
    fit: Sequence[Lfm2SemanticMlpEndpointEvidence],
    inventory: Lfm2DirectLocalMlpSourceInventory,
    incumbent: Mapping[str, Any],
    *,
    ridge: float,
    maximum_delta_ratio: float | None,
    cluster_ridge: float,
    cluster_max_iterations: int,
    cluster_distance_chunk_size: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
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
        max_prior_delta_ratio=maximum_delta_ratio,
        cluster_ridge=cluster_ridge,
        cluster_max_iterations=cluster_max_iterations,
        cluster_distance_chunk_size=cluster_distance_chunk_size,
    )
    if (
        not isinstance(compiled, Lfm2CompiledStructuralMlpFold)
        or not compiled.full_source_coverage
        or compiled.source_state_sha256_before != compiled.source_state_sha256_after
    ):
        raise Lfm2SemanticMlpRefoldError(
            "Full structural MLP compiler did not preserve complete source coverage."
        )
    receipt = {
        **compiled.as_dict(),
        "solver": "fit-only-full-source-cluster-plus-three-closed-form-linear-solves",
        "fit_rows": len(fit),
        "heldout_rows_used_by_solver": 0,
        "analytic_linear_solve": True,
        "cluster_solve": True,
        "current_dense_c61_w1_w3_w2_used_as_priors": True,
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
) -> Lfm2PreparedSemanticMlpRefold:
    """Compile and freeze both FIT-only candidates without mutating the child."""

    cache = _digest(cache_key, "cache key")
    manifest = _digest(cache_manifest_sha256, "cache manifest")
    bridge = _digest(bridge_artifact_sha256, "bridge artifact")
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
        isinstance(item, Lfm2SemanticMlpEndpointEvidence) and item.verify()
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
    target_intermediate, target_hidden = map(int, incumbent_state["w1.weight"].shape)
    if (
        tuple(incumbent_state["w3.weight"].shape)
        != (target_intermediate, target_hidden)
        or tuple(incumbent_state["w2.weight"].shape)
        != (target_hidden, target_intermediate)
        or any(item.target_hidden_size != target_hidden for item in raw)
    ):
        raise Lfm2SemanticMlpRefoldError(
            "Evidence and current child SwiGLU topology differ."
        )
    partitions = _split_evidence(raw)
    source_before = inventory.inventory_sha256
    writer_state, writer_solver = _solve_writer(
        partitions["fit"],
        incumbent_state,
        ridge=float(ridge),
        maximum_delta_ratio=(
            None if maximum_delta_ratio is None else float(maximum_delta_ratio)
        ),
    )
    full_state, full_solver = _full_candidate_state(
        partitions["fit"],
        inventory,
        incumbent_state,
        ridge=float(ridge),
        maximum_delta_ratio=(
            None if maximum_delta_ratio is None else float(maximum_delta_ratio)
        ),
        cluster_ridge=float(cluster_ridge),
        cluster_max_iterations=int(cluster_max_iterations),
        cluster_distance_chunk_size=int(cluster_distance_chunk_size),
    )
    if inventory.inventory_sha256 != source_before or not inventory.verify():
        raise Lfm2SemanticMlpRefoldError(
            "Direct-local parent W1/W3/W2 bundle changed during compilation."
        )
    incumbent = Lfm2FrozenSemanticMlpCandidate(
        role="incumbent",
        gate_weight=incumbent_state["w1.weight"],
        up_weight=incumbent_state["w3.weight"],
        down_weight=incumbent_state["w2.weight"],
        metrics=_all_metrics(incumbent_state, partitions),
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
        metrics=_all_metrics(writer_state, partitions),
        solver_receipt=writer_solver,
    )
    full = Lfm2FrozenSemanticMlpCandidate(
        role="full-w1-w3-w2",
        gate_weight=full_state["w1.weight"],
        up_weight=full_state["w3.weight"],
        down_weight=full_state["w2.weight"],
        metrics=_all_metrics(full_state, partitions),
        solver_receipt=full_solver,
    )
    heldout = ("fit_check", "selection", "validation")
    eligibility = {
        candidate.role: {
            name: _nonregressing(
                candidate.metrics[name],
                incumbent.metrics[name],
                rmse_relative_tolerance=float(rmse_relative_tolerance),
                cosine_tolerance=float(cosine_tolerance),
            )
            for name in heldout
        }
        for candidate in (writer, full)
    }
    writer_changed = bool(_changed_roles(incumbent_state, writer_state))
    full_changed = bool(_changed_roles(incumbent_state, full_state))
    writer_selection_vs_full = _nonregressing(
        writer.metrics["selection"],
        full.metrics["selection"],
        rmse_relative_tolerance=float(rmse_relative_tolerance),
        cosine_tolerance=float(cosine_tolerance),
    )
    writer_prevalidation = all(
        eligibility[writer.role][name] for name in ("fit_check", "selection")
    )
    full_prevalidation = all(
        eligibility[full.role][name] for name in ("fit_check", "selection")
    )
    if (
        writer_changed
        and writer_prevalidation
        and (not full_prevalidation or writer_selection_vs_full)
    ):
        selected = writer
        rationale = "simplest-w2-only-candidate-nonregressing-versus-incumbent-and-full"
    elif full_changed and full_prevalidation:
        selected = full
        rationale = "full-fold-selection-quality-exceeded-conservative-writer"
    else:
        raise Lfm2SemanticMlpRefoldError(
            "No nonempty semantic MLP candidate passed FIT-CHECK and SELECTION."
        )
    if not eligibility[selected.role]["validation"]:
        raise Lfm2SemanticMlpRefoldError(
            "The frozen selected semantic MLP candidate regressed on VALIDATION."
        )
    if fingerprint_lfm2_parent(target_model).identity_sha256 != current_state:
        raise Lfm2SemanticMlpRefoldError(
            "Candidate preparation mutated the live child."
        )
    terminal = _terminal_banks(
        partitions,
        required=require_terminal_action_evidence,
    )
    selection = MappingProxyType(
        {
            "selected_role": selected.role,
            "rationale": rationale,
            "writer_changed_roles": list(_changed_roles(incumbent_state, writer_state)),
            "full_changed_roles": list(_changed_roles(incumbent_state, full_state)),
            "heldout_nonregression": eligibility,
            "writer_selection_nonregressing_versus_full": writer_selection_vs_full,
            "candidate_frozen_before_fit_check": True,
            "candidate_frozen_before_selection": True,
            "candidate_frozen_before_validation": True,
            "validation_triggered_refit": False,
            "fit_partition_only_used_for_solves": True,
            "heldout_partitions_used_for_solves": False,
            "endpoint_metrics": {
                candidate.role: {
                    name: candidate.metrics[name].as_dict() for name in _PARTITIONS
                }
                for candidate in (incumbent, writer, full)
            },
        }
    )
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
        selected_role=selected.role,
        selection_receipt=selection,
        terminal_action_evidence=terminal,
    )


def _copy_candidate(parameter: Any, value: Any) -> None:
    """Single testable mutation point; rollback deliberately bypasses it."""

    parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


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
    expected_changed = tuple(
        role
        for role in install_roles
        if not torch.equal(
            state[role],
            incumbent[prefix + role],
        )
    )
    if not expected_changed:
        if not defer_down_weight:
            raise Lfm2SemanticMlpRefoldError(
                "Selected semantic MLP candidate contains no installable correction."
            )
        if set(prepared.terminal_action_evidence) != set(_PARTITIONS):
            raise Lfm2SemanticMlpRefoldError(
                "A fully deferred W2 requires all four terminal-action banks."
            )
        receipt_body = {
            "schema": LFM2_SEMANTIC_MLP_INSTALLATION_SCHEMA,
            "status": "prepared-final-w2-deferred",
            "installed": False,
            "layer_index": prepared.layer_index,
            "preparation_sha256": prepared.preparation_sha256,
            "selected_role": prepared.selected_role,
            "model_state_before_sha256": current,
            "model_state_after_sha256": current,
            "parameter_count_before": prepared.target_parameter_count,
            "parameter_count_after": prepared.target_parameter_count,
            "parameter_count_changed": False,
            "topology_changed": False,
            "install_count": 0,
            "changed_parameter_roles": [],
            "final_layer_mlp_w2_deferred_to_fused_terminal": True,
            "downstream_terminal_replay_and_validation_required": True,
            "terminal_action_evidence_sha256": {
                name: value.authentication_sha256
                for name, value in prepared.terminal_action_evidence.items()
            },
            "operation_audit": dict(_ZERO_OPERATIONS),
        }
        receipt = MappingProxyType(
            {
                **receipt_body,
                "result_sha256": canonical_sha256(receipt_body),
            }
        )
        return Lfm2SemanticMlpPostfoldResult(
            receipt=receipt,
            prepared=prepared,
            final_layer_mlp_evidence=prepared.terminal_action_evidence,
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
            name.removeprefix(prefix)
            for name, parameter in parameters.items()
            if not torch.equal(parameter.detach().cpu(), incumbent[name])
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
            "evidence_endpoint_count": len(prepared.evidence_sha256),
            "parent_exact_endpoint_count": len(prepared.evidence_sha256),
            "inserted_target_rows_used": 0,
            "query_scope": "one-parent-exact-semantic-endpoint-per-phase",
            "candidate_scope": (
                "incumbent-producer-dense-w2-residual-writer-and-"
                "full-source-clustered-dense-w1-w3-w2"
            ),
            "numeric_parent_child_weight_match_objective": False,
            "analytic_linear_and_cluster_solves_only": True,
            "sequential_replay_freshness_enforced": True,
            "operation_audit": dict(_ZERO_OPERATIONS),
        }
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
    "LFM2_SEMANTIC_MLP_EVIDENCE_SCHEMA",
    "LFM2_SEMANTIC_MLP_INSTALLATION_SCHEMA",
    "LFM2_SEMANTIC_MLP_PREPARATION_SCHEMA",
    "LFM2_SEMANTIC_MLP_SOURCE_SCHEMA",
    "Lfm2DirectLocalMlpSourceInventory",
    "Lfm2FrozenSemanticMlpCandidate",
    "Lfm2PreparedSemanticMlpRefold",
    "Lfm2SemanticMlpEndpointEvidence",
    "Lfm2SemanticMlpEndpointMetrics",
    "Lfm2SemanticMlpPostfoldResult",
    "Lfm2SemanticMlpRefoldError",
    "compile_install_lfm2_semantic_mlp_postfold",
    "install_lfm2_prepared_semantic_mlp_postfold",
    "prepare_lfm2_semantic_mlp_postfold",
]
