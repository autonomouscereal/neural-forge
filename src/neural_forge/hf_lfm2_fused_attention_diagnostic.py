"""Fold-only, role-localized LFM2 attention diagnostics.

The main exact4x compiler emits an ordinary compact LFM2 attention operator and
the progressive component diagnostic places that operator between explicit
parent-to-child and child-to-parent residual bridges.  This module adds a
strictly diagnostic second stage without changing either pipeline.

Because the residual input bridge is immediately followed by the linear Q/K/V
projections, and the linear O projection is immediately followed by the
residual output bridge, their factors can be composed exactly in real
arithmetic::

    Q_f = Q_c @ E.T       K_f = K_c @ E.T
    V_f = V_c @ E.T       O_f = D.T @ O_c

The resulting tensors form one ordinary parent-input/parent-output
``Lfm2Attention`` with the compact head topology.  No bridge modules or
sidecars remain at runtime.  A native selected-head control and role-mixed
QK/VO candidates then localize failures while preserving intact GQA groups.

All producer functions are tensor-only closed-form operations.  They never
construct or forward a parent model, recapture traces, train, prune, quantize,
share parameters, or use network loaders.  Numeric diagnostics have no
semantic decision authority; saved decoded responses remain the evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    lfm2_authenticated_tensor_sha256,
)


LFM2_FUSED_ATTENTION_INPUT_SCHEMA = (
    "neural-forge-lfm2-fused-attention-diagnostic-input-v1"
)
LFM2_FUSED_ATTENTION_CANDIDATE_SCHEMA = (
    "neural-forge-lfm2-fused-attention-diagnostic-candidate-v1"
)
LFM2_FUSED_ATTENTION_BANK_SCHEMA = (
    "neural-forge-lfm2-fused-attention-diagnostic-bank-v1"
)
LFM2_FUSED_ATTENTION_RECONSTRUCTION_SCHEMA = (
    "neural-forge-lfm2-fused-attention-diagnostic-reconstruction-v1"
)
LFM2_FUSED_ATTENTION_DESCRIPTOR_SCHEMA = (
    "neural-forge-lfm2-fused-attention-reconstruction-descriptor-v1"
)

Lfm2FusedAttentionGroupRoleBundle = Literal["qk", "vo", "qkvo"]

_STATE_KEYS = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "out_proj.weight",
    "q_layernorm.weight",
    "k_layernorm.weight",
)
_BASE_INPUT_TENSOR_NAMES = (
    "parent.q_proj.weight",
    "parent.k_proj.weight",
    "parent.v_proj.weight",
    "parent.out_proj.weight",
    "parent.q_layernorm.weight",
    "parent.k_layernorm.weight",
    "compact.q_proj.pristine.weight",
    "compact.k_proj.pristine.weight",
    "compact.v_proj.pristine.weight",
    "compact.out_proj.pristine.weight",
    "compact.q_proj.final.weight",
    "compact.k_proj.final.weight",
    "compact.v_proj.final.weight",
    "compact.out_proj.final.weight",
    "compact.q_layernorm.weight",
    "compact.k_layernorm.weight",
    "bridge.input_encoder",
    "bridge.output_decoder",
    "selection.context_reconstruction",
)
_OPTIONAL_LOCAL_O_TENSOR_NAME = "compact.out_proj.local_corrected.weight"
_BASE_VARIANT_ORDER = (
    "native-selected",
    "qk-structural-s0-only",
    "qk-final-s2-only",
    "v-structural-s0-only",
    "v-final-s2-only",
    "o-structural-s0-only",
    "o-final-s2-only",
    "vo-structural-s0",
    "vo-final-s2",
    "qk-structural-s0-vo-final-s2",
    "qk-final-s2-vo-structural-s0",
    "fused-qkvo-structural-s0",
    "fused-qkvo-final-s2",
)
_OPTIONAL_LOCAL_VARIANT_ORDER = (
    "o-local-s1-only",
    "vo-structural-v-s0-local-o-s1",
    "fused-structural-qkv-s0-local-o-s1",
)
_TOPOLOGY_KEYS = {
    "layer_index",
    "parent_hidden_size",
    "compact_hidden_size",
    "source_query_head_count",
    "source_key_value_head_count",
    "target_query_head_count",
    "target_key_value_head_count",
    "queries_per_key_value",
    "head_dim",
    "source_query_width",
    "source_key_value_width",
    "target_query_width",
    "target_key_value_width",
    "parameter_dtype",
    "norm_eps",
    "attention_implementation",
    "qk_norm_policy",
    "local_corrected_o_supplied",
}
_ZERO_OPERATION_AUDIT = MappingProxyType(
    {
        "parent_model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
        "compact_model_forward_calls": 0,
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
        "runtime_parameter_aliases_created": 0,
        "network_access": False,
    }
)


class Lfm2FusedAttentionDiagnosticError(RuntimeError):
    """Authenticated fused-attention construction or reconstruction failed."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - required runtime dependency
        raise RuntimeError("Fused LFM2 attention diagnostics require PyTorch.") from exc
    return torch


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2FusedAttentionDiagnosticError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2FusedAttentionDiagnosticError(
            f"{label} must be one positive integer."
        )
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2FusedAttentionDiagnosticError(
            f"{label} must be one nonnegative integer."
        )
    return value


def _canonical_text(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character.isspace() for character in value)
    ):
        raise Lfm2FusedAttentionDiagnosticError(
            f"{label} must be one nonempty canonical token."
        )
    return value


def _plain_tensor(
    value: Any,
    *,
    name: str,
    ndim: int,
    clone: bool,
) -> Any:
    value = _validated_tensor(value, name=name, ndim=ndim)
    result = value.detach().to(device="cpu").contiguous()
    return result.clone() if clone else result


def _validated_tensor(
    value: Any,
    *,
    name: str,
    ndim: int,
) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != ndim
        or any(int(size) <= 0 for size in value.shape)
        or not value.dtype.is_floating_point
        or bool(getattr(value, "is_quantized", False))
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise Lfm2FusedAttentionDiagnosticError(
            f"{name} must be one finite dense detached floating rank-{ndim} tensor."
        )
    return value.detach()


def _dtype_name(dtype: Any) -> str:
    torch = _torch()
    names = {
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.float64: "float64",
    }
    if dtype not in names:
        raise Lfm2FusedAttentionDiagnosticError(
            "Fused attention supports bfloat16, float32, or float64 parameters only."
        )
    return names[dtype]


def _dtype_from_name(name: Any) -> Any:
    torch = _torch()
    values = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if name not in values:
        raise Lfm2FusedAttentionDiagnosticError(
            "Reconstruction descriptor names an unsupported parameter dtype."
        )
    return values[name]


def _assert_no_storage_aliases(
    tensors: Mapping[str, Any],
    *,
    label: str,
) -> None:
    groups: dict[tuple[str, str, int, int], list[str]] = {}
    for name, tensor in tensors.items():
        identity = (
            str(tensor.device),
            str(tensor.dtype),
            int(tensor.untyped_storage().data_ptr()),
            int(tensor.untyped_storage().nbytes()),
        )
        groups.setdefault(identity, []).append(name)
    aliases = [sorted(names) for names in groups.values() if len(names) > 1]
    if aliases:
        raise Lfm2FusedAttentionDiagnosticError(
            f"{label} contains shared or aliased tensor storage: {aliases!r}."
        )


def _tensor_record(name: str, tensor: Any) -> dict[str, Any]:
    return {
        "name": name,
        "shape": [int(size) for size in tensor.shape],
        "dtype": _dtype_name(tensor.dtype),
        "sha256": lfm2_authenticated_tensor_sha256(tensor),
    }


def _state_sha256(state: Mapping[str, Any]) -> str:
    if tuple(sorted(state)) != tuple(sorted(_STATE_KEYS)):
        raise Lfm2FusedAttentionDiagnosticError(
            "Fused attention state must contain exactly Q/K/V/O and Q/K norm tensors."
        )
    return canonical_sha256(
        [_tensor_record(name, state[name]) for name in sorted(state)]
    )


def lfm2_fused_attention_candidate_state_sha256(
    state: Mapping[str, Any],
) -> str:
    """Return the authenticated logical state digest for one candidate."""

    if tuple(sorted(state)) != tuple(sorted(_STATE_KEYS)):
        raise Lfm2FusedAttentionDiagnosticError(
            "Candidate state must contain exactly the supported LFM2 attention keys."
        )
    normalized = {
        name: _plain_tensor(
            state[name], name=name, ndim=1 if "norm" in name else 2, clone=False
        )
        for name in _STATE_KEYS
    }
    _assert_no_storage_aliases(normalized, label="candidate state")
    return _state_sha256(normalized)


def _module_parameter_alias_group_count(module: Any) -> int:
    try:
        named_parameters = module.named_parameters(remove_duplicate=False)
    except TypeError as exc:  # pragma: no cover - requires supported PyTorch
        raise Lfm2FusedAttentionDiagnosticError(
            "Runtime alias verification requires named_parameters(remove_duplicate=False)."
        ) from exc
    groups: dict[tuple[str, str, int, int], list[str]] = {}
    for name, parameter in named_parameters:
        identity = (
            str(parameter.device),
            str(parameter.dtype),
            int(parameter.untyped_storage().data_ptr()),
            int(parameter.untyped_storage().nbytes()),
        )
        groups.setdefault(identity, []).append(str(name))
    return sum(len(names) > 1 for names in groups.values())


def _validated_topology(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TOPOLOGY_KEYS:
        raise Lfm2FusedAttentionDiagnosticError(
            "Candidate topology fields are not canonical."
        )
    topology = dict(value)
    _nonnegative_int(topology["layer_index"], label="topology layer_index")
    for name in (
        "parent_hidden_size",
        "compact_hidden_size",
        "source_query_head_count",
        "source_key_value_head_count",
        "target_query_head_count",
        "target_key_value_head_count",
        "queries_per_key_value",
        "head_dim",
        "source_query_width",
        "source_key_value_width",
        "target_query_width",
        "target_key_value_width",
    ):
        _positive_int(topology[name], label=f"topology {name}")
    source_q = topology["source_query_head_count"]
    source_kv = topology["source_key_value_head_count"]
    target_q = topology["target_query_head_count"]
    target_kv = topology["target_key_value_head_count"]
    head_dim = topology["head_dim"]
    if (
        topology["compact_hidden_size"] >= topology["parent_hidden_size"]
        or source_q % source_kv
        or target_q % target_kv
        or source_q // source_kv != topology["queries_per_key_value"]
        or target_q // target_kv != topology["queries_per_key_value"]
        or target_kv >= source_kv
        or topology["source_query_width"] != source_q * head_dim
        or topology["source_key_value_width"] != source_kv * head_dim
        or topology["target_query_width"] != target_q * head_dim
        or topology["target_key_value_width"] != target_kv * head_dim
    ):
        raise Lfm2FusedAttentionDiagnosticError(
            "Candidate topology does not encode reduced intact-GQA geometry."
        )
    _dtype_from_name(topology["parameter_dtype"])
    norm_eps = topology["norm_eps"]
    if (
        isinstance(norm_eps, bool)
        or not isinstance(norm_eps, (int, float))
        or not math.isfinite(float(norm_eps))
        or float(norm_eps) <= 0.0
    ):
        raise Lfm2FusedAttentionDiagnosticError(
            "Candidate topology norm_eps must be finite and positive."
        )
    _canonical_text(
        topology["attention_implementation"],
        label="topology attention implementation",
    )
    if topology[
        "qk_norm_policy"
    ] != "fixed-direct-parent-across-all-variants" or not isinstance(
        topology["local_corrected_o_supplied"], bool
    ):
        raise Lfm2FusedAttentionDiagnosticError(
            "Candidate topology does not preserve the authenticated norm/O policy."
        )
    return MappingProxyType(topology)


@dataclass(frozen=True, slots=True)
class Lfm2FusedAttentionDiagnosticInput:
    """Pinned tensor sources for one LFM2 attention-layer diagnostic bank."""

    layer_index: int
    source_query_head_count: int
    source_key_value_head_count: int
    target_query_head_count: int
    target_key_value_head_count: int
    head_dim: int
    selected_source_key_value_groups: Sequence[int]
    selected_source_query_heads: Sequence[int]
    norm_eps: float
    attention_implementation: str
    parent_query_weight: Any = field(repr=False, compare=False)
    parent_key_weight: Any = field(repr=False, compare=False)
    parent_value_weight: Any = field(repr=False, compare=False)
    parent_out_weight: Any = field(repr=False, compare=False)
    parent_query_norm_weight: Any = field(repr=False, compare=False)
    parent_key_norm_weight: Any = field(repr=False, compare=False)
    compact_pristine_query_weight: Any = field(repr=False, compare=False)
    compact_pristine_key_weight: Any = field(repr=False, compare=False)
    compact_pristine_value_weight: Any = field(repr=False, compare=False)
    compact_pristine_out_weight: Any = field(repr=False, compare=False)
    compact_final_query_weight: Any = field(repr=False, compare=False)
    compact_final_key_weight: Any = field(repr=False, compare=False)
    compact_final_value_weight: Any = field(repr=False, compare=False)
    compact_final_out_weight: Any = field(repr=False, compare=False)
    compact_query_norm_weight: Any = field(repr=False, compare=False)
    compact_key_norm_weight: Any = field(repr=False, compare=False)
    input_encoder: Any = field(repr=False, compare=False)
    output_decoder: Any = field(repr=False, compare=False)
    context_reconstruction: Any = field(repr=False, compare=False)
    compact_local_corrected_out_weight: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _nonnegative_int(self.layer_index, label="layer_index")
        for name in (
            "source_query_head_count",
            "source_key_value_head_count",
            "target_query_head_count",
            "target_key_value_head_count",
            "head_dim",
        ):
            _positive_int(getattr(self, name), label=name)
        groups = tuple(self.selected_source_key_value_groups)
        heads = tuple(self.selected_source_query_heads)
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in groups
        ):
            raise Lfm2FusedAttentionDiagnosticError(
                "Selected source KV groups must be integers."
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in heads
        ):
            raise Lfm2FusedAttentionDiagnosticError(
                "Selected source query heads must be integers."
            )
        object.__setattr__(self, "selected_source_key_value_groups", groups)
        object.__setattr__(self, "selected_source_query_heads", heads)
        if (
            isinstance(self.norm_eps, bool)
            or not isinstance(self.norm_eps, (int, float))
            or not math.isfinite(float(self.norm_eps))
            or float(self.norm_eps) <= 0.0
        ):
            raise Lfm2FusedAttentionDiagnosticError(
                "norm_eps must be one finite positive value."
            )
        object.__setattr__(self, "norm_eps", float(self.norm_eps))
        _canonical_text(
            self.attention_implementation,
            label="attention_implementation",
        )

    def named_tensors(self) -> Mapping[str, Any]:
        """Return the fixed input tensor namespace without cloning values."""

        return MappingProxyType(
            {
                "parent.q_proj.weight": self.parent_query_weight,
                "parent.k_proj.weight": self.parent_key_weight,
                "parent.v_proj.weight": self.parent_value_weight,
                "parent.out_proj.weight": self.parent_out_weight,
                "parent.q_layernorm.weight": self.parent_query_norm_weight,
                "parent.k_layernorm.weight": self.parent_key_norm_weight,
                "compact.q_proj.pristine.weight": self.compact_pristine_query_weight,
                "compact.k_proj.pristine.weight": self.compact_pristine_key_weight,
                "compact.v_proj.pristine.weight": self.compact_pristine_value_weight,
                "compact.out_proj.pristine.weight": (self.compact_pristine_out_weight),
                "compact.q_proj.final.weight": self.compact_final_query_weight,
                "compact.k_proj.final.weight": self.compact_final_key_weight,
                "compact.v_proj.final.weight": self.compact_final_value_weight,
                "compact.out_proj.final.weight": self.compact_final_out_weight,
                "compact.q_layernorm.weight": self.compact_query_norm_weight,
                "compact.k_layernorm.weight": self.compact_key_norm_weight,
                "bridge.input_encoder": self.input_encoder,
                "bridge.output_decoder": self.output_decoder,
                "selection.context_reconstruction": self.context_reconstruction,
                **(
                    {}
                    if self.compact_local_corrected_out_weight is None
                    else {
                        _OPTIONAL_LOCAL_O_TENSOR_NAME: (
                            self.compact_local_corrected_out_weight
                        )
                    }
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class Lfm2FusedAttentionDiagnosticCandidate:
    """One ordinary parent-I/O reduced-head attention state and its seals."""

    variant_id: str
    query_weight: Any = field(repr=False, compare=False)
    key_weight: Any = field(repr=False, compare=False)
    value_weight: Any = field(repr=False, compare=False)
    out_weight: Any = field(repr=False, compare=False)
    query_norm_weight: Any = field(repr=False, compare=False)
    key_norm_weight: Any = field(repr=False, compare=False)
    receipt: Mapping[str, Any]
    reconstruction_descriptor: Mapping[str, Any]

    def state_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "q_proj.weight": self.query_weight.detach().cpu().contiguous().clone(),
                "k_proj.weight": self.key_weight.detach().cpu().contiguous().clone(),
                "v_proj.weight": self.value_weight.detach().cpu().contiguous().clone(),
                "out_proj.weight": self.out_weight.detach().cpu().contiguous().clone(),
                "q_layernorm.weight": (
                    self.query_norm_weight.detach().cpu().contiguous().clone()
                ),
                "k_layernorm.weight": (
                    self.key_norm_weight.detach().cpu().contiguous().clone()
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class Lfm2FusedAttentionDiagnosticBank:
    """The complete fixed variant bank for one attention layer."""

    candidates: Mapping[str, Lfm2FusedAttentionDiagnosticCandidate]
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Lfm2FusedAttentionReconstruction:
    """A freshly reconstructed concrete LFM2 module and its receipt."""

    module: Any = field(repr=False, compare=False)
    receipt: Mapping[str, Any]


def _normalized_input(
    source: Lfm2FusedAttentionDiagnosticInput,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(source, Lfm2FusedAttentionDiagnosticInput):
        raise TypeError("source must be Lfm2FusedAttentionDiagnosticInput.")
    raw = dict(source.named_tensors())
    expected_names = _BASE_INPUT_TENSOR_NAMES + (
        ()
        if source.compact_local_corrected_out_weight is None
        else (_OPTIONAL_LOCAL_O_TENSOR_NAME,)
    )
    if tuple(raw) != expected_names:
        raise Lfm2FusedAttentionDiagnosticError(
            "Fused attention input tensor namespace is not canonical."
        )
    raw_checked = {
        name: _validated_tensor(
            tensor,
            name=name,
            ndim=1 if "layernorm" in name else 2,
        )
        for name, tensor in raw.items()
    }
    _assert_no_storage_aliases(raw_checked, label="fused attention inputs")
    tensors = {
        name: _plain_tensor(
            tensor,
            name=name,
            ndim=1 if "layernorm" in name else 2,
            clone=True,
        )
        for name, tensor in raw.items()
    }

    source_q = source.source_query_head_count
    source_kv = source.source_key_value_head_count
    target_q = source.target_query_head_count
    target_kv = source.target_key_value_head_count
    head_dim = source.head_dim
    if source_q % source_kv or target_q % target_kv:
        raise Lfm2FusedAttentionDiagnosticError(
            "Source and target query heads must divide exactly into GQA groups."
        )
    source_q_per_kv = source_q // source_kv
    target_q_per_kv = target_q // target_kv
    if source_q_per_kv != target_q_per_kv:
        raise Lfm2FusedAttentionDiagnosticError(
            "Source and target must preserve the same query-to-KV GQA ratio."
        )
    if target_kv >= source_kv:
        raise Lfm2FusedAttentionDiagnosticError(
            "Diagnostic target must contain fewer intact GQA groups than the source."
        )
    groups = tuple(source.selected_source_key_value_groups)
    if (
        len(groups) != target_kv
        or len(set(groups)) != target_kv
        or any(group < 0 or group >= source_kv for group in groups)
    ):
        raise Lfm2FusedAttentionDiagnosticError(
            "Selected source KV groups do not exactly realize the target topology."
        )
    expected_query_heads = tuple(
        group * source_q_per_kv + offset
        for group in groups
        for offset in range(source_q_per_kv)
    )
    if tuple(source.selected_source_query_heads) != expected_query_heads:
        raise Lfm2FusedAttentionDiagnosticError(
            "Selected source query heads are not complete ordered intact GQA groups."
        )

    parent_q = tensors["parent.q_proj.weight"]
    compact_q = tensors["compact.q_proj.final.weight"]
    parent_hidden = int(parent_q.shape[1])
    compact_hidden = int(compact_q.shape[1])
    source_q_width = source_q * head_dim
    source_kv_width = source_kv * head_dim
    target_q_width = target_q * head_dim
    target_kv_width = target_kv * head_dim
    expected_shapes = {
        "parent.q_proj.weight": (source_q_width, parent_hidden),
        "parent.k_proj.weight": (source_kv_width, parent_hidden),
        "parent.v_proj.weight": (source_kv_width, parent_hidden),
        "parent.out_proj.weight": (parent_hidden, source_q_width),
        "parent.q_layernorm.weight": (head_dim,),
        "parent.k_layernorm.weight": (head_dim,),
        "compact.q_proj.pristine.weight": (target_q_width, compact_hidden),
        "compact.k_proj.pristine.weight": (target_kv_width, compact_hidden),
        "compact.v_proj.pristine.weight": (target_kv_width, compact_hidden),
        "compact.out_proj.pristine.weight": (compact_hidden, target_q_width),
        "compact.q_proj.final.weight": (target_q_width, compact_hidden),
        "compact.k_proj.final.weight": (target_kv_width, compact_hidden),
        "compact.v_proj.final.weight": (target_kv_width, compact_hidden),
        "compact.out_proj.final.weight": (compact_hidden, target_q_width),
        "compact.q_layernorm.weight": (head_dim,),
        "compact.k_layernorm.weight": (head_dim,),
        "bridge.input_encoder": (parent_hidden, compact_hidden),
        "bridge.output_decoder": (compact_hidden, parent_hidden),
        "selection.context_reconstruction": (target_q_width, source_q_width),
        **(
            {}
            if source.compact_local_corrected_out_weight is None
            else {
                _OPTIONAL_LOCAL_O_TENSOR_NAME: (compact_hidden, target_q_width),
            }
        ),
    }
    if any(
        tuple(tensors[name].shape) != shape for name, shape in expected_shapes.items()
    ):
        observed = {name: list(tensors[name].shape) for name in expected_shapes}
        raise Lfm2FusedAttentionDiagnosticError(
            "Fused attention source geometry is inconsistent: "
            f"expected={expected_shapes!r}, observed={observed!r}."
        )
    parameter_dtypes = {
        tensor.dtype
        for name, tensor in tensors.items()
        if not name.startswith("bridge.") and name != "selection.context_reconstruction"
    }
    if len(parameter_dtypes) != 1:
        raise Lfm2FusedAttentionDiagnosticError(
            "Parent, compact, pristine, final, optional local-O, and norm parameters "
            "must share one dtype."
        )
    parameter_dtype = next(iter(parameter_dtypes))
    parameter_dtype_name = _dtype_name(parameter_dtype)
    if lfm2_authenticated_tensor_sha256(
        tensors["parent.q_layernorm.weight"]
    ) != lfm2_authenticated_tensor_sha256(
        tensors["compact.q_layernorm.weight"]
    ) or lfm2_authenticated_tensor_sha256(
        tensors["parent.k_layernorm.weight"]
    ) != lfm2_authenticated_tensor_sha256(tensors["compact.k_layernorm.weight"]):
        raise Lfm2FusedAttentionDiagnosticError(
            "Compact Q/K norms are not the authenticated fixed direct-parent norms."
        )
    topology = {
        "layer_index": source.layer_index,
        "parent_hidden_size": parent_hidden,
        "compact_hidden_size": compact_hidden,
        "source_query_head_count": source_q,
        "source_key_value_head_count": source_kv,
        "target_query_head_count": target_q,
        "target_key_value_head_count": target_kv,
        "queries_per_key_value": source_q_per_kv,
        "head_dim": head_dim,
        "source_query_width": source_q_width,
        "source_key_value_width": source_kv_width,
        "target_query_width": target_q_width,
        "target_key_value_width": target_kv_width,
        "parameter_dtype": parameter_dtype_name,
        "norm_eps": source.norm_eps,
        "attention_implementation": source.attention_implementation,
        "qk_norm_policy": "fixed-direct-parent-across-all-variants",
        "local_corrected_o_supplied": (
            source.compact_local_corrected_out_weight is not None
        ),
    }
    return tensors, topology


def fingerprint_lfm2_fused_attention_diagnostic_input(
    source: Lfm2FusedAttentionDiagnosticInput,
) -> Mapping[str, Any]:
    """Fingerprint all producer tensors and topology for external pinning."""

    tensors, topology = _normalized_input(source)
    body = {
        "schema": LFM2_FUSED_ATTENTION_INPUT_SCHEMA,
        "status": "fused-attention-inputs-authenticated",
        "topology": topology,
        "selection": {
            "selected_source_key_value_groups": list(
                source.selected_source_key_value_groups
            ),
            "selected_source_query_heads": list(source.selected_source_query_heads),
            "whole_gqa_groups_only": True,
        },
        "tensor_records": {
            name: _tensor_record(name, tensor) for name, tensor in tensors.items()
        },
        "input_storage_alias_group_count": 0,
        "input_parameters_quantized": 0,
        "input_parameters_sparse": 0,
    }
    return MappingProxyType({**body, "input_bundle_sha256": canonical_sha256(body)})


def _authenticated_input(
    source: Lfm2FusedAttentionDiagnosticInput,
    *,
    expected_input_bundle_sha256: str,
    expected_tensor_sha256: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any]]:
    supplied_bundle = _digest(
        expected_input_bundle_sha256,
        label="expected input bundle",
    )
    if not isinstance(expected_tensor_sha256, Mapping):
        raise Lfm2FusedAttentionDiagnosticError(
            "expected_tensor_sha256 must be one complete mapping."
        )
    expected_names = tuple(source.named_tensors())
    if tuple(sorted(expected_tensor_sha256)) != tuple(sorted(expected_names)):
        raise Lfm2FusedAttentionDiagnosticError(
            "Expected tensor pins do not cover the exact fused-attention input namespace."
        )
    expected_pins = {
        name: _digest(value, label=f"expected {name}")
        for name, value in expected_tensor_sha256.items()
    }
    authentication = fingerprint_lfm2_fused_attention_diagnostic_input(source)
    observed_pins = {
        name: record["sha256"]
        for name, record in authentication["tensor_records"].items()
    }
    if (
        authentication["input_bundle_sha256"] != supplied_bundle
        or observed_pins != expected_pins
    ):
        raise Lfm2FusedAttentionDiagnosticError(
            "Fused-attention inputs differ from their independent pins."
        )
    tensors, topology = _normalized_input(source)
    return tensors, topology, authentication


def lfm2_fused_attention_parameter_count(
    *,
    hidden_size: int,
    query_head_count: int,
    key_value_head_count: int,
    head_dim: int,
) -> int:
    """Count one dense bias-free LFM2 attention, including Q/K RMSNorms."""

    hidden = _positive_int(hidden_size, label="hidden_size")
    query_heads = _positive_int(query_head_count, label="query_head_count")
    key_value_heads = _positive_int(
        key_value_head_count,
        label="key_value_head_count",
    )
    dimension = _positive_int(head_dim, label="head_dim")
    query_width = query_heads * dimension
    key_value_width = key_value_heads * dimension
    return hidden * (2 * query_width + 2 * key_value_width) + 2 * dimension


def _matmul_fp64(left: Any, right: Any, *, label: str) -> Any:
    torch = _torch()
    result = (
        left.to(device="cpu", dtype=torch.float64)
        @ right.to(device="cpu", dtype=torch.float64)
    ).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2FusedAttentionDiagnosticError(
            f"{label} produced non-finite fused parameters."
        )
    return result


def _candidate_state(
    *,
    query: Any,
    key: Any,
    value: Any,
    out: Any,
    query_norm: Any,
    key_norm: Any,
    dtype: Any,
) -> dict[str, Any]:
    values = {
        "q_proj.weight": query,
        "k_proj.weight": key,
        "v_proj.weight": value,
        "out_proj.weight": out,
        "q_layernorm.weight": query_norm,
        "k_layernorm.weight": key_norm,
    }
    result = {
        name: tensor.detach().to(device="cpu", dtype=dtype).contiguous().clone()
        for name, tensor in values.items()
    }
    result = {
        name: _plain_tensor(
            tensor,
            name=name,
            ndim=1 if "norm" in name else 2,
            clone=True,
        )
        for name, tensor in result.items()
    }
    _assert_no_storage_aliases(result, label="materialized candidate")
    return result


def _candidate_shapes(topology: Mapping[str, Any]) -> Mapping[str, tuple[int, ...]]:
    parent_hidden = int(topology["parent_hidden_size"])
    query_width = int(topology["target_query_width"])
    key_value_width = int(topology["target_key_value_width"])
    head_dim = int(topology["head_dim"])
    return MappingProxyType(
        {
            "q_proj.weight": (query_width, parent_hidden),
            "k_proj.weight": (key_value_width, parent_hidden),
            "v_proj.weight": (key_value_width, parent_hidden),
            "out_proj.weight": (parent_hidden, query_width),
            "q_layernorm.weight": (head_dim,),
            "k_layernorm.weight": (head_dim,),
        }
    )


def _build_candidate(
    *,
    variant_id: str,
    state: Mapping[str, Any],
    topology: Mapping[str, Any],
    input_bundle_sha256: str,
    role_sources: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> Lfm2FusedAttentionDiagnosticCandidate:
    variant = _canonical_text(variant_id, label="variant_id")
    if tuple(sorted(state)) != tuple(sorted(_STATE_KEYS)):
        raise Lfm2FusedAttentionDiagnosticError(
            "Candidate builder received incomplete attention state."
        )
    normalized = {
        name: _plain_tensor(
            state[name],
            name=name,
            ndim=1 if "norm" in name else 2,
            clone=True,
        )
        for name in _STATE_KEYS
    }
    _assert_no_storage_aliases(normalized, label=f"candidate {variant}")
    expected_shapes = _candidate_shapes(topology)
    if any(
        tuple(normalized[name].shape) != expected_shapes[name] for name in _STATE_KEYS
    ):
        raise Lfm2FusedAttentionDiagnosticError(
            f"Candidate {variant} does not match its parent-I/O reduced-head geometry."
        )
    dtypes = {tensor.dtype for tensor in normalized.values()}
    if (
        len(dtypes) != 1
        or _dtype_name(next(iter(dtypes))) != topology["parameter_dtype"]
    ):
        raise Lfm2FusedAttentionDiagnosticError(
            f"Candidate {variant} does not preserve the authenticated parameter dtype."
        )
    state_sha256 = _state_sha256(normalized)
    candidate_elements = sum(int(tensor.numel()) for tensor in normalized.values())
    expected_elements = lfm2_fused_attention_parameter_count(
        hidden_size=int(topology["parent_hidden_size"]),
        query_head_count=int(topology["target_query_head_count"]),
        key_value_head_count=int(topology["target_key_value_head_count"]),
        head_dim=int(topology["head_dim"]),
    )
    original_elements = lfm2_fused_attention_parameter_count(
        hidden_size=int(topology["parent_hidden_size"]),
        query_head_count=int(topology["source_query_head_count"]),
        key_value_head_count=int(topology["source_key_value_head_count"]),
        head_dim=int(topology["head_dim"]),
    )
    if (
        candidate_elements != expected_elements
        or candidate_elements >= original_elements
    ):
        raise Lfm2FusedAttentionDiagnosticError(
            "Fused attention candidate is not one literal physical parameter reduction."
        )
    element_size = next(iter(normalized.values())).element_size()
    descriptor_body = {
        "schema": LFM2_FUSED_ATTENTION_DESCRIPTOR_SCHEMA,
        "status": "production-safe-ordinary-lfm2-attention-descriptor",
        "family_adapter": "lfm2-concrete-runtime-fused-attention-v1",
        "variant_id": variant,
        "layer_index": int(topology["layer_index"]),
        "source_parent_geometry": {
            "hidden_size": int(topology["parent_hidden_size"]),
            "num_attention_heads": int(topology["source_query_head_count"]),
            "num_key_value_heads": int(topology["source_key_value_head_count"]),
            "head_dim": int(topology["head_dim"]),
        },
        "operator_geometry": {
            "hidden_size": int(topology["parent_hidden_size"]),
            "num_attention_heads": int(topology["target_query_head_count"]),
            "num_key_value_heads": int(topology["target_key_value_head_count"]),
            "head_dim": int(topology["head_dim"]),
            "norm_eps": float(topology["norm_eps"]),
            "attention_implementation": topology["attention_implementation"],
        },
        "parameter_dtype": topology["parameter_dtype"],
        "state_keys": list(_STATE_KEYS),
        "candidate_state_sha256": state_sha256,
        "input_bridge_module_present": False,
        "output_bridge_module_present": False,
        "bridge_factors_retained_as_runtime_parameters": False,
        "ordinary_transformers_lfm2_attention": True,
        "runtime_parameter_alias_group_count": 0,
        "runtime_parameter_sharing_used": False,
    }
    descriptor = {
        **descriptor_body,
        "descriptor_sha256": canonical_sha256(descriptor_body),
    }
    physical = {
        "original_parent_attention_parameter_elements": original_elements,
        "fused_candidate_parameter_elements": candidate_elements,
        "physical_parameter_reduction": original_elements - candidate_elements,
        "fused_candidate_parameter_bytes": candidate_elements * element_size,
        "input_bridge_parameter_elements": 0,
        "output_bridge_parameter_elements": 0,
        "sidecar_parameter_elements": 0,
        "runtime_shared_parameter_elements": 0,
        "stored_tensor_elements": candidate_elements,
        "parameter_reduction_attributed_to_sharing": False,
        "parameter_reduction_attributed_to_quantization": False,
        "parameter_reduction_attributed_to_pruning": False,
        "parameter_reduction_attributed_to_sparsity": False,
    }
    receipt_body = {
        "schema": LFM2_FUSED_ATTENTION_CANDIDATE_SCHEMA,
        "status": "NON-FINAL-FOLD-ONLY-FUSED-ATTENTION-DIAGNOSTIC",
        "variant_id": variant,
        "topology": dict(topology),
        "input_bundle_sha256": _digest(
            input_bundle_sha256,
            label="candidate input bundle",
        ),
        "role_sources": dict(role_sources),
        "lineage": dict(lineage),
        "candidate_tensor_records": {
            name: _tensor_record(name, normalized[name]) for name in _STATE_KEYS
        },
        "candidate_state_sha256": state_sha256,
        "reconstruction_descriptor_sha256": descriptor["descriptor_sha256"],
        "physical_parameter_accounting": physical,
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
        "interpretation": {
            "parent_input_output_width_retained_for_diagnostic_only": True,
            "compact_head_topology_installed": True,
            "whole_decoded_behavior_required": True,
            "numeric_metrics_have_semantic_decision_authority": False,
            "standalone_exact4x_child_claimed": False,
            "does_not_modify_main_exact4x_compiler": True,
        },
    }
    receipt = {
        **receipt_body,
        "result_sha256": canonical_sha256(receipt_body),
    }
    return Lfm2FusedAttentionDiagnosticCandidate(
        variant_id=variant,
        query_weight=normalized["q_proj.weight"],
        key_weight=normalized["k_proj.weight"],
        value_weight=normalized["v_proj.weight"],
        out_weight=normalized["out_proj.weight"],
        query_norm_weight=normalized["q_layernorm.weight"],
        key_norm_weight=normalized["k_layernorm.weight"],
        receipt=MappingProxyType(receipt),
        reconstruction_descriptor=MappingProxyType(descriptor),
    )


def compile_lfm2_fused_attention_diagnostic_bank(
    source: Lfm2FusedAttentionDiagnosticInput,
    *,
    expected_input_bundle_sha256: str,
    expected_tensor_sha256: Mapping[str, str],
) -> Lfm2FusedAttentionDiagnosticBank:
    """Build the fixed native/QK/VO/O/QKVO candidate bank without forwards."""

    torch = _torch()
    tensors, topology, authentication = _authenticated_input(
        source,
        expected_input_bundle_sha256=expected_input_bundle_sha256,
        expected_tensor_sha256=expected_tensor_sha256,
    )
    dtype = _dtype_from_name(topology["parameter_dtype"])
    head_dim = int(topology["head_dim"])
    query_rows = [
        head * head_dim + coordinate
        for head in source.selected_source_query_heads
        for coordinate in range(head_dim)
    ]
    key_value_rows = [
        group * head_dim + coordinate
        for group in source.selected_source_key_value_groups
        for coordinate in range(head_dim)
    ]
    parent_q = tensors["parent.q_proj.weight"]
    parent_k = tensors["parent.k_proj.weight"]
    parent_v = tensors["parent.v_proj.weight"]
    parent_o = tensors["parent.out_proj.weight"]
    pristine_q = tensors["compact.q_proj.pristine.weight"]
    pristine_k = tensors["compact.k_proj.pristine.weight"]
    pristine_v = tensors["compact.v_proj.pristine.weight"]
    pristine_o = tensors["compact.out_proj.pristine.weight"]
    final_q = tensors["compact.q_proj.final.weight"]
    final_k = tensors["compact.k_proj.final.weight"]
    final_v = tensors["compact.v_proj.final.weight"]
    final_o = tensors["compact.out_proj.final.weight"]
    encoder = tensors["bridge.input_encoder"]
    decoder = tensors["bridge.output_decoder"]
    reconstruction = tensors["selection.context_reconstruction"]

    native_q = parent_q[query_rows, :].to(dtype=torch.float64).contiguous()
    native_k = parent_k[key_value_rows, :].to(dtype=torch.float64).contiguous()
    native_v = parent_v[key_value_rows, :].to(dtype=torch.float64).contiguous()
    native_o = _matmul_fp64(
        parent_o,
        reconstruction.T,
        label="native selected-head O reconstruction",
    )
    fused_q_pristine = _matmul_fp64(
        pristine_q,
        encoder.T,
        label="fused structural S0 Q",
    )
    fused_k_pristine = _matmul_fp64(
        pristine_k,
        encoder.T,
        label="fused structural S0 K",
    )
    fused_v_pristine = _matmul_fp64(
        pristine_v,
        encoder.T,
        label="fused structural S0 V",
    )
    fused_o_pristine = _matmul_fp64(
        decoder.T,
        pristine_o,
        label="fused structural S0 O",
    )
    fused_q_final = _matmul_fp64(
        final_q,
        encoder.T,
        label="fused final/post-global S2 Q",
    )
    fused_k_final = _matmul_fp64(
        final_k,
        encoder.T,
        label="fused final/post-global S2 K",
    )
    fused_v_final = _matmul_fp64(
        final_v,
        encoder.T,
        label="fused final/post-global S2 V",
    )
    fused_o_final = _matmul_fp64(
        decoder.T,
        final_o,
        label="fused final/post-global O",
    )
    parent_q_norm = tensors["parent.q_layernorm.weight"]
    parent_k_norm = tensors["parent.k_layernorm.weight"]

    def roles(*, query: str, key: str, value: str, out: str) -> dict[str, str]:
        return {
            "q": query,
            "k": key,
            "v": value,
            "o": out,
            "qk_norms": "fixed-direct-parent",
        }

    sources = {
        "native-selected": (
            native_q,
            native_k,
            native_v,
            native_o,
            parent_q_norm,
            parent_k_norm,
            roles(
                query="native-selected",
                key="native-selected",
                value="native-selected",
                out="native-reconstructed",
            ),
        ),
        "qk-structural-s0-only": (
            fused_q_pristine,
            fused_k_pristine,
            native_v,
            native_o,
            parent_q_norm,
            parent_k_norm,
            roles(
                query="bridge-fused-structural-s0",
                key="bridge-fused-structural-s0",
                value="native-selected",
                out="native-reconstructed",
            ),
        ),
        "qk-final-s2-only": (
            fused_q_final,
            fused_k_final,
            native_v,
            native_o,
            parent_q_norm,
            parent_k_norm,
            roles(
                query="bridge-fused-final-post-global-s2",
                key="bridge-fused-final-post-global-s2",
                value="native-selected",
                out="native-reconstructed",
            ),
        ),
        "v-structural-s0-only": (
            native_q,
            native_k,
            fused_v_pristine,
            native_o,
            parent_q_norm,
            parent_k_norm,
            roles(
                query="native-selected",
                key="native-selected",
                value="bridge-fused-structural-s0",
                out="native-reconstructed",
            ),
        ),
        "v-final-s2-only": (
            native_q,
            native_k,
            fused_v_final,
            native_o,
            parent_q_norm,
            parent_k_norm,
            roles(
                query="native-selected",
                key="native-selected",
                value="bridge-fused-final-post-global-s2",
                out="native-reconstructed",
            ),
        ),
        "o-structural-s0-only": (
            native_q,
            native_k,
            native_v,
            fused_o_pristine,
            parent_q_norm,
            parent_k_norm,
            roles(
                query="native-selected",
                key="native-selected",
                value="native-selected",
                out="bridge-fused-pristine-s0",
            ),
        ),
        "o-final-s2-only": (
            native_q,
            native_k,
            native_v,
            fused_o_final,
            parent_q_norm,
            parent_k_norm,
            roles(
                query="native-selected",
                key="native-selected",
                value="native-selected",
                out="bridge-fused-final-post-global-s2",
            ),
        ),
        "vo-structural-s0": (
            native_q,
            native_k,
            fused_v_pristine,
            fused_o_pristine,
            parent_q_norm,
            parent_k_norm,
            roles(
                query="native-selected",
                key="native-selected",
                value="bridge-fused-structural-s0",
                out="bridge-fused-pristine-s0",
            ),
        ),
        "vo-final-s2": (
            native_q,
            native_k,
            fused_v_final,
            fused_o_final,
            parent_q_norm,
            parent_k_norm,
            roles(
                query="native-selected",
                key="native-selected",
                value="bridge-fused-final-post-global-s2",
                out="bridge-fused-final-post-global-s2",
            ),
        ),
        "qk-structural-s0-vo-final-s2": (
            fused_q_pristine,
            fused_k_pristine,
            fused_v_final,
            fused_o_final,
            parent_q_norm,
            parent_k_norm,
            roles(
                query="bridge-fused-structural-s0",
                key="bridge-fused-structural-s0",
                value="bridge-fused-final-post-global-s2",
                out="bridge-fused-final-post-global-s2",
            ),
        ),
        "qk-final-s2-vo-structural-s0": (
            fused_q_final,
            fused_k_final,
            fused_v_pristine,
            fused_o_pristine,
            parent_q_norm,
            parent_k_norm,
            roles(
                query="bridge-fused-final-post-global-s2",
                key="bridge-fused-final-post-global-s2",
                value="bridge-fused-structural-s0",
                out="bridge-fused-pristine-s0",
            ),
        ),
        "fused-qkvo-structural-s0": (
            fused_q_pristine,
            fused_k_pristine,
            fused_v_pristine,
            fused_o_pristine,
            parent_q_norm,
            parent_k_norm,
            roles(
                query="bridge-fused-structural-s0",
                key="bridge-fused-structural-s0",
                value="bridge-fused-structural-s0",
                out="bridge-fused-pristine-s0",
            ),
        ),
        "fused-qkvo-final-s2": (
            fused_q_final,
            fused_k_final,
            fused_v_final,
            fused_o_final,
            parent_q_norm,
            parent_k_norm,
            roles(
                query="bridge-fused-final-post-global-s2",
                key="bridge-fused-final-post-global-s2",
                value="bridge-fused-final-post-global-s2",
                out="bridge-fused-final-post-global-s2",
            ),
        ),
    }
    variant_order = _BASE_VARIANT_ORDER
    if source.compact_local_corrected_out_weight is not None:
        fused_o_local = _matmul_fp64(
            decoder.T,
            tensors[_OPTIONAL_LOCAL_O_TENSOR_NAME],
            label="fused externally authenticated source-local-only O",
        )
        sources.update(
            {
                "o-local-s1-only": (
                    native_q,
                    native_k,
                    native_v,
                    fused_o_local,
                    parent_q_norm,
                    parent_k_norm,
                    roles(
                        query="native-selected",
                        key="native-selected",
                        value="native-selected",
                        out="bridge-fused-externally-authenticated-local-only",
                    ),
                ),
                "vo-structural-v-s0-local-o-s1": (
                    native_q,
                    native_k,
                    fused_v_pristine,
                    fused_o_local,
                    parent_q_norm,
                    parent_k_norm,
                    roles(
                        query="native-selected",
                        key="native-selected",
                        value="bridge-fused-structural-s0",
                        out="bridge-fused-externally-authenticated-local-only",
                    ),
                ),
                "fused-structural-qkv-s0-local-o-s1": (
                    fused_q_pristine,
                    fused_k_pristine,
                    fused_v_pristine,
                    fused_o_local,
                    parent_q_norm,
                    parent_k_norm,
                    roles(
                        query="bridge-fused-structural-s0",
                        key="bridge-fused-structural-s0",
                        value="bridge-fused-structural-s0",
                        out="bridge-fused-externally-authenticated-local-only",
                    ),
                ),
            }
        )
        variant_order += _OPTIONAL_LOCAL_VARIANT_ORDER
    compact_stage_byte_equality = {
        role: (
            authentication["tensor_records"][f"compact.{role}_proj.pristine.weight"][
                "sha256"
            ]
            == authentication["tensor_records"][f"compact.{role}_proj.final.weight"][
                "sha256"
            ]
        )
        for role in ("q", "k", "v", "out")
    }
    candidates: dict[str, Lfm2FusedAttentionDiagnosticCandidate] = {}
    for variant in variant_order:
        query, key, value, out, query_norm, key_norm, role_sources = sources[variant]
        state = _candidate_state(
            query=query,
            key=key,
            value=value,
            out=out,
            query_norm=query_norm,
            key_norm=key_norm,
            dtype=dtype,
        )
        candidates[variant] = _build_candidate(
            variant_id=variant,
            state=state,
            topology=topology,
            input_bundle_sha256=authentication["input_bundle_sha256"],
            role_sources=role_sources,
            lineage={
                "construction": "closed-form-fp64-bridge-fusion-and-intact-gqa-selection",
                "input_authentication_schema": authentication["schema"],
                "selected_source_key_value_groups": list(
                    source.selected_source_key_value_groups
                ),
                "selected_source_query_heads": list(source.selected_source_query_heads),
                "q_formula": "compact_q @ input_encoder.T",
                "k_formula": "compact_k @ input_encoder.T",
                "v_formula": "compact_v @ input_encoder.T",
                "o_formula": "output_decoder.T @ compact_o",
                "native_o_formula": "parent_o @ context_reconstruction.T",
                "pristine_qkvo_artifact_stage": "s0-pristine-structural-archive",
                "final_qkvo_artifact_stage": "s2-final-post-global-checkpoint",
                "source_local_only_o_requires_external_authenticated_bytes": True,
                "qk_norms_fixed_direct_parent_across_variants": True,
            },
        )
    body = {
        "schema": LFM2_FUSED_ATTENTION_BANK_SCHEMA,
        "status": "NON-FINAL-FUSED-ATTENTION-DIAGNOSTIC-BANK-COMPLETE",
        "input_bundle_sha256": authentication["input_bundle_sha256"],
        "variant_ids": list(variant_order),
        "candidate_result_sha256": {
            name: candidates[name].receipt["result_sha256"] for name in variant_order
        },
        "candidate_state_sha256": {
            name: candidates[name].receipt["candidate_state_sha256"]
            for name in variant_order
        },
        "compact_qkvo_artifact_lanes": {
            "pristine_qkvo_s0_supplied_and_authenticated": True,
            "final_qkvo_post_global_s2_supplied_and_authenticated": True,
            "source_local_only_s1_bytes_supplied_and_authenticated": (
                source.compact_local_corrected_out_weight is not None
            ),
            "source_local_only_s1_inferred_from_receipt_hash_without_bytes": False,
        },
        "compact_s0_s2_byte_equality_by_role": compact_stage_byte_equality,
        "compact_s0_s2_full_qkvo_byte_equal": all(compact_stage_byte_equality.values()),
        "compact_s0_s2_stage_transition_is_authenticated_noop": all(
            compact_stage_byte_equality.values()
        ),
        "candidate_deduplication_performed": False,
        "candidate_deduplication_allowed_only_on_candidate_state_sha256_equality": True,
        "qk_norms_fixed_across_all_variants": True,
        "all_candidates_are_ordinary_parent_io_reduced_head_lfm2_attention": True,
        "bridge_sidecar_count": 0,
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
        "semantic_acceptance_performed": False,
        "standalone_exact4x_child_claimed": False,
    }
    receipt = {**body, "result_sha256": canonical_sha256(body)}
    return Lfm2FusedAttentionDiagnosticBank(
        candidates=MappingProxyType(candidates),
        receipt=MappingProxyType(receipt),
    )


def authenticate_lfm2_fused_attention_reconstruction_descriptor(
    descriptor: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate one independently supplied reconstruction descriptor."""

    if not isinstance(descriptor, Mapping):
        raise Lfm2FusedAttentionDiagnosticError(
            "Reconstruction descriptor must be one mapping."
        )
    value = dict(descriptor)
    supplied = _digest(
        value.pop("descriptor_sha256", None),
        label="descriptor",
    )
    if canonical_sha256(value) != supplied:
        raise Lfm2FusedAttentionDiagnosticError(
            "Reconstruction descriptor seal is invalid."
        )
    if (
        value.get("schema") != LFM2_FUSED_ATTENTION_DESCRIPTOR_SCHEMA
        or value.get("status") != "production-safe-ordinary-lfm2-attention-descriptor"
        or value.get("family_adapter") != "lfm2-concrete-runtime-fused-attention-v1"
        or value.get("state_keys") != list(_STATE_KEYS)
        or value.get("input_bridge_module_present") is not False
        or value.get("output_bridge_module_present") is not False
        or value.get("bridge_factors_retained_as_runtime_parameters") is not False
        or value.get("ordinary_transformers_lfm2_attention") is not True
        or value.get("runtime_parameter_alias_group_count") != 0
        or value.get("runtime_parameter_sharing_used") is not False
    ):
        raise Lfm2FusedAttentionDiagnosticError(
            "Reconstruction descriptor does not describe the supported fused LFM2 seam."
        )
    _canonical_text(value.get("variant_id"), label="descriptor variant_id")
    _nonnegative_int(value.get("layer_index"), label="descriptor layer_index")
    _digest(value.get("candidate_state_sha256"), label="descriptor candidate state")
    _dtype_from_name(value.get("parameter_dtype"))
    for name in ("source_parent_geometry", "operator_geometry"):
        if not isinstance(value.get(name), Mapping):
            raise Lfm2FusedAttentionDiagnosticError(f"Descriptor {name} is missing.")
    source_geometry = value["source_parent_geometry"]
    operator_geometry = value["operator_geometry"]
    if set(source_geometry) != {
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    } or set(operator_geometry) != {
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "norm_eps",
        "attention_implementation",
    }:
        raise Lfm2FusedAttentionDiagnosticError(
            "Reconstruction descriptor geometry fields are not canonical."
        )
    for geometry, prefix in (
        (source_geometry, "source"),
        (operator_geometry, "operator"),
    ):
        for name in (
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        ):
            _positive_int(geometry.get(name), label=f"descriptor {prefix} {name}")
    source_q = int(source_geometry["num_attention_heads"])
    source_kv = int(source_geometry["num_key_value_heads"])
    target_q = int(operator_geometry["num_attention_heads"])
    target_kv = int(operator_geometry["num_key_value_heads"])
    if (
        source_geometry["hidden_size"] != operator_geometry["hidden_size"]
        or source_geometry["head_dim"] != operator_geometry["head_dim"]
        or source_q % source_kv
        or target_q % target_kv
        or source_q // source_kv != target_q // target_kv
        or target_kv >= source_kv
    ):
        raise Lfm2FusedAttentionDiagnosticError(
            "Reconstruction descriptor does not preserve reduced intact-GQA geometry."
        )
    norm_eps = operator_geometry.get("norm_eps")
    if (
        isinstance(norm_eps, bool)
        or not isinstance(norm_eps, (int, float))
        or not math.isfinite(float(norm_eps))
        or float(norm_eps) <= 0.0
    ):
        raise Lfm2FusedAttentionDiagnosticError(
            "Descriptor norm_eps must be finite and positive."
        )
    _canonical_text(
        operator_geometry.get("attention_implementation"),
        label="descriptor attention implementation",
    )
    return MappingProxyType({**value, "descriptor_sha256": supplied})


def authenticate_lfm2_fused_attention_candidate(
    candidate: Lfm2FusedAttentionDiagnosticCandidate,
) -> Mapping[str, Any]:
    """Authenticate candidate tensors, receipt, and reconstruction descriptor."""

    if not isinstance(candidate, Lfm2FusedAttentionDiagnosticCandidate):
        raise TypeError("candidate must be Lfm2FusedAttentionDiagnosticCandidate.")
    receipt = dict(candidate.receipt)
    supplied = _digest(receipt.pop("result_sha256", None), label="candidate result")
    if (
        canonical_sha256(receipt) != supplied
        or receipt.get("schema") != LFM2_FUSED_ATTENTION_CANDIDATE_SCHEMA
        or receipt.get("status") != "NON-FINAL-FOLD-ONLY-FUSED-ATTENTION-DIAGNOSTIC"
        or receipt.get("variant_id") != candidate.variant_id
        or receipt.get("operation_audit") != dict(_ZERO_OPERATION_AUDIT)
    ):
        raise Lfm2FusedAttentionDiagnosticError(
            "Fused attention candidate receipt is invalid."
        )
    descriptor = authenticate_lfm2_fused_attention_reconstruction_descriptor(
        candidate.reconstruction_descriptor
    )
    raw_state = {
        "q_proj.weight": candidate.query_weight,
        "k_proj.weight": candidate.key_weight,
        "v_proj.weight": candidate.value_weight,
        "out_proj.weight": candidate.out_weight,
        "q_layernorm.weight": candidate.query_norm_weight,
        "k_layernorm.weight": candidate.key_norm_weight,
    }
    raw_checked = {
        name: _validated_tensor(
            tensor,
            name=f"candidate {name}",
            ndim=1 if "norm" in name else 2,
        )
        for name, tensor in raw_state.items()
    }
    _assert_no_storage_aliases(raw_checked, label="candidate raw state")
    state = candidate.state_dict()
    state_sha = lfm2_fused_attention_candidate_state_sha256(state)
    topology = _validated_topology(receipt.get("topology"))
    physical = receipt.get("physical_parameter_accounting")
    if not isinstance(physical, Mapping):
        raise Lfm2FusedAttentionDiagnosticError(
            "Candidate receipt lacks topology or physical accounting."
        )
    expected_source_geometry = {
        "hidden_size": int(topology["parent_hidden_size"]),
        "num_attention_heads": int(topology["source_query_head_count"]),
        "num_key_value_heads": int(topology["source_key_value_head_count"]),
        "head_dim": int(topology["head_dim"]),
    }
    expected_operator_geometry = {
        "hidden_size": int(topology["parent_hidden_size"]),
        "num_attention_heads": int(topology["target_query_head_count"]),
        "num_key_value_heads": int(topology["target_key_value_head_count"]),
        "head_dim": int(topology["head_dim"]),
        "norm_eps": float(topology["norm_eps"]),
        "attention_implementation": topology["attention_implementation"],
    }
    observed_elements = sum(int(tensor.numel()) for tensor in state.values())
    expected_elements = lfm2_fused_attention_parameter_count(
        hidden_size=expected_operator_geometry["hidden_size"],
        query_head_count=expected_operator_geometry["num_attention_heads"],
        key_value_head_count=expected_operator_geometry["num_key_value_heads"],
        head_dim=expected_operator_geometry["head_dim"],
    )
    original_elements = lfm2_fused_attention_parameter_count(
        hidden_size=expected_source_geometry["hidden_size"],
        query_head_count=expected_source_geometry["num_attention_heads"],
        key_value_head_count=expected_source_geometry["num_key_value_heads"],
        head_dim=expected_source_geometry["head_dim"],
    )
    element_size = next(iter(state.values())).element_size()
    expected_records = {name: _tensor_record(name, state[name]) for name in _STATE_KEYS}
    expected_physical = {
        "original_parent_attention_parameter_elements": original_elements,
        "fused_candidate_parameter_elements": expected_elements,
        "physical_parameter_reduction": original_elements - expected_elements,
        "fused_candidate_parameter_bytes": expected_elements * element_size,
        "input_bridge_parameter_elements": 0,
        "output_bridge_parameter_elements": 0,
        "sidecar_parameter_elements": 0,
        "runtime_shared_parameter_elements": 0,
        "stored_tensor_elements": expected_elements,
        "parameter_reduction_attributed_to_sharing": False,
        "parameter_reduction_attributed_to_quantization": False,
        "parameter_reduction_attributed_to_pruning": False,
        "parameter_reduction_attributed_to_sparsity": False,
    }
    if (
        state_sha != receipt.get("candidate_state_sha256")
        or state_sha != descriptor.get("candidate_state_sha256")
        or receipt.get("candidate_tensor_records") != expected_records
        or descriptor["descriptor_sha256"]
        != receipt.get("reconstruction_descriptor_sha256")
        or descriptor["source_parent_geometry"] != expected_source_geometry
        or descriptor["operator_geometry"] != expected_operator_geometry
        or descriptor["parameter_dtype"] != topology["parameter_dtype"]
        or observed_elements != expected_elements
        or dict(physical) != expected_physical
    ):
        raise Lfm2FusedAttentionDiagnosticError(
            "Candidate state, descriptor, and receipt are not bound together."
        )
    return MappingProxyType({**receipt, "result_sha256": supplied})


def substitute_lfm2_fused_attention_gqa_group(
    *,
    base: Lfm2FusedAttentionDiagnosticCandidate,
    donor: Lfm2FusedAttentionDiagnosticCandidate,
    target_group_index: int,
    role_bundle: Lfm2FusedAttentionGroupRoleBundle,
    variant_id: str,
) -> Lfm2FusedAttentionDiagnosticCandidate:
    """Substitute one intact target GQA group without changing topology/count."""

    base_receipt = authenticate_lfm2_fused_attention_candidate(base)
    donor_receipt = authenticate_lfm2_fused_attention_candidate(donor)
    topology = base_receipt.get("topology")
    if (
        not isinstance(topology, Mapping)
        or donor_receipt.get("topology") != topology
        or donor_receipt.get("input_bundle_sha256")
        != base_receipt.get("input_bundle_sha256")
    ):
        raise Lfm2FusedAttentionDiagnosticError(
            "GQA-group substitution candidates do not share one authenticated topology."
        )
    group = _nonnegative_int(target_group_index, label="target_group_index")
    target_kv = int(topology["target_key_value_head_count"])
    if group >= target_kv:
        raise Lfm2FusedAttentionDiagnosticError(
            "GQA-group substitution index is outside the target topology."
        )
    if role_bundle not in {"qk", "vo", "qkvo"}:
        raise Lfm2FusedAttentionDiagnosticError("role_bundle must be qk, vo, or qkvo.")
    head_dim = int(topology["head_dim"])
    queries_per_kv = int(topology["queries_per_key_value"])
    query_start = group * queries_per_kv * head_dim
    query_stop = query_start + queries_per_kv * head_dim
    key_value_start = group * head_dim
    key_value_stop = key_value_start + head_dim
    base_state = {name: tensor.clone() for name, tensor in base.state_dict().items()}
    donor_state = donor.state_dict()
    torch = _torch()
    with torch.no_grad():
        if role_bundle in {"qk", "qkvo"}:
            base_state["q_proj.weight"][query_start:query_stop, :].copy_(
                donor_state["q_proj.weight"][query_start:query_stop, :]
            )
            base_state["k_proj.weight"][key_value_start:key_value_stop, :].copy_(
                donor_state["k_proj.weight"][key_value_start:key_value_stop, :]
            )
        if role_bundle in {"vo", "qkvo"}:
            base_state["v_proj.weight"][key_value_start:key_value_stop, :].copy_(
                donor_state["v_proj.weight"][key_value_start:key_value_stop, :]
            )
            base_state["out_proj.weight"][:, query_start:query_stop].copy_(
                donor_state["out_proj.weight"][:, query_start:query_stop]
            )
    # Q/K norms operate on the shared head coordinate and cannot be replaced for
    # only one GQA group; the base norms are intentionally retained.
    return _build_candidate(
        variant_id=variant_id,
        state=base_state,
        topology=topology,
        input_bundle_sha256=base_receipt["input_bundle_sha256"],
        role_sources={
            "base_candidate": base.variant_id,
            "donor_candidate": donor.variant_id,
            "target_group_index": group,
            "role_bundle": role_bundle,
            "shared_qk_norm_source": base.variant_id,
        },
        lineage={
            "construction": "one-intact-gqa-group-row-column-substitution",
            "base_candidate_result_sha256": base_receipt["result_sha256"],
            "donor_candidate_result_sha256": donor_receipt["result_sha256"],
            "query_row_range": [query_start, query_stop],
            "key_value_row_range": [key_value_start, key_value_stop],
            "out_column_range": [query_start, query_stop],
            "qk_norms_are_shared_head_coordinate_not_per_group": True,
        },
    )


def reconstruct_lfm2_fused_attention_candidate(
    *,
    candidate: Lfm2FusedAttentionDiagnosticCandidate,
    parent_config: Any,
    expected_candidate_result_sha256: str,
    expected_descriptor_sha256: str,
    device: Any = "cpu",
) -> Lfm2FusedAttentionReconstruction:
    """Freshly build one ordinary Transformers LFM2 attention without forwarding it."""

    torch = _torch()
    receipt = authenticate_lfm2_fused_attention_candidate(candidate)
    descriptor = authenticate_lfm2_fused_attention_reconstruction_descriptor(
        candidate.reconstruction_descriptor
    )
    if receipt["result_sha256"] != _digest(
        expected_candidate_result_sha256,
        label="expected candidate result",
    ) or descriptor["descriptor_sha256"] != _digest(
        expected_descriptor_sha256,
        label="expected reconstruction descriptor",
    ):
        raise Lfm2FusedAttentionDiagnosticError(
            "Candidate or reconstruction descriptor differs from its external pin."
        )
    to_dict = getattr(parent_config, "to_dict", None)
    if not callable(to_dict):
        raise Lfm2FusedAttentionDiagnosticError(
            "Reconstruction requires one concrete LFM2 parent config."
        )
    source_geometry = descriptor["source_parent_geometry"]
    operator_geometry = descriptor["operator_geometry"]
    parent_observed = {
        "hidden_size": int(getattr(parent_config, "hidden_size")),
        "num_attention_heads": int(getattr(parent_config, "num_attention_heads")),
        "num_key_value_heads": int(getattr(parent_config, "num_key_value_heads")),
        "head_dim": int(getattr(parent_config, "head_dim")),
    }
    if parent_observed != dict(source_geometry):
        raise Lfm2FusedAttentionDiagnosticError(
            "Parent config differs from the descriptor source geometry."
        )
    if (
        float(getattr(parent_config, "norm_eps"))
        != float(operator_geometry["norm_eps"])
        or getattr(parent_config, "_attn_implementation", None)
        != operator_geometry["attention_implementation"]
    ):
        raise Lfm2FusedAttentionDiagnosticError(
            "Parent config differs from the descriptor normalization/attention seam."
        )
    layer_index = int(descriptor["layer_index"])
    if layer_index >= int(getattr(parent_config, "num_hidden_layers")):
        raise Lfm2FusedAttentionDiagnosticError(
            "Descriptor layer index is outside the parent LFM2 config."
        )
    requested_device = torch.device(device)
    if requested_device.type == "meta":
        raise Lfm2FusedAttentionDiagnosticError(
            "Fused attention reconstruction cannot target the meta device."
        )
    try:
        from transformers import Lfm2Config
        from transformers.models.lfm2.modeling_lfm2 import Lfm2Attention
    except ImportError as exc:  # pragma: no cover - required runtime dependency
        raise Lfm2FusedAttentionDiagnosticError(
            "Concrete Transformers LFM2 classes are required for reconstruction."
        ) from exc
    config_value = dict(to_dict())
    config_value.update(
        {
            "hidden_size": int(operator_geometry["hidden_size"]),
            "num_attention_heads": int(operator_geometry["num_attention_heads"]),
            "num_key_value_heads": int(operator_geometry["num_key_value_heads"]),
            "head_dim": int(operator_geometry["head_dim"]),
            "norm_eps": float(operator_geometry["norm_eps"]),
        }
    )
    config = Lfm2Config(**config_value)
    config._attn_implementation = _canonical_text(
        operator_geometry["attention_implementation"],
        label="descriptor attention implementation",
    )
    module = Lfm2Attention(config, layer_index)
    dtype = _dtype_from_name(descriptor["parameter_dtype"])
    module.to(device=requested_device, dtype=dtype)
    state = {
        name: tensor.to(device=requested_device, dtype=dtype)
        for name, tensor in candidate.state_dict().items()
    }
    incompatible = module.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise Lfm2FusedAttentionDiagnosticError(
            "Fresh LFM2 attention did not accept the complete fused state."
        )
    module.requires_grad_(False)
    module.eval()
    runtime_alias_group_count = _module_parameter_alias_group_count(module)
    if runtime_alias_group_count:
        raise Lfm2FusedAttentionDiagnosticError(
            "Fresh LFM2 fused attention unexpectedly contains parameter aliases."
        )
    module_state = {
        name: _plain_tensor(
            tensor,
            name=name,
            ndim=1 if "norm" in name else 2,
            clone=False,
        )
        for name, tensor in module.state_dict().items()
    }
    _assert_no_storage_aliases(module_state, label="reconstructed attention")
    module_state_sha = _state_sha256(module_state)
    parameter_elements = sum(
        int(parameter.numel()) for parameter in module.parameters()
    )
    expected_elements = receipt["physical_parameter_accounting"][
        "fused_candidate_parameter_elements"
    ]
    if (
        module_state_sha != descriptor["candidate_state_sha256"]
        or parameter_elements != expected_elements
    ):
        raise Lfm2FusedAttentionDiagnosticError(
            "Fresh LFM2 reconstruction differs from its authenticated candidate."
        )
    body = {
        "schema": LFM2_FUSED_ATTENTION_RECONSTRUCTION_SCHEMA,
        "status": "ordinary-lfm2-fused-attention-freshly-reconstructed",
        "candidate_result_sha256": receipt["result_sha256"],
        "descriptor_sha256": descriptor["descriptor_sha256"],
        "candidate_state_sha256": descriptor["candidate_state_sha256"],
        "reconstructed_state_sha256": module_state_sha,
        "parameter_elements": parameter_elements,
        "runtime_parameter_alias_group_count": runtime_alias_group_count,
        "input_bridge_module_present": False,
        "output_bridge_module_present": False,
        "model_forward_calls": 0,
        "training_steps": 0,
    }
    reconstruction_receipt = {
        **body,
        "result_sha256": canonical_sha256(body),
    }
    return Lfm2FusedAttentionReconstruction(
        module=module,
        receipt=MappingProxyType(reconstruction_receipt),
    )


__all__ = [
    "LFM2_FUSED_ATTENTION_BANK_SCHEMA",
    "LFM2_FUSED_ATTENTION_CANDIDATE_SCHEMA",
    "LFM2_FUSED_ATTENTION_DESCRIPTOR_SCHEMA",
    "LFM2_FUSED_ATTENTION_INPUT_SCHEMA",
    "LFM2_FUSED_ATTENTION_RECONSTRUCTION_SCHEMA",
    "Lfm2FusedAttentionDiagnosticBank",
    "Lfm2FusedAttentionDiagnosticCandidate",
    "Lfm2FusedAttentionDiagnosticError",
    "Lfm2FusedAttentionDiagnosticInput",
    "Lfm2FusedAttentionReconstruction",
    "authenticate_lfm2_fused_attention_candidate",
    "authenticate_lfm2_fused_attention_reconstruction_descriptor",
    "compile_lfm2_fused_attention_diagnostic_bank",
    "fingerprint_lfm2_fused_attention_diagnostic_input",
    "lfm2_fused_attention_candidate_state_sha256",
    "lfm2_fused_attention_parameter_count",
    "reconstruct_lfm2_fused_attention_candidate",
    "substitute_lfm2_fused_attention_gqa_group",
]
