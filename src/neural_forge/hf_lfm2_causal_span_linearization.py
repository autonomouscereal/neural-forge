"""Authenticated causal composition across one LFM2 source-depth span.

This module is deliberately narrower than a fold installer.  It binds an
existing :class:`~neural_forge.hf_lfm2_model_fold.Lfm2ModelFoldSpan` to frozen
full-history source primals and exposes the missing matrix-free map from a
residual perturbation at any source boundary to the terminal span boundary.

The implementation reuses the exact analytic FP64 JVP/VJP operators in
``hf_lfm2_global_linearization``.  It does not execute a model, fit weights,
invoke autograd, install parameters, or decide whether decoded behavior is
acceptable.  Architecture-neutral folding code can consume the ``jvp`` and
``vjp`` surface while an LFM2 evidence adapter remains responsible for
constructing the authenticated primals from cached histories and direct local
weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from neural_forge.hf_lfm2_global_linearization import (
    Lfm2WholeStackPrimalSnapshot,
    lfm2_mlp_segment_jvp,
    lfm2_mlp_segment_vjp,
)
from neural_forge.hf_lfm2_global_residual_closure import (
    fingerprint_lfm2_global_residual_tensor,
)
from neural_forge.hf_lfm2_model_fold import Lfm2ModelFoldSpan


LFM2_CAUSAL_SPAN_LINEARIZATION_SCHEMA = "neural-forge-lfm2-causal-span-linearization-v1"

Lfm2CausalSpanBoundaryRole = Literal[
    "pre_operator",
    "post_operator",
    "post_layer",
]

_BOUNDARY_ROLES = frozenset(
    {
        "pre_operator",
        "post_operator",
        "post_layer",
    }
)

_ZERO_OPERATION_AUDIT = MappingProxyType(
    {
        "parent_model_objects_constructed": 0,
        "parent_forward_calls": 0,
        "parent_generate_calls": 0,
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
        "parameter_sharing_used": False,
        "sidecar_parameters_created": 0,
        "numeric_behavior_gate_created": False,
        "manual_behavior_gate_created": False,
    }
)


class Lfm2CausalSpanLinearizationError(ValueError):
    """One causal-span request has invalid topology or tensor geometry."""


class Lfm2CausalSpanAuthenticationError(Lfm2CausalSpanLinearizationError):
    """One source-span evidence object or derived tensor bundle is stale."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for LFM2 causal-span linearization."
        ) from exc
    return torch


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2CausalSpanAuthenticationError(
            "Causal-span evidence must be canonical finite JSON."
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CausalSpanAuthenticationError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _boundary_role(value: Any) -> Lfm2CausalSpanBoundaryRole:
    result = str(value).strip().lower()
    if result not in _BOUNDARY_ROLES:
        raise Lfm2CausalSpanLinearizationError(
            "boundary_role must be pre_operator, post_operator, or post_layer."
        )
    return result  # type: ignore[return-value]


def _owned_runtime_history(value: Any, label: str) -> Any:
    """Clone one exact cached runtime history without changing its dtype."""

    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 3
        or value.numel() <= 0
        or not torch.is_floating_point(value)
        or value.is_complex()
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        raise Lfm2CausalSpanLinearizationError(
            f"{label} must be one detached finite dense rank-three history."
        )
    result = value.detach().to(device="cpu").contiguous().clone()
    result.requires_grad_(False)
    return result


def _owned_fp64(value: Any, label: str, *, shape: tuple[int, ...] | None = None) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 3
        or value.numel() <= 0
        or not torch.is_floating_point(value)
        or value.is_complex()
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value.detach()).all().item())
        or (shape is not None and tuple(value.shape) != shape)
    ):
        expected = "" if shape is None else f" with shape {shape!r}"
        raise Lfm2CausalSpanLinearizationError(
            f"{label} must be one detached finite dense rank-three tensor{expected}."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()
    result.requires_grad_(False)
    return result


def _tensor_hashes(values: Sequence[Any]) -> list[str]:
    return [fingerprint_lfm2_global_residual_tensor(value) for value in values]


def _owned_parameter_state(value: Any, label: str) -> Mapping[str, Any]:
    torch = _torch()
    if not isinstance(value, Mapping) or not value:
        raise Lfm2CausalSpanLinearizationError(
            f"{label} must be one nonempty parameter-state mapping."
        )
    result = {}
    for raw_name, raw_tensor in sorted(value.items(), key=lambda item: str(item[0])):
        name = str(raw_name).strip()
        if (
            not name
            or name in result
            or not isinstance(raw_tensor, torch.Tensor)
            or raw_tensor.layout != torch.strided
            or raw_tensor.device.type == "meta"
            or raw_tensor.numel() <= 0
            or raw_tensor.requires_grad
            or raw_tensor.grad_fn is not None
            or (
                (torch.is_floating_point(raw_tensor) or raw_tensor.is_complex())
                and not bool(torch.isfinite(raw_tensor.detach()).all().item())
            )
        ):
            raise Lfm2CausalSpanLinearizationError(
                f"{label} contains an invalid parameter tensor."
            )
        tensor = raw_tensor.detach().to(device="cpu").contiguous().clone()
        tensor.requires_grad_(False)
        result[name] = tensor
    return MappingProxyType(result)


def _parameter_state_sha256(value: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {
            name: fingerprint_lfm2_global_residual_tensor(tensor)
            for name, tensor in sorted(value.items())
        }
    )


def _owned_json_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2CausalSpanLinearizationError(
            f"{label} must be one canonical JSON mapping."
        )
    try:
        copied = json.loads(
            json.dumps(
                dict(value),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise Lfm2CausalSpanLinearizationError(
            f"{label} must be one canonical finite JSON mapping."
        ) from exc
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class Lfm2SourceSpanDirectLayerRecord:
    """One absolute checkpoint layer and its independently owned parameters."""

    source_layer_index: int
    operator_kind: str
    parameter_state: Mapping[str, Any] = field(repr=False, compare=False)
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_layer_index, bool)
            or not isinstance(self.source_layer_index, int)
            or self.source_layer_index < 0
        ):
            raise Lfm2CausalSpanLinearizationError(
                "source_layer_index must be a nonnegative integer."
            )
        kind = str(self.operator_kind).strip().lower()
        if kind not in {"full_attention", "short_conv"}:
            raise Lfm2CausalSpanLinearizationError(
                "operator_kind must be full_attention or short_conv."
            )
        object.__setattr__(self, "operator_kind", kind)
        object.__setattr__(
            self,
            "parameter_state",
            _owned_parameter_state(
                self.parameter_state,
                f"direct layer {self.source_layer_index} parameter_state",
            ),
        )
        expected = self.recompute_authentication_sha256()
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2CausalSpanAuthenticationError(
                "Direct layer record authentication_sha256 is incorrect."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    @property
    def parameter_state_sha256(self) -> str:
        return _parameter_state_sha256(self.parameter_state)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CAUSAL_SPAN_LINEARIZATION_SCHEMA,
            "kind": "direct_parent_layer_record",
            "source_layer_index": self.source_layer_index,
            "operator_kind": self.operator_kind,
            "parameter_state_sha256": self.parameter_state_sha256,
            "parameter_tensor_sha256": {
                name: fingerprint_lfm2_global_residual_tensor(value)
                for name, value in sorted(self.parameter_state.items())
            },
        }

    def recompute_authentication_sha256(self) -> str:
        return _canonical_sha256(self.authentication_body())

    def verify_authentication(self) -> bool:
        try:
            replayed = Lfm2SourceSpanDirectLayerRecord(
                source_layer_index=self.source_layer_index,
                operator_kind=self.operator_kind,
                parameter_state=self.parameter_state,
            )
            return replayed.authentication_sha256 == self.authentication_sha256
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False


@dataclass(frozen=True, slots=True)
class Lfm2SourceSpanDirectParentReceipt:
    """Replayable whole-parent checkpoint identity with absolute layer records."""

    parent_graph_fingerprint: str
    checkpoint_configuration: Mapping[str, Any]
    layers: tuple[Lfm2SourceSpanDirectLayerRecord, ...] = field(
        repr=False,
        compare=False,
    )
    parent_model_identity_sha256: str = field(init=False)
    checkpoint_manifest_sha256: str = field(init=False)
    session_sha256: str = field(init=False)
    authentication_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        graph = _digest(self.parent_graph_fingerprint, "parent graph fingerprint")
        configuration = _owned_json_mapping(
            self.checkpoint_configuration,
            "checkpoint_configuration",
        )
        layers = tuple(self.layers)
        if (
            not layers
            or any(
                not isinstance(value, Lfm2SourceSpanDirectLayerRecord)
                or not value.verify_authentication()
                for value in layers
            )
            or tuple(value.source_layer_index for value in layers)
            != tuple(range(len(layers)))
        ):
            raise Lfm2CausalSpanAuthenticationError(
                "Direct parent layers must exactly cover the checkpoint in order."
            )
        configured_depth = configuration.get("num_hidden_layers")
        if configured_depth is not None and configured_depth != len(layers):
            raise Lfm2CausalSpanAuthenticationError(
                "Direct parent checkpoint depth differs from its layer records."
            )
        object.__setattr__(self, "parent_graph_fingerprint", graph)
        object.__setattr__(self, "checkpoint_configuration", configuration)
        object.__setattr__(self, "layers", layers)
        manifest_body = {
            "schema": LFM2_CAUSAL_SPAN_LINEARIZATION_SCHEMA,
            "kind": "direct_parent_checkpoint_manifest",
            "parent_graph_fingerprint": graph,
            "checkpoint_configuration": dict(configuration),
            "ordered_layer_sha256": [value.authentication_sha256 for value in layers],
        }
        manifest = _canonical_sha256(manifest_body)
        identity = _canonical_sha256(
            {
                "schema": LFM2_CAUSAL_SPAN_LINEARIZATION_SCHEMA,
                "kind": "direct_parent_model_identity",
                "checkpoint_manifest_sha256": manifest,
                "ordered_layer_parameter_sha256": [
                    value.parameter_state_sha256 for value in layers
                ],
            }
        )
        session = _canonical_sha256(
            {
                "schema": LFM2_CAUSAL_SPAN_LINEARIZATION_SCHEMA,
                "kind": "direct_parent_replay_session",
                "parent_model_identity_sha256": identity,
                "checkpoint_manifest_sha256": manifest,
                "loaded_absolute_layer_indices": list(range(len(layers))),
                "operation_audit": dict(_ZERO_OPERATION_AUDIT),
            }
        )
        object.__setattr__(self, "checkpoint_manifest_sha256", manifest)
        object.__setattr__(self, "parent_model_identity_sha256", identity)
        object.__setattr__(self, "session_sha256", session)
        object.__setattr__(
            self,
            "authentication_sha256",
            self.recompute_authentication_sha256(),
        )

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CAUSAL_SPAN_LINEARIZATION_SCHEMA,
            "kind": "direct_parent_receipt",
            "parent_graph_fingerprint": self.parent_graph_fingerprint,
            "parent_model_identity_sha256": self.parent_model_identity_sha256,
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "session_sha256": self.session_sha256,
            "checkpoint_configuration": dict(self.checkpoint_configuration),
            "ordered_layer_sha256": [
                value.authentication_sha256 for value in self.layers
            ],
            "ordered_layer_parameter_sha256": [
                value.parameter_state_sha256 for value in self.layers
            ],
            "operation_audit": dict(_ZERO_OPERATION_AUDIT),
        }

    def recompute_authentication_sha256(self) -> str:
        return _canonical_sha256(self.authentication_body())

    def verify_authentication(self) -> bool:
        try:
            replayed = Lfm2SourceSpanDirectParentReceipt(
                parent_graph_fingerprint=self.parent_graph_fingerprint,
                checkpoint_configuration=self.checkpoint_configuration,
                layers=self.layers,
            )
            return (
                all(value.verify_authentication() for value in self.layers)
                and replayed.parent_model_identity_sha256
                == self.parent_model_identity_sha256
                and replayed.checkpoint_manifest_sha256
                == self.checkpoint_manifest_sha256
                and replayed.session_sha256 == self.session_sha256
                and replayed.authentication_sha256 == self.authentication_sha256
            )
        except (RuntimeError, TypeError, ValueError):
            return False


@dataclass(frozen=True, slots=True)
class Lfm2SourceSpanPlanReceipt:
    """Replayable complete depth plan bound to one parent identity."""

    parent_model_identity_sha256: str
    parent_graph_fingerprint: str
    ordered_source_spans: tuple[tuple[int, ...], ...]
    target_configuration: Mapping[str, Any]
    authentication_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        identity = _digest(
            self.parent_model_identity_sha256,
            "plan parent model identity",
        )
        graph = _digest(self.parent_graph_fingerprint, "plan parent graph")
        spans = tuple(tuple(span) for span in self.ordered_source_spans)
        flattened = tuple(index for span in spans for index in span)
        if (
            not spans
            or any(not span for span in spans)
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in flattened
            )
            or flattened != tuple(range(len(flattened)))
        ):
            raise Lfm2CausalSpanLinearizationError(
                "Plan spans must exactly partition every source layer in order."
            )
        configuration = _owned_json_mapping(
            self.target_configuration,
            "target_configuration",
        )
        configured_depth = configuration.get("num_hidden_layers")
        if configured_depth is not None and configured_depth != len(spans):
            raise Lfm2CausalSpanLinearizationError(
                "Target configuration depth differs from the source-span plan."
            )
        object.__setattr__(self, "parent_model_identity_sha256", identity)
        object.__setattr__(self, "parent_graph_fingerprint", graph)
        object.__setattr__(self, "ordered_source_spans", spans)
        object.__setattr__(self, "target_configuration", configuration)
        object.__setattr__(
            self,
            "authentication_sha256",
            self.recompute_authentication_sha256(),
        )

    @property
    def source_layer_count(self) -> int:
        return sum(len(value) for value in self.ordered_source_spans)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CAUSAL_SPAN_LINEARIZATION_SCHEMA,
            "kind": "source_span_plan_receipt",
            "parent_model_identity_sha256": self.parent_model_identity_sha256,
            "parent_graph_fingerprint": self.parent_graph_fingerprint,
            "ordered_source_spans": [
                list(value) for value in self.ordered_source_spans
            ],
            "source_layer_count": self.source_layer_count,
            "target_configuration": dict(self.target_configuration),
        }

    def recompute_authentication_sha256(self) -> str:
        return _canonical_sha256(self.authentication_body())

    def verify_authentication(self) -> bool:
        try:
            replayed = Lfm2SourceSpanPlanReceipt(
                parent_model_identity_sha256=self.parent_model_identity_sha256,
                parent_graph_fingerprint=self.parent_graph_fingerprint,
                ordered_source_spans=self.ordered_source_spans,
                target_configuration=self.target_configuration,
            )
            return replayed.authentication_sha256 == self.authentication_sha256
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False


@dataclass(frozen=True, slots=True)
class Lfm2SourceSpanBatchChain:
    """Owned cache history for one batch and one absolute source span.

    Every absolute layer identifier, operator kind, pre-operator residual,
    post-operator residual, and post-layer residual participates in the seal.
    The final post-layer residual is intentionally included even though it has
    no following adjacency edge.
    """

    batch_index: int
    semantic_alignment_sha256: str
    source_layer_indices: tuple[int, ...]
    operator_kinds: tuple[str, ...]
    layer_pre_residuals: tuple[Any, ...] = field(repr=False, compare=False)
    layer_post_operator_residuals: tuple[Any, ...] = field(
        repr=False,
        compare=False,
    )
    layer_post_residuals: tuple[Any, ...] = field(repr=False, compare=False)
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        torch = _torch()
        if (
            isinstance(self.batch_index, bool)
            or not isinstance(self.batch_index, int)
            or self.batch_index < 0
        ):
            raise Lfm2CausalSpanLinearizationError(
                "batch_index must be a nonnegative integer."
            )
        object.__setattr__(
            self,
            "semantic_alignment_sha256",
            _digest(self.semantic_alignment_sha256, "semantic alignment"),
        )
        source = tuple(self.source_layer_indices)
        if (
            not source
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in source
            )
            or source != tuple(range(source[0], source[-1] + 1))
        ):
            raise Lfm2CausalSpanLinearizationError(
                "source_layer_indices must be one nonempty contiguous absolute span."
            )
        kinds = tuple(str(value).strip().lower() for value in self.operator_kinds)
        if len(kinds) != len(source) or any(
            value not in {"full_attention", "short_conv"} for value in kinds
        ):
            raise Lfm2CausalSpanLinearizationError(
                "operator_kinds must identify every absolute source layer."
            )
        pre = tuple(
            _owned_runtime_history(value, f"layer_pre_residuals[{index}]")
            for index, value in enumerate(self.layer_pre_residuals)
        )
        post_operator = tuple(
            _owned_runtime_history(
                value,
                f"layer_post_operator_residuals[{index}]",
            )
            for index, value in enumerate(self.layer_post_operator_residuals)
        )
        post = tuple(
            _owned_runtime_history(value, f"layer_post_residuals[{index}]")
            for index, value in enumerate(self.layer_post_residuals)
        )
        if not pre or not (len(pre) == len(post_operator) == len(post) == len(source)):
            raise Lfm2CausalSpanLinearizationError(
                "A source batch chain needs complete pre/operator/post history."
            )
        reference_shape = tuple(pre[0].shape)
        reference_dtype = pre[0].dtype
        if any(
            tuple(value.shape) != reference_shape or value.dtype != reference_dtype
            for value in (*pre, *post_operator, *post)
        ):
            raise Lfm2CausalSpanLinearizationError(
                "Source batch chain histories disagree on runtime shape or dtype."
            )
        for layer_index in range(len(pre) - 1):
            if not bool(torch.equal(post[layer_index], pre[layer_index + 1])):
                raise Lfm2CausalSpanAuthenticationError(
                    "Cached source histories do not form one exact adjacent "
                    f"residual chain at local layer {layer_index}."
                )
        object.__setattr__(self, "source_layer_indices", source)
        object.__setattr__(self, "operator_kinds", kinds)
        object.__setattr__(self, "layer_pre_residuals", pre)
        object.__setattr__(self, "layer_post_operator_residuals", post_operator)
        object.__setattr__(self, "layer_post_residuals", post)
        expected = self.recompute_authentication_sha256()
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2CausalSpanAuthenticationError(
                "Source batch chain authentication_sha256 is incorrect."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    @property
    def layer_count(self) -> int:
        return len(self.source_layer_indices)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CAUSAL_SPAN_LINEARIZATION_SCHEMA,
            "kind": "source_span_batch_chain",
            "batch_index": self.batch_index,
            "semantic_alignment_sha256": self.semantic_alignment_sha256,
            "source_layer_indices": list(self.source_layer_indices),
            "operator_kinds": list(self.operator_kinds),
            "layer_count": self.layer_count,
            "runtime_dtype": str(self.layer_pre_residuals[0].dtype),
            "runtime_shape": [
                int(value) for value in self.layer_pre_residuals[0].shape
            ],
            "layer_pre_residual_sha256": _tensor_hashes(self.layer_pre_residuals),
            "layer_post_operator_residual_sha256": _tensor_hashes(
                self.layer_post_operator_residuals
            ),
            "layer_post_residual_sha256": _tensor_hashes(self.layer_post_residuals),
            "terminal_post_layer_residual_included": True,
            "exact_adjacent_runtime_chain": True,
        }

    def recompute_authentication_sha256(self) -> str:
        return _canonical_sha256(self.authentication_body())

    def verify_authentication(self) -> bool:
        torch = _torch()
        try:
            replayed = Lfm2SourceSpanBatchChain(
                batch_index=self.batch_index,
                semantic_alignment_sha256=self.semantic_alignment_sha256,
                source_layer_indices=self.source_layer_indices,
                operator_kinds=self.operator_kinds,
                layer_pre_residuals=self.layer_pre_residuals,
                layer_post_operator_residuals=self.layer_post_operator_residuals,
                layer_post_residuals=self.layer_post_residuals,
            )
            return (
                self.layer_count > 0
                and len(self.operator_kinds) == self.layer_count
                and len(self.layer_pre_residuals) == self.layer_count
                and len(self.layer_post_operator_residuals) == self.layer_count
                and len(self.layer_post_residuals) == self.layer_count
                and all(
                    bool(
                        torch.equal(
                            self.layer_post_residuals[index],
                            self.layer_pre_residuals[index + 1],
                        )
                    )
                    for index in range(self.layer_count - 1)
                )
                and replayed.authentication_sha256 == self.authentication_sha256
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False


def _clone_source_span_batch_chain(
    value: Lfm2SourceSpanBatchChain,
) -> Lfm2SourceSpanBatchChain:
    if not isinstance(value, Lfm2SourceSpanBatchChain) or not (
        value.verify_authentication()
    ):
        raise Lfm2CausalSpanAuthenticationError(
            "A cache receipt can only own authenticated source batch chains."
        )
    return Lfm2SourceSpanBatchChain(
        batch_index=value.batch_index,
        semantic_alignment_sha256=value.semantic_alignment_sha256,
        source_layer_indices=value.source_layer_indices,
        operator_kinds=value.operator_kinds,
        layer_pre_residuals=value.layer_pre_residuals,
        layer_post_operator_residuals=value.layer_post_operator_residuals,
        layer_post_residuals=value.layer_post_residuals,
        authentication_sha256=value.authentication_sha256,
    )


@dataclass(frozen=True, slots=True)
class Lfm2SourceSpanCacheReceipt:
    """Independently replayable cache authority for one absolute source span."""

    parent_model_identity_sha256: str
    layer_topology: tuple[str, ...]
    batch_chains: tuple[Lfm2SourceSpanBatchChain, ...] = field(
        repr=False,
        compare=False,
    )
    cache_key: str = field(init=False)
    manifest_sha256: str = field(init=False)
    session_sha256: str = field(init=False)
    authentication_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        identity = _digest(
            self.parent_model_identity_sha256,
            "cache parent model identity",
        )
        topology = tuple(str(value).strip().lower() for value in self.layer_topology)
        if not topology or any(
            value not in {"full_attention", "short_conv"} for value in topology
        ):
            raise Lfm2CausalSpanLinearizationError(
                "layer_topology must identify every parent operator in order."
            )
        chains = tuple(
            _clone_source_span_batch_chain(value) for value in self.batch_chains
        )
        if (
            not chains
            or tuple(value.batch_index for value in chains) != tuple(range(len(chains)))
            or any(
                value.source_layer_indices != chains[0].source_layer_indices
                for value in chains
            )
        ):
            raise Lfm2CausalSpanAuthenticationError(
                "Cache batch chains must be canonical and share one absolute span."
            )
        source = chains[0].source_layer_indices
        if source[-1] >= len(topology) or any(
            chain.operator_kinds
            != tuple(topology[index] for index in chain.source_layer_indices)
            for chain in chains
        ):
            raise Lfm2CausalSpanAuthenticationError(
                "Cache histories do not join the declared absolute layer topology."
            )
        object.__setattr__(self, "parent_model_identity_sha256", identity)
        object.__setattr__(self, "layer_topology", topology)
        object.__setattr__(self, "batch_chains", chains)
        cache_key = _canonical_sha256(
            {
                "schema": LFM2_CAUSAL_SPAN_LINEARIZATION_SCHEMA,
                "kind": "source_span_cache_key",
                "parent_model_identity_sha256": identity,
                "source_layer_indices": list(source),
                "semantic_alignment_sha256": [
                    value.semantic_alignment_sha256 for value in chains
                ],
            }
        )
        manifest = _canonical_sha256(
            {
                "schema": LFM2_CAUSAL_SPAN_LINEARIZATION_SCHEMA,
                "kind": "source_span_cache_manifest",
                "cache_key": cache_key,
                "parent_model_identity_sha256": identity,
                "layer_topology": list(topology),
                "ordered_batch_chain_sha256": [
                    value.authentication_sha256 for value in chains
                ],
            }
        )
        session = _canonical_sha256(
            {
                "schema": LFM2_CAUSAL_SPAN_LINEARIZATION_SCHEMA,
                "kind": "source_span_cache_replay_session",
                "cache_key": cache_key,
                "manifest_sha256": manifest,
                "source_layer_indices": list(source),
                "operation_audit": dict(_ZERO_OPERATION_AUDIT),
            }
        )
        object.__setattr__(self, "cache_key", cache_key)
        object.__setattr__(self, "manifest_sha256", manifest)
        object.__setattr__(self, "session_sha256", session)
        object.__setattr__(
            self,
            "authentication_sha256",
            self.recompute_authentication_sha256(),
        )

    @property
    def source_layer_indices(self) -> tuple[int, ...]:
        return self.batch_chains[0].source_layer_indices

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CAUSAL_SPAN_LINEARIZATION_SCHEMA,
            "kind": "source_span_cache_receipt",
            "parent_model_identity_sha256": self.parent_model_identity_sha256,
            "layer_topology": list(self.layer_topology),
            "source_layer_indices": list(self.source_layer_indices),
            "cache_key": self.cache_key,
            "manifest_sha256": self.manifest_sha256,
            "session_sha256": self.session_sha256,
            "ordered_batch_chain_sha256": [
                value.authentication_sha256 for value in self.batch_chains
            ],
            "operation_audit": dict(_ZERO_OPERATION_AUDIT),
        }

    def recompute_authentication_sha256(self) -> str:
        return _canonical_sha256(self.authentication_body())

    def verify_authentication(self) -> bool:
        try:
            replayed = Lfm2SourceSpanCacheReceipt(
                parent_model_identity_sha256=self.parent_model_identity_sha256,
                layer_topology=self.layer_topology,
                batch_chains=self.batch_chains,
            )
            return (
                replayed.cache_key == self.cache_key
                and replayed.manifest_sha256 == self.manifest_sha256
                and replayed.session_sha256 == self.session_sha256
                and replayed.authentication_sha256 == self.authentication_sha256
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False


def lfm2_source_span_provenance_root_sha256(
    plan_receipt: Lfm2SourceSpanPlanReceipt,
    cache_receipt: Lfm2SourceSpanCacheReceipt,
    direct_parent_receipt: Lfm2SourceSpanDirectParentReceipt,
) -> str:
    """Compute the externally pinned root of three replayable authorities."""

    receipts = (
        (plan_receipt, Lfm2SourceSpanPlanReceipt, "plan"),
        (cache_receipt, Lfm2SourceSpanCacheReceipt, "cache"),
        (direct_parent_receipt, Lfm2SourceSpanDirectParentReceipt, "direct parent"),
    )
    if any(
        not isinstance(value, expected) or not value.verify_authentication()
        for value, expected, _label in receipts
    ):
        raise Lfm2CausalSpanAuthenticationError(
            "A provenance root requires three authenticated replay receipts."
        )
    return _canonical_sha256(
        {
            "schema": LFM2_CAUSAL_SPAN_LINEARIZATION_SCHEMA,
            "kind": "source_span_provenance_root",
            "plan_receipt_sha256": plan_receipt.authentication_sha256,
            "cache_receipt_sha256": cache_receipt.authentication_sha256,
            "direct_parent_receipt_sha256": (
                direct_parent_receipt.authentication_sha256
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class Lfm2SourceSpanPrimalEvidence:
    """A source span joined to replayed plan, cache, and parent authorities."""

    span: Lfm2ModelFoldSpan
    snapshot: Lfm2WholeStackPrimalSnapshot = field(repr=False, compare=False)
    batch_chains: tuple[Lfm2SourceSpanBatchChain, ...] = field(
        repr=False,
        compare=False,
    )
    plan_receipt: Lfm2SourceSpanPlanReceipt
    cache_receipt: Lfm2SourceSpanCacheReceipt = field(repr=False, compare=False)
    direct_parent_receipt: Lfm2SourceSpanDirectParentReceipt = field(
        repr=False,
        compare=False,
    )
    expected_provenance_root_sha256: str
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.span, Lfm2ModelFoldSpan):
            raise TypeError("span must be Lfm2ModelFoldSpan.")
        snapshot = self.snapshot
        if not isinstance(snapshot, Lfm2WholeStackPrimalSnapshot) or not (
            snapshot.verify_authentication()
        ):
            raise Lfm2CausalSpanAuthenticationError(
                "snapshot must be an authenticated whole-stack primal."
            )
        if snapshot.terminal_present:
            raise Lfm2CausalSpanLinearizationError(
                "A source span ends at post-layer residual, before terminal norm."
            )
        chains = tuple(
            _clone_source_span_batch_chain(value) for value in self.batch_chains
        )
        object.__setattr__(self, "batch_chains", chains)
        object.__setattr__(
            self,
            "expected_provenance_root_sha256",
            _digest(
                self.expected_provenance_root_sha256,
                "expected provenance root",
            ),
        )
        self._require_replayed_joins()
        expected = self.recompute_authentication_sha256()
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2CausalSpanAuthenticationError(
                "Source span primal authentication_sha256 is incorrect."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    @property
    def absolute_to_local_layer(self) -> Mapping[int, int]:
        return MappingProxyType(
            {
                absolute: local
                for local, absolute in enumerate(self.span.source_layer_indices)
            }
        )

    def _require_replayed_joins(self) -> None:
        torch = _torch()
        plan = self.plan_receipt
        cache = self.cache_receipt
        direct = self.direct_parent_receipt
        if (
            not isinstance(plan, Lfm2SourceSpanPlanReceipt)
            or not plan.verify_authentication()
            or not isinstance(cache, Lfm2SourceSpanCacheReceipt)
            or not cache.verify_authentication()
            or not isinstance(direct, Lfm2SourceSpanDirectParentReceipt)
            or not direct.verify_authentication()
        ):
            raise Lfm2CausalSpanAuthenticationError(
                "Source span evidence requires replayable authenticated receipts."
            )
        root = lfm2_source_span_provenance_root_sha256(plan, cache, direct)
        if root != self.expected_provenance_root_sha256:
            raise Lfm2CausalSpanAuthenticationError(
                "Replayed receipts differ from the externally pinned provenance root."
            )
        identity = direct.parent_model_identity_sha256
        if (
            plan.parent_model_identity_sha256 != identity
            or cache.parent_model_identity_sha256 != identity
            or plan.parent_graph_fingerprint != direct.parent_graph_fingerprint
            or plan.source_layer_count != len(direct.layers)
            or len(cache.layer_topology) != len(direct.layers)
        ):
            raise Lfm2CausalSpanAuthenticationError(
                "Plan, cache, and direct-parent receipts do not describe one parent."
            )
        if (
            self.span.target_layer_index >= len(plan.ordered_source_spans)
            or plan.ordered_source_spans[self.span.target_layer_index]
            != self.span.source_layer_indices
            or cache.source_layer_indices != self.span.source_layer_indices
        ):
            raise Lfm2CausalSpanAuthenticationError(
                "The requested span does not join the replayed complete fold plan."
            )
        snapshot = self.snapshot
        if (
            snapshot.layer_count != len(self.span.source_layer_indices)
            or len(self.batch_chains) != len(snapshot.batches)
            or len(self.batch_chains) != len(cache.batch_chains)
            or tuple(value.batch_index for value in self.batch_chains)
            != tuple(range(len(self.batch_chains)))
        ):
            raise Lfm2CausalSpanAuthenticationError(
                "Local primals, histories, and cache receipt have different geometry."
            )
        for local_index, absolute_index in enumerate(self.span.source_layer_indices):
            direct_layer = direct.layers[absolute_index]
            cache_kind = cache.layer_topology[absolute_index]
            if (
                direct_layer.source_layer_index != absolute_index
                or direct_layer.operator_kind != cache_kind
                or direct_layer.parameter_state_sha256
                != snapshot.layer_parameter_sha256[local_index]
                or any(
                    batch.macrocells[local_index].operator_kind != cache_kind
                    for batch in snapshot.batches
                )
            ):
                raise Lfm2CausalSpanAuthenticationError(
                    "Direct parameters or operator topology do not join the local "
                    f"snapshot at absolute layer {absolute_index}."
                )
        for batch_index, (chain, authoritative, batch) in enumerate(
            zip(
                self.batch_chains,
                cache.batch_chains,
                snapshot.batches,
                strict=True,
            )
        ):
            if (
                not chain.verify_authentication()
                or chain.authentication_sha256 != authoritative.authentication_sha256
                or chain.source_layer_indices != self.span.source_layer_indices
                or chain.operator_kinds
                != tuple(
                    cache.layer_topology[index]
                    for index in self.span.source_layer_indices
                )
            ):
                raise Lfm2CausalSpanAuthenticationError(
                    f"Batch {batch_index} is not the authoritative cached span."
                )
            for local_index, (cached_pre, macrocell) in enumerate(
                zip(chain.layer_pre_residuals, batch.macrocells, strict=True)
            ):
                converted = cached_pre.to(device="cpu", dtype=torch.float64)
                if tuple(converted.shape) != tuple(
                    macrocell.operator.residual.shape
                ) or not bool(torch.equal(converted, macrocell.operator.residual)):
                    raise Lfm2CausalSpanAuthenticationError(
                        "Analytic source primal is not centered on cached "
                        f"residual.pre for batch {batch_index}, local layer "
                        f"{local_index}."
                    )

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CAUSAL_SPAN_LINEARIZATION_SCHEMA,
            "kind": "source_span_primal_evidence",
            "span": self.span.as_dict(),
            "absolute_to_local_layer": {
                str(key): value for key, value in self.absolute_to_local_layer.items()
            },
            "snapshot_sha256": self.snapshot.authentication_sha256,
            "snapshot_layer_parameter_sha256": list(
                self.snapshot.layer_parameter_sha256
            ),
            "batch_chain_sha256": [
                value.authentication_sha256 for value in self.batch_chains
            ],
            "plan_receipt_sha256": self.plan_receipt.authentication_sha256,
            "cache_receipt_sha256": self.cache_receipt.authentication_sha256,
            "direct_parent_receipt_sha256": (
                self.direct_parent_receipt.authentication_sha256
            ),
            "expected_provenance_root_sha256": (self.expected_provenance_root_sha256),
            "primal_semantics": (
                "authenticated_runtime_history_centered_continuous_fp64_"
                "linearization_with_inter_layer_cast_straight_through_identity"
            ),
            "terminal_endpoint": "source_span_last_layer_post_layer_residual",
            "operation_audit": dict(_ZERO_OPERATION_AUDIT),
        }

    def recompute_authentication_sha256(self) -> str:
        return _canonical_sha256(self.authentication_body())

    def verify_authentication(self) -> bool:
        try:
            if not self.snapshot.verify_authentication():
                return False
            self._require_replayed_joins()
            return self.authentication_sha256 == self.recompute_authentication_sha256()
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            return False


@dataclass(frozen=True, slots=True)
class Lfm2CausalSpanSeed:
    """One authenticated full-history residual seed at a source boundary."""

    evidence_sha256: str
    source_layer_index: int
    boundary_role: Lfm2CausalSpanBoundaryRole
    batch_tangents: tuple[Any, ...] = field(repr=False, compare=False)
    semantic_alignment_sha256: tuple[str, ...]
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_layer_index, bool)
            or not isinstance(self.source_layer_index, int)
            or self.source_layer_index < 0
        ):
            raise Lfm2CausalSpanLinearizationError(
                "source_layer_index must be a nonnegative integer."
            )
        object.__setattr__(
            self,
            "evidence_sha256",
            _digest(self.evidence_sha256, "source span evidence"),
        )
        object.__setattr__(self, "boundary_role", _boundary_role(self.boundary_role))
        tangents = tuple(
            _owned_fp64(value, f"batch_tangents[{index}]")
            for index, value in enumerate(self.batch_tangents)
        )
        if not tangents:
            raise Lfm2CausalSpanLinearizationError(
                "A causal-span seed needs at least one batch tangent."
            )
        alignments = tuple(
            _digest(value, f"semantic_alignment_sha256[{index}]")
            for index, value in enumerate(self.semantic_alignment_sha256)
        )
        if len(alignments) != len(tangents):
            raise Lfm2CausalSpanLinearizationError(
                "Seed tensors and semantic alignment digests must align."
            )
        object.__setattr__(self, "batch_tangents", tangents)
        object.__setattr__(self, "semantic_alignment_sha256", alignments)
        expected = self.recompute_authentication_sha256()
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2CausalSpanAuthenticationError(
                "Causal-span seed authentication_sha256 is incorrect."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CAUSAL_SPAN_LINEARIZATION_SCHEMA,
            "kind": "source_span_boundary_seed",
            "evidence_sha256": self.evidence_sha256,
            "source_layer_index": self.source_layer_index,
            "boundary_role": self.boundary_role,
            "batch_tangent_sha256": _tensor_hashes(self.batch_tangents),
            "semantic_alignment_sha256": list(self.semantic_alignment_sha256),
        }

    def recompute_authentication_sha256(self) -> str:
        return _canonical_sha256(self.authentication_body())

    def verify_authentication(self) -> bool:
        try:
            return self.authentication_sha256 == self.recompute_authentication_sha256()
        except (RuntimeError, TypeError, ValueError):
            return False


@dataclass(frozen=True, slots=True)
class Lfm2CausalSpanEndpointCotangent:
    """One sealed cotangent at the terminal post-layer span boundary."""

    evidence_sha256: str
    batch_cotangents: tuple[Any, ...] = field(repr=False, compare=False)
    semantic_alignment_sha256: tuple[str, ...]
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_sha256",
            _digest(self.evidence_sha256, "source span evidence"),
        )
        cotangents = tuple(
            _owned_fp64(value, f"batch_cotangents[{index}]")
            for index, value in enumerate(self.batch_cotangents)
        )
        alignments = tuple(
            _digest(value, f"semantic_alignment_sha256[{index}]")
            for index, value in enumerate(self.semantic_alignment_sha256)
        )
        if not cotangents or len(cotangents) != len(alignments):
            raise Lfm2CausalSpanLinearizationError(
                "Endpoint cotangents and semantic alignments must be nonempty "
                "and aligned."
            )
        object.__setattr__(self, "batch_cotangents", cotangents)
        object.__setattr__(self, "semantic_alignment_sha256", alignments)
        expected = self.recompute_authentication_sha256()
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2CausalSpanAuthenticationError(
                "Endpoint cotangent authentication_sha256 is incorrect."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CAUSAL_SPAN_LINEARIZATION_SCHEMA,
            "kind": "source_span_endpoint_cotangent",
            "evidence_sha256": self.evidence_sha256,
            "batch_cotangent_sha256": _tensor_hashes(self.batch_cotangents),
            "semantic_alignment_sha256": list(self.semantic_alignment_sha256),
        }

    def recompute_authentication_sha256(self) -> str:
        return _canonical_sha256(self.authentication_body())

    def verify_authentication(self) -> bool:
        try:
            return self.authentication_sha256 == self.recompute_authentication_sha256()
        except (RuntimeError, TypeError, ValueError):
            return False


@dataclass(frozen=True, slots=True)
class Lfm2CausalSpanEndpointTangent:
    """Authenticated endpoint image of one causal-span boundary seed."""

    evidence_sha256: str
    seed_sha256: str
    batch_tangents: tuple[Any, ...] = field(repr=False, compare=False)
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_sha256",
            _digest(self.evidence_sha256, "source span evidence"),
        )
        object.__setattr__(
            self,
            "seed_sha256",
            _digest(self.seed_sha256, "source span seed"),
        )
        tangents = tuple(
            _owned_fp64(value, f"endpoint_batch_tangents[{index}]")
            for index, value in enumerate(self.batch_tangents)
        )
        if not tangents:
            raise Lfm2CausalSpanLinearizationError(
                "Endpoint tangent bundle cannot be empty."
            )
        object.__setattr__(self, "batch_tangents", tangents)
        expected = self.recompute_authentication_sha256()
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2CausalSpanAuthenticationError(
                "Endpoint tangent authentication_sha256 is incorrect."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CAUSAL_SPAN_LINEARIZATION_SCHEMA,
            "kind": "source_span_endpoint_tangent",
            "evidence_sha256": self.evidence_sha256,
            "seed_sha256": self.seed_sha256,
            "batch_tangent_sha256": _tensor_hashes(self.batch_tangents),
            "operation_audit": dict(_ZERO_OPERATION_AUDIT),
        }

    def recompute_authentication_sha256(self) -> str:
        return _canonical_sha256(self.authentication_body())

    def verify_authentication(self) -> bool:
        try:
            return self.authentication_sha256 == self.recompute_authentication_sha256()
        except (RuntimeError, TypeError, ValueError):
            return False


@dataclass(frozen=True, slots=True)
class Lfm2CausalSpanBoundaryCotangent:
    """Authenticated transpose image at one selected source boundary."""

    evidence_sha256: str
    endpoint_cotangent_sha256: str
    source_layer_index: int
    boundary_role: Lfm2CausalSpanBoundaryRole
    batch_cotangents: tuple[Any, ...] = field(repr=False, compare=False)
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_sha256",
            _digest(self.evidence_sha256, "source span evidence"),
        )
        object.__setattr__(
            self,
            "endpoint_cotangent_sha256",
            _digest(self.endpoint_cotangent_sha256, "endpoint cotangent"),
        )
        if (
            isinstance(self.source_layer_index, bool)
            or not isinstance(self.source_layer_index, int)
            or self.source_layer_index < 0
        ):
            raise Lfm2CausalSpanLinearizationError(
                "source_layer_index must be a nonnegative integer."
            )
        object.__setattr__(self, "boundary_role", _boundary_role(self.boundary_role))
        cotangents = tuple(
            _owned_fp64(value, f"boundary_batch_cotangents[{index}]")
            for index, value in enumerate(self.batch_cotangents)
        )
        if not cotangents:
            raise Lfm2CausalSpanLinearizationError(
                "Boundary cotangent bundle cannot be empty."
            )
        object.__setattr__(self, "batch_cotangents", cotangents)
        expected = self.recompute_authentication_sha256()
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2CausalSpanAuthenticationError(
                "Boundary cotangent authentication_sha256 is incorrect."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CAUSAL_SPAN_LINEARIZATION_SCHEMA,
            "kind": "source_span_boundary_cotangent",
            "evidence_sha256": self.evidence_sha256,
            "endpoint_cotangent_sha256": self.endpoint_cotangent_sha256,
            "source_layer_index": self.source_layer_index,
            "boundary_role": self.boundary_role,
            "batch_cotangent_sha256": _tensor_hashes(self.batch_cotangents),
            "operation_audit": dict(_ZERO_OPERATION_AUDIT),
        }

    def recompute_authentication_sha256(self) -> str:
        return _canonical_sha256(self.authentication_body())

    def verify_authentication(self) -> bool:
        try:
            return self.authentication_sha256 == self.recompute_authentication_sha256()
        except (RuntimeError, TypeError, ValueError):
            return False


class Lfm2CausalSpanEndpointOperator:
    """Matrix-free JVP/VJP from a source boundary to its span endpoint."""

    __slots__ = ("_evidence", "_zero_output_basis")

    def __init__(self, evidence: Lfm2SourceSpanPrimalEvidence) -> None:
        torch = _torch()
        if not isinstance(evidence, Lfm2SourceSpanPrimalEvidence) or not (
            evidence.verify_authentication()
        ):
            raise Lfm2CausalSpanAuthenticationError(
                "evidence must be an authenticated source-span primal."
            )
        self._evidence = evidence
        self._zero_output_basis = torch.zeros(
            (evidence.snapshot.hidden_size, 1),
            dtype=torch.float64,
            device="cpu",
        )

    @property
    def evidence(self) -> Lfm2SourceSpanPrimalEvidence:
        return self._evidence

    def _require_authenticated(self) -> None:
        if not self._evidence.verify_authentication():
            raise Lfm2CausalSpanAuthenticationError(
                "Source-span primal evidence changed after operator construction."
            )

    def _local_layer(self, source_layer_index: int) -> int:
        local = self._evidence.absolute_to_local_layer.get(source_layer_index)
        if local is None:
            raise Lfm2CausalSpanLinearizationError(
                f"Source layer {source_layer_index} is outside the sealed span."
            )
        return local

    def make_seed(
        self,
        *,
        source_layer_index: int,
        boundary_role: Lfm2CausalSpanBoundaryRole,
        batch_tangents: Sequence[Any],
    ) -> Lfm2CausalSpanSeed:
        """Own and seal one seed against this evidence and token geometry."""

        self._require_authenticated()
        local = self._local_layer(source_layer_index)
        values = tuple(batch_tangents)
        if len(values) != len(self._evidence.snapshot.batches):
            raise Lfm2CausalSpanLinearizationError(
                "Seed batch count differs from source-span evidence."
            )
        for batch_index, (value, batch) in enumerate(
            zip(values, self._evidence.snapshot.batches, strict=True)
        ):
            expected_shape = tuple(batch.macrocells[local].operator.residual.shape)
            _owned_fp64(
                value,
                f"seed batch {batch_index}",
                shape=expected_shape,
            )
        return Lfm2CausalSpanSeed(
            evidence_sha256=self._evidence.authentication_sha256,
            source_layer_index=source_layer_index,
            boundary_role=boundary_role,
            batch_tangents=values,
            semantic_alignment_sha256=tuple(
                value.semantic_alignment_sha256 for value in self._evidence.batch_chains
            ),
        )

    def make_endpoint_cotangent(
        self,
        batch_cotangents: Sequence[Any],
    ) -> Lfm2CausalSpanEndpointCotangent:
        """Own and seal one cotangent at the terminal post-layer boundary."""

        self._require_authenticated()
        values = tuple(batch_cotangents)
        if len(values) != len(self._evidence.snapshot.batches):
            raise Lfm2CausalSpanLinearizationError(
                "Endpoint cotangent batch count differs from source evidence."
            )
        for batch_index, (value, batch) in enumerate(
            zip(values, self._evidence.snapshot.batches, strict=True)
        ):
            expected_shape = tuple(batch.macrocells[-1].mlp.output.shape)
            _owned_fp64(
                value,
                f"endpoint cotangent batch {batch_index}",
                shape=expected_shape,
            )
        return Lfm2CausalSpanEndpointCotangent(
            evidence_sha256=self._evidence.authentication_sha256,
            batch_cotangents=values,
            semantic_alignment_sha256=tuple(
                value.semantic_alignment_sha256 for value in self._evidence.batch_chains
            ),
        )

    def _validate_seed(self, seed: Lfm2CausalSpanSeed) -> int:
        if (
            not isinstance(seed, Lfm2CausalSpanSeed)
            or not seed.verify_authentication()
            or seed.evidence_sha256 != self._evidence.authentication_sha256
            or seed.semantic_alignment_sha256
            != tuple(
                value.semantic_alignment_sha256 for value in self._evidence.batch_chains
            )
            or len(seed.batch_tangents) != len(self._evidence.snapshot.batches)
        ):
            raise Lfm2CausalSpanAuthenticationError(
                "Seed is stale or not bound to this source-span evidence."
            )
        local = self._local_layer(seed.source_layer_index)
        for value, batch in zip(
            seed.batch_tangents,
            self._evidence.snapshot.batches,
            strict=True,
        ):
            if tuple(value.shape) != tuple(
                batch.macrocells[local].operator.residual.shape
            ):
                raise Lfm2CausalSpanLinearizationError(
                    "Seed batch geometry differs from its source boundary."
                )
        return local

    def _validate_endpoint_cotangent(
        self,
        cotangent: Lfm2CausalSpanEndpointCotangent,
    ) -> None:
        if (
            not isinstance(cotangent, Lfm2CausalSpanEndpointCotangent)
            or not cotangent.verify_authentication()
            or cotangent.evidence_sha256 != self._evidence.authentication_sha256
            or cotangent.semantic_alignment_sha256
            != tuple(
                value.semantic_alignment_sha256 for value in self._evidence.batch_chains
            )
            or len(cotangent.batch_cotangents) != len(self._evidence.snapshot.batches)
        ):
            raise Lfm2CausalSpanAuthenticationError(
                "Endpoint cotangent is stale or bound to different evidence."
            )
        for value, batch in zip(
            cotangent.batch_cotangents,
            self._evidence.snapshot.batches,
            strict=True,
        ):
            if tuple(value.shape) != tuple(batch.macrocells[-1].mlp.output.shape):
                raise Lfm2CausalSpanLinearizationError(
                    "Endpoint cotangent geometry differs from the span endpoint."
                )

    def jvp(
        self,
        seed: Lfm2CausalSpanSeed,
    ) -> Lfm2CausalSpanEndpointTangent:
        """Propagate one boundary seed through every causally later source block."""

        torch = _torch()
        self._require_authenticated()
        local_start = self._validate_seed(seed)
        results = []
        with torch.no_grad():
            for batch, source_tangent in zip(
                self._evidence.snapshot.batches,
                seed.batch_tangents,
                strict=True,
            ):
                tangent = source_tangent
                first = batch.macrocells[local_start]
                if seed.boundary_role == "pre_operator":
                    tangent = first.residual_jvp(
                        tangent,
                        self._zero_output_basis,
                    )[1]
                elif seed.boundary_role == "post_operator":
                    zero_writer = torch.zeros(
                        (1, first.mlp.parameters.writer_input_width),
                        dtype=torch.float64,
                        device="cpu",
                    )
                    tangent = lfm2_mlp_segment_jvp(
                        first.mlp,
                        tangent,
                        zero_writer,
                        self._zero_output_basis,
                    ).output
                for local_index in range(local_start + 1, len(batch.macrocells)):
                    tangent = batch.macrocells[local_index].residual_jvp(
                        tangent,
                        self._zero_output_basis,
                    )[1]
                results.append(tangent)
        self._require_authenticated()
        return Lfm2CausalSpanEndpointTangent(
            evidence_sha256=self._evidence.authentication_sha256,
            seed_sha256=seed.authentication_sha256,
            batch_tangents=tuple(results),
        )

    def vjp(
        self,
        cotangent: Lfm2CausalSpanEndpointCotangent,
        *,
        source_layer_index: int,
        boundary_role: Lfm2CausalSpanBoundaryRole,
    ) -> Lfm2CausalSpanBoundaryCotangent:
        """Apply the exact transpose of the selected boundary-to-endpoint map."""

        torch = _torch()
        self._require_authenticated()
        self._validate_endpoint_cotangent(cotangent)
        local_start = self._local_layer(source_layer_index)
        role = _boundary_role(boundary_role)
        results = []
        with torch.no_grad():
            for batch, endpoint_value in zip(
                self._evidence.snapshot.batches,
                cotangent.batch_cotangents,
                strict=True,
            ):
                value = endpoint_value
                for local_index in range(
                    len(batch.macrocells) - 1,
                    local_start,
                    -1,
                ):
                    macrocell = batch.macrocells[local_index]
                    value = macrocell.residual_vjp(
                        (
                            torch.zeros_like(macrocell.operator.output),
                            value,
                        ),
                        self._zero_output_basis,
                    )
                first = batch.macrocells[local_start]
                if role == "pre_operator":
                    value = first.residual_vjp(
                        (
                            torch.zeros_like(first.operator.output),
                            value,
                        ),
                        self._zero_output_basis,
                    )
                elif role == "post_operator":
                    value = lfm2_mlp_segment_vjp(
                        first.mlp,
                        value,
                        self._zero_output_basis,
                    ).incoming_residual
                results.append(value)
        self._require_authenticated()
        return Lfm2CausalSpanBoundaryCotangent(
            evidence_sha256=self._evidence.authentication_sha256,
            endpoint_cotangent_sha256=cotangent.authentication_sha256,
            source_layer_index=source_layer_index,
            boundary_role=role,
            batch_cotangents=tuple(results),
        )


__all__ = [
    "LFM2_CAUSAL_SPAN_LINEARIZATION_SCHEMA",
    "Lfm2CausalSpanAuthenticationError",
    "Lfm2CausalSpanBoundaryCotangent",
    "Lfm2CausalSpanBoundaryRole",
    "Lfm2CausalSpanEndpointCotangent",
    "Lfm2CausalSpanEndpointOperator",
    "Lfm2CausalSpanEndpointTangent",
    "Lfm2CausalSpanLinearizationError",
    "Lfm2CausalSpanSeed",
    "Lfm2SourceSpanBatchChain",
    "Lfm2SourceSpanCacheReceipt",
    "Lfm2SourceSpanDirectLayerRecord",
    "Lfm2SourceSpanDirectParentReceipt",
    "Lfm2SourceSpanPlanReceipt",
    "Lfm2SourceSpanPrimalEvidence",
    "lfm2_source_span_provenance_root_sha256",
]
