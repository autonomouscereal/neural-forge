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


@dataclass(frozen=True, slots=True)
class Lfm2SourceSpanBatchChain:
    """Exact cached full-history residual chain for one source batch.

    ``layer_pre_residuals`` and ``layer_post_residuals`` remain in the cache's
    runtime dtype.  Exact adjacency is therefore authenticated before any
    conversion to the continuous FP64 linearization surrogate.
    """

    batch_index: int
    semantic_alignment_sha256: str
    cache_payload_sha256: str
    layer_pre_residuals: tuple[Any, ...] = field(repr=False, compare=False)
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
        object.__setattr__(
            self,
            "cache_payload_sha256",
            _digest(self.cache_payload_sha256, "cache payload"),
        )
        pre = tuple(
            _owned_runtime_history(value, f"layer_pre_residuals[{index}]")
            for index, value in enumerate(self.layer_pre_residuals)
        )
        post = tuple(
            _owned_runtime_history(value, f"layer_post_residuals[{index}]")
            for index, value in enumerate(self.layer_post_residuals)
        )
        if not pre or len(pre) != len(post):
            raise Lfm2CausalSpanLinearizationError(
                "A source batch chain needs equal nonempty pre/post histories."
            )
        reference_shape = tuple(pre[0].shape)
        reference_dtype = pre[0].dtype
        if any(
            tuple(value.shape) != reference_shape or value.dtype != reference_dtype
            for value in (*pre, *post)
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
        object.__setattr__(self, "layer_pre_residuals", pre)
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
        return len(self.layer_pre_residuals)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CAUSAL_SPAN_LINEARIZATION_SCHEMA,
            "kind": "source_span_batch_chain",
            "batch_index": self.batch_index,
            "semantic_alignment_sha256": self.semantic_alignment_sha256,
            "cache_payload_sha256": self.cache_payload_sha256,
            "layer_count": self.layer_count,
            "runtime_dtype": str(self.layer_pre_residuals[0].dtype),
            "runtime_shape": [
                int(value) for value in self.layer_pre_residuals[0].shape
            ],
            "layer_pre_residual_sha256": _tensor_hashes(self.layer_pre_residuals),
            "layer_post_residual_sha256": _tensor_hashes(self.layer_post_residuals),
            "exact_adjacent_runtime_chain": True,
        }

    def recompute_authentication_sha256(self) -> str:
        return _canonical_sha256(self.authentication_body())

    def verify_authentication(self) -> bool:
        torch = _torch()
        try:
            return (
                all(
                    bool(
                        torch.equal(
                            self.layer_post_residuals[index],
                            self.layer_pre_residuals[index + 1],
                        )
                    )
                    for index in range(self.layer_count - 1)
                )
                and self.authentication_sha256 == self.recompute_authentication_sha256()
            )
        except (RuntimeError, TypeError, ValueError):
            return False


@dataclass(frozen=True, slots=True)
class Lfm2SourceSpanPrimalEvidence:
    """A sealed source-depth plan joined to full-history analytic primals."""

    span: Lfm2ModelFoldSpan
    snapshot: Lfm2WholeStackPrimalSnapshot = field(repr=False, compare=False)
    batch_chains: tuple[Lfm2SourceSpanBatchChain, ...] = field(
        repr=False,
        compare=False,
    )
    source_plan_sha256: str
    cache_key: str
    cache_manifest_sha256: str
    cache_session_sha256: str
    direct_parent_session_sha256: str
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        torch = _torch()
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
        if snapshot.layer_count != len(self.span.source_layer_indices):
            raise Lfm2CausalSpanLinearizationError(
                "Source span and local primal snapshot disagree on layer count."
            )
        chains = tuple(self.batch_chains)
        if (
            len(chains) != len(snapshot.batches)
            or not all(
                isinstance(value, Lfm2SourceSpanBatchChain)
                and value.verify_authentication()
                for value in chains
            )
            or tuple(value.batch_index for value in chains) != tuple(range(len(chains)))
        ):
            raise Lfm2CausalSpanAuthenticationError(
                "batch_chains must match snapshot batches in canonical order."
            )
        for batch_index, (chain, batch) in enumerate(
            zip(chains, snapshot.batches, strict=True)
        ):
            if chain.layer_count != snapshot.layer_count:
                raise Lfm2CausalSpanLinearizationError(
                    f"Source chain {batch_index} has the wrong layer count."
                )
            for local_index, (cached_pre, macrocell) in enumerate(
                zip(
                    chain.layer_pre_residuals,
                    batch.macrocells,
                    strict=True,
                )
            ):
                analytic_pre = macrocell.operator.residual
                converted = cached_pre.to(
                    device="cpu",
                    dtype=torch.float64,
                )
                if tuple(converted.shape) != tuple(analytic_pre.shape) or not bool(
                    torch.equal(converted, analytic_pre)
                ):
                    raise Lfm2CausalSpanAuthenticationError(
                        "Analytic source primal is not centered on cached "
                        f"residual.pre for batch {batch_index}, local layer "
                        f"{local_index}."
                    )
        object.__setattr__(self, "batch_chains", chains)
        for name in (
            "source_plan_sha256",
            "cache_key",
            "cache_manifest_sha256",
            "cache_session_sha256",
            "direct_parent_session_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
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
            "source_plan_sha256": self.source_plan_sha256,
            "cache_key": self.cache_key,
            "cache_manifest_sha256": self.cache_manifest_sha256,
            "cache_session_sha256": self.cache_session_sha256,
            "direct_parent_session_sha256": self.direct_parent_session_sha256,
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
        torch = _torch()
        try:
            if (
                not self.snapshot.verify_authentication()
                or any(not value.verify_authentication() for value in self.batch_chains)
                or self.snapshot.layer_count != len(self.span.source_layer_indices)
                or len(self.snapshot.batches) != len(self.batch_chains)
            ):
                return False
            for chain, batch in zip(
                self.batch_chains,
                self.snapshot.batches,
                strict=True,
            ):
                if any(
                    not bool(
                        torch.equal(
                            cached.to(device="cpu", dtype=torch.float64),
                            macrocell.operator.residual,
                        )
                    )
                    for cached, macrocell in zip(
                        chain.layer_pre_residuals,
                        batch.macrocells,
                        strict=True,
                    )
                ):
                    return False
            return self.authentication_sha256 == self.recompute_authentication_sha256()
        except (RuntimeError, TypeError, ValueError):
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
    "Lfm2SourceSpanPrimalEvidence",
]
