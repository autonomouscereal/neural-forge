"""Executable dense blank-first LFM2 width folding.

The dense parameter frontier materializes a real smaller LFM2 architecture
whose learned tensors are compiler-zeroed.  This module fills that blank in
one transactional, zero-gradient model action:

* one complete-weight or circuit-balanced residual coordinate basis is shared
  by the embedding, every residual-facing circuit, and the output interface;
* GQA attention channels are folded with the existing RoPE-pair-constrained
  maps selected independently from each materialized layer's physical shape;
* every source SwiGLU neuron is assigned to the smaller coupled gate/up/down
  circuit through the existing feature-fold primitive;
* every dense ShortConv uses one coupled internal-channel fold distinct from
  the shared residual map; and
* an optional cached child-closed-loop pass refines attention consumers,
  SwiGLU, final normalization, and the tied output interface without executing
  the parent or training.

Depth composition is deliberately outside this transaction.  A target must
retain the source layer count and operator order; callers get an explicit
failure instead of a misleading partial fold.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from .dynamic_circuit_ir import JsonScalar, ModelCircuitGraph, canonical_sha256
from .dynamic_closed_loop_fold import (
    ChildClosedLoopCompiler,
    ClosedLoopCompilerConfig,
)
from .dynamic_fold_executor import (
    FoldDecision,
    FoldEvaluationContext,
    FoldEvidence,
    FoldExecutionContext,
    FoldTransaction,
    fingerprint_model_state,
)
from .dynamic_fold_diagnostics import (
    TRANSACTION_REPORT_BINDING_KEY,
    WHOLE_CHILD_DIAGNOSTIC_OPERATOR_CAPABILITY,
    AuthenticatedWholeChildComponentDispatch,
    WholeChildComponentDiagnosticError,
    authenticate_whole_child_component_dispatch,
)
from .dynamic_fold_planner import DynamicFoldPlan, FoldAction
from .dynamic_lfm2_adapter import (
    Lfm2DenseBudgetedSelectedTarget,
    encode_lfm2_layer_types,
    encode_lfm2_source_spans,
    validate_lfm2_dense_budgeted_child,
)
from .hf_attention_fold import build_gqa_attention_fold_maps
from .hf_feature_fold import build_feature_fold_map
from .hf_lfm2_circuit_trace_cache import (
    CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
    CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2CircuitTraceCache,
    Lfm2CircuitTraceCaptureConfig,
    Lfm2CircuitTraceSelectiveAuthenticatedSession,
    capture_lfm2_child_replay,
)
from .hf_lfm2_dynamic_closed_loop_adapter import (
    Lfm2DynamicClosedLoopAdapter,
    Lfm2DynamicClosedLoopAdapterConfig,
)
from .hf_lfm2_macro_cell_attractor_gate import (
    Lfm2MacroCellAttractorGate,
    Lfm2MacroCellAttractorGateConfig,
)
from .hf_lfm2_tokenizer_bridge_trace_adapter import (
    Lfm2StrictTokenizerBridgeTraceAdapter,
    Lfm2StrictTokenizerBridgeTraceError,
)
from .hf_lfm2_weight_complete_basis import build_lfm2_weight_complete_basis
from .hf_tokenizer_bridge import (
    StrictTokenizerBridgeArtifact,
    compile_strict_vocabulary_operators,
)
from .lfm2_native_exact4x_target import (
    lfm2_identity_tokenizer_evidence_sha256,
)


DENSE_BUDGET_MODEL_METHOD = "lfm2_dense_budget_residual_attention_model"
DENSE_BUDGET_FOLD_SCHEMA = "neural-forge-lfm2-dense-budget-model-fold-v1"
VOCABULARY_MARGIN_CANDIDATE_GATE_SCHEMA = (
    "neural-forge-lfm2-dense-budget-vocabulary-margin-candidate-gate-v1"
)
LFM2_COMPONENT_FOLD_STAGE_CONFIGURATION_KEY = "lfm2_component_fold_stage"
LFM2_COMPONENT_STRUCTURAL_STAGE = "structural_transport"
LFM2_COMPONENT_CLOSED_LOOP_STAGE = "closed_loop_refinement"
LFM2_COMPONENT_DISPATCH_EXECUTION_SCHEMA = (
    "neural-forge-lfm2-component-dispatch-execution-v1"
)


class Lfm2DenseBudgetFoldError(RuntimeError):
    """The dense target cannot be filled under the authenticated contract."""


class Lfm2DenseCacheIdentitySession(Protocol):
    """Cheap identity recheck exposed by an already-authenticated session."""

    @property
    def cache_key(self) -> str: ...

    def verify(self) -> Any: ...


class Lfm2DenseCacheVerificationLease:
    """Bind one fully authenticated cache to cheap subsequent identity checks.

    Constructing a lease does not authenticate cache payload bytes.  The caller
    must create it only after completing that full authentication, and must
    supply either a cheap identity callback or the authenticated session that
    performed it.  Every use still pins the exact cache handle and cache digest;
    callers that do not supply a lease retain the conservative full ``verify``
    path.
    """

    __slots__ = (
        "_trace_cache",
        "_cache_key",
        "_identity_check",
        "_authenticated_session",
    )

    def __init__(
        self,
        trace_cache: Lfm2CircuitTraceCache,
        expected_cache_key: str,
        *,
        identity_check: Callable[[], str] | None = None,
        authenticated_session: Lfm2DenseCacheIdentitySession | None = None,
    ) -> None:
        if not isinstance(trace_cache, Lfm2CircuitTraceCache):
            raise Lfm2DenseBudgetFoldError(
                "Dense cache verification lease requires Lfm2CircuitTraceCache."
            )
        if (identity_check is None) == (authenticated_session is None):
            raise Lfm2DenseBudgetFoldError(
                "Dense cache verification lease requires exactly one identity "
                "callback or authenticated session."
            )
        if identity_check is not None and not callable(identity_check):
            raise Lfm2DenseBudgetFoldError(
                "Dense cache verification identity_check must be callable."
            )
        if authenticated_session is not None and not callable(
            getattr(authenticated_session, "verify", None)
        ):
            raise Lfm2DenseBudgetFoldError(
                "Dense cache verification session must expose verify()."
            )
        self._trace_cache = trace_cache
        self._cache_key = _digest(
            expected_cache_key,
            "dense cache verification lease cache key",
        )
        self._identity_check = identity_check
        self._authenticated_session = authenticated_session
        self.verify(trace_cache)

    @property
    def trace_cache(self) -> Lfm2CircuitTraceCache:
        return self._trace_cache

    @property
    def cache_key(self) -> str:
        return self._cache_key

    def verify(
        self,
        trace_cache: Lfm2CircuitTraceCache,
    ) -> Lfm2CircuitTraceCache:
        if trace_cache is not self._trace_cache:
            raise Lfm2DenseBudgetFoldError(
                "Dense cache verification lease received a different cache handle."
            )
        if trace_cache.cache_key != self._cache_key:
            raise Lfm2DenseBudgetFoldError(
                "Dense cache verification lease cache key changed."
            )
        try:
            if self._authenticated_session is not None:
                verified = self._authenticated_session.verify()
                if verified is not None and verified is not self._authenticated_session:
                    raise Lfm2DenseBudgetFoldError(
                        "Dense cache authenticated session returned a different "
                        "session handle."
                    )
                observed = getattr(
                    self._authenticated_session,
                    "cache_key",
                    None,
                )
            else:
                assert self._identity_check is not None
                observed = self._identity_check()
        except Lfm2DenseBudgetFoldError:
            raise
        except Exception as exc:
            raise Lfm2DenseBudgetFoldError(
                "Dense cache verification lease identity recheck failed."
            ) from exc
        if _digest(observed, "dense cache verification identity result") != (
            self._cache_key
        ):
            raise Lfm2DenseBudgetFoldError(
                "Dense cache verification lease identity differs from its "
                "bound cache key."
            )
        return trace_cache


def _verify_dense_trace_cache(
    trace_cache: Lfm2CircuitTraceCache,
    verification_lease: Lfm2DenseCacheVerificationLease | None,
) -> Lfm2CircuitTraceCache:
    if verification_lease is None:
        return trace_cache.verify()
    if not isinstance(verification_lease, Lfm2DenseCacheVerificationLease):
        raise Lfm2DenseBudgetFoldError(
            "verification_lease must be Lfm2DenseCacheVerificationLease or None."
        )
    return verification_lease.verify(trace_cache)


class Lfm2SealedDenseBudgetEvidence:
    """Executor-safe non-dataclass seal around the typed trace cache."""

    __slots__ = ("_trace_cache", "_cache_key", "_verification_lease")

    def __init__(
        self,
        trace_cache: Lfm2CircuitTraceCache,
        *,
        verification_lease: Lfm2DenseCacheVerificationLease | None = None,
    ) -> None:
        if not isinstance(trace_cache, Lfm2CircuitTraceCache):
            raise Lfm2DenseBudgetFoldError("trace_cache must be Lfm2CircuitTraceCache.")
        self._verification_lease = verification_lease
        self._trace_cache = _verify_dense_trace_cache(
            trace_cache,
            verification_lease,
        )
        self._cache_key = _digest(
            self._trace_cache.cache_key,
            "trace_cache.cache_key",
        )

    @property
    def trace_cache(self) -> Lfm2CircuitTraceCache:
        return self._trace_cache

    @property
    def cache_key(self) -> str:
        return self._cache_key

    def verify(self) -> "Lfm2SealedDenseBudgetEvidence":
        _verify_dense_trace_cache(
            self._trace_cache,
            self._verification_lease,
        )
        if self._trace_cache.cache_key != self._cache_key:
            raise Lfm2DenseBudgetFoldError(
                "Dense trace-cache seal changed after binding."
            )
        return self


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise Lfm2DenseBudgetFoldError(
            "PyTorch is required for dense LFM2 folding."
        ) from exc
    return torch


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2DenseBudgetFoldError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2DenseBudgetFoldError(f"{name} must be a positive integer.")
    return value


def _unit_fraction(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2DenseBudgetFoldError(
            f"{name} must be a finite fraction within [0, 1]."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2DenseBudgetFoldError(
            f"{name} must be a finite fraction within [0, 1]."
        ) from exc
    if not 0.0 <= result <= 1.0:
        raise Lfm2DenseBudgetFoldError(
            f"{name} must be a finite fraction within [0, 1]."
        )
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2DenseBudgetFoldError(f"{name} must be nonempty.")
    return value.strip()


def _layers(model: Any) -> tuple[Any, ...]:
    body = getattr(model, "model", None)
    layers = getattr(body, "layers", None)
    if layers is None:
        raise Lfm2DenseBudgetFoldError("LFM2 model does not expose model.layers.")
    result = tuple(layers)
    if not result:
        raise Lfm2DenseBudgetFoldError("LFM2 decoder has no layers.")
    return result


def _fold_layer_kind(layer: Any, *, label: str) -> str:
    """Read the executed operator kind without assuming uniform dimensions.

    A physically materialized heterogeneous child retains native decoder and
    operator semantics while owning descriptor-local attention and ShortConv
    dimensions.  The fold installer therefore authenticates its kind at the
    live module boundary rather than assuming one uniform configuration.
    """

    is_attention = getattr(layer, "is_attention_layer", None)
    if type(is_attention) is not bool:
        raise Lfm2DenseBudgetFoldError(f"{label}.is_attention_layer must be boolean.")
    operator_name = "self_attn" if is_attention else "conv"
    if getattr(layer, operator_name, None) is None:
        raise Lfm2DenseBudgetFoldError(
            f"{label} lacks its declared {operator_name} operator."
        )
    return "full_attention" if is_attention else "conv"


def _linear_weight_shape(operator: Any, name: str, *, label: str) -> tuple[int, int]:
    torch = _torch()
    projection = getattr(operator, name, None)
    weight = getattr(projection, "weight", None)
    if (
        not isinstance(weight, torch.Tensor)
        or weight.ndim != 2
        or any(int(size) <= 0 for size in weight.shape)
    ):
        raise Lfm2DenseBudgetFoldError(
            f"{label}.{name}.weight must be a nonempty matrix."
        )
    return int(weight.shape[0]), int(weight.shape[1])


def _attention_physical_geometry(
    attention: Any,
    *,
    label: str,
) -> tuple[int, int, int, int]:
    """Infer Q/KV/head geometry from one executed attention module."""

    head_dim = _positive_int(
        getattr(attention, "head_dim", None),
        f"{label}.head_dim",
    )
    if head_dim % 2:
        raise Lfm2DenseBudgetFoldError(
            f"{label}.head_dim must preserve complete RoPE pairs."
        )
    query_width, query_input = _linear_weight_shape(
        attention,
        "q_proj",
        label=label,
    )
    key_width, key_input = _linear_weight_shape(
        attention,
        "k_proj",
        label=label,
    )
    value_width, value_input = _linear_weight_shape(
        attention,
        "v_proj",
        label=label,
    )
    output_width, context_width = _linear_weight_shape(
        attention,
        "out_proj",
        label=label,
    )
    if (
        query_width % head_dim
        or key_width % head_dim
        or value_width != key_width
        or query_input != key_input
        or query_input != value_input
        or output_width != query_input
        or context_width != query_width
    ):
        raise Lfm2DenseBudgetFoldError(
            f"{label} projection matrices disagree with one coupled GQA geometry."
        )
    query_heads = query_width // head_dim
    key_value_heads = key_width // head_dim
    if query_heads % key_value_heads:
        raise Lfm2DenseBudgetFoldError(
            f"{label} query heads must divide into complete KV groups."
        )
    declared_query_heads = getattr(attention, "query_heads", None)
    if declared_query_heads is not None and declared_query_heads != query_heads:
        raise Lfm2DenseBudgetFoldError(
            f"{label}.query_heads disagrees with q_proj.weight."
        )
    declared_key_value_heads = getattr(attention, "key_value_heads", None)
    if (
        declared_key_value_heads is not None
        and declared_key_value_heads != key_value_heads
    ):
        raise Lfm2DenseBudgetFoldError(
            f"{label}.key_value_heads disagrees with k_proj.weight."
        )
    return query_heads, key_value_heads, head_dim, query_input


def _dense_short_conv_physical_geometry(
    short_conv: Any,
    *,
    label: str,
) -> tuple[int, int, int]:
    """Authenticate one dense ShortConv writer and return ``(C, H, taps)``."""

    torch = _torch()
    writer = getattr(short_conv, "out_proj", None)
    if isinstance(writer, torch.nn.Sequential):
        raise Lfm2DenseBudgetFoldError(
            "Factorized nn.Sequential ShortConv writers are supported by the "
            "physical blank ABI but are not yet installable by this dense "
            f"structural semantic lane ({label}.out_proj)."
        )
    input_rows, residual_width = _linear_weight_shape(
        short_conv,
        "in_proj",
        label=label,
    )
    output_width, output_channels = _linear_weight_shape(
        short_conv,
        "out_proj",
        label=label,
    )
    if input_rows % 3:
        raise Lfm2DenseBudgetFoldError(
            f"{label}.in_proj must contain three complete B/C/x blocks."
        )
    internal_channels = input_rows // 3
    convolution = getattr(short_conv, "conv", None)
    kernel = getattr(convolution, "weight", None)
    if (
        not isinstance(kernel, torch.Tensor)
        or kernel.ndim != 3
        or tuple(int(size) for size in kernel.shape[:2]) != (internal_channels, 1)
        or int(kernel.shape[2]) <= 0
        or output_width != residual_width
        or output_channels != internal_channels
    ):
        raise Lfm2DenseBudgetFoldError(
            f"{label} does not expose one dense C-channel ShortConv circuit."
        )
    return internal_channels, residual_width, int(kernel.shape[2])


def _topology_fingerprint(model: Any) -> str:
    parameters = tuple(model.named_parameters(recurse=True, remove_duplicate=False))
    aliases: dict[int, list[str]] = {}
    for name, parameter in parameters:
        aliases.setdefault(id(parameter), []).append(name)
    alias_group = {
        parameter_id: index
        for index, parameter_id in enumerate(
            sorted(
                aliases,
                key=lambda value: tuple(sorted(aliases[value])),
            )
        )
    }
    return canonical_sha256(
        {
            "schema": "neural-forge-lfm2-live-parameter-topology-v1",
            "parameters": [
                {
                    "name": name,
                    "shape": [int(size) for size in parameter.shape],
                    "dtype": str(parameter.dtype),
                    "layout": str(parameter.layout),
                    "requires_grad": bool(parameter.requires_grad),
                    "alias_group": alias_group[id(parameter)],
                }
                for name, parameter in parameters
            ],
        }
    )


def _state_snapshot(model: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (
            name,
            value.detach().to(device="cpu").contiguous().clone(),
        )
        for name, value in sorted(model.state_dict().items())
    )


def _restore_state(model: Any, snapshot: Sequence[tuple[str, Any]]) -> None:
    torch = _torch()
    current = model.state_dict()
    saved = dict(snapshot)
    if set(current) != set(saved):
        raise Lfm2DenseBudgetFoldError(
            "Dense child state keys changed before rollback."
        )
    with torch.no_grad():
        for name, value in current.items():
            source = saved[name]
            if tuple(value.shape) != tuple(source.shape) or value.dtype != source.dtype:
                raise Lfm2DenseBudgetFoldError(
                    "Dense child state shape/dtype changed before rollback."
                )
            value.copy_(source.to(device=value.device, dtype=value.dtype))


def _parameter_snapshot(model: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (
            name,
            parameter.detach().to(device="cpu").contiguous().clone(),
        )
        for name, parameter in sorted(
            model.named_parameters(recurse=True, remove_duplicate=False)
        )
    )


def _buffer_snapshot(model: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (
            name,
            buffer.detach().to(device="cpu").contiguous().clone(),
        )
        for name, buffer in sorted(
            model.named_buffers(recurse=True, remove_duplicate=False)
        )
    )


def _assert_buffer_snapshot_equal(
    before: Sequence[tuple[str, Any]],
    after: Sequence[tuple[str, Any]],
) -> None:
    torch = _torch()
    left = dict(before)
    right = dict(after)
    if set(left) != set(right) or any(
        not torch.equal(left[name], right[name]) for name in left
    ):
        raise Lfm2DenseBudgetFoldError(
            "Component diagnostic refinement changed unowned model buffers."
        )


def _tensor_sha256(value: Any) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(int(size) for size in tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.reshape(-1).view(_torch().uint8).numpy().tobytes())
    return digest.hexdigest()


def _component_stage_assignments(
    dispatch: AuthenticatedWholeChildComponentDispatch,
) -> tuple[tuple[str, str], ...]:
    strategy_stages: dict[str, str] = {}
    for fingerprint, strategy in dispatch.strategies:
        configuration = dict(strategy.configuration)
        if set(configuration) != {LFM2_COMPONENT_FOLD_STAGE_CONFIGURATION_KEY}:
            raise Lfm2DenseBudgetFoldError(
                "LFM2 production diagnostics accept only the explicit component "
                "fold-stage strategy configuration."
            )
        stage = configuration[LFM2_COMPONENT_FOLD_STAGE_CONFIGURATION_KEY]
        if stage not in {
            LFM2_COMPONENT_STRUCTURAL_STAGE,
            LFM2_COMPONENT_CLOSED_LOOP_STAGE,
        }:
            raise Lfm2DenseBudgetFoldError(
                "LFM2 component fold stage must select structural transport or "
                "cached closed-loop refinement."
            )
        strategy_stages[fingerprint] = str(stage)
    if len(set(strategy_stages.values())) != 2:
        raise Lfm2DenseBudgetFoldError(
            "LFM2 diagnostic strategies must execute distinct production stages."
        )
    return tuple(
        (path, strategy_stages[fingerprint])
        for path, fingerprint in dispatch.parameter_strategy_fingerprints
    )


def _compose_component_stage_parameters(
    model: Any,
    *,
    path_stages: Mapping[str, str],
    structural_snapshot: Sequence[tuple[str, Any]],
    closed_loop_snapshot: Sequence[tuple[str, Any]],
) -> None:
    torch = _torch()
    structural = dict(structural_snapshot)
    closed_loop = dict(closed_loop_snapshot)
    parameters = tuple(model.named_parameters(recurse=True, remove_duplicate=False))
    names = {name for name, _parameter in parameters}
    if (
        names != set(path_stages)
        or names != set(structural)
        or names != set(closed_loop)
    ):
        raise Lfm2DenseBudgetFoldError(
            "Component dispatch snapshots differ from live whole-child ownership."
        )
    aliases: dict[int, list[tuple[str, Any]]] = {}
    for name, parameter in parameters:
        aliases.setdefault(id(parameter), []).append((name, parameter))
    with torch.no_grad():
        for values in aliases.values():
            stages = {path_stages[name] for name, _parameter in values}
            if len(stages) != 1:
                raise Lfm2DenseBudgetFoldError(
                    "Aliased target parameter paths received different fold stages."
                )
            stage = stages.pop()
            source_snapshot = (
                structural if stage == LFM2_COMPONENT_STRUCTURAL_STAGE else closed_loop
            )
            first_name, parameter = values[0]
            source = source_snapshot[first_name]
            if tuple(source.shape) != tuple(parameter.shape):
                raise Lfm2DenseBudgetFoldError(
                    "Component dispatch changed target parameter geometry."
                )
            parameter.copy_(source.to(device=parameter.device, dtype=parameter.dtype))
            parameter.grad = None
            for alias_name, _alias in values[1:]:
                if not torch.equal(source_snapshot[alias_name], source):
                    raise Lfm2DenseBudgetFoldError(
                        "Aliased component snapshots contain different values."
                    )


def _component_operator_names(semantics: Sequence[str]) -> tuple[str, ...]:
    operators = set()
    for semantic in semantics:
        if semantic.startswith("attention_"):
            operators.add("gqa_rope_residual_attention_fold")
        elif semantic.startswith("mlp_"):
            operators.add("coupled_swiglu_feature_fold")
        elif semantic == "state_mixer":
            operators.add("coupled_short_conv_channel_fold")
        elif semantic in {"token_embedding", "output_head"}:
            operators.add("token_vocabulary_interface_fold")
        elif semantic == "normalization":
            operators.add("residual_diagonal_norm_fold")
        else:
            operators.add("topology_preservation")
    return tuple(sorted(operators))


def _component_dispatch_execution_report(
    dispatch: AuthenticatedWholeChildComponentDispatch,
    model: Any,
    *,
    path_stages: Mapping[str, str],
    structural_snapshot: Sequence[tuple[str, Any]],
    closed_loop_snapshot: Sequence[tuple[str, Any]] | None,
    exact_target_unique_parameters: int,
) -> dict[str, Any]:
    torch = _torch()
    structural = dict(structural_snapshot)
    closed_loop = (
        structural if closed_loop_snapshot is None else dict(closed_loop_snapshot)
    )
    final = dict(_parameter_snapshot(model))
    if set(final) != set(path_stages) or set(final) != set(structural):
        raise Lfm2DenseBudgetFoldError(
            "Component execution report lost whole-child path ownership."
        )
    changed_between_stages = {
        path for path in final if not torch.equal(structural[path], closed_loop[path])
    }
    for path, stage in path_stages.items():
        expected = (
            structural[path]
            if stage == LFM2_COMPONENT_STRUCTURAL_STAGE
            else closed_loop[path]
        )
        if not torch.equal(final[path], expected):
            raise Lfm2DenseBudgetFoldError(
                "Final diagnostic parameter differs from its assigned fold stage."
            )
    focus_paths = set(dispatch.focused_parameter_paths)
    focus_changed = focus_paths.intersection(changed_between_stages)
    if dispatch.binding.focus_scope_id is not None and not focus_changed:
        raise Lfm2DenseBudgetFoldError(
            "Focused LFM2 production strategy installed no distinct learned-state "
            "delta; this component diagnostic is not executable."
        )

    strategy_by_path = dispatch.parameter_strategy_by_path
    component_actions = []
    parameterized_component_ids = set()
    for component in dispatch.components:
        if not component.parameter_paths:
            continue
        parameterized_component_ids.add(component.component_id)
        fingerprints = {strategy_by_path[path] for path in component.parameter_paths}
        stages = {path_stages[path] for path in component.parameter_paths}
        if len(fingerprints) != 1 or len(stages) != 1:
            raise Lfm2DenseBudgetFoldError(
                "One discovered component was split across production strategies."
            )
        fingerprint = fingerprints.pop()
        stage = stages.pop()
        parameter_state = {
            path: _tensor_sha256(final[path]) for path in component.parameter_paths
        }
        component_actions.append(
            {
                "component_id": component.component_id,
                "semantics": list(component.semantics),
                "strategy_fingerprint": fingerprint,
                "fold_stage": stage,
                "production_fold_operators": list(
                    _component_operator_names(component.semantics)
                ),
                "target_parameter_paths": list(component.parameter_paths),
                "target_storage_groups": list(component.storage_groups),
                "final_parameter_state_sha256": canonical_sha256(parameter_state),
                "stage_delta_parameter_path_count": len(
                    set(component.parameter_paths).intersection(changed_between_stages)
                ),
                "executor_action_performed": True,
            }
        )
    if parameterized_component_ids != {
        value.component_id for value in dispatch.components if value.parameter_paths
    }:
        raise Lfm2DenseBudgetFoldError(
            "Component dispatch did not execute every learned child component."
        )
    unique_parameters = {
        id(parameter)
        for _name, parameter in model.named_parameters(
            recurse=True, remove_duplicate=False
        )
    }
    if sum(parameter.numel() for parameter in model.parameters()) != (
        exact_target_unique_parameters
    ) or len(unique_parameters) != len(tuple(model.parameters())):
        raise Lfm2DenseBudgetFoldError(
            "Component execution changed exact dense parameter accounting."
        )
    path_receipt = {
        path: {
            "strategy_fingerprint": strategy_by_path[path],
            "fold_stage": path_stages[path],
            "final_state_sha256": _tensor_sha256(final[path]),
        }
        for path in sorted(final)
    }
    structural_nodes = tuple(
        sorted(
            node
            for component in dispatch.components
            if not component.parameter_paths
            for node in component.node_ids
        )
    )
    body = {
        "schema": LFM2_COMPONENT_DISPATCH_EXECUTION_SCHEMA,
        "binding": dispatch.binding.as_dict(),
        "dispatch_sha256": dispatch.dispatch_sha256,
        "exact_target_unique_parameters": exact_target_unique_parameters,
        "complete_parameter_path_coverage": True,
        "parameter_path_count": len(path_receipt),
        "parameter_path_execution_receipt_sha256": canonical_sha256(path_receipt),
        "actual_executor_action_count": len(component_actions),
        "actual_executor_actions": component_actions,
        "structural_nodes_preserved": list(structural_nodes),
        "structural_topology_preserved": True,
        "closed_loop_stage_materialized": closed_loop_snapshot is not None,
        "stage_delta_parameter_path_count": len(changed_between_stages),
        "stage_delta_parameter_paths_sha256": canonical_sha256(
            sorted(changed_between_stages)
        ),
        "focus_parameter_path_count": len(focus_paths),
        "focus_parameter_paths_sha256": canonical_sha256(sorted(focus_paths)),
        "focus_changed_parameter_path_count": len(focus_changed),
        "focus_changed_parameter_paths_sha256": canonical_sha256(sorted(focus_changed)),
        "all_parameters_match_assigned_production_stage": True,
        "parent_forward_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    return {**body, "execution_sha256": canonical_sha256(body)}


def _dense_basis(mapping: Any, *, dtype: Any) -> Any:
    source_width = _positive_int(
        getattr(mapping, "source_width", None),
        "coordinate_map.source_width",
    )
    target_width = _positive_int(
        getattr(mapping, "target_width", None),
        "coordinate_map.target_width",
    )
    materialize = getattr(mapping, "materialize_dense", None)
    if not callable(materialize):
        raise Lfm2DenseBudgetFoldError(
            "Residual coordinate map cannot materialize its dense basis."
        )
    result = materialize(
        max_elements=source_width * target_width,
        dtype=dtype,
        device="cpu",
    )
    if tuple(result.shape) != (source_width, target_width):
        raise Lfm2DenseBudgetFoldError("Residual coordinate basis has the wrong shape.")
    return result


def _transport_diagonal(mapping: Any, source: Any) -> Any:
    torch = _torch()
    basis = _dense_basis(mapping, dtype=torch.float64)
    value = source.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if int(value.numel()) != int(mapping.source_width):
        raise Lfm2DenseBudgetFoldError(
            "Diagonal source width differs from the residual coordinate map."
        )
    return (basis.square().T @ value).contiguous()


def _transport_feature_diagonal(feature_map: Any, source: Any) -> Any:
    torch = _torch()
    basis = feature_map.materialize_dense(
        max_elements=feature_map.source_width * feature_map.target_width,
        dtype=torch.float64,
        device="cpu",
    )
    value = source.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if int(value.numel()) != feature_map.source_width:
        raise Lfm2DenseBudgetFoldError(
            "Feature-diagonal source width differs from its fold map."
        )
    return (basis.square().T @ value).contiguous()


def _transport_depthwise_kernel(mapping: Any, source: Any) -> Any:
    torch = _torch()
    value = source.detach().to(device="cpu", dtype=torch.float64)
    if value.ndim != 2 or int(value.shape[0]) != int(mapping.source_width):
        raise Lfm2DenseBudgetFoldError(
            "Depthwise FIR kernel must be [source_channels, taps]."
        )
    basis = _dense_basis(mapping, dtype=torch.float64)
    return (basis.square().T @ value).contiguous()


def _copy_parameter(
    target: Any,
    value: Any,
    *,
    name: str,
    written: set[int],
) -> None:
    torch = _torch()
    if tuple(target.shape) != tuple(value.shape):
        raise Lfm2DenseBudgetFoldError(
            f"Folded shape mismatch for {name}: "
            f"{tuple(value.shape)} != {tuple(target.shape)}."
        )
    if not bool(torch.isfinite(value.detach()).all()):
        raise Lfm2DenseBudgetFoldError(f"Folded value for {name} is non-finite.")
    target.copy_(value.to(device=target.device, dtype=target.dtype))
    target.grad = None
    written.add(id(target))


def _copy_optional_bias(
    source: Any,
    target: Any,
    value: Any,
    *,
    name: str,
    written: set[int],
) -> None:
    if (source is None) != (target is None):
        raise Lfm2DenseBudgetFoldError(
            f"Source/target bias presence differs for {name}."
        )
    if source is not None:
        _copy_parameter(target, value, name=name, written=written)


@dataclass(frozen=True, slots=True)
class Lfm2DenseBudgetFoldConfig:
    """Residual-basis selection plus optional cached closed-loop refinement."""

    residual_basis_mode: str = "weight_complete"
    run_closed_loop: bool = True
    circuit_validation_fraction: float = 0.25
    circuit_split_seed: str = "lfm2-dense-budget-circuit-basis-v1"
    circuit_max_rows_per_block: int = 512
    minimum_heldout_retained_energy_fraction: float = 0.0
    vocabulary_margin_observability: bool = False
    vocabulary_margin_top_k: int = 8
    vocabulary_margin_floor_fraction: float = 0.05
    streaming_vocabulary_margin_v2: bool = False
    streaming_vocabulary_margin_pair_chunk_size: int = 256
    vocabulary_margin_candidate_gate_enabled: bool = False
    vocabulary_margin_candidate_replay_mode: str = CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE
    vocabulary_margin_candidate_minimum_sign_agreement_fraction: float = 0.9
    vocabulary_margin_candidate_minimum_top_competitor_nonnegative_fraction: float = 0.9
    vocabulary_margin_candidate_minimum_all_top_k_nonnegative_fraction: float = 0.9
    vocabulary_margin_candidate_minimum_exact_top1_agreement_fraction: float = 0.9
    closed_loop_adapter_config: Lfm2DynamicClosedLoopAdapterConfig = field(
        default_factory=Lfm2DynamicClosedLoopAdapterConfig
    )
    closed_loop_compiler_config: ClosedLoopCompilerConfig = field(
        default_factory=ClosedLoopCompilerConfig
    )
    macro_cell_attractor_gate_config: Lfm2MacroCellAttractorGateConfig | None = None

    def __post_init__(self) -> None:
        mode = _text(self.residual_basis_mode, "residual_basis_mode")
        if mode not in {"weight_complete", "circuit_balanced"}:
            raise Lfm2DenseBudgetFoldError(
                "residual_basis_mode must be weight_complete or circuit_balanced."
            )
        object.__setattr__(self, "residual_basis_mode", mode)
        if not isinstance(self.run_closed_loop, bool):
            raise Lfm2DenseBudgetFoldError("run_closed_loop must be boolean.")
        fraction = float(self.circuit_validation_fraction)
        if not 0.0 < fraction < 1.0:
            raise Lfm2DenseBudgetFoldError(
                "circuit_validation_fraction must be within (0, 1)."
            )
        object.__setattr__(self, "circuit_validation_fraction", fraction)
        object.__setattr__(
            self,
            "circuit_split_seed",
            _text(self.circuit_split_seed, "circuit_split_seed"),
        )
        object.__setattr__(
            self,
            "circuit_max_rows_per_block",
            _positive_int(
                self.circuit_max_rows_per_block,
                "circuit_max_rows_per_block",
            ),
        )
        retained = float(self.minimum_heldout_retained_energy_fraction)
        if not 0.0 <= retained <= 1.0:
            raise Lfm2DenseBudgetFoldError(
                "minimum_heldout_retained_energy_fraction must be within [0, 1]."
            )
        object.__setattr__(
            self,
            "minimum_heldout_retained_energy_fraction",
            retained,
        )
        if not isinstance(self.vocabulary_margin_observability, bool):
            raise Lfm2DenseBudgetFoldError(
                "vocabulary_margin_observability must be boolean."
            )
        if self.vocabulary_margin_observability and mode != "circuit_balanced":
            raise Lfm2DenseBudgetFoldError(
                "vocabulary_margin_observability requires "
                "residual_basis_mode='circuit_balanced'."
            )
        object.__setattr__(
            self,
            "vocabulary_margin_top_k",
            _positive_int(
                self.vocabulary_margin_top_k,
                "vocabulary_margin_top_k",
            ),
        )
        margin_floor = float(self.vocabulary_margin_floor_fraction)
        if not 0.0 < margin_floor <= 1.0:
            raise Lfm2DenseBudgetFoldError(
                "vocabulary_margin_floor_fraction must be within (0, 1]."
            )
        object.__setattr__(
            self,
            "vocabulary_margin_floor_fraction",
            margin_floor,
        )
        if not isinstance(self.streaming_vocabulary_margin_v2, bool):
            raise Lfm2DenseBudgetFoldError(
                "streaming_vocabulary_margin_v2 must be boolean."
            )
        if (
            self.streaming_vocabulary_margin_v2
            and not self.vocabulary_margin_observability
        ):
            raise Lfm2DenseBudgetFoldError(
                "streaming_vocabulary_margin_v2 requires "
                "vocabulary_margin_observability=True."
            )
        object.__setattr__(
            self,
            "streaming_vocabulary_margin_pair_chunk_size",
            _positive_int(
                self.streaming_vocabulary_margin_pair_chunk_size,
                "streaming_vocabulary_margin_pair_chunk_size",
            ),
        )
        if not isinstance(
            self.vocabulary_margin_candidate_gate_enabled,
            bool,
        ):
            raise Lfm2DenseBudgetFoldError(
                "vocabulary_margin_candidate_gate_enabled must be boolean."
            )
        if (
            self.vocabulary_margin_candidate_gate_enabled
            and not self.vocabulary_margin_observability
        ):
            raise Lfm2DenseBudgetFoldError(
                "vocabulary_margin_candidate_gate_enabled requires "
                "vocabulary_margin_observability=True."
            )
        if not isinstance(
            self.vocabulary_margin_candidate_replay_mode,
            str,
        ) or self.vocabulary_margin_candidate_replay_mode not in {
            CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
            CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
        }:
            raise Lfm2DenseBudgetFoldError(
                "vocabulary_margin_candidate_replay_mode must be "
                "'prefix_by_prefix_canonical' or 'single_pass_causal'."
            )
        for attribute in (
            "vocabulary_margin_candidate_minimum_sign_agreement_fraction",
            ("vocabulary_margin_candidate_minimum_top_competitor_nonnegative_fraction"),
            ("vocabulary_margin_candidate_minimum_all_top_k_nonnegative_fraction"),
            ("vocabulary_margin_candidate_minimum_exact_top1_agreement_fraction"),
        ):
            object.__setattr__(
                self,
                attribute,
                _unit_fraction(getattr(self, attribute), attribute),
            )
        if not isinstance(
            self.closed_loop_adapter_config,
            Lfm2DynamicClosedLoopAdapterConfig,
        ):
            raise Lfm2DenseBudgetFoldError(
                "closed_loop_adapter_config has the wrong type."
            )
        if not isinstance(
            self.closed_loop_compiler_config,
            ClosedLoopCompilerConfig,
        ):
            raise Lfm2DenseBudgetFoldError(
                "closed_loop_compiler_config has the wrong type."
            )
        if self.macro_cell_attractor_gate_config is not None and not isinstance(
            self.macro_cell_attractor_gate_config,
            Lfm2MacroCellAttractorGateConfig,
        ):
            raise Lfm2DenseBudgetFoldError(
                "macro_cell_attractor_gate_config has the wrong type."
            )
        if (
            self.macro_cell_attractor_gate_config is not None
            and not self.run_closed_loop
        ):
            raise Lfm2DenseBudgetFoldError(
                "macro-cell attractor gating requires run_closed_loop=True."
            )


@dataclass(frozen=True, slots=True)
class Lfm2DenseBudgetFoldPlan:
    """Immutable audit report for one installed dense child candidate."""

    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.report, Mapping):
            raise Lfm2DenseBudgetFoldError("report must be a mapping.")
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    @property
    def zero_gradient(self) -> bool:
        return bool(self.report["zero_gradient"])

    @property
    def seed_erasure_by_construction(self) -> bool:
        return bool(self.report["seed_erasure_by_construction"])

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def _build_residual_basis(
    parent_model: Any,
    parent_graph: ModelCircuitGraph,
    trace_cache: Lfm2CircuitTraceCache,
    target_width: int,
    config: Lfm2DenseBudgetFoldConfig,
    selective_session: Lfm2CircuitTraceSelectiveAuthenticatedSession | None = None,
) -> Any:
    if config.residual_basis_mode == "weight_complete":
        return build_lfm2_weight_complete_basis(
            parent_model,
            target_width,
        )
    from .hf_lfm2_balanced_circuit_basis import (
        build_lfm2_circuit_balanced_basis,
    )

    return build_lfm2_circuit_balanced_basis(
        parent_model,
        trace_cache,
        target_width,
        selective_session=selective_session,
        validation_fraction=config.circuit_validation_fraction,
        split_seed=config.circuit_split_seed,
        max_rows_per_block=config.circuit_max_rows_per_block,
        minimum_heldout_retained_energy_fraction=(
            config.minimum_heldout_retained_energy_fraction
        ),
        circuit_graph=parent_graph,
        vocabulary_margin_observability=(config.vocabulary_margin_observability),
        vocabulary_margin_top_k=config.vocabulary_margin_top_k,
        vocabulary_margin_floor_fraction=(config.vocabulary_margin_floor_fraction),
        streaming_vocabulary_margin_v2=(
            config.streaming_vocabulary_margin_v2
        ),
        streaming_vocabulary_margin_pair_chunk_size=(
            config.streaming_vocabulary_margin_pair_chunk_size
        ),
    )


def _candidate_replay_capture_config(
    trace_cache: Lfm2CircuitTraceCache,
) -> Lfm2CircuitTraceCaptureConfig:
    manifest_capture = trace_cache.manifest.get("capture")
    if not isinstance(manifest_capture, Mapping):
        raise Lfm2DenseBudgetFoldError(
            "The sealed parent cache omits its capture configuration."
        )
    try:
        return Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=int(manifest_capture.get("sample_last_tokens", 8)),
            max_decode_steps=int(manifest_capture.get("max_decode_steps", 32)),
            max_capture_bytes=int(
                manifest_capture.get(
                    "max_capture_bytes",
                    512 * 1024 * 1024,
                )
            ),
            logit_rtol=float(manifest_capture.get("logit_rtol", 1e-5)),
            logit_atol=float(manifest_capture.get("logit_atol", 1e-6)),
            require_top1_agreement=bool(
                manifest_capture.get("require_top1_agreement", True)
            ),
            decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2DenseBudgetFoldError(
            "The sealed parent cache has an invalid replay configuration."
        ) from exc


def _disabled_vocabulary_margin_candidate_gate_report() -> dict[str, Any]:
    body = {
        "schema": VOCABULARY_MARGIN_CANDIDATE_GATE_SCHEMA,
        "enabled": False,
        "gate_passed": None,
        "candidate_replay_capture_calls": 0,
        "actual_child_forward_calls": 0,
        "parent_forward_calls": 0,
        "trace_regeneration_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    return {**body, "audit_sha256": canonical_sha256(body)}


def _evaluate_vocabulary_margin_candidate_gate(
    child_model: Any,
    trace_cache: Lfm2CircuitTraceCache,
    basis: Any,
    config: Lfm2DenseBudgetFoldConfig,
) -> dict[str, Any]:
    from .hf_lfm2_balanced_circuit_basis import (
        Lfm2CircuitBalancedBasisResult,
    )

    if (
        not isinstance(basis, Lfm2CircuitBalancedBasisResult)
        or basis.vocabulary_margin_gate is None
    ):
        raise Lfm2DenseBudgetFoldError(
            "The enabled vocabulary-margin candidate gate requires a "
            "circuit-balanced basis with sealed held-out margin evidence."
        )
    heldout_record_ids = frozenset(basis.vocabulary_margin_gate.heldout_record_ids)
    if not heldout_record_ids:
        raise Lfm2DenseBudgetFoldError(
            "The vocabulary-margin candidate gate has no held-out records."
        )
    heldout_batches = tuple(
        batch
        for batch in trace_cache.iter_batches()
        if batch.input_sha256 in heldout_record_ids
    )
    replay_record_ids = tuple(batch.input_sha256 for batch in heldout_batches)
    if (
        len(replay_record_ids) != len(heldout_record_ids)
        or len(set(replay_record_ids)) != len(replay_record_ids)
        or set(replay_record_ids) != set(heldout_record_ids)
    ):
        raise Lfm2DenseBudgetFoldError(
            "The sealed parent cache cannot materialize the exact held-out "
            "vocabulary-margin replay split."
        )

    child_before = fingerprint_model_state(child_model)
    replay = capture_lfm2_child_replay(
        child_model,
        heldout_batches,
        selected_layer_indices=(len(_layers(child_model)) - 1,),
        config=_candidate_replay_capture_config(trace_cache),
        replay_capture_mode=(config.vocabulary_margin_candidate_replay_mode),
    )
    child_after = fingerprint_model_state(child_model)
    if child_after != child_before:
        raise Lfm2DenseBudgetFoldError(
            "Vocabulary-margin candidate replay mutated the folded child."
        )

    metadata = replay.metadata
    runtime = metadata.get("runtime")
    forwards = metadata.get("forward_counts")
    selection = metadata.get("selection")
    benchmark = metadata.get("replay_benchmark")
    if (
        not isinstance(runtime, Mapping)
        or not isinstance(forwards, Mapping)
        or not isinstance(selection, Mapping)
        or not isinstance(benchmark, Mapping)
        or runtime.get("parent_model_supplied") is not False
        or runtime.get("parent_forward_count") != 0
        or runtime.get("tokenizer_accessed") is not False
        or runtime.get("disk_artifacts_written") != 0
        or runtime.get("child_state_preserved") is not True
        or forwards.get("parent") != 0
        or selection.get("replay_capture_mode")
        != config.vocabulary_margin_candidate_replay_mode
    ):
        raise Lfm2DenseBudgetFoldError(
            "Vocabulary-margin candidate replay crossed a sealed runtime "
            "or provenance tripwire."
        )
    actual_child_forwards = benchmark.get("actual_child_forwards")
    canonical_prefix_child_forwards = benchmark.get("canonical_prefix_child_forwards")
    saved_child_forwards = benchmark.get("saved_child_forwards")
    if (
        isinstance(actual_child_forwards, bool)
        or not isinstance(actual_child_forwards, int)
        or actual_child_forwards <= 0
        or isinstance(canonical_prefix_child_forwards, bool)
        or not isinstance(canonical_prefix_child_forwards, int)
        or canonical_prefix_child_forwards < actual_child_forwards
        or isinstance(saved_child_forwards, bool)
        or not isinstance(saved_child_forwards, int)
        or saved_child_forwards
        != canonical_prefix_child_forwards - actual_child_forwards
        or forwards.get("child") != actual_child_forwards
    ):
        raise Lfm2DenseBudgetFoldError(
            "Vocabulary-margin candidate replay forward accounting is invalid."
        )

    causal_contract_sha256: str | None = None
    single_pass_authenticated = False
    if (
        config.vocabulary_margin_candidate_replay_mode
        == CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE
    ):
        if actual_child_forwards != len(heldout_batches):
            raise Lfm2DenseBudgetFoldError(
                "Authenticated single-pass vocabulary replay did not use "
                "exactly one child forward per held-out batch."
            )
        contracts = []
        for batch in replay.iter_batches():
            contract = batch.capture.get("causal_equivalence_contract")
            if not isinstance(contract, Mapping):
                raise Lfm2DenseBudgetFoldError(
                    "Single-pass vocabulary replay omitted its authenticated "
                    "causal contract."
                )
            contracts.append(
                _digest(
                    contract.get("contract_sha256"),
                    "candidate replay causal contract",
                )
            )
        if len(set(contracts)) != 1:
            raise Lfm2DenseBudgetFoldError(
                "Single-pass vocabulary replay changed causal contract "
                "between held-out batches."
            )
        causal_contract_sha256 = contracts[0]
        single_pass_authenticated = True

    evaluation = basis.evaluate_vocabulary_margin_candidate_gate(
        replay,
        minimum_sign_agreement_fraction=(
            config.vocabulary_margin_candidate_minimum_sign_agreement_fraction
        ),
        minimum_top_competitor_nonnegative_fraction=(
            config.vocabulary_margin_candidate_minimum_top_competitor_nonnegative_fraction
        ),
        minimum_all_top_k_nonnegative_fraction=(
            config.vocabulary_margin_candidate_minimum_all_top_k_nonnegative_fraction
        ),
        minimum_exact_top1_agreement_fraction=(
            config.vocabulary_margin_candidate_minimum_exact_top1_agreement_fraction
        ),
    )
    if (
        not isinstance(evaluation, Mapping)
        or not isinstance(evaluation.get("gate_passed"), bool)
        or evaluation.get("parent_forward_calls") != 0
        or evaluation.get("candidate_forward_calls") != 0
        or evaluation.get("training_steps") != 0
    ):
        raise Lfm2DenseBudgetFoldError(
            "Vocabulary-margin candidate gate returned an invalid audit."
        )

    body = {
        "schema": VOCABULARY_MARGIN_CANDIDATE_GATE_SCHEMA,
        "enabled": True,
        "gate_passed": bool(evaluation["gate_passed"]),
        "sealed_parent_cache_key": trace_cache.cache_key,
        "heldout_record_count": len(heldout_batches),
        "heldout_record_ids_sha256": canonical_sha256(sorted(heldout_record_ids)),
        "heldout_used_for_basis_selection": False,
        "candidate_replay_capture_calls": 1,
        "candidate_replay_sha256": replay.replay_sha256,
        "candidate_replay_mode": (config.vocabulary_margin_candidate_replay_mode),
        "single_pass_causal_authenticated": single_pass_authenticated,
        "causal_contract_sha256": causal_contract_sha256,
        "selected_layer_indices": list(replay.selected_layer_indices),
        "actual_child_forward_calls": actual_child_forwards,
        "canonical_prefix_child_forward_calls": (canonical_prefix_child_forwards),
        "saved_child_forward_calls": saved_child_forwards,
        "evaluation_forward_calls": 0,
        "candidate_state_preserved_during_replay": True,
        "candidate_state_fingerprint": child_before,
        "evaluation": dict(evaluation),
        "parent_forward_calls": 0,
        "trace_regeneration_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    return {**body, "audit_sha256": canonical_sha256(body)}


def _fold_dense_blank_from_parent(
    parent_model: Any,
    child_model: Any,
    mapping: Any,
    tokenizer_bridge_artifact: StrictTokenizerBridgeArtifact | None = None,
) -> dict[str, Any]:
    """Overwrite every learned child tensor from the complete parent state."""

    torch = _torch()
    source_layers = _layers(parent_model)
    target_layers = _layers(child_model)
    if len(source_layers) != len(target_layers):
        raise Lfm2DenseBudgetFoldError(
            "Ordered source-span composition is required for a depth-changing "
            "dense target; the executable dense lane currently requires equal "
            "source/target depth."
        )
    source_types = tuple(
        _fold_layer_kind(layer, label=f"source layer {index}")
        for index, layer in enumerate(source_layers)
    )
    target_types = tuple(
        _fold_layer_kind(layer, label=f"target layer {index}")
        for index, layer in enumerate(target_layers)
    )
    if source_types != target_types:
        raise Lfm2DenseBudgetFoldError(
            "Dense width folding requires identical source/target operator order."
        )
    attention_folds: dict[int, tuple[Any, Any, tuple[int, ...], tuple[int, ...]]] = {}
    short_conv_geometries: dict[
        int,
        tuple[tuple[int, int, int], tuple[int, int, int]],
    ] = {}
    for layer_index, (source_layer, target_layer, layer_type) in enumerate(
        zip(source_layers, target_layers, source_types, strict=True)
    ):
        if layer_type == "full_attention":
            source_geometry = _attention_physical_geometry(
                source_layer.self_attn,
                label=f"source layer {layer_index}.self_attn",
            )
            target_geometry = _attention_physical_geometry(
                target_layer.self_attn,
                label=f"target layer {layer_index}.self_attn",
            )
            (
                source_query_heads,
                source_key_value_heads,
                source_head_dim,
                source_residual_width,
            ) = source_geometry
            (
                target_query_heads,
                target_key_value_heads,
                target_head_dim,
                target_residual_width,
            ) = target_geometry
            if source_residual_width != int(
                mapping.source_width
            ) or target_residual_width != int(mapping.target_width):
                raise Lfm2DenseBudgetFoldError(
                    f"Layer {layer_index} attention residual widths disagree "
                    "with the shared residual coordinate map."
                )
            attention_maps = build_gqa_attention_fold_maps(
                source_query_heads=source_query_heads,
                source_key_value_heads=source_key_value_heads,
                source_head_dim=source_head_dim,
                target_query_heads=target_query_heads,
                target_key_value_heads=target_key_value_heads,
                target_head_dim=target_head_dim,
            )
            rope_pair_map = build_gqa_attention_fold_maps(
                source_query_heads=1,
                source_key_value_heads=1,
                source_head_dim=source_head_dim,
                target_query_heads=1,
                target_key_value_heads=1,
                target_head_dim=target_head_dim,
            ).query
            attention_folds[layer_index] = (
                attention_maps,
                rope_pair_map,
                source_geometry,
                target_geometry,
            )
        else:
            source_geometry = _dense_short_conv_physical_geometry(
                source_layer.conv,
                label=f"source layer {layer_index}.conv",
            )
            target_geometry = _dense_short_conv_physical_geometry(
                target_layer.conv,
                label=f"target layer {layer_index}.conv",
            )
            if source_geometry[1] != int(mapping.source_width) or target_geometry[
                1
            ] != int(mapping.target_width):
                raise Lfm2DenseBudgetFoldError(
                    f"Layer {layer_index} ShortConv residual widths disagree "
                    "with the shared residual coordinate map."
                )
            if source_geometry[2] != target_geometry[2]:
                raise Lfm2DenseBudgetFoldError(
                    f"Layer {layer_index} changes its ShortConv FIR length."
                )
            short_conv_geometries[layer_index] = (
                source_geometry,
                target_geometry,
            )
    written: set[int] = set()
    layer_reports = []

    with torch.no_grad():
        source_embedding = parent_model.get_input_embeddings().weight
        target_embedding = child_model.get_input_embeddings().weight
        source_head = parent_model.get_output_embeddings().weight
        target_head = child_model.get_output_embeddings().weight
        target_tied = id(target_head) == id(target_embedding)
        source_tied = id(source_head) == id(source_embedding)
        vocabulary_transport_report: Mapping[str, Any] | None = None
        if tokenizer_bridge_artifact is not None:
            if target_tied and not source_tied:
                raise Lfm2DenseBudgetFoldError(
                    "A tied dense target cannot exactly represent an untied "
                    "source token interface."
                )
            transported = compile_strict_vocabulary_operators(
                source_embedding,
                source_head,
                tokenizer_bridge_artifact,
                mapping,
                mapping,
                source_lm_head_bias=getattr(
                    parent_model.get_output_embeddings(),
                    "bias",
                    None,
                ),
                require_tied_target=target_tied,
            )
            _copy_parameter(
                target_embedding,
                transported.embedding_weight,
                name="model.embed_tokens.weight",
                written=written,
            )
            if target_tied:
                written.add(id(target_head))
            else:
                _copy_parameter(
                    target_head,
                    transported.lm_head_weight,
                    name="lm_head.weight",
                    written=written,
                )
            target_bias = getattr(
                child_model.get_output_embeddings(),
                "bias",
                None,
            )
            if target_bias is not None:
                if transported.lm_head_bias is None:
                    raise Lfm2DenseBudgetFoldError(
                        "Bridged target LM-head bias lacks transported source values."
                    )
                _copy_parameter(
                    target_bias,
                    transported.lm_head_bias,
                    name="lm_head.bias",
                    written=written,
                )
            elif transported.lm_head_bias is not None:
                raise Lfm2DenseBudgetFoldError(
                    "Bridged source LM-head bias has no target parameter."
                )
            vocabulary_transport_report = dict(transported.report)
        else:
            _copy_parameter(
                target_embedding,
                mapping.fold_consumer_weight(
                    source_embedding.detach().to(
                        device="cpu",
                        dtype=torch.float64,
                    )
                ),
                name="model.embed_tokens.weight",
                written=written,
            )
            if target_tied:
                if not source_tied:
                    raise Lfm2DenseBudgetFoldError(
                        "A tied dense target cannot merge an untied source token "
                        "interface without an explicit tokenizer/output bridge."
                    )
                written.add(id(target_head))
            else:
                _copy_parameter(
                    target_head,
                    mapping.fold_consumer_weight(
                        source_head.detach().to(
                            device="cpu",
                            dtype=torch.float64,
                        )
                    ),
                    name="lm_head.weight",
                    written=written,
                )

        for layer_index, (source_layer, target_layer) in enumerate(
            zip(source_layers, target_layers, strict=True)
        ):
            layer_type = source_types[layer_index]
            canonical_kind = (
                "full_attention" if layer_type == "full_attention" else "short_conv"
            )
            _copy_parameter(
                target_layer.operator_norm.weight,
                _transport_diagonal(mapping, source_layer.operator_norm.weight),
                name=f"model.layers.{layer_index}.operator_norm.weight",
                written=written,
            )
            _copy_parameter(
                target_layer.ffn_norm.weight,
                _transport_diagonal(mapping, source_layer.ffn_norm.weight),
                name=f"model.layers.{layer_index}.ffn_norm.weight",
                written=written,
            )

            source_mlp = source_layer.feed_forward
            target_mlp = target_layer.feed_forward
            source_intermediate = int(source_mlp.w1.weight.shape[0])
            target_intermediate = int(target_mlp.w1.weight.shape[0])
            mlp_map = build_feature_fold_map(
                (
                    source_mlp.w1.weight.detach().T,
                    source_mlp.w3.weight.detach().T,
                    source_mlp.w2.weight.detach(),
                ),
                target_intermediate,
                source_width=source_intermediate,
                chunk_size=min(256, source_intermediate),
                sketch_size=min(64, max(8, int(mapping.source_width))),
                balance_strength=0.35,
                allow_sign_flip=False,
            )
            folded_gate = mapping.fold_consumer_weight(
                source_mlp.w1.weight.detach().to(device="cpu", dtype=torch.float64)
            )
            folded_up = mapping.fold_consumer_weight(
                source_mlp.w3.weight.detach().to(device="cpu", dtype=torch.float64)
            )
            folded_down = mlp_map.fold_consumer_weight(
                source_mlp.w2.weight.detach().to(device="cpu", dtype=torch.float64)
            )
            _copy_parameter(
                target_mlp.w1.weight,
                mlp_map.fold_producer_weight(folded_gate),
                name=f"model.layers.{layer_index}.feed_forward.w1.weight",
                written=written,
            )
            _copy_parameter(
                target_mlp.w3.weight,
                mlp_map.fold_producer_weight(folded_up),
                name=f"model.layers.{layer_index}.feed_forward.w3.weight",
                written=written,
            )
            _copy_parameter(
                target_mlp.w2.weight,
                mapping.fold_producer_weight(folded_down),
                name=f"model.layers.{layer_index}.feed_forward.w2.weight",
                written=written,
            )

            if layer_type == "full_attention":
                source_attention = source_layer.self_attn
                target_attention = target_layer.self_attn
                (
                    attention_maps,
                    rope_pair_map,
                    source_attention_geometry,
                    target_attention_geometry,
                ) = attention_folds[layer_index]
                for role, projection in (
                    ("query", "q_proj"),
                    ("key", "k_proj"),
                    ("value", "v_proj"),
                ):
                    source_projection = getattr(source_attention, projection)
                    target_projection = getattr(target_attention, projection)
                    channel_map = getattr(attention_maps, role)
                    folded = mapping.fold_consumer_weight(
                        source_projection.weight.detach().to(
                            device="cpu", dtype=torch.float64
                        )
                    )
                    _copy_parameter(
                        target_projection.weight,
                        channel_map.fold_producer_weight(folded),
                        name=(
                            f"model.layers.{layer_index}.self_attn.{projection}.weight"
                        ),
                        written=written,
                    )
                    _copy_optional_bias(
                        source_projection.bias,
                        target_projection.bias,
                        (
                            None
                            if source_projection.bias is None
                            else channel_map.fold_producer_weight(
                                source_projection.bias.detach()
                                .to(device="cpu", dtype=torch.float64)
                                .reshape(-1, 1)
                            ).reshape(-1)
                        ),
                        name=(
                            f"model.layers.{layer_index}.self_attn.{projection}.bias"
                        ),
                        written=written,
                    )
                source_output = source_attention.out_proj
                target_output = target_attention.out_proj
                context_folded = attention_maps.context.fold_consumer_weight(
                    source_output.weight.detach().to(device="cpu", dtype=torch.float64)
                )
                _copy_parameter(
                    target_output.weight,
                    mapping.fold_producer_weight(context_folded),
                    name=(f"model.layers.{layer_index}.self_attn.out_proj.weight"),
                    written=written,
                )
                _copy_optional_bias(
                    source_output.bias,
                    target_output.bias,
                    (
                        None
                        if source_output.bias is None
                        else mapping.fold_producer_weight(
                            source_output.bias.detach()
                            .to(device="cpu", dtype=torch.float64)
                            .reshape(-1, 1)
                        ).reshape(-1)
                    ),
                    name=f"model.layers.{layer_index}.self_attn.out_proj.bias",
                    written=written,
                )
                for norm_name in ("q_layernorm", "k_layernorm"):
                    source_norm = getattr(source_attention, norm_name, None)
                    target_norm = getattr(target_attention, norm_name, None)
                    if (source_norm is None) != (target_norm is None):
                        raise Lfm2DenseBudgetFoldError(
                            f"Attention {norm_name} presence differs."
                        )
                    if source_norm is not None:
                        _copy_parameter(
                            target_norm.weight,
                            _transport_feature_diagonal(
                                rope_pair_map,
                                source_norm.weight,
                            ),
                            name=(
                                f"model.layers.{layer_index}.self_attn."
                                f"{norm_name}.weight"
                            ),
                            written=written,
                        )
                operator_report = {
                    "kind": "gqa_rope_pair_constrained_full_source_fold",
                    "query": attention_maps.query.report.as_dict(),
                    "key": attention_maps.key.report.as_dict(),
                    "value": attention_maps.value.report.as_dict(),
                    "context": attention_maps.context.report.as_dict(),
                    "source_physical_geometry": {
                        "query_heads": source_attention_geometry[0],
                        "key_value_heads": source_attention_geometry[1],
                        "head_dim": source_attention_geometry[2],
                    },
                    "target_physical_geometry": {
                        "query_heads": target_attention_geometry[0],
                        "key_value_heads": target_attention_geometry[1],
                        "head_dim": target_attention_geometry[2],
                    },
                    "source_heads_dropped": 0,
                    "rope_pairs_preserved": True,
                }
            else:
                source_conv = source_layer.conv
                target_conv = target_layer.conv
                source_blocks = source_conv.in_proj.weight.chunk(3, dim=0)
                source_kernel = source_conv.conv.weight.detach()[:, 0, :]
                source_channels = short_conv_geometries[layer_index][0][0]
                target_channels = short_conv_geometries[layer_index][1][0]
                internal_map = build_feature_fold_map(
                    (
                        *(block.detach().T for block in source_blocks),
                        source_conv.out_proj.weight.detach(),
                        source_kernel.T,
                    ),
                    target_channels,
                    source_width=source_channels,
                    chunk_size=min(256, source_channels),
                    sketch_size=min(64, max(8, source_channels)),
                    balance_strength=0.35,
                    allow_sign_flip=False,
                )
                _copy_parameter(
                    target_conv.in_proj.weight,
                    torch.cat(
                        tuple(
                            internal_map.fold_producer_weight(
                                mapping.fold_consumer_weight(
                                    block.detach().to(
                                        device="cpu",
                                        dtype=torch.float64,
                                    )
                                )
                            )
                            for block in source_blocks
                        ),
                        dim=0,
                    ),
                    name=f"model.layers.{layer_index}.conv.in_proj.weight",
                    written=written,
                )
                _copy_parameter(
                    target_conv.out_proj.weight,
                    mapping.fold_producer_weight(
                        internal_map.fold_consumer_weight(
                            source_conv.out_proj.weight.detach().to(
                                device="cpu",
                                dtype=torch.float64,
                            )
                        )
                    ),
                    name=f"model.layers.{layer_index}.conv.out_proj.weight",
                    written=written,
                )
                _copy_parameter(
                    target_conv.conv.weight,
                    _transport_depthwise_kernel(
                        internal_map,
                        source_kernel,
                    ).unsqueeze(1),
                    name=f"model.layers.{layer_index}.conv.conv.weight",
                    written=written,
                )
                source_in_bias = source_conv.in_proj.bias
                target_in_bias = target_conv.in_proj.bias
                _copy_optional_bias(
                    source_in_bias,
                    target_in_bias,
                    (
                        None
                        if source_in_bias is None
                        else torch.cat(
                            tuple(
                                internal_map.fold_producer_weight(
                                    block.detach()
                                    .to(device="cpu", dtype=torch.float64)
                                    .reshape(-1, 1)
                                ).reshape(-1)
                                for block in source_in_bias.chunk(3)
                            )
                        )
                    ),
                    name=f"model.layers.{layer_index}.conv.in_proj.bias",
                    written=written,
                )
                for projection_name in ("out_proj",):
                    source_bias = getattr(source_conv, projection_name).bias
                    target_bias = getattr(target_conv, projection_name).bias
                    _copy_optional_bias(
                        source_bias,
                        target_bias,
                        (
                            None
                            if source_bias is None
                            else mapping.fold_producer_weight(
                                source_bias.detach()
                                .to(device="cpu", dtype=torch.float64)
                                .reshape(-1, 1)
                            ).reshape(-1)
                        ),
                        name=(
                            f"model.layers.{layer_index}.conv.{projection_name}.bias"
                        ),
                        written=written,
                    )
                _copy_optional_bias(
                    source_conv.conv.bias,
                    target_conv.conv.bias,
                    (
                        None
                        if source_conv.conv.bias is None
                        else internal_map.fold_producer_weight(
                            source_conv.conv.bias.detach()
                            .to(device="cpu", dtype=torch.float64)
                            .reshape(-1, 1)
                        ).reshape(-1)
                    ),
                    name=f"model.layers.{layer_index}.conv.conv.bias",
                    written=written,
                )
                operator_report = {
                    "kind": "dense_residual_internal_short_conv_transport",
                    "input_projection": "F.T @ W_block @ B",
                    "output_projection": "B.T @ W @ F",
                    "depthwise_fir": "diag(F.T @ diag(kernel_tap) @ F)",
                    "source_internal_channels": source_channels,
                    "target_internal_channels": target_channels,
                    "internal_feature_fold": internal_map.report.as_dict(),
                    "source_channels_dropped": 0,
                }

            layer_reports.append(
                {
                    "layer_index": layer_index,
                    "kind": canonical_kind,
                    "swiglu": {
                        "source_neurons": source_intermediate,
                        "target_neurons": target_intermediate,
                        "source_neurons_dropped": 0,
                        "full_source_coverage": True,
                        "feature_fold": mlp_map.report.as_dict(),
                    },
                    "operator": operator_report,
                }
            )

        source_final_norm = parent_model.model.embedding_norm
        target_final_norm = child_model.model.embedding_norm
        _copy_parameter(
            target_final_norm.weight,
            _transport_diagonal(mapping, source_final_norm.weight),
            name="model.embedding_norm.weight",
            written=written,
        )

    target_unique = {id(parameter) for parameter in child_model.parameters()}
    missing = target_unique - written
    if missing:
        names = [
            name
            for name, parameter in child_model.named_parameters(
                recurse=True,
                remove_duplicate=False,
            )
            if id(parameter) in missing
        ]
        raise Lfm2DenseBudgetFoldError(
            "Dense fold did not overwrite every target parameter: "
            + ", ".join(sorted(names))
        )
    nonzero = sum(
        int(torch.count_nonzero(parameter.detach()).item())
        for parameter in child_model.parameters()
    )
    if nonzero <= 0:
        raise Lfm2DenseBudgetFoldError("Dense fold left the compiler-zero child blank.")
    return {
        "schema": DENSE_BUDGET_FOLD_SCHEMA,
        "layers": layer_reports,
        "complete_target_parameter_overwrite": True,
        "target_unique_parameter_objects_written": len(written),
        "target_nonzero_parameter_elements": nonzero,
        "source_layer_count": len(source_layers),
        "target_layer_count": len(target_layers),
        "source_layer_types": list(source_types),
        "target_layer_types": list(target_types),
        "depth_composition_used": False,
        "quantization_used": False,
        "pruning_used": False,
        "training_used": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "parent_forward_calls": 0,
        "parent_trace_regeneration_calls": 0,
        "tokenizer_bridge_used": tokenizer_bridge_artifact is not None,
        "strict_vocabulary_operator_transport": vocabulary_transport_report,
    }


class Lfm2DenseBudgetModelTransaction:
    """Whole-child transaction around initial transport and closed-loop repair."""

    method_id = DENSE_BUDGET_MODEL_METHOD

    def __init__(
        self,
        context: FoldExecutionContext,
        snapshot: tuple[tuple[str, Any], ...],
        topology_before: str,
        incumbent_state_fingerprint: str,
    ) -> None:
        self.context = context
        self._snapshot = snapshot
        self._topology_before = topology_before
        self.incumbent_state_fingerprint = incumbent_state_fingerprint
        self.candidate_state_fingerprint = incumbent_state_fingerprint
        self.plan = Lfm2DenseBudgetFoldPlan(
            {
                "schema": DENSE_BUDGET_FOLD_SCHEMA,
                "zero_gradient": True,
                "seed_erasure_by_construction": False,
                "prepared": False,
            }
        )
        self.active = True

    def bind_candidate(
        self,
        *,
        plan: Lfm2DenseBudgetFoldPlan,
        candidate_state_fingerprint: str,
    ) -> None:
        if not self.active:
            raise Lfm2DenseBudgetFoldError("Cannot bind an inactive dense transaction.")
        candidate = _digest(
            candidate_state_fingerprint,
            "candidate_state_fingerprint",
        )
        if candidate == self.incumbent_state_fingerprint:
            raise Lfm2DenseBudgetFoldError(
                "Dense fold candidate is identical to the blank child."
            )
        if _topology_fingerprint(self.context.child_model) != self._topology_before:
            raise Lfm2DenseBudgetFoldError("Dense fold changed Parameter topology.")
        self.plan = plan
        self.candidate_state_fingerprint = candidate

    def commit(self) -> Mapping[str, Any]:
        if not self.active:
            raise Lfm2DenseBudgetFoldError("Dense fold transaction is inactive.")
        final = self.context.child_state_fingerprint
        if (
            final != self.candidate_state_fingerprint
            or _topology_fingerprint(self.context.child_model) != self._topology_before
        ):
            raise Lfm2DenseBudgetFoldError(
                "Dense fold candidate changed before commit."
            )
        self.active = False
        result = {
            **self.plan.as_dict(),
            "action": "commit",
            "accepted": True,
            "exact": True,
            "candidate_state_fingerprint": self.candidate_state_fingerprint,
            "final_state_fingerprint": final,
            "parameter_topology_preserved": True,
        }
        diagnostic = self.plan.report.get("whole_child_component_diagnostic_execution")
        if isinstance(diagnostic, Mapping):
            result[TRANSACTION_REPORT_BINDING_KEY] = diagnostic["binding"]
        return result

    def rollback(self) -> Mapping[str, Any]:
        if not self.active:
            raise Lfm2DenseBudgetFoldError("Dense fold transaction is inactive.")
        _restore_state(self.context.child_model, self._snapshot)
        final = self.context.child_state_fingerprint
        topology = _topology_fingerprint(self.context.child_model)
        if (
            final != self.incumbent_state_fingerprint
            or topology != self._topology_before
        ):
            raise Lfm2DenseBudgetFoldError(
                "Dense fold rollback did not exactly restore the blank."
            )
        self.active = False
        result = {
            **self.plan.as_dict(),
            "action": "rollback",
            "accepted": False,
            "exact": True,
            "rollback_exact": True,
            "candidate_state_fingerprint": self.candidate_state_fingerprint,
            "final_state_fingerprint": final,
            "parameter_topology_preserved": True,
            "zero_gradient": True,
        }
        diagnostic = self.plan.report.get("whole_child_component_diagnostic_execution")
        if isinstance(diagnostic, Mapping):
            result[TRANSACTION_REPORT_BINDING_KEY] = diagnostic["binding"]
        return result


class Lfm2DenseBudgetModelOperator:
    """One authenticated, full-coverage dense-width model FoldOperator."""

    method_id = DENSE_BUDGET_MODEL_METHOD
    whole_child_component_diagnostic_capability = (
        WHOLE_CHILD_DIAGNOSTIC_OPERATOR_CAPABILITY
    )

    def __init__(
        self,
        selected_target: Lfm2DenseBudgetedSelectedTarget,
        trace_cache: Lfm2CircuitTraceCache,
        *,
        config: Lfm2DenseBudgetFoldConfig | None = None,
        verification_lease: Lfm2DenseCacheVerificationLease | None = None,
        selective_session: Lfm2CircuitTraceSelectiveAuthenticatedSession | None = None,
        tokenizer_bridge_artifact: StrictTokenizerBridgeArtifact | None = None,
        tokenizer_bridge_trace_adapter: (
            Lfm2StrictTokenizerBridgeTraceAdapter | None
        ) = None,
    ) -> None:
        if not isinstance(
            selected_target,
            Lfm2DenseBudgetedSelectedTarget,
        ):
            raise Lfm2DenseBudgetFoldError(
                "selected_target must be Lfm2DenseBudgetedSelectedTarget."
            )
        if not isinstance(trace_cache, Lfm2CircuitTraceCache):
            raise Lfm2DenseBudgetFoldError(
                "Dense folding requires a typed sealed Lfm2CircuitTraceCache."
            )
        self.selected_target = selected_target
        if selective_session is not None and not isinstance(
            selective_session,
            Lfm2CircuitTraceSelectiveAuthenticatedSession,
        ):
            raise Lfm2DenseBudgetFoldError(
                "selective_session must be an authenticated LFM2 cache session."
            )
        if selective_session is not None and (
            selective_session.cache_key != trace_cache.cache_key
            or selective_session.path != trace_cache.path
            or len(selective_session) != len(trace_cache)
        ):
            raise Lfm2DenseBudgetFoldError(
                "selective_session differs from the dense parent cache."
            )
        self.selective_session = selective_session
        self.verification_lease = verification_lease
        self.trace_cache = _verify_dense_trace_cache(
            trace_cache,
            verification_lease,
        )
        self.config = config or Lfm2DenseBudgetFoldConfig()
        if not isinstance(self.config, Lfm2DenseBudgetFoldConfig):
            raise Lfm2DenseBudgetFoldError("config must be Lfm2DenseBudgetFoldConfig.")
        if selected_target.tokenizer_bridge_required:
            if not isinstance(
                tokenizer_bridge_artifact,
                StrictTokenizerBridgeArtifact,
            ):
                raise Lfm2DenseBudgetFoldError(
                    "Bridged dense target requires its explicit strict "
                    "tokenizer bridge artifact."
                )
            try:
                tokenizer_bridge_artifact.verify_files()
            except ValueError as exc:
                raise Lfm2DenseBudgetFoldError(
                    "Dense tokenizer bridge artifact failed verification."
                ) from exc
            if (
                tokenizer_bridge_artifact.artifact_sha256
                != selected_target.tokenizer_bridge_artifact_sha256
                or tokenizer_bridge_artifact.profile_fingerprint
                != selected_target.tokenizer_transport_profile_sha256
                or tokenizer_bridge_artifact.donor_tokenizer_semantics_sha256
                != selected_target.parent_tokenizer_semantics_sha256
                or tokenizer_bridge_artifact.target_tokenizer_semantics_sha256
                != selected_target.child_tokenizer_semantics_sha256
                or tokenizer_bridge_artifact.transport_profile.target_row_capacity
                != selected_target.vocabulary_size
            ):
                raise Lfm2DenseBudgetFoldError(
                    "Dense tokenizer bridge artifact differs from selected "
                    "target metadata."
                )
            if tokenizer_bridge_trace_adapter is not None:
                if not isinstance(
                    tokenizer_bridge_trace_adapter,
                    Lfm2StrictTokenizerBridgeTraceAdapter,
                ):
                    raise Lfm2DenseBudgetFoldError(
                        "Dense tokenizer bridge trace adapter has the wrong type."
                    )
                try:
                    tokenizer_bridge_trace_adapter.verify()
                except Lfm2StrictTokenizerBridgeTraceError as exc:
                    raise Lfm2DenseBudgetFoldError(
                        "Dense tokenizer bridge trace adapter failed verification."
                    ) from exc
                ordered_cache_inputs = tuple(
                    str(value)
                    for value in self.trace_cache.manifest["corpus"][
                        "ordered_batch_input_sha256"
                    ]
                )
                if (
                    tokenizer_bridge_trace_adapter.bridge_artifact_sha256
                    != tokenizer_bridge_artifact.artifact_sha256
                    or tokenizer_bridge_trace_adapter.transport_profile_fingerprint
                    != tokenizer_bridge_artifact.profile_fingerprint
                    or tokenizer_bridge_trace_adapter.record_count
                    != len(self.trace_cache)
                    or any(
                        record.ordinal != index
                        or record.donor_cache_input_sha256 != input_sha256
                        for index, (record, input_sha256) in enumerate(
                            zip(
                                tokenizer_bridge_trace_adapter.records,
                                ordered_cache_inputs,
                                strict=True,
                            )
                        )
                    )
                ):
                    raise Lfm2DenseBudgetFoldError(
                        "Dense tokenizer bridge trace evidence differs from the "
                        "artifact or sealed parent cache."
                    )
            elif self.config.run_closed_loop:
                raise Lfm2DenseBudgetFoldError(
                    "Bridged closed-loop folding requires its sealed target-token "
                    "trace adapter."
                )
            if (
                self.config.vocabulary_margin_candidate_gate_enabled
                or self.config.macro_cell_attractor_gate_config is not None
            ):
                raise Lfm2DenseBudgetFoldError(
                    "Bridged vocabulary-margin or macro-attractor replay requires "
                    "its target-token-aware evaluator; use the bridged closed loop "
                    "and external whole-output semantic evaluation for this run."
                )
        elif tokenizer_bridge_artifact is not None:
            raise Lfm2DenseBudgetFoldError(
                "Unbridged dense target received an unused tokenizer bridge artifact."
            )
        elif tokenizer_bridge_trace_adapter is not None:
            raise Lfm2DenseBudgetFoldError(
                "Unbridged dense target received an unused tokenizer bridge "
                "trace adapter."
            )
        self.tokenizer_bridge_artifact = tokenizer_bridge_artifact
        self.tokenizer_bridge_trace_adapter = tokenizer_bridge_trace_adapter

    def _validate_action(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
    ) -> tuple[
        str,
        Any,
        ModelCircuitGraph,
        AuthenticatedWholeChildComponentDispatch | None,
    ]:
        if action.method_id != self.method_id:
            raise Lfm2DenseBudgetFoldError(
                "Dense LFM2 operator received a different method."
            )
        if len(action.sources) != 1 or action.compression_delta <= 0:
            raise Lfm2DenseBudgetFoldError(
                "Dense LFM2 execution requires one parent and positive compaction."
            )
        source = action.sources[0]
        if source.parent_name != self.selected_target.source_parent:
            raise Lfm2DenseBudgetFoldError(
                "Dense action source differs from the selected catalog parent."
            )
        parent_graph = context.parent_graphs[source.parent_name]
        if (
            parent_graph.fingerprint != self.selected_target.source_graph_fingerprint
            or parent_graph.tokenizer_fingerprint
            != self.selected_target.parent_tokenizer_semantics_sha256
            or set(source.node_ids) != {node.node_id for node in parent_graph.nodes}
            or set(action.target_node_ids)
            != {node.node_id for node in context.child_graph.nodes}
        ):
            raise Lfm2DenseBudgetFoldError(
                "Dense action must own the authenticated complete source and target."
            )
        expected_paths = {
            name
            for name, _parameter in context.child_model.named_parameters(
                recurse=True,
                remove_duplicate=False,
            )
        }
        if set(action.target_parameter_paths) != expected_paths:
            raise Lfm2DenseBudgetFoldError(
                "Dense action does not own every target parameter path."
            )
        configuration = dict(action.configuration)
        expected = {
            "catalog_id": self.selected_target.catalog_id,
            "source_depth": len(parent_graph.layer_indices),
            "target_depth": self.selected_target.target_depth,
            "source_spans": encode_lfm2_source_spans(
                self.selected_target.ordered_source_spans
            ),
            "target_layer_types": encode_lfm2_layer_types(
                self.selected_target.layer_types
            ),
            "target_hidden_size": self.selected_target.hidden_size,
            "target_intermediate_size": self.selected_target.intermediate_size,
            "target_query_heads": self.selected_target.query_heads,
            "target_key_value_heads": self.selected_target.key_value_heads,
            "target_head_dim": self.selected_target.head_dim,
        }
        if self.selected_target.tokenizer_bridge_required:
            expected.update(
                {
                    "tokenizer_bridge_required": True,
                    "tokenizer_bridge_artifact_sha256": (
                        self.selected_target.tokenizer_bridge_artifact_sha256
                    ),
                    "tokenizer_transport_profile_sha256": (
                        self.selected_target.tokenizer_transport_profile_sha256
                    ),
                    "parent_tokenizer_semantics_sha256": (
                        self.selected_target.parent_tokenizer_semantics_sha256
                    ),
                    "child_tokenizer_semantics_sha256": (
                        self.selected_target.child_tokenizer_semantics_sha256
                    ),
                }
            )
        if any(configuration.get(name) != value for name, value in expected.items()):
            raise Lfm2DenseBudgetFoldError(
                "Dense action topology differs from its selected catalog target."
            )
        if len(parent_graph.layer_indices) != self.selected_target.target_depth:
            raise Lfm2DenseBudgetFoldError(
                "Ordered source-span composition is required for this depth-"
                "changing dense target; 4x/10x depth compaction is fail-closed "
                "until that composition operator is implemented."
            )
        expected_spans = tuple(
            (index,) for index in range(len(parent_graph.layer_indices))
        )
        if self.selected_target.ordered_source_spans != expected_spans:
            raise Lfm2DenseBudgetFoldError(
                "Same-depth dense folding requires one source layer per target."
            )
        parent_model = context.parent_models[source.parent_name]
        if fingerprint_model_state(parent_model) != (
            self.selected_target.source_state_fingerprint
        ):
            raise Lfm2DenseBudgetFoldError(
                "Dense source checkpoint changed after target selection."
            )
        validate_lfm2_dense_budgeted_child(
            self.selected_target,
            context.child_model,
            context.child_graph,
        )
        try:
            dispatch = authenticate_whole_child_component_dispatch(
                action,
                context.child_graph,
                expected_method_id=self.method_id,
            )
        except WholeChildComponentDiagnosticError as exc:
            raise Lfm2DenseBudgetFoldError(
                "Dense LFM2 component dispatch failed authentication."
            ) from exc
        if dispatch is not None:
            _component_stage_assignments(dispatch)
        return source.parent_name, parent_model, parent_graph, dispatch

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
    ) -> FoldTransaction:
        (
            source_name,
            parent_model,
            parent_graph,
            component_dispatch,
        ) = self._validate_action(action, context)
        if not isinstance(
            evidence.payload,
            Lfm2SealedDenseBudgetEvidence,
        ):
            raise Lfm2DenseBudgetFoldError(
                "Dense fold evidence lost its verified trace-cache seal."
            )
        sealed_evidence = evidence.payload.verify()
        if sealed_evidence.trace_cache is not self.trace_cache:
            raise Lfm2DenseBudgetFoldError(
                "Dense fold evidence contains a different trace-cache handle."
            )
        incumbent = context.child_state_fingerprint
        topology_before = _topology_fingerprint(context.child_model)
        snapshot = _state_snapshot(context.child_model)
        parent_before = fingerprint_model_state(parent_model)
        transaction = Lfm2DenseBudgetModelTransaction(
            context,
            snapshot,
            topology_before,
            incumbent,
        )
        try:
            basis = _build_residual_basis(
                parent_model,
                parent_graph,
                self.trace_cache,
                self.selected_target.hidden_size,
                self.config,
                selective_session=self.selective_session,
            )
            initial_report = _fold_dense_blank_from_parent(
                parent_model,
                context.child_model,
                basis,
                self.tokenizer_bridge_artifact,
            )
            initial_state = context.child_state_fingerprint
            structural_snapshot = (
                None
                if component_dispatch is None
                else _parameter_snapshot(context.child_model)
            )
            structural_buffers = (
                None
                if component_dispatch is None
                else _buffer_snapshot(context.child_model)
            )
            component_path_stages = (
                {}
                if component_dispatch is None
                else dict(_component_stage_assignments(component_dispatch))
            )
            execute_closed_loop = (
                self.config.run_closed_loop
                if component_dispatch is None
                else LFM2_COMPONENT_CLOSED_LOOP_STAGE
                in set(component_path_stages.values())
            )
            if execute_closed_loop and not self.config.run_closed_loop:
                raise Lfm2DenseBudgetFoldError(
                    "Component diagnostics requested cached closed-loop refinement "
                    "from an operator configured without that production stage."
                )
            closed_loop_report: Mapping[str, Any] | None = None
            if execute_closed_loop:
                macro_cell_attractor_gate = (
                    None
                    if self.config.macro_cell_attractor_gate_config is None
                    else Lfm2MacroCellAttractorGate(
                        self.trace_cache,
                        config=(self.config.macro_cell_attractor_gate_config),
                    )
                )
                adapter = Lfm2DynamicClosedLoopAdapter(
                    source_model=parent_model,
                    child_model=context.child_model,
                    parent_cache=self.trace_cache,
                    coordinate_map=basis,
                    initial_child_basis_sha256=basis.map_sha256,
                    config=self.config.closed_loop_adapter_config,
                    macro_cell_attractor_gate=macro_cell_attractor_gate,
                    tokenizer_bridge_artifact=self.tokenizer_bridge_artifact,
                    tokenizer_bridge_trace_adapter=(
                        self.tokenizer_bridge_trace_adapter
                    ),
                    selective_session=self.selective_session,
                    source_graph_fingerprint=(
                        parent_graph.fingerprint
                        if self.selected_target.heterogeneous_layout
                        else None
                    ),
                    heterogeneous_layout_selection_receipt=(
                        self.selected_target.heterogeneous_layout_selection_receipt
                    ),
                    identity_tokenizer_evidence_sha256=(
                        None
                        if (
                            not self.selected_target.heterogeneous_layout
                            or self.tokenizer_bridge_artifact is not None
                        )
                        else lfm2_identity_tokenizer_evidence_sha256(
                            tokenizer_semantics_sha256=(
                                self.selected_target.parent_tokenizer_semantics_sha256
                            ),
                            vocabulary_size=(self.selected_target.vocabulary_size),
                        )
                    ),
                )
                closed_loop_report = ChildClosedLoopCompiler(
                    adapter,
                    self.config.closed_loop_compiler_config,
                ).run()
            component_dispatch_report: Mapping[str, Any] | None = None
            if component_dispatch is not None:
                assert structural_snapshot is not None
                assert structural_buffers is not None
                closed_loop_snapshot = (
                    _parameter_snapshot(context.child_model)
                    if execute_closed_loop
                    else None
                )
                if execute_closed_loop:
                    _assert_buffer_snapshot_equal(
                        structural_buffers,
                        _buffer_snapshot(context.child_model),
                    )
                    assert closed_loop_snapshot is not None
                    _compose_component_stage_parameters(
                        context.child_model,
                        path_stages=component_path_stages,
                        structural_snapshot=structural_snapshot,
                        closed_loop_snapshot=closed_loop_snapshot,
                    )
                component_dispatch_report = _component_dispatch_execution_report(
                    component_dispatch,
                    context.child_model,
                    path_stages=component_path_stages,
                    structural_snapshot=structural_snapshot,
                    closed_loop_snapshot=closed_loop_snapshot,
                    exact_target_unique_parameters=(
                        self.selected_target.exact_target_unique_parameters
                    ),
                )
            candidate = context.child_state_fingerprint
            if fingerprint_model_state(parent_model) != parent_before:
                raise Lfm2DenseBudgetFoldError("Dense model fold mutated its parent.")
            if _topology_fingerprint(context.child_model) != topology_before:
                raise Lfm2DenseBudgetFoldError(
                    "Dense model fold changed child Parameter topology."
                )
            vocabulary_margin_candidate_gate = (
                _evaluate_vocabulary_margin_candidate_gate(
                    context.child_model,
                    self.trace_cache,
                    basis,
                    self.config,
                )
                if self.config.vocabulary_margin_candidate_gate_enabled
                else _disabled_vocabulary_margin_candidate_gate_report()
            )
            if context.child_state_fingerprint != candidate:
                raise Lfm2DenseBudgetFoldError(
                    "Vocabulary-margin candidate gating changed the folded child."
                )
            if fingerprint_model_state(parent_model) != parent_before:
                raise Lfm2DenseBudgetFoldError(
                    "Vocabulary-margin candidate gating mutated its parent."
                )
            report = {
                "schema": DENSE_BUDGET_FOLD_SCHEMA,
                "method_id": self.method_id,
                "source_parent": source_name,
                "catalog_id": self.selected_target.catalog_id,
                "tokenizer_bridge_required": (
                    self.selected_target.tokenizer_bridge_required
                ),
                "tokenizer_bridge_artifact_sha256": (
                    self.selected_target.tokenizer_bridge_artifact_sha256
                ),
                "tokenizer_transport_profile_sha256": (
                    self.selected_target.tokenizer_transport_profile_sha256
                ),
                **(
                    {
                        "tokenizer_bridge_trace_evidence_sha256": (
                            self.tokenizer_bridge_trace_adapter.evidence_sha256
                        ),
                        "tokenizer_bridge_target_token_replay": bool(
                            execute_closed_loop and closed_loop_report is not None
                        ),
                    }
                    if self.tokenizer_bridge_trace_adapter is not None
                    else {}
                ),
                "exact_target_unique_parameters": (
                    self.selected_target.exact_target_unique_parameters
                ),
                "residual_basis_mode": self.config.residual_basis_mode,
                "streaming_vocabulary_margin_v2": (
                    self.config.streaming_vocabulary_margin_v2
                ),
                "residual_basis_sha256": basis.map_sha256,
                "residual_basis": basis.as_dict(),
                "initial_structural_fold": initial_report,
                "initial_structural_state_fingerprint": initial_state,
                "closed_loop_enabled": execute_closed_loop,
                "macro_cell_attractor_gate_enabled": (
                    self.config.macro_cell_attractor_gate_config is not None
                ),
                "closed_loop": (
                    None if closed_loop_report is None else dict(closed_loop_report)
                ),
                **(
                    {
                        "closed_loop_configured": self.config.run_closed_loop,
                        "whole_child_component_diagnostic_execution": dict(
                            component_dispatch_report
                        ),
                    }
                    if component_dispatch_report is not None
                    else {}
                ),
                "vocabulary_margin_candidate_gate": (vocabulary_margin_candidate_gate),
                "complete_target_parameter_overwrite": True,
                "seed_erasure_by_construction": True,
                "parameter_topology_preserved": True,
                "parameter_count_preserved": True,
                "parent_immutable": True,
                "parent_trace_cache_reused": True,
                "parent_traces_regenerated": False,
                "quantization_used": False,
                "pruning_used": False,
                "training_used": False,
                "zero_gradient": True,
                "optimizer_steps": 0,
                "backward_calls": 0,
            }
            plan = Lfm2DenseBudgetFoldPlan(report)
            transaction.bind_candidate(
                plan=plan,
                candidate_state_fingerprint=candidate,
            )
            return transaction
        except BaseException:
            if transaction.active:
                transaction.rollback()
            raise


class Lfm2DenseBudgetEvidenceProvider:
    """Expose one verified cache handle without regenerating parent traces."""

    def __init__(
        self,
        trace_cache: Lfm2CircuitTraceCache,
        parent_cache_fingerprint: str,
        *,
        verification_lease: Lfm2DenseCacheVerificationLease | None = None,
    ) -> None:
        if not isinstance(trace_cache, Lfm2CircuitTraceCache):
            raise Lfm2DenseBudgetFoldError("trace_cache must be Lfm2CircuitTraceCache.")
        self.verification_lease = verification_lease
        self.trace_cache = _verify_dense_trace_cache(
            trace_cache,
            verification_lease,
        )
        self.sealed_evidence = Lfm2SealedDenseBudgetEvidence(
            self.trace_cache,
            verification_lease=verification_lease,
        )
        self.parent_cache_fingerprint = _digest(
            parent_cache_fingerprint,
            "parent_cache_fingerprint",
        )
        if self.trace_cache.cache_key != self.parent_cache_fingerprint:
            raise Lfm2DenseBudgetFoldError(
                "Trace cache key differs from the stack evidence fingerprint."
            )
        self.prepare_count = 0
        self.refresh_count = 0
        self.restore_count = 0

    @property
    def base_fingerprint(self) -> str:
        return self.parent_cache_fingerprint

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
    ) -> FoldEvidence:
        del context
        if action.method_id != DENSE_BUDGET_MODEL_METHOD:
            raise Lfm2DenseBudgetFoldError(
                "Dense evidence provider received an unsupported method."
            )
        _verify_dense_trace_cache(
            self.trace_cache,
            self.verification_lease,
        )
        self.prepare_count += 1
        return FoldEvidence(
            self.parent_cache_fingerprint,
            payload=self.sealed_evidence,
            channels=action.evidence_requirements,
            parent_forward_count=0,
            child_forward_count=0,
            root_fingerprint=self.parent_cache_fingerprint,
        )

    def refresh(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        transaction_report: Mapping[str, Any],
    ) -> None:
        del action, context, transaction_report
        _verify_dense_trace_cache(
            self.trace_cache,
            self.verification_lease,
        )
        self.refresh_count += 1

    def restore(
        self,
        context: FoldExecutionContext,
        state_fingerprint: str,
        action_ids: Sequence[str],
    ) -> None:
        del action_ids
        if context.child_state_fingerprint != _digest(
            state_fingerprint,
            "state_fingerprint",
        ):
            raise Lfm2DenseBudgetFoldError(
                "Dense evidence restore observed an unrestored child."
            )
        _verify_dense_trace_cache(
            self.trace_cache,
            self.verification_lease,
        )
        self.restore_count += 1


class Lfm2DenseBudgetFoldOnlyInitializer:
    """Non-mutating proof that the plan owns one complete dense transaction."""

    def __init__(
        self,
        parent_cache_fingerprint: str,
        selected_target: Lfm2DenseBudgetedSelectedTarget,
    ) -> None:
        self.parent_cache_fingerprint = _digest(
            parent_cache_fingerprint,
            "parent_cache_fingerprint",
        )
        if not isinstance(
            selected_target,
            Lfm2DenseBudgetedSelectedTarget,
        ):
            raise Lfm2DenseBudgetFoldError(
                "selected_target must be Lfm2DenseBudgetedSelectedTarget."
            )
        self.selected_target = selected_target

    def initialize(
        self,
        plan: DynamicFoldPlan,
        context: FoldExecutionContext,
    ) -> Mapping[str, Any]:
        before = context.child_state_fingerprint
        if (
            plan.evidence_fingerprint != self.parent_cache_fingerprint
            or plan.exact_target_unique_parameters
            != self.selected_target.exact_target_unique_parameters
            or len(plan.actions) != 1
        ):
            raise Lfm2DenseBudgetFoldError(
                "Dense plan evidence, budget, or action count is invalid."
            )
        action = plan.actions[0]
        if (
            action.method_id != DENSE_BUDGET_MODEL_METHOD
            or not action.signature.zero_gradient
            or action.compression_delta <= 0
            or set(action.target_node_ids)
            != {node.node_id for node in context.child_graph.nodes}
        ):
            raise Lfm2DenseBudgetFoldError(
                "Dense target must compile to one positive-compaction, "
                "zero-gradient, complete-model action."
            )
        if len(context.parent_graphs) != 1:
            raise Lfm2DenseBudgetFoldError(
                "Dense LFM2 folding currently accepts exactly one parent."
            )
        parent_graph = next(iter(context.parent_graphs.values()))
        if (
            parent_graph.model_name != self.selected_target.source_parent
            or parent_graph.fingerprint != self.selected_target.source_graph_fingerprint
            or dict(plan.parent_state_fingerprints)
            != {
                self.selected_target.source_parent: (
                    self.selected_target.source_state_fingerprint
                )
            }
        ):
            raise Lfm2DenseBudgetFoldError(
                "Dense execution parent differs from target selection."
            )
        if len(parent_graph.layer_indices) != self.selected_target.target_depth:
            raise Lfm2DenseBudgetFoldError(
                "Ordered source-span composition is required for this depth-"
                "changing target; the current executable lane is width-only."
            )
        validate_lfm2_dense_budgeted_child(
            self.selected_target,
            context.child_model,
            context.child_graph,
        )
        after = context.child_state_fingerprint
        if after != before:
            raise Lfm2DenseBudgetFoldError("Dense fold initializer mutated the child.")
        return {
            "kind": "lfm2_dense_budget_fold_only",
            "child_state_fingerprint": after,
            "complete_target_coverage": True,
            "parent_cache_fingerprint": self.parent_cache_fingerprint,
            "parent_traces_regenerated": False,
            "zero_gradient": True,
            "optimizer_steps": 0,
            "backward_calls": 0,
        }


class Lfm2DenseBudgetAdmissionEvaluator:
    """Enforce fold integrity while retaining numeric behavior as diagnostics."""

    def evaluate(self, context: FoldEvaluationContext) -> FoldDecision:
        transaction = context.transaction
        if not isinstance(transaction, Lfm2DenseBudgetModelTransaction):
            return FoldDecision.reject(
                "Dense LFM2 action lacks its typed whole-model transaction."
            )
        report = transaction.plan.as_dict()
        metrics: dict[str, JsonScalar] = {
            "local_gate_only": True,
            "model_level_action": True,
            "zero_gradient": bool(report.get("zero_gradient", False)),
            "seed_erasure_by_construction": bool(
                report.get("seed_erasure_by_construction", False)
            ),
            "complete_target_parameter_overwrite": bool(
                report.get("complete_target_parameter_overwrite", False)
            ),
            "parameter_topology_preserved": bool(
                report.get("parameter_topology_preserved", False)
            ),
            "quantization_used": bool(report.get("quantization_used", True)),
            "pruning_used": bool(report.get("pruning_used", True)),
            "training_used": bool(report.get("training_used", True)),
        }
        candidate_gate = report.get("vocabulary_margin_candidate_gate")
        candidate_gate_enabled = (
            isinstance(candidate_gate, Mapping)
            and candidate_gate.get("enabled") is True
        )
        candidate_gate_passed = (
            isinstance(candidate_gate, Mapping)
            and candidate_gate.get("gate_passed") is True
        )
        metrics["vocabulary_margin_candidate_gate_enabled"] = candidate_gate_enabled
        metrics["vocabulary_margin_candidate_gate_passed"] = candidate_gate_passed
        metrics["vocabulary_margin_candidate_is_decision_authority"] = False
        if candidate_gate_enabled:
            evaluation = candidate_gate.get("evaluation")
            measured = (
                evaluation.get("metrics") if isinstance(evaluation, Mapping) else None
            )
            minimums = (
                evaluation.get("minimums") if isinstance(evaluation, Mapping) else None
            )
            checks = (
                evaluation.get("checks") if isinstance(evaluation, Mapping) else None
            )
            metric_names = (
                "candidate_vs_cached_sign_agreement_fraction",
                ("candidate_top1_vs_top_competitor_nonnegative_fraction"),
                ("candidate_top1_vs_all_top_k_nonnegative_fraction"),
                "candidate_exact_parent_top1_agreement_fraction",
            )
            if (
                isinstance(measured, Mapping)
                and isinstance(minimums, Mapping)
                and isinstance(checks, Mapping)
            ):
                for name in metric_names:
                    value = measured.get(name)
                    minimum = minimums.get(name)
                    check = checks.get(name)
                    if (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and isinstance(minimum, (int, float))
                        and not isinstance(minimum, bool)
                        and isinstance(check, bool)
                    ):
                        metrics[f"vocabulary_margin_candidate_{name}"] = float(value)
                        metrics[f"vocabulary_margin_candidate_minimum_{name}"] = float(
                            minimum
                        )
                        metrics[f"vocabulary_margin_candidate_check_{name}"] = check
            for source_name in (
                "heldout_record_count",
                "actual_child_forward_calls",
                "parent_forward_calls",
                "training_steps",
            ):
                value = candidate_gate.get(source_name)
                if (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                ):
                    metrics[f"vocabulary_margin_candidate_{source_name}"] = value
            audit_sha256 = candidate_gate.get("audit_sha256")
            if isinstance(audit_sha256, str):
                metrics["vocabulary_margin_candidate_audit_sha256"] = audit_sha256
        required = (
            metrics["zero_gradient"],
            metrics["seed_erasure_by_construction"],
            metrics["complete_target_parameter_overwrite"],
            metrics["parameter_topology_preserved"],
            not metrics["quantization_used"],
            not metrics["pruning_used"],
            not metrics["training_used"],
        )
        if not all(required):
            return FoldDecision.reject(
                "Dense LFM2 model transaction failed a local fold-only invariant.",
                metrics,
            )
        return FoldDecision.accept(
            "Dense LFM2 blank was fully filled from parent-derived structural "
            "maps; numeric replay is diagnostic and complete decoded behavior "
            "remains mandatory evidence.",
            metrics,
        )


__all__ = [
    "DENSE_BUDGET_FOLD_SCHEMA",
    "DENSE_BUDGET_MODEL_METHOD",
    "LFM2_COMPONENT_CLOSED_LOOP_STAGE",
    "LFM2_COMPONENT_DISPATCH_EXECUTION_SCHEMA",
    "LFM2_COMPONENT_FOLD_STAGE_CONFIGURATION_KEY",
    "LFM2_COMPONENT_STRUCTURAL_STAGE",
    "Lfm2DenseBudgetAdmissionEvaluator",
    "Lfm2DenseBudgetEvidenceProvider",
    "Lfm2DenseBudgetFoldConfig",
    "Lfm2DenseBudgetFoldError",
    "Lfm2DenseBudgetFoldOnlyInitializer",
    "Lfm2DenseBudgetFoldPlan",
    "Lfm2DenseBudgetModelOperator",
    "Lfm2DenseBudgetModelTransaction",
    "Lfm2SealedDenseBudgetEvidence",
]
