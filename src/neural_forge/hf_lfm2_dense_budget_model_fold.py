"""Executable dense blank-first LFM2 width folding.

The dense parameter frontier materializes a real smaller LFM2 architecture
whose learned tensors are compiler-zeroed.  This module fills that blank in
one transactional, zero-gradient model action:

* one complete-weight or circuit-balanced residual coordinate basis is shared
  by the embedding, every residual-facing circuit, and the output interface;
* GQA attention channels are folded with the existing RoPE-pair-constrained
  maps;
* every source SwiGLU neuron is assigned to the smaller coupled gate/up/down
  circuit through the existing feature-fold primitive; and
* an optional cached child-closed-loop pass refines attention consumers,
  SwiGLU, final normalization, and the tied output interface without executing
  the parent or training.

Depth composition is deliberately outside this transaction.  A target must
retain the source layer count and operator order; callers get an explicit
failure instead of a misleading partial fold.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

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
from .dynamic_fold_planner import DynamicFoldPlan, FoldAction
from .dynamic_lfm2_adapter import (
    Lfm2DenseBudgetedSelectedTarget,
    encode_lfm2_layer_types,
    encode_lfm2_source_spans,
    validate_lfm2_dense_budgeted_child,
)
from .hf_attention_fold import build_gqa_attention_fold_maps
from .hf_feature_fold import build_feature_fold_map
from .hf_lfm2_canonical import inspect_lfm2_layer
from .hf_lfm2_circuit_trace_cache import (
    CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
    CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2CircuitTraceCache,
    Lfm2CircuitTraceCaptureConfig,
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
from .hf_lfm2_weight_complete_basis import build_lfm2_weight_complete_basis
from .hf_tokenizer_bridge import (
    StrictTokenizerBridgeArtifact,
    compile_strict_vocabulary_operators,
)


DENSE_BUDGET_MODEL_METHOD = "lfm2_dense_budget_residual_attention_model"
DENSE_BUDGET_FOLD_SCHEMA = "neural-forge-lfm2-dense-budget-model-fold-v1"
VOCABULARY_MARGIN_CANDIDATE_GATE_SCHEMA = (
    "neural-forge-lfm2-dense-budget-vocabulary-margin-candidate-gate-v1"
)


class Lfm2DenseBudgetFoldError(RuntimeError):
    """The dense target cannot be filled under the authenticated contract."""


class Lfm2SealedDenseBudgetEvidence:
    """Executor-safe non-dataclass seal around the typed trace cache."""

    __slots__ = ("_trace_cache", "_cache_key")

    def __init__(self, trace_cache: Lfm2CircuitTraceCache) -> None:
        if not isinstance(trace_cache, Lfm2CircuitTraceCache):
            raise Lfm2DenseBudgetFoldError(
                "trace_cache must be Lfm2CircuitTraceCache."
            )
        self._trace_cache = trace_cache.verify()
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
        self._trace_cache.verify()
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
        raise Lfm2DenseBudgetFoldError(
            f"{name} must be a lowercase SHA-256 digest."
        )
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
        raise Lfm2DenseBudgetFoldError(
            "LFM2 model does not expose model.layers."
        )
    result = tuple(layers)
    if not result:
        raise Lfm2DenseBudgetFoldError("LFM2 decoder has no layers.")
    return result


def _ordered_layer_types(model: Any) -> tuple[str, ...]:
    result = []
    for layer in _layers(model):
        contract = inspect_lfm2_layer(layer)
        if contract.kind not in {"full_attention", "short_conv"}:
            raise Lfm2DenseBudgetFoldError(
                f"Unsupported canonical LFM2 layer kind {contract.kind!r}."
            )
        result.append(
            "full_attention"
            if contract.kind == "full_attention"
            else "conv"
        )
    return tuple(result)


def _topology_fingerprint(model: Any) -> str:
    parameters = tuple(
        model.named_parameters(recurse=True, remove_duplicate=False)
    )
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
        raise Lfm2DenseBudgetFoldError(
            "Residual coordinate basis has the wrong shape."
        )
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


def _transport_square(mapping: Any, source: Any) -> Any:
    value = source.detach().to(device="cpu", dtype=_torch().float64)
    return mapping.fold_producer_weight(
        mapping.fold_consumer_weight(value)
    ).contiguous()


def _transport_depthwise_kernel(mapping: Any, source: Any) -> Any:
    torch = _torch()
    value = source.detach().to(device="cpu", dtype=torch.float64)
    if value.ndim != 2 or int(value.shape[0]) != int(mapping.source_width):
        raise Lfm2DenseBudgetFoldError(
            "Depthwise FIR kernel must be [source_hidden, taps]."
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
        raise Lfm2DenseBudgetFoldError(
            f"Folded value for {name} is non-finite."
        )
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
    vocabulary_margin_candidate_gate_enabled: bool = False
    vocabulary_margin_candidate_replay_mode: str = (
        CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE
    )
    vocabulary_margin_candidate_minimum_sign_agreement_fraction: float = 0.9
    vocabulary_margin_candidate_minimum_top_competitor_nonnegative_fraction: (
        float
    ) = 0.9
    vocabulary_margin_candidate_minimum_all_top_k_nonnegative_fraction: (
        float
    ) = 0.9
    vocabulary_margin_candidate_minimum_exact_top1_agreement_fraction: (
        float
    ) = 0.9
    closed_loop_adapter_config: Lfm2DynamicClosedLoopAdapterConfig = field(
        default_factory=Lfm2DynamicClosedLoopAdapterConfig
    )
    closed_loop_compiler_config: ClosedLoopCompilerConfig = field(
        default_factory=ClosedLoopCompilerConfig
    )
    macro_cell_attractor_gate_config: (
        Lfm2MacroCellAttractorGateConfig | None
    ) = None

    def __post_init__(self) -> None:
        mode = _text(self.residual_basis_mode, "residual_basis_mode")
        if mode not in {"weight_complete", "circuit_balanced"}:
            raise Lfm2DenseBudgetFoldError(
                "residual_basis_mode must be weight_complete or circuit_balanced."
            )
        object.__setattr__(self, "residual_basis_mode", mode)
        if not isinstance(self.run_closed_loop, bool):
            raise Lfm2DenseBudgetFoldError(
                "run_closed_loop must be boolean."
            )
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
        if (
            self.vocabulary_margin_observability
            and mode != "circuit_balanced"
        ):
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
        if (
            not isinstance(
                self.vocabulary_margin_candidate_replay_mode,
                str,
            )
            or self.vocabulary_margin_candidate_replay_mode
            not in {
                CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
                CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
            }
        ):
            raise Lfm2DenseBudgetFoldError(
                "vocabulary_margin_candidate_replay_mode must be "
                "'prefix_by_prefix_canonical' or 'single_pass_causal'."
            )
        for attribute in (
            "vocabulary_margin_candidate_minimum_sign_agreement_fraction",
            (
                "vocabulary_margin_candidate_minimum_"
                "top_competitor_nonnegative_fraction"
            ),
            (
                "vocabulary_margin_candidate_minimum_"
                "all_top_k_nonnegative_fraction"
            ),
            (
                "vocabulary_margin_candidate_minimum_"
                "exact_top1_agreement_fraction"
            ),
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
        if (
            self.macro_cell_attractor_gate_config is not None
            and not isinstance(
                self.macro_cell_attractor_gate_config,
                Lfm2MacroCellAttractorGateConfig,
            )
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
        validation_fraction=config.circuit_validation_fraction,
        split_seed=config.circuit_split_seed,
        max_rows_per_block=config.circuit_max_rows_per_block,
        minimum_heldout_retained_energy_fraction=(
            config.minimum_heldout_retained_energy_fraction
        ),
        circuit_graph=parent_graph,
        vocabulary_margin_observability=(
            config.vocabulary_margin_observability
        ),
        vocabulary_margin_top_k=config.vocabulary_margin_top_k,
        vocabulary_margin_floor_fraction=(
            config.vocabulary_margin_floor_fraction
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
            sample_last_tokens=int(
                manifest_capture.get("sample_last_tokens", 8)
            ),
            max_decode_steps=int(
                manifest_capture.get("max_decode_steps", 32)
            ),
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
    heldout_record_ids = frozenset(
        basis.vocabulary_margin_gate.heldout_record_ids
    )
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
        replay_capture_mode=(
            config.vocabulary_margin_candidate_replay_mode
        ),
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
    canonical_prefix_child_forwards = benchmark.get(
        "canonical_prefix_child_forwards"
    )
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
            config
            .vocabulary_margin_candidate_minimum_top_competitor_nonnegative_fraction
        ),
        minimum_all_top_k_nonnegative_fraction=(
            config
            .vocabulary_margin_candidate_minimum_all_top_k_nonnegative_fraction
        ),
        minimum_exact_top1_agreement_fraction=(
            config
            .vocabulary_margin_candidate_minimum_exact_top1_agreement_fraction
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
        "heldout_record_ids_sha256": canonical_sha256(
            sorted(heldout_record_ids)
        ),
        "heldout_used_for_basis_selection": False,
        "candidate_replay_capture_calls": 1,
        "candidate_replay_sha256": replay.replay_sha256,
        "candidate_replay_mode": (
            config.vocabulary_margin_candidate_replay_mode
        ),
        "single_pass_causal_authenticated": single_pass_authenticated,
        "causal_contract_sha256": causal_contract_sha256,
        "selected_layer_indices": list(replay.selected_layer_indices),
        "actual_child_forward_calls": actual_child_forwards,
        "canonical_prefix_child_forward_calls": (
            canonical_prefix_child_forwards
        ),
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
    source_types = _ordered_layer_types(parent_model)
    target_types = _ordered_layer_types(child_model)
    if source_types != target_types:
        raise Lfm2DenseBudgetFoldError(
            "Dense width folding requires identical source/target operator order."
        )
    source_config = parent_model.config
    target_config = child_model.config
    source_query_heads = int(source_config.num_attention_heads)
    source_key_value_heads = int(source_config.num_key_value_heads)
    target_query_heads = int(target_config.num_attention_heads)
    target_key_value_heads = int(target_config.num_key_value_heads)
    source_head_dim = int(
        getattr(source_config, "head_dim", 0)
        or next(
            layer.self_attn.head_dim
            for layer in source_layers
            if getattr(layer, "is_attention_layer", False)
        )
    )
    target_head_dim = int(
        getattr(target_config, "head_dim", 0)
        or next(
            layer.self_attn.head_dim
            for layer in target_layers
            if getattr(layer, "is_attention_layer", False)
        )
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
                        "Bridged target LM-head bias lacks transported source "
                        "values."
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
            source_contract = inspect_lfm2_layer(source_layer)
            target_contract = inspect_lfm2_layer(target_layer)
            if source_contract.kind != target_contract.kind:
                raise Lfm2DenseBudgetFoldError(
                    f"Layer {layer_index} changes its canonical operator kind."
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
                source_mlp.w1.weight.detach().to(
                    device="cpu", dtype=torch.float64
                )
            )
            folded_up = mapping.fold_consumer_weight(
                source_mlp.w3.weight.detach().to(
                    device="cpu", dtype=torch.float64
                )
            )
            folded_down = mlp_map.fold_consumer_weight(
                source_mlp.w2.weight.detach().to(
                    device="cpu", dtype=torch.float64
                )
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

            if source_contract.kind == "full_attention":
                source_attention = source_layer.self_attn
                target_attention = target_layer.self_attn
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
                            f"model.layers.{layer_index}.self_attn."
                            f"{projection}.weight"
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
                            f"model.layers.{layer_index}.self_attn."
                            f"{projection}.bias"
                        ),
                        written=written,
                    )
                source_output = source_attention.out_proj
                target_output = target_attention.out_proj
                context_folded = attention_maps.context.fold_consumer_weight(
                    source_output.weight.detach().to(
                        device="cpu", dtype=torch.float64
                    )
                )
                _copy_parameter(
                    target_output.weight,
                    mapping.fold_producer_weight(context_folded),
                    name=(
                        f"model.layers.{layer_index}.self_attn.out_proj.weight"
                    ),
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
                    "source_heads_dropped": 0,
                    "rope_pairs_preserved": True,
                }
            else:
                source_conv = source_layer.conv
                target_conv = target_layer.conv
                source_blocks = source_conv.in_proj.weight.chunk(3, dim=0)
                _copy_parameter(
                    target_conv.in_proj.weight,
                    torch.cat(
                        tuple(
                            _transport_square(mapping, block)
                            for block in source_blocks
                        ),
                        dim=0,
                    ),
                    name=f"model.layers.{layer_index}.conv.in_proj.weight",
                    written=written,
                )
                _copy_parameter(
                    target_conv.out_proj.weight,
                    _transport_square(mapping, source_conv.out_proj.weight),
                    name=f"model.layers.{layer_index}.conv.out_proj.weight",
                    written=written,
                )
                source_kernel = source_conv.conv.weight.detach()[:, 0, :]
                _copy_parameter(
                    target_conv.conv.weight,
                    _transport_depthwise_kernel(
                        mapping,
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
                                mapping.fold_producer_weight(
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
                            f"model.layers.{layer_index}.conv."
                            f"{projection_name}.bias"
                        ),
                        written=written,
                    )
                _copy_optional_bias(
                    source_conv.conv.bias,
                    target_conv.conv.bias,
                    (
                        None
                        if source_conv.conv.bias is None
                        else mapping.fold_producer_weight(
                            source_conv.conv.bias.detach()
                            .to(device="cpu", dtype=torch.float64)
                            .reshape(-1, 1)
                        ).reshape(-1)
                    ),
                    name=f"model.layers.{layer_index}.conv.conv.bias",
                    written=written,
                )
                operator_report = {
                    "kind": "dense_residual_short_conv_transport",
                    "input_projection": "B.T @ W_block @ B",
                    "output_projection": "B.T @ W @ B",
                    "depthwise_fir": "diag(B.T @ diag(kernel_tap) @ B)",
                    "source_channels_dropped": 0,
                }

            layer_reports.append(
                {
                    "layer_index": layer_index,
                    "kind": source_contract.kind,
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
        raise Lfm2DenseBudgetFoldError(
            "Dense fold left the compiler-zero child blank."
        )
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
            raise Lfm2DenseBudgetFoldError(
                "Cannot bind an inactive dense transaction."
            )
        candidate = _digest(
            candidate_state_fingerprint,
            "candidate_state_fingerprint",
        )
        if candidate == self.incumbent_state_fingerprint:
            raise Lfm2DenseBudgetFoldError(
                "Dense fold candidate is identical to the blank child."
            )
        if _topology_fingerprint(self.context.child_model) != self._topology_before:
            raise Lfm2DenseBudgetFoldError(
                "Dense fold changed Parameter topology."
            )
        self.plan = plan
        self.candidate_state_fingerprint = candidate

    def commit(self) -> Mapping[str, Any]:
        if not self.active:
            raise Lfm2DenseBudgetFoldError(
                "Dense fold transaction is inactive."
            )
        final = self.context.child_state_fingerprint
        if (
            final != self.candidate_state_fingerprint
            or _topology_fingerprint(self.context.child_model)
            != self._topology_before
        ):
            raise Lfm2DenseBudgetFoldError(
                "Dense fold candidate changed before commit."
            )
        self.active = False
        return {
            **self.plan.as_dict(),
            "action": "commit",
            "accepted": True,
            "exact": True,
            "candidate_state_fingerprint": self.candidate_state_fingerprint,
            "final_state_fingerprint": final,
            "parameter_topology_preserved": True,
        }

    def rollback(self) -> Mapping[str, Any]:
        if not self.active:
            raise Lfm2DenseBudgetFoldError(
                "Dense fold transaction is inactive."
            )
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
        return {
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


class Lfm2DenseBudgetModelOperator:
    """One authenticated, full-coverage dense-width model FoldOperator."""

    method_id = DENSE_BUDGET_MODEL_METHOD

    def __init__(
        self,
        selected_target: Lfm2DenseBudgetedSelectedTarget,
        trace_cache: Lfm2CircuitTraceCache,
        *,
        config: Lfm2DenseBudgetFoldConfig | None = None,
        tokenizer_bridge_artifact: StrictTokenizerBridgeArtifact | None = None,
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
        self.trace_cache = trace_cache.verify()
        self.config = config or Lfm2DenseBudgetFoldConfig()
        if not isinstance(self.config, Lfm2DenseBudgetFoldConfig):
            raise Lfm2DenseBudgetFoldError(
                "config must be Lfm2DenseBudgetFoldConfig."
            )
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
            if (
                self.config.run_closed_loop
                or self.config.vocabulary_margin_candidate_gate_enabled
                or self.config.macro_cell_attractor_gate_config is not None
            ):
                raise Lfm2DenseBudgetFoldError(
                    "Bridged dense folding requires sealed child-tokenizer "
                    "trace retokenization before closed-loop or candidate "
                    "replay can run."
                )
        elif tokenizer_bridge_artifact is not None:
            raise Lfm2DenseBudgetFoldError(
                "Unbridged dense target received an unused tokenizer bridge "
                "artifact."
            )
        self.tokenizer_bridge_artifact = tokenizer_bridge_artifact

    def _validate_action(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
    ) -> tuple[str, Any, ModelCircuitGraph]:
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
            parent_graph.fingerprint
            != self.selected_target.source_graph_fingerprint
            or parent_graph.tokenizer_fingerprint
            != self.selected_target.parent_tokenizer_semantics_sha256
            or set(source.node_ids)
            != {node.node_id for node in parent_graph.nodes}
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
        return source.parent_name, parent_model, parent_graph

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
    ) -> FoldTransaction:
        source_name, parent_model, parent_graph = self._validate_action(
            action,
            context,
        )
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
            )
            initial_report = _fold_dense_blank_from_parent(
                parent_model,
                context.child_model,
                basis,
                self.tokenizer_bridge_artifact,
            )
            initial_state = context.child_state_fingerprint
            closed_loop_report: Mapping[str, Any] | None = None
            if self.config.run_closed_loop:
                macro_cell_attractor_gate = (
                    None
                    if self.config.macro_cell_attractor_gate_config is None
                    else Lfm2MacroCellAttractorGate(
                        self.trace_cache,
                        config=(
                            self.config.macro_cell_attractor_gate_config
                        ),
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
                )
                closed_loop_report = ChildClosedLoopCompiler(
                    adapter,
                    self.config.closed_loop_compiler_config,
                ).run()
            candidate = context.child_state_fingerprint
            if fingerprint_model_state(parent_model) != parent_before:
                raise Lfm2DenseBudgetFoldError(
                    "Dense model fold mutated its parent."
                )
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
                "exact_target_unique_parameters": (
                    self.selected_target.exact_target_unique_parameters
                ),
                "residual_basis_mode": self.config.residual_basis_mode,
                "residual_basis_sha256": basis.map_sha256,
                "residual_basis": basis.as_dict(),
                "initial_structural_fold": initial_report,
                "initial_structural_state_fingerprint": initial_state,
                "closed_loop_enabled": self.config.run_closed_loop,
                "macro_cell_attractor_gate_enabled": (
                    self.config.macro_cell_attractor_gate_config is not None
                ),
                "closed_loop": (
                    None if closed_loop_report is None else dict(closed_loop_report)
                ),
                "vocabulary_margin_candidate_gate": (
                    vocabulary_margin_candidate_gate
                ),
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
    ) -> None:
        if not isinstance(trace_cache, Lfm2CircuitTraceCache):
            raise Lfm2DenseBudgetFoldError(
                "trace_cache must be Lfm2CircuitTraceCache."
            )
        self.trace_cache = trace_cache.verify()
        self.sealed_evidence = Lfm2SealedDenseBudgetEvidence(
            self.trace_cache
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
        self.trace_cache.verify()
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
        self.trace_cache.verify()
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
        self.trace_cache.verify()
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
            or parent_graph.fingerprint
            != self.selected_target.source_graph_fingerprint
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
            raise Lfm2DenseBudgetFoldError(
                "Dense fold initializer mutated the child."
            )
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
        metrics["vocabulary_margin_candidate_gate_enabled"] = (
            candidate_gate_enabled
        )
        metrics["vocabulary_margin_candidate_gate_passed"] = (
            candidate_gate_passed
        )
        metrics["vocabulary_margin_candidate_is_decision_authority"] = False
        if candidate_gate_enabled:
            evaluation = candidate_gate.get("evaluation")
            measured = (
                evaluation.get("metrics")
                if isinstance(evaluation, Mapping)
                else None
            )
            minimums = (
                evaluation.get("minimums")
                if isinstance(evaluation, Mapping)
                else None
            )
            checks = (
                evaluation.get("checks")
                if isinstance(evaluation, Mapping)
                else None
            )
            metric_names = (
                "candidate_vs_cached_sign_agreement_fraction",
                (
                    "candidate_top1_vs_top_competitor_"
                    "nonnegative_fraction"
                ),
                (
                    "candidate_top1_vs_all_top_k_"
                    "nonnegative_fraction"
                ),
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
                        metrics[
                            f"vocabulary_margin_candidate_{name}"
                        ] = float(value)
                        metrics[
                            f"vocabulary_margin_candidate_minimum_{name}"
                        ] = float(minimum)
                        metrics[
                            f"vocabulary_margin_candidate_check_{name}"
                        ] = check
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
                    metrics[
                        f"vocabulary_margin_candidate_{source_name}"
                    ] = value
            audit_sha256 = candidate_gate.get("audit_sha256")
            if isinstance(audit_sha256, str):
                metrics["vocabulary_margin_candidate_audit_sha256"] = (
                    audit_sha256
                )
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
