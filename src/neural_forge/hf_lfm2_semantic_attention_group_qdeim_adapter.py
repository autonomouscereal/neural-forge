"""Sealed-cache adapter for intact-GQA-group semantic attention Q-DEIM.

The parent trace cache already contains the actual pre-O attention context as
``attention.context``.  The collector records it after softmax/value
aggregation and authenticates it as part of the immutable safetensors batch.
Enhanced caches can additionally contain
``attention.context_from_probabilities`` or the exact cached
``attention.probabilities`` plus ``attention.value_history`` from which that
context can be reconstructed.

Only the captured ``attention.context`` route is admitted as functional FIT
evidence by this adapter.  If any required phase lacks it, compilation stops
with a typed evidence-deficit result.  A separately labelled nearest
cache-derived structural context is materialized when possible, but is never
silently substituted into the compiler.

The normalized-input decoder and residual chart are accepted only with the
existing layer-stack receipts that created them.  This module does not refit
either map, call a model, recapture a parent trace, train, prune, quantize, or
consult held-out tensor payloads.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_attention_group_qdeim import (
    Lfm2AttentionGroupQdeimInsufficientRankError,
    Lfm2CompiledAttentionGroupQdeim,
    compile_lfm2_attention_group_qdeim,
    lfm2_attention_group_qdeim_context_evidence_sha256,
    lfm2_attention_group_qdeim_source_state_sha256,
    lfm2_attention_group_qdeim_target_template_sha256,
)
from neural_forge.hf_lfm2_cached_endpoint_qk_residual_fold import (
    Lfm2DenseQkvoSourceInventory,
    seal_lfm2_dense_qkvo_source_inventory,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    CACHE_SCHEMA,
    SELECTIVE_AUTHENTICATED_SESSION_SCHEMA,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    lfm2_authenticated_tensor_sha256,
)


LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_ADAPTER_SCHEMA = (
    "neural-forge-lfm2-semantic-attention-group-qdeim-cache-adapter-v1"
)
LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_CAPABILITY_SCHEMA = (
    "neural-forge-lfm2-semantic-attention-group-qdeim-cache-capability-v1"
)
LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_EVIDENCE_DEFICIT_SCHEMA = (
    "neural-forge-lfm2-semantic-attention-group-qdeim-evidence-deficit-v1"
)

_RESIDUAL_CHART_SCHEMA = "neural-forge-lfm2-layer-stack-residual-chart-v1"
_NORMALIZED_DECODER_SCHEMA = "neural-forge-lfm2-normalized-operator-input-decoder-v1"
_ZERO_OPERATIONS = MappingProxyType(
    {
        "parent_model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
        "parent_model_generate_calls": 0,
        "parent_trace_recaptures": 0,
        "child_model_forward_calls": 0,
        "child_model_generate_calls": 0,
        "network_calls": 0,
        "huggingface_loader_calls": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
)


class Lfm2SemanticAttentionGroupQdeimAdapterError(RuntimeError):
    """Cache, map lineage, topology, or candidate authentication failed."""


@dataclass(frozen=True, slots=True)
class Lfm2SemanticAttentionGroupQdeimEvidenceDeficit:
    """Typed no-candidate result with an optional explicit structural proxy."""

    layer_index: int
    deficit_kind: str
    structural_context_rows: Any | None = field(repr=False, compare=False)
    structural_row_sample_weights: tuple[float, ...]
    report: Mapping[str, Any]

    @property
    def candidate_materialized(self) -> bool:
        return False

    def as_dict(self) -> dict[str, Any]:
        return _deep_thaw(self.report)


@dataclass(frozen=True, slots=True)
class Lfm2SemanticAttentionGroupQdeimCandidate:
    """Dense review candidate compiled only from captured FIT context rows."""

    layer_index: int
    compiled: Lfm2CompiledAttentionGroupQdeim = field(repr=False, compare=False)
    report: Mapping[str, Any]

    @property
    def candidate_materialized(self) -> bool:
        return True

    def state_dict(self) -> Mapping[str, Any]:
        return self.compiled.state_dict()

    def as_dict(self) -> dict[str, Any]:
        return _deep_thaw(self.report)


@dataclass(frozen=True, slots=True)
class _PhaseContextRoute:
    phase: str
    true_context_name: str
    true_context_available: bool
    nearest_method: str | None
    nearest_tensor_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _BatchContextPlan:
    batch_index: int
    phases: tuple[str, ...]
    routes: tuple[_PhaseContextRoute, ...]
    input_sha256: str
    record_sha256: str


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "PyTorch is required for semantic attention Q-DEIM adaptation."
        ) from exc
    return torch


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    if isinstance(value, list):
        return [_deep_thaw(item) for item in value]
    return value


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _positive_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            f"{label} must be positive and finite."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            f"{label} must be positive and finite."
        ) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            f"{label} must be positive and finite."
        )
    return result


def _tensor_sha256(value: Any, *, label: str) -> str:
    try:
        return lfm2_authenticated_tensor_sha256(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            f"{label} is not one hashable materialized tensor."
        ) from exc


def _finite_tensor(value: Any, *, label: str, ndim: int) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != ndim
        or any(int(size) <= 0 for size in value.shape)
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
        or value.requires_grad
        or not bool(torch.isfinite(value).all().item())
    ):
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            f"{label} must be one detached finite dense rank-{ndim} tensor."
        )
    return value.detach().to(device="cpu").contiguous().clone()


def _verify_self_sealed(
    value: Mapping[str, Any],
    *,
    seal_name: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            f"{label} must be a sealed mapping."
        )
    result = _deep_thaw(value)
    supplied = result.pop(seal_name, None)
    if supplied != canonical_sha256(result):
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            f"{label} self-seal is invalid."
        )
    return {**result, seal_name: supplied}


def _phase_sort_key(value: str) -> tuple[int, int, str]:
    if value == "prefill":
        return (0, -1, value)
    if value.startswith("decode."):
        suffix = value.removeprefix("decode.")
        if suffix.isdigit():
            return (1, int(suffix), value)
    return (2, 0, value)


def _fit_indices_and_weights(
    references: Sequence[Any],
    fit_batch_indices: Sequence[int],
    fit_batch_sample_weights: Mapping[int, float],
) -> tuple[tuple[int, ...], Mapping[int, float]]:
    if isinstance(fit_batch_indices, (str, bytes)):
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            "fit_batch_indices must be a sequence of cache batch indices."
        )
    indices = tuple(fit_batch_indices)
    if (
        not indices
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in indices
        )
        or len(set(indices)) != len(indices)
    ):
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            "fit_batch_indices must contain unique integer cache indices."
        )
    normalized = tuple(sorted(indices))
    available = {int(reference.index) for reference in references}
    if any(value not in available for value in normalized):
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            "A requested FIT batch is absent from the authenticated cache session."
        )
    if not isinstance(fit_batch_sample_weights, Mapping) or set(
        fit_batch_sample_weights
    ) != set(normalized):
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            "fit_batch_sample_weights must map exactly the FIT batch indices."
        )
    weights = MappingProxyType(
        {
            index: _positive_float(
                fit_batch_sample_weights[index],
                label=f"fit_batch_sample_weights[{index}]",
            )
            for index in normalized
        }
    )
    return normalized, weights


def _authenticated_session(
    session: Any,
) -> tuple[dict[str, Any], dict[str, Any], tuple[Any, ...]]:
    required = (
        "manifest",
        "cache_key",
        "evidence_sha256",
        "batch_references",
        "load_selected_tensors",
        "release_selected_tensors",
        "verify",
        "as_dict",
    )
    if any(not hasattr(session, name) for name in required):
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            "session is not a selective authenticated LFM2 cache session."
        )
    manifest = _verify_self_sealed(
        session.manifest,
        seal_name="manifest_sha256",
        label="parent cache manifest",
    )
    receipt = _verify_self_sealed(
        session.as_dict(),
        seal_name="session_sha256",
        label="selective cache session",
    )
    references = tuple(session.batch_references)
    cache_key = _digest(session.cache_key, label="cache key")
    evidence_sha = _digest(
        session.evidence_sha256,
        label="selective session SHA-256",
    )
    if (
        manifest.get("schema") != CACHE_SCHEMA
        or manifest.get("cache_key") != cache_key
        or receipt.get("schema") != SELECTIVE_AUTHENTICATED_SESSION_SCHEMA
        or receipt.get("cache_key") != cache_key
        or receipt.get("manifest_sha256") != manifest["manifest_sha256"]
        or receipt.get("session_sha256") != evidence_sha
        or receipt.get("batch_count") != len(references)
        or len({int(reference.index) for reference in references}) != len(references)
        or receipt.get("authentication_contract", {}).get("parent_forward_calls") != 0
        or receipt.get("authentication_contract", {}).get("training_steps") != 0
    ):
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            "Selective cache session is not bound to its immutable parent manifest."
        )
    return manifest, receipt, references


def _attention_contract(
    manifest: Mapping[str, Any],
    *,
    layer_index: int,
) -> Mapping[str, Any]:
    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or layer_index < 0
    ):
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            "layer_index must be a nonnegative integer."
        )
    layers = manifest.get("layer_topology", {}).get("layers")
    if not isinstance(layers, list) or layer_index >= len(layers):
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            "Cache manifest has no requested LFM2 layer."
        )
    entry = layers[layer_index]
    contract = entry.get("contract") if isinstance(entry, Mapping) else None
    operator = contract.get("operator") if isinstance(contract, Mapping) else None
    if (
        not isinstance(contract, Mapping)
        or contract.get("kind") != "full_attention"
        or not isinstance(operator, Mapping)
        or operator.get("operator") != "full_attention"
        or operator.get("query_heads") is None
        or operator.get("key_value_heads") is None
        or operator.get("head_dim") is None
    ):
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            "Requested cache layer is not an authenticated full-attention operator."
        )
    return MappingProxyType(_deep_thaw(operator))


def _context_plans(
    references: Sequence[Any],
    *,
    fit_indices: Sequence[int],
    layer_index: int,
) -> tuple[_BatchContextPlan, ...]:
    reference_by_index = {int(reference.index): reference for reference in references}
    marker = f".layer.{layer_index:03d}.attention.q"
    plans = []
    for batch_index in fit_indices:
        reference = reference_by_index[batch_index]
        inventory = set(reference.ordered_tensor_names)
        phases = tuple(
            sorted(
                {name[: -len(marker)] for name in inventory if name.endswith(marker)},
                key=_phase_sort_key,
            )
        )
        if not phases:
            raise Lfm2SemanticAttentionGroupQdeimAdapterError(
                f"FIT cache batch {batch_index} has no attention Q phases at layer "
                f"{layer_index}."
            )
        routes = []
        for phase in phases:
            prefix = f"{phase}.layer.{layer_index:03d}.attention."
            true_name = prefix + "context"
            if true_name in inventory:
                nearest_method = "captured_true_post_softmax_pre_o_context"
                nearest_names = (true_name,)
            elif prefix + "context_from_probabilities" in inventory:
                nearest_method = "cached_context_from_probabilities"
                nearest_names = (prefix + "context_from_probabilities",)
            elif {
                prefix + "probabilities",
                prefix + "value_history",
            }.issubset(inventory):
                nearest_method = "probabilities_matmul_repeated_value_history"
                nearest_names = (
                    prefix + "probabilities",
                    prefix + "value_history",
                )
            else:
                nearest_method = None
                nearest_names = ()
            routes.append(
                _PhaseContextRoute(
                    phase=phase,
                    true_context_name=true_name,
                    true_context_available=true_name in inventory,
                    nearest_method=nearest_method,
                    nearest_tensor_names=tuple(sorted(nearest_names)),
                )
            )
        plans.append(
            _BatchContextPlan(
                batch_index=batch_index,
                phases=phases,
                routes=tuple(routes),
                input_sha256=_digest(
                    reference.input_sha256,
                    label=f"cache batch {batch_index} input SHA-256",
                ),
                record_sha256=_digest(
                    reference.record_sha256,
                    label=f"cache batch {batch_index} record SHA-256",
                ),
            )
        )
    return tuple(plans)


def inspect_lfm2_semantic_attention_group_qdeim_cache(
    *,
    session: Any,
    layer_index: int,
    fit_batch_indices: Sequence[int],
) -> Mapping[str, Any]:
    """Inspect exact true/proxy context availability without loading payloads."""

    manifest, session_receipt, references = _authenticated_session(session)
    fit_indices = tuple(fit_batch_indices)
    if (
        not fit_indices
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in fit_indices
        )
        or len(set(fit_indices)) != len(fit_indices)
    ):
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            "fit_batch_indices must contain unique integer cache indices."
        )
    normalized_indices = tuple(sorted(fit_indices))
    available = {int(reference.index) for reference in references}
    if any(index not in available for index in normalized_indices):
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            "A requested FIT batch is absent from the authenticated session."
        )
    contract = _attention_contract(manifest, layer_index=layer_index)
    plans = _context_plans(
        references,
        fit_indices=normalized_indices,
        layer_index=layer_index,
    )
    missing_true = [
        route.true_context_name
        for plan in plans
        for route in plan.routes
        if not route.true_context_available
    ]
    missing_nearest = [
        f"{route.phase}.layer.{layer_index:03d}.attention"
        for plan in plans
        for route in plan.routes
        if route.nearest_method is None
    ]
    body = {
        "schema": LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_CAPABILITY_SCHEMA,
        "cache_key": manifest["cache_key"],
        "cache_manifest_sha256": manifest["manifest_sha256"],
        "cache_selective_session_sha256": session_receipt["session_sha256"],
        "layer_index": layer_index,
        "attention_operator_contract": dict(contract),
        "fit_batch_indices": list(normalized_indices),
        "heldout_batch_indices": sorted(available.difference(normalized_indices)),
        "phase_routes": [
            {
                "batch_index": plan.batch_index,
                "phase": route.phase,
                "true_context_name": route.true_context_name,
                "true_context_available": route.true_context_available,
                "nearest_cache_derived_method": route.nearest_method,
                "nearest_cache_derived_tensor_names": list(route.nearest_tensor_names),
            }
            for plan in plans
            for route in plan.routes
        ],
        "true_post_softmax_context_complete": not missing_true,
        "missing_true_post_softmax_context_tensor_names": missing_true,
        "nearest_structural_context_complete": not missing_nearest,
        "missing_nearest_structural_context_routes": missing_nearest,
        "captured_true_context_semantics": (
            "actual-pre-o-input-after-softmax-value-aggregation"
        ),
        "context_from_probabilities_semantics": (
            "separately-cached-reconstruction-not-silently-equivalent-to-captured-context"
        ),
        "tensor_payloads_loaded": 0,
        "parent_forward_calls": 0,
        "trace_recaptures": 0,
        "training_steps": 0,
    }
    return MappingProxyType({**body, "result_sha256": canonical_sha256(body)})


def _verified_chart_and_decoder(
    *,
    layer_index: int,
    residual_output_chart: Any,
    residual_chart_receipt: Mapping[str, Any],
    normalized_input_decoder: Any,
    normalized_input_decoder_receipt: Mapping[str, Any],
) -> tuple[Any, Mapping[str, Any], Any, Mapping[str, Any]]:
    chart = _finite_tensor(
        residual_output_chart,
        label="residual_output_chart",
        ndim=2,
    )
    decoder = _finite_tensor(
        normalized_input_decoder,
        label="normalized_input_decoder",
        ndim=2,
    )
    chart_receipt = _verify_self_sealed(
        residual_chart_receipt,
        seal_name="result_sha256",
        label="layer-stack residual chart receipt",
    )
    decoder_receipt = _verify_self_sealed(
        normalized_input_decoder_receipt,
        seal_name="decoder_sha256",
        label="normalized-input decoder receipt",
    )
    chart_sha = _tensor_sha256(chart, label="residual output chart")
    decoder_sha = _tensor_sha256(decoder, label="normalized input decoder")
    if (
        chart_receipt.get("schema") != _RESIDUAL_CHART_SCHEMA
        or chart_receipt.get("layer_index") != layer_index
        or chart_receipt.get("chart_sha256") != chart_sha
        or chart_receipt.get("fit_only") is not True
        or chart_receipt.get("validation_rows_used_for_fit") != 0
        or chart_receipt.get("numeric_metric_used_for_candidate_selection") is not False
        or chart_receipt.get("numeric_metric_used_for_publication_or_rollback")
        is not False
        or chart_receipt.get("training_steps") != 0
    ):
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            "Residual chart is not the authenticated FIT-only layer-stack chart."
        )
    if (
        decoder_receipt.get("schema") != _NORMALIZED_DECODER_SCHEMA
        or decoder_receipt.get("residual_chart_sha256")
        != chart_receipt["result_sha256"]
        or decoder_receipt.get("residual_chart_observed_sha256") != chart_sha
        or decoder_receipt.get("published_decoder_sha256") != decoder_sha
        or decoder_receipt.get("fit_rows_only_used_for_residual") is not True
        or decoder_receipt.get("heldout_rows_used_for_fit") != 0
        or decoder_receipt.get("gradients_used") is not False
        or decoder_receipt.get("training_steps") != 0
    ):
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            "Normalized decoder is not bound to the selected residual-chart receipt."
        )
    return (
        chart,
        MappingProxyType(chart_receipt),
        decoder,
        MappingProxyType(decoder_receipt),
    )


def _verified_source_inventory(
    *,
    source_query_weight: Any,
    source_key_weight: Any,
    source_value_weight: Any,
    source_out_weight: Any,
    source_inventory: Lfm2DenseQkvoSourceInventory,
) -> Lfm2DenseQkvoSourceInventory:
    if not isinstance(source_inventory, Lfm2DenseQkvoSourceInventory):
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            "source_inventory must be an authenticated dense Q/K/V/O inventory."
        )
    rebuilt = seal_lfm2_dense_qkvo_source_inventory(
        source_query_weight=source_query_weight,
        source_key_weight=source_key_weight,
        source_value_weight=source_value_weight,
        source_output_weight=source_out_weight,
    )
    if (
        source_inventory.verify() != source_inventory.inventory_sha256
        or rebuilt.inventory_sha256 != source_inventory.inventory_sha256
    ):
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            "Direct source Q/K/V/O tensors differ from their authenticated inventory."
        )
    return source_inventory


def _normalized_context(
    value: Any,
    *,
    label: str,
    source_query_heads: int,
    head_dim: int,
) -> Any:
    context = _finite_tensor(value, label=label, ndim=4)
    if tuple(context.shape[1::2]) != (source_query_heads, head_dim):
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            f"{label} does not match the authenticated parent head topology."
        )
    batch, _heads, queries, _head_dim = map(int, context.shape)
    return (
        context.permute(0, 2, 1, 3)
        .reshape(batch * queries, source_query_heads, head_dim)
        .contiguous()
    )


def _context_from_probabilities(
    probabilities: Any,
    value_history: Any,
    *,
    label: str,
    source_query_heads: int,
    source_key_value_heads: int,
    head_dim: int,
) -> Any:
    torch = _torch()
    probability = _finite_tensor(
        probabilities,
        label=f"{label} probabilities",
        ndim=4,
    ).to(torch.float64)
    values = _finite_tensor(
        value_history,
        label=f"{label} value history",
        ndim=4,
    ).to(torch.float64)
    if source_query_heads % source_key_value_heads:
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            "Authenticated parent attention contract has an invalid GQA ratio."
        )
    q_per_kv = source_query_heads // source_key_value_heads
    batch, query_heads, query_rows, key_rows = map(int, probability.shape)
    if (
        query_heads != source_query_heads
        or tuple(values.shape) != (batch, source_key_value_heads, key_rows, head_dim)
        or bool((probability < 0.0).any().item())
        or not bool(
            torch.allclose(
                probability.sum(dim=-1),
                torch.ones(
                    (batch, source_query_heads, query_rows),
                    dtype=torch.float64,
                ),
                rtol=2e-3,
                atol=2e-4,
            )
        )
    ):
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            f"{label} probability/value history is not one valid cached GQA route."
        )
    repeated_values = values.repeat_interleave(q_per_kv, dim=1)
    context = probability @ repeated_values
    return _normalized_context(
        context,
        label=f"{label} reconstructed context",
        source_query_heads=source_query_heads,
        head_dim=head_dim,
    )


def _payload_receipt(payload: Any, *, expected_names: Sequence[str]) -> dict[str, Any]:
    names = tuple(expected_names)
    if (
        tuple(payload.ordered_tensor_names) != names
        or payload.released
        or isinstance(payload.batch_index, bool)
        or not isinstance(payload.batch_index, int)
    ):
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            "Selective cache payload does not match its requested tensor set."
        )
    return {
        "batch_index": payload.batch_index,
        "input_sha256": _digest(
            payload.input_sha256,
            label="selected payload input SHA-256",
        ),
        "selection_sha256": _digest(
            payload.selection_sha256,
            label="selected payload selection SHA-256",
        ),
        "session_sha256": _digest(
            payload.session_sha256,
            label="selected payload session SHA-256",
        ),
        "ordered_tensor_names": list(names),
        "logical_bytes": int(payload.logical_bytes),
    }


def _load_context_rows(
    *,
    session: Any,
    plans: Sequence[_BatchContextPlan],
    batch_weights: Mapping[int, float],
    source_query_heads: int,
    source_key_value_heads: int,
    head_dim: int,
    route: str,
) -> tuple[Any, tuple[float, ...], Mapping[str, Any]]:
    torch = _torch()
    if route not in {"captured_true", "nearest_structural"}:
        raise AssertionError("invalid internal context route")
    all_rows = []
    all_weights: list[float] = []
    payload_receipts = []
    phase_receipts = []
    for plan in plans:
        selected_names = []
        for phase_route in plan.routes:
            if route == "captured_true":
                if not phase_route.true_context_available:
                    raise AssertionError("true context route is incomplete")
                selected_names.append(phase_route.true_context_name)
            else:
                if phase_route.nearest_method is None:
                    raise AssertionError("nearest structural route is incomplete")
                selected_names.extend(phase_route.nearest_tensor_names)
        names = tuple(sorted(set(selected_names)))
        payload = session.load_selected_tensors(plan.batch_index, names)
        try:
            receipt = _payload_receipt(payload, expected_names=names)
            if (
                receipt["batch_index"] != plan.batch_index
                or receipt["input_sha256"] != plan.input_sha256
                or receipt["session_sha256"] != session.evidence_sha256
            ):
                raise Lfm2SemanticAttentionGroupQdeimAdapterError(
                    "Selected context payload changed cache/session lineage."
                )
            tensors = payload.tensors
            batch_rows = []
            phase_count = len(plan.routes)
            for phase_route in plan.routes:
                method = (
                    "captured_true_post_softmax_pre_o_context"
                    if route == "captured_true"
                    else phase_route.nearest_method
                )
                if method == "probabilities_matmul_repeated_value_history":
                    probability_name, value_name = phase_route.nearest_tensor_names
                    rows = _context_from_probabilities(
                        tensors[probability_name],
                        tensors[value_name],
                        label=f"batch {plan.batch_index} phase {phase_route.phase}",
                        source_query_heads=source_query_heads,
                        source_key_value_heads=source_key_value_heads,
                        head_dim=head_dim,
                    )
                    source_names = (probability_name, value_name)
                else:
                    tensor_name = (
                        phase_route.true_context_name
                        if method == "captured_true_post_softmax_pre_o_context"
                        else phase_route.nearest_tensor_names[0]
                    )
                    rows = _normalized_context(
                        tensors[tensor_name],
                        label=(
                            f"batch {plan.batch_index} phase {phase_route.phase} "
                            f"{method}"
                        ),
                        source_query_heads=source_query_heads,
                        head_dim=head_dim,
                    )
                    source_names = (tensor_name,)
                batch_rows.append(rows)
                phase_weight_budget = batch_weights[plan.batch_index] / phase_count
                per_row_weight = phase_weight_budget / int(rows.shape[0])
                all_weights.extend(per_row_weight for _ in range(int(rows.shape[0])))
                phase_receipts.append(
                    {
                        "batch_index": plan.batch_index,
                        "phase": phase_route.phase,
                        "method": method,
                        "source_tensor_names": list(source_names),
                        "source_tensor_sha256": {
                            name: _tensor_sha256(
                                tensors[name],
                                label=f"cached context source {name}",
                            )
                            for name in source_names
                        },
                        "materialized_context_rows": int(rows.shape[0]),
                        "phase_weight_budget": phase_weight_budget,
                        "per_row_sample_weight": per_row_weight,
                        "materialized_row_weight_sum": (
                            per_row_weight * int(rows.shape[0])
                        ),
                        "materialized_context_sha256": _tensor_sha256(
                            rows,
                            label="materialized context rows",
                        ),
                    }
                )
            concatenated = torch.cat(batch_rows, dim=0).contiguous()
            all_rows.append(concatenated)
            payload_receipts.append(receipt)
        finally:
            session.release_selected_tensors(payload)
    materialized = torch.cat(all_rows, dim=0).contiguous()
    body = {
        "schema": "neural-forge-lfm2-semantic-attention-qdeim-context-bank-v1",
        "route": route,
        "fit_batch_indices": [plan.batch_index for plan in plans],
        "trajectory_balanced_batch_weights": {
            str(index): batch_weights[index] for index in sorted(batch_weights)
        },
        "phase_balancing_policy": (
            "batch_weight_divided_equally_across_phases_then_phase_weight_"
            "divided_equally_across_rows"
        ),
        "phase_weight_sums": [
            {
                "batch_index": value["batch_index"],
                "phase": value["phase"],
                "phase_weight_budget": value["phase_weight_budget"],
                "materialized_row_weight_sum": value["materialized_row_weight_sum"],
            }
            for value in phase_receipts
        ],
        "row_sample_weights": list(all_weights),
        "context_rows_sha256": _tensor_sha256(
            materialized,
            label="semantic attention Q-DEIM context bank",
        ),
        "context_row_count": int(materialized.shape[0]),
        "source_query_head_count": source_query_heads,
        "head_dim": head_dim,
        "selected_payloads": payload_receipts,
        "phase_materializations": phase_receipts,
        "heldout_tensor_payload_loads": 0,
        "parent_forward_calls": 0,
        "trace_recaptures": 0,
        "training_steps": 0,
    }
    return (
        materialized,
        tuple(all_weights),
        MappingProxyType({**body, "result_sha256": canonical_sha256(body)}),
    )


def _deficit(
    *,
    layer_index: int,
    deficit_kind: str,
    capability: Mapping[str, Any],
    source_inventory: Lfm2DenseQkvoSourceInventory,
    chart_receipt: Mapping[str, Any],
    decoder_receipt: Mapping[str, Any],
    structural_context_rows: Any | None,
    structural_row_weights: tuple[float, ...],
    structural_context_receipt: Mapping[str, Any] | None,
    true_context_receipt: Mapping[str, Any] | None = None,
    compiler_rank_deficit: Mapping[str, Any] | None = None,
) -> Lfm2SemanticAttentionGroupQdeimEvidenceDeficit:
    structural_sha = (
        None
        if structural_context_rows is None
        else _tensor_sha256(
            structural_context_rows,
            label="nearest structural context rows",
        )
    )
    body = {
        "schema": LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_EVIDENCE_DEFICIT_SCHEMA,
        "status": "no-candidate-materialized",
        "deficit_kind": deficit_kind,
        "layer_index": layer_index,
        "cache_capability_sha256": capability["result_sha256"],
        "source_inventory_sha256": source_inventory.inventory_sha256,
        "residual_chart_receipt_sha256": chart_receipt["result_sha256"],
        "normalized_input_decoder_receipt_sha256": decoder_receipt["decoder_sha256"],
        "missing_true_post_softmax_context_tensor_names": capability[
            "missing_true_post_softmax_context_tensor_names"
        ],
        "nearest_structural_context_materialized": (
            structural_context_rows is not None
        ),
        "nearest_structural_context_sha256": structural_sha,
        "nearest_structural_context_row_count": (
            0
            if structural_context_rows is None
            else int(structural_context_rows.shape[0])
        ),
        "nearest_structural_row_sample_weights": list(structural_row_weights),
        "nearest_structural_context_receipt": (
            None
            if structural_context_receipt is None
            else _deep_thaw(structural_context_receipt)
        ),
        "authenticated_true_context_receipt": (
            None if true_context_receipt is None else _deep_thaw(true_context_receipt)
        ),
        "nearest_structural_context_is_not_silently_admitted_as_true_fit_context": True,
        "compiler_rank_deficit": (
            None if compiler_rank_deficit is None else _deep_thaw(compiler_rank_deficit)
        ),
        "dense_qkvo_candidate_materialized": False,
        "numeric_diagnostics_have_semantic_decision_authority": False,
        "binary_manual_threshold_has_iteration_veto_authority": False,
        "whole_decoded_behavior_is_primary_intelligence_evidence": True,
        "binary_manual_acceptance_gate_required": False,
        "intelligence_retention_target_is_objective_not_numeric_cutoff": True,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    report = MappingProxyType({**body, "result_sha256": canonical_sha256(body)})
    return Lfm2SemanticAttentionGroupQdeimEvidenceDeficit(
        layer_index=layer_index,
        deficit_kind=deficit_kind,
        structural_context_rows=(
            None
            if structural_context_rows is None
            else structural_context_rows.detach().cpu().contiguous().clone()
        ),
        structural_row_sample_weights=structural_row_weights,
        report=report,
    )


def compile_lfm2_semantic_attention_group_qdeim_from_cache(
    *,
    session: Any,
    layer_index: int,
    fit_batch_indices: Sequence[int],
    fit_batch_sample_weights: Mapping[int, float],
    source_query_weight: Any,
    source_key_weight: Any,
    source_value_weight: Any,
    source_out_weight: Any,
    source_inventory: Lfm2DenseQkvoSourceInventory,
    target_query_weight_template: Any,
    target_key_weight_template: Any,
    target_value_weight_template: Any,
    target_out_weight_template: Any,
    normalized_input_decoder: Any,
    normalized_input_decoder_receipt: Mapping[str, Any],
    residual_output_chart: Any,
    residual_chart_receipt: Mapping[str, Any],
    pod_rank_relative_tolerance: float | None = None,
) -> (
    Lfm2SemanticAttentionGroupQdeimCandidate
    | Lfm2SemanticAttentionGroupQdeimEvidenceDeficit
):
    """Compile a dense Q/K/V/O review candidate from true cached FIT context."""

    manifest, session_receipt, references = _authenticated_session(session)
    fit_indices, batch_weights = _fit_indices_and_weights(
        references,
        fit_batch_indices,
        fit_batch_sample_weights,
    )
    contract = _attention_contract(manifest, layer_index=layer_index)
    source_q_heads = int(contract["query_heads"])
    source_kv_heads = int(contract["key_value_heads"])
    head_dim = int(contract["head_dim"])
    chart, chart_receipt, decoder, decoder_receipt = _verified_chart_and_decoder(
        layer_index=layer_index,
        residual_output_chart=residual_output_chart,
        residual_chart_receipt=residual_chart_receipt,
        normalized_input_decoder=normalized_input_decoder,
        normalized_input_decoder_receipt=normalized_input_decoder_receipt,
    )
    inventory = _verified_source_inventory(
        source_query_weight=source_query_weight,
        source_key_weight=source_key_weight,
        source_value_weight=source_value_weight,
        source_out_weight=source_out_weight,
        source_inventory=source_inventory,
    )
    capability = inspect_lfm2_semantic_attention_group_qdeim_cache(
        session=session,
        layer_index=layer_index,
        fit_batch_indices=fit_indices,
    )
    plans = _context_plans(
        references,
        fit_indices=fit_indices,
        layer_index=layer_index,
    )

    if capability["true_post_softmax_context_complete"] is not True:
        structural_rows = None
        structural_weights: tuple[float, ...] = ()
        structural_receipt = None
        if capability["nearest_structural_context_complete"] is True:
            structural_rows, structural_weights, structural_receipt = (
                _load_context_rows(
                    session=session,
                    plans=plans,
                    batch_weights=batch_weights,
                    source_query_heads=source_q_heads,
                    source_key_value_heads=source_kv_heads,
                    head_dim=head_dim,
                    route="nearest_structural",
                )
            )
        session.verify()
        if session.evidence_sha256 != session_receipt["session_sha256"]:
            raise Lfm2SemanticAttentionGroupQdeimAdapterError(
                "Cache session lineage changed while materializing the evidence deficit."
            )
        return _deficit(
            layer_index=layer_index,
            deficit_kind="missing-captured-true-post-softmax-context",
            capability=capability,
            source_inventory=inventory,
            chart_receipt=chart_receipt,
            decoder_receipt=decoder_receipt,
            structural_context_rows=structural_rows,
            structural_row_weights=structural_weights,
            structural_context_receipt=structural_receipt,
        )

    context_rows, row_weights, context_receipt = _load_context_rows(
        session=session,
        plans=plans,
        batch_weights=batch_weights,
        source_query_heads=source_q_heads,
        source_key_value_heads=source_kv_heads,
        head_dim=head_dim,
        route="captured_true",
    )
    session.verify()
    if session.evidence_sha256 != session_receipt["session_sha256"]:
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            "Cache session lineage changed during true-context consumption."
        )

    source_state_sha = lfm2_attention_group_qdeim_source_state_sha256(
        source_query_weight,
        source_key_weight,
        source_value_weight,
        source_out_weight,
    )
    target_template_sha = lfm2_attention_group_qdeim_target_template_sha256(
        target_query_weight_template,
        target_key_weight_template,
        target_value_weight_template,
        target_out_weight_template,
    )
    context_evidence_sha = lfm2_attention_group_qdeim_context_evidence_sha256(
        context_rows,
        row_weights,
    )
    target_q_rows = int(target_query_weight_template.shape[0])
    target_kv_rows = int(target_key_weight_template.shape[0])
    if target_q_rows % head_dim or target_kv_rows % head_dim:
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            "Target templates do not preserve the authenticated parent head dimension."
        )
    target_q_heads = target_q_rows // head_dim
    target_kv_heads = target_kv_rows // head_dim
    try:
        compiled = compile_lfm2_attention_group_qdeim(
            source_query_weight=source_query_weight,
            source_key_weight=source_key_weight,
            source_value_weight=source_value_weight,
            source_out_weight=source_out_weight,
            expected_source_state_sha256=source_state_sha,
            target_query_weight_template=target_query_weight_template,
            target_key_weight_template=target_key_weight_template,
            target_value_weight_template=target_value_weight_template,
            target_out_weight_template=target_out_weight_template,
            expected_target_template_sha256=target_template_sha,
            source_query_head_count=source_q_heads,
            source_key_value_head_count=source_kv_heads,
            target_query_head_count=target_q_heads,
            target_key_value_head_count=target_kv_heads,
            normalized_input_decoder=decoder,
            expected_normalized_input_decoder_sha256=_tensor_sha256(
                decoder,
                label="normalized input decoder",
            ),
            residual_output_chart=chart,
            expected_residual_output_chart_sha256=_tensor_sha256(
                chart,
                label="residual output chart",
            ),
            parent_post_softmax_context_rows=context_rows,
            row_sample_weights=row_weights,
            expected_context_evidence_sha256=context_evidence_sha,
            pod_rank_relative_tolerance=pod_rank_relative_tolerance,
        )
    except Lfm2AttentionGroupQdeimInsufficientRankError as exc:
        return _deficit(
            layer_index=layer_index,
            deficit_kind="insufficient-authenticated-true-context-group-rank",
            capability=capability,
            source_inventory=inventory,
            chart_receipt=chart_receipt,
            decoder_receipt=decoder_receipt,
            structural_context_rows=None,
            structural_row_weights=(),
            structural_context_receipt=None,
            true_context_receipt=context_receipt,
            compiler_rank_deficit=exc.report,
        )
    if not isinstance(compiled, Lfm2CompiledAttentionGroupQdeim):
        raise Lfm2SemanticAttentionGroupQdeimAdapterError(
            "Standalone intact-group compiler returned no dense candidate."
        )
    state = compiled.state_dict()
    body = {
        "schema": LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_ADAPTER_SCHEMA,
        "status": "dense-candidate-materialized-for-whole-response-review",
        "layer_index": layer_index,
        "cache_key": manifest["cache_key"],
        "cache_parent_fingerprint": manifest["parent"]["identity_sha256"],
        "cache_manifest_sha256": manifest["manifest_sha256"],
        "cache_selective_session_sha256": session_receipt["session_sha256"],
        "cache_capability_sha256": capability["result_sha256"],
        "source_attention_contract": dict(contract),
        "source_inventory_sha256": inventory.inventory_sha256,
        "source_state_sha256": source_state_sha,
        "target_template_sha256": target_template_sha,
        "residual_chart_receipt_sha256": chart_receipt["result_sha256"],
        "residual_chart_sha256": _tensor_sha256(
            chart,
            label="residual output chart",
        ),
        "normalized_input_decoder_receipt_sha256": decoder_receipt["decoder_sha256"],
        "normalized_input_decoder_sha256": _tensor_sha256(
            decoder,
            label="normalized input decoder",
        ),
        "fit_batch_indices": list(fit_indices),
        "heldout_batch_indices": capability["heldout_batch_indices"],
        "heldout_tensor_payload_loads": 0,
        "captured_true_post_softmax_context_only": True,
        "context_bank_receipt": _deep_thaw(context_receipt),
        "context_bank_receipt_sha256": context_receipt["result_sha256"],
        "standalone_context_evidence_sha256": context_evidence_sha,
        "standalone_compiler_result_sha256": compiled.report["result_sha256"],
        "selected_source_key_value_groups": list(
            compiled.selected_source_key_value_groups
        ),
        "selected_source_query_heads": list(compiled.selected_source_query_heads),
        "candidate_tensor_sha256": {
            name: _tensor_sha256(value, label=f"candidate {name}")
            for name, value in state.items()
        },
        "candidate_state_sha256": compiled.report["materialization"][
            "candidate_state_sha256"
        ],
        "candidate_dense_bfloat16_standalone": True,
        "qkv_producer_averaging_performed": False,
        "dense_all_parent_output_columns_used": True,
        "numeric_diagnostics_are_advisory": True,
        "numeric_metrics_have_semantic_decision_authority": False,
        "binary_manual_threshold_has_iteration_veto_authority": False,
        "binary_manual_acceptance_gate_required": False,
        "intelligence_retention_target_is_objective_not_numeric_cutoff": True,
        "whole_decoded_behavior_is_primary_intelligence_evidence": True,
        "automatic_semantic_acceptance_or_rejection_permitted": False,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    report = MappingProxyType({**body, "result_sha256": canonical_sha256(body)})
    return Lfm2SemanticAttentionGroupQdeimCandidate(
        layer_index=layer_index,
        compiled=compiled,
        report=report,
    )


__all__ = [
    "LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_ADAPTER_SCHEMA",
    "LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_CAPABILITY_SCHEMA",
    "LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_EVIDENCE_DEFICIT_SCHEMA",
    "Lfm2SemanticAttentionGroupQdeimAdapterError",
    "Lfm2SemanticAttentionGroupQdeimCandidate",
    "Lfm2SemanticAttentionGroupQdeimEvidenceDeficit",
    "compile_lfm2_semantic_attention_group_qdeim_from_cache",
    "inspect_lfm2_semantic_attention_group_qdeim_cache",
]
