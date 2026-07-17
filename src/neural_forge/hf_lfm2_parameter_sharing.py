"""Activation-aligned, no-training partial FFN sharing for LFM2.

This primitive keeps every decoder layer and every attention/short-convolution
operator in place.  It compresses physical storage only: selected disjoint
layer pairs execute their original full-width SwiGLU sublayers while sharing
one aligned-and-merged ``w1``/``w3``/``w2`` parameter bundle.

Pair selection uses held-out, already-cached post-SwiGLU rows.  Layer
similarity is the mean of permutation-invariant linear CKA for the post-SwiGLU
rows and for their full down-projected outputs.  Only after disjoint pairs are
selected is every source neuron aligned to an anchor neuron with one shared
permutation.  The same permutation is applied to the source ``w1`` and ``w3``
rows and ``w2`` columns by :mod:`neural_forge.hf_parameter_sharing`.

No forward capture, optimizer, backward pass, or gradient update exists here.
Installation is one fail-closed whole-model transaction.  Rollback restores
the exact original Parameter objects.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

from neural_forge.hf_lfm2_canonical import (
    Lfm2CanonicalError,
    Lfm2LayerContract,
    inspect_lfm2_layer,
)
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_parameter_sharing import (
    DEFAULT_CORRELATION_CHUNK_SIZE,
    DEFAULT_EXACT_ASSIGNMENT_LIMIT,
    MLPMergeResult,
    MLPModuleLayout,
    MLPTensorBundle,
    align_and_merge_mlp_bundles,
    extract_mlp_bundle,
    tie_mlp_modules,
)


EVIDENCE_SCHEMA = "neural-forge-lfm2-post-swiglu-sharing-evidence-v1"
PLAN_SCHEMA = "neural-forge-lfm2-activation-aligned-ffn-sharing-plan-v1"
REPORT_SCHEMA = "neural-forge-lfm2-activation-aligned-ffn-sharing-report-v1"
SELECTION_METHOD = "heldout-post-swiglu-and-down-output-linear-cka"
ALIGNMENT_METHOD = "full-width-post-swiglu-correlation-permutation"
_SHA256_LENGTH = 64
_ROLES = ("w1", "w3", "w2")
_BUNDLE_ROLES = (
    "input_weight",
    "gate_weight",
    "up_weight",
    "down_weight",
    "input_bias",
    "gate_bias",
    "up_bias",
    "down_bias",
)
_LAYOUT = MLPModuleLayout.gated(
    "w1",
    "w3",
    "w2",
    gate_bias=False,
    up_bias=False,
    down_bias=False,
)
_TRANSACTION_LOCK = threading.RLock()


class Lfm2ParameterSharingError(RuntimeError):
    """Base error for LFM2 activation-aligned parameter sharing."""


class Lfm2ParameterSharingEvidenceError(Lfm2ParameterSharingError):
    """Cached activation evidence is absent, stale, incomplete, or modified."""


class Lfm2ParameterSharingTopologyError(Lfm2ParameterSharingError):
    """The model is not a supported, independent-FFN Transformers LFM2."""


class Lfm2ParameterSharingTransactionError(Lfm2ParameterSharingError):
    """Atomic installation or exact rollback could not be authenticated."""


@dataclass(frozen=True, slots=True)
class Lfm2ActivationEvidence:
    """Hash-bound cached post-SwiGLU rows for one exact loaded parent."""

    schema: str
    cache_key: str
    heldout_corpus_sha256: str
    parent_identity_sha256: str
    layer_rows: tuple[Any, ...]
    layer_row_sha256: tuple[str, ...]
    row_count: int
    intermediate_size: int
    post_swiglu_verified: bool
    evidence_sha256: str

    def rows_for_layer(self, layer_index: int) -> Any:
        """Return the sealed rows for one decoder layer."""

        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or layer_index < 0
            or layer_index >= len(self.layer_rows)
        ):
            raise Lfm2ParameterSharingEvidenceError(
                "layer_index is outside the sealed activation evidence."
            )
        return self.layer_rows[layer_index]

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe evidence metadata without embedding tensor values."""

        return {
            "schema": self.schema,
            "cache_key": self.cache_key,
            "heldout_corpus_sha256": self.heldout_corpus_sha256,
            "parent_identity_sha256": self.parent_identity_sha256,
            "layer_count": len(self.layer_rows),
            "layer_row_sha256": list(self.layer_row_sha256),
            "row_count": self.row_count,
            "intermediate_size": self.intermediate_size,
            "post_swiglu_verified": self.post_swiglu_verified,
            "evidence_sha256": self.evidence_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2PairSimilarity:
    """Held-out similarity used before any parameters are changed."""

    anchor_layer: int
    source_layer: int
    activation_similarity: float
    output_similarity: float
    combined_similarity: float

    @property
    def layers(self) -> tuple[int, int]:
        return (self.anchor_layer, self.source_layer)

    def as_dict(self) -> dict[str, Any]:
        return {
            "anchor_layer": self.anchor_layer,
            "source_layer": self.source_layer,
            "activation_linear_cka": self.activation_similarity,
            "down_output_linear_cka": self.output_similarity,
            "combined_similarity": self.combined_similarity,
        }


@dataclass(frozen=True, slots=True)
class Lfm2FfnPairPlan:
    """One complete full-width pair merge and its assignment evidence."""

    similarity: Lfm2PairSimilarity
    merge_result: MLPMergeResult
    aligned_neuron_count: int
    parameters_saved: int

    @property
    def layers(self) -> tuple[int, int]:
        return self.similarity.layers

    @property
    def permutation(self) -> Any:
        """Map anchor-neuron index to the selected source-neuron index."""

        return self.merge_result.permutations[1]

    @property
    def merged_bundle(self) -> MLPTensorBundle:
        return self.merge_result.merged_bundle

    def as_dict(self) -> dict[str, Any]:
        assignment = self.merge_result.assignments[1].diagnostics
        return {
            **self.similarity.as_dict(),
            "alignment_method": ALIGNMENT_METHOD,
            "aligned_neuron_count": self.aligned_neuron_count,
            "parameters_saved": self.parameters_saved,
            "normalized_merge_weights": [
                float(value) for value in self.merge_result.normalized_weights.tolist()
            ],
            "assignment": {
                "solver": assignment.solver,
                "complete": assignment.complete,
                "collision_count": assignment.collision_count,
                "deterministic": assignment.deterministic,
                "mean_assigned_correlation": (assignment.mean_assigned_correlation),
                "minimum_assigned_correlation": (
                    assignment.minimum_assigned_correlation
                ),
                "maximum_assigned_correlation": (
                    assignment.maximum_assigned_correlation
                ),
            },
        }


@dataclass(frozen=True, slots=True)
class Lfm2ParameterSharingPlan:
    """Immutable model-level plan built entirely before installation."""

    schema: str
    parent_identity_sha256: str
    evidence_sha256: str
    requested_pair_count: int
    pair_plans: tuple[Lfm2FfnPairPlan, ...]
    pair_score_weights: tuple[float, float]
    merge_weights: tuple[float, float]
    selection_solver: str
    logical_parameters: int
    unique_parameters_before: int
    projected_unique_parameters_after: int
    projected_parameters_saved: int
    layer_count: int
    hidden_size: int
    intermediate_size: int
    plan_sha256: str = ""
    training_steps: int = 0
    gradients_used: bool = False

    @property
    def selected_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(pair.layers for pair in self.pair_plans)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "parent_identity_sha256": self.parent_identity_sha256,
            "evidence_sha256": self.evidence_sha256,
            "requested_pair_count": self.requested_pair_count,
            "selected_pairs": [list(pair) for pair in self.selected_pairs],
            "pair_score_weights": list(self.pair_score_weights),
            "merge_weights": list(self.merge_weights),
            "selection_method": SELECTION_METHOD,
            "selection_solver": self.selection_solver,
            "alignment_method": ALIGNMENT_METHOD,
            "pairs": [pair.as_dict() for pair in self.pair_plans],
            "logical_parameters": self.logical_parameters,
            "unique_parameters_before": self.unique_parameters_before,
            "projected_unique_parameters_after": (
                self.projected_unique_parameters_after
            ),
            "projected_parameters_saved": self.projected_parameters_saved,
            "layer_count": self.layer_count,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "preserves_execution_depth": True,
            "preserves_attention_and_recurrent_operators": True,
            "training_steps": self.training_steps,
            "gradients_used": self.gradients_used,
            "plan_sha256": self.plan_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2ParameterSharingReport:
    """Authenticated logical-versus-physical counts after installation."""

    schema: str
    parent_identity_before: str
    model_identity_after: str
    evidence_sha256: str
    selected_pairs: tuple[tuple[int, int], ...]
    logical_parameters: int
    unique_parameters_before: int
    unique_parameters_after: int
    parameters_saved: int
    tied_parameter_objects: int
    aligned_neurons_per_pair: int
    topology_preserved: bool
    recurrent_state_preserved: bool
    only_selected_ffn_parameters_replaced: bool
    training_steps: int = 0
    gradients_used: bool = False

    @property
    def compression_fraction(self) -> float:
        if self.unique_parameters_before <= 0:
            return 0.0
        return self.parameters_saved / self.unique_parameters_before

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "parent_identity_before": self.parent_identity_before,
            "model_identity_after": self.model_identity_after,
            "evidence_sha256": self.evidence_sha256,
            "selected_pairs": [list(pair) for pair in self.selected_pairs],
            "logical_parameters": self.logical_parameters,
            "unique_parameters_before": self.unique_parameters_before,
            "unique_parameters_after": self.unique_parameters_after,
            "parameters_saved": self.parameters_saved,
            "compression_fraction": self.compression_fraction,
            "tied_parameter_objects": self.tied_parameter_objects,
            "aligned_neurons_per_pair": self.aligned_neurons_per_pair,
            "topology_preserved": self.topology_preserved,
            "recurrent_state_preserved": self.recurrent_state_preserved,
            "only_selected_ffn_parameters_replaced": (
                self.only_selected_ffn_parameters_replaced
            ),
            "execution_depth_preserved": True,
            "training_steps": self.training_steps,
            "gradients_used": self.gradients_used,
        }


@dataclass(frozen=True, slots=True)
class _ParameterBinding:
    owner: Any
    name: str
    parameter: Any


class InstalledLfm2ParameterSharing:
    """Live installation handle with authenticated exact-object rollback."""

    def __init__(
        self,
        *,
        model: Any,
        report: Lfm2ParameterSharingReport,
        original_bindings: tuple[_ParameterBinding, ...],
        installed_parameters: tuple[Any, ...],
        original_buffer_snapshot: Mapping[str, tuple[Any, Any]],
    ) -> None:
        self.model = model
        self.report = report
        self._original_bindings = original_bindings
        self._installed_parameters = installed_parameters
        self._original_buffer_snapshot = dict(original_buffer_snapshot)
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    def rollback(self) -> bool:
        """Restore the exact original Parameters; return false if already done."""

        torch = _require_torch()
        with _TRANSACTION_LOCK, torch.no_grad():
            if not self._active:
                return False
            for binding, installed in zip(
                self._original_bindings,
                self._installed_parameters,
            ):
                current = binding.owner._parameters.get(binding.name)
                if current is not installed:
                    raise Lfm2ParameterSharingTransactionError(
                        "A tied FFN Parameter changed after installation; refusing "
                        "to clobber external state during rollback."
                    )
            for binding in self._original_bindings:
                binding.owner._parameters[binding.name] = binding.parameter
            self._active = False
            if any(
                binding.owner._parameters.get(binding.name) is not binding.parameter
                for binding in self._original_bindings
            ):
                raise Lfm2ParameterSharingTransactionError(
                    "Exact Parameter-object rollback verification failed."
                )
            _verify_buffers_unchanged(
                torch,
                self.model,
                self._original_buffer_snapshot,
            )
            restored = _parent_identity(self.model)
            if restored != self.report.parent_identity_before:
                raise Lfm2ParameterSharingTransactionError(
                    "Rollback restored this surgery's Parameter objects, but the "
                    "whole model no longer matches the original parent identity."
                )
            return True

    def __enter__(self) -> InstalledLfm2ParameterSharing:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.rollback()


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError("PyTorch is required for LFM2 parameter sharing.") from exc
    return torch


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Lfm2ParameterSharingEvidenceError(
            "Evidence metadata must be canonical JSON."
        ) from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _tensor_sha256(torch: Any, value: Any) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-sharing-row-v1\0")
    digest.update(str(tensor.dtype).replace("torch.", "").encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json([int(size) for size in tensor.shape]))
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _plan_tensor_record(torch: Any, value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise Lfm2ParameterSharingTransactionError(
            f"Sealed plan tensor {name!r} is missing or unsupported."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-sharing-plan-tensor-v1\0")
    digest.update(name.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tensor.dtype).replace("torch.", "").encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json([int(size) for size in tensor.shape]))
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "shape": [int(size) for size in tensor.shape],
        "sha256": digest.hexdigest(),
    }


def _plan_seal_body(
    torch: Any,
    plan: Lfm2ParameterSharingPlan,
) -> dict[str, Any]:
    metadata = plan.as_dict()
    metadata.pop("plan_sha256", None)
    sealed_pairs = []
    for pair_index, pair in enumerate(plan.pair_plans):
        permutations = [
            _plan_tensor_record(
                torch,
                permutation,
                f"pair.{pair_index}.permutation.{permutation_index}",
            )
            for permutation_index, permutation in enumerate(
                pair.merge_result.permutations
            )
        ]
        merged_tensors = {}
        for role in _BUNDLE_ROLES:
            value = getattr(pair.merged_bundle, role)
            merged_tensors[role] = (
                None
                if value is None
                else _plan_tensor_record(
                    torch,
                    value,
                    f"pair.{pair_index}.merged_bundle.{role}",
                )
            )
        sealed_pairs.append(
            {
                "layers": list(pair.layers),
                "permutations": permutations,
                "merged_tensors": merged_tensors,
            }
        )
    return {
        "schema": "neural-forge-lfm2-sharing-plan-seal-v1",
        "metadata": metadata,
        "selected_pairs": [list(pair) for pair in plan.selected_pairs],
        "payloads": sealed_pairs,
    }


def _seal_plan(
    torch: Any,
    plan: Lfm2ParameterSharingPlan,
) -> Lfm2ParameterSharingPlan:
    return replace(
        plan,
        plan_sha256=_json_sha256(_plan_seal_body(torch, plan)),
    )


def _validate_plan_seal(torch: Any, plan: Lfm2ParameterSharingPlan) -> None:
    expected = _json_sha256(_plan_seal_body(torch, plan))
    if not _valid_sha256(plan.plan_sha256) or plan.plan_sha256 != expected:
        raise Lfm2ParameterSharingTransactionError(
            "The sealed sharing plan changed after planning."
        )


def _parent_identity(model: Any) -> str:
    try:
        return fingerprint_lfm2_parent(model).identity_sha256
    except Exception as exc:
        raise Lfm2ParameterSharingTopologyError(
            "Could not authenticate the exact loaded LFM2 parent."
        ) from exc


def _locate_layers(model: Any, torch: Any) -> tuple[Any, ...]:
    candidates: list[Any] = []
    direct = getattr(model, "layers", None)
    if direct is not None:
        candidates.append(direct)
    inner = getattr(model, "model", None)
    nested = getattr(inner, "layers", None)
    if nested is not None and all(nested is not value for value in candidates):
        candidates.append(nested)
    if len(candidates) != 1:
        raise Lfm2ParameterSharingTopologyError(
            "The model must expose exactly one decoder layer sequence at "
            "model.layers or model.model.layers."
        )
    try:
        layers = tuple(candidates[0])
    except TypeError as exc:
        raise Lfm2ParameterSharingTopologyError(
            "The LFM2 decoder layers are not iterable."
        ) from exc
    if len(layers) < 2 or any(
        not isinstance(layer, torch.nn.Module) for layer in layers
    ):
        raise Lfm2ParameterSharingTopologyError(
            "LFM2 parameter sharing requires at least two decoder modules."
        )
    return layers


def _named_parameters_no_dedup(model: Any) -> tuple[tuple[str, Any], ...]:
    try:
        return tuple(model.named_parameters(remove_duplicate=False))
    except TypeError as exc:  # pragma: no cover - torch >=2.5 supports this
        raise Lfm2ParameterSharingTopologyError(
            "This PyTorch build cannot audit aliased Parameter names."
        ) from exc


def _validate_independent_ffns(
    torch: Any,
    model: Any,
    layers: Sequence[Any],
) -> None:
    occurrences: dict[int, list[tuple[int, str]]] = {}
    storage: dict[tuple[str, int], tuple[int, str]] = {}
    ffn_ids: set[int] = set()
    for layer_index, layer in enumerate(layers):
        ffn = layer.feed_forward
        for role in _ROLES:
            linear = getattr(ffn, role)
            parameter = linear._parameters.get("weight")
            if not isinstance(parameter, torch.nn.Parameter):
                raise Lfm2ParameterSharingTopologyError(
                    f"Layer {layer_index} {role}.weight is not a Parameter."
                )
            if parameter.grad is not None:
                raise Lfm2ParameterSharingTopologyError(
                    "Existing gradients are forbidden for zero-training sharing."
                )
            occurrences.setdefault(id(parameter), []).append((layer_index, role))
            ffn_ids.add(id(parameter))
            key = (
                str(parameter.device),
                int(parameter.untyped_storage().data_ptr()),
            )
            prior = storage.get(key)
            if prior is not None:
                raise Lfm2ParameterSharingTopologyError(
                    "Pre-existing partial aliasing or shared FFN storage is "
                    f"forbidden ({prior} and {(layer_index, role)})."
                )
            storage[key] = (layer_index, role)
    duplicated = [values for values in occurrences.values() if len(values) != 1]
    if duplicated:
        raise Lfm2ParameterSharingTopologyError(
            "Pre-existing partial aliasing among LFM2 w1/w3/w2 Parameters is "
            f"forbidden: {duplicated[:3]}."
        )
    full_occurrences: dict[int, int] = {}
    for _, parameter in _named_parameters_no_dedup(model):
        if id(parameter) in ffn_ids:
            full_occurrences[id(parameter)] = full_occurrences.get(id(parameter), 0) + 1
    if set(full_occurrences) != ffn_ids or any(
        count != 1 for count in full_occurrences.values()
    ):
        raise Lfm2ParameterSharingTopologyError(
            "An LFM2 FFN Parameter is missing from the model or aliased outside "
            "its own projection path."
        )


def _inspect_model(
    model: Any,
    *,
    reject_aliasing: bool,
) -> tuple[tuple[Any, ...], tuple[Lfm2LayerContract, ...]]:
    torch = _require_torch()
    if not isinstance(model, torch.nn.Module):
        raise Lfm2ParameterSharingTopologyError("model must be a torch.nn.Module.")
    if model.training:
        raise Lfm2ParameterSharingTopologyError(
            "The loaded LFM2 must be in eval mode; this primitive never trains."
        )
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "lfm2":
        raise Lfm2ParameterSharingTopologyError("model.config.model_type must be lfm2.")
    layers = _locate_layers(model, torch)
    contracts: list[Lfm2LayerContract] = []
    try:
        for index, layer in enumerate(layers):
            contract = inspect_lfm2_layer(layer)
            if contract.layer_index != index:
                raise Lfm2ParameterSharingTopologyError(
                    "LFM2 layer indices must be contiguous and ordered."
                )
            contracts.append(contract)
    except Lfm2CanonicalError as exc:
        raise Lfm2ParameterSharingTopologyError(
            "The model is not a supported concrete Transformers LFM2 topology."
        ) from exc
    hidden_sizes = {contract.hidden_size for contract in contracts}
    intermediate_sizes = {contract.ffn_intermediate_size for contract in contracts}
    if len(hidden_sizes) != 1 or len(intermediate_sizes) != 1:
        raise Lfm2ParameterSharingTopologyError(
            "Every LFM2 layer must have the same hidden and SwiGLU widths."
        )
    all_parameters = tuple(model.parameters())
    if any(parameter.grad is not None for parameter in all_parameters):
        raise Lfm2ParameterSharingTopologyError(
            "Existing model gradients are forbidden for zero-training sharing."
        )
    if reject_aliasing:
        _validate_independent_ffns(torch, model, layers)
    return layers, tuple(contracts)


def _coerce_rows(
    torch: Any,
    post_swiglu_rows: Any,
    *,
    layer_count: int,
    intermediate_size: int,
) -> tuple[Any, ...]:
    if isinstance(post_swiglu_rows, Mapping):
        if set(post_swiglu_rows) != set(range(layer_count)):
            raise Lfm2ParameterSharingEvidenceError(
                "Cached post-SwiGLU evidence must cover every layer exactly once."
            )
        values = tuple(post_swiglu_rows[index] for index in range(layer_count))
    elif isinstance(post_swiglu_rows, Sequence) and not isinstance(
        post_swiglu_rows,
        (str, bytes),
    ):
        values = tuple(post_swiglu_rows)
        if len(values) != layer_count:
            raise Lfm2ParameterSharingEvidenceError(
                "Cached post-SwiGLU evidence count must equal the layer count."
            )
    else:
        raise Lfm2ParameterSharingEvidenceError(
            "post_swiglu_rows must be a layer-index mapping or sequence."
        )
    sealed: list[Any] = []
    row_count: int | None = None
    for index, value in enumerate(values):
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.ndim != 2
            or not torch.is_floating_point(value)
            or torch.is_complex(value)
            or value.device.type == "meta"
        ):
            raise Lfm2ParameterSharingEvidenceError(
                f"Layer {index} rows must be a materialized real rank-2 tensor."
            )
        if tuple(value.shape)[1] != intermediate_size or int(value.shape[0]) < 2:
            raise Lfm2ParameterSharingEvidenceError(
                f"Layer {index} rows must have shape [at least 2, {intermediate_size}]."
            )
        if row_count is None:
            row_count = int(value.shape[0])
        elif int(value.shape[0]) != row_count:
            raise Lfm2ParameterSharingEvidenceError(
                "Every layer must use the same ordered held-out row count."
            )
        result = value.detach().to(device="cpu").contiguous().clone()
        if result.requires_grad or result.grad_fn is not None:
            raise Lfm2ParameterSharingEvidenceError(
                "Cached activation rows retained a gradient graph."
            )
        if not bool(torch.isfinite(result).all()):
            raise Lfm2ParameterSharingEvidenceError(
                f"Layer {index} activation rows contain NaN or infinity."
            )
        sealed.append(result)
    return tuple(sealed)


def _evidence_body(
    evidence: Lfm2ActivationEvidence,
) -> dict[str, Any]:
    return {
        "schema": evidence.schema,
        "cache_key": evidence.cache_key,
        "heldout_corpus_sha256": evidence.heldout_corpus_sha256,
        "parent_identity_sha256": evidence.parent_identity_sha256,
        "layer_count": len(evidence.layer_rows),
        "layer_row_sha256": list(evidence.layer_row_sha256),
        "row_count": evidence.row_count,
        "intermediate_size": evidence.intermediate_size,
        "post_swiglu_verified": evidence.post_swiglu_verified,
    }


def seal_lfm2_activation_evidence(
    model: Any,
    post_swiglu_rows: Mapping[int, Any] | Sequence[Any],
    *,
    cache_key: str,
    heldout_corpus_sha256: str,
) -> Lfm2ActivationEvidence:
    """Seal already-cached post-SwiGLU rows to the exact loaded LFM2 parent."""

    torch = _require_torch()
    with _TRANSACTION_LOCK, torch.no_grad():
        _, contracts = _inspect_model(model, reject_aliasing=True)
        if not _valid_sha256(cache_key):
            raise Lfm2ParameterSharingEvidenceError(
                "cache_key must be a lowercase SHA-256 digest."
            )
        if not _valid_sha256(heldout_corpus_sha256):
            raise Lfm2ParameterSharingEvidenceError(
                "heldout_corpus_sha256 must be a lowercase SHA-256 digest."
            )
        rows = _coerce_rows(
            torch,
            post_swiglu_rows,
            layer_count=len(contracts),
            intermediate_size=contracts[0].ffn_intermediate_size,
        )
        row_hashes = tuple(_tensor_sha256(torch, value) for value in rows)
        evidence = Lfm2ActivationEvidence(
            schema=EVIDENCE_SCHEMA,
            cache_key=cache_key,
            heldout_corpus_sha256=heldout_corpus_sha256,
            parent_identity_sha256=_parent_identity(model),
            layer_rows=rows,
            layer_row_sha256=row_hashes,
            row_count=int(rows[0].shape[0]),
            intermediate_size=contracts[0].ffn_intermediate_size,
            post_swiglu_verified=True,
            evidence_sha256="",
        )
        return Lfm2ActivationEvidence(
            **{
                **{
                    field: getattr(evidence, field)
                    for field in (
                        "schema",
                        "cache_key",
                        "heldout_corpus_sha256",
                        "parent_identity_sha256",
                        "layer_rows",
                        "layer_row_sha256",
                        "row_count",
                        "intermediate_size",
                        "post_swiglu_verified",
                    )
                },
                "evidence_sha256": _json_sha256(_evidence_body(evidence)),
            }
        )


def _validate_evidence(
    model: Any,
    evidence: Any,
) -> tuple[tuple[Any, ...], tuple[Lfm2LayerContract, ...]]:
    torch = _require_torch()
    layers, contracts = _inspect_model(model, reject_aliasing=True)
    if not isinstance(evidence, Lfm2ActivationEvidence):
        raise Lfm2ParameterSharingEvidenceError(
            "A sealed Lfm2ActivationEvidence object is required."
        )
    if evidence.schema != EVIDENCE_SCHEMA or evidence.post_swiglu_verified is not True:
        raise Lfm2ParameterSharingEvidenceError(
            "Activation evidence schema or post-SwiGLU verification is invalid."
        )
    if not _valid_sha256(evidence.cache_key) or not _valid_sha256(
        evidence.heldout_corpus_sha256
    ):
        raise Lfm2ParameterSharingEvidenceError(
            "Activation evidence cache identity is invalid."
        )
    if not _valid_sha256(evidence.parent_identity_sha256) or not _valid_sha256(
        evidence.evidence_sha256
    ):
        raise Lfm2ParameterSharingEvidenceError(
            "Activation evidence identity digest is invalid."
        )
    expected_intermediate = contracts[0].ffn_intermediate_size
    if (
        len(evidence.layer_rows) != len(layers)
        or len(evidence.layer_row_sha256) != len(layers)
        or evidence.intermediate_size != expected_intermediate
        or evidence.row_count < 2
    ):
        raise Lfm2ParameterSharingEvidenceError(
            "Activation evidence does not cover this LFM2 topology."
        )
    for index, (rows, digest) in enumerate(
        zip(evidence.layer_rows, evidence.layer_row_sha256)
    ):
        if (
            not isinstance(rows, torch.Tensor)
            or rows.layout != torch.strided
            or rows.ndim != 2
            or tuple(rows.shape) != (evidence.row_count, evidence.intermediate_size)
            or not torch.is_floating_point(rows)
            or torch.is_complex(rows)
            or not bool(torch.isfinite(rows).all())
        ):
            raise Lfm2ParameterSharingEvidenceError(
                f"Layer {index} cached rows are missing or malformed."
            )
        if _tensor_sha256(torch, rows) != digest:
            raise Lfm2ParameterSharingEvidenceError(
                f"Layer {index} cached rows changed after evidence sealing."
            )
    if _json_sha256(_evidence_body(evidence)) != evidence.evidence_sha256:
        raise Lfm2ParameterSharingEvidenceError(
            "Activation evidence metadata changed after sealing."
        )
    current_identity = _parent_identity(model)
    if current_identity != evidence.parent_identity_sha256:
        raise Lfm2ParameterSharingEvidenceError(
            "Cached activation evidence is stale for the loaded LFM2 parent."
        )
    return layers, contracts


def _normalized_weights(values: Any, name: str) -> tuple[float, float]:
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or len(values) != 2
    ):
        raise Lfm2ParameterSharingError(f"{name} must contain two weights.")
    result: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise Lfm2ParameterSharingError(
                f"{name} weights must be finite and non-negative."
            )
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise Lfm2ParameterSharingError(
                f"{name} weights must be finite and non-negative."
            ) from exc
        if not math.isfinite(number) or number < 0.0:
            raise Lfm2ParameterSharingError(
                f"{name} weights must be finite and non-negative."
            )
        result.append(number)
    total = sum(result)
    if total <= 0.0:
        raise Lfm2ParameterSharingError(f"{name} weights must have a positive total.")
    return (result[0] / total, result[1] / total)


def _linear_cka(torch: Any, left: Any, right: Any, name: str) -> float:
    if tuple(left.shape) != tuple(right.shape):
        raise Lfm2ParameterSharingEvidenceError(
            f"{name} matrices must have equal shapes."
        )
    left64 = left.detach().to(device="cpu", dtype=torch.float64)
    right64 = right.detach().to(device="cpu", dtype=torch.float64)
    left64 = left64 - left64.mean(dim=0, keepdim=True)
    right64 = right64 - right64.mean(dim=0, keepdim=True)
    left_gram = left64 @ left64.transpose(0, 1)
    right_gram = right64 @ right64.transpose(0, 1)
    left_gram = (
        left_gram
        - left_gram.mean(dim=0, keepdim=True)
        - left_gram.mean(dim=1, keepdim=True)
        + left_gram.mean()
    )
    right_gram = (
        right_gram
        - right_gram.mean(dim=0, keepdim=True)
        - right_gram.mean(dim=1, keepdim=True)
        + right_gram.mean()
    )
    numerator = float((left_gram * right_gram).sum().item())
    left_norm = float(left_gram.square().sum().item())
    right_norm = float(right_gram.square().sum().item())
    denominator = math.sqrt(left_norm * right_norm)
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or denominator <= 0.0
    ):
        raise Lfm2ParameterSharingEvidenceError(
            f"{name} evidence is degenerate for linear CKA."
        )
    value = numerator / denominator
    if not math.isfinite(value):
        raise Lfm2ParameterSharingEvidenceError(f"{name} linear CKA is non-finite.")
    return min(1.0, max(0.0, value))


def _parameter_counts(model: Any) -> tuple[int, int]:
    named = _named_parameters_no_dedup(model)
    logical = sum(int(parameter.numel()) for _, parameter in named)
    unique: dict[int, Any] = {}
    for _, parameter in named:
        unique[id(parameter)] = parameter
    physical = sum(int(parameter.numel()) for parameter in unique.values())
    return logical, physical


def lfm2_ffn_parameter_sharing_savings(
    hidden_size: int,
    intermediate_size: int,
    pair_count: int,
) -> int:
    """Return exact physical savings for full-width bias-free LFM2 FFN pairs.

    For LFM2.5-230M, ``hidden_size=1024``, ``intermediate_size=2560``, and
    ``pair_count=3`` saves exactly 23,592,960 parameters.
    """

    for value, name, allow_zero in (
        (hidden_size, "hidden_size", False),
        (intermediate_size, "intermediate_size", False),
        (pair_count, "pair_count", True),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or (not allow_zero and value == 0)
        ):
            qualifier = "non-negative" if allow_zero else "positive"
            raise Lfm2ParameterSharingError(f"{name} must be a {qualifier} integer.")
    return 3 * hidden_size * intermediate_size * pair_count


def _pair_similarities(
    torch: Any,
    layers: Sequence[Any],
    evidence: Lfm2ActivationEvidence,
    score_weights: tuple[float, float],
    requested_pairs: set[tuple[int, int]] | None,
) -> tuple[Lfm2PairSimilarity, ...]:
    # Pair scoring needs only w2. Avoid materializing all three float64 FFN
    # matrices for every production layer before the disjoint pairs are known.
    # Complete bundles are extracted below only for selected pairs.
    outputs = tuple(
        evidence.layer_rows[index].to(torch.float64)
        @ layer.feed_forward.w2.weight.detach()
        .to(device="cpu", dtype=torch.float64)
        .transpose(0, 1)
        for index, layer in enumerate(layers)
    )
    result: list[Lfm2PairSimilarity] = []
    for anchor in range(len(layers)):
        for source in range(anchor + 1, len(layers)):
            pair = (anchor, source)
            if requested_pairs is not None and pair not in requested_pairs:
                continue
            activation = _linear_cka(
                torch,
                evidence.layer_rows[anchor],
                evidence.layer_rows[source],
                f"layers {anchor}/{source} post-SwiGLU",
            )
            output = _linear_cka(
                torch,
                outputs[anchor],
                outputs[source],
                f"layers {anchor}/{source} down output",
            )
            combined = score_weights[0] * activation + score_weights[1] * output
            result.append(
                Lfm2PairSimilarity(
                    anchor_layer=anchor,
                    source_layer=source,
                    activation_similarity=activation,
                    output_similarity=output,
                    combined_similarity=combined,
                )
            )
    return tuple(result)


def _validated_explicit_pairs(
    layer_pairs: Any,
    *,
    layer_count: int,
    pair_count: int,
) -> tuple[tuple[int, int], ...] | None:
    if layer_pairs is None:
        return None
    if not isinstance(layer_pairs, Sequence) or isinstance(layer_pairs, (str, bytes)):
        raise Lfm2ParameterSharingTopologyError(
            "layer_pairs must be a sequence of two-index pairs."
        )
    normalized: list[tuple[int, int]] = []
    used: set[int] = set()
    for value in layer_pairs:
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != 2
        ):
            raise Lfm2ParameterSharingTopologyError(
                "Each explicit layer pair must contain two indices."
            )
        left, right = value
        if any(
            isinstance(index, bool) or not isinstance(index, int) for index in value
        ):
            raise Lfm2ParameterSharingTopologyError(
                "Explicit layer-pair indices must be integers."
            )
        if (
            left == right
            or left < 0
            or right < 0
            or left >= layer_count
            or right >= layer_count
        ):
            raise Lfm2ParameterSharingTopologyError(
                "Explicit layer-pair indices must be distinct and in range."
            )
        pair = (min(left, right), max(left, right))
        if pair[0] in used or pair[1] in used:
            raise Lfm2ParameterSharingTopologyError(
                "Overlapping layer pairs are forbidden."
            )
        used.update(pair)
        normalized.append(pair)
    if len(normalized) != pair_count:
        raise Lfm2ParameterSharingTopologyError(
            "Explicit layer-pair count must equal pair_count."
        )
    if len(set(normalized)) != len(normalized):
        raise Lfm2ParameterSharingTopologyError("Duplicate layer pairs are forbidden.")
    return tuple(sorted(normalized))


def _exact_disjoint_selection(
    similarities: Sequence[Lfm2PairSimilarity],
    *,
    layer_count: int,
    pair_count: int,
) -> tuple[Lfm2PairSimilarity, ...]:
    by_pair = {value.layers: value for value in similarities}
    full_mask = (1 << layer_count) - 1

    @lru_cache(maxsize=None)
    def solve(
        mask: int, remaining: int
    ) -> tuple[float, tuple[tuple[int, int], ...]] | None:
        if remaining == 0:
            return (0.0, ())
        if mask.bit_count() < 2 * remaining:
            return None
        first = (mask & -mask).bit_length() - 1
        best = solve(mask & ~(1 << first), remaining)
        for source in range(first + 1, layer_count):
            if not mask & (1 << source):
                continue
            tail = solve(
                mask & ~(1 << first) & ~(1 << source),
                remaining - 1,
            )
            if tail is None:
                continue
            pair = (first, source)
            candidate = (
                by_pair[pair].combined_similarity + tail[0],
                (pair, *tail[1]),
            )
            if (
                best is None
                or candidate[0] > best[0]
                or (candidate[0] == best[0] and candidate[1] < best[1])
            ):
                best = candidate
        return best

    selected = solve(full_mask, pair_count)
    if selected is None:
        raise Lfm2ParameterSharingTopologyError(
            "Could not select the requested number of disjoint layer pairs."
        )
    return tuple(by_pair[pair] for pair in selected[1])


def _greedy_disjoint_selection(
    similarities: Sequence[Lfm2PairSimilarity],
    *,
    pair_count: int,
) -> tuple[Lfm2PairSimilarity, ...]:
    ordered = sorted(
        similarities,
        key=lambda value: (
            -value.combined_similarity,
            -value.activation_similarity,
            -value.output_similarity,
            value.anchor_layer,
            value.source_layer,
        ),
    )
    used: set[int] = set()
    selected: list[Lfm2PairSimilarity] = []
    for value in ordered:
        if value.anchor_layer in used or value.source_layer in used:
            continue
        selected.append(value)
        used.update(value.layers)
        if len(selected) == pair_count:
            break
    if len(selected) != pair_count:
        raise Lfm2ParameterSharingTopologyError(
            "Could not select the requested number of disjoint layer pairs."
        )
    return tuple(sorted(selected, key=lambda value: value.layers))


def plan_lfm2_activation_aligned_parameter_sharing(
    model: Any,
    evidence: Lfm2ActivationEvidence,
    *,
    pair_count: int,
    layer_pairs: Sequence[Sequence[int]] | None = None,
    pair_score_weights: Sequence[float] = (0.5, 0.5),
    merge_weights: Sequence[float] = (0.5, 0.5),
    exact_assignment_limit: int = DEFAULT_EXACT_ASSIGNMENT_LIMIT,
    chunk_size: int = DEFAULT_CORRELATION_CHUNK_SIZE,
    variance_floor: float = 0.0,
) -> Lfm2ParameterSharingPlan:
    """Select disjoint pairs and stage every full-width merge without mutation."""

    torch = _require_torch()
    with _TRANSACTION_LOCK, torch.no_grad():
        layers, contracts = _validate_evidence(model, evidence)
        if (
            isinstance(pair_count, bool)
            or not isinstance(pair_count, int)
            or pair_count <= 0
            or pair_count > len(layers) // 2
        ):
            raise Lfm2ParameterSharingTopologyError(
                "pair_count must be positive and cannot exceed half the layer count."
            )
        explicit = _validated_explicit_pairs(
            layer_pairs,
            layer_count=len(layers),
            pair_count=pair_count,
        )
        score_weights = _normalized_weights(
            pair_score_weights,
            "pair_score_weights",
        )
        normalized_merge = _normalized_weights(merge_weights, "merge_weights")
        requested = set(explicit) if explicit is not None else None
        similarities = _pair_similarities(
            torch,
            layers,
            evidence,
            score_weights,
            requested,
        )
        if explicit is not None:
            similarity_by_pair = {value.layers: value for value in similarities}
            selected = tuple(similarity_by_pair[pair] for pair in explicit)
            selection_solver = "explicit-disjoint-pairs"
        elif len(layers) <= 20:
            selected = _exact_disjoint_selection(
                similarities,
                layer_count=len(layers),
                pair_count=pair_count,
            )
            selection_solver = "exact-cardinality-disjoint-pair-dynamic-program"
        else:
            selected = _greedy_disjoint_selection(
                similarities,
                pair_count=pair_count,
            )
            selection_solver = "deterministic-ranked-disjoint-greedy"

        pair_plans: list[Lfm2FfnPairPlan] = []
        intermediate = contracts[0].ffn_intermediate_size
        hidden = contracts[0].hidden_size
        expected_pair_savings = lfm2_ffn_parameter_sharing_savings(
            hidden,
            intermediate,
            1,
        )
        for similarity in selected:
            anchor, source = similarity.layers
            bundles = (
                extract_mlp_bundle(layers[anchor].feed_forward, _LAYOUT),
                extract_mlp_bundle(layers[source].feed_forward, _LAYOUT),
            )
            merge = align_and_merge_mlp_bundles(
                bundles,
                (
                    evidence.layer_rows[anchor],
                    evidence.layer_rows[source],
                ),
                weights=normalized_merge,
                exact_assignment_limit=exact_assignment_limit,
                chunk_size=chunk_size,
                variance_floor=variance_floor,
            )
            permutation = merge.permutations[1]
            if (
                tuple(permutation.shape) != (intermediate,)
                or len(set(int(value) for value in permutation.tolist()))
                != intermediate
                or merge.parameter_counts.parameters_saved != expected_pair_savings
            ):
                raise Lfm2ParameterSharingTransactionError(
                    "Full-width coupled SwiGLU alignment failed its invariants."
                )
            pair_plans.append(
                Lfm2FfnPairPlan(
                    similarity=similarity,
                    merge_result=merge,
                    aligned_neuron_count=intermediate,
                    parameters_saved=expected_pair_savings,
                )
            )
        logical, unique_before = _parameter_counts(model)
        total_savings = sum(pair.parameters_saved for pair in pair_plans)
        plan = Lfm2ParameterSharingPlan(
            schema=PLAN_SCHEMA,
            parent_identity_sha256=evidence.parent_identity_sha256,
            evidence_sha256=evidence.evidence_sha256,
            requested_pair_count=pair_count,
            pair_plans=tuple(pair_plans),
            pair_score_weights=score_weights,
            merge_weights=normalized_merge,
            selection_solver=selection_solver,
            logical_parameters=logical,
            unique_parameters_before=unique_before,
            projected_unique_parameters_after=unique_before - total_savings,
            projected_parameters_saved=total_savings,
            layer_count=len(layers),
            hidden_size=hidden,
            intermediate_size=intermediate,
            plan_sha256="",
        )
        if plan.projected_unique_parameters_after <= 0:
            raise Lfm2ParameterSharingTransactionError(
                "Projected unique parameter count is invalid."
            )
        return _seal_plan(torch, plan)


def _buffer_snapshot(torch: Any, model: Any) -> dict[str, tuple[Any, Any]]:
    return {
        name: (value, value.detach().to(device="cpu").contiguous().clone())
        for name, value in model.named_buffers(remove_duplicate=False)
    }


def _verify_buffers_unchanged(
    torch: Any,
    model: Any,
    snapshot: Mapping[str, tuple[Any, Any]],
) -> None:
    current = dict(model.named_buffers(remove_duplicate=False))
    if set(current) != set(snapshot):
        raise Lfm2ParameterSharingTransactionError(
            "Model buffers changed during the sharing transaction."
        )
    for name, value in current.items():
        original, original_value = snapshot[name]
        if value is not original or not torch.equal(
            value.detach().to(device="cpu"),
            original_value,
        ):
            raise Lfm2ParameterSharingTransactionError(
                f"Model buffer {name!r} changed during the sharing transaction."
            )


def _structure_snapshot(layers: Sequence[Any]) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (
            id(layer),
            id(layer.feed_forward),
            id(layer.self_attn if layer.is_attention_layer else layer.conv),
        )
        for layer in layers
    )


def _selected_bindings(
    layers: Sequence[Any],
    pairs: Sequence[Lfm2FfnPairPlan],
) -> tuple[_ParameterBinding, ...]:
    result: list[_ParameterBinding] = []
    for pair in pairs:
        for layer_index in pair.layers:
            ffn = layers[layer_index].feed_forward
            for role in _ROLES:
                owner = getattr(ffn, role)
                parameter = owner._parameters.get("weight")
                result.append(
                    _ParameterBinding(
                        owner=owner,
                        name="weight",
                        parameter=parameter,
                    )
                )
    return tuple(result)


def _verify_installed_aliases(
    layers: Sequence[Any],
    pair_plans: Sequence[Lfm2FfnPairPlan],
) -> None:
    expected: set[frozenset[tuple[int, str]]] = set()
    for pair in pair_plans:
        for role in _ROLES:
            expected.add(
                frozenset(
                    {
                        (pair.layers[0], role),
                        (pair.layers[1], role),
                    }
                )
            )
    occurrences: dict[int, set[tuple[int, str]]] = {}
    for layer_index, layer in enumerate(layers):
        for role in _ROLES:
            parameter = getattr(layer.feed_forward, role).weight
            occurrences.setdefault(id(parameter), set()).add((layer_index, role))
    actual = {frozenset(values) for values in occurrences.values() if len(values) > 1}
    if actual != expected:
        raise Lfm2ParameterSharingTransactionError(
            "Installed FFN aliases do not exactly equal the selected disjoint pairs."
        )


def install_lfm2_parameter_sharing_plan(
    model: Any,
    evidence: Lfm2ActivationEvidence,
    plan: Lfm2ParameterSharingPlan,
) -> InstalledLfm2ParameterSharing:
    """Atomically install a precomputed full-width partial-sharing plan."""

    torch = _require_torch()
    with _TRANSACTION_LOCK, torch.no_grad():
        layers, contracts = _validate_evidence(model, evidence)
        if not isinstance(plan, Lfm2ParameterSharingPlan) or plan.schema != PLAN_SCHEMA:
            raise Lfm2ParameterSharingTransactionError(
                "A valid LFM2 parameter-sharing plan is required."
            )
        if (
            plan.parent_identity_sha256 != evidence.parent_identity_sha256
            or plan.evidence_sha256 != evidence.evidence_sha256
            or plan.layer_count != len(layers)
            or plan.hidden_size != contracts[0].hidden_size
            or plan.intermediate_size != contracts[0].ffn_intermediate_size
            or len(plan.pair_plans) != plan.requested_pair_count
        ):
            raise Lfm2ParameterSharingTransactionError(
                "The plan is stale or does not match this model/evidence pair."
            )
        _validated_explicit_pairs(
            plan.selected_pairs,
            layer_count=len(layers),
            pair_count=plan.requested_pair_count,
        )
        logical_before, unique_before = _parameter_counts(model)
        if (
            logical_before != plan.logical_parameters
            or unique_before != plan.unique_parameters_before
        ):
            raise Lfm2ParameterSharingTransactionError(
                "Model parameter counts changed after planning."
            )

        parameters_before = dict(_named_parameters_no_dedup(model))
        buffers_before = _buffer_snapshot(torch, model)
        structure_before = _structure_snapshot(layers)
        originals = _selected_bindings(layers, plan.pair_plans)
        original_ids = {id(binding.parameter) for binding in originals}
        expected_names = {
            name
            for name, parameter in parameters_before.items()
            if id(parameter) in original_ids
        }
        if len(expected_names) != len(originals):
            raise Lfm2ParameterSharingTransactionError(
                "Selected FFN Parameter names are aliased or incomplete."
            )

        # This must remain the last operation before the first model mutation.
        # Plan dataclasses are frozen, but their tensor payloads are mutable.
        _validate_plan_seal(torch, plan)
        try:
            for pair in plan.pair_plans:
                anchor, source = pair.layers
                report = tie_mlp_modules(
                    (
                        layers[anchor].feed_forward,
                        layers[source].feed_forward,
                    ),
                    pair.merged_bundle,
                    _LAYOUT,
                )
                if (
                    not report.fully_shared
                    or report.parameters_saved != pair.parameters_saved
                    or report.unique_parameter_objects != 3
                ):
                    raise Lfm2ParameterSharingTransactionError(
                        "A selected FFN pair did not become exactly three shared "
                        "Parameter objects."
                    )

            post_layers, _ = _inspect_model(model, reject_aliasing=False)
            if _structure_snapshot(post_layers) != structure_before:
                raise Lfm2ParameterSharingTransactionError(
                    "A decoder layer, FFN module, or recurrent/attention operator "
                    "was replaced."
                )
            _verify_installed_aliases(post_layers, plan.pair_plans)
            parameters_after = dict(_named_parameters_no_dedup(model))
            if set(parameters_after) != set(parameters_before):
                raise Lfm2ParameterSharingTransactionError(
                    "The model Parameter namespace changed during installation."
                )
            changed_names = {
                name
                for name in parameters_before
                if parameters_after[name] is not parameters_before[name]
            }
            if changed_names != expected_names:
                raise Lfm2ParameterSharingTransactionError(
                    "Installation replaced Parameters outside the selected FFN pairs."
                )
            _verify_buffers_unchanged(torch, model, buffers_before)
            logical_after, unique_after = _parameter_counts(model)
            if (
                logical_after != logical_before
                or unique_before - unique_after != plan.projected_parameters_saved
                or unique_after != plan.projected_unique_parameters_after
            ):
                raise Lfm2ParameterSharingTransactionError(
                    "Logical/unique parameter accounting failed after installation."
                )
            model_identity_after = _parent_identity(model)
            installed = tuple(
                binding.owner._parameters[binding.name] for binding in originals
            )
            sharing_report = Lfm2ParameterSharingReport(
                schema=REPORT_SCHEMA,
                parent_identity_before=evidence.parent_identity_sha256,
                model_identity_after=model_identity_after,
                evidence_sha256=evidence.evidence_sha256,
                selected_pairs=plan.selected_pairs,
                logical_parameters=logical_after,
                unique_parameters_before=unique_before,
                unique_parameters_after=unique_after,
                parameters_saved=unique_before - unique_after,
                tied_parameter_objects=3 * plan.requested_pair_count,
                aligned_neurons_per_pair=plan.intermediate_size,
                topology_preserved=True,
                recurrent_state_preserved=True,
                only_selected_ffn_parameters_replaced=True,
            )
            return InstalledLfm2ParameterSharing(
                model=model,
                report=sharing_report,
                original_bindings=originals,
                installed_parameters=installed,
                original_buffer_snapshot=buffers_before,
            )
        except BaseException as exc:
            for binding in originals:
                binding.owner._parameters[binding.name] = binding.parameter
            try:
                restored = _parent_identity(model)
                _verify_buffers_unchanged(torch, model, buffers_before)
                restored_parameters = dict(_named_parameters_no_dedup(model))
                objects_restored = bool(
                    set(restored_parameters) == set(parameters_before)
                    and all(
                        restored_parameters[name] is parameter
                        for name, parameter in parameters_before.items()
                    )
                )
            except BaseException as rollback_exc:
                raise Lfm2ParameterSharingTransactionError(
                    "Installation failed and exact whole-model rollback could not "
                    "be authenticated."
                ) from rollback_exc
            if (
                restored != evidence.parent_identity_sha256
                or not objects_restored
            ):
                raise Lfm2ParameterSharingTransactionError(
                    "Installation failed and rollback did not restore the parent."
                ) from exc
            raise


def install_lfm2_activation_aligned_parameter_sharing(
    model: Any,
    evidence: Lfm2ActivationEvidence,
    *,
    pair_count: int,
    layer_pairs: Sequence[Sequence[int]] | None = None,
    pair_score_weights: Sequence[float] = (0.5, 0.5),
    merge_weights: Sequence[float] = (0.5, 0.5),
    exact_assignment_limit: int = DEFAULT_EXACT_ASSIGNMENT_LIMIT,
    chunk_size: int = DEFAULT_CORRELATION_CHUNK_SIZE,
    variance_floor: float = 0.0,
) -> InstalledLfm2ParameterSharing:
    """Plan and atomically install activation-aligned partial FFN sharing."""

    plan = plan_lfm2_activation_aligned_parameter_sharing(
        model,
        evidence,
        pair_count=pair_count,
        layer_pairs=layer_pairs,
        pair_score_weights=pair_score_weights,
        merge_weights=merge_weights,
        exact_assignment_limit=exact_assignment_limit,
        chunk_size=chunk_size,
        variance_floor=variance_floor,
    )
    return install_lfm2_parameter_sharing_plan(model, evidence, plan)


build_lfm2_parameter_sharing_plan = plan_lfm2_activation_aligned_parameter_sharing
install_lfm2_parameter_sharing = install_lfm2_activation_aligned_parameter_sharing


__all__ = [
    "ALIGNMENT_METHOD",
    "EVIDENCE_SCHEMA",
    "InstalledLfm2ParameterSharing",
    "Lfm2ActivationEvidence",
    "Lfm2FfnPairPlan",
    "Lfm2PairSimilarity",
    "Lfm2ParameterSharingError",
    "Lfm2ParameterSharingEvidenceError",
    "Lfm2ParameterSharingPlan",
    "Lfm2ParameterSharingReport",
    "Lfm2ParameterSharingTopologyError",
    "Lfm2ParameterSharingTransactionError",
    "PLAN_SCHEMA",
    "REPORT_SCHEMA",
    "SELECTION_METHOD",
    "build_lfm2_parameter_sharing_plan",
    "install_lfm2_activation_aligned_parameter_sharing",
    "install_lfm2_parameter_sharing",
    "install_lfm2_parameter_sharing_plan",
    "lfm2_ffn_parameter_sharing_savings",
    "plan_lfm2_activation_aligned_parameter_sharing",
    "seal_lfm2_activation_evidence",
]
