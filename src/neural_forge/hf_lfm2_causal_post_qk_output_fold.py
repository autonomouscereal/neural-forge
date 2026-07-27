"""Causal cache-derived attention-output folding after target-lattice Q/K.

The target-lattice Q/K pass leaves the child's value and output projections
fixed.  This stage resumes from that freshly reloaded checkpoint and folds the
attention output projections in causal layer order.  Parent local actions are
loaded only from the authenticated circuit cache and child contexts are
obtained only by analytic macrocell replay.  A caller cannot supply fit rows,
replay tensors, or replay digests.

Each accepted ``out_proj`` is replayed before the residual bank advances to the
next layer.  The complete operation is covered by one whole-model snapshot;
any authentication, compilation, installation, or replay failure restores the
exact entry state.  This is an analytic fold-only stage: it contains no parent
model construction, trace capture, model forward, generation, gradient,
optimizer, training, pruning, quantization, sparsity, or parameter sharing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_attention_group_qdeim import (
    LFM2_ATTENTION_GROUP_QDEIM_SCHEMA,
    LFM2_ATTENTION_GROUP_QDEIM_SOURCE_LOCAL_OUTPUT_RESIDUAL_SCHEMA,
    Lfm2CompiledAttentionGroupQdeim,
    lfm2_attention_group_qdeim_candidate_state_sha256,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceCache,
    fingerprint_lfm2_parent,
)
from neural_forge.hf_lfm2_semantic_attention_source_local_output import (
    LFM2_SEMANTIC_ATTENTION_SOURCE_LOCAL_OUTPUT_INSTALLATION_SCHEMA,
    compile_install_lfm2_semantic_attention_source_local_output,
)
from neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter import (
    LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA,
    _WholeModelSnapshot,
    _build_attention_source_local_output_fit_cases,
    _initial_residual_bank,
    _load_parent_layer_trajectories,
    _next_residual_bank,
    _parameter_topology,
    _replay_current_layer,
)
from neural_forge.hf_lfm2_semantic_postfold import (
    LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_LAYER_INSTALLATION_SCHEMA,
    _changed_state_keys,
    _parameter_count,
    _semantic_alignments_from_session,
    _state_tensor_hashes,
    _tensor_sha256,
)
from neural_forge.hf_lfm2_semantic_span_cpcf_alignment import (
    audit_lfm2_semantic_span_cpcf_refinement,
)
from neural_forge.hf_lfm2_structural_attention_candidate_archive import (
    LFM2_STRUCTURAL_ATTENTION_CANDIDATE_ARCHIVE_SCHEMA,
)
from neural_forge.hf_tokenizer_bridge import StrictTokenizerBridgeArtifact
from neural_forge import hf_lfm2_post_target_lattice_qk_output_fold as _post_qk
from neural_forge import hf_lfm2_semantic_attention_source_local_output as _local
from neural_forge import hf_lfm2_structural_attention_candidate_archive as _archive


LFM2_CAUSAL_POST_QK_OUTPUT_FOLD_SCHEMA = (
    "neural-forge-lfm2-causal-post-target-lattice-qk-output-fold-v1"
)
LFM2_CAUSAL_POST_QK_OUTPUT_LAYER_SCHEMA = (
    "neural-forge-lfm2-causal-post-target-lattice-qk-output-layer-v1"
)
LFM2_CAUSAL_POST_QK_REPLAY_STEP_SCHEMA = (
    "neural-forge-lfm2-causal-post-target-lattice-qk-replay-step-v1"
)
_RESIDUAL_CHART_SCHEMA = "neural-forge-lfm2-layer-stack-residual-chart-v1"
_SEMANTIC_LAYER_TRANSACTION_SCHEMA = (
    "neural-forge-lfm2-semantic-layer-transaction-v1"
)
_STRUCTURAL_ATTENTION_LAYER_SCHEMA = (
    LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_LAYER_INSTALLATION_SCHEMA
)
_ARCHIVE_TENSOR_FIELDS = (
    "query_weight",
    "key_weight",
    "value_weight",
    "out_weight",
    "group_pod_basis",
    "group_interpolation",
    "context_reconstruction",
    "normalized_input_decoder",
    "residual_output_chart",
)
_QKV_ROLES = ("q_proj.weight", "k_proj.weight", "v_proj.weight")


class Lfm2CausalPostQkOutputFoldError(RuntimeError):
    """Causal post-Q/K O folding failed closed and was rolled back."""


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_deep_thaw(item) for item in value]
    return value


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CausalPostQkOutputFoldError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _sealed(body: Mapping[str, Any]) -> Mapping[str, Any]:
    plain = _deep_thaw(body)
    return MappingProxyType({**plain, "result_sha256": canonical_sha256(plain)})


def _verified_result(
    value: Any,
    *,
    label: str,
    expected_schema: str | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2CausalPostQkOutputFoldError(
            f"{label} must be one sealed mapping."
        )
    body = _deep_thaw(value)
    supplied = body.pop("result_sha256", None)
    if (
        supplied != canonical_sha256(body)
        or (expected_schema is not None and body.get("schema") != expected_schema)
    ):
        raise Lfm2CausalPostQkOutputFoldError(
            f"{label} is stale, malformed, or unauthenticated."
        )
    return MappingProxyType({**body, "result_sha256": supplied})


def _runtime_inventory(model: Any) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            str(name): {
                "object_id": id(parameter),
                "storage_pointer": int(parameter.data_ptr()),
                "requires_grad": bool(parameter.requires_grad),
                "gradient_object_id": (
                    None if parameter.grad is None else id(parameter.grad)
                ),
                "gradient_sha256": (
                    None
                    if parameter.grad is None
                    else _tensor_sha256(parameter.grad)
                ),
            }
            for name, parameter in model.named_parameters()
        }
    )


@dataclass(frozen=True, slots=True)
class _AttentionStackBinding:
    layer_index: int
    fit_batch_indices: tuple[int, ...]
    fit_batch_sample_weights: Mapping[int, float] = field(
        repr=False,
        compare=False,
    )
    structural_receipt: Mapping[str, Any] = field(repr=False, compare=False)
    compiler_receipt: Mapping[str, Any] = field(repr=False, compare=False)
    residual_chart_receipt: Mapping[str, Any] = field(repr=False, compare=False)


def _stack_and_attention_bindings(
    *,
    semantic_layer_stack_receipt: Mapping[str, Any],
    expected_semantic_layer_stack_receipt_sha256: str,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
) -> tuple[
    Mapping[str, Any],
    tuple[str, ...],
    Mapping[int, _AttentionStackBinding],
]:
    expected_stack = _digest(
        expected_semantic_layer_stack_receipt_sha256,
        label="expected semantic layer-stack receipt",
    )
    stack = _verified_result(
        semantic_layer_stack_receipt,
        expected_schema=LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA,
        label="semantic layer-stack receipt",
    )
    if stack["result_sha256"] != expected_stack:
        raise Lfm2CausalPostQkOutputFoldError(
            "The semantic layer stack does not match its independent pin."
        )
    raw_layers = stack.get("layer_installations")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise Lfm2CausalPostQkOutputFoldError(
            "The semantic layer stack has no causal layer transactions."
        )
    layer_receipts = tuple(
        _verified_result(
            raw,
            expected_schema=_SEMANTIC_LAYER_TRANSACTION_SCHEMA,
            label="semantic layer transaction",
        )
        for raw in raw_layers
    )
    if tuple(receipt.get("layer_index") for receipt in layer_receipts) != tuple(
        range(len(layer_receipts))
    ):
        raise Lfm2CausalPostQkOutputFoldError(
            "Semantic layer transactions are not in exact numeric causal order."
        )
    kinds = tuple(str(receipt.get("operator_kind")) for receipt in layer_receipts)
    if any(kind not in {"full_attention", "short_conv"} for kind in kinds):
        raise Lfm2CausalPostQkOutputFoldError(
            "Semantic layer transactions contain an unsupported operator kind."
        )
    topology = cache.manifest.get("layer_topology", {}).get("layers")
    if not isinstance(topology, list) or len(topology) != len(kinds):
        raise Lfm2CausalPostQkOutputFoldError(
            "The authenticated cache and semantic stack layer counts diverge."
        )
    cache_kinds = tuple(
        (
            value.get("contract", {}).get("kind")
            if isinstance(value, Mapping)
            and isinstance(value.get("contract"), Mapping)
            else None
        )
        for value in topology
    )
    if cache_kinds != kinds:
        raise Lfm2CausalPostQkOutputFoldError(
            "The authenticated cache and semantic stack operator topology diverge."
        )

    bindings: dict[int, _AttentionStackBinding] = {}
    for layer in layer_receipts:
        if layer["operator_kind"] != "full_attention":
            continue
        index = int(layer["layer_index"])
        operator = _verified_result(
            layer.get("attention_operator_installation_receipt"),
            label=f"layer {index} attention installation",
        )
        structural_value = operator.get("structural_attention_operator_receipt")
        structural = (
            _verified_result(
                structural_value,
                expected_schema=_STRUCTURAL_ATTENTION_LAYER_SCHEMA,
                label=f"layer {index} structural attention installation",
            )
            if isinstance(structural_value, Mapping)
            else operator
        )
        if structural.get("schema") != _STRUCTURAL_ATTENTION_LAYER_SCHEMA:
            raise Lfm2CausalPostQkOutputFoldError(
                f"Layer {index} has no intact-group structural attention receipt."
            )
        compiler = _verified_result(
            structural.get("coupled_attention_compiler_receipt"),
            expected_schema=LFM2_ATTENTION_GROUP_QDEIM_SCHEMA,
            label=f"layer {index} structural attention compiler",
        )
        chart = _verified_result(
            structural.get("residual_chart_receipt"),
            expected_schema=_RESIDUAL_CHART_SCHEMA,
            label=f"layer {index} residual output chart",
        )
        fit_indices = structural.get("fit_batch_indices")
        raw_weights = structural.get("fit_batch_sample_weights")
        chart_partitions = chart.get("partition_batch_indices")
        chart_fit = (
            chart_partitions.get("fit")
            if isinstance(chart_partitions, Mapping)
            else None
        )
        if (
            structural.get("layer_index") != index
            or compiler.get("selection") is None
            or chart.get("layer_index") != index
            or not isinstance(fit_indices, list)
            or not fit_indices
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in fit_indices
            )
            or len(fit_indices) != len(set(fit_indices))
            or chart_fit != fit_indices
            or not isinstance(raw_weights, list)
            or len(raw_weights) != len(fit_indices)
            or structural.get("cache_key") != cache.cache_key
            or structural.get("bridge_artifact_sha256") != bridge.artifact_sha256
        ):
            raise Lfm2CausalPostQkOutputFoldError(
                f"Layer {index} lost its cache-bound FIT partition."
            )
        weights: dict[int, float] = {}
        for expected_index, raw in zip(fit_indices, raw_weights, strict=True):
            if not isinstance(raw, Mapping):
                raise Lfm2CausalPostQkOutputFoldError(
                    f"Layer {index} FIT weight receipt is malformed."
                )
            weight = raw.get("sample_weight")
            if (
                raw.get("batch_index") != expected_index
                or isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not 0.0 < float(weight) < float("inf")
            ):
                raise Lfm2CausalPostQkOutputFoldError(
                    f"Layer {index} FIT trajectory weight is invalid."
                )
            weights[expected_index] = float(weight)
        bindings[index] = _AttentionStackBinding(
            layer_index=index,
            fit_batch_indices=tuple(fit_indices),
            fit_batch_sample_weights=MappingProxyType(weights),
            structural_receipt=structural,
            compiler_receipt=compiler,
            residual_chart_receipt=chart,
        )
    if not bindings:
        raise Lfm2CausalPostQkOutputFoldError(
            "The semantic stack contains no attention layers."
        )
    return stack, kinds, MappingProxyType(bindings)


def _authenticate_archive_loaded_candidates(
    *,
    candidates: Mapping[int, Lfm2CompiledAttentionGroupQdeim],
    manifest: Mapping[str, Any],
    expected_manifest_sha256: str,
    stack: Mapping[str, Any],
    bindings: Mapping[int, _AttentionStackBinding],
) -> Mapping[str, Any]:
    expected = _digest(
        expected_manifest_sha256,
        label="expected structural candidate archive manifest",
    )
    if not isinstance(candidates, Mapping) or not isinstance(manifest, Mapping):
        raise Lfm2CausalPostQkOutputFoldError(
            "Archive-loaded structural candidates and manifest must be mappings."
        )
    body = _deep_thaw(manifest)
    fresh_reload = body.pop("fresh_reload_verified", None)
    supplied = body.pop("manifest_sha256", None)
    indices = tuple(sorted(bindings))
    compiler_receipts = {
        str(index): _deep_thaw(bindings[index].compiler_receipt)
        for index in indices
    }
    if (
        fresh_reload is not True
        or body.get("schema")
        != LFM2_STRUCTURAL_ATTENTION_CANDIDATE_ARCHIVE_SCHEMA
        or supplied != expected
        or supplied != canonical_sha256(body)
        or body.get("semantic_layer_stack_result_sha256")
        != stack["result_sha256"]
        or body.get("attention_layer_indices") != list(indices)
        or body.get("attention_layer_count") != len(indices)
        or body.get("compiler_receipts") != compiler_receipts
        or set(candidates) != set(indices)
        or body.get("parent_weight_tensor_loads") != 0
        or body.get("parent_trace_recaptures") != 0
        or body.get("model_forward_calls") != 0
        or body.get("training_steps") != 0
    ):
        raise Lfm2CausalPostQkOutputFoldError(
            "The archive-loaded candidates are not freshly reloaded and bound "
            "to the independently pinned semantic stack."
        )
    records = body.get("tensor_records")
    candidate_states = body.get("candidate_state_sha256")
    if not isinstance(records, Mapping) or not isinstance(candidate_states, Mapping):
        raise Lfm2CausalPostQkOutputFoldError(
            "The structural candidate archive omitted tensor authentication."
        )
    for index in indices:
        candidate = candidates[index]
        try:
            _archive._authenticate_candidate(
                candidate,
                compiler_receipt=bindings[index].compiler_receipt,
                layer_index=index,
            )
        except Exception as exc:
            raise Lfm2CausalPostQkOutputFoldError(
                f"Layer {index} candidate geometry differs from its stack/archive "
                "lineage."
            ) from exc
        for name in _ARCHIVE_TENSOR_FIELDS:
            key = f"layer.{index:03d}.{name}"
            try:
                observed = _archive._tensor_record(
                    getattr(candidate, name),
                    label=key,
                )
            except Exception as exc:
                raise Lfm2CausalPostQkOutputFoldError(
                    f"Layer {index} archived geometry tensor {name} is invalid."
                ) from exc
            if records.get(key) != observed:
                raise Lfm2CausalPostQkOutputFoldError(
                    f"Layer {index} archived geometry/chart tensor {name} changed."
                )
        state = lfm2_attention_group_qdeim_candidate_state_sha256(
            candidate.query_weight,
            candidate.key_weight,
            candidate.value_weight,
            candidate.out_weight,
        )
        if candidate_states.get(str(index)) != state:
            raise Lfm2CausalPostQkOutputFoldError(
                f"Layer {index} structural Q/K/V/O state changed after archive load."
            )
        chart_sha = _tensor_sha256(candidate.residual_output_chart)
        if (
            bindings[index].residual_chart_receipt.get("chart_sha256") != chart_sha
            or bindings[index].compiler_receipt.get("authentication", {}).get(
                "residual_output_chart_sha256"
            )
            != chart_sha
        ):
            raise Lfm2CausalPostQkOutputFoldError(
                f"Layer {index} residual chart is not bound to its semantic stack."
            )
    return MappingProxyType(
        {**body, "manifest_sha256": supplied, "fresh_reload_verified": True}
    )


def _validate_live_topology(
    target_model: Any,
    kinds: Sequence[str],
) -> None:
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    if len(layers) != len(kinds):
        raise Lfm2CausalPostQkOutputFoldError(
            "The live Q/K child and authenticated cache layer counts diverge."
        )
    for index, (layer, kind) in enumerate(zip(layers, kinds, strict=True)):
        is_attention = getattr(layer, "is_attention_layer", None)
        if not isinstance(is_attention, bool) or is_attention != (
            kind == "full_attention"
        ):
            raise Lfm2CausalPostQkOutputFoldError(
                f"Live child layer {index} differs from the authenticated topology."
            )
    configured = tuple(
        getattr(getattr(target_model, "config", None), "layer_types", ())
    )
    if configured:
        normalized = tuple(
            "short_conv" if value in {"conv", "short_conv"} else str(value)
            for value in configured
        )
        if normalized != tuple(kinds):
            raise Lfm2CausalPostQkOutputFoldError(
                "The live child config differs from the authenticated topology."
            )


def _attention_hashes(target_model: Any, layer_index: int) -> Mapping[str, str]:
    parameters = _local._attention_parameters(target_model, layer_index)
    return MappingProxyType(
        {
            role: _tensor_sha256(parameters[role])
            for role in (*_QKV_ROLES, "out_proj.weight")
        }
    )


def _residual_bank_hashes(value: Mapping[int, Any]) -> Mapping[str, str]:
    return MappingProxyType(
        {
            str(index): _tensor_sha256(tensor)
            for index, tensor in sorted(value.items())
        }
    )


def _materialization_receipt(
    *,
    compiled: Any,
    installed: bool,
    prior_sha256: str,
    corrected_sha256: str,
) -> Mapping[str, Any]:
    if installed:
        return _post_qk._direction_materialization_receipt(compiled)
    if prior_sha256 != corrected_sha256:
        raise Lfm2CausalPostQkOutputFoldError(
            "An O decision reported no installation but changed target-dtype bytes."
        )
    body = {
        "schema": "neural-forge-lfm2-post-qk-o-materialization-noop-v1",
        "status": "authenticated-target-dtype-representational-noop",
        "structural_prior_sha256": prior_sha256,
        "corrected_out_weight_sha256": corrected_sha256,
        "effectful_target_dtype_materialization": False,
        "authenticated_representational_noop": True,
        "direction_preservation_not_applicable_to_zero_published_delta": True,
        "numeric_quality_gate_used": False,
    }
    return _sealed(body)


@dataclass(frozen=True, slots=True)
class Lfm2CausalPostQkOutputFoldInstallation:
    """Sealed whole-body result with one sealed causal receipt per O layer."""

    receipt: Mapping[str, Any]
    layer_receipts: tuple[Mapping[str, Any], ...] = field(init=False)

    def __post_init__(self) -> None:
        receipt = MappingProxyType(dict(self.receipt))
        object.__setattr__(self, "receipt", receipt)
        layers = receipt.get("attention_layer_receipts")
        if not isinstance(layers, list):
            raise Lfm2CausalPostQkOutputFoldError(
                "Causal O result omitted per-layer receipts."
            )
        object.__setattr__(
            self,
            "layer_receipts",
            tuple(MappingProxyType(dict(value)) for value in layers),
        )
        if not self.verify():
            raise Lfm2CausalPostQkOutputFoldError(
                "Causal post-Q/K O result failed authentication."
            )

    def verify(self) -> bool:
        try:
            outer = _verified_result(
                self.receipt,
                expected_schema=LFM2_CAUSAL_POST_QK_OUTPUT_FOLD_SCHEMA,
                label="causal post-Q/K O receipt",
            )
            indices = tuple(outer["attention_layer_indices"])
            if indices != tuple(sorted(indices)) or len(indices) != len(set(indices)):
                return False
            if len(self.layer_receipts) != len(indices):
                return False
            verified = tuple(
                _verified_result(
                    value,
                    expected_schema=LFM2_CAUSAL_POST_QK_OUTPUT_LAYER_SCHEMA,
                    label="causal post-Q/K O layer receipt",
                )
                for value in self.layer_receipts
            )
            raw_steps = outer.get("causal_replay_steps")
            if not isinstance(raw_steps, list) or not raw_steps:
                return False
            steps = tuple(
                _verified_result(
                    value,
                    expected_schema=LFM2_CAUSAL_POST_QK_REPLAY_STEP_SCHEMA,
                    label="causal post-Q/K replay step",
                )
                for value in raw_steps
            )
            changed = sorted(
                name
                for value in verified
                for name in value.get("changed_state_keys", ())
            )
            install_count = sum(
                int(value.get("o_decision_installed") is True) for value in verified
            )
            return bool(
                tuple(value["layer_index"] for value in verified) == indices
                and tuple(value["causal_attention_sequence_number"] for value in verified)
                == tuple(range(len(indices)))
                and all(
                    tuple(value.get("changed_state_keys", ()))
                    in {
                        (),
                        (
                            f"model.layers.{value['layer_index']}.self_attn."
                            "out_proj.weight",
                        ),
                    }
                    for value in verified
                )
                and outer["attention_layer_receipt_sha256"]
                == [value["result_sha256"] for value in verified]
                and outer["causal_replay_step_sha256"]
                == [value["result_sha256"] for value in steps]
                and tuple(value["layer_index"] for value in steps)
                == tuple(range(len(steps)))
                and outer["changed_state_keys"] == changed
                and outer["attention_o_decision_count"] == len(verified)
                and outer["attention_o_install_count"] == install_count
                and outer["attention_o_authenticated_noop_count"]
                == len(verified) - install_count
                and outer["actual_changed_state_keys_are_out_proj_only"] is True
                and outer["qkv_byte_identical_before_and_after"] is True
                and outer["causal_downstream_replay_after_each_o_install"] is True
            )
        except Exception:
            return False


def compile_install_lfm2_causal_post_qk_output_fold(
    *,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    donor_tokenizer: Any,
    target_tokenizer: Any,
    target_model: Any,
    qk_treatment_receipt: Mapping[str, Any],
    qk_fresh_reload_provenance_receipt: Mapping[str, Any],
    expected_qk_fresh_reload_provenance_sha256: str,
    semantic_layer_stack_receipt: Mapping[str, Any],
    expected_semantic_layer_stack_receipt_sha256: str,
    structural_candidates: Mapping[int, Lfm2CompiledAttentionGroupQdeim],
    structural_candidate_archive_manifest: Mapping[str, Any],
    expected_structural_candidate_archive_manifest_sha256: str,
    authenticated_cache_session: Any | None = None,
    ridge: float = 1.0e-6,
    fold_compute_device: str = "cpu",
) -> Lfm2CausalPostQkOutputFoldInstallation:
    """Causally compile/install every O from internally reconstructed evidence."""

    if not isinstance(cache, Lfm2CircuitTraceCache):
        raise TypeError("cache must be Lfm2CircuitTraceCache.")
    if not isinstance(bridge, StrictTokenizerBridgeArtifact):
        raise TypeError("bridge must be StrictTokenizerBridgeArtifact.")
    snapshot = _WholeModelSnapshot.capture(target_model)
    entry_hashes = _state_tensor_hashes(target_model)
    entry_runtime = _runtime_inventory(target_model)
    entry_topology = _local._topology_sha256(target_model)
    entry_count = _parameter_count(target_model)
    entry_identity = snapshot.fingerprint
    result: Lfm2CausalPostQkOutputFoldInstallation | None = None
    try:
        stack, kinds, bindings = _stack_and_attention_bindings(
            semantic_layer_stack_receipt=semantic_layer_stack_receipt,
            expected_semantic_layer_stack_receipt_sha256=(
                expected_semantic_layer_stack_receipt_sha256
            ),
            cache=cache,
            bridge=bridge,
        )
        _validate_live_topology(target_model, kinds)
        manifest = _authenticate_archive_loaded_candidates(
            candidates=structural_candidates,
            manifest=structural_candidate_archive_manifest,
            expected_manifest_sha256=(
                expected_structural_candidate_archive_manifest_sha256
            ),
            stack=stack,
            bindings=bindings,
        )
        attention_indices = tuple(sorted(bindings))
        provenance = _post_qk._verified_fresh_reload_provenance(
            qk_fresh_reload_provenance_receipt,
            expected_sha256=expected_qk_fresh_reload_provenance_sha256,
        )
        treatment_sha = _digest(
            provenance.get("target_lattice_qk_treatment_sha256"),
            label="fresh-reload target-lattice Q/K treatment",
        )
        (
            treatment,
            qk_installations,
            treated_identity,
            parameter_dtype,
            parameter_dtype_name,
        ) = _post_qk._authenticate_treated_qk_seed(
            target_model=target_model,
            treatment_receipt=qk_treatment_receipt,
            expected_treatment_receipt_sha256=treatment_sha,
            layer_indices=attention_indices,
        )
        if treated_identity != entry_identity:
            raise Lfm2CausalPostQkOutputFoldError(
                "The whole-body snapshot is not the authenticated fresh Q/K seed."
            )
        _post_qk._bind_fresh_reload_provenance_to_live_seed(
            provenance=provenance,
            treatment=treatment,
            target_model=target_model,
            current_identity=treated_identity,
            dtype_name=parameter_dtype_name,
        )
        qk_steps = _post_qk._layer_step_by_index(treatment)
        qkv_before = {
            index: {
                role: _attention_hashes(target_model, index)[role]
                for role in _QKV_ROLES
            }
            for index in attention_indices
        }
        cache_context = (
            cache.selective_authenticated_session()
            if authenticated_cache_session is None
            else nullcontext(authenticated_cache_session)
        )
        attention_receipts: list[Mapping[str, Any]] = []
        causal_steps: list[Mapping[str, Any]] = []
        with cache_context as session:
            alignments, tokenizer_session = _semantic_alignments_from_session(
                cache=cache,
                session=session,
                bridge=bridge,
                donor_tokenizer=donor_tokenizer,
                target_tokenizer=target_tokenizer,
            )
            refinement = dict(
                audit_lfm2_semantic_span_cpcf_refinement(
                    alignments,
                    expected_batch_count=cache.batch_count,
                )
            )
            if not refinement.get(
                "probability_preserving_attention_transport_go",
                False,
            ):
                raise Lfm2CausalPostQkOutputFoldError(
                    "The authenticated tokenizer bridge has no exact semantic "
                    "target lattice for causal O folding."
                )
            residual_bank = _initial_residual_bank(target_model, alignments)
            previous_state = treated_identity
            installed_o_keys: list[str] = []
            for layer_index, kind in enumerate(kinds):
                entry_replay = _replay_current_layer(
                    target_model,
                    alignments=alignments,
                    layer_index=layer_index,
                    pre_residual_by_batch=residual_bank,
                )
                if entry_replay.captured_child_model_state_sha256 != previous_state:
                    raise Lfm2CausalPostQkOutputFoldError(
                        f"Layer {layer_index} entry replay is not causally fresh."
                    )
                propagation = entry_replay
                attention_receipt_sha: str | None = None
                parent_trajectory_sha: list[str] = []
                if kind == "full_attention":
                    binding = bindings[layer_index]
                    parent = _load_parent_layer_trajectories(
                        session,
                        alignments=alignments,
                        layer_index=layer_index,
                        operator_kind="full_attention",
                    )
                    parent_trajectory_sha = [
                        parent[index].evidence_sha256 for index in sorted(parent)
                    ]
                    fit_cases, fit_receipt = (
                        _build_attention_source_local_output_fit_cases(
                            alignments=alignments,
                            parent=parent,
                            structural_replay=entry_replay,
                            fit_batch_indices=binding.fit_batch_indices,
                            fit_batch_sample_weights=(
                                binding.fit_batch_sample_weights
                            ),
                            layer_index=layer_index,
                        )
                    )
                    live_parameters = _local._attention_parameters(
                        target_model,
                        layer_index,
                    )
                    live_before = _attention_hashes(target_model, layer_index)
                    candidate = structural_candidates[layer_index]
                    treated_candidate = (
                        _post_qk._adapt_structural_candidate_to_treated_qk(
                            structural_candidate=candidate,
                            live_parameters=live_parameters,
                            treatment_result_sha256=treatment["result_sha256"],
                            qk_installation=qk_installations[layer_index],
                            model_identity_sha256=previous_state,
                            dtype=parameter_dtype,
                            dtype_name=parameter_dtype_name,
                        )
                    )
                    installation = (
                        compile_install_lfm2_semantic_attention_source_local_output(
                            target_model=target_model,
                            layer_index=layer_index,
                            structural_candidate=treated_candidate,
                            fit_cases=fit_cases,
                            ridge=ridge,
                            fold_compute_device=fold_compute_device,
                        )
                    )
                    low = _verified_result(
                        installation.receipt,
                        expected_schema=(
                            LFM2_SEMANTIC_ATTENTION_SOURCE_LOCAL_OUTPUT_INSTALLATION_SCHEMA
                        ),
                        label=f"layer {layer_index} source-local O installation",
                    )
                    compiler = _verified_result(
                        installation.compiled.report,
                        expected_schema=(
                            LFM2_ATTENTION_GROUP_QDEIM_SOURCE_LOCAL_OUTPUT_RESIDUAL_SCHEMA
                        ),
                        label=f"layer {layer_index} source-local O compiler",
                    )
                    output_key = (
                        f"model.layers.{layer_index}.self_attn.out_proj.weight"
                    )
                    live_after = _attention_hashes(target_model, layer_index)
                    corrected_sha = _tensor_sha256(
                        installation.compiled.out_weight_source_local_corrected
                    )
                    prior_sha = _tensor_sha256(
                        installation.compiled.out_weight_structural_prior
                    )
                    changed_keys = tuple(low.get("changed_state_keys", ()))
                    installed = changed_keys == (output_key,)
                    direction = _materialization_receipt(
                        compiled=installation.compiled,
                        installed=installed,
                        prior_sha256=prior_sha,
                        corrected_sha256=corrected_sha,
                    )
                    if (
                        low.get("model_state_before_sha256") != previous_state
                        or changed_keys not in {(), (output_key,)}
                        or low.get("installed") is not installed
                        or low.get("install_count") != int(installed)
                        or low.get("only_out_projection_permitted_to_change")
                        is not True
                        or low.get("qkv_bit_identical_to_structural_candidate")
                        is not True
                        or compiler.get("authentication", {}).get(
                            "fit_evidence_sha256_before"
                        )
                        != fit_receipt["fit_evidence_sha256"]
                        or prior_sha != live_before["out_proj.weight"]
                        or corrected_sha != live_after["out_proj.weight"]
                        or any(
                            live_after[role] != live_before[role]
                            for role in _QKV_ROLES
                        )
                    ):
                        raise Lfm2CausalPostQkOutputFoldError(
                            f"Layer {layer_index} did not publish exactly one "
                            "cache-derived O correction."
                        )
                    propagation = _replay_current_layer(
                        target_model,
                        alignments=alignments,
                        layer_index=layer_index,
                        pre_residual_by_batch=residual_bank,
                    )
                    if (
                        propagation.captured_child_model_state_sha256
                        != low["model_state_after_sha256"]
                    ):
                        raise Lfm2CausalPostQkOutputFoldError(
                            f"Layer {layer_index} post-O replay is stale."
                        )
                    geometry_sha = {
                        name: _tensor_sha256(getattr(candidate, name))
                        for name in _ARCHIVE_TENSOR_FIELDS[4:]
                    }
                    layer_body = {
                        "schema": LFM2_CAUSAL_POST_QK_OUTPUT_LAYER_SCHEMA,
                        "layer_index": layer_index,
                        "causal_attention_sequence_number": len(
                            attention_receipts
                        ),
                        "semantic_layer_stack_result_sha256": stack[
                            "result_sha256"
                        ],
                        "structural_candidate_archive_manifest_sha256": manifest[
                            "manifest_sha256"
                        ],
                        "pre_qk_structural_compiler_result_sha256": binding.compiler_receipt[
                            "result_sha256"
                        ],
                        "pre_qk_structural_candidate_state_sha256": (
                            lfm2_attention_group_qdeim_candidate_state_sha256(
                                candidate.query_weight,
                                candidate.key_weight,
                                candidate.value_weight,
                                candidate.out_weight,
                            )
                        ),
                        "archive_geometry_and_chart_sha256": geometry_sha,
                        "residual_chart_receipt_sha256": binding.residual_chart_receipt[
                            "result_sha256"
                        ],
                        "qk_treatment_result_sha256": treatment["result_sha256"],
                        "qk_layer_installation_result_sha256": qk_installations[
                            layer_index
                        ]["result_sha256"],
                        "qk_layer_step_result_sha256": qk_steps[layer_index][
                            "result_sha256"
                        ],
                        "entry_replay_bank_sha256": entry_replay.bank_sha256,
                        "entry_replay_model_state_sha256": entry_replay.captured_child_model_state_sha256,
                        "parent_trajectory_evidence_sha256": parent_trajectory_sha,
                        "fit_cases_receipt": dict(fit_receipt),
                        "fit_cases_receipt_sha256": fit_receipt["result_sha256"],
                        "fit_rows_derived_inside_transaction": True,
                        "caller_supplied_fit_or_replay_tensors": False,
                        "treated_structural_candidate_result_sha256": (
                            treated_candidate.report["result_sha256"]
                        ),
                        "source_local_compiler_receipt": dict(compiler),
                        "source_local_compiler_result_sha256": compiler[
                            "result_sha256"
                        ],
                        "source_local_installation_receipt": dict(low),
                        "source_local_installation_result_sha256": low[
                            "result_sha256"
                        ],
                        "direction_materialization_receipt": dict(direction),
                        "direction_materialization_result_sha256": direction[
                            "result_sha256"
                        ],
                        "model_state_before_sha256": previous_state,
                        "model_state_after_sha256": low[
                            "model_state_after_sha256"
                        ],
                        "changed_state_keys": list(changed_keys),
                        "post_o_replay_bank_sha256": propagation.bank_sha256,
                        "post_o_replay_model_state_sha256": propagation.captured_child_model_state_sha256,
                        "propagated_residual_sha256": dict(
                            _residual_bank_hashes(_next_residual_bank(propagation))
                        ),
                        "qkv_byte_identical_before_and_after": True,
                        "o_decision_installed": installed,
                        "o_install_count": int(installed),
                        "authenticated_o_representational_noop": not installed,
                        "only_out_projection_permitted_to_change": True,
                        "parameter_dtype": parameter_dtype_name,
                        "checkpoint_dtype_preserved": True,
                        "archive_geometry_chart_bound_to_semantic_stack": True,
                        "cached_parent_rows_loaded_internally": True,
                        "fresh_live_child_replay_built_internally": True,
                        "post_install_replay_precedes_downstream_layer": True,
                        "parent_model_objects_constructed": 0,
                        "parent_trace_recaptures": 0,
                        "model_forward_calls": 0,
                        "training_steps": 0,
                    }
                    layer_receipt = _sealed(layer_body)
                    attention_receipts.append(layer_receipt)
                    attention_receipt_sha = layer_receipt["result_sha256"]
                    if installed:
                        installed_o_keys.append(output_key)
                    previous_state = low["model_state_after_sha256"]
                residual_bank = _next_residual_bank(propagation)
                step_body = {
                    "schema": LFM2_CAUSAL_POST_QK_REPLAY_STEP_SCHEMA,
                    "layer_index": layer_index,
                    "operator_kind": kind,
                    "entry_replay_bank_sha256": entry_replay.bank_sha256,
                    "entry_model_state_sha256": entry_replay.captured_child_model_state_sha256,
                    "attention_o_receipt_sha256": attention_receipt_sha,
                    "propagation_replay_bank_sha256": propagation.bank_sha256,
                    "propagation_model_state_sha256": propagation.captured_child_model_state_sha256,
                    "propagated_residual_sha256": dict(
                        _residual_bank_hashes(residual_bank)
                    ),
                    "upstream_installed_o_state_keys": list(installed_o_keys),
                    "uses_all_preceding_accepted_o_mutations": True,
                }
                causal_steps.append(_sealed(step_body))

        after_hashes = _state_tensor_hashes(target_model)
        changed = _changed_state_keys(entry_hashes, after_hashes)
        expected_changed = tuple(sorted(installed_o_keys))
        qkv_after = {
            index: {
                role: _attention_hashes(target_model, index)[role]
                for role in _QKV_ROLES
            }
            for index in attention_indices
        }
        final_identity = fingerprint_lfm2_parent(target_model).identity_sha256
        if (
            len(attention_receipts) != len(attention_indices)
            or tuple(receipt["layer_index"] for receipt in attention_receipts)
            != attention_indices
            or changed != expected_changed
            or qkv_after != qkv_before
            or (bool(changed) and final_identity == entry_identity)
            or (not changed and final_identity != entry_identity)
            or _local._topology_sha256(target_model) != entry_topology
            or _parameter_count(target_model) != entry_count
            or _parameter_topology(target_model) != snapshot.topology
            or _runtime_inventory(target_model) != entry_runtime
            or tuple(
                (str(name), bool(module.training))
                for name, module in target_model.named_modules()
            )
            != snapshot.modes
            or any(
                parameter.dtype != parameter_dtype
                for parameter in target_model.parameters()
            )
        ):
            raise Lfm2CausalPostQkOutputFoldError(
                "The causal whole-body stage changed non-O state, lost an O "
                "effect, or altered dtype/topology/identity/storage/gradients."
            )
        body = {
            "schema": LFM2_CAUSAL_POST_QK_OUTPUT_FOLD_SCHEMA,
            "status": "completed-cache-derived-causal-post-qk-attention-o",
            "cache_key": cache.cache_key,
            "bridge_artifact_sha256": bridge.artifact_sha256,
            "tokenizer_session_sha256": tokenizer_session.get("session_sha256"),
            "semantic_refinement_sha256": refinement.get("audit_sha256"),
            "semantic_layer_stack_result_sha256": stack["result_sha256"],
            "semantic_layer_stack_independently_pinned": True,
            "structural_candidate_archive_manifest_sha256": manifest[
                "manifest_sha256"
            ],
            "structural_candidate_archive_fresh_reload_verified": True,
            "qk_treatment_result_sha256": treatment["result_sha256"],
            "qk_fresh_reload_provenance_sha256": provenance["receipt_sha256"],
            "qk_fresh_reload_provenance_independently_pinned": True,
            "model_state_before_sha256": entry_identity,
            "model_state_after_sha256": final_identity,
            "parameter_dtype": parameter_dtype_name,
            "checkpoint_dtype_preserved": True,
            "attention_layer_indices": list(attention_indices),
            "attention_layer_count": len(attention_indices),
            "attention_o_decision_count": len(attention_receipts),
            "attention_o_install_count": len(installed_o_keys),
            "attention_o_authenticated_noop_count": (
                len(attention_receipts) - len(installed_o_keys)
            ),
            "attention_layer_receipt_sha256": [
                value["result_sha256"] for value in attention_receipts
            ],
            "attention_layer_receipts": [dict(value) for value in attention_receipts],
            "causal_replay_step_sha256": [
                value["result_sha256"] for value in causal_steps
            ],
            "causal_replay_steps": [dict(value) for value in causal_steps],
            "changed_state_keys": list(changed),
            "allowed_changed_state_keys": [
                f"model.layers.{index}.self_attn.out_proj.weight"
                for index in attention_indices
            ],
            "actual_changed_state_keys_are_out_proj_only": True,
            "all_attention_o_decisions_completed": True,
            "authenticated_representational_noops_permitted": True,
            "qkv_tensor_sha256_before": qkv_before,
            "qkv_tensor_sha256_after": qkv_after,
            "qkv_byte_identical_before_and_after": True,
            "target_topology_sha256_before": entry_topology,
            "target_topology_sha256_after": _local._topology_sha256(target_model),
            "parameter_count_before": entry_count,
            "parameter_count_after": _parameter_count(target_model),
            "parameter_count_changed": False,
            "parameter_identities_preserved": True,
            "parameter_storage_preserved": True,
            "gradient_state_preserved": True,
            "module_modes_preserved": True,
            "one_whole_body_snapshot_and_rollback_boundary": True,
            "cache_session_owned_by_stage": authenticated_cache_session is None,
            "cache_session_reused_from_pipeline": authenticated_cache_session
            is not None,
            "cached_parent_rows_loaded_internally": True,
            "caller_supplied_parent_or_child_fit_tensors": False,
            "caller_supplied_replay_digests": False,
            "child_macrocell_replay_only": True,
            "causal_downstream_replay_after_each_o_install": True,
            "numeric_metrics_have_semantic_decision_authority": False,
            "whole_decoded_behavior_is_primary_intelligence_evidence": True,
            "parent_model_objects_constructed": 0,
            "parent_model_forward_calls": 0,
            "parent_trace_recaptures": 0,
            "child_model_forward_calls": 0,
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "pruning_operations": 0,
            "quantization_operations": 0,
            "sparsity_used": False,
            "parameter_sharing_used": False,
        }
        result = Lfm2CausalPostQkOutputFoldInstallation(receipt=_sealed(body))
    except BaseException as exc:
        try:
            snapshot.restore(target_model)
        except BaseException as rollback_exc:
            raise Lfm2CausalPostQkOutputFoldError(
                "Causal post-Q/K O folding failed and exact whole-body rollback "
                "could not be proven."
            ) from rollback_exc
        raise Lfm2CausalPostQkOutputFoldError(
            "Causal post-Q/K O folding failed; exact entry state was restored."
        ) from exc
    if result is None:  # pragma: no cover - defensive totality guard
        raise AssertionError("Causal post-Q/K O folding returned no result.")
    return result


__all__ = [
    "LFM2_CAUSAL_POST_QK_OUTPUT_FOLD_SCHEMA",
    "LFM2_CAUSAL_POST_QK_OUTPUT_LAYER_SCHEMA",
    "LFM2_CAUSAL_POST_QK_REPLAY_STEP_SCHEMA",
    "Lfm2CausalPostQkOutputFoldError",
    "Lfm2CausalPostQkOutputFoldInstallation",
    "compile_install_lfm2_causal_post_qk_output_fold",
]
