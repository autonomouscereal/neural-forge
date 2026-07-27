"""Cache-derived source-local global reclosure after causal Q/K/O folding.

This stage starts from the exact live child published by the causal post-Q/K
attention-output fold.  It derives semantic alignments internally, loads only
authenticated cached parent trajectories, and analytically replays the live
child one macrocell at a time.  The resulting whole-stack evidence is passed
to the output-writer-only source-local closure.  Attention Q/K/V, convolution
inputs, MLP gate/up projections, and the live terminal W2 are outside the
publication set and are required to remain byte-identical.

There is no parent model construction, trace capture, model forward,
generation, autograd, optimizer, training, pruning, quantization, sparsity, or
parameter sharing in this stage.  One whole-model snapshot covers evidence
construction and publication; every failure restores the exact entry state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_causal_post_qk_output_fold import (
    LFM2_CAUSAL_POST_QK_OUTPUT_FOLD_SCHEMA,
    Lfm2CausalPostQkOutputFoldInstallation,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceCache,
    fingerprint_lfm2_parent,
)
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (
    Lfm2CpcfTerminalActionDownWeightPrior,
    lfm2_cpcf_terminal_action_tensor_sha256,
)
from neural_forge.hf_lfm2_postfold_terminal_w2_reclosure import (
    LFM2_POSTFOLD_TERMINAL_W2_REBASE_SCHEMA,
    Lfm2PostfoldTerminalW2Rebase,
)
from neural_forge.hf_lfm2_postfold_reclosure_seed_archive import (
    _foundation_tensor_sha256,
)
from neural_forge.hf_lfm2_semantic_global_residual_integration import (
    LFM2_SEMANTIC_SOURCE_LOCAL_OUTPUT_GLOBAL_RESIDUAL_INTEGRATION_SCHEMA,
    Lfm2SemanticGlobalResidualEvidencePackage,
    Lfm2SemanticGlobalResidualInstallation,
    compile_install_lfm2_semantic_source_local_output_residual_closure,
)
from neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter import (
    LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA,
    _WholeModelSnapshot,
    _anchor_interface_tensor_sha256,
    _build_global_residual_evidence_from_accumulators,
    _capture_global_child_layer_primals,
    _capture_global_parent_layer_boundaries,
    _fit_residual_chart,
    _initial_residual_bank,
    _load_parent_layer_trajectories,
    _next_residual_bank,
    _parameter_topology,
    _prepare_global_residual_accumulators,
    _replay_current_layer,
)
from neural_forge.hf_lfm2_semantic_postfold import (
    _changed_state_keys,
    _interface_tensor_sha256,
    _parameter_count,
    _semantic_alignments_from_session,
    _state_tensor_hashes,
    _tensor_sha256,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    lfm2_authenticated_tensor_sha256,
)
from neural_forge.hf_lfm2_semantic_span_cpcf_alignment import (
    audit_lfm2_semantic_span_cpcf_refinement,
)
from neural_forge.hf_tokenizer_bridge import StrictTokenizerBridgeArtifact


LFM2_POST_CAUSAL_O_GLOBAL_RECLOSURE_SCHEMA = (
    "neural-forge-lfm2-post-causal-o-source-local-global-reclosure-v1"
)
LFM2_POST_CAUSAL_O_GLOBAL_REPLAY_STEP_SCHEMA = (
    "neural-forge-lfm2-post-causal-o-global-replay-step-v1"
)


class Lfm2PostCausalOGlobalReclosureError(RuntimeError):
    """Post-causal-O evidence or publication failed closed and rolled back."""


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
        raise Lfm2PostCausalOGlobalReclosureError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _sealed(body: Mapping[str, Any]) -> Mapping[str, Any]:
    plain = _deep_thaw(body)
    return MappingProxyType({**plain, "result_sha256": canonical_sha256(plain)})


def _verified_result(
    value: Any,
    *,
    expected_sha256: str,
    label: str,
    expected_schema: str | None = None,
) -> Mapping[str, Any]:
    expected = _digest(expected_sha256, label=f"expected {label}")
    if not isinstance(value, Mapping):
        raise Lfm2PostCausalOGlobalReclosureError(
            f"{label} must be one sealed mapping."
        )
    body = _deep_thaw(value)
    supplied = body.pop("result_sha256", None)
    if (
        supplied != expected
        or supplied != canonical_sha256(body)
        or (expected_schema is not None and body.get("schema") != expected_schema)
    ):
        raise Lfm2PostCausalOGlobalReclosureError(
            f"{label} is stale, malformed, or not independently pinned."
        )
    return MappingProxyType({**body, "result_sha256": supplied})


def _verified_coordinate_receipt(
    value: Any,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[Mapping[str, Any], str]:
    """Verify either a compiler report or an ordinary sealed result."""

    expected = _digest(expected_sha256, label=f"expected {label}")
    if not isinstance(value, Mapping):
        raise Lfm2PostCausalOGlobalReclosureError(
            f"{label} must be one sealed mapping."
        )
    plain = _deep_thaw(value)
    present = tuple(
        field
        for field in ("manifest_sha256", "report_sha256", "result_sha256")
        if field in plain
    )
    if len(present) != 1:
        raise Lfm2PostCausalOGlobalReclosureError(
            f"{label} must have exactly one recognized top-level seal."
        )
    seal_field = present[0]
    supplied = plain.pop(seal_field)
    fresh_reload_verified = plain.pop("fresh_reload_verified", None)
    if seal_field == "manifest_sha256" and fresh_reload_verified is not True:
        raise Lfm2PostCausalOGlobalReclosureError(
            f"{label} archive was not freshly reloaded and authenticated."
        )
    if seal_field != "manifest_sha256" and fresh_reload_verified is not None:
        raise Lfm2PostCausalOGlobalReclosureError(
            f"{label} has an invalid freshness wrapper."
        )
    if supplied != expected or supplied != canonical_sha256(plain):
        raise Lfm2PostCausalOGlobalReclosureError(
            f"{label} is stale, malformed, or not independently pinned."
        )
    return MappingProxyType(
        {
            **plain,
            seal_field: supplied,
            **(
                {"fresh_reload_verified": True}
                if seal_field == "manifest_sha256"
                else {}
            ),
        }
    ), seal_field


def _coordinate_hashes(value: Any) -> frozenset[str]:
    """Recognize the tensor seals used by current interface artifacts."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Post-causal-O reclosure requires PyTorch.") from exc
    if not isinstance(value, torch.Tensor):
        raise Lfm2PostCausalOGlobalReclosureError(
            "Coordinate evidence must be one materialized Tensor."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    return frozenset(
        {
            _tensor_sha256(tensor),
            _interface_tensor_sha256(tensor),
            _anchor_interface_tensor_sha256(tensor),
            _foundation_tensor_sha256(tensor),
            lfm2_authenticated_tensor_sha256(tensor),
            _interface_tensor_sha256(tensor.to(dtype=torch.float64)),
        }
    )


def lfm2_post_causal_o_global_coordinate_sha256(value: Any) -> str:
    """Return the archive tensor pin consumed by this public seam."""

    return lfm2_authenticated_tensor_sha256(value)


def _bind_coordinate(
    value: Any,
    *,
    expected_tensor_sha256: str,
    receipt: Mapping[str, Any],
    expected_receipt_sha256: str,
    claim_fields: Sequence[str],
    receipt_tensor_record_name: str | None,
    label: str,
) -> tuple[Mapping[str, Any], str, str]:
    expected_tensor = _digest(
        expected_tensor_sha256,
        label=f"expected {label} tensor",
    )
    raw_sha = lfm2_post_causal_o_global_coordinate_sha256(value)
    if raw_sha != expected_tensor:
        raise Lfm2PostCausalOGlobalReclosureError(
            f"{label} tensor differs from its independent pin."
        )
    verified, seal_field = _verified_coordinate_receipt(
        receipt,
        expected_sha256=expected_receipt_sha256,
        label=f"{label} receipt",
    )
    if seal_field == "manifest_sha256":
        records = verified.get("tensor_records")
        record = (
            records.get(receipt_tensor_record_name)
            if isinstance(records, Mapping)
            and isinstance(receipt_tensor_record_name, str)
            else None
        )
        if not isinstance(record, Mapping) or record.get("sha256") != raw_sha:
            raise Lfm2PostCausalOGlobalReclosureError(
                f"{label} differs from its freshly reloaded archive tensor record."
            )
    claims = tuple(
        str(verified[field])
        for field in claim_fields
        if isinstance(verified.get(field), str)
    )
    tensor_hashes = _coordinate_hashes(value)
    matching = tuple(claim for claim in claims if claim in tensor_hashes)
    if len(set(matching)) != 1:
        raise Lfm2PostCausalOGlobalReclosureError(
            f"{label} receipt does not uniquely authenticate the supplied tensor."
        )
    return verified, seal_field, matching[0]


def _coordinate_receipt_interface_report_sha256(
    receipt: Mapping[str, Any],
    seal_field: str,
) -> str:
    linked = receipt.get("foundation_active_interface_report_sha256")
    if isinstance(linked, str):
        return _digest(linked, label="coordinate foundation interface report")
    if seal_field == "report_sha256":
        return _digest(receipt[seal_field], label="coordinate interface report")
    raise Lfm2PostCausalOGlobalReclosureError(
        "Coordinate receipt does not bind the active foundation interface."
    )


def _live_interface_target_table(
    target_model: Any,
    *,
    interface_lineage: Mapping[str, Any],
) -> tuple[Any, str]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Post-causal-O reclosure requires PyTorch.") from exc
    input_weight = getattr(target_model.get_input_embeddings(), "weight", None)
    output_weight = getattr(target_model.get_output_embeddings(), "weight", None)
    target_claim = _digest(
        interface_lineage.get("target_weight_sha256"),
        label="live interface target lineage",
    )
    if (
        not isinstance(input_weight, torch.nn.Parameter)
        or not isinstance(output_weight, torch.nn.Parameter)
        or output_weight is input_weight
        or int(output_weight.untyped_storage().data_ptr())
        == int(input_weight.untyped_storage().data_ptr())
        or input_weight.ndim != 2
        or output_weight.ndim != 2
        or tuple(output_weight.shape) != tuple(input_weight.shape)
        or getattr(
            getattr(target_model, "config", None),
            "tie_word_embeddings",
            None,
        )
        is True
        or interface_lineage.get("target_weight_matches_live_input_child") is not True
        or interface_lineage.get("interface_runtime_mode") != "independent_output"
        or interface_lineage.get("live_output_parameter_independent") is not True
        or interface_lineage.get(
            "output_table_matches_structural_input_at_body_entry"
        )
        is not True
        or not torch.equal(
            output_weight.detach().to(device="cpu", dtype=input_weight.dtype),
            input_weight.detach().to(device="cpu"),
        )
    ):
        raise Lfm2PostCausalOGlobalReclosureError(
            "Live independent input/output tables do not reproduce the "
            "authenticated child-coordinate interface target."
        )
    return input_weight.detach().to(device="cpu").contiguous().clone(), target_claim


def _runtime_inventory(model: Any) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            str(name): {
                "parameter_id": id(parameter),
                "storage_pointer": int(parameter.data_ptr()),
                "requires_grad": bool(parameter.requires_grad),
                "gradient_id": None if parameter.grad is None else id(parameter.grad),
                "gradient_sha256": (
                    None if parameter.grad is None else _tensor_sha256(parameter.grad)
                ),
            }
            for name, parameter in model.named_parameters()
        }
    )


def _model_modes(model: Any) -> tuple[tuple[str, bool], ...]:
    return tuple(
        (str(name), bool(module.training)) for name, module in model.named_modules()
    )


def _live_topology(
    target_model: Any,
    *,
    stack: Mapping[str, Any],
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
) -> tuple[str, ...]:
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    kinds = tuple(stack.get("operator_layer_kinds", ()))
    live = tuple(
        "full_attention" if bool(getattr(layer, "is_attention_layer", False))
        else "short_conv"
        for layer in layers
    )
    attention = tuple(
        index for index, kind in enumerate(kinds) if kind == "full_attention"
    )
    short_conv = tuple(
        index for index, kind in enumerate(kinds) if kind == "short_conv"
    )
    if (
        not layers
        or kinds != live
        or any(kind not in {"full_attention", "short_conv"} for kind in kinds)
        or stack.get("layer_count") != len(layers)
        or stack.get("terminal_layer_index") != len(layers) - 1
        or tuple(stack.get("attention_layer_indices", ())) != attention
        or tuple(stack.get("short_conv_layer_indices", ())) != short_conv
        or stack.get("cache_key") != cache.cache_key
        or stack.get("cache_parent_fingerprint") != cache.parent_fingerprint
        or stack.get("cache_layer_topology_sha256")
        != cache.layer_topology_fingerprint
        or stack.get("bridge_artifact_sha256") != bridge.artifact_sha256
        or stack.get("complete_causal_layer_stack_adjudicated") is not True
        or stack.get("final_layer_mlp_w2_deferred_to_fused_terminal") is not True
        or stack.get("final_layer_mlp_w2_install_count") != 0
    ):
        raise Lfm2PostCausalOGlobalReclosureError(
            "Semantic stack, cache, bridge, and live child topology disagree."
        )
    return kinds


def _terminal_parameter(
    target_model: Any,
    *,
    rebase: Lfm2PostfoldTerminalW2Rebase,
    stack_sha256: str,
    expected_rebase_sha256: str,
    expected_prior_sha256: str,
) -> tuple[Any, Mapping[str, Any], Lfm2CpcfTerminalActionDownWeightPrior]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Post-causal-O reclosure requires PyTorch.") from exc
    if not isinstance(rebase, Lfm2PostfoldTerminalW2Rebase) or not rebase.verify():
        raise Lfm2PostCausalOGlobalReclosureError(
            "Terminal W2 rebase is not authenticated."
        )
    receipt = _verified_result(
        rebase.receipt,
        expected_sha256=expected_rebase_sha256,
        expected_schema=LFM2_POSTFOLD_TERMINAL_W2_REBASE_SCHEMA,
        label="terminal W2 rebase receipt",
    )
    prior = rebase.terminal_prior
    expected_prior = _digest(
        expected_prior_sha256,
        label="expected terminal prior authentication",
    )
    name = receipt.get("final_w2_parameter_name")
    parameter = dict(target_model.named_parameters()).get(name)
    if (
        not isinstance(prior, Lfm2CpcfTerminalActionDownWeightPrior)
        or not prior.verify_authentication()
        or prior.authentication_sha256 != expected_prior
        or receipt.get("semantic_layer_stack_result_sha256") != stack_sha256
        or receipt.get("terminal_prior_authentication_sha256") != expected_prior
        or receipt.get("terminal_prior_tensor_sha256")
        != lfm2_cpcf_terminal_action_tensor_sha256(prior.down_weight)
        or receipt.get("restored_live_w2_sha256")
        != prior.live_deferred_w2_sha256
        or not isinstance(parameter, torch.nn.Parameter)
        or lfm2_cpcf_terminal_action_tensor_sha256(parameter)
        != prior.live_deferred_w2_sha256
    ):
        raise Lfm2PostCausalOGlobalReclosureError(
            "Terminal prior, rebase receipt, semantic stack, and live W2 disagree."
        )
    return parameter, receipt, prior


def _residual_hashes(bank: Mapping[int, Any]) -> Mapping[str, str]:
    return MappingProxyType(
        {str(index): _tensor_sha256(value) for index, value in sorted(bank.items())}
    )


@dataclass(frozen=True, slots=True)
class Lfm2PostCausalOGlobalReclosureInstallation:
    """Authenticated output-writer reclosure and deferred terminal handoff."""

    installation: Lfm2SemanticGlobalResidualInstallation = field(
        repr=False,
        compare=False,
    )
    evidence: Lfm2SemanticGlobalResidualEvidencePackage = field(
        repr=False,
        compare=False,
    )
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))
        if not self.verify():
            raise Lfm2PostCausalOGlobalReclosureError(
                "Post-causal-O global reclosure result failed authentication."
            )

    @property
    def corrected_terminal_w2_prior(self) -> Lfm2CpcfTerminalActionDownWeightPrior:
        return self.installation.corrected_final_w2_prior

    def verify(self) -> bool:
        try:
            supplied = self.receipt.get("result_sha256")
            receipt = _verified_result(
                self.receipt,
                expected_sha256=supplied,
                expected_schema=LFM2_POST_CAUSAL_O_GLOBAL_RECLOSURE_SCHEMA,
                label="post-causal-O global reclosure receipt",
            )
            raw_steps = receipt.get("replay_steps")
            if not isinstance(raw_steps, list):
                return False
            steps = tuple(
                _verified_result(
                    value,
                    expected_sha256=value.get("result_sha256"),
                    expected_schema=LFM2_POST_CAUSAL_O_GLOBAL_REPLAY_STEP_SCHEMA,
                    label="post-causal-O global replay step",
                )
                for value in raw_steps
                if isinstance(value, Mapping)
            )
            changed = tuple(receipt.get("changed_state_keys", ()))
            allowed = tuple(receipt.get("allowed_changed_state_keys", ()))
            _digest(
                receipt.get("tokenizer_session_sha256"),
                label="tokenizer session SHA-256",
            )
            _digest(
                receipt.get("semantic_refinement_sha256"),
                label="semantic refinement SHA-256",
            )
            _digest(
                receipt.get("cache_session_sha256"),
                label="cache session SHA-256",
            )
            return bool(
                isinstance(
                    self.installation,
                    Lfm2SemanticGlobalResidualInstallation,
                )
                and self.installation.verify_authentication()
                and isinstance(
                    self.evidence,
                    Lfm2SemanticGlobalResidualEvidencePackage,
                )
                and self.evidence.verify_authentication()
                and receipt.get("global_residual_evidence_sha256")
                == self.evidence.evidence_sha256
                and receipt.get("global_reclosure_report_sha256")
                == self.installation.report.get("report_sha256")
                and receipt.get("corrected_terminal_w2_prior_sha256")
                == self.installation.corrected_final_w2_prior.authentication_sha256
                and len(steps) == len(raw_steps)
                and tuple(value.get("layer_index") for value in steps)
                == tuple(range(len(steps)))
                and receipt.get("causal_layer_indices") == list(range(len(steps)))
                and receipt.get("replay_step_sha256")
                == [value["result_sha256"] for value in steps]
                and tuple(sorted(changed)) == changed
                and tuple(sorted(allowed)) == allowed
                and set(changed).issubset(set(allowed))
                and receipt.get("changed_state_keys_are_output_writers_only") is True
                and receipt.get("protected_producer_tensors_byte_identical") is True
                and receipt.get("live_terminal_w2_unchanged_and_deferred") is True
                and receipt.get(
                    "current_live_child_replayed_in_causal_layer_order"
                )
                is True
            )
        except Exception:
            return False


def compile_install_lfm2_post_causal_o_source_local_global_reclosure(
    *,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    donor_tokenizer: Any,
    target_tokenizer: Any,
    target_model: Any,
    causal_o_installation: Lfm2CausalPostQkOutputFoldInstallation,
    expected_causal_o_receipt_sha256: str,
    semantic_layer_stack_receipt: Mapping[str, Any],
    expected_semantic_layer_stack_receipt_sha256: str,
    mapped_parent_tied_table: Any,
    mapped_parent_tied_table_receipt: Mapping[str, Any],
    expected_mapped_parent_tied_table_sha256: str,
    expected_mapped_parent_tied_table_receipt_sha256: str,
    producer_encoder: Any,
    producer_encoder_receipt: Mapping[str, Any],
    expected_producer_encoder_sha256: str,
    expected_producer_encoder_receipt_sha256: str,
    terminal_rebase: Lfm2PostfoldTerminalW2Rebase,
    expected_terminal_rebase_receipt_sha256: str,
    expected_terminal_prior_authentication_sha256: str,
    authenticated_cache_session: Any | None = None,
    closure_policy: Any | None = None,
) -> Lfm2PostCausalOGlobalReclosureInstallation:
    """Build cached whole-stack evidence and reclose output writers only."""

    if not isinstance(cache, Lfm2CircuitTraceCache):
        raise TypeError("cache must be Lfm2CircuitTraceCache.")
    if not isinstance(bridge, StrictTokenizerBridgeArtifact):
        raise TypeError("bridge must be StrictTokenizerBridgeArtifact.")
    snapshot = _WholeModelSnapshot.capture(target_model)
    entry_hashes = _state_tensor_hashes(target_model)
    entry_runtime = _runtime_inventory(target_model)
    entry_modes = _model_modes(target_model)
    entry_identity = snapshot.fingerprint
    entry_count = _parameter_count(target_model)
    result: Lfm2PostCausalOGlobalReclosureInstallation | None = None
    try:
        stack = _verified_result(
            semantic_layer_stack_receipt,
            expected_sha256=expected_semantic_layer_stack_receipt_sha256,
            expected_schema=LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA,
            label="semantic layer-stack receipt",
        )
        kinds = _live_topology(
            target_model,
            stack=stack,
            cache=cache,
            bridge=bridge,
        )
        if (
            not isinstance(
                causal_o_installation,
                Lfm2CausalPostQkOutputFoldInstallation,
            )
            or not causal_o_installation.verify()
        ):
            raise Lfm2PostCausalOGlobalReclosureError(
                "Causal post-Q/K O installation is not authenticated."
            )
        causal = _verified_result(
            causal_o_installation.receipt,
            expected_sha256=expected_causal_o_receipt_sha256,
            expected_schema=LFM2_CAUSAL_POST_QK_OUTPUT_FOLD_SCHEMA,
            label="causal post-Q/K O receipt",
        )
        if (
            causal.get("semantic_layer_stack_result_sha256")
            != stack["result_sha256"]
            or causal.get("cache_key") != cache.cache_key
            or causal.get("bridge_artifact_sha256") != bridge.artifact_sha256
            or causal.get("model_state_after_sha256") != entry_identity
            or causal.get("all_attention_o_decisions_completed") is not True
            or causal.get("qkv_byte_identical_before_and_after") is not True
            or causal.get("causal_downstream_replay_after_each_o_install") is not True
        ):
            raise Lfm2PostCausalOGlobalReclosureError(
                "Live child is not the independently pinned causal-O result."
            )

        producer_receipt, producer_seal, producer_claim = _bind_coordinate(
            producer_encoder,
            expected_tensor_sha256=expected_producer_encoder_sha256,
            receipt=producer_encoder_receipt,
            expected_receipt_sha256=expected_producer_encoder_receipt_sha256,
            claim_fields=(
                "producer_encoder_foundation_sha256",
                "producer_encoder_sha256",
            ),
            receipt_tensor_record_name="producer_encoder",
            label="producer encoder",
        )
        tied_receipt, tied_seal, tied_claim = _bind_coordinate(
            mapped_parent_tied_table,
            expected_tensor_sha256=expected_mapped_parent_tied_table_sha256,
            receipt=mapped_parent_tied_table_receipt,
            expected_receipt_sha256=(
                expected_mapped_parent_tied_table_receipt_sha256
            ),
            claim_fields=(
                "mapped_parent_tied_table_foundation_sha256",
                "source_tied_table_sha256",
                "mapped_table_sha256",
            ),
            receipt_tensor_record_name="mapped_parent_tied_table",
            label="mapped parent tied table",
        )
        interface_lineage = stack.get("tied_interface_lineage")
        global_gauge_required = bool(
            stack.get("body_global_coordinate_completion_enabled") is True
            or (
                isinstance(interface_lineage, Mapping)
                and interface_lineage.get("global_residual_gauge_required") is True
            )
        )
        if (
            not isinstance(interface_lineage, Mapping)
            or not isinstance(
                interface_lineage.get("global_residual_gauge_required"), bool
            )
            or interface_lineage.get("interface_report_sha256")
            != _coordinate_receipt_interface_report_sha256(
                producer_receipt,
                producer_seal,
            )
            or interface_lineage.get("interface_report_sha256")
            != _coordinate_receipt_interface_report_sha256(
                tied_receipt,
                tied_seal,
            )
            or interface_lineage.get("producer_encoder_sha256")
            not in _coordinate_hashes(producer_encoder)
            or interface_lineage.get(
                "producer_encoder_taken_from_authenticated_compiler_lineage"
            )
            is not True
        ):
            raise Lfm2PostCausalOGlobalReclosureError(
                "Interface coordinates do not bind to the semantic stack."
            )
        live_interface_table, live_interface_table_claim = (
            _live_interface_target_table(
                target_model,
                interface_lineage=interface_lineage,
            )
        )

        final_w2, rebase_receipt, terminal_prior = _terminal_parameter(
            target_model,
            rebase=terminal_rebase,
            stack_sha256=stack["result_sha256"],
            expected_rebase_sha256=expected_terminal_rebase_receipt_sha256,
            expected_prior_sha256=(expected_terminal_prior_authentication_sha256),
        )
        cache_context = (
            cache.selective_authenticated_session()
            if authenticated_cache_session is None
            else nullcontext(authenticated_cache_session)
        )
        replay_steps: list[Mapping[str, Any]] = []
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
                raise Lfm2PostCausalOGlobalReclosureError(
                    "Tokenizer bridge lacks an exact semantic target lattice."
                )
            accumulators, row_maps = _prepare_global_residual_accumulators(
                alignments
            )
            residual_bank = _initial_residual_bank(target_model, alignments)
            terminal_layer_index = len(kinds) - 1
            for layer_index, operator_kind in enumerate(kinds):
                replay = _replay_current_layer(
                    target_model,
                    alignments=alignments,
                    layer_index=layer_index,
                    pre_residual_by_batch=residual_bank,
                )
                current_identity = fingerprint_lfm2_parent(
                    target_model
                ).identity_sha256
                if replay.captured_child_model_state_sha256 != current_identity:
                    raise Lfm2PostCausalOGlobalReclosureError(
                        f"Layer {layer_index} replay is stale."
                    )
                parent = _load_parent_layer_trajectories(
                    session,
                    alignments=alignments,
                    layer_index=layer_index,
                    operator_kind=operator_kind,
                    include_final_norm=layer_index == terminal_layer_index,
                )
                chart, chart_receipt = _fit_residual_chart(
                    alignments=alignments,
                    parent=parent,
                    child_bank=replay,
                    producer_encoder=producer_encoder,
                    layer_index=layer_index,
                    global_gauge_required=global_gauge_required,
                )
                _capture_global_parent_layer_boundaries(
                    accumulators,
                    parent=parent,
                    residual_chart=chart,
                    producer_encoder=producer_encoder,
                    layer_index=layer_index,
                    terminal_layer_index=terminal_layer_index,
                    capture_source_local_pre=True,
                )
                _capture_global_child_layer_primals(
                    accumulators,
                    target_model=target_model,
                    alignments=alignments,
                    replay_bank=replay,
                    layer_index=layer_index,
                    deferred_final_w2_prior=(
                        terminal_prior
                        if layer_index == terminal_layer_index
                        else None
                    ),
                )
                residual_bank = _next_residual_bank(replay)
                step = _sealed(
                    {
                        "schema": LFM2_POST_CAUSAL_O_GLOBAL_REPLAY_STEP_SCHEMA,
                        "layer_index": layer_index,
                        "operator_kind": operator_kind,
                        "captured_child_model_state_sha256": current_identity,
                        "replay_bank_sha256": replay.bank_sha256,
                        "parent_trajectory_evidence_sha256": {
                            str(index): value.evidence_sha256
                            for index, value in sorted(parent.items())
                        },
                        "residual_chart_receipt_sha256": chart_receipt[
                            "result_sha256"
                        ],
                        "residual_chart_sha256": _tensor_sha256(chart),
                        "propagated_residual_sha256": dict(
                            _residual_hashes(residual_bank)
                        ),
                        "uses_all_preceding_live_child_layers": True,
                        "parent_rows_loaded_from_authenticated_cache": True,
                        "child_macrocell_replay_only": True,
                        "source_local_pre_post_operator_post_layer_captured": True,
                        "terminal_final_norm_captured": (
                            layer_index == terminal_layer_index
                        ),
                    }
                )
                replay_steps.append(step)
                del parent, chart
            session.verify()
            cache_session_receipt = session.as_dict()

        if (
            fingerprint_lfm2_parent(target_model).identity_sha256 != entry_identity
            or _state_tensor_hashes(target_model) != entry_hashes
        ):
            raise Lfm2PostCausalOGlobalReclosureError(
                "Evidence construction mutated the live child."
            )
        evidence, writer_parameters = (
            _build_global_residual_evidence_from_accumulators(
                accumulators,
                target_model=target_model,
                mapped_parent_tied_table=live_interface_table,
                deferred_final_w2_prior=terminal_prior,
                source_local=True,
            )
        )
        protected_before = {
            name: digest
            for name, digest in entry_hashes.items()
            if name not in writer_parameters
        }
        installation = (
            compile_install_lfm2_semantic_source_local_output_residual_closure(
                evidence,
                live_writer_parameters=writer_parameters,
                deferred_final_writer_parameter=final_w2,
                policy=closure_policy,
            )
        )
        if (
            not isinstance(installation, Lfm2SemanticGlobalResidualInstallation)
            or not installation.verify_authentication()
            or installation.report.get("schema")
            != LFM2_SEMANTIC_SOURCE_LOCAL_OUTPUT_GLOBAL_RESIDUAL_INTEGRATION_SCHEMA
        ):
            raise Lfm2PostCausalOGlobalReclosureError(
                "Source-local output-writer closure returned no authenticated result."
            )
        after_hashes = _state_tensor_hashes(target_model)
        changed = _changed_state_keys(entry_hashes, after_hashes)
        allowed = tuple(sorted(writer_parameters))
        protected_after = {
            name: digest
            for name, digest in after_hashes.items()
            if name not in writer_parameters
        }
        final_w2_sha = lfm2_cpcf_terminal_action_tensor_sha256(final_w2)
        if (
            not set(changed).issubset(set(allowed))
            or protected_after != protected_before
            or final_w2_sha != terminal_prior.live_deferred_w2_sha256
            or installation.report.get("final_w2_unchanged") is not True
            or installation.report.get("final_w2_installation_count") != 0
            or _parameter_count(target_model) != entry_count
            or _parameter_topology(target_model) != snapshot.topology
            or _runtime_inventory(target_model) != entry_runtime
            or _model_modes(target_model) != entry_modes
        ):
            raise Lfm2PostCausalOGlobalReclosureError(
                "Global reclosure mutated protected producer state, terminal W2, "
                "or runtime topology."
            )
        final_identity = fingerprint_lfm2_parent(target_model).identity_sha256
        body = {
            "schema": LFM2_POST_CAUSAL_O_GLOBAL_RECLOSURE_SCHEMA,
            "status": "completed-cached-source-local-output-writer-reclosure",
            "cache_key": cache.cache_key,
            "bridge_artifact_sha256": bridge.artifact_sha256,
            "semantic_layer_stack_result_sha256": stack["result_sha256"],
            "causal_o_result_sha256": causal["result_sha256"],
            "causal_o_live_entry_state_sha256": entry_identity,
            "terminal_rebase_result_sha256": rebase_receipt["result_sha256"],
            "terminal_prior_authentication_sha256": (
                terminal_prior.authentication_sha256
            ),
            "producer_encoder_tensor_sha256": _tensor_sha256(producer_encoder),
            "producer_encoder_archive_tensor_sha256": (
                lfm2_post_causal_o_global_coordinate_sha256(producer_encoder)
            ),
            "producer_encoder_receipt_sha256": producer_receipt[producer_seal],
            "producer_encoder_receipt_tensor_claim_sha256": producer_claim,
            "mapped_parent_tied_table_tensor_sha256": _tensor_sha256(
                mapped_parent_tied_table
            ),
            "mapped_parent_tied_table_archive_tensor_sha256": (
                lfm2_post_causal_o_global_coordinate_sha256(
                    mapped_parent_tied_table
                )
            ),
            "mapped_parent_tied_table_receipt_sha256": tied_receipt[tied_seal],
            "mapped_parent_tied_table_receipt_tensor_claim_sha256": tied_claim,
            "live_interface_target_table_sha256": _tensor_sha256(
                live_interface_table
            ),
            "live_interface_target_table_lineage_claim_sha256": (
                live_interface_table_claim
            ),
            "global_evidence_terminal_table_source": (
                "authenticated_live_child_interface_target_table"
            ),
            "tokenizer_session_sha256": _digest(
                tokenizer_session.get("receipt_sha256"),
                label="tokenizer session SHA-256",
            ),
            "semantic_refinement_sha256": _digest(
                refinement.get("report_sha256"),
                label="semantic refinement SHA-256",
            ),
            "cache_session_sha256": _digest(
                cache_session_receipt.get("session_sha256"),
                label="cache session SHA-256",
            ),
            "global_residual_semantic_row_map_sha256": row_maps["result_sha256"],
            "replay_step_sha256": [
                value["result_sha256"] for value in replay_steps
            ],
            "replay_steps": [dict(value) for value in replay_steps],
            "causal_layer_indices": list(range(len(kinds))),
            "global_residual_evidence_sha256": evidence.evidence_sha256,
            "global_reclosure_report_sha256": installation.report[
                "report_sha256"
            ],
            "corrected_terminal_w2_prior_sha256": (
                installation.corrected_final_w2_prior.authentication_sha256
            ),
            "model_state_before_sha256": entry_identity,
            "model_state_after_sha256": final_identity,
            "changed_state_keys": list(changed),
            "allowed_changed_state_keys": list(allowed),
            "changed_state_keys_are_output_writers_only": True,
            "protected_producer_tensors_byte_identical": True,
            "attention_qkv_byte_identical": True,
            "short_conv_input_byte_identical": True,
            "mlp_gate_up_byte_identical": True,
            "live_terminal_w2_unchanged_and_deferred": True,
            "corrected_terminal_w2_prior_returned_not_installed": True,
            "parameter_count_changed": False,
            "parameter_topology_preserved": True,
            "parameter_identities_preserved": True,
            "parameter_storage_preserved": True,
            "gradient_state_preserved": True,
            "module_modes_preserved": True,
            "one_whole_model_rollback_boundary": True,
            "semantic_alignments_derived_internally": True,
            "cached_parent_trajectories_loaded_internally": True,
            "current_live_child_replayed_in_causal_layer_order": True,
            "residual_charts_refit_from_authenticated_producer_encoder": True,
            "source_local_pre_post_operator_post_layer_boundaries_captured": True,
            "terminal_parent_final_norm_captured": True,
            "current_child_primals_captured": True,
            "caller_supplied_parent_or_child_trace_tensors": False,
            "parent_model_objects_constructed": 0,
            "parent_model_forward_calls": 0,
            "parent_trace_recaptures": 0,
            "child_model_forward_calls": 0,
            "generation_calls": 0,
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "pruning_operations": 0,
            "quantization_operations": 0,
            "sparsity_used": False,
            "parameter_sharing_used": False,
        }
        result = Lfm2PostCausalOGlobalReclosureInstallation(
            installation=installation,
            evidence=evidence,
            receipt=_sealed(body),
        )
    except BaseException as exc:
        try:
            snapshot.restore(target_model)
            if (
                fingerprint_lfm2_parent(target_model).identity_sha256
                != entry_identity
                or _state_tensor_hashes(target_model) != entry_hashes
                or _parameter_count(target_model) != entry_count
                or _parameter_topology(target_model) != snapshot.topology
                or _runtime_inventory(target_model) != entry_runtime
                or _model_modes(target_model) != entry_modes
            ):
                raise Lfm2PostCausalOGlobalReclosureError(
                    "Post-causal-O rollback did not restore exact tensor and "
                    "runtime state."
                )
        except BaseException as rollback_exc:
            raise Lfm2PostCausalOGlobalReclosureError(
                "Post-causal-O global reclosure failed and exact rollback could "
                "not be proven."
            ) from rollback_exc
        raise Lfm2PostCausalOGlobalReclosureError(
            "Post-causal-O global reclosure failed; exact entry state was restored."
        ) from exc
    if result is None:  # pragma: no cover - defensive totality guard
        raise AssertionError("Post-causal-O global reclosure produced no result.")
    return result


__all__ = [
    "LFM2_POST_CAUSAL_O_GLOBAL_RECLOSURE_SCHEMA",
    "LFM2_POST_CAUSAL_O_GLOBAL_REPLAY_STEP_SCHEMA",
    "Lfm2PostCausalOGlobalReclosureError",
    "Lfm2PostCausalOGlobalReclosureInstallation",
    "compile_install_lfm2_post_causal_o_source_local_global_reclosure",
    "lfm2_post_causal_o_global_coordinate_sha256",
]
