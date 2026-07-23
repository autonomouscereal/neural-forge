"""Transactional no-training semantic causal-stack postfold.

This module resumes from a *saved standalone* CPCF-v5 LFM2 child.  It never
constructs or executes the parent model.  Instead it:

* authenticates the dense child artifact and the complete causal trace cache;
* authenticates legacy c61 or fast semantic-foundation ancestry and
  direct-local parent Q/K/V/O bytes;
* builds per-phase donor/target semantic-span attention evidence;
* requires causal attention/short-conv/replay-fresh-MLP installation in order;
* defers the shared final-layer W2 writer;
* jointly solves final-MLP, parent-terminal, and lexical-continuation evidence;
* atomically publishes and freshly reloads a new dense standalone checkpoint.

The module does not equate an attention-only correction with a coherent child.
Until a complete causal layer-stack adapter exists, the default path fails
closed before final-W2 mutation or checkpoint publication and reports both the
long-horizon short-conv and replay-fresh MLP gaps.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any
from weakref import WeakKeyDictionary

from neural_forge.dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from neural_forge import dynamic_lfm2_cpcf_resources as cpcf_resources
from neural_forge import hf_lfm2_cpcf_model_compiler as cpcf_model_compiler
from neural_forge import hf_lfm2_cpcf_model_fold as cpcf_model_fold
from neural_forge import hf_lfm2_cpcf_v5_model_compiler as cpcf_v5_compiler
from neural_forge.dynamic_fold_executor import fingerprint_model_state
from neural_forge.dynamic_fold_artifact import (
    STANDALONE_FOLD_ARTIFACT_INVENTORY_SCHEMA,
    canonical_standalone_fold_artifact_inventory,
    scan_standalone_fold_artifact_tree,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceCache,
    fingerprint_lfm2_parent,
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_cpcf_v5_quarantine import (
    LFM2_CPCF_V5_ARTIFACT_METADATA,
    LFM2_CPCF_V5_QUARANTINE_ARTIFACT_SCHEMA,
    Lfm2CpcfV5QuarantineSaver,
    Lfm2CpcfV5StrictLocalLoader,
)
from neural_forge.hf_lfm2_cpcf_model_fold import (
    Lfm2CpcfModelFoldPolicy,
    independent_parameter_inventory,
)
from neural_forge.hf_lfm2_cpcf_macrocell_replay import (
    replay_lfm2_cpcf_macrocell,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import _rope_from_target
from neural_forge.hf_lfm2_global_residual_closure import (
    Lfm2GlobalResidualClosurePolicy,
)
from neural_forge.hf_lfm2_producer_aware_semantic_integration import (
    Lfm2ProducerAwareCorrectionScalePolicy,
)
from neural_forge.hf_lfm2_cpcf_selective_evidence import (
    open_lfm2_cpcf_selective_evidence_provider_from_catalog,
)
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (
    LFM2_CPCF_TERMINAL_ACTION_EXACT_ZERO_ELIGIBILITY_SCHEMA,
    LFM2_CPCF_TERMINAL_ACTION_REVIEW_PENDING_SCHEMA,
    LFM2_CPCF_TERMINAL_ACTION_RESULT_SCHEMA,
    LFM2_CPCF_TERMINAL_ACTION_STRUCTURAL_NONVETO_SCHEMA,
    Lfm2CpcfTerminalActionEvidence,
    Lfm2CpcfTerminalActionDownWeightPrior,
    Lfm2CpcfTerminalActionPolicy,
    Lfm2CpcfTerminalActionResidualResult,
    Lfm2CpcfTerminalActionSeed,
    _all_passed,
    _evaluate_split,
    _metric_gates,
    _prepare_split,
    _tensor_sha256 as _terminal_action_tensor_sha256,
    compile_lfm2_cpcf_terminal_action_residual,
)
from neural_forge.hf_lfm2_cached_endpoint_qk_residual_fold import (
    Lfm2DenseQkvoSourceInventory,
    seal_lfm2_dense_qkvo_source_inventory,
)
from neural_forge.hf_lfm2_long_horizon_terminal_refold import (
    LFM2_DIRECT_LOCAL_MAPPED_TIED_TABLE_SOURCE_SCHEMA,
    LFM2_LONG_HORIZON_TERMINAL_CACHE_COMPATIBILITY_SCHEMA,
    Lfm2LongHorizonTerminalCompositeLineage,
    Lfm2LongHorizonTerminalSourceLineage,
    Lfm2LongHorizonTrajectorySplitPlan,
    Lfm2LocalParentFinalRmsNormLoad,
    _build_split_evidence,
    _build_terminal_composite_lineage,
    _bind_terminal_cache_compatibility_to_payload_session,
    _collect_cached_semantic_trajectories,
    _summarize_terminal_cache_payload_loads,
    _target_runtime_seed,
    _tensor_sha256 as _long_horizon_tensor_sha256,
    authenticate_lfm2_long_horizon_terminal_composite_lineage,
    build_lfm2_long_horizon_terminal_source_lineage,
    load_lfm2_parent_final_rmsnorm_from_local_safetensors,
    preflight_lfm2_long_horizon_terminal_cache,
)
from neural_forge.hf_lfm2_semantic_attention_refold_adapter import (
    Lfm2SemanticAttentionReplayEndpoint,
)
from neural_forge.hf_lfm2_semantic_body_recipe import (
    LFM2_SEMANTIC_BODY_RECIPE_IDS,
    Lfm2SemanticBodyRecipe,
    is_supported_lfm2_semantic_component_equivalence_inventory,
    resolve_lfm2_semantic_body_recipe,
)
from neural_forge.hf_lfm2_semantic_foundation import (
    LFM2_SEMANTIC_FOUNDATION_LINEAGE_SCHEMA,
    LFM2_SEMANTIC_FOUNDATION_SCHEMA,
    Lfm2SemanticFoundationError,
    authenticate_lfm2_semantic_foundation_receipt,
    rehydrate_lfm2_semantic_foundation_interface,
)
from neural_forge.hf_lfm2_semantic_span_cpcf_alignment import (
    Lfm2SemanticSpanCpcfAlignment,
    Lfm2SemanticSpanCpcfPhaseAlignment,
    audit_lfm2_semantic_span_cpcf_refinement,
    build_lfm2_semantic_span_cpcf_alignment,
)
from neural_forge.hf_lfm2_semantic_span_phase_bridge import (
    _decode_exact,
    _encode_exact,
    build_lfm2_semantic_span_selected_input_evidence,
    build_lfm2_semantic_span_phase_bridge,
)
from neural_forge.hf_semantic_attention_lattice import (
    Lfm2SemanticAttentionCacheCapability,
    SemanticAttentionExpansionTransport,
    SemanticAttentionPreRopeHistory,
    assemble_lfm2_explicit_pre_rope_history,
    fold_semantic_attention_parent_pre_rope_priors,
    inspect_lfm2_semantic_attention_cache_capability,
    transport_lfm2_phase_probability_preserving_attention,
)
from neural_forge.hf_tied_causal_interface_fold import (
    TIED_CAUSAL_INTERFACE_SELF_DUAL_SCHEMA,
    TiedCausalInterfaceFold,
    whiten_tied_causal_interface,
)
from neural_forge.hf_tied_residual_congruence import (
    TIED_RESIDUAL_CONGRUENCE_MODES,
    TIED_RESIDUAL_CONGRUENCE_SCHEMA,
    TIED_RESIDUAL_CURRENT_SELF_DUAL_MODE,
    TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE,
    TIED_RESIDUAL_STRONG_CONGRUENCE_MODE,
    TiedResidualAnchorTransportWitness,
    authenticate_tied_residual_congruence_interface,
    compile_tied_residual_congruence_candidate,
)
from neural_forge.hf_tokenizer_bridge import (
    StrictTokenizerBridgeArtifact,
    fingerprint_hf_tokenizer_semantics,
    load_strict_tokenizer_bridge_artifact,
)


LFM2_SEMANTIC_POSTFOLD_SCHEMA = "neural-forge-lfm2-semantic-postfold-v1"
LFM2_SEMANTIC_ATTENTION_STAGE_SCHEMA = "neural-forge-lfm2-semantic-attention-stage-v1"
LFM2_SEMANTIC_ATTENTION_INSTALLATION_SCHEMA = (
    "neural-forge-lfm2-semantic-attention-installation-v1"
)
LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_LAYER_INSTALLATION_SCHEMA = (
    "neural-forge-lfm2-semantic-attention-group-qdeim-layer-installation-v1"
)
_ATTENTION_STRUCTURAL_NOOP_STATUS = "structurally-identical-or-invalid-incumbent-noop"
_ATTENTION_CANDIDATE_ELIGIBILITY_BASIS = (
    "finite_dense_bfloat16_exact_shape_and_nonincumbent_effect"
)
LFM2_SEMANTIC_POSTFOLD_CHECKPOINT_SCHEMA = (
    "neural-forge-lfm2-semantic-postfold-checkpoint-v1"
)
LFM2_SEMANTIC_POSTFOLD_FAILURE_SCHEMA = "neural-forge-lfm2-semantic-postfold-failure-v1"
LFM2_SEMANTIC_COMPILER_LINEAGE_SCHEMA = "neural-forge-lfm2-semantic-compiler-lineage-v1"
LFM2_SEMANTIC_RESIDUAL_CHART_SCHEMA = "neural-forge-lfm2-semantic-residual-chart-v1"
LFM2_SEMANTIC_LEXICAL_COMPLETION_SCHEMA = (
    "neural-forge-lfm2-semantic-lexical-completion-v1"
)
LFM2_SEMANTIC_TERMINAL_REVIEW_PENDING_SCHEMA = (
    "neural-forge-lfm2-fused-semantic-terminal-review-pending-v1"
)
LFM2_SEMANTIC_TERMINAL_FRONTIER_EXPORT_CONTEXT_SCHEMA = (
    "neural-forge-lfm2-semantic-terminal-frontier-export-context-v1"
)
LFM2_SEMANTIC_FOUNDATION_ANCHOR_SELECTION_SCHEMA = (
    "neural-forge-lfm2-semantic-foundation-anchor-selection-v1"
)
SEMANTIC_ATTENTION_INSTALLER_CALLABLE = (
    "compile_install_lfm2_semantic_attention_postfold"
)
SEMANTIC_LAYER_STACK_INSTALLER_CALLABLE = (
    "compile_install_lfm2_semantic_layer_stack_postfold"
)

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
    }
)

_TERMINAL_FRONTIER_EXPORT_CONTEXT_ISSUER = object()
_TERMINAL_FRONTIER_LIVE_FACT_NAMES = (
    "mapped_parent_tied_table_sha256",
    "tied_interface_authentication_sha256",
    "terminal_cache_compatibility_sha256",
    "terminal_cache_payload_binding_sha256",
    "body_recipe_sha256",
    "interface_anchor_selection_sha256",
)


class _CompilerIssuedTerminalFrontierExportContext(Mapping[str, Any]):
    """Immutable live context that cannot be recreated by resealing a dict."""

    __slots__ = (
        "_issued_fact_sha256",
        "_issued_fact_tuple",
        "_issuer",
        "_values",
    )

    def __init__(
        self,
        values: Mapping[str, Any],
        issued_facts: Mapping[str, str],
        *,
        issuer: object,
    ) -> None:
        if issuer is not _TERMINAL_FRONTIER_EXPORT_CONTEXT_ISSUER:
            raise Lfm2SemanticPostfoldError(
                "Only the fused compiler may issue a terminal frontier context."
            )
        facts = dict(issued_facts)
        if set(facts) != set(_TERMINAL_FRONTIER_LIVE_FACT_NAMES):
            raise Lfm2SemanticPostfoldError(
                "Terminal frontier issued facts are incomplete."
            )
        fact_tuple = tuple(
            (name, facts[name]) for name in _TERMINAL_FRONTIER_LIVE_FACT_NAMES
        )
        object.__setattr__(self, "_values", MappingProxyType(dict(values)))
        object.__setattr__(self, "_issued_fact_tuple", fact_tuple)
        object.__setattr__(
            self,
            "_issued_fact_sha256",
            canonical_sha256([list(value) for value in fact_tuple]),
        )
        object.__setattr__(self, "_issuer", issuer)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Compiler-issued terminal frontier contexts are frozen.")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Compiler-issued terminal frontier contexts are frozen.")

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def _authenticate_issued_facts(
        self,
        issuer: object,
    ) -> tuple[tuple[tuple[str, str], ...], str]:
        if issuer is not self._issuer or issuer is not (
            _TERMINAL_FRONTIER_EXPORT_CONTEXT_ISSUER
        ):
            raise Lfm2SemanticPostfoldError(
                "Terminal frontier context was not issued by this compiler."
            )
        fact_tuple = self._issued_fact_tuple
        fact_sha256 = self._issued_fact_sha256
        if fact_sha256 != canonical_sha256([list(value) for value in fact_tuple]):
            raise Lfm2SemanticPostfoldError(
                "Terminal frontier compiler-issued fact binding changed."
            )
        return fact_tuple, fact_sha256


_TERMINAL_ZERO_OPERATION_AUDIT = MappingProxyType(
    {
        "parent_model_forward_calls": 0,
        "child_model_forward_calls": 0,
        "model_generate_calls": 0,
        "parent_cache_generation_calls": 0,
        "trace_recaptures": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "new_parameters": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
)
_TERMINAL_ACTION_RESULT_METHOD = (
    "closed_form_rmsnorm_gauss_newton_then_weighted_ridge_"
    "exact_bfloat16_terminal_writer_frontier"
)
_ATTENTION_SIX_TENSOR_ROLES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "out_proj.weight",
    "q_layernorm.weight",
    "k_layernorm.weight",
)
_ATTENTION_REFOLD_ZERO_OPERATIONS = MappingProxyType(
    {
        **dict(_ZERO_OPERATIONS),
        "model_forward_calls": 0,
    }
)


class Lfm2SemanticPostfoldError(RuntimeError):
    """A sealed input, analytic fold, or checkpoint transaction failed."""


class Lfm2SemanticPostfoldBlockedError(Lfm2SemanticPostfoldError):
    """The exact no-training path is unavailable and published nothing."""

    def __init__(self, message: str, report: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.report = MappingProxyType(dict(report))


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError("Semantic postfold requires PyTorch.") from exc
    return torch


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2SemanticPostfoldError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _is_digest(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024**2), b""):
                digest.update(block)
    except OSError as exc:
        raise Lfm2SemanticPostfoldError(
            f"Could not hash authenticated local file {path}."
        ) from exc
    return digest.hexdigest()


def _sealed(body: Mapping[str, Any], seal_name: str) -> dict[str, Any]:
    if seal_name in body:
        raise Lfm2SemanticPostfoldError(f"Receipt body already contains {seal_name!r}.")
    value = dict(body)
    value[seal_name] = canonical_sha256(value)
    return value


def _verify_sealed(
    value: Mapping[str, Any],
    *,
    seal_name: str,
    label: str,
) -> dict[str, Any]:
    result = dict(value)
    supplied = result.pop(seal_name, None)
    if supplied != canonical_sha256(result):
        raise Lfm2SemanticPostfoldError(f"{label} self-seal is invalid.")
    return {**result, seal_name: supplied}


def _authenticate_body_recipe_binding(
    value: Mapping[str, Any],
    *,
    body_recipe: str | Lfm2SemanticBodyRecipe | None = None,
    component: str | None = None,
    label: str,
) -> Lfm2SemanticBodyRecipe:
    """Authenticate one complete-body recipe and, optionally, its component lane."""

    declared_id = value.get("body_recipe_id")
    try:
        recipe = resolve_lfm2_semantic_body_recipe(
            declared_id if body_recipe is None else body_recipe
        )
    except (TypeError, ValueError) as exc:
        raise Lfm2SemanticPostfoldError(
            f"{label} names an invalid semantic body recipe."
        ) from exc
    expected_lane = None if component is None else recipe.component_lanes.get(component)
    if (
        declared_id != recipe.recipe_id
        or value.get("body_recipe_sha256") != recipe.recipe_sha256
        or value.get("body_recipe") != recipe.as_dict()
        or (component is not None and expected_lane is None)
        or (
            component is not None
            and value.get("recipe_component_lane") != expected_lane
        )
    ):
        raise Lfm2SemanticPostfoldError(
            f"{label} did not bind the authenticated semantic body recipe"
            + (" and component lane." if component is not None else ".")
        )
    return recipe


def _foundation_lineage_tensor_sha256(value: Any) -> str:
    """Match the semantic-foundation tensor hash used by companion receipts."""

    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise Lfm2SemanticPostfoldError(
            "Foundation anchor selection references a non-tensor value."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
    }
    digest = hashlib.sha256(canonical_json_bytes(descriptor))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _authenticate_foundation_anchor_selection(
    value: Mapping[str, Any],
    *,
    selected_interface: TiedCausalInterfaceFold | None = None,
) -> Mapping[str, Any]:
    """Authenticate one explicit lazy anchor selection and its result hashes."""

    selection = _verify_sealed(
        _json_owned_receipt_value(value),
        seal_name="selection_sha256",
        label="semantic-foundation anchor selection",
    )
    indices = selection.get("causal_anchor_materializable_candidate_indices")
    selected_index = selection.get("selected_anchor_candidate_index")
    default_index = selection.get("causal_anchor_compatibility_default_candidate_index")
    recipe = selection.get("selected_anchor_candidate_recipe")
    recipe_body = dict(recipe) if isinstance(recipe, Mapping) else {}
    recipe_sha = recipe_body.pop("recipe_sha256", None)
    if (
        selection.get("schema") != LFM2_SEMANTIC_FOUNDATION_ANCHOR_SELECTION_SCHEMA
        or not isinstance(indices, list)
        or not indices
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in indices
        )
        or len(set(indices)) != len(indices)
        or isinstance(selected_index, bool)
        or not isinstance(selected_index, int)
        or selected_index not in indices
        or isinstance(default_index, bool)
        or not isinstance(default_index, int)
        or default_index not in indices
        or selection.get("anchor_base_candidate_index") != 0
        or 0 not in indices
        or not isinstance(recipe, Mapping)
        or recipe.get("candidate_index") != selected_index
        or recipe.get("base_target_weight_sha256")
        != selection.get("anchor_base_target_weight_sha256")
        or recipe_sha != canonical_sha256(recipe_body)
        or selection.get("selected_anchor_candidate_recipe_sha256") != recipe_sha
        or selection.get("saved_child_exactly_matches_compatibility_default")
        is not True
        or selection.get("selected_candidate_lazily_materialized") is not True
        or selection.get(
            "target_model_installation_deferred_until_authenticated_pipeline_session"
        )
        is not True
        or selection.get("dense_anchor_candidate_checkpoints_persisted") != 0
        or selection.get("numeric_metrics_have_final_intelligence_authority")
        is not False
        or selection.get("selected_anchor_exactly_equals_base_plus_residual")
        is not True
    ):
        raise Lfm2SemanticPostfoldError(
            "Semantic-foundation anchor selection binding is invalid."
        )
    for name in (
        "interface_frontier_companion_file_sha256",
        "interface_frontier_companion_sha256",
        "interface_statistics_artifact_sha256",
        "causal_anchor_frontier_sha256",
        "anchor_base_interface_report_sha256",
        "anchor_base_target_weight_sha256",
        "anchor_base_target_weight_foundation_sha256",
        "compatibility_default_interface_report_sha256",
        "compatibility_default_target_weight_sha256",
        "saved_target_weight_sha256",
        "selected_anchor_candidate_recipe_sha256",
        "selected_interface_report_sha256",
        "selected_target_weight_sha256",
        "selected_target_weight_foundation_sha256",
        "selected_residual_weight_sha256",
        "selected_residual_weight_foundation_sha256",
        "selected_producer_encoder_sha256",
        "selection_sha256",
    ):
        _digest(selection.get(name), f"foundation anchor selection {name}")
    if selected_interface is not None and (
        not isinstance(selected_interface, TiedCausalInterfaceFold)
        or selected_interface.report.get("report_sha256")
        != selection.get("selected_interface_report_sha256")
        or selected_interface.report.get("target_weight_sha256")
        != selection.get("selected_target_weight_sha256")
        or selected_interface.report.get("producer_encoder_sha256")
        != selection.get("selected_producer_encoder_sha256")
        or _foundation_lineage_tensor_sha256(selected_interface.target_weight)
        != selection.get("selected_target_weight_foundation_sha256")
    ):
        raise Lfm2SemanticPostfoldError(
            "Selected anchor interface differs from its companion binding."
        )
    return MappingProxyType(selection)


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2SemanticPostfoldError(
            "Cannot authenticate a non-materialized dense tensor."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    return canonical_sha256(
        {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": [int(size) for size in tensor.shape],
            "data_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )


def _owned_receipt_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _owned_receipt_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_owned_receipt_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_owned_receipt_value(item) for item in value)
    return value


def _json_owned_receipt_value(value: Any) -> Any:
    """Own a frozen receipt while restoring its canonical JSON container types."""

    if isinstance(value, Mapping):
        return {key: _json_owned_receipt_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_owned_receipt_value(item) for item in value]
    return value


class _CompilerIssuedFusedSemanticTerminalReceipt(Mapping[str, Any]):
    """Identity-only mapping proof for one live fused-terminal compilation."""

    __slots__ = ("__weakref__",)
    __hash__ = object.__hash__
    __eq__ = object.__eq__

    def __new__(cls):
        raise Lfm2SemanticPostfoldError(
            "Fused semantic terminal receipts are issued only by the real compiler."
        )

    def __getitem__(self, key: str) -> Any:
        return _fused_semantic_terminal_receipt_values(self)[key]

    def __iter__(self):
        return iter(_fused_semantic_terminal_receipt_values(self))

    def __len__(self) -> int:
        return len(_fused_semantic_terminal_receipt_values(self))

    def __copy__(self):
        return object.__new__(type(self))

    def __deepcopy__(self, memo):
        copied = object.__new__(type(self))
        memo[id(self)] = copied
        return copied


@dataclass(frozen=True, slots=True)
class _FusedSemanticTerminalReceiptRegistration:
    outer_result_sha256: str
    terminal_action_result_sha256: str
    terminal_action_selected_w2_sha256: str
    terminal_action_published_w2_sha256: str
    semantic_selected_w2_sha256: str
    semantic_published_w2_sha256: str
    values: Mapping[str, Any] = field(repr=False, compare=False)


def _fused_semantic_terminal_receipt_registration(
    value: Mapping[str, Any],
    *,
    values: Mapping[str, Any],
) -> _FusedSemanticTerminalReceiptRegistration:
    return _FusedSemanticTerminalReceiptRegistration(
        outer_result_sha256=_digest(
            value.get("result_sha256"),
            "fused semantic terminal receipt",
        ),
        terminal_action_result_sha256=_digest(
            value.get("terminal_action_result_sha256"),
            "fused semantic terminal-action result",
        ),
        terminal_action_selected_w2_sha256=_digest(
            value.get("terminal_action_selected_w2_sha256"),
            "fused semantic terminal-action selected W2",
        ),
        terminal_action_published_w2_sha256=_digest(
            value.get("terminal_action_published_w2_sha256"),
            "fused semantic terminal-action published W2",
        ),
        semantic_selected_w2_sha256=_digest(
            value.get("semantic_selected_w2_sha256"),
            "fused semantic selected W2",
        ),
        semantic_published_w2_sha256=_digest(
            value.get("semantic_published_w2_sha256"),
            "fused semantic published W2",
        ),
        values=values,
    )


def _build_fused_semantic_terminal_receipt_registry():
    registry: WeakKeyDictionary[
        _CompilerIssuedFusedSemanticTerminalReceipt,
        _FusedSemanticTerminalReceiptRegistration,
    ] = WeakKeyDictionary()
    registry_lock = RLock()

    def issue(
        value: Mapping[str, Any],
    ) -> _CompilerIssuedFusedSemanticTerminalReceipt:
        authenticated = _verify_sealed(
            value,
            seal_name="result_sha256",
            label="fused semantic terminal compiler receipt",
        )
        owned = MappingProxyType(_owned_receipt_value(authenticated))
        proof = object.__new__(_CompilerIssuedFusedSemanticTerminalReceipt)
        registration = _fused_semantic_terminal_receipt_registration(
            owned,
            values=owned,
        )
        with registry_lock:
            registry[proof] = registration
        return proof

    def authenticate(
        value: Mapping[str, Any],
    ) -> dict[str, Any]:
        if type(value) is not _CompilerIssuedFusedSemanticTerminalReceipt:
            raise Lfm2SemanticPostfoldError(
                "Fused semantic terminal receipt is not compiler-issued."
            )
        with registry_lock:
            registration = registry.get(value)
        if registration is None:
            raise Lfm2SemanticPostfoldError(
                "Fused semantic terminal receipt proof is unregistered."
            )
        authenticated = _verify_sealed(
            registration.values,
            seal_name="result_sha256",
            label="compiler-issued fused semantic terminal receipt",
        )
        observed = _fused_semantic_terminal_receipt_registration(
            authenticated,
            values=registration.values,
        )
        if observed != registration:
            raise Lfm2SemanticPostfoldError(
                "Fused semantic terminal receipt diverges from compiler-issued lineage."
            )
        return authenticated

    def values(
        proof: _CompilerIssuedFusedSemanticTerminalReceipt,
    ) -> Mapping[str, Any]:
        with registry_lock:
            registration = registry.get(proof)
        if registration is None:
            raise Lfm2SemanticPostfoldError(
                "Fused semantic terminal receipt proof is unregistered."
            )
        return registration.values

    def size() -> int:
        with registry_lock:
            return len(registry)

    return issue, authenticate, values, size


(
    _issue_fused_semantic_terminal_receipt,
    _authenticate_fused_semantic_terminal_receipt_lineage,
    _fused_semantic_terminal_receipt_values,
    _fused_semantic_terminal_receipt_registry_size,
) = _build_fused_semantic_terminal_receipt_registry()
del _build_fused_semantic_terminal_receipt_registry


def _interface_tensor_sha256(value: Any) -> str:
    """Match ``hf_tied_causal_interface_fold`` tensor authentication."""

    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            [int(size) for size in tensor.shape],
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _parameter_count(model: Any) -> int:
    seen: set[int] = set()
    total = 0
    for value in model.parameters():
        if id(value) in seen:
            continue
        seen.add(id(value))
        total += int(value.numel())
    return total


def _state_tensor_hashes(model: Any) -> dict[str, str]:
    return {
        name: _tensor_sha256(value)
        for name, value in sorted(model.state_dict().items())
    }


_DENSE_PRODUCER_WEIGHT_ROLES = MappingProxyType(
    {
        "full_attention": (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.out_proj.weight",
        ),
        "short_conv": (
            "conv.in_proj.weight",
            "conv.conv.weight",
            "conv.out_proj.weight",
        ),
        "mlp": (
            "feed_forward.w1.weight",
            "feed_forward.w3.weight",
            "feed_forward.w2.weight",
        ),
    }
)


def _dense_weight_structural_coverage(
    value: Any,
    *,
    parameter_name: str,
    require_parameter: bool = True,
) -> Mapping[str, Any]:
    """Authenticate that a dense producer spans every output and input axis.

    This is deliberately an exact structural gate, not a quality metric.  A
    dense folded weight is considered planted only when it is finite, contains
    a nonzero coefficient, and leaves no output row/channel or input/lag column
    structurally disconnected.  It does not require parent/child numeric weight
    equality and it does not impose an arbitrary magnitude threshold.
    """

    torch = _torch()
    if (
        (require_parameter and not isinstance(value, torch.nn.Parameter))
        or not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or not torch.is_floating_point(value)
        or value.ndim < 2
        or value.numel() <= 0
    ):
        raise Lfm2SemanticPostfoldError(
            f"{parameter_name} is not a materialized dense floating "
            + ("Parameter." if require_parameter else "tensor.")
        )
    tensor = value.detach().to(device="cpu").contiguous()
    finite = bool(torch.isfinite(tensor.to(dtype=torch.float32)).all().item())
    nonzero = tensor.ne(0)
    nonzero_count = int(nonzero.sum().item())
    output_units = int(tensor.shape[0])
    input_units = int(tensor.shape[-1])
    covered_output_units = int(
        nonzero.reshape(output_units, -1).any(dim=1).sum().item()
    )
    covered_input_units = int(nonzero.reshape(-1, input_units).any(dim=0).sum().item())
    body = {
        "schema": "neural-forge-dense-weight-structural-coverage-v1",
        "parameter_name": parameter_name,
        "ordinary_parameter_required": require_parameter,
        "ordinary_parameter": isinstance(value, torch.nn.Parameter),
        "shape": [int(size) for size in tensor.shape],
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "tensor_sha256": _tensor_sha256(tensor),
        "element_count": int(tensor.numel()),
        "nonzero_element_count": nonzero_count,
        "nonzero_element_fraction": nonzero_count / int(tensor.numel()),
        "finite": finite,
        "output_structural_unit_count": output_units,
        "covered_output_structural_unit_count": covered_output_units,
        "input_structural_unit_count": input_units,
        "covered_input_structural_unit_count": covered_input_units,
        "all_output_structural_units_covered": (covered_output_units == output_units),
        "all_input_structural_units_covered": covered_input_units == input_units,
        "structurally_nonzero": bool(
            finite
            and nonzero_count > 0
            and covered_output_units == output_units
            and covered_input_units == input_units
        ),
    }
    return MappingProxyType(_sealed(body, "result_sha256"))


def _lfm2_dense_producer_chain_coverage(
    target_model: Any,
    *,
    layer_index: int,
    producer_kind: str,
    defer_down_weight: bool = False,
) -> Mapping[str, Any]:
    """Seal structural coverage of one existing dense LFM2 producer chain.

    The final MLP may defer only W2.  Its W1/W3 producer must still be planted
    before terminal evidence can be trusted; the deferred W2 bytes are recorded
    and authenticated but are not counted as required until terminal publish.
    """

    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or layer_index < 0
        or producer_kind not in _DENSE_PRODUCER_WEIGHT_ROLES
        or (defer_down_weight and producer_kind != "mlp")
    ):
        raise Lfm2SemanticPostfoldError(
            "Dense producer-chain coverage request is invalid."
        )
    roles = tuple(_DENSE_PRODUCER_WEIGHT_ROLES[producer_kind])
    deferred_roles = ("feed_forward.w2.weight",) if defer_down_weight else ()
    required_roles = tuple(role for role in roles if role not in deferred_roles)
    named_rows = list(target_model.named_parameters(remove_duplicate=False))
    named: dict[str, Any] = {}
    duplicate_names: set[str] = set()
    for name, parameter in named_rows:
        if name in named:
            duplicate_names.add(name)
        named[name] = parameter
    receipts: dict[str, Mapping[str, Any]] = {}
    for role in roles:
        parameter_name = f"model.layers.{layer_index}.{role}"
        if parameter_name in duplicate_names or parameter_name not in named:
            raise Lfm2SemanticPostfoldError(
                f"Dense producer Parameter {parameter_name} is missing or ambiguous."
            )
        receipts[role] = _dense_weight_structural_coverage(
            named[parameter_name],
            parameter_name=parameter_name,
        )
    role_hashes = {role: receipts[role]["tensor_sha256"] for role in roles}
    body = {
        "schema": "neural-forge-lfm2-dense-producer-chain-coverage-v1",
        "layer_index": layer_index,
        "producer_kind": producer_kind,
        "required_roles": list(required_roles),
        "deferred_roles": list(deferred_roles),
        "role_receipts": {role: dict(receipts[role]) for role in roles},
        "required_role_state_sha256": canonical_sha256(
            {role: role_hashes[role] for role in required_roles}
        ),
        "all_role_state_sha256": canonical_sha256(role_hashes),
        "required_role_count": len(required_roles),
        "structurally_nonzero_required_role_count": sum(
            int(receipts[role]["structurally_nonzero"]) for role in required_roles
        ),
        "blank_or_partially_blank_required_role_count": sum(
            int(not receipts[role]["structurally_nonzero"]) for role in required_roles
        ),
        "all_required_roles_structurally_nonzero": all(
            receipts[role]["structurally_nonzero"] for role in required_roles
        ),
        "dense_existing_topology_only": True,
        "numeric_parent_child_weight_match_objective": False,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    return MappingProxyType(_sealed(body, "result_sha256"))


def _changed_state_keys(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> tuple[str, ...]:
    if set(before) != set(after):
        raise Lfm2SemanticPostfoldError(
            "Analytic attention installation changed the child state topology."
        )
    return tuple(name for name in sorted(before) if before[name] != after[name])


def _semantic_layer_stack_state_invariant_failures(
    stack: Mapping[str, Any],
) -> tuple[str, ...]:
    """Name every failed top-level structural invariant in a stack receipt."""

    before = stack.get("model_state_before_sha256")
    after = stack.get("model_state_after_sha256")
    changed = stack.get("changed_state_keys")
    parameter_count = stack.get("parameter_element_count")

    def is_digest(value: Any) -> bool:
        return bool(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    failures: list[str] = []
    if not is_digest(before):
        failures.append("model_state_before_sha256")
    if not is_digest(after):
        failures.append("model_state_after_sha256")
    if is_digest(before) and is_digest(after) and before == after:
        failures.append("model_state_changed")
    if not isinstance(changed, list) or not changed:
        failures.append("changed_state_keys_nonempty_list")
    elif any(not isinstance(name, str) or not name for name in changed):
        failures.append("changed_state_keys_nonempty_strings")
    elif changed != sorted(set(changed)):
        failures.append("changed_state_keys_canonical")
    if type(parameter_count) is not int or parameter_count <= 0:
        failures.append("parameter_element_count_positive_int")
    if stack.get("parameter_count_changed") is not False:
        failures.append("parameter_count_changed_false")
    if stack.get("parameter_topology_changed") is not False:
        failures.append("parameter_topology_changed_false")
    if stack.get("whole_stage_contains_meaningful_parameter_changes") is not True:
        failures.append("whole_stage_contains_meaningful_parameter_changes_true")
    if stack.get("complete_causal_layer_stack_adjudicated") is not True:
        failures.append("complete_causal_layer_stack_adjudicated_true")
    if (
        stack.get("causal_install_order")
        != "operator_then_replay_fresh_mlp_then_next_layer"
    ):
        failures.append("causal_install_order")
    return tuple(failures)


def _authenticate_semantic_global_residual_transition(
    stack: Mapping[str, Any],
    recipe: Lfm2SemanticBodyRecipe,
) -> Mapping[str, Any] | None:
    """Authenticate the optional post-stack closure as a second transition."""

    closure_keys = {
        key
        for key in stack
        if key.startswith("global_residual_")
        or key
        in {
            "global_residual_tied_table_source",
            "global_residual_v1_shared_attention_head_dimension",
            "global_residual_general_per_site_head_dimensions_claimed",
            "causal_layer_stack_model_state_after_sha256",
            "causal_layer_stack_changed_state_keys",
            "composed_transition_changed_state_keys",
            "decoded_behavior_review_is_evidence_for_understanding",
            "decoded_behavior_verdict_required_for_continuation",
            "final_layer_mlp_down_weight_prior_closure_lineage_sha256",
        }
    }
    applied = stack.get("global_residual_closure_applied")
    if applied is None:
        if closure_keys:
            raise Lfm2SemanticPostfoldError(
                "Semantic layer stack contains a partial global residual "
                "closure transition."
            )
        return None
    if applied is not True:
        raise Lfm2SemanticPostfoldError(
            "Global residual closure application flag is not true."
        )
    expected_lane = "intact_gqa_group_qdeim_coupled_qkvo"
    if (
        recipe.component_lanes.get("qk") != expected_lane
        or recipe.component_lanes.get("value_output") != expected_lane
        or recipe.component_lanes.get("mlp")
        not in {
            "nonlinear_qdeim_structural_priority",
            "nonlinear_qdeim_fit_corrected_priority",
        }
        or stack.get("global_residual_closure_required_by_body_recipe") is not True
        or not isinstance(stack.get("global_residual_closure_scope"), str)
        or not stack.get("global_residual_closure_scope")
    ):
        raise Lfm2SemanticPostfoldError(
            "Global residual closure is not bound to its coupled nonlinear "
            "Q-DEIM recipe."
        )

    def digest(value: Any, label: str) -> str:
        return _digest(value, label)

    def canonical_names(value: Any, label: str) -> tuple[str, ...]:
        if (
            not isinstance(value, list)
            or any(not isinstance(name, str) or not name for name in value)
            or value != sorted(set(value))
        ):
            raise Lfm2SemanticPostfoldError(
                f"{label} must be a canonical changed-key list."
            )
        return tuple(value)

    row_map_value = stack.get("global_residual_semantic_row_maps")
    if not isinstance(row_map_value, Mapping):
        raise Lfm2SemanticPostfoldError(
            "Global residual closure omitted its semantic row maps."
        )
    row_map = dict(row_map_value)
    row_map_sha = row_map.pop("result_sha256", None)
    rows = row_map.get("trajectories")
    if (
        row_map_sha != canonical_sha256(row_map)
        or stack.get("global_residual_semantic_row_map_sha256") != row_map_sha
        or row_map.get("kind") != "global_residual_semantic_row_maps"
        or row_map.get("exact_target_donor_canonical_maps") is not True
        or row_map.get("parent_histories_compacted_to_selected_donor_rows") is not True
        or row_map.get("numeric_or_manual_gate_used") is not False
        or not isinstance(rows, list)
        or not rows
        or row_map.get("trajectory_count") != len(rows)
    ):
        raise Lfm2SemanticPostfoldError(
            "Global residual semantic row-map receipt is invalid."
        )
    canonical_rows: list[int] = []
    for batch_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise Lfm2SemanticPostfoldError(
                "Global residual semantic row-map row is invalid."
            )
        target_positions = row.get("selected_target_positions")
        source_positions = row.get("source_donor_positions")
        compact_positions = row.get("compact_donor_positions")
        canonical_positions = row.get("canonical_boundary_row_indices")
        row_count = len(source_positions) if isinstance(source_positions, list) else -1
        if (
            row.get("batch_index") != batch_index
            or not isinstance(target_positions, list)
            or not isinstance(source_positions, list)
            or not isinstance(compact_positions, list)
            or not isinstance(canonical_positions, list)
            or row_count <= 0
            or len(target_positions) != row_count
            or len(compact_positions) != row_count
            or len(canonical_positions) != row_count
            or source_positions != list(range(row_count))
            or compact_positions != list(range(row_count))
            or target_positions != sorted(set(target_positions))
            or any(type(value) is not int or value < 0 for value in target_positions)
            or any(type(value) is not int or value < 0 for value in canonical_positions)
        ):
            raise Lfm2SemanticPostfoldError(
                "Global residual target/donor/canonical row map is invalid."
            )
        canonical_rows.extend(canonical_positions)
    if canonical_rows != list(range(len(canonical_rows))) or row_map.get(
        "canonical_boundary_row_count"
    ) != len(canonical_rows):
        raise Lfm2SemanticPostfoldError(
            "Global residual canonical rows do not form one exact partition."
        )

    report_value = stack.get("global_residual_closure_receipt")
    if not isinstance(report_value, Mapping):
        raise Lfm2SemanticPostfoldError(
            "Global residual closure omitted its installation receipt."
        )
    report = dict(report_value)
    report_sha = report.pop("report_sha256", None)
    closure_version = stack.get("global_residual_closure_version")
    correction_frontier = stack.get("global_residual_correction_scale_frontier")
    correction_scale = stack.get("global_residual_correction_scale")
    residual_backend = stack.get("global_residual_backend")
    if (
        closure_version not in {"producer_aware_v2a", "output_writer_v1_diagnostic"}
        or residual_backend not in {"auto", "cpu", "cuda"}
        or not isinstance(correction_frontier, list)
        or not correction_frontier
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in correction_frontier
        )
        or [float(value) for value in correction_frontier]
        != sorted(set(float(value) for value in correction_frontier))
        or isinstance(correction_scale, bool)
        or not isinstance(correction_scale, (int, float))
        or not math.isfinite(float(correction_scale))
        or float(correction_scale)
        not in {float(value) for value in correction_frontier}
        or not isinstance(
            stack.get("global_residual_correction_scale_policy_sha256"), str
        )
        or not isinstance(stack.get("global_residual_closure_policy_sha256"), str)
    ):
        raise Lfm2SemanticPostfoldError(
            "Global residual closure mode, backend, or correction-scale policy is "
            "invalid."
        )
    expected_closure_policy = Lfm2GlobalResidualClosurePolicy(backend=residual_backend)
    expected_scale_policy = Lfm2ProducerAwareCorrectionScalePolicy(
        materialization_frontier=tuple(float(value) for value in correction_frontier),
        materialization_scale=float(correction_scale),
    )
    if (
        stack.get("global_residual_closure_policy_sha256")
        != expected_closure_policy.authentication_sha256
        or stack.get("global_residual_correction_scale_policy_sha256")
        != expected_scale_policy.authentication_sha256
    ):
        raise Lfm2SemanticPostfoldError(
            "Global residual closure controls failed policy authentication."
        )
    before_by_writer: Mapping[str, Any]
    after_by_writer: Mapping[str, Any]
    installed_ids: list[str]
    site_specifications = report.get("site_specifications")
    writer_site_count = report.get("writer_site_count")
    topology_site_ids = (
        [
            value.get("site_id")
            for value in site_specifications
            if isinstance(value, Mapping)
        ]
        if isinstance(site_specifications, list)
        else []
    )
    if (
        report_sha != canonical_sha256(report)
        or stack.get("global_residual_closure_receipt_sha256") != report_sha
        or report.get("evidence_package_sha256")
        != stack.get("global_residual_closure_evidence_sha256")
        or report.get("closure_result_sha256")
        != stack.get("global_residual_closure_result_sha256")
        or report.get("closure_scope") != stack.get("global_residual_closure_scope")
        or not isinstance(site_specifications, list)
        or len(site_specifications) != writer_site_count
        or len(topology_site_ids) != writer_site_count
        or any(not isinstance(name, str) or not name for name in topology_site_ids)
        or len(set(topology_site_ids)) != writer_site_count
        or report.get("final_w2_unchanged") is not True
        or report.get("final_w2_installation_count") != 0
        or report.get("quality_gate") is not False
        or report.get("numeric_diagnostics_used_for_auto_veto") is not False
        or report.get("verdict_required_for_continuation") is not False
        or stack.get("global_residual_closure_numeric_quality_gate") is not False
        or stack.get("global_residual_closure_manual_quality_gate") is not False
        or stack.get("decoded_behavior_review_is_evidence_for_understanding")
        is not True
        or stack.get("decoded_behavior_verdict_required_for_continuation") is not False
    ):
        raise Lfm2SemanticPostfoldError(
            "Global residual closure installation receipt is invalid."
        )

    if closure_version == "producer_aware_v2a":
        topology_value = report.get("topology_receipt")
        dense_value = report.get("dense_transaction_receipt")
        installation_value = report.get("physical_installation_receipt")
        scale_policy_value = report.get("correction_scale_policy")
        closure_policy_value = report.get("closure_policy")
        materialization_compute_value = report.get(
            "materialization_compute_execution_receipt"
        )
        physical_specs = report.get("physical_parameter_specifications")
        if not all(
            isinstance(value, Mapping)
            for value in (
                topology_value,
                dense_value,
                installation_value,
                scale_policy_value,
                closure_policy_value,
                materialization_compute_value,
            )
        ) or not isinstance(physical_specs, list):
            raise Lfm2SemanticPostfoldError(
                "Producer-aware closure omitted its authenticated physical receipts."
            )
        topology = dict(topology_value)
        topology_sha = topology.pop("topology_sha256", None)
        dense = dict(dense_value)
        dense_sha = dense.pop("report_sha256", None)
        installation = dict(installation_value)
        installation_sha = installation.pop("report_sha256", None)
        scale_policy = dict(scale_policy_value)
        scale_policy_sha = scale_policy.pop("authentication_sha256", None)
        closure_policy = dict(closure_policy_value)
        closure_policy_sha = closure_policy.pop("authentication_sha256", None)
        materialization_compute = dict(materialization_compute_value)
        materialization_compute_sha = materialization_compute.pop(
            "report_sha256",
            None,
        )
        before_value = report.get("dense_live_physical_sha256_before")
        after_value = report.get("dense_live_physical_sha256_after")
        installed_value = report.get("installed_parameter_ids")
        dense_parameter_ids = dense.get("parameter_ids")
        dense_roles = dense.get("roles")
        dense_layers = dense.get("layer_indices")
        final_parameter_id = report.get("final_w2_parameter_id")
        layer_count = report.get("layer_count")
        expected_site_orders = [
            order
            for layer_index in range(layer_count if type(layer_count) is int else 0)
            for order in (
                6 * layer_index,
                6 * layer_index + 1,
                6 * layer_index + 3,
                6 * layer_index + 4,
            )
        ]
        spec_ids = [
            value.get("parameter_id") if isinstance(value, Mapping) else None
            for value in physical_specs
        ]
        deferred_specs = [
            value
            for value in physical_specs
            if isinstance(value, Mapping) and value.get("deferred") is True
        ]
        installed_specs = [
            value
            for value in physical_specs
            if isinstance(value, Mapping) and value.get("installed") is True
        ]
        changed_ids = report.get("changed_physical_parameter_ids")
        if (
            report.get("schema")
            != "neural-forge-lfm2-producer-aware-semantic-integration-v2a"
            or report.get("closure_backend") != "producer_aware_v2a"
            or report.get("global_residual_backend") != residual_backend
            or report.get("closure_policy_sha256") != closure_policy_sha
            or closure_policy_sha != stack.get("global_residual_closure_policy_sha256")
            or closure_policy_sha != canonical_sha256(closure_policy)
            or closure_policy.get("backend") != residual_backend
            or materialization_compute_sha != canonical_sha256(materialization_compute)
            or materialization_compute_sha
            != report.get("materialization_compute_execution_receipt_sha256")
            or report.get("materialization_compute_device")
            != stack.get("fold_compute_device", "cpu")
            or materialization_compute.get("requested_device")
            != stack.get("fold_compute_device", "cpu")
            or materialization_compute.get("schema")
            != "neural-forge-lfm2-producer-aware-dense-absorption-compute-v1"
            or materialization_compute.get("kind")
            != "producer_aware_dense_absorption_compute_execution"
            or materialization_compute.get("resolved_device")
            != stack.get("fold_compute_device", "cpu")
            or materialization_compute.get("backend")
            != str(stack.get("fold_compute_device", "cpu")).partition(":")[0]
            or materialization_compute.get("compute_dtype") != "float64"
            or materialization_compute.get("archive_devices")
            != {
                "layer_seed_weights": "cpu",
                "output_basis": "cpu",
                "coefficient_blocks": "cpu",
            }
            or materialization_compute.get("output_devices")
            != {
                "transaction_seed_weights": "cpu",
                "dense_corrected_weights": "cpu",
            }
            or any(
                materialization_compute.get(name) is not True
                for name in (
                    "output_basis_staged_to_resolved_device",
                    "coefficient_blocks_staged_to_resolved_device",
                    "active_layer_seed_weights_staged_to_resolved_device",
                    "active_layers_staged_one_at_a_time",
                    "system_memory_is_immutable_archive",
                )
            )
            or materialization_compute.get("silent_device_fallback_performed")
            is not False
            or any(
                materialization_compute.get(name) != 0
                for name in (
                    "autograd_grad_calls",
                    "backward_calls",
                    "optimizer_steps",
                    "training_steps",
                    "parameters_pruned",
                    "parameters_quantized",
                )
            )
            or dense.get("compute_execution_receipt_sha256")
            != materialization_compute_sha
            or dense.get("compute_execution_receipt") != materialization_compute_value
            or report.get("correction_scale_policy_sha256") != scale_policy_sha
            or scale_policy_sha
            != stack.get("global_residual_correction_scale_policy_sha256")
            or scale_policy_sha != canonical_sha256(scale_policy)
            or scale_policy.get("materialization_frontier")
            != [float(value) for value in correction_frontier]
            or scale_policy.get("materialization_scale") != float(correction_scale)
            or report.get("materialization_frontier")
            != [float(value) for value in correction_frontier]
            or report.get("materialization_scale") != float(correction_scale)
            or report.get("materialization_scale_is_execution_variant_not_winner")
            is not True
            or report.get("frontier_member_selected_as_winner") is not False
            or report.get(
                "separate_fold_decode_retain_required_for_each_frontier_scale"
            )
            is not True
            or stack.get("global_residual_producer_aware_v2a") is not True
            or stack.get("global_residual_v1_diagnostic_baseline") is not False
            or stack.get("global_residual_dynamic_layer_count") != layer_count
            or type(layer_count) is not int
            or layer_count <= 0
            or writer_site_count != 4 * layer_count
            or stack.get("global_residual_producer_aware_site_count")
            != writer_site_count
            or report.get("four_sites_per_layer") is not True
            or report.get("dynamic_layer_count_derived_from_authenticated_evidence")
            is not True
            or topology_sha != canonical_sha256(topology)
            or topology_sha != report.get("topology_sha256")
            or topology.get("sites") != site_specifications
            or [value.get("causal_order") for value in site_specifications]
            != expected_site_orders
            or [value.get("layer_index") for value in site_specifications]
            != [index for index in range(layer_count) for _ in range(4)]
            or dense_sha != canonical_sha256(dense)
            or dense_sha != report.get("dense_transaction_sha256")
            or installation_sha != canonical_sha256(installation)
            or installation_sha != report.get("installation_receipt_sha256")
            or installation.get("transaction_report_sha256") != dense_sha
            or not isinstance(dense_parameter_ids, list)
            or not isinstance(dense_roles, list)
            or not isinstance(dense_layers, list)
            or len(dense_parameter_ids) != len(physical_specs)
            or len(set(dense_parameter_ids)) != len(dense_parameter_ids)
            or spec_ids != dense_parameter_ids
            or [value.get("role") for value in physical_specs] != dense_roles
            or [value.get("layer_index") for value in physical_specs] != dense_layers
            or len(deferred_specs) != 1
            or deferred_specs[0].get("parameter_id") != final_parameter_id
            or deferred_specs[0].get("installed") is not False
            or any(value.get("deferred") is not False for value in installed_specs)
            or not isinstance(installed_value, list)
            or installed_value
            != [value.get("parameter_id") for value in installed_specs]
            or report.get("installed_parameter_count") != len(installed_value)
            or report.get("all_nondeferred_physical_parameters_installed") is not True
            or installation.get("installed_parameter_ids") != installed_value
            or installation.get("deferred_parameter_ids") != [final_parameter_id]
            or installation.get("all_nondeferred_physical_parameters_installed")
            is not True
            or not isinstance(before_value, Mapping)
            or not isinstance(after_value, Mapping)
            or set(before_value) != set(installed_value)
            or set(after_value) != set(installed_value)
            or installation.get("post_install_sha256") != dict(after_value)
            or not isinstance(changed_ids, list)
            or changed_ids
            != sorted(
                name
                for name in installed_value
                if before_value.get(name) != after_value.get(name)
            )
            or report.get("autograd_grad_calls") != 0
            or report.get("backward_calls") != 0
            or report.get("optimizer_steps") != 0
            or report.get("training_steps") != 0
            or stack.get("global_residual_full_physical_parameter_transaction")
            is not True
            or stack.get("global_residual_input_and_output_producer_roles_folded")
            is not True
            or stack.get(
                "global_residual_materialization_scale_is_execution_variant_not_winner"
            )
            is not True
            or stack.get("global_residual_frontier_member_selected_as_winner")
            is not False
            or stack.get(
                "global_residual_each_frontier_scale_requires_separate_fold_decode_retain"
            )
            is not True
            or stack.get("global_residual_materialization_compute_device")
            != stack.get("fold_compute_device", "cpu")
            or stack.get(
                "global_residual_materialization_compute_execution_receipt_sha256"
            )
            != materialization_compute_sha
            or stack.get("global_residual_materialization_compute_execution_receipt")
            != materialization_compute_value
        ):
            raise Lfm2SemanticPostfoldError(
                "Producer-aware global residual physical transaction is invalid."
            )
        before_by_writer = before_value
        after_by_writer = after_value
        installed_ids = installed_value
    else:
        expected_live_writer_count = (
            writer_site_count - 1
            if type(writer_site_count) is int and writer_site_count > 1
            else -1
        )
        before_value = report.get("dense_live_writer_sha256_before")
        after_value = report.get("dense_live_writer_sha256_after")
        installed_value = report.get("installed_writer_site_ids")
        if (
            correction_frontier != [1.0]
            or float(correction_scale) != 1.0
            or stack.get("global_residual_producer_aware_v2a") is not False
            or stack.get("global_residual_v1_diagnostic_baseline") is not True
            or stack.get("global_residual_v1_shared_attention_head_dimension")
            is not True
            or stack.get("global_residual_general_per_site_head_dimensions_claimed")
            is not False
            or not report.get("closure_scope", "").endswith(
                "output_writers_only_baseline_v1"
            )
            or report.get("v2_input_transport_extension_roadmap", {}).get(
                "implemented_in_this_result"
            )
            is not False
            or report.get("installed_writer_count") != expected_live_writer_count
            or not isinstance(installed_value, list)
            or len(installed_value) != expected_live_writer_count
            or len(set(installed_value)) != expected_live_writer_count
            or any(not isinstance(name, str) or not name for name in installed_value)
            or installed_value != topology_site_ids[:-1]
            or report.get("final_w2_site_id") != topology_site_ids[-1]
            or not isinstance(before_value, Mapping)
            or not isinstance(after_value, Mapping)
            or set(before_value) != set(installed_value)
            or set(after_value) != set(installed_value)
        ):
            raise Lfm2SemanticPostfoldError(
                "Output-writer v1 diagnostic closure receipt is invalid."
            )
        before_by_writer = before_value
        after_by_writer = after_value
        installed_ids = installed_value

    local_after = digest(
        stack.get("causal_layer_stack_model_state_after_sha256"),
        "causal layer-stack state after",
    )
    closure_before = digest(
        stack.get("global_residual_closure_model_state_before_sha256"),
        "global residual closure state before",
    )
    closure_after = digest(
        stack.get("global_residual_closure_model_state_after_sha256"),
        "global residual closure state after",
    )
    stack_after = digest(
        stack.get("model_state_after_sha256"),
        "semantic layer-stack final state",
    )
    local_changed = canonical_names(
        stack.get("causal_layer_stack_changed_state_keys"),
        "causal layer-stack changed keys",
    )
    closure_changed = canonical_names(
        stack.get("global_residual_closure_changed_state_keys"),
        "global residual closure changed keys",
    )
    composed_changed = canonical_names(
        stack.get("composed_transition_changed_state_keys"),
        "composed transition changed keys",
    )
    final_changed = canonical_names(
        stack.get("changed_state_keys"),
        "semantic layer-stack final changed keys",
    )
    report_changed = tuple(
        sorted(
            name
            for name in installed_ids
            if before_by_writer.get(name) != after_by_writer.get(name)
        )
    )
    if (
        local_after != closure_before
        or closure_after != stack_after
        or closure_changed != report_changed
        or composed_changed != tuple(sorted(set(local_changed).union(closure_changed)))
        or not set(final_changed).issubset(composed_changed)
        or stack.get("global_residual_closure_live_final_w2_unchanged") is not True
        or stack.get("global_residual_closure_parent_histories_compacted") is not True
        or report.get("corrected_final_w2_prior_sha256")
        != stack.get("global_residual_corrected_final_w2_prior_sha256")
        or report.get("final_layer_mlp_down_weight_prior_authentication_sha256")
        != stack.get("final_layer_mlp_down_weight_prior_authentication_sha256")
        or report.get("final_layer_mlp_down_weight_prior_tensor_sha256")
        != stack.get("final_layer_mlp_down_weight_prior_tensor_sha256")
        or report.get("final_layer_mlp_down_weight_prior_install_count") != 0
        or report.get("final_layer_mlp_down_weight_prior_closure_lineage_sha256")
        != stack.get("final_layer_mlp_down_weight_prior_closure_lineage_sha256")
        or report.get("corrected_final_w2_closure_lineage_sha256")
        != stack.get("global_residual_corrected_final_w2_closure_lineage_sha256")
    ):
        raise Lfm2SemanticPostfoldError(
            "Global residual closure state transition or final-prior lineage "
            "is invalid."
        )
    return MappingProxyType(
        {
            "causal_layer_stack_model_state_after_sha256": local_after,
            "causal_layer_stack_changed_state_keys": list(local_changed),
            "closure_model_state_before_sha256": closure_before,
            "closure_model_state_after_sha256": closure_after,
            "closure_changed_state_keys": list(closure_changed),
            "composed_transition_changed_state_keys": list(composed_changed),
            "closure_receipt_sha256": report_sha,
            "semantic_row_map_sha256": row_map_sha,
            "global_residual_closure_version": closure_version,
            "global_residual_backend": residual_backend,
            "correction_scale_frontier": [
                float(value) for value in correction_frontier
            ],
            "correction_scale": float(correction_scale),
            **(
                {
                    "materialization_compute_device": report[
                        "materialization_compute_device"
                    ],
                    "materialization_compute_execution_receipt_sha256": report[
                        "materialization_compute_execution_receipt_sha256"
                    ],
                }
                if closure_version == "producer_aware_v2a"
                else {}
            ),
            "materialization_scale_is_execution_variant_not_winner": (
                closure_version == "producer_aware_v2a"
            ),
            "frontier_member_selected_as_winner": False,
            "two_transition_chain_authenticated": True,
            "numeric_or_manual_quality_gate_used": False,
        }
    )


def _parameter_runtime_snapshot(model: Any) -> Mapping[str, Any]:
    """Capture topology, object/storage identity, and exact gradient state."""

    snapshot: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        snapshot[name] = {
            "shape": tuple(int(size) for size in parameter.shape),
            "dtype": str(parameter.dtype),
            "device": str(parameter.device),
            "parameter_id": id(parameter),
            "storage_data_ptr": int(parameter.data_ptr()),
            "requires_grad": bool(parameter.requires_grad),
            "gradient": (
                None
                if gradient is None
                else {
                    "shape": tuple(int(size) for size in gradient.shape),
                    "dtype": str(gradient.dtype),
                    "device": str(gradient.device),
                    "tensor_id": id(gradient),
                    "storage_data_ptr": int(gradient.data_ptr()),
                    "sha256": _tensor_sha256(gradient),
                }
            ),
        }
    return MappingProxyType(snapshot)


def _parameter_runtime_proof(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> Mapping[str, Any]:
    before_names = tuple(sorted(before))
    after_names = tuple(sorted(after))
    topology_fields = ("shape", "dtype", "device")
    topology_preserved = bool(
        before_names == after_names
        and all(
            tuple(before[name][field] for field in topology_fields)
            == tuple(after[name][field] for field in topology_fields)
            for name in before_names
        )
    )
    identities_preserved = bool(
        topology_preserved
        and all(
            before[name]["parameter_id"] == after[name]["parameter_id"]
            for name in before_names
        )
    )
    storage_preserved = bool(
        topology_preserved
        and all(
            before[name]["storage_data_ptr"] == after[name]["storage_data_ptr"]
            for name in before_names
        )
    )
    gradient_state_preserved = bool(
        topology_preserved
        and all(
            before[name]["requires_grad"] == after[name]["requires_grad"]
            and before[name]["gradient"] == after[name]["gradient"]
            for name in before_names
        )
    )
    topology_body = {
        name: {field: before[name][field] for field in topology_fields}
        for name in before_names
    }
    after_topology_body = {
        name: {field: after[name][field] for field in topology_fields}
        for name in after_names
    }
    return MappingProxyType(
        {
            "parameter_topology_sha256_before": canonical_sha256(topology_body),
            "parameter_topology_sha256_after": canonical_sha256(after_topology_body),
            "parameter_topology_preserved": topology_preserved,
            "parameter_identities_preserved": identities_preserved,
            "parameter_storage_preserved": storage_preserved,
            "gradient_state_preserved": gradient_state_preserved,
        }
    )


def install_lfm2_self_dual_tied_interface_gauge(
    target_model: Any,
    interface: TiedCausalInterfaceFold,
) -> tuple[TiedCausalInterfaceFold, Mapping[str, Any]]:
    """Whiten and transactionally install one global tied/residual gauge."""

    torch = _torch()
    self_dual = whiten_tied_causal_interface(interface)
    if self_dual.report.get("schema") != TIED_CAUSAL_INTERFACE_SELF_DUAL_SCHEMA:
        raise Lfm2SemanticPostfoldError(
            "Tied-interface whitening returned an unsupported gauge."
        )
    input_module = target_model.get_input_embeddings()
    output_module = target_model.get_output_embeddings()
    input_weight = getattr(input_module, "weight", None)
    output_weight = getattr(output_module, "weight", None)
    if (
        not isinstance(input_weight, torch.nn.Parameter)
        or output_weight is not input_weight
        or tuple(input_weight.shape) != tuple(self_dual.target_weight.shape)
        or getattr(output_module, "bias", None) is not None
    ):
        raise Lfm2SemanticPostfoldError(
            "Child has no unbiased native tied table matching the self-dual interface."
        )
    before_state = fingerprint_lfm2_parent(target_model).identity_sha256
    before_hashes = _state_tensor_hashes(target_model)
    before_runtime = _parameter_runtime_snapshot(target_model)
    before_count = _parameter_count(target_model)
    incumbent = input_weight.detach().to(device="cpu").contiguous().clone()
    tied_pointer = int(input_weight.untyped_storage().data_ptr())
    tied_state_names = tuple(
        sorted(
            name
            for name, value in target_model.state_dict(keep_vars=True).items()
            if isinstance(value, torch.Tensor)
            and int(value.untyped_storage().data_ptr()) == tied_pointer
        )
    )
    if not tied_state_names:
        raise Lfm2SemanticPostfoldError(
            "Native tied table has no authenticated state-dict aliases."
        )
    materialized = self_dual.target_weight.to(
        device=input_weight.device,
        dtype=input_weight.dtype,
    )
    try:
        with torch.no_grad():
            input_weight.copy_(materialized)
        after_state = fingerprint_lfm2_parent(target_model).identity_sha256
        after_hashes = _state_tensor_hashes(target_model)
        runtime_proof = _parameter_runtime_proof(
            before_runtime,
            _parameter_runtime_snapshot(target_model),
        )
        changed_state_keys = _changed_state_keys(before_hashes, after_hashes)
        source_operator = (
            interface.producer_encoder.to(torch.float64)
            @ interface.target_weight.to(torch.float64).T
        )
        materialized_operator = (
            self_dual.producer_encoder.to(torch.float64)
            @ input_weight.detach().to(device="cpu", dtype=torch.float64).T
        )
        operator_residual = materialized_operator - source_operator
        materialized_relative_error = float(
            torch.linalg.vector_norm(operator_residual).item()
            / max(
                float(torch.linalg.vector_norm(source_operator).item()),
                torch.finfo(torch.float64).tiny,
            )
        )
        materialized_maximum_error = float(operator_residual.abs().max().item())
        runtime_preserved = all(
            runtime_proof[name] is True
            for name in (
                "parameter_topology_preserved",
                "parameter_identities_preserved",
                "parameter_storage_preserved",
                "gradient_state_preserved",
            )
        )
        if (
            not runtime_preserved
            or _parameter_count(target_model) != before_count
            or target_model.get_input_embeddings().weight
            is not target_model.get_output_embeddings().weight
            or not torch.equal(input_weight.detach(), materialized)
            or tuple(changed_state_keys) != tied_state_names
            or after_state == before_state
            or not math.isfinite(materialized_relative_error)
            or materialized_relative_error > 0.01
        ):
            raise Lfm2SemanticPostfoldError(
                "Self-dual gauge installation changed topology/aliases or lost "
                "the tied source-to-vocabulary operator."
            )
    except BaseException:
        with torch.no_grad():
            input_weight.copy_(incumbent.to(device=input_weight.device))
        rollback_runtime = _parameter_runtime_proof(
            before_runtime,
            _parameter_runtime_snapshot(target_model),
        )
        if (
            fingerprint_lfm2_parent(target_model).identity_sha256 != before_state
            or _state_tensor_hashes(target_model) != before_hashes
            or not all(
                rollback_runtime[name] is True
                for name in (
                    "parameter_topology_preserved",
                    "parameter_identities_preserved",
                    "parameter_storage_preserved",
                    "gradient_state_preserved",
                )
            )
        ):
            raise Lfm2SemanticPostfoldError(
                "Self-dual tied-interface rollback was not exact."
            )
        raise
    body = {
        "schema": "neural-forge-lfm2-self-dual-tied-gauge-install-v1",
        "source_interface_report_sha256": interface.report["report_sha256"],
        "self_dual_interface_report_sha256": self_dual.report["report_sha256"],
        "model_state_before_sha256": before_state,
        "model_state_after_sha256": after_state,
        "changed_state_keys": list(changed_state_keys),
        "native_tied_state_aliases": list(tied_state_names),
        "source_target_weight_sha256": _interface_tensor_sha256(
            interface.target_weight
        ),
        "self_dual_target_weight_sha256": _interface_tensor_sha256(
            self_dual.target_weight
        ),
        "installed_target_weight_sha256": _interface_tensor_sha256(input_weight),
        "self_dual_producer_encoder_sha256": _interface_tensor_sha256(
            self_dual.producer_encoder
        ),
        "materialized_source_to_vocabulary_operator_relative_error": (
            materialized_relative_error
        ),
        "materialized_source_to_vocabulary_operator_maximum_absolute_error": (
            materialized_maximum_error
        ),
        "native_tie_preserved": True,
        "parameter_topology_preserved": True,
        "parameter_identities_preserved": True,
        "parameter_storage_preserved": True,
        "gradient_state_preserved": True,
        "unique_parameter_count_preserved": True,
        "full_vocabulary_rows_used": int(input_weight.shape[0]),
        "target_width": int(input_weight.shape[1]),
        "global_residual_gauge_frozen": True,
        "independent_per_layer_gauge_rotation_allowed": False,
        "parent_model_forward_calls": 0,
        "child_model_forward_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    return self_dual, MappingProxyType(_sealed(body, "result_sha256"))


def install_lfm2_semantic_foundation_anchor_selection(
    target_model: Any,
    interface: TiedCausalInterfaceFold,
    selection_binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Install exactly one authenticated lazy anchor candidate transactionally."""

    torch = _torch()
    selection = _authenticate_foundation_anchor_selection(
        selection_binding,
        selected_interface=interface,
    )
    input_weight = getattr(target_model.get_input_embeddings(), "weight", None)
    output_weight = getattr(target_model.get_output_embeddings(), "weight", None)
    if (
        not isinstance(input_weight, torch.nn.Parameter)
        or output_weight is not input_weight
        or tuple(input_weight.shape) != tuple(interface.target_weight.shape)
    ):
        raise Lfm2SemanticPostfoldError(
            "Child has no native tied table matching the selected anchor candidate."
        )
    incumbent = input_weight.detach().to(device="cpu").contiguous().clone()
    if _foundation_lineage_tensor_sha256(incumbent) != selection.get(
        "saved_target_weight_sha256"
    ):
        raise Lfm2SemanticPostfoldError(
            "Saved child changed after compatibility-default anchor verification."
        )
    materialized = interface.target_weight.to(
        device=input_weight.device,
        dtype=input_weight.dtype,
    )
    before_state = fingerprint_lfm2_parent(target_model).identity_sha256
    before_hashes = _state_tensor_hashes(target_model)
    before_runtime = _parameter_runtime_snapshot(target_model)
    before_count = _parameter_count(target_model)
    install = not torch.equal(input_weight.detach(), materialized)
    try:
        if install:
            with torch.no_grad():
                input_weight.copy_(materialized)
        after_state = fingerprint_lfm2_parent(target_model).identity_sha256
        after_hashes = _state_tensor_hashes(target_model)
        runtime_proof = _parameter_runtime_proof(
            before_runtime,
            _parameter_runtime_snapshot(target_model),
        )
        changed_state_keys = _changed_state_keys(before_hashes, after_hashes)
        tied_pointer = int(input_weight.untyped_storage().data_ptr())
        tied_state_names = tuple(
            sorted(
                name
                for name, value in target_model.state_dict(keep_vars=True).items()
                if isinstance(value, torch.Tensor)
                and int(value.untyped_storage().data_ptr()) == tied_pointer
            )
        )
        if (
            _parameter_count(target_model) != before_count
            or target_model.get_output_embeddings().weight is not input_weight
            or not all(runtime_proof.values())
            or not torch.equal(input_weight.detach(), materialized)
            or (
                install
                and (
                    after_state == before_state
                    or tuple(changed_state_keys) != tied_state_names
                )
            )
            or (
                not install
                and (after_state != before_state or bool(changed_state_keys))
            )
        ):
            raise Lfm2SemanticPostfoldError(
                "Anchor selection installation changed topology, aliases, or "
                "unexpected child state."
            )
    except BaseException:
        with torch.no_grad():
            input_weight.copy_(incumbent.to(device=input_weight.device))
        raise
    body = {
        "schema": "neural-forge-lfm2-semantic-foundation-anchor-install-v1",
        "interface_anchor_selection": dict(selection),
        "interface_anchor_selection_sha256": selection["selection_sha256"],
        "selected_anchor_candidate_index": selection["selected_anchor_candidate_index"],
        "selected_anchor_candidate_recipe_sha256": selection[
            "selected_anchor_candidate_recipe_sha256"
        ],
        "selected_interface_report_sha256": selection[
            "selected_interface_report_sha256"
        ],
        "selected_target_weight_sha256": selection["selected_target_weight_sha256"],
        "model_state_before_sha256": before_state,
        "model_state_after_sha256": after_state,
        "changed_state_keys": list(changed_state_keys),
        "installation_performed": install,
        "parameter_copy_calls": int(install),
        "parameter_element_count_before": before_count,
        "parameter_element_count_after": _parameter_count(target_model),
        "parameter_count_changed": False,
        "selected_candidate_installed_inside_authenticated_pipeline_session": True,
        "numeric_metric_gate_applied": False,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    return MappingProxyType(_sealed(body, "result_sha256"))


def install_lfm2_tied_residual_congruence_gauge(
    target_model: Any,
    interface: TiedCausalInterfaceFold,
    mapped_parent_tied_table: Any,
    *,
    mode: str = TIED_RESIDUAL_STRONG_CONGRUENCE_MODE,
    anchor_base_target_weight: Any | None = None,
    anchor_source_residual_weight: Any | None = None,
) -> tuple[TiedCausalInterfaceFold, Mapping[str, Any]]:
    """Transactionally install a self-dual control or strong tied congruence.

    The strong candidate keeps the self-dual producer basis ``B`` fixed and
    installs ``mapped_parent_tied_table @ B`` into the child's one native tied
    embedding/LM-head parameter.  The control retains the historical
    self-dual output factor.  Both paths use the same topology-preserving
    transaction and a BF16-exact materialization witness.
    """

    torch = _torch()
    if mode not in TIED_RESIDUAL_CONGRUENCE_MODES:
        raise Lfm2SemanticPostfoldError(
            f"Unsupported tied residual interface mode: {mode!r}."
        )
    candidate = compile_tied_residual_congruence_candidate(
        interface,
        mapped_parent_tied_table,
        mode=mode,
        anchor_base_target_weight=anchor_base_target_weight,
        anchor_source_residual_weight=anchor_source_residual_weight,
    )
    selected = candidate.interface
    input_module = target_model.get_input_embeddings()
    output_module = target_model.get_output_embeddings()
    input_weight = getattr(input_module, "weight", None)
    output_weight = getattr(output_module, "weight", None)
    if (
        not isinstance(input_weight, torch.nn.Parameter)
        or output_weight is not input_weight
        or input_weight.dtype != torch.bfloat16
        or tuple(input_weight.shape) != tuple(selected.target_weight.shape)
        or getattr(output_module, "bias", None) is not None
    ):
        raise Lfm2SemanticPostfoldError(
            "Child has no unbiased native BF16 tied table matching the "
            "residual-congruence candidate."
        )

    before_state = fingerprint_lfm2_parent(target_model).identity_sha256
    before_hashes = _state_tensor_hashes(target_model)
    before_runtime = _parameter_runtime_snapshot(target_model)
    before_count = _parameter_count(target_model)
    incumbent = input_weight.detach().to(device="cpu").contiguous().clone()
    tied_pointer = int(input_weight.untyped_storage().data_ptr())
    tied_state_names = tuple(
        sorted(
            name
            for name, value in target_model.state_dict(keep_vars=True).items()
            if isinstance(value, torch.Tensor)
            and int(value.untyped_storage().data_ptr()) == tied_pointer
        )
    )
    if not tied_state_names:
        raise Lfm2SemanticPostfoldError(
            "Native tied table has no authenticated state-dict aliases."
        )
    materialized = candidate.materialized_target_weight.to(
        device=input_weight.device,
        dtype=input_weight.dtype,
    )
    install = not torch.equal(input_weight.detach(), materialized)
    try:
        if install:
            with torch.no_grad():
                input_weight.copy_(materialized)
        after_state = fingerprint_lfm2_parent(target_model).identity_sha256
        after_hashes = _state_tensor_hashes(target_model)
        runtime_proof = _parameter_runtime_proof(
            before_runtime,
            _parameter_runtime_snapshot(target_model),
        )
        changed_state_keys = _changed_state_keys(before_hashes, after_hashes)
        after_tied_pointer = int(input_weight.untyped_storage().data_ptr())
        after_tied_state_names = tuple(
            sorted(
                name
                for name, value in target_model.state_dict(keep_vars=True).items()
                if isinstance(value, torch.Tensor)
                and int(value.untyped_storage().data_ptr()) == after_tied_pointer
            )
        )
        runtime_preserved = all(
            runtime_proof[name] is True
            for name in (
                "parameter_topology_preserved",
                "parameter_identities_preserved",
                "parameter_storage_preserved",
                "gradient_state_preserved",
            )
        )
        mutation_valid = bool(
            install
            and after_state != before_state
            and tuple(changed_state_keys) == tied_state_names
        )
        noop_valid = bool(
            not install and after_state == before_state and not changed_state_keys
        )
        installed_cpu = input_weight.detach().to(device="cpu").contiguous()
        installed64 = installed_cpu.to(torch.float64)
        selected64 = selected.target_weight.to(torch.float64)
        materialization_residual = installed64 - selected64
        materialization_relative_error = float(
            torch.linalg.vector_norm(materialization_residual).item()
            / max(
                float(torch.linalg.vector_norm(selected64).item()),
                torch.finfo(torch.float64).tiny,
            )
        )
        materialization_maximum_error = float(
            materialization_residual.abs().max().item()
        )
        if (
            not runtime_preserved
            or _parameter_count(target_model) != before_count
            or target_model.get_input_embeddings().weight
            is not target_model.get_output_embeddings().weight
            or after_tied_state_names != tied_state_names
            or not torch.equal(installed_cpu, candidate.materialized_target_weight)
            or candidate.report.get("bf16_materialization_exact") is not True
            or (
                mode == TIED_RESIDUAL_STRONG_CONGRUENCE_MODE
                and candidate.report.get("bf16_congruence_exact") is not True
            )
            or not (mutation_valid or noop_valid)
            or not math.isfinite(materialization_relative_error)
        ):
            raise Lfm2SemanticPostfoldError(
                "Tied residual-congruence installation changed topology, "
                "aliases, parameter count, gradients, or expected BF16 bytes."
            )
    except BaseException:
        with torch.no_grad():
            input_weight.copy_(incumbent.to(device=input_weight.device))
        rollback_runtime = _parameter_runtime_proof(
            before_runtime,
            _parameter_runtime_snapshot(target_model),
        )
        if (
            fingerprint_lfm2_parent(target_model).identity_sha256 != before_state
            or _state_tensor_hashes(target_model) != before_hashes
            or not all(
                rollback_runtime[name] is True
                for name in (
                    "parameter_topology_preserved",
                    "parameter_identities_preserved",
                    "parameter_storage_preserved",
                    "gradient_state_preserved",
                )
            )
        ):
            raise Lfm2SemanticPostfoldError(
                "Tied residual-congruence rollback was not exact."
            )
        raise

    body = {
        "schema": "neural-forge-lfm2-tied-residual-congruence-install-v1",
        "status": (
            "installed-selected-tied-residual-interface"
            if install
            else "authenticated-selected-tied-residual-interface-noop"
        ),
        "mode": mode,
        "source_interface_report_sha256": interface.report["report_sha256"],
        "candidate_report_sha256": candidate.report["report_sha256"],
        "model_state_before_sha256": before_state,
        "model_state_after_sha256": after_state,
        "changed_state_keys": list(changed_state_keys),
        "native_tied_state_aliases_before": list(tied_state_names),
        "native_tied_state_aliases_after": list(after_tied_state_names),
        "incumbent_target_weight_sha256": _interface_tensor_sha256(incumbent),
        "selected_fp64_target_weight_sha256": _interface_tensor_sha256(
            selected.target_weight
        ),
        "expected_bf16_target_weight_sha256": _interface_tensor_sha256(
            candidate.materialized_target_weight
        ),
        "installed_target_weight_sha256": _interface_tensor_sha256(input_weight),
        "selected_producer_encoder_sha256": _interface_tensor_sha256(
            selected.producer_encoder
        ),
        "anchor_residual_transport_applied": candidate.report[
            "anchor_residual_transport_applied"
        ],
        "anchor_base_target_weight_sha256": candidate.report[
            "anchor_base_target_weight_sha256"
        ],
        "anchor_base_target_weight_foundation_sha256": (
            None
            if anchor_base_target_weight is None
            else _foundation_lineage_tensor_sha256(anchor_base_target_weight)
        ),
        "anchor_source_target_residual_sha256": candidate.report[
            "anchor_source_target_residual_sha256"
        ],
        "anchor_source_target_residual_foundation_sha256": (
            None
            if anchor_source_residual_weight is None
            else _foundation_lineage_tensor_sha256(anchor_source_residual_weight)
        ),
        "anchor_selfdual_residual_sha256": candidate.report[
            "anchor_selfdual_residual_sha256"
        ],
        "installed_vs_selected_fp64_relative_error": materialization_relative_error,
        "installed_vs_selected_fp64_maximum_absolute_error": (
            materialization_maximum_error
        ),
        "producer_basis_orthogonality_authenticated": candidate.report[
            "producer_basis_orthogonality_authenticated"
        ],
        "fp64_congruence_exact": candidate.report["fp64_congruence_exact"],
        "bf16_materialization_exact": candidate.report["bf16_materialization_exact"],
        "bf16_congruence_exact": candidate.report["bf16_congruence_exact"],
        "congruent_vs_old_output_factor_relative_error": candidate.report[
            "congruent_vs_old_output_factor_relative_error"
        ],
        "congruent_vs_old_output_factor_row_cosine": candidate.report[
            "congruent_vs_old_output_factor_row_cosine"
        ],
        "congruent_vs_old_output_factor_column_cosine": candidate.report[
            "congruent_vs_old_output_factor_column_cosine"
        ],
        "native_tie_preserved": True,
        "native_tied_alias_set_preserved": True,
        **dict(runtime_proof),
        "unique_parameter_count_before": before_count,
        "unique_parameter_count_after": _parameter_count(target_model),
        "unique_parameter_count_preserved": True,
        "full_vocabulary_rows_used": int(input_weight.shape[0]),
        "target_width": int(input_weight.shape[1]),
        "global_residual_basis_frozen": True,
        "independent_per_layer_gauge_rotation_allowed": False,
        "installation_performed": install,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    return selected, MappingProxyType(_sealed(body, "result_sha256"))


def _ordinary_directory(value: str | os.PathLike[str], name: str) -> Path:
    lexical = Path(value).expanduser()
    if lexical.is_symlink():
        raise Lfm2SemanticPostfoldError(f"{name} must not be a symlink.")
    path = lexical.resolve()
    if not path.is_dir() or path.is_symlink():
        raise Lfm2SemanticPostfoldError(f"{name} must be a real local directory.")
    return path


def _ordinary_file(value: str | os.PathLike[str], name: str) -> Path:
    lexical = Path(value).expanduser()
    if lexical.is_symlink():
        raise Lfm2SemanticPostfoldError(f"{name} must not be a symlink.")
    path = lexical.resolve()
    if not path.is_file() or path.is_symlink():
        raise Lfm2SemanticPostfoldError(f"{name} must be a real local file.")
    return path


def _new_output_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise Lfm2SemanticPostfoldError(
            f"Refusing to replace existing output checkpoint {path}."
        )
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise Lfm2SemanticPostfoldError(
            "Output checkpoint parent must be a real existing directory."
        )
    return path


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _local_tokenizer_source_paths(tokenizer: Any) -> tuple[Path, ...]:
    """Collect already-local tokenizer roots without invoking tokenizer code."""

    attributes = getattr(tokenizer, "__dict__", None)
    if not isinstance(attributes, Mapping):
        return ()
    candidates = []
    for name in (
        "name_or_path",
        "tokenizer_file",
        "vocab_file",
        "merges_file",
        "sp_model_file",
    ):
        candidates.append(attributes.get(name))
    init_kwargs = attributes.get("init_kwargs")
    if isinstance(init_kwargs, Mapping):
        candidates.extend(
            init_kwargs.get(name)
            for name in (
                "name_or_path",
                "tokenizer_file",
                "vocab_file",
                "merges_file",
                "sp_model_file",
            )
        )
    paths = set()
    for value in candidates:
        if not isinstance(value, (str, os.PathLike)):
            continue
        # Several local tokenizer implementations use an empty string to mean
        # "no source path".  Resolving that sentinel would turn it into the
        # process working directory and falsely make every nested output look
        # like it overlaps a tokenizer input.
        raw_value = os.fspath(value)
        if not isinstance(raw_value, str) or not raw_value.strip():
            continue
        path = Path(raw_value).expanduser().resolve()
        if path.exists() or path.is_symlink():
            paths.add(path)
    return tuple(sorted(paths, key=str))


def _validate_preterminal_capsule_path_disjointness(
    *,
    output_checkpoint_directory: str | os.PathLike[str],
    preterminal_capsule_directory: str | os.PathLike[str] | None,
    input_checkpoint_directory: str | os.PathLike[str],
    cache_directory: str | os.PathLike[str],
    bridge_directory: str | os.PathLike[str],
    parent_model_directory: str | os.PathLike[str],
    compiler_diagnostic_json: str | os.PathLike[str],
    interface_statistics_directory: str | os.PathLike[str] | None,
    interface_frontier_companion_json: str | os.PathLike[str] | None,
    donor_tokenizer: Any,
    target_tokenizer: Any,
) -> tuple[Path, Path | None]:
    """Reject capsule/output ancestry collisions before any artifact is loaded."""

    output = Path(output_checkpoint_directory).expanduser().resolve()
    if preterminal_capsule_directory is None:
        return output, None
    capsule = Path(preterminal_capsule_directory).expanduser().resolve()
    readonly = {
        "input checkpoint": Path(input_checkpoint_directory).expanduser().resolve(),
        "trace cache": Path(cache_directory).expanduser().resolve(),
        "tokenizer bridge": Path(bridge_directory).expanduser().resolve(),
        "parent checkpoint": Path(parent_model_directory).expanduser().resolve(),
        "compiler diagnostic": Path(compiler_diagnostic_json).expanduser().resolve(),
    }
    if interface_statistics_directory is not None:
        readonly["interface statistics"] = (
            Path(interface_statistics_directory).expanduser().resolve()
        )
    if interface_frontier_companion_json is not None:
        readonly["interface frontier companion"] = (
            Path(interface_frontier_companion_json).expanduser().resolve()
        )
    for index, path in enumerate(
        (
            *_local_tokenizer_source_paths(donor_tokenizer),
            *_local_tokenizer_source_paths(target_tokenizer),
        )
    ):
        readonly[f"local tokenizer source {index}"] = path
    collisions = [
        label
        for label, path in (("final output", output), *readonly.items())
        if _paths_overlap(capsule, path)
    ]
    if collisions:
        raise Lfm2SemanticPostfoldError(
            "Preterminal capsule path must be disjoint from every input and "
            f"output path; overlaps {', '.join(collisions)}."
        )
    return output, capsule


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise Lfm2SemanticPostfoldError(f"{label} is missing or unsafe.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2SemanticPostfoldError(f"Could not parse {label}.") from exc
    if not isinstance(value, dict):
        raise Lfm2SemanticPostfoldError(f"{label} must contain a JSON object.")
    return value


def _read_canonical_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    """Read one canonical runner artifact and retain its exact file identity."""

    path = _ordinary_file(path, label)
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2SemanticPostfoldError(f"Could not parse {label}.") from exc
    if not isinstance(value, dict):
        raise Lfm2SemanticPostfoldError(f"{label} must contain a JSON object.")
    expected = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    if raw != expected:
        raise Lfm2SemanticPostfoldError(f"{label} is not canonical JSON.")
    return value, hashlib.sha256(raw).hexdigest()


def _authenticated_c61_diagnostic(
    diagnostic_json: str | os.PathLike[str],
    *,
    expected_diagnostic_sha256: str,
) -> tuple[Mapping[str, Any], str]:
    path = _ordinary_file(diagnostic_json, "c61 compiler diagnostic")
    diagnostic, file_sha256 = _read_canonical_json(
        path,
        "c61 compiler diagnostic",
    )
    expected = _digest(
        expected_diagnostic_sha256,
        "expected c61 compiler diagnostic SHA-256",
    )
    verified = _verify_sealed(
        diagnostic,
        seal_name="receipt_sha256",
        label="c61 compiler diagnostic",
    )
    if (
        verified["receipt_sha256"] != expected
        or verified.get("schema")
        != "neural-forge-lfm2-cpcf-v5-selective-blank-fold-run-v1"
        or verified.get("status") != "compiled_in_memory_diagnostic_not_published"
    ):
        raise Lfm2SemanticPostfoldError(
            "c61 compiler diagnostic differs from its externally pinned receipt."
        )
    return MappingProxyType(verified), file_sha256


def _claims_semantic_foundation_schema(
    ancestry_json: str | os.PathLike[str],
) -> bool:
    """Peek only for dispatch; the selected authenticator remains authoritative."""

    try:
        path = Path(ancestry_json).expanduser().resolve()
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(value, Mapping)
        and value.get("schema") == LFM2_SEMANTIC_FOUNDATION_SCHEMA
    )


def _input_artifact_metadata(
    checkpoint: Path,
    *,
    expected_artifact_sha256: str,
) -> dict[str, Any]:
    expected = _digest(expected_artifact_sha256, "expected input artifact SHA-256")
    value = _read_json(
        checkpoint / LFM2_CPCF_V5_ARTIFACT_METADATA,
        "standalone CPCF-v5 artifact metadata",
    )
    body = dict(value)
    supplied = body.pop("artifact_sha256", None)
    if (
        supplied != expected
        or supplied != canonical_sha256(body)
        or body.get("schema") != LFM2_CPCF_V5_QUARANTINE_ARTIFACT_SCHEMA
        or body.get("dense_safetensors_only") is not True
        or body.get("target_tokenizer_saved_with_child") is not True
        or body.get("local_reload_only") is not True
        or body.get("network_access") is not False
        or body.get("hub_loader_used") is not False
    ):
        raise Lfm2SemanticPostfoldError(
            "Input child artifact differs from its externally rooted identity."
        )
    return {**body, "artifact_sha256": supplied}


def load_lfm2_cpcf_v5_standalone_child(
    checkpoint_directory: str | os.PathLike[str],
    *,
    expected_artifact_sha256: str,
    device: str,
    dtype: str = "bfloat16",
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Strictly load one saved dense child through the existing local contract."""

    checkpoint = _ordinary_directory(checkpoint_directory, "input checkpoint")
    metadata = _input_artifact_metadata(
        checkpoint,
        expected_artifact_sha256=expected_artifact_sha256,
    )
    inventory = metadata.get("parameter_inventory")
    if not isinstance(inventory, Mapping):
        raise Lfm2SemanticPostfoldError(
            "Input artifact has no authenticated parameter inventory."
        )
    loader = Lfm2CpcfV5StrictLocalLoader(
        context_binding_sha256=_digest(
            metadata.get("context_binding_sha256"),
            "input context binding",
        ),
        device=device,
        dtype=dtype,
        expected_ffn_width=int(metadata.get("materialized_ffn_width", 0)),
        expected_model_state_sha256=_digest(
            metadata.get("model_state_sha256"),
            "input model state",
        ),
        expected_parameter_inventory_sha256=_digest(
            metadata.get("parameter_inventory_sha256"),
            "input parameter inventory",
        ),
        expected_tokenizer_semantics_sha256=_digest(
            metadata.get("tokenizer_semantics_sha256"),
            "input tokenizer semantics",
        ),
    )
    model = loader(checkpoint)
    tokenizer = getattr(model, "_neural_forge_quarantine_tokenizer", None)
    if tokenizer is None:
        raise Lfm2SemanticPostfoldError(
            "Strict local child reload did not retain its target tokenizer."
        )
    return model, tokenizer, MappingProxyType(metadata)


def _local_safetensor_shards(directory: Path) -> tuple[Path, ...]:
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file() and not index_path.is_symlink():
        index = _read_json(index_path, "parent safetensors index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise Lfm2SemanticPostfoldError(
                "Parent safetensors index has no weight map."
            )
        shards = tuple(directory / name for name in sorted(set(weight_map.values())))
    else:
        single = directory / "model.safetensors"
        shards = (
            (single,)
            if single.is_file()
            else tuple(sorted(directory.glob("model-*.safetensors")))
        )
    if not shards or any(not path.is_file() or path.is_symlink() for path in shards):
        raise Lfm2SemanticPostfoldError(
            "Parent direct-local safetensors are incomplete or unsafe."
        )
    return shards


def load_lfm2_mapped_parent_tied_table_from_local_safetensors(
    model_directory: str | os.PathLike[str],
    *,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    authenticated_parent_final_norm: Lfm2LocalParentFinalRmsNormLoad,
    tensor_key: str = "model.embed_tokens.weight",
    include_source_lineage_report: bool = False,
) -> (
    tuple[Any, Mapping[str, Any]]
    | tuple[
        Any,
        Mapping[str, Any],
        Mapping[str, Any],
    ]
):
    """Read only the mapped tied rows; never construct the parent module."""

    torch = _torch()
    if type(include_source_lineage_report) is not bool:
        raise TypeError("include_source_lineage_report must be bool.")
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "Direct-local tied-table loading needs safetensors."
        ) from exc
    directory = _ordinary_directory(model_directory, "parent model directory")
    config = _read_json(directory / "config.json", "parent LFM2 config")
    if (
        config.get("model_type") != "lfm2"
        or config.get("tie_word_embeddings") is not True
        or not authenticated_parent_final_norm.final_norm.verify_authentication()
        or authenticated_parent_final_norm.final_norm.cache_key != cache.cache_key
    ):
        raise Lfm2SemanticPostfoldError(
            "Parent tied-table source is not bound to the authenticated cache."
        )
    selected = None
    source_shard = None
    for shard in _local_safetensor_shards(directory):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            if tensor_key in handle.keys():
                if selected is not None:
                    raise Lfm2SemanticPostfoldError(
                        "Parent tied table appears in more than one shard."
                    )
                selected = (
                    handle.get_tensor(tensor_key)
                    .detach()
                    .to(device="cpu")
                    .contiguous()
                    .clone()
                )
                source_shard = shard
    if selected is None or selected.ndim != 2:
        raise Lfm2SemanticPostfoldError(
            f"Parent checkpoint lacks tied table {tensor_key!r}."
        )
    mapping = tuple(bridge.transport_profile.target_to_donor_row_indices)
    if (
        not mapping
        or max(mapping) >= int(selected.shape[0])
        or int(selected.shape[0]) != bridge.transport_profile.donor_row_capacity
    ):
        raise Lfm2SemanticPostfoldError(
            "Tokenizer bridge does not address the authenticated parent table."
        )
    rows = selected.index_select(0, torch.tensor(mapping, dtype=torch.long))
    rows = rows.detach().to(device="cpu", dtype=torch.bfloat16).contiguous().clone()
    body = {
        "schema": "neural-forge-lfm2-direct-local-mapped-tied-table-v1",
        "cache_key": cache.cache_key,
        "cache_parent_fingerprint": cache.parent_fingerprint,
        "parent_final_norm_load_sha256": (
            authenticated_parent_final_norm.result_sha256
        ),
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "tensor_key": tensor_key,
        "source_shard_name": source_shard.name,
        "target_to_donor_rows_sha256": canonical_sha256(list(mapping)),
        "mapped_table_sha256": _interface_tensor_sha256(rows.to(torch.float64)),
        "mapped_table_shape": [int(size) for size in rows.shape],
        "parent_model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
        "network_access_performed": False,
    }
    mapped_receipt = MappingProxyType(_sealed(body, "result_sha256"))
    if not include_source_lineage_report:
        return rows, mapped_receipt

    checkpoint_files = [
        {
            "name": shard.name,
            "bytes": shard.stat().st_size,
            "sha256": _file_sha256(shard),
        }
        for shard in _local_safetensor_shards(directory)
    ]
    source_shard_record = next(
        value for value in checkpoint_files if value["name"] == source_shard.name
    )
    index_path = directory / "model.safetensors.index.json"
    index_record = (
        {
            "name": index_path.name,
            "bytes": index_path.stat().st_size,
            "sha256": _file_sha256(index_path),
        }
        if index_path.is_file() and not index_path.is_symlink()
        else None
    )
    config_path = directory / "config.json"
    parent_load_report = authenticated_parent_final_norm.as_dict()
    parent_checkpoint_files = parent_load_report.get("checkpoint_files")
    parent_identity = parent_load_report.get("reconstructed_parent_identity")
    if (
        not isinstance(parent_checkpoint_files, list)
        or not parent_checkpoint_files
        or not isinstance(parent_identity, Mapping)
        or _file_sha256(config_path) != parent_load_report.get("config_file_sha256")
        or checkpoint_files != parent_checkpoint_files
        or canonical_sha256(parent_identity) != cache.parent_fingerprint
    ):
        raise Lfm2SemanticPostfoldError(
            "Mapped tied-table checkpoint differs from the independently "
            "authenticated parent checkpoint identity."
        )
    source_body = {
        "schema": LFM2_DIRECT_LOCAL_MAPPED_TIED_TABLE_SOURCE_SCHEMA,
        "cache_key": cache.cache_key,
        "cache_manifest_sha256": cache.manifest["manifest_sha256"],
        "cache_parent_fingerprint": cache.parent_fingerprint,
        "cache_layer_topology_fingerprint": cache.layer_topology_fingerprint,
        "parent_final_norm_load_sha256": (
            authenticated_parent_final_norm.result_sha256
        ),
        "parent_final_norm_authentication_sha256": (
            authenticated_parent_final_norm.final_norm.authentication_sha256
        ),
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_transport_profile_fingerprint": bridge.profile_fingerprint,
        "target_to_donor_row_indices_sha256": canonical_sha256(list(mapping)),
        "source_directory": str(directory),
        "config_file": {
            "name": config_path.name,
            "bytes": config_path.stat().st_size,
            "sha256": _file_sha256(config_path),
        },
        "safetensors_index_file": index_record,
        "checkpoint_files": checkpoint_files,
        "independently_authenticated_parent_checkpoint_files_sha256": (
            canonical_sha256(parent_checkpoint_files)
        ),
        "independently_authenticated_parent_identity_sha256": (
            canonical_sha256(parent_identity)
        ),
        "source_checkpoint_exactly_matches_authenticated_parent": True,
        "source_tensor_key": tensor_key,
        "source_tensor_shard": source_shard_record,
        "source_tensor_sha256": _long_horizon_tensor_sha256(
            selected,
            "parent_tied_source_tensor",
        ),
        "source_tensor_dtype": str(selected.dtype).removeprefix("torch."),
        "source_tensor_shape": [int(size) for size in selected.shape],
        "mapped_table_bfloat16_sha256": _long_horizon_tensor_sha256(
            rows,
            "mapped_parent_tied_table",
        ),
        "mapped_table_interface_source_sha256": _interface_tensor_sha256(
            rows.to(torch.float64)
        ),
        "mapped_table_shape": [int(size) for size in rows.shape],
        "mapped_table_dtype": "bfloat16",
        "mapping_operation": (
            "exact_target_to_donor_index_select_then_cpu_bfloat16_contiguous_clone"
        ),
        "deterministic_cast_to_bfloat16": True,
        "mapped_table_legacy_receipt_sha256": mapped_receipt["result_sha256"],
        "parent_model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
        "network_access_performed": False,
    }
    return (
        rows,
        mapped_receipt,
        MappingProxyType(_sealed(source_body, "result_sha256")),
    )


def reconstruct_lfm2_saved_tied_interface(
    target_model: Any,
    source_selected_tied_table: Any,
) -> TiedCausalInterfaceFold:
    """Rebuild the analytical producer map from the saved native tied table.

    CPCF's mean-preserving tied fold stores a constant softmax-invariant first
    column and scaled vocabulary basis columns.  The checkpoint needs only the
    target table for inference; this inverse reconstructs the producer map
    needed by the later terminal compiler without consulting old run memory.
    """

    torch = _torch()
    source = source_selected_tied_table
    target = getattr(target_model.get_input_embeddings(), "weight", None)
    output = getattr(target_model.get_output_embeddings(), "weight", None)
    if (
        not isinstance(source, torch.Tensor)
        or source.ndim != 2
        or source.dtype != torch.bfloat16
        or not isinstance(target, torch.nn.Parameter)
        or output is not target
        or target.ndim != 2
        or int(source.shape[0]) != int(target.shape[0])
        or int(target.shape[1]) < 2
    ):
        raise Lfm2SemanticPostfoldError(
            "Saved child/source tables cannot reconstruct a tied interface."
        )
    source64 = source.detach().to(device="cpu", dtype=torch.float64).contiguous()
    target64 = target.detach().to(device="cpu", dtype=torch.float64).contiguous()
    first = target64[:, 0]
    component_rms = float(first.mean().item())
    constant_error = float((first - component_rms).abs().max().item())
    if (
        not math.isfinite(component_rms)
        or component_rms <= 0.0
        or constant_error != 0.0
    ):
        raise Lfm2SemanticPostfoldError(
            "Saved target tied table lacks CPCF's exact mean coordinate."
        )
    vocabulary = int(target64.shape[0])
    column_scale = component_rms * math.sqrt(vocabulary)
    vocabulary_vectors = target64[:, 1:] / column_scale
    mean = source64.mean(dim=0)
    centered = source64 - mean
    encoder = torch.cat(
        (
            (mean / component_rms).unsqueeze(1),
            centered.T @ vocabulary_vectors / column_scale,
        ),
        dim=1,
    ).contiguous()
    if tuple(encoder.shape) != (int(source.shape[1]), int(target.shape[1])) or not bool(
        torch.isfinite(encoder).all().item()
    ):
        raise Lfm2SemanticPostfoldError(
            "Reconstructed saved-interface producer map is invalid."
        )
    body = {
        "schema": "neural-forge-tied-causal-interface-resume-v1",
        "method": "inverse_saved_mean_coordinate_and_vocabulary_basis",
        "source_width": int(source.shape[1]),
        "target_width": int(target.shape[1]),
        "vocabulary_size": vocabulary,
        "source_table_sha256": _interface_tensor_sha256(source64),
        "target_weight_sha256": _interface_tensor_sha256(target64),
        "producer_encoder_sha256": _interface_tensor_sha256(encoder),
        "mean_coordinate_value": component_rms,
        "mean_coordinate_maximum_absolute_error": constant_error,
        "checkpoint_target_bytes_are_authoritative": True,
        "parent_model_calls": 0,
        "child_model_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    report = _sealed(body, "report_sha256")
    return TiedCausalInterfaceFold(
        target_weight=target64,
        producer_encoder=encoder,
        report=report,
    )


def rehydrate_lfm2_c61_tied_interface(
    target_model: Any,
    source_selected_tied_table: Any,
    *,
    semantic_cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    target_tokenizer: Any,
    compiler_diagnostic_json: str | os.PathLike[str],
    expected_compiler_diagnostic_sha256: str,
) -> tuple[TiedCausalInterfaceFold, Mapping[str, Any]]:
    """Replay c61's exact tied-interface compiler from its sealed evidence.

    Anchor repair changes the saved target table while deliberately keeping the
    producer encoder fixed.  Consequently the encoder cannot be inverted from
    an anchor-active checkpoint.  This routine instead warm-opens c61's exact
    compiler cache/catalog, loads only final-normalization evidence, and reruns
    the deterministic interface compiler.  Every resulting tensor and report
    is matched to both the c61 lineage and the saved dense child.
    """

    torch = _torch()
    diagnostic, diagnostic_file_sha256 = _authenticated_c61_diagnostic(
        compiler_diagnostic_json,
        expected_diagnostic_sha256=expected_compiler_diagnostic_sha256,
    )
    inputs = diagnostic.get("inputs")
    request = diagnostic.get("request")
    preparation_value = diagnostic.get("selective_catalog_artifact")
    compiled_value = diagnostic.get("compiled_model_fold")
    bundle_value = diagnostic.get("boundary_bundle")
    install_value = diagnostic.get("interface_install")
    seam = diagnostic.get("compiled_candidate_consumer_seam")
    if any(
        not isinstance(value, Mapping)
        for value in (
            inputs,
            request,
            preparation_value,
            compiled_value,
            bundle_value,
            install_value,
            seam,
        )
    ):
        raise Lfm2SemanticPostfoldError(
            "c61 diagnostic lacks exact interface-rehydration ancestry."
        )
    preparation = _verify_sealed(
        preparation_value,
        seal_name="preparation_receipt_sha256",
        label="c61 selective-catalog preparation",
    )
    compiled = _verify_sealed(
        compiled_value,
        seal_name="result_sha256",
        label="c61 compiled-model receipt",
    )
    bundle = _verify_sealed(
        bundle_value,
        seal_name="bundle_sha256",
        label="c61 boundary bundle",
    )
    install = _verify_sealed(
        install_value,
        seal_name="report_sha256",
        label="c61 tied-interface installation",
    )
    compiler_cache_key = _digest(
        inputs.get("cache_key"),
        "c61 compiler cache key",
    )
    common_parent = _digest(
        inputs.get("parent_fingerprint"),
        "c61 parent fingerprint",
    )
    if (
        semantic_cache.parent_fingerprint != common_parent
        or inputs.get("bridge_artifact_sha256") != bridge.artifact_sha256
        or seam.get("cache_sha256") != compiler_cache_key
        or seam.get("parent_sha256") != common_parent
        or seam.get("bridge_sha256") != bridge.artifact_sha256
        or preparation.get("cache_key") != compiler_cache_key
        or preparation.get("trace_cache_directory")
        != inputs.get("trace_cache_directory")
        or preparation.get("catalog_store_directory")
        != request.get("selective_catalog_store_directory")
        or inputs.get("selective_catalog_preparation_receipt_sha256")
        != preparation["preparation_receipt_sha256"]
        or inputs.get("selective_catalog_binding_sha256")
        != preparation.get("binding_sha256")
        or inputs.get("selective_catalog_artifact_manifest_sha256")
        != preparation.get("artifact_manifest_sha256")
        or inputs.get("selective_catalog_artifact_sha256")
        != preparation.get("artifact_catalog_sha256")
        or compiled.get("selective_catalog_sha256")
        != preparation.get("selective_catalog_sha256")
        or compiled.get("selective_source_sha256")
        != preparation.get("selective_source_sha256")
        or compiled.get("selective_provider_sha256")
        != preparation.get("selective_provider_sha256")
    ):
        raise Lfm2SemanticPostfoldError(
            "Compiler-cache, catalog, bridge, or parent ancestry diverged."
        )

    preparation_path = _ordinary_file(
        preparation.get("catalog_receipt_json"),
        "c61 selective-catalog preparation receipt",
    )
    preparation_disk, preparation_file_sha256 = _read_canonical_json(
        preparation_path,
        "c61 selective-catalog preparation receipt",
    )
    if preparation_disk != preparation:
        raise Lfm2SemanticPostfoldError(
            "On-disk selective-catalog receipt differs from c61's sealed copy."
        )
    compiler_cache_directory = _ordinary_directory(
        inputs.get("trace_cache_directory"),
        "c61 compiler trace cache",
    )
    catalog_store_directory = _ordinary_directory(
        preparation.get("catalog_store_directory"),
        "c61 selective-catalog store",
    )
    artifact_path = _ordinary_directory(
        preparation.get("artifact_path"),
        "c61 selective-catalog artifact",
    )
    try:
        artifact_path.relative_to(catalog_store_directory)
    except ValueError as exc:
        raise Lfm2SemanticPostfoldError(
            "c61 selective-catalog artifact lies outside its sealed store."
        ) from exc
    trajectory_path = _ordinary_file(
        inputs.get("trajectory_corpus"),
        "c61 trajectory corpus",
    )
    trajectory_sha256 = _file_sha256(trajectory_path)
    if trajectory_sha256 != _digest(
        inputs.get("trajectory_corpus_sha256"),
        "c61 trajectory corpus SHA-256",
    ):
        raise Lfm2SemanticPostfoldError(
            "c61 trajectory corpus changed before interface rehydration."
        )

    compiler_cache = open_lfm2_circuit_trace_cache(
        compiler_cache_directory,
        verify=True,
    )
    if (
        compiler_cache.cache_key != compiler_cache_key
        or compiler_cache.parent_fingerprint != common_parent
    ):
        raise Lfm2SemanticPostfoldError(
            "The reopened c61 compiler cache differs from its sealed identity."
        )
    rows = cpcf_resources._read_trajectory_rows(trajectory_path)
    trace_adapter = cpcf_resources._build_trace_adapter(
        rows=rows,
        target_tokenizer=target_tokenizer,
        bridge=bridge,
        cache=compiler_cache,
    )
    if trace_adapter.evidence_sha256 != _digest(
        inputs.get("trace_adapter_sha256"),
        "c61 trace-adapter SHA-256",
    ):
        raise Lfm2SemanticPostfoldError(
            "Rebuilt c61 target-token trace adapter changed."
        )

    provider = None
    final_payload = None
    final_norm_authentication_sha256 = None
    try:
        provider = open_lfm2_cpcf_selective_evidence_provider_from_catalog(
            compiler_cache,
            artifact_store=catalog_store_directory,
            expected_binding_sha256=_digest(
                preparation.get("binding_sha256"),
                "c61 selective-catalog binding SHA-256",
            ),
            expected_manifest_sha256=_digest(
                preparation.get("artifact_manifest_sha256"),
                "c61 selective-catalog manifest SHA-256",
            ),
            expected_catalog_sha256=_digest(
                preparation.get("artifact_catalog_sha256"),
                "c61 selective-catalog artifact SHA-256",
            ),
        )
        catalog = provider.catalog
        if (
            catalog.catalog_sha256 != preparation.get("selective_catalog_sha256")
            or catalog.source_sha256 != preparation.get("selective_source_sha256")
            or catalog.provider_sha256 != preparation.get("selective_provider_sha256")
        ):
            raise Lfm2SemanticPostfoldError(
                "Warm-opened selective catalog differs from c61."
            )
        final_payload = provider.load_final_norm_payload()
        final_norm_authentication_sha256 = provider.authenticate_final_norm_payload(
            final_payload
        )
        if final_norm_authentication_sha256 != _digest(
            compiled.get("final_norm_authentication_sha256"),
            "c61 final-normalization authentication SHA-256",
        ):
            raise Lfm2SemanticPostfoldError(
                "Final-normalization evidence changed since c61 compilation."
            )
        shells = cpcf_v5_compiler.build_lfm2_cpcf_v5_target_phase_shells(
            trace_adapter,
            catalog,
            None,
            final_norm_payload=final_payload,
        )
        final_phases = cpcf_v5_compiler.materialize_lfm2_cpcf_v5_final_norm_phases(
            shells,
            catalog,
            final_payload,
        )
        boundary_compiler = cpcf_v5_compiler._SelectiveBoundaryCompiler(
            shells,
            catalog,
            source_selected_tied_table,
            target_hidden_width=int(target_model.config.hidden_size),
            model_policy=Lfm2CpcfModelFoldPolicy(),
        )
        interface = boundary_compiler.fit_interface(
            final_phases,
            target_norm_epsilon=cpcf_model_fold._final_norm_epsilon(target_model),
        )
    finally:
        if provider is not None and final_payload is not None:
            provider.release_final_norm_payload(final_payload)
        if provider is not None:
            provider.close()

    lineage = bundle.get("interface_lineage")
    if not isinstance(lineage, Mapping):
        raise Lfm2SemanticPostfoldError(
            "c61 boundary bundle has no tied-interface lineage."
        )
    base_report = _verify_sealed(
        lineage.get("base_interface_report"),
        seal_name="report_sha256",
        label="c61 base tied-interface report",
    )
    anchor_report = _verify_sealed(
        lineage.get("causal_anchor_report"),
        seal_name="report_sha256",
        label="c61 causal-anchor interface report",
    )
    active_report = _verify_sealed(
        interface.report,
        seal_name="report_sha256",
        label="rehydrated c61 active interface",
    )
    active_anchor_value = anchor_report.get("active_anchor_residual_selected")
    base_control_value = anchor_report.get("selected_candidate_is_base_control")
    if (
        not isinstance(active_anchor_value, bool)
        or not isinstance(base_control_value, bool)
        or active_anchor_value is base_control_value
    ):
        raise Lfm2SemanticPostfoldError(
            "c61 causal-anchor selection flags are not exact booleans."
        )
    active_anchor = active_anchor_value
    source64 = (
        source_selected_tied_table.detach()
        .to(
            device="cpu",
            dtype=torch.float64,
        )
        .contiguous()
    )
    target = getattr(target_model.get_input_embeddings(), "weight", None)
    output = getattr(target_model.get_output_embeddings(), "weight", None)
    if not isinstance(target, torch.nn.Parameter) or output is not target:
        raise Lfm2SemanticPostfoldError(
            "Saved c61 child no longer exposes one native tied table."
        )
    saved_target = target.detach().to(device="cpu").contiguous()
    expected_saved_target = interface.target_weight.to(
        dtype=saved_target.dtype,
    ).contiguous()
    saved_target_runtime_sha256 = cpcf_model_compiler._tensor_sha256(saved_target)
    if (
        dict(boundary_compiler.base_interface.report) != base_report
        or dict(boundary_compiler.anchor_fold.report) != anchor_report
        or active_report != anchor_report
        or lineage.get("base_interface_report_sha256") != base_report["report_sha256"]
        or lineage.get("causal_anchor_report_sha256") != anchor_report["report_sha256"]
        or lineage.get("active_interface_report_sha256")
        != active_report["report_sha256"]
        or lineage.get("active_interface_is_anchor_repaired") is not active_anchor
        or lineage.get("active_interface_is_base_control") != (not active_anchor)
        or lineage.get("producer_encoder_fixed_by_anchor_selection") is not True
        or bundle.get("interface_report_sha256") != active_report["report_sha256"]
        or bundle.get("base_interface_report_sha256") != base_report["report_sha256"]
        or bundle.get("causal_anchor_report_sha256") != anchor_report["report_sha256"]
        or compiled.get("boundary_bundle_sha256") != bundle["bundle_sha256"]
        or install.get("interface_report_sha256") != active_report["report_sha256"]
        or install.get("target_weight_sha256") != saved_target_runtime_sha256
        or not torch.equal(saved_target, expected_saved_target)
        or active_report.get("target_weight_sha256")
        != _interface_tensor_sha256(interface.target_weight)
        or active_report.get("producer_encoder_sha256")
        != _interface_tensor_sha256(interface.producer_encoder)
        or base_report.get("source_table_sha256") != _interface_tensor_sha256(source64)
        or anchor_report.get("source_tied_table_sha256")
        != _interface_tensor_sha256(source64)
    ):
        raise Lfm2SemanticPostfoldError(
            "Rehydrated producer coordinates differ from c61 or the saved child."
        )
    body = {
        "schema": "neural-forge-lfm2-c61-interface-rehydration-v1",
        "compiler_diagnostic_receipt_sha256": diagnostic["receipt_sha256"],
        "compiler_diagnostic_file_sha256": diagnostic_file_sha256,
        "compiler_cache_key": compiler_cache_key,
        "semantic_evidence_cache_key": semantic_cache.cache_key,
        "common_parent_fingerprint": common_parent,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "trajectory_corpus_sha256": trajectory_sha256,
        "trace_adapter_sha256": trace_adapter.evidence_sha256,
        "selective_catalog_preparation_receipt_sha256": (
            preparation["preparation_receipt_sha256"]
        ),
        "selective_catalog_preparation_file_sha256": preparation_file_sha256,
        "selective_catalog_binding_sha256": preparation["binding_sha256"],
        "selective_catalog_manifest_sha256": preparation["artifact_manifest_sha256"],
        "selective_catalog_artifact_sha256": preparation["artifact_catalog_sha256"],
        "final_norm_authentication_sha256": final_norm_authentication_sha256,
        "base_interface_report_sha256": base_report["report_sha256"],
        "active_interface_report_sha256": active_report["report_sha256"],
        "active_interface_is_anchor_repaired": active_anchor,
        "active_interface_is_base_control": not active_anchor,
        "source_table_sha256": _interface_tensor_sha256(source64),
        "active_target_weight_sha256": active_report["target_weight_sha256"],
        "saved_target_weight_runtime_sha256": saved_target_runtime_sha256,
        "producer_encoder_sha256": active_report["producer_encoder_sha256"],
        "producer_encoder_recovered_from_repaired_target_table": False,
        "producer_encoder_recomputed_from_authenticated_final_norm_evidence": True,
        "compiler_and_semantic_cache_identities_bound_separately": True,
        "parent_model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
        "child_model_forward_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    return interface, MappingProxyType(_sealed(body, "result_sha256"))


@dataclass(frozen=True, slots=True)
class Lfm2SemanticCompilerLayerLineage:
    """Normalized, authenticated c61 attention ancestry for one layer."""

    layer_index: int
    source_inventory: Lfm2DenseQkvoSourceInventory
    query_groups: tuple[tuple[int, ...], ...]
    query_group_weights: tuple[tuple[float, ...], ...]
    key_value_groups: tuple[tuple[int, ...], ...]
    key_value_group_weights: tuple[tuple[float, ...], ...]
    compiler_lineage_sha256: str
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
            or not isinstance(self.source_inventory, Lfm2DenseQkvoSourceInventory)
            or self.source_inventory.verify() != self.source_inventory.inventory_sha256
        ):
            raise Lfm2SemanticPostfoldError(
                "Compiler layer lineage has invalid coordinates or inventory."
            )
        groups = tuple(
            tuple(int(item) for item in group) for group in self.query_groups
        )
        weights = tuple(
            tuple(float(item) for item in group) for group in self.query_group_weights
        )
        key_value_groups = tuple(
            tuple(int(item) for item in group) for group in self.key_value_groups
        )
        key_value_weights = tuple(
            tuple(float(item) for item in group)
            for group in self.key_value_group_weights
        )
        if (
            not groups
            or len(groups) != len(weights)
            or any(not group for group in groups)
            or any(
                len(group) != len(weight)
                for group, weight in zip(groups, weights, strict=True)
            )
            or any(
                not math.isclose(math.fsum(weight), 1.0, rel_tol=1e-8, abs_tol=1e-10)
                or any(not math.isfinite(value) or value <= 0.0 for value in weight)
                for weight in weights
            )
            or not key_value_groups
            or len(key_value_groups) != len(key_value_weights)
            or any(not group for group in key_value_groups)
            or any(
                len(group) != weight_count
                for group, weight_count in zip(
                    key_value_groups,
                    map(len, key_value_weights),
                    strict=True,
                )
            )
            or any(
                not math.isclose(
                    math.fsum(weight),
                    1.0,
                    rel_tol=1e-8,
                    abs_tol=1e-10,
                )
                or any(not math.isfinite(value) or value <= 0.0 for value in weight)
                for weight in key_value_weights
            )
        ):
            raise Lfm2SemanticPostfoldError(
                "Compiler layer lineage has invalid Q/KV head groups."
            )
        lineage = _digest(
            self.compiler_lineage_sha256,
            "compiler layer lineage SHA-256",
        )
        report = _verify_sealed(
            self.report,
            seal_name="lineage_sha256",
            label="compiler layer lineage",
        )
        if (
            report["lineage_sha256"] != lineage
            or report.get("layer_index") != self.layer_index
            or report.get("source_inventory_sha256")
            != self.source_inventory.inventory_sha256
            or report.get("query_groups") != [list(value) for value in groups]
            or report.get("query_group_weights") != [list(value) for value in weights]
            or report.get("key_value_groups")
            != [list(value) for value in key_value_groups]
            or report.get("key_value_group_weights")
            != [list(value) for value in key_value_weights]
        ):
            raise Lfm2SemanticPostfoldError(
                "Normalized compiler layer lineage differs from its seal."
            )
        object.__setattr__(self, "query_groups", groups)
        object.__setattr__(self, "query_group_weights", weights)
        object.__setattr__(self, "key_value_groups", key_value_groups)
        object.__setattr__(self, "key_value_group_weights", key_value_weights)
        object.__setattr__(self, "compiler_lineage_sha256", lineage)
        object.__setattr__(self, "report", MappingProxyType(report))


def _typed_qkvo_inventory(value: Any) -> Lfm2DenseQkvoSourceInventory:
    if not isinstance(value, Mapping):
        raise Lfm2SemanticPostfoldError(
            "Compiler diagnostic has no typed parent Q/K/V/O inventory."
        )
    try:
        inventory = Lfm2DenseQkvoSourceInventory(
            tensor_receipts=value.get("tensor_receipts", {}),
            source_state_sha256=value.get("source_state_sha256"),
            structural_source_state_sha256=value.get("structural_source_state_sha256"),
            logical_parameters=value.get("logical_parameters"),
            inventory_sha256=value.get("inventory_sha256", ""),
        )
    except (TypeError, ValueError) as exc:
        raise Lfm2SemanticPostfoldError(
            "Compiler parent Q/K/V/O inventory is invalid."
        ) from exc
    if inventory.as_dict() != dict(value):
        raise Lfm2SemanticPostfoldError(
            "Compiler parent Q/K/V/O inventory is not canonical."
        )
    return inventory


def _load_lfm2_semantic_foundation_lineage(
    foundation_receipt_json: str | os.PathLike[str],
    *,
    expected_foundation_receipt_sha256: str,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    input_metadata: Mapping[str, Any],
    parent_model_directory: str | os.PathLike[str],
) -> Mapping[int, Lfm2SemanticCompilerLayerLineage]:
    """Normalize the fast foundation's direct-local attention ancestry."""

    try:
        verified, receipt_file_sha256 = authenticate_lfm2_semantic_foundation_receipt(
            foundation_receipt_json,
            expected_receipt_sha256=expected_foundation_receipt_sha256,
            semantic_cache=cache,
            bridge=bridge,
            input_metadata=input_metadata,
            parent_model_directory=parent_model_directory,
        )
    except Lfm2SemanticFoundationError as exc:
        raise Lfm2SemanticPostfoldError(
            "Semantic-foundation ancestry authentication failed."
        ) from exc
    raw_lineages = verified.get("attention_lineage")
    topology = verified.get("target_topology")
    context = verified.get("foundation_context")
    parent_inventory = verified.get("parent_source_inventory")
    if (
        not isinstance(raw_lineages, Mapping)
        or not isinstance(topology, Mapping)
        or not isinstance(context, Mapping)
        or not isinstance(parent_inventory, Mapping)
        or context.get("attention_lineage_sha256") != canonical_sha256(raw_lineages)
    ):
        raise Lfm2SemanticPostfoldError(
            "Semantic-foundation receipt lacks bound attention lineage."
        )
    expected_layers = topology.get("attention_layer_indices")
    if (
        not isinstance(expected_layers, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in expected_layers
        )
        or set(raw_lineages) != {str(value) for value in expected_layers}
    ):
        raise Lfm2SemanticPostfoldError(
            "Semantic-foundation attention lineage differs from target topology."
        )
    normalized: dict[int, Lfm2SemanticCompilerLayerLineage] = {}
    for layer_text, raw in sorted(raw_lineages.items(), key=lambda item: int(item[0])):
        layer_index = int(layer_text)
        lineage = _verify_sealed(
            raw,
            seal_name="lineage_sha256",
            label=f"semantic-foundation layer {layer_index} lineage",
        )
        group_report = _verify_sealed(
            lineage.get("head_group_report"),
            seal_name="result_sha256",
            label=f"semantic-foundation layer {layer_index} head groups",
        )
        inventory = _typed_qkvo_inventory(lineage.get("source_inventory"))
        groups = tuple(
            tuple(int(value) for value in group)
            for group in lineage.get("query_groups", ())
        )
        weights = tuple(
            tuple(float(value) for value in group)
            for group in lineage.get("query_group_weights", ())
        )
        key_value_groups = tuple(
            tuple(int(value) for value in group)
            for group in group_report.get("key_value_groups", ())
        )
        key_value_weights = tuple(
            tuple(1.0 / len(group) for _ in group)
            for group in key_value_groups
            if group
        )
        target_head_dim = topology.get("head_dim")
        target_query_heads = topology.get("query_heads")
        q_shape = inventory.tensor_receipts["q_proj.weight"]["shape"]
        source_query_heads = (
            0
            if isinstance(target_head_dim, bool)
            or not isinstance(target_head_dim, int)
            or target_head_dim <= 0
            or int(q_shape[0]) % target_head_dim
            else int(q_shape[0]) // target_head_dim
        )
        if (
            lineage.get("schema") != LFM2_SEMANTIC_FOUNDATION_LINEAGE_SCHEMA
            or lineage.get("layer_index") != layer_index
            or lineage.get("parent_inventory_sha256")
            != parent_inventory.get("result_sha256")
            or lineage.get("source_inventory_sha256") != inventory.inventory_sha256
            or lineage.get("head_group_report_sha256") != group_report["result_sha256"]
            or group_report.get("query_groups") != [list(value) for value in groups]
            or group_report.get("query_group_weights")
            != [list(value) for value in weights]
            or group_report.get("key_value_groups")
            != [list(value) for value in key_value_groups]
            or group_report.get("full_source_query_coverage") is not True
            or group_report.get("all_weights_positive") is not True
            or group_report.get("source_query_heads") != source_query_heads
            or group_report.get("target_query_heads") != target_query_heads
            or source_query_heads <= 0
            or len(groups) != target_query_heads
            or sorted(value for group in groups for value in group)
            != list(range(source_query_heads))
            or sorted(value for group in key_value_groups for value in group)
            != list(range(int(group_report.get("source_key_value_heads", -1))))
            or len(key_value_groups) != topology.get("key_value_heads")
            or lineage.get("target_hidden_size") != topology.get("hidden_size")
            or lineage.get("target_query_heads") != target_query_heads
            or lineage.get("target_key_value_heads") != topology.get("key_value_heads")
            or lineage.get("target_head_dim") != topology.get("head_dim")
            or lineage.get("legacy_layer_compiler_calls") != 0
        ):
            raise Lfm2SemanticPostfoldError(
                "Semantic-foundation layer lineage is internally inconsistent."
            )
        report = {
            **lineage,
            "key_value_groups": [list(value) for value in key_value_groups],
            "key_value_group_weights": [list(value) for value in key_value_weights],
            "foundation_lineage_sha256": lineage["lineage_sha256"],
            "foundation_receipt_sha256": verified["receipt_sha256"],
            "foundation_receipt_file_sha256": receipt_file_sha256,
            "compiler_cache_key": verified["compiler_cache_key"],
            "semantic_evidence_cache_key": cache.cache_key,
            "common_parent_fingerprint": cache.parent_fingerprint,
            "compiler_and_semantic_cache_identities_bound_separately": True,
        }
        report.pop("lineage_sha256")
        report = _sealed(report, "lineage_sha256")
        normalized[layer_index] = Lfm2SemanticCompilerLayerLineage(
            layer_index=layer_index,
            source_inventory=inventory,
            query_groups=groups,
            query_group_weights=weights,
            key_value_groups=key_value_groups,
            key_value_group_weights=key_value_weights,
            compiler_lineage_sha256=report["lineage_sha256"],
            report=report,
        )
    if set(normalized) != set(expected_layers):
        raise Lfm2SemanticPostfoldError(
            "Semantic-foundation attention lineage is incomplete."
        )
    return MappingProxyType(normalized)


def load_lfm2_semantic_compiler_lineage(
    diagnostic_json: str | os.PathLike[str],
    *,
    expected_diagnostic_sha256: str,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    input_metadata: Mapping[str, Any],
    parent_model_directory: str | os.PathLike[str] | None = None,
) -> Mapping[int, Lfm2SemanticCompilerLayerLineage]:
    """Load legacy c61 or fast-foundation attention ancestry."""

    if _claims_semantic_foundation_schema(diagnostic_json):
        if parent_model_directory is None:
            raise Lfm2SemanticPostfoldError(
                "Semantic-foundation ancestry requires the direct-local parent."
            )
        return _load_lfm2_semantic_foundation_lineage(
            diagnostic_json,
            expected_foundation_receipt_sha256=expected_diagnostic_sha256,
            cache=cache,
            bridge=bridge,
            input_metadata=input_metadata,
            parent_model_directory=parent_model_directory,
        )

    verified, diagnostic_file_sha256 = _authenticated_c61_diagnostic(
        diagnostic_json,
        expected_diagnostic_sha256=expected_diagnostic_sha256,
    )
    expected = verified["receipt_sha256"]
    inputs = verified.get("inputs")
    seam = verified.get("compiled_candidate_consumer_seam")
    compiled = verified.get("compiled_model_fold")
    compiler_cache_key = (
        None if not isinstance(inputs, Mapping) else inputs.get("cache_key")
    )
    if (
        not isinstance(inputs, Mapping)
        or _digest(compiler_cache_key, "c61 compiler cache key") != compiler_cache_key
        or inputs.get("parent_fingerprint") != cache.parent_fingerprint
        or inputs.get("bridge_artifact_sha256") != bridge.artifact_sha256
        or not isinstance(seam, Mapping)
        or seam.get("cache_sha256") != compiler_cache_key
        or seam.get("parent_sha256") != cache.parent_fingerprint
        or seam.get("bridge_sha256") != bridge.artifact_sha256
        or seam.get("target_state_sha256") != input_metadata.get("model_state_sha256")
        or not isinstance(compiled, Mapping)
    ):
        raise Lfm2SemanticPostfoldError(
            "c61 diagnostic is not the compiler ancestry of this saved child."
        )
    compiled_report = _verify_sealed(
        compiled,
        seal_name="result_sha256",
        label="c61 compiled-model receipt",
    )
    layers = verified.get("layer_results")
    if not isinstance(layers, list) or not layers:
        raise Lfm2SemanticPostfoldError(
            "c61 diagnostic contains no ordered layer results."
        )
    normalized: dict[int, Lfm2SemanticCompilerLayerLineage] = {}
    for raw in layers:
        if not isinstance(raw, Mapping):
            raise Lfm2SemanticPostfoldError("c61 layer result is not a mapping.")
        layer = _verify_sealed(
            raw,
            seal_name="result_sha256",
            label="c61 layer result",
        )
        if layer.get("operator_kind") != "full_attention":
            continue
        layer_index = layer.get("layer_index")
        solver = layer.get("solver_evidence_policy")
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or layer_index < 0
            or layer_index in normalized
            or not isinstance(solver, Mapping)
        ):
            raise Lfm2SemanticPostfoldError(
                "c61 attention layer lineage is missing or duplicated."
            )
        evaluations = solver.get("attention_candidate_evaluations")
        direct = (
            None
            if not isinstance(evaluations, Mapping)
            else evaluations.get("direct_monolithic")
        )
        direct_receipt = _verify_sealed(
            direct,
            seal_name="report_sha256",
            label=f"c61 layer {layer_index} direct attention candidate",
        )
        artifact = direct_receipt.get("operator_artifact")
        context = direct_receipt.get("selection_context_metrics")
        if not isinstance(artifact, Mapping) or not isinstance(context, Mapping):
            raise Lfm2SemanticPostfoldError(
                "c61 direct attention candidate lacks groups or context audit."
            )
        groups = tuple(
            tuple(int(item) for item in group)
            for group in artifact.get("query_groups", ())
        )
        weights = tuple(
            tuple(float(item) for item in group)
            for group in artifact.get("query_barycentric_weights", ())
        )
        key_value_groups = tuple(
            tuple(int(item) for item in group)
            for group in artifact.get("key_value_groups", ())
        )
        key_value_weights = tuple(
            tuple(float(item) for item in group)
            for group in artifact.get("key_value_barycentric_weights", ())
        )
        group_hash = canonical_sha256(
            {
                "groups": [list(value) for value in groups],
                "weights": [list(value) for value in weights],
            }
        )
        endpoint = solver.get("endpoint_residual_attention_artifact")
        endpoint_receipt = _verify_sealed(
            endpoint,
            seal_name="artifact_sha256",
            label=f"c61 layer {layer_index} endpoint-residual artifact",
        )
        inventory = _typed_qkvo_inventory(endpoint_receipt.get("source_inventory"))
        if (
            direct_receipt.get("candidate_id") != "direct_monolithic"
            or direct_receipt.get("operator_artifact_sha256")
            != canonical_sha256(dict(artifact))
            or context.get("query_groups_sha256") != group_hash
            or endpoint_receipt.get("source_inventory_sha256")
            != inventory.inventory_sha256
            or solver.get("endpoint_residual_attention_artifact_sha256")
            != endpoint_receipt["artifact_sha256"]
        ):
            raise Lfm2SemanticPostfoldError(
                "c61 attention groups or parent inventory changed after compilation."
            )
        body = {
            "schema": LFM2_SEMANTIC_COMPILER_LINEAGE_SCHEMA,
            "diagnostic_receipt_sha256": expected,
            "diagnostic_file_sha256": diagnostic_file_sha256,
            "compiler_cache_key": compiler_cache_key,
            "semantic_evidence_cache_key": cache.cache_key,
            "common_parent_fingerprint": cache.parent_fingerprint,
            "compiler_and_semantic_cache_identities_bound_separately": True,
            "compiled_model_fold_sha256": compiled_report["result_sha256"],
            "layer_index": layer_index,
            "layer_result_sha256": layer["result_sha256"],
            "direct_candidate_receipt_sha256": direct_receipt["report_sha256"],
            "direct_operator_artifact_sha256": direct_receipt[
                "operator_artifact_sha256"
            ],
            "endpoint_artifact_sha256": endpoint_receipt["artifact_sha256"],
            "source_inventory_sha256": inventory.inventory_sha256,
            "query_groups": [list(value) for value in groups],
            "query_group_weights": [list(value) for value in weights],
            "key_value_groups": [list(value) for value in key_value_groups],
            "key_value_group_weights": [list(value) for value in key_value_weights],
            "query_groups_sha256": group_hash,
            "numeric_parent_child_weight_match_objective": False,
        }
        report = _sealed(body, "lineage_sha256")
        normalized[layer_index] = Lfm2SemanticCompilerLayerLineage(
            layer_index=layer_index,
            source_inventory=inventory,
            query_groups=groups,
            query_group_weights=weights,
            key_value_groups=key_value_groups,
            key_value_group_weights=key_value_weights,
            compiler_lineage_sha256=report["lineage_sha256"],
            report=report,
        )
    if not normalized:
        raise Lfm2SemanticPostfoldError(
            "c61 diagnostic contains no authenticated attention lineage."
        )
    return MappingProxyType(normalized)


def _load_lfm2_parent_qkvo_from_local_safetensors(
    model_directory: str | os.PathLike[str],
    *,
    layer_index: int,
    expected_inventory: Lfm2DenseQkvoSourceInventory,
) -> tuple[Lfm2DenseQkvoSourceInventory, Mapping[str, Any]]:
    """Load and authenticate one direct-local parent Q/K/V/O layer."""

    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError("Direct-local Q/K/V/O loading needs safetensors.") from exc
    directory = _ordinary_directory(model_directory, "parent model directory")
    prefix = f"model.layers.{layer_index}.self_attn."
    keys = {
        "q": prefix + "q_proj.weight",
        "k": prefix + "k_proj.weight",
        "v": prefix + "v_proj.weight",
        "o": prefix + "out_proj.weight",
        "q_norm": prefix + "q_layernorm.weight",
        "k_norm": prefix + "k_layernorm.weight",
        "operator_norm": f"model.layers.{layer_index}.operator_norm.weight",
    }
    tensors: dict[str, Any] = {}
    for shard in _local_safetensor_shards(directory):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name, key in keys.items():
                if key in handle.keys():
                    if name in tensors:
                        raise Lfm2SemanticPostfoldError(
                            f"Parent tensor {key!r} appears in multiple shards."
                        )
                    tensors[name] = handle.get_tensor(key)
    if set(tensors) != set(keys):
        raise Lfm2SemanticPostfoldError(
            f"Direct-local parent layer {layer_index} lacks dense Q/K/V/O."
        )
    observed = seal_lfm2_dense_qkvo_source_inventory(
        source_query_weight=tensors["q"],
        source_key_weight=tensors["k"],
        source_value_weight=tensors["v"],
        source_output_weight=tensors["o"],
    )
    if (
        not isinstance(expected_inventory, Lfm2DenseQkvoSourceInventory)
        or observed.as_dict() != expected_inventory.as_dict()
    ):
        raise Lfm2SemanticPostfoldError(
            f"Direct-local parent Q/K/V/O bytes changed for layer {layer_index}."
        )
    return observed, MappingProxyType(
        {
            name: value.detach().to(device="cpu").contiguous()
            for name, value in tensors.items()
        }
    )


def _local_lfm2_norm_epsilon(model_directory: str | os.PathLike[str]) -> float:
    directory = _ordinary_directory(model_directory, "parent model directory")
    config_path = directory / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Lfm2SemanticPostfoldError(
            "Direct-local parent config.json is unavailable or invalid."
        ) from exc
    epsilon = config.get("norm_eps") if isinstance(config, Mapping) else None
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(float(epsilon))
        or float(epsilon) <= 0.0
    ):
        raise Lfm2SemanticPostfoldError(
            "Direct-local parent config has no finite positive norm_eps."
        )
    return float(epsilon)


def load_lfm2_parent_qkvo_inventory_from_local_safetensors(
    model_directory: str | os.PathLike[str],
    *,
    layer_index: int,
    expected_inventory: Lfm2DenseQkvoSourceInventory,
) -> Lfm2DenseQkvoSourceInventory:
    """Hash direct-local parent Q/K/V/O bytes and match c61's sealed inventory."""

    observed, _tensors = _load_lfm2_parent_qkvo_from_local_safetensors(
        model_directory,
        layer_index=layer_index,
        expected_inventory=expected_inventory,
    )
    return observed


def _fold_parent_qk_weight_priors(
    *,
    parent_query_weight: Any,
    parent_key_weight: Any,
    parent_value_weight: Any,
    parent_output_weight: Any,
    parent_query_norm_weight: Any,
    parent_key_norm_weight: Any,
    residual_chart: Any,
    query_groups: Sequence[Sequence[int]],
    query_group_weights: Sequence[Sequence[float]],
    key_value_groups: Sequence[Sequence[int]],
    key_value_group_weights: Sequence[Sequence[float]],
    head_dim: int,
    source_inventory_sha256: str,
    residual_chart_sha256: str,
    normalized_input_decoder: Any | None = None,
    normalized_input_decoder_receipt: Mapping[str, Any] | None = None,
) -> tuple[Any, Any, Any, Any, Any, Any, Mapping[str, Any]]:
    """Transport direct parent Q/K operators through sealed head/chart coordinates."""

    torch = _torch()
    chart = residual_chart.detach().to(device="cpu", dtype=torch.float64)
    parent_query = parent_query_weight.detach().to(device="cpu", dtype=torch.float64)
    parent_key = parent_key_weight.detach().to(device="cpu", dtype=torch.float64)
    parent_value = parent_value_weight.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    parent_output = parent_output_weight.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    parent_query_norm = parent_query_norm_weight.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    parent_key_norm = parent_key_norm_weight.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    query_groups_value = tuple(
        tuple(int(value) for value in group) for group in query_groups
    )
    query_weights_value = tuple(
        tuple(float(value) for value in group) for group in query_group_weights
    )
    key_groups_value = tuple(
        tuple(int(value) for value in group) for group in key_value_groups
    )
    key_weights_value = tuple(
        tuple(float(value) for value in group) for group in key_value_group_weights
    )
    if (
        chart.ndim != 2
        or parent_query.ndim != 2
        or parent_key.ndim != 2
        or parent_value.ndim != 2
        or parent_output.ndim != 2
        or parent_query_norm.ndim != 1
        or parent_key_norm.ndim != 1
        or tuple(parent_query_norm.shape) != (head_dim,)
        or tuple(parent_key_norm.shape) != (head_dim,)
        or head_dim <= 0
        or int(parent_query.shape[0]) % head_dim
        or int(parent_key.shape[0]) % head_dim
    ):
        raise Lfm2SemanticPostfoldError(
            "Direct parent Q/K weights cannot enter the semantic residual chart."
        )
    source_hidden, target_hidden = map(int, chart.shape)
    if (
        int(parent_query.shape[1]) != source_hidden
        or int(parent_key.shape[1]) != source_hidden
        or tuple(parent_value.shape) != tuple(parent_key.shape)
        or tuple(parent_output.shape) != (source_hidden, int(parent_query.shape[0]))
    ):
        raise Lfm2SemanticPostfoldError(
            "Direct parent Q/K/V/O weights disagree on source topology."
        )
    source_query_heads = int(parent_query.shape[0]) // head_dim
    source_key_heads = int(parent_key.shape[0]) // head_dim

    def validated_groups(
        groups: tuple[tuple[int, ...], ...],
        weights: tuple[tuple[float, ...], ...],
        source_heads: int,
        role: str,
    ) -> tuple[tuple[tuple[int, ...], tuple[float, ...]], ...]:
        if (
            not groups
            or len(groups) != len(weights)
            or sorted(value for group in groups for value in group)
            != list(range(source_heads))
            or any(
                len(group) != len(coefficients)
                or not group
                or any(
                    value <= 0.0 or not math.isfinite(value) for value in coefficients
                )
                or not math.isclose(
                    math.fsum(coefficients),
                    1.0,
                    rel_tol=1e-8,
                    abs_tol=1e-10,
                )
                for group, coefficients in zip(groups, weights, strict=True)
            )
        ):
            raise Lfm2SemanticPostfoldError(
                f"Authenticated {role} head coordinates do not partition the parent."
            )
        return tuple(zip(groups, weights, strict=True))

    query_coordinates = validated_groups(
        query_groups_value,
        query_weights_value,
        source_query_heads,
        "query",
    )
    key_coordinates = validated_groups(
        key_groups_value,
        key_weights_value,
        source_key_heads,
        "key/value",
    )
    singular = torch.linalg.svdvals(chart)
    maximum_singular = float(singular.max().item())
    rank_threshold = (
        maximum_singular
        * max(source_hidden, target_hidden)
        * torch.finfo(torch.float64).eps
    )
    chart_rank = int((singular > rank_threshold).sum().item())
    raw_chart_decoder = torch.linalg.pinv(
        chart,
        rtol=max(source_hidden, target_hidden) * torch.finfo(torch.float64).eps,
    )
    if normalized_input_decoder is None:
        decoder = raw_chart_decoder
        decoder_receipt_sha256 = None
        decoder_method = "moore_penrose_pseudoinverse_raw_residual_fallback"
    else:
        decoder = (
            normalized_input_decoder.detach()
            .to(
                device="cpu",
                dtype=torch.float64,
            )
            .contiguous()
        )
        if (
            tuple(decoder.shape) != (target_hidden, source_hidden)
            or normalized_input_decoder_receipt is None
        ):
            raise Lfm2SemanticPostfoldError(
                "Normalized operator-input decoder changed residual coordinates."
            )
        decoder_receipt = _verify_sealed(
            normalized_input_decoder_receipt,
            seal_name="decoder_sha256",
            label="normalized operator-input decoder",
        )
        if decoder_receipt.get("published_decoder_sha256") != _tensor_sha256(decoder):
            raise Lfm2SemanticPostfoldError(
                "Normalized operator-input decoder changed after sealing."
            )
        decoder_receipt_sha256 = decoder_receipt["decoder_sha256"]
        decoder_method = "prior_centered_cached_normalized_input_residual"

    def fold(
        parent_weight: Any,
        coordinates: Sequence[tuple[Sequence[int], Sequence[float]]],
    ) -> tuple[Any, Any]:
        blocked = parent_weight.reshape(-1, head_dim, source_hidden)
        folded = []
        for members, coefficients in coordinates:
            index = torch.tensor(tuple(members), dtype=torch.long)
            alpha = torch.tensor(tuple(coefficients), dtype=torch.float64).reshape(
                -1,
                1,
                1,
            )
            folded.append((blocked.index_select(0, index) * alpha).sum(dim=0))
        source_operator = torch.stack(folded, dim=0).reshape(-1, source_hidden)
        target_operator = (source_operator @ decoder.T).contiguous()
        return source_operator, target_operator

    folded_query_source, query_prior = fold(parent_query, query_coordinates)
    folded_key_source, key_prior = fold(parent_key, key_coordinates)
    folded_value_source, value_prior = fold(parent_value, key_coordinates)
    query_head_fold = torch.zeros(
        (len(query_coordinates) * head_dim, source_query_heads * head_dim),
        dtype=torch.float64,
    )
    for target_head, (members, coefficients) in enumerate(query_coordinates):
        for source_head, coefficient in zip(members, coefficients, strict=True):
            for coordinate in range(head_dim):
                query_head_fold[
                    target_head * head_dim + coordinate,
                    source_head * head_dim + coordinate,
                ] = coefficient
    context_decoder = torch.linalg.pinv(
        query_head_fold.T,
        rtol=max(map(int, query_head_fold.shape)) * torch.finfo(torch.float64).eps,
    )
    output_prior = (chart.T @ parent_output @ context_decoder.T).contiguous()
    if not all(
        bool(torch.isfinite(value).all().item())
        for value in (query_prior, key_prior, value_prior, output_prior)
    ):
        raise Lfm2SemanticPostfoldError(
            "Parent-derived Q/K/V/O weight priors are nonfinite."
        )

    def replay_metrics(source_operator: Any, target_operator: Any) -> Mapping[str, Any]:
        replay = chart @ target_operator.T
        target = source_operator.T
        difference = replay - target
        denominator = target.square().mean().sqrt().clamp_min(1e-30)
        return MappingProxyType(
            {
                "relative_rmse": float(
                    (difference.square().mean().sqrt() / denominator).item()
                ),
                "maximum_absolute_error": float(difference.abs().max().item()),
                "replay_sha256": _tensor_sha256(replay),
                "source_operator_sha256": _tensor_sha256(source_operator),
            }
        )

    body = {
        "schema": "neural-forge-lfm2-parent-qk-weight-prior-v1",
        "source_inventory_sha256": source_inventory_sha256,
        "residual_chart_sha256": residual_chart_sha256,
        "residual_chart_shape": [source_hidden, target_hidden],
        "residual_chart_sha256_observed": _tensor_sha256(chart),
        "residual_decoder_method": decoder_method,
        "residual_decoder_sha256": _tensor_sha256(decoder),
        "normalized_input_decoder_receipt_sha256": decoder_receipt_sha256,
        "raw_residual_chart_decoder_sha256": _tensor_sha256(raw_chart_decoder),
        "residual_chart_rank": chart_rank,
        "residual_chart_rank_threshold": rank_threshold,
        "residual_chart_singular_values_sha256": _tensor_sha256(singular),
        "query_groups": [list(value) for value in query_groups_value],
        "query_group_weights": [list(value) for value in query_weights_value],
        "key_value_groups": [list(value) for value in key_groups_value],
        "key_value_group_weights": [list(value) for value in key_weights_value],
        "query_prior_sha256": _tensor_sha256(query_prior),
        "key_prior_sha256": _tensor_sha256(key_prior),
        "query_norm_weight_sha256": _tensor_sha256(parent_query_norm),
        "key_norm_weight_sha256": _tensor_sha256(parent_key_norm),
        "query_raw_chart_projection_diagnostic": dict(
            replay_metrics(folded_query_source, query_prior)
        ),
        "key_raw_chart_projection_diagnostic": dict(
            replay_metrics(folded_key_source, key_prior)
        ),
        "raw_chart_projection_is_not_normalized_input_replay": True,
        "value_prior_sha256": _tensor_sha256(value_prior),
        "output_prior_sha256": _tensor_sha256(output_prior),
        "folded_value_source_sha256": _tensor_sha256(folded_value_source),
        "query_context_head_fold_sha256": _tensor_sha256(query_head_fold),
        "query_context_decoder_sha256": _tensor_sha256(context_decoder),
        "value_prior_uses_normalized_input_decoder": True,
        "output_prior_uses_residual_chart_and_context_decoder": True,
        "direct_parent_weights_used": True,
        "fit_rows_used": 0,
        "gradients_used": False,
        "training_steps": 0,
    }
    return (
        query_prior,
        key_prior,
        value_prior,
        output_prior,
        parent_query_norm.contiguous(),
        parent_key_norm.contiguous(),
        MappingProxyType({**body, "prior_sha256": canonical_sha256(body)}),
    )


def _fit_normalized_operator_input_decoder(
    *,
    parent_raw_rows: Sequence[Any],
    child_normalized_rows: Sequence[Any],
    parent_operator_norm_weight: Any,
    target_operator_norm_weight: Any,
    parent_norm_epsilon: float,
    target_norm_epsilon: float,
    residual_chart: Any,
    residual_chart_sha256: str,
    fit_trajectory_count: int,
    ridge: float = 1e-6,
) -> tuple[Any, Mapping[str, Any]]:
    """Fit cached normalized-input residual around chart/gamma structural decoder."""

    torch = _torch()
    parent_gamma = parent_operator_norm_weight.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    target_gamma = target_operator_norm_weight.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    chart = residual_chart.detach().to(device="cpu", dtype=torch.float64)
    parents = tuple(
        value.detach().to(device="cpu", dtype=torch.float64)
        for value in parent_raw_rows
    )
    children = tuple(
        value.detach().to(device="cpu", dtype=torch.float64)
        for value in child_normalized_rows
    )
    if (
        chart.ndim != 2
        or len(parents) != len(children)
        or len(parents) < 2
        or fit_trajectory_count <= 0
        or fit_trajectory_count >= len(parents)
        or tuple(parent_gamma.shape) != (int(chart.shape[0]),)
        or tuple(target_gamma.shape) != (int(chart.shape[1]),)
        or bool((target_gamma.abs() <= torch.finfo(torch.float64).eps).any().item())
        or not math.isfinite(parent_norm_epsilon)
        or parent_norm_epsilon <= 0.0
        or not math.isfinite(target_norm_epsilon)
        or target_norm_epsilon <= 0.0
    ):
        raise Lfm2SemanticPostfoldError(
            "Normalized operator-input decoder has incompatible chart/gamma evidence."
        )

    normalized_parent = []
    normalized_child = []
    for parent_raw, child_input in zip(parents, children, strict=True):
        if (
            parent_raw.ndim != 2
            or child_input.ndim != 2
            or int(parent_raw.shape[0]) != int(child_input.shape[0])
            or int(parent_raw.shape[1]) != int(chart.shape[0])
            or int(child_input.shape[1]) != int(chart.shape[1])
        ):
            raise Lfm2SemanticPostfoldError(
                "Normalized operator-input trajectory rows are misaligned."
            )
        variance = parent_raw.square().mean(dim=-1, keepdim=True)
        parent_input = (
            parent_raw * torch.rsqrt(variance + parent_norm_epsilon)
        ) * parent_gamma
        normalized_parent.append(parent_input)
        normalized_child.append(child_input)

    raw_decoder = torch.linalg.pinv(
        chart,
        rtol=max(map(int, chart.shape)) * torch.finfo(torch.float64).eps,
    )
    unscaled_prior = (
        torch.diag(target_gamma.reciprocal()) @ raw_decoder @ torch.diag(parent_gamma)
    ).contiguous()
    fit_parent = torch.cat(normalized_parent[:fit_trajectory_count], dim=0)
    fit_child = torch.cat(normalized_child[:fit_trajectory_count], dim=0)
    fit_source_moments = torch.tensor(
        [
            float(value.square().mean().item())
            for value in parents[:fit_trajectory_count]
        ],
        dtype=torch.float64,
    )
    fit_projected_moments = torch.tensor(
        [
            float((value @ chart).square().mean().item())
            for value in parents[:fit_trajectory_count]
        ],
        dtype=torch.float64,
    )
    source_moment = fit_source_moments.mean()
    projected_moment = fit_projected_moments.mean()
    covariance_scale = torch.sqrt(
        (projected_moment + float(target_norm_epsilon))
        / (source_moment + float(parent_norm_epsilon))
    )
    if (
        not bool(torch.isfinite(covariance_scale).item())
        or float(covariance_scale.item()) <= 0.0
    ):
        raise Lfm2SemanticPostfoldError(
            "Normalized operator-input structural covariance scale is invalid."
        )
    scale_body = {
        "schema": "neural-forge-lfm2-normalized-input-covariance-scale-v1",
        "formula": (
            "sqrt((mean_fit_trajectory(mean((parent_raw@chart)^2))+"
            "target_eps)/(mean_fit_trajectory(mean(parent_raw^2))+parent_eps))"
        ),
        "fit_trajectory_count": fit_trajectory_count,
        "trajectory_balanced": True,
        "fit_source_moment_sha256": _tensor_sha256(fit_source_moments),
        "fit_projected_moment_sha256": _tensor_sha256(fit_projected_moments),
        "mean_fit_source_element_energy": float(source_moment.item()),
        "mean_fit_projected_element_energy": float(projected_moment.item()),
        "parent_norm_epsilon": float(parent_norm_epsilon),
        "target_norm_epsilon": float(target_norm_epsilon),
        "covariance_scale": float(covariance_scale.item()),
        "actual_child_rows_used_for_scale": 0,
        "heldout_rows_used_for_scale": 0,
        "fit_parent_raw_rows_only": True,
    }
    scale_receipt = MappingProxyType(
        {**scale_body, "scale_sha256": canonical_sha256(scale_body)}
    )
    prior = (unscaled_prior * covariance_scale).contiguous()
    fit_residual = fit_parent - fit_child @ prior
    gram = fit_child @ fit_child.T
    scale = float(gram.diagonal().mean().clamp_min(1.0).item())
    effective_ridge = float(ridge) * scale
    correction = fit_child.T @ torch.linalg.solve(
        gram + effective_ridge * torch.eye(int(gram.shape[0]), dtype=torch.float64),
        fit_residual,
    )
    decoder = (prior + correction).contiguous()
    if not bool(torch.isfinite(decoder).all().item()):
        raise Lfm2SemanticPostfoldError(
            "Normalized operator-input decoder residual solve is nonfinite."
        )
    _left, singular, right_h = torch.linalg.svd(fit_child, full_matrices=True)
    maximum_singular = (
        0.0 if int(singular.numel()) == 0 else float(singular.max().item())
    )
    threshold = (
        maximum_singular
        * max(map(int, fit_child.shape))
        * torch.finfo(torch.float64).eps
    )
    rank = int((singular > threshold).sum().item())
    null_basis = right_h[rank:]
    nullspace_change = (
        torch.zeros((), dtype=torch.float64)
        if int(null_basis.shape[0]) == 0
        else (null_basis @ correction).abs().max()
    )

    def metrics(child_rows: Any, parent_rows: Any) -> Mapping[str, Any]:
        difference = child_rows @ decoder - parent_rows
        denominator = parent_rows.square().mean().sqrt().clamp_min(1e-30)
        return MappingProxyType(
            {
                "row_count": int(child_rows.shape[0]),
                "relative_rmse": float(
                    (difference.square().mean().sqrt() / denominator).item()
                ),
                "maximum_absolute_error": float(difference.abs().max().item()),
            }
        )

    heldout_parent = torch.cat(normalized_parent[fit_trajectory_count:], dim=0)
    heldout_child = torch.cat(normalized_child[fit_trajectory_count:], dim=0)
    heldout_rowwise_denominator_ratios = torch.cat(
        tuple(
            torch.sqrt(
                ((value @ chart).square().mean(dim=-1) + float(target_norm_epsilon))
                / (value.square().mean(dim=-1) + float(parent_norm_epsilon))
            )
            for value in parents[fit_trajectory_count:]
        ),
        dim=0,
    )
    body = {
        "schema": "neural-forge-lfm2-normalized-operator-input-decoder-v1",
        "residual_chart_sha256": residual_chart_sha256,
        "residual_chart_observed_sha256": _tensor_sha256(chart),
        "parent_operator_norm_weight_sha256": _tensor_sha256(parent_gamma),
        "target_operator_norm_weight_sha256": _tensor_sha256(target_gamma),
        "parent_norm_epsilon": float(parent_norm_epsilon),
        "target_norm_epsilon": float(target_norm_epsilon),
        "parent_raw_trajectory_sha256": [_tensor_sha256(value) for value in parents],
        "child_normalized_trajectory_sha256": [
            _tensor_sha256(value) for value in children
        ],
        "structural_prior": (
            "alpha_cov@diag(target_gamma)^-1@pinv(residual_chart)@diag(parent_gamma)"
        ),
        "unscaled_structural_prior_sha256": _tensor_sha256(unscaled_prior),
        "structural_covariance_scale": dict(scale_receipt),
        "structural_prior_sha256": _tensor_sha256(prior),
        "residual_correction_sha256": _tensor_sha256(correction),
        "published_decoder_sha256": _tensor_sha256(decoder),
        "fit_trajectory_count": fit_trajectory_count,
        "heldout_trajectory_count": len(parents) - fit_trajectory_count,
        "fit_metrics": dict(metrics(fit_child, fit_parent)),
        "heldout_metrics": dict(metrics(heldout_child, heldout_parent)),
        "heldout_rowwise_denominator_ratio_diagnostic": {
            "row_count": int(heldout_rowwise_denominator_ratios.numel()),
            "minimum": float(heldout_rowwise_denominator_ratios.min().item()),
            "maximum": float(heldout_rowwise_denominator_ratios.max().item()),
            "mean": float(heldout_rowwise_denominator_ratios.mean().item()),
            "standard_deviation": float(
                heldout_rowwise_denominator_ratios.std(unbiased=False).item()
            ),
            "ratio_sha256": _tensor_sha256(heldout_rowwise_denominator_ratios),
            "diagnostic_only": True,
        },
        "fit_design_rank": rank,
        "fit_design_width": int(fit_child.shape[1]),
        "fit_design_nullity": int(null_basis.shape[0]),
        "fit_nullspace_correction_maximum_absolute": float(nullspace_change.item()),
        "prior_preserved_off_fit_design": True,
        "ridge": float(ridge),
        "effective_ridge": effective_ridge,
        "fit_rows_only_used_for_residual": True,
        "heldout_rows_used_for_fit": 0,
        "gradients_used": False,
        "training_steps": 0,
    }
    return decoder, MappingProxyType({**body, "decoder_sha256": canonical_sha256(body)})


@dataclass(frozen=True, slots=True)
class Lfm2SemanticAttentionTransportEvidence:
    """One bounded batch/layer teacher lattice handed to the installer."""

    batch_index: int
    layer_index: int
    alignment_sha256: str
    alignment: Lfm2SemanticSpanCpcfPhaseAlignment = field(
        repr=False,
        compare=False,
    )
    capability: Mapping[str, Any]
    history: Mapping[str, Any]
    actual_child_replay: Lfm2SemanticAttentionReplayEndpoint = field(
        repr=False,
        compare=False,
    )
    source_inventory: Lfm2DenseQkvoSourceInventory = field(
        repr=False,
        compare=False,
    )
    query_groups: tuple[tuple[int, ...], ...]
    query_group_weights: tuple[tuple[float, ...], ...]
    key_value_groups: tuple[tuple[int, ...], ...]
    key_value_group_weights: tuple[tuple[float, ...], ...]
    compiler_lineage_sha256: str
    transport: SemanticAttentionExpansionTransport = field(
        repr=False,
        compare=False,
    )
    folded_parent_query_pre_rope: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    folded_parent_key_pre_rope: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    parent_pre_rope_prior_receipt: Mapping[str, Any] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    parent_query_weight_prior: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    parent_key_weight_prior: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    parent_value_weight_prior: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    parent_output_weight_prior: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    parent_query_norm_weight: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    parent_key_norm_weight: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    parent_weight_prior_receipt: Mapping[str, Any] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    evidence_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.batch_index, bool)
            or not isinstance(self.batch_index, int)
            or self.batch_index < 0
            or isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
            or not isinstance(
                self.alignment,
                Lfm2SemanticSpanCpcfPhaseAlignment,
            )
            or not isinstance(self.transport, SemanticAttentionExpansionTransport)
            or not isinstance(
                self.actual_child_replay,
                Lfm2SemanticAttentionReplayEndpoint,
            )
            or not isinstance(
                self.source_inventory,
                Lfm2DenseQkvoSourceInventory,
            )
        ):
            raise Lfm2SemanticPostfoldError(
                "Semantic attention evidence coordinates are invalid."
            )
        _digest(self.alignment_sha256, "semantic alignment SHA-256")
        capability = _verify_sealed(
            self.capability,
            seal_name="capability_sha256",
            label="attention capability",
        )
        history = _verify_sealed(
            self.history,
            seal_name="history_sha256",
            label="attention history",
        )
        groups = tuple(
            tuple(int(item) for item in group) for group in self.query_groups
        )
        weights = tuple(
            tuple(float(item) for item in group) for group in self.query_group_weights
        )
        key_value_groups = tuple(
            tuple(int(item) for item in group) for group in self.key_value_groups
        )
        key_value_weights = tuple(
            tuple(float(item) for item in group)
            for group in self.key_value_group_weights
        )
        lineage = _digest(
            self.compiler_lineage_sha256,
            "compiler lineage SHA-256",
        )
        if (
            self.alignment.alignment_sha256 != self.alignment_sha256
            or self.transport.semantic_alignment_sha256 != self.alignment_sha256
            or self.actual_child_replay.alignment_sha256 != self.alignment_sha256
            or self.actual_child_replay.batch_index != self.batch_index
            or self.actual_child_replay.layer_index != self.layer_index
            or self.source_inventory.verify() != self.source_inventory.inventory_sha256
            or not groups
            or len(groups) != len(weights)
            or not key_value_groups
            or len(key_value_groups) != len(key_value_weights)
        ):
            raise Lfm2SemanticPostfoldError(
                "Semantic attention transport/replay/compiler lineage diverges."
            )
        batch = int(self.transport.probabilities.shape[0])
        target_query_heads = len(groups)
        target_key_value_heads = len(key_value_groups)
        query_tokens = len(self.transport.query_donor_positions)
        key_tokens = len(self.transport.target_to_donor_span_owner_positions)
        head_dim = int(self.transport.query_rope_cos.shape[-1])
        prior_values = (
            self.folded_parent_query_pre_rope,
            self.folded_parent_key_pre_rope,
            self.parent_pre_rope_prior_receipt,
        )
        prior_receipt: Mapping[str, Any] | None = None
        if any(value is not None for value in prior_values):
            torch = _torch()
            query_prior = self.folded_parent_query_pre_rope
            key_prior = self.folded_parent_key_pre_rope
            if (
                not isinstance(query_prior, torch.Tensor)
                or not isinstance(key_prior, torch.Tensor)
                or query_prior.ndim != 4
                or key_prior.ndim != 4
                or query_prior.layout != torch.strided
                or key_prior.layout != torch.strided
                or query_prior.device.type == "meta"
                or key_prior.device.type == "meta"
                or query_prior.requires_grad
                or key_prior.requires_grad
                or not torch.is_floating_point(query_prior)
                or not torch.is_floating_point(key_prior)
                or not bool(torch.isfinite(query_prior).all().item())
                or not bool(torch.isfinite(key_prior).all().item())
            ):
                raise Lfm2SemanticPostfoldError(
                    "Folded parent pre-RoPE priors must be finite detached tensors."
                )
            query_prior = query_prior.detach().to(device="cpu").contiguous()
            key_prior = key_prior.detach().to(device="cpu").contiguous()
            if (
                tuple(query_prior.shape)
                != (batch, target_query_heads, query_tokens, head_dim)
                or tuple(key_prior.shape)
                != (batch, target_key_value_heads, key_tokens, head_dim)
                or self.parent_pre_rope_prior_receipt is None
            ):
                raise Lfm2SemanticPostfoldError(
                    "Folded parent pre-RoPE priors changed semantic geometry."
                )
            prior_receipt = _verify_sealed(
                self.parent_pre_rope_prior_receipt,
                seal_name="prior_sha256",
                label="parent pre-RoPE prior",
            )
            if (
                prior_receipt.get("source_history_sha256") != history["history_sha256"]
                or prior_receipt.get("transport_sha256")
                != self.transport.transport_sha256
                or prior_receipt.get("query_prior_sha256")
                != _tensor_sha256(query_prior)
                or prior_receipt.get("key_prior_sha256") != _tensor_sha256(key_prior)
                or prior_receipt.get("query_groups")
                != [list(group) for group in groups]
                or prior_receipt.get("query_group_weights")
                != [list(group) for group in weights]
                or prior_receipt.get("key_value_groups")
                != [list(group) for group in key_value_groups]
                or prior_receipt.get("key_value_group_weights")
                != [list(group) for group in key_value_weights]
                or prior_receipt.get("parent_activation_evidence_only") is not True
            ):
                raise Lfm2SemanticPostfoldError(
                    "Folded parent pre-RoPE prior is not bound to its evidence."
                )
            object.__setattr__(self, "folded_parent_query_pre_rope", query_prior)
            object.__setattr__(self, "folded_parent_key_pre_rope", key_prior)
        elif not all(value is None for value in prior_values):
            raise AssertionError("Internal parent pre-RoPE prior validation failed.")
        weight_prior_values = (
            self.parent_query_weight_prior,
            self.parent_key_weight_prior,
            self.parent_value_weight_prior,
            self.parent_output_weight_prior,
            self.parent_query_norm_weight,
            self.parent_key_norm_weight,
            self.parent_weight_prior_receipt,
        )
        weight_prior_receipt: Mapping[str, Any] | None = None
        if any(value is not None for value in weight_prior_values):
            torch = _torch()
            query_weight_prior = self.parent_query_weight_prior
            key_weight_prior = self.parent_key_weight_prior
            value_weight_prior = self.parent_value_weight_prior
            output_weight_prior = self.parent_output_weight_prior
            query_norm_weight = self.parent_query_norm_weight
            key_norm_weight = self.parent_key_norm_weight
            if (
                not isinstance(query_weight_prior, torch.Tensor)
                or not isinstance(key_weight_prior, torch.Tensor)
                or query_weight_prior.ndim != 2
                or key_weight_prior.ndim != 2
                or not isinstance(value_weight_prior, torch.Tensor)
                or not isinstance(output_weight_prior, torch.Tensor)
                or value_weight_prior.ndim != 2
                or output_weight_prior.ndim != 2
                or not isinstance(query_norm_weight, torch.Tensor)
                or not isinstance(key_norm_weight, torch.Tensor)
                or query_norm_weight.ndim != 1
                or key_norm_weight.ndim != 1
                or query_weight_prior.requires_grad
                or key_weight_prior.requires_grad
                or query_weight_prior.device.type == "meta"
                or key_weight_prior.device.type == "meta"
                or not bool(torch.isfinite(query_weight_prior).all().item())
                or not bool(torch.isfinite(key_weight_prior).all().item())
                or int(query_weight_prior.shape[0]) != target_query_heads * head_dim
                or int(key_weight_prior.shape[0]) != target_key_value_heads * head_dim
                or int(query_weight_prior.shape[1]) != int(key_weight_prior.shape[1])
                or tuple(value_weight_prior.shape) != tuple(key_weight_prior.shape)
                or tuple(output_weight_prior.shape)
                != (
                    int(query_weight_prior.shape[1]),
                    int(query_weight_prior.shape[0]),
                )
                or tuple(query_norm_weight.shape) != (head_dim,)
                or tuple(key_norm_weight.shape) != (head_dim,)
                or not bool(torch.isfinite(query_norm_weight).all().item())
                or not bool(torch.isfinite(key_norm_weight).all().item())
                or self.parent_weight_prior_receipt is None
            ):
                raise Lfm2SemanticPostfoldError(
                    "Parent-derived Q/K weight priors changed target topology."
                )
            query_weight_prior = (
                query_weight_prior.detach()
                .to(
                    device="cpu",
                    dtype=torch.float64,
                )
                .contiguous()
            )
            key_weight_prior = (
                key_weight_prior.detach()
                .to(
                    device="cpu",
                    dtype=torch.float64,
                )
                .contiguous()
            )
            value_weight_prior = (
                value_weight_prior.detach()
                .to(
                    device="cpu",
                    dtype=torch.float64,
                )
                .contiguous()
            )
            output_weight_prior = (
                output_weight_prior.detach()
                .to(
                    device="cpu",
                    dtype=torch.float64,
                )
                .contiguous()
            )
            query_norm_weight = (
                query_norm_weight.detach()
                .to(
                    device="cpu",
                    dtype=torch.float64,
                )
                .contiguous()
            )
            key_norm_weight = (
                key_norm_weight.detach()
                .to(
                    device="cpu",
                    dtype=torch.float64,
                )
                .contiguous()
            )
            weight_prior_receipt = _verify_sealed(
                self.parent_weight_prior_receipt,
                seal_name="prior_sha256",
                label="parent Q/K weight prior",
            )
            if (
                weight_prior_receipt.get("source_inventory_sha256")
                != self.source_inventory.inventory_sha256
                or weight_prior_receipt.get("query_prior_sha256")
                != _tensor_sha256(query_weight_prior)
                or weight_prior_receipt.get("key_prior_sha256")
                != _tensor_sha256(key_weight_prior)
                or weight_prior_receipt.get("value_prior_sha256")
                != _tensor_sha256(value_weight_prior)
                or weight_prior_receipt.get("output_prior_sha256")
                != _tensor_sha256(output_weight_prior)
                or weight_prior_receipt.get("query_norm_weight_sha256")
                != _tensor_sha256(query_norm_weight)
                or weight_prior_receipt.get("key_norm_weight_sha256")
                != _tensor_sha256(key_norm_weight)
                or weight_prior_receipt.get("query_groups")
                != [list(group) for group in groups]
                or weight_prior_receipt.get("query_group_weights")
                != [list(group) for group in weights]
                or weight_prior_receipt.get("key_value_groups")
                != [list(group) for group in key_value_groups]
                or weight_prior_receipt.get("key_value_group_weights")
                != [list(group) for group in key_value_weights]
                or weight_prior_receipt.get("direct_parent_weights_used") is not True
            ):
                raise Lfm2SemanticPostfoldError(
                    "Parent-derived Q/K weight prior is not bound to its lineage."
                )
            object.__setattr__(self, "parent_query_weight_prior", query_weight_prior)
            object.__setattr__(self, "parent_key_weight_prior", key_weight_prior)
            object.__setattr__(self, "parent_value_weight_prior", value_weight_prior)
            object.__setattr__(self, "parent_output_weight_prior", output_weight_prior)
            object.__setattr__(self, "parent_query_norm_weight", query_norm_weight)
            object.__setattr__(self, "parent_key_norm_weight", key_norm_weight)
        elif not all(value is None for value in weight_prior_values):
            raise AssertionError("Internal parent Q/K weight-prior validation failed.")
        object.__setattr__(self, "capability", MappingProxyType(capability))
        object.__setattr__(self, "history", MappingProxyType(history))
        object.__setattr__(self, "query_groups", groups)
        object.__setattr__(self, "query_group_weights", weights)
        object.__setattr__(self, "key_value_groups", key_value_groups)
        object.__setattr__(self, "key_value_group_weights", key_value_weights)
        object.__setattr__(self, "compiler_lineage_sha256", lineage)
        object.__setattr__(
            self,
            "parent_pre_rope_prior_receipt",
            None if prior_receipt is None else MappingProxyType(dict(prior_receipt)),
        )
        object.__setattr__(
            self,
            "parent_weight_prior_receipt",
            (
                None
                if weight_prior_receipt is None
                else MappingProxyType(dict(weight_prior_receipt))
            ),
        )
        object.__setattr__(self, "evidence_sha256", canonical_sha256(self._body()))

    def endpoint_residual_evidence(self) -> dict[str, Any]:
        return self.transport.endpoint_residual_evidence()

    def _body(self) -> dict[str, Any]:
        return {
            "schema": "neural-forge-lfm2-semantic-attention-transport-evidence-v1",
            "batch_index": self.batch_index,
            "layer_index": self.layer_index,
            "alignment_sha256": self.alignment_sha256,
            "phase": self.alignment.phase,
            "capability_sha256": self.capability["capability_sha256"],
            "history_sha256": self.history["history_sha256"],
            "transport_sha256": self.transport.transport_sha256,
            "actual_child_replay_sha256": self.actual_child_replay.replay_sha256,
            "source_inventory_sha256": self.source_inventory.inventory_sha256,
            "query_groups_sha256": canonical_sha256(
                {
                    "groups": [list(value) for value in self.query_groups],
                    "weights": [list(value) for value in self.query_group_weights],
                }
            ),
            "key_value_groups_sha256": canonical_sha256(
                {
                    "groups": [list(value) for value in self.key_value_groups],
                    "weights": [list(value) for value in self.key_value_group_weights],
                }
            ),
            "compiler_lineage_sha256": self.compiler_lineage_sha256,
            "parent_pre_rope_prior_available": (
                self.parent_pre_rope_prior_receipt is not None
            ),
            "parent_pre_rope_prior_sha256": (
                None
                if self.parent_pre_rope_prior_receipt is None
                else self.parent_pre_rope_prior_receipt["prior_sha256"]
            ),
            "parent_weight_prior_available": (
                self.parent_weight_prior_receipt is not None
            ),
            "parent_weight_prior_sha256": (
                None
                if self.parent_weight_prior_receipt is None
                else self.parent_weight_prior_receipt["prior_sha256"]
            ),
            "direct_endpoint_case_suffix_compatible": (
                self.transport.direct_endpoint_case_suffix_compatible
            ),
            "probability_mass_error": (self.transport.maximum_probability_mass_error),
            "context_absolute_error": (self.transport.maximum_context_absolute_error),
        }

    def to_evidence_dict(self) -> dict[str, Any]:
        return {**self._body(), "evidence_sha256": self.evidence_sha256}


@dataclass(frozen=True, slots=True)
class Lfm2SemanticLayerStackInstallation:
    """Installed causal stack plus deferred evidence for its shared final W2."""

    receipt: Mapping[str, Any]
    final_layer_mlp_evidence: Mapping[
        str,
        Lfm2CpcfTerminalActionEvidence,
    ] = field(repr=False, compare=False)
    final_layer_mlp_down_weight_prior: Lfm2CpcfTerminalActionDownWeightPrior | None = (
        field(default=None, repr=False, compare=False)
    )

    def __post_init__(self) -> None:
        receipt = _verify_sealed(
            self.receipt,
            seal_name="result_sha256",
            label="semantic layer-stack installation",
        )
        evidence = dict(self.final_layer_mlp_evidence)
        expected_splits = {"fit", "fit_check", "selection", "validation"}
        if set(evidence) != expected_splits or any(
            not isinstance(value, Lfm2CpcfTerminalActionEvidence)
            or value.split != split
            or not value.verify_authentication()
            for split, value in evidence.items()
        ):
            raise Lfm2SemanticPostfoldError(
                "Layer-stack final-MLP evidence does not cover every split."
            )
        evidence_sha = {
            split: evidence[split].authentication_sha256 for split in sorted(evidence)
        }
        if (
            receipt.get("final_layer_mlp_w2_deferred_to_fused_terminal") is not True
            or receipt.get("final_layer_mlp_w2_install_count") != 0
            or receipt.get("final_layer_mlp_evidence_sha256") != evidence_sha
        ):
            raise Lfm2SemanticPostfoldError(
                "Layer-stack receipt does not authenticate its deferred final W2."
            )
        recipe = receipt.get("body_recipe")
        component_lanes = (
            recipe.get("component_lanes") if isinstance(recipe, Mapping) else None
        )
        qdeim_lane = bool(
            isinstance(component_lanes, Mapping)
            and component_lanes.get("mlp")
            in {
                "nonlinear_qdeim_structural_priority",
                "nonlinear_qdeim_fit_corrected_priority",
            }
        )
        prior = self.final_layer_mlp_down_weight_prior
        if qdeim_lane:
            if (
                not isinstance(prior, Lfm2CpcfTerminalActionDownWeightPrior)
                or not prior.verify_authentication()
                or prior.body_recipe_id != receipt.get("body_recipe_id")
                or prior.body_recipe_sha256 != receipt.get("body_recipe_sha256")
                or prior.mlp_lane != component_lanes.get("mlp")
                or receipt.get(
                    "final_layer_mlp_down_weight_prior_authentication_sha256"
                )
                != prior.authentication_sha256
                or receipt.get("final_layer_mlp_down_weight_prior_tensor_sha256")
                != _terminal_action_tensor_sha256(
                    prior.down_weight,
                    "final_layer_mlp_down_weight_prior",
                )
                or receipt.get("final_layer_mlp_down_weight_prior_live_w2_sha256")
                != prior.live_deferred_w2_sha256
                or receipt.get(
                    "final_layer_mlp_down_weight_prior_installed_before_terminal"
                )
                is not False
                or receipt.get("final_layer_mlp_down_weight_prior_install_count") != 0
            ):
                raise Lfm2SemanticPostfoldError(
                    "Layer-stack receipt does not authenticate its Q-DEIM W2 prior."
                )
        elif prior is not None:
            raise Lfm2SemanticPostfoldError(
                "A legacy layer-stack result unexpectedly carries a Q-DEIM W2 prior."
            )
        object.__setattr__(self, "receipt", MappingProxyType(receipt))
        object.__setattr__(
            self,
            "final_layer_mlp_evidence",
            MappingProxyType(evidence),
        )
        object.__setattr__(self, "final_layer_mlp_down_weight_prior", prior)


def seal_lfm2_semantic_attention_installation_receipt(
    *,
    layer_index: int,
    transport_evidence_sha256: Sequence[str],
    model_state_before_sha256: str,
    model_state_after_sha256: str,
    changed_state_keys: Sequence[str],
    parameter_element_count: int,
) -> Mapping[str, Any]:
    """Canonical receipt helper for the future analytic attention compiler."""

    evidence = tuple(
        _digest(value, "transport evidence SHA-256")
        for value in transport_evidence_sha256
    )
    changed = tuple(changed_state_keys)
    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or layer_index < 0
        or not evidence
        or not changed
        or any(not isinstance(value, str) or not value for value in changed)
        or isinstance(parameter_element_count, bool)
        or not isinstance(parameter_element_count, int)
        or parameter_element_count <= 0
    ):
        raise Lfm2SemanticPostfoldError(
            "Attention installation receipt inputs are incomplete."
        )
    before = _digest(model_state_before_sha256, "model state before")
    after = _digest(model_state_after_sha256, "model state after")
    if before == after:
        raise Lfm2SemanticPostfoldError(
            "An installed attention correction must change model state."
        )
    body = {
        "schema": LFM2_SEMANTIC_ATTENTION_INSTALLATION_SCHEMA,
        "layer_index": layer_index,
        "installed": True,
        "transport_evidence_sha256": list(evidence),
        "model_state_before_sha256": before,
        "model_state_after_sha256": after,
        "changed_state_keys": list(changed),
        "parameter_element_count_before": parameter_element_count,
        "parameter_element_count_after": parameter_element_count,
        "parameter_count_changed": False,
        "analytic_cached_evidence_only": True,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    return MappingProxyType(_sealed(body, "result_sha256"))


def _validate_attention_installation_receipt(
    value: Mapping[str, Any],
    *,
    layer_index: int,
    evidence: Sequence[Lfm2SemanticAttentionTransportEvidence],
    before_sha256: str,
    after_sha256: str,
    changed_keys: Sequence[str],
    parameter_count: int,
) -> dict[str, Any]:
    receipt = _verify_sealed(
        value,
        seal_name="result_sha256",
        label="semantic attention installation receipt",
    )
    allowed_prefix = f"model.layers.{layer_index}.self_attn."
    expected_evidence = [item.evidence_sha256 for item in evidence]
    if (
        receipt.get("schema") != LFM2_SEMANTIC_ATTENTION_INSTALLATION_SCHEMA
        or receipt.get("layer_index") != layer_index
        or receipt.get("installed") is not True
        or receipt.get("transport_evidence_sha256") != expected_evidence
        or receipt.get("model_state_before_sha256") != before_sha256
        or receipt.get("model_state_after_sha256") != after_sha256
        or receipt.get("changed_state_keys") != list(changed_keys)
        or not changed_keys
        or any(not name.startswith(allowed_prefix) for name in changed_keys)
        or receipt.get("parameter_element_count_before") != parameter_count
        or receipt.get("parameter_element_count_after") != parameter_count
        or receipt.get("parameter_count_changed") is not False
        or receipt.get("analytic_cached_evidence_only") is not True
        or receipt.get("operation_audit") != dict(_ZERO_OPERATIONS)
    ):
        raise Lfm2SemanticPostfoldError(
            "Analytic attention installer returned an invalid or over-broad receipt."
        )
    return receipt


def _authenticate_lfm2_coupled_attention_group_qdeim_operator_receipt(
    receipt: Mapping[str, Any],
    *,
    layer_index: int,
    body_recipe: str | Lfm2SemanticBodyRecipe | None = None,
) -> Mapping[str, Any]:
    """Authenticate one persisted intact-group Q-DEIM six-tensor install."""

    recipe = _authenticate_body_recipe_binding(
        receipt,
        body_recipe=body_recipe,
        component="qk",
        label=f"coupled attention Q-DEIM layer {layer_index} receipt",
    )
    raw_value = receipt.get("coupled_attention_installation_receipt")
    adapter_value = receipt.get("coupled_attention_candidate_receipt")
    compiler_value = receipt.get("coupled_attention_compiler_receipt")
    norm_value = receipt.get("source_norm_inventory")
    direct_value = receipt.get("direct_attention_source_receipt")
    decoder_value = receipt.get("normalized_input_decoder_receipt")
    chart_value = receipt.get("residual_chart_receipt")
    if not all(
        isinstance(value, Mapping)
        for value in (
            raw_value,
            adapter_value,
            compiler_value,
            norm_value,
            direct_value,
            decoder_value,
            chart_value,
        )
    ):
        raise Lfm2SemanticPostfoldError(
            f"Coupled attention Q-DEIM layer {layer_index} omitted provenance."
        )
    raw = _verify_sealed(
        raw_value,
        seal_name="result_sha256",
        label=f"coupled attention Q-DEIM layer {layer_index} installation",
    )
    adapter = _verify_sealed(
        adapter_value,
        seal_name="result_sha256",
        label=f"coupled attention Q-DEIM layer {layer_index} cache adapter",
    )
    compiler = _verify_sealed(
        compiler_value,
        seal_name="result_sha256",
        label=f"coupled attention Q-DEIM layer {layer_index} compiler",
    )
    capability_value = adapter.get("cache_capability_receipt")
    context_bank_value = adapter.get("context_bank_receipt")
    if not isinstance(capability_value, Mapping) or not isinstance(
        context_bank_value,
        Mapping,
    ):
        raise Lfm2SemanticPostfoldError(
            f"Coupled attention Q-DEIM layer {layer_index} omitted cache evidence roots."
        )
    capability = _verify_sealed(
        capability_value,
        seal_name="result_sha256",
        label=f"coupled attention Q-DEIM layer {layer_index} cache capability",
    )
    context_bank = _verify_sealed(
        context_bank_value,
        seal_name="result_sha256",
        label=f"coupled attention Q-DEIM layer {layer_index} context bank",
    )
    direct = _verify_sealed(
        direct_value,
        seal_name="result_sha256",
        label=f"coupled attention Q-DEIM layer {layer_index} direct source",
    )
    decoder = _verify_sealed(
        decoder_value,
        seal_name="decoder_sha256",
        label=f"coupled attention Q-DEIM layer {layer_index} decoder",
    )
    chart = _verify_sealed(
        chart_value,
        seal_name="result_sha256",
        label=f"coupled attention Q-DEIM layer {layer_index} residual chart",
    )
    norm = dict(norm_value)
    norm_sha256 = norm.pop("inventory_sha256", None)
    if norm_sha256 != canonical_sha256(norm):
        raise Lfm2SemanticPostfoldError(
            f"Coupled attention Q-DEIM layer {layer_index} norm inventory is stale."
        )
    norm = {**norm, "inventory_sha256": norm_sha256}
    changed_keys = receipt.get("changed_state_keys")
    transport_evidence = receipt.get("transport_evidence_sha256")
    prefix = f"model.layers.{layer_index}.self_attn."
    required_projection_roles = _ATTENTION_SIX_TENSOR_ROLES[:4]
    optional_norm_roles = _ATTENTION_SIX_TENSOR_ROLES[4:]
    required_projection_keys = {prefix + role for role in required_projection_roles}
    allowed_changed_keys = required_projection_keys | {
        prefix + role for role in optional_norm_roles
    }
    transaction_covered_keys = sorted(allowed_changed_keys)
    candidate_hashes = adapter.get("candidate_tensor_sha256")
    norm_hashes = norm.get("tensor_receipts")
    if not isinstance(candidate_hashes, Mapping) or not isinstance(
        norm_hashes,
        Mapping,
    ):
        raise Lfm2SemanticPostfoldError(
            f"Coupled attention Q-DEIM layer {layer_index} omitted tensor hashes."
        )
    six_tensor_bundle = {
        "q_proj.weight": candidate_hashes.get("q_proj.weight"),
        "k_proj.weight": candidate_hashes.get("k_proj.weight"),
        "v_proj.weight": candidate_hashes.get("v_proj.weight"),
        "out_proj.weight": candidate_hashes.get("out_proj.weight"),
        "q_layernorm.weight": (
            norm_hashes.get("q_layernorm.weight", {}).get("tensor_sha256")
            if isinstance(norm_hashes.get("q_layernorm.weight"), Mapping)
            else None
        ),
        "k_layernorm.weight": (
            norm_hashes.get("k_layernorm.weight", {}).get("tensor_sha256")
            if isinstance(norm_hashes.get("k_layernorm.weight"), Mapping)
            else None
        ),
    }
    target_seed_authentication = raw.get("target_seed_authentication")
    norm_seed_transitions = raw.get("norm_seed_transitions")
    compiler_topology = compiler.get("topology")
    seed_shapes: dict[str, tuple[int, ...]] = {}
    if isinstance(compiler_topology, Mapping):
        target_hidden_size = compiler_topology.get("target_hidden_size")
        target_query_head_count = compiler_topology.get("target_query_head_count")
        target_key_value_head_count = compiler_topology.get(
            "target_key_value_head_count"
        )
        head_dim = compiler_topology.get("head_dim")
        dimensions = (
            target_hidden_size,
            target_query_head_count,
            target_key_value_head_count,
            head_dim,
        )
        if all(
            not isinstance(value, bool) and isinstance(value, int) and value > 0
            for value in dimensions
        ):
            query_width = target_query_head_count * head_dim
            key_value_width = target_key_value_head_count * head_dim
            seed_shapes = {
                "q_proj.weight": (query_width, target_hidden_size),
                "k_proj.weight": (key_value_width, target_hidden_size),
                "v_proj.weight": (key_value_width, target_hidden_size),
                "out_proj.weight": (target_hidden_size, query_width),
                "q_layernorm.weight": (head_dim,),
                "k_layernorm.weight": (head_dim,),
            }
    torch = _torch()
    exact_zero_seed_sha256 = {
        role: _tensor_sha256(torch.zeros(shape, dtype=torch.bfloat16))
        for role, shape in seed_shapes.items()
    }
    exact_one_norm_seed_sha256 = {
        role: _tensor_sha256(torch.ones(seed_shapes[role], dtype=torch.bfloat16))
        for role in optional_norm_roles
        if role in seed_shapes
    }
    expected_changed_roles: list[str] = list(required_projection_roles)
    expected_changed_norm_roles: list[str] = []
    norm_transition_contract_valid = bool(
        isinstance(target_seed_authentication, Mapping)
        and set(target_seed_authentication) == set(_ATTENTION_SIX_TENSOR_ROLES)
        and set(seed_shapes) == set(_ATTENTION_SIX_TENSOR_ROLES)
        and set(norm_hashes) == set(optional_norm_roles)
        and isinstance(norm_hashes.get("q_layernorm.weight"), Mapping)
        and isinstance(norm_hashes.get("k_layernorm.weight"), Mapping)
        and norm_hashes["q_layernorm.weight"].get("shape")
        == list(seed_shapes.get("q_layernorm.weight", ()))
        and norm_hashes["k_layernorm.weight"].get("shape")
        == list(seed_shapes.get("k_layernorm.weight", ()))
        and norm_hashes["q_layernorm.weight"].get("dtype") == "torch.bfloat16"
        and norm_hashes["k_layernorm.weight"].get("dtype") == "torch.bfloat16"
        and raw.get("target_seed_authentication_sha256")
        == canonical_sha256(target_seed_authentication)
        and isinstance(norm_seed_transitions, Mapping)
        and set(norm_seed_transitions) == set(optional_norm_roles)
    )
    if norm_transition_contract_valid:
        for role in required_projection_roles:
            seed = target_seed_authentication.get(role)
            norm_transition_contract_valid = bool(
                norm_transition_contract_valid
                and isinstance(seed, Mapping)
                and seed.get("state_key") == prefix + role
                and seed.get("seed_policy") == "exact-zero-blank-required"
                and seed.get("seed_kind") == "exact-zero-blank"
                and seed.get("seed_tensor_sha256") == exact_zero_seed_sha256.get(role)
                and _digest(
                    seed.get("seed_tensor_sha256"),
                    f"coupled attention {role} seed",
                )
                == seed.get("seed_tensor_sha256")
            )
        for role in optional_norm_roles:
            seed = target_seed_authentication.get(role)
            transition = norm_seed_transitions.get(role)
            source_sha256 = six_tensor_bundle[role]
            seed_sha256 = (
                seed.get("seed_tensor_sha256")
                if isinstance(
                    seed,
                    Mapping,
                )
                else None
            )
            bytes_changed = bool(source_sha256 != seed_sha256)
            expected_transition = (
                {
                    **dict(seed),
                    "source_norm_tensor_sha256": source_sha256,
                    "candidate_equal_to_seed": not bytes_changed,
                    "bytes_changed": bytes_changed,
                }
                if isinstance(seed, Mapping)
                else None
            )
            norm_transition_contract_valid = bool(
                norm_transition_contract_valid
                and isinstance(seed, Mapping)
                and seed.get("state_key") == prefix + role
                and seed.get("seed_policy")
                == "exact-zero-legacy-blank-or-exact-one-canonical-neutral"
                and seed.get("seed_kind")
                in {
                    "exact-zero-legacy-blank",
                    "exact-one-canonical-neutral",
                }
                and seed.get("seed_tensor_sha256")
                == (
                    exact_zero_seed_sha256.get(role)
                    if seed.get("seed_kind") == "exact-zero-legacy-blank"
                    else exact_one_norm_seed_sha256.get(role)
                )
                and _digest(
                    seed_sha256,
                    f"coupled attention {role} seed",
                )
                == seed_sha256
                and isinstance(transition, Mapping)
                and dict(transition) == expected_transition
            )
            if bytes_changed:
                expected_changed_roles.append(role)
                expected_changed_norm_roles.append(role)
    expected_changed_roles = sorted(expected_changed_roles)
    expected_unchanged_roles = sorted(
        set(_ATTENTION_SIX_TENSOR_ROLES) - set(expected_changed_roles)
    )
    expected_changed_norm_roles = sorted(expected_changed_norm_roles)
    expected_equal_norm_roles = sorted(
        set(optional_norm_roles) - set(expected_changed_norm_roles)
    )
    changed_roles = (
        sorted(name.removeprefix(prefix) for name in changed_keys)
        if (isinstance(changed_keys, list))
        else []
    )
    adapter_authentication = compiler.get("authentication")
    compiler_evidence = compiler.get("evidence")
    compiler_selection = compiler.get("selection")
    compiler_materialization = compiler.get("materialization")
    compiler_policy = compiler.get("semantic_policy")
    source_parent_identity_sha256 = raw.get("source_parent_identity_sha256")
    source_inventory_sha256 = raw.get("source_qkvo_inventory_sha256")
    cache_manifest_sha256 = raw.get("cache_manifest_sha256")
    cache_capability_sha256 = raw.get("cache_capability_sha256")
    cache_session_sha256 = raw.get("cache_selective_session_sha256")
    direct_receipt_sha256 = direct.get("result_sha256")
    cache_key = receipt.get("cache_key")
    bridge_artifact_sha256 = receipt.get("bridge_artifact_sha256")
    parameter_count = receipt.get("parameter_element_count_before")
    operation_audit = raw.get("operation_audit")
    selected_context_payloads = context_bank.get("selected_payloads")
    invalid = bool(
        receipt.get("schema")
        != LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_LAYER_INSTALLATION_SCHEMA
        or receipt.get("layer_index") != layer_index
        or receipt.get("status") != "installed-coupled-intact-group-qdeim-attention"
        or receipt.get("installed") is not True
        or receipt.get("install_count") != 1
        or receipt.get("recipe_qk_lane") != "intact_gqa_group_qdeim_coupled_qkvo"
        or receipt.get("recipe_value_output_lane")
        != "intact_gqa_group_qdeim_coupled_qkvo"
        or recipe.component_lanes["qk"] != "intact_gqa_group_qdeim_coupled_qkvo"
        or recipe.component_lanes["value_output"]
        != "intact_gqa_group_qdeim_coupled_qkvo"
        or receipt.get("coupled_qkvo_single_candidate") is not True
        or receipt.get("split_qk_or_vo_fallback_used") is not False
        or receipt.get("legacy_split_attention_installer_called") is not False
        or not isinstance(changed_keys, list)
        or changed_keys != sorted(changed_keys)
        or len(changed_keys) != len(set(changed_keys))
        or not required_projection_keys.issubset(changed_keys)
        or not set(changed_keys).issubset(allowed_changed_keys)
        or raw.get("changed_state_keys") != changed_keys
        or receipt.get("actual_changed_state_key_count") != len(changed_keys)
        or receipt.get("transaction_covered_state_keys") != transaction_covered_keys
        or receipt.get("transaction_covered_parameter_role_count") != 6
        or receipt.get("all_six_roles_supplied_to_atomic_transaction") is not True
        or receipt.get("six_role_transaction_coverage_proven") is not True
        or receipt.get("actual_changed_bytes_cover_four_to_six_roles") is not True
        or raw.get("changed_state_key_count") != len(changed_keys)
        or raw.get("prepared_expected_changed_state_keys") != changed_keys
        or raw.get("prepared_expected_changed_state_keys_matched") is not True
        or raw.get("transaction_covered_state_keys") != transaction_covered_keys
        or raw.get("transaction_covered_parameter_role_count") != 6
        or raw.get("all_six_roles_supplied_to_atomic_transaction") is not True
        or raw.get("six_role_transaction_coverage_proven") is not True
        or raw.get("actual_changed_bytes_cover_four_to_six_roles") is not True
        or raw.get("all_six_roles_changed") is not (len(changed_keys) == 6)
        or norm_transition_contract_valid is not True
        or changed_roles != expected_changed_roles
        or raw.get("changed_parameter_roles") != expected_changed_roles
        or raw.get("unchanged_transaction_parameter_roles") != expected_unchanged_roles
        or raw.get("transaction_covered_parameter_roles")
        != list(_ATTENTION_SIX_TENSOR_ROLES)
        or raw.get("source_norm_changed_from_target_seed_roles")
        != expected_changed_norm_roles
        or raw.get("source_norm_equal_to_target_seed_roles")
        != expected_equal_norm_roles
        or raw.get("blank_query_norm_zero_escape_proven")
        is not (
            target_seed_authentication["q_layernorm.weight"]["seed_kind"]
            == "exact-zero-legacy-blank"
            and "q_layernorm.weight" in expected_changed_norm_roles
        )
        or raw.get("blank_key_norm_zero_escape_proven")
        is not (
            target_seed_authentication["k_layernorm.weight"]["seed_kind"]
            == "exact-zero-legacy-blank"
            and "k_layernorm.weight" in expected_changed_norm_roles
        )
        or receipt.get("model_state_before_sha256")
        != raw.get("model_state_before_sha256")
        or receipt.get("model_state_after_sha256")
        != raw.get("model_state_after_sha256")
        or receipt.get("model_state_before_sha256")
        == receipt.get("model_state_after_sha256")
        or receipt.get("parameter_element_count_before")
        != receipt.get("parameter_element_count_after")
        or isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or parameter_count <= 0
        or parameter_count != raw.get("parameter_count_before")
        or parameter_count != raw.get("parameter_count_after")
        or receipt.get("parameter_count_changed") is not False
        or receipt.get("target_topology_sha256_before")
        != receipt.get("target_topology_sha256_after")
        or receipt.get("target_topology_sha256_before")
        != raw.get("target_topology_sha256_before")
        or receipt.get("topology_changed") is not False
        or receipt.get("parameter_identities_preserved") is not True
        or receipt.get("parameter_storage_preserved") is not True
        or receipt.get("gradient_state_preserved") is not True
        or receipt.get("same_selective_cache_session_as_layer_stack") is not True
        or not isinstance(transport_evidence, list)
        or transport_evidence != [adapter.get("result_sha256")]
        or raw.get("schema")
        != "neural-forge-lfm2-semantic-attention-group-qdeim-installation-v1"
        or raw.get("candidate_adapter_result_sha256") != adapter.get("result_sha256")
        or raw.get("candidate_compiler_result_sha256") != compiler.get("result_sha256")
        or raw.get("source_norm_inventory_sha256") != norm_sha256
        or source_inventory_sha256 != adapter.get("source_inventory_sha256")
        or source_inventory_sha256 != direct.get("source_inventory_sha256")
        or source_inventory_sha256 != norm.get("source_qkvo_inventory_sha256")
        or source_parent_identity_sha256 != norm.get("source_parent_identity_sha256")
        or source_parent_identity_sha256 != adapter.get("cache_parent_fingerprint")
        or cache_session_sha256 != adapter.get("cache_selective_session_sha256")
        or cache_session_sha256 != receipt.get("cache_selective_session_sha256")
        or cache_manifest_sha256 != adapter.get("cache_manifest_sha256")
        or cache_capability_sha256 != adapter.get("cache_capability_sha256")
        or capability.get("schema")
        != "neural-forge-lfm2-semantic-attention-group-qdeim-cache-capability-v1"
        or capability.get("result_sha256") != cache_capability_sha256
        or capability.get("cache_key") != cache_key
        or capability.get("cache_manifest_sha256") != cache_manifest_sha256
        or capability.get("cache_selective_session_sha256") != cache_session_sha256
        or capability.get("layer_index") != layer_index
        or capability.get("fit_batch_indices") != adapter.get("fit_batch_indices")
        or capability.get("true_post_softmax_context_complete") is not True
        or capability.get("tensor_payloads_loaded") != 0
        or capability.get("parent_forward_calls") != 0
        or capability.get("trace_recaptures") != 0
        or capability.get("training_steps") != 0
        or context_bank.get("schema")
        != "neural-forge-lfm2-semantic-attention-qdeim-context-bank-v1"
        or context_bank.get("result_sha256")
        != adapter.get("context_bank_receipt_sha256")
        or context_bank.get("route") != "captured_true"
        or context_bank.get("fit_batch_indices") != adapter.get("fit_batch_indices")
        or not isinstance(selected_context_payloads, list)
        or not selected_context_payloads
        or any(
            not isinstance(payload, Mapping)
            or payload.get("session_sha256") != cache_session_sha256
            for payload in selected_context_payloads
        )
        or context_bank.get("heldout_tensor_payload_loads") != 0
        or context_bank.get("parent_forward_calls") != 0
        or context_bank.get("trace_recaptures") != 0
        or context_bank.get("training_steps") != 0
        or raw.get("residual_chart_receipt_sha256") != chart.get("result_sha256")
        or raw.get("normalized_input_decoder_receipt_sha256")
        != decoder.get("decoder_sha256")
        or raw.get("context_bank_receipt_sha256")
        != adapter.get("context_bank_receipt_sha256")
        or adapter.get("schema")
        != "neural-forge-lfm2-semantic-attention-group-qdeim-cache-adapter-v1"
        or adapter.get("layer_index") != layer_index
        or adapter.get("cache_key") != cache_key
        or adapter.get("source_state_sha256") != raw.get("source_state_sha256")
        or adapter.get("target_template_sha256") != raw.get("target_template_sha256")
        or adapter.get("candidate_state_sha256") != raw.get("candidate_state_sha256")
        or adapter.get("residual_chart_receipt_sha256")
        != raw.get("residual_chart_receipt_sha256")
        or adapter.get("residual_chart_sha256")
        != raw.get("residual_output_chart_sha256")
        or adapter.get("normalized_input_decoder_receipt_sha256")
        != raw.get("normalized_input_decoder_receipt_sha256")
        or adapter.get("normalized_input_decoder_sha256")
        != raw.get("normalized_input_decoder_sha256")
        or adapter.get("context_bank_receipt_sha256")
        != raw.get("context_bank_receipt_sha256")
        or adapter.get("standalone_compiler_result_sha256")
        != compiler.get("result_sha256")
        or compiler.get("schema")
        != "neural-forge-lfm2-attention-intact-gqa-group-qdeim-v1"
        or not isinstance(adapter_authentication, Mapping)
        or adapter_authentication.get("source_state_sha256_before")
        != raw.get("source_state_sha256")
        or adapter_authentication.get("source_state_sha256_after")
        != raw.get("source_state_sha256")
        or adapter_authentication.get("source_state_immutable") is not True
        or adapter_authentication.get("target_template_sha256")
        != raw.get("target_template_sha256")
        or adapter_authentication.get("normalized_input_decoder_sha256")
        != raw.get("normalized_input_decoder_sha256")
        or adapter_authentication.get("residual_output_chart_sha256")
        != raw.get("residual_output_chart_sha256")
        or not isinstance(compiler_evidence, Mapping)
        or compiler_evidence.get("fit_partition_only") is not True
        or compiler_evidence.get("heldout_rows_used") is not False
        or not isinstance(compiler_selection, Mapping)
        or compiler_selection.get("selected_source_key_value_groups")
        != adapter.get("selected_source_key_value_groups")
        or compiler_selection.get("selected_source_query_heads")
        != adapter.get("selected_source_query_heads")
        or compiler_selection.get("whole_gqa_groups_only") is not True
        or compiler_selection.get("qkv_producer_averaging_performed") is not False
        or not isinstance(compiler_materialization, Mapping)
        or compiler_materialization.get("candidate_state_sha256")
        != raw.get("candidate_state_sha256")
        or compiler_materialization.get("standalone_dense_tensors") is not True
        or not isinstance(compiler_policy, Mapping)
        or compiler_policy.get("numeric_threshold_auto_accept_permitted") is not False
        or compiler_policy.get("numeric_threshold_auto_veto_permitted") is not False
        or compiler_policy.get("automatic_semantic_publication_permitted") is not False
        or compiler_policy.get("whole_response_intelligence_evaluation_is_primary")
        is not True
        or norm.get("schema")
        != ("neural-forge-lfm2-semantic-attention-group-qdeim-parent-norm-inventory-v1")
        or norm.get("layer_index") != layer_index
        or norm.get("direct_parent_norms_only") is not True
        or norm.get("head_averaging_or_refit_performed") is not False
        or set(norm_hashes) != set(optional_norm_roles)
        or direct.get("schema")
        != "neural-forge-lfm2-direct-parent-qkvo-session-load-v1"
        or direct.get("layer_index") != layer_index
        or direct.get("query_norm_weight_sha256")
        != six_tensor_bundle["q_layernorm.weight"]
        or direct.get("key_norm_weight_sha256")
        != six_tensor_bundle["k_layernorm.weight"]
        or direct.get("qk_norms_loaded_as_atomic_attention_source") is not True
        or direct.get("direct_local_safetensors") is not True
        or direct.get("parent_model_objects_constructed") != 0
        or not isinstance(cache_key, str)
        or not cache_key
        or _digest(
            source_parent_identity_sha256,
            "coupled attention source parent identity",
        )
        != source_parent_identity_sha256
        or _digest(cache_session_sha256, "coupled attention cache session")
        != cache_session_sha256
        or _digest(cache_manifest_sha256, "coupled attention cache manifest")
        != cache_manifest_sha256
        or _digest(cache_capability_sha256, "coupled attention cache capability")
        != cache_capability_sha256
        or _digest(
            bridge_artifact_sha256,
            "coupled attention tokenizer bridge artifact",
        )
        != bridge_artifact_sha256
        or _digest(
            direct.get("parent_session_open_receipt_sha256"),
            "coupled attention direct parent session",
        )
        != direct.get("parent_session_open_receipt_sha256")
        or _digest(
            direct.get("operator_norm_weight_sha256"),
            "coupled attention direct operator norm",
        )
        != direct.get("operator_norm_weight_sha256")
        or _digest(
            direct.get("compiler_lineage_sha256"),
            "coupled attention direct compiler lineage",
        )
        != direct.get("compiler_lineage_sha256")
        or _digest(
            direct_receipt_sha256,
            "coupled attention direct source receipt",
        )
        != direct_receipt_sha256
        or raw.get("complete_six_parameter_bundle_installed_transactionally")
        is not True
        or raw.get("qkvo_installed_from_one_coupled_intact_group_candidate") is not True
        or raw.get("query_and_key_norms_installed_from_direct_parent_tensors")
        is not True
        or raw.get("heldout_payload_used_for_compilation") is not False
        or receipt.get("analytic_cached_evidence_only") is not True
        or receipt.get("heldout_metrics_are_diagnostic_only") is not True
        or receipt.get("numeric_metric_used_for_candidate_selection") is not False
        or receipt.get("numeric_metric_used_for_publication_or_rollback") is not False
        or receipt.get("numeric_metrics_have_final_intelligence_authority") is not False
        or receipt.get("binary_manual_threshold_has_iteration_veto_authority")
        is not False
        or receipt.get("binary_manual_acceptance_gate_required") is not False
        or receipt.get("intelligence_retention_target_is_objective_not_numeric_cutoff")
        is not True
        or receipt.get("whole_decoded_behavior_is_primary_intelligence_evidence")
        is not True
        or receipt.get("operation_audit") != dict(_ZERO_OPERATIONS)
        or not isinstance(operation_audit, Mapping)
        or any(operation_audit.get(name) != 0 for name in _ZERO_OPERATIONS)
        or set(six_tensor_bundle) != set(_ATTENTION_SIX_TENSOR_ROLES)
        or any(
            not isinstance(value, str)
            or _digest(value, f"coupled attention {role}") != value
            for role, value in six_tensor_bundle.items()
        )
    )
    if invalid:
        raise Lfm2SemanticPostfoldError(
            f"Coupled attention Q-DEIM layer {layer_index} receipt is incomplete."
        )
    changed_projection = [
        name
        for name in changed_keys
        if name.rsplit(".", 2)[-2] in {"q_proj", "k_proj", "v_proj", "out_proj"}
    ]
    changed_norm = [
        name
        for name in changed_keys
        if name.rsplit(".", 2)[-2] in {"q_layernorm", "k_layernorm"}
    ]
    return MappingProxyType(
        {
            "layer_index": layer_index,
            "operator_receipt_sha256": receipt["result_sha256"],
            "model_state_before_sha256": receipt["model_state_before_sha256"],
            "model_state_after_sha256": receipt["model_state_after_sha256"],
            "transport_evidence_sha256": list(transport_evidence),
            "parameter_element_count": parameter_count,
            "installed": True,
            "changed_state_keys": list(changed_keys),
            "changed_projection_state_keys": changed_projection,
            "changed_norm_state_keys": changed_norm,
            "six_tensor_bundle": six_tensor_bundle,
            "six_tensor_bundle_sha256": canonical_sha256(six_tensor_bundle),
            "qk_bundle_sha256": canonical_sha256(
                {
                    "q": six_tensor_bundle["q_proj.weight"],
                    "k": six_tensor_bundle["k_proj.weight"],
                }
            ),
            "qkvo_bundle_sha256": canonical_sha256(
                {
                    "q": six_tensor_bundle["q_proj.weight"],
                    "k": six_tensor_bundle["k_proj.weight"],
                    "v": six_tensor_bundle["v_proj.weight"],
                    "o": six_tensor_bundle["out_proj.weight"],
                }
            ),
            "body_recipe": recipe.as_dict(),
            "body_recipe_sha256": recipe.recipe_sha256,
            "body_recipe_id": recipe.recipe_id,
            "recipe_qk_lane": recipe.component_lanes["qk"],
            "recipe_value_output_lane": recipe.component_lanes["value_output"],
            "coupled_attention_group_qdeim": True,
            "cache_key": cache_key,
            "cache_parent_fingerprint": source_parent_identity_sha256,
            "source_parent_identity_sha256": source_parent_identity_sha256,
            "cache_selective_session_sha256": cache_session_sha256,
            "bridge_artifact_sha256": bridge_artifact_sha256,
            "cache_manifest_sha256": cache_manifest_sha256,
            "cache_capability_sha256": cache_capability_sha256,
            "direct_attention_source_receipt_sha256": direct_receipt_sha256,
            "direct_parent_session_open_receipt_sha256": direct[
                "parent_session_open_receipt_sha256"
            ],
            "direct_compiler_lineage_sha256": direct["compiler_lineage_sha256"],
            "source_qkvo_inventory_sha256": source_inventory_sha256,
            "source_norm_inventory_sha256": norm_sha256,
            "candidate_adapter_result_sha256": adapter["result_sha256"],
            "candidate_compiler_result_sha256": compiler["result_sha256"],
            "numeric_behavior_gate_used": False,
            "binary_manual_gate_used": False,
            "retention_target_is_objective_not_numeric_cutoff": True,
        }
    )


def _authenticate_lfm2_semantic_attention_operator_receipt(
    value: Mapping[str, Any],
    *,
    layer_index: int,
    body_recipe: str | Lfm2SemanticBodyRecipe | None = None,
) -> Mapping[str, Any]:
    """Authenticate one persisted six-tensor attention transaction receipt."""

    receipt = _verify_sealed(
        value,
        seal_name="result_sha256",
        label=f"semantic attention layer {layer_index} operator receipt",
    )
    if (
        receipt.get("schema")
        == LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_LAYER_INSTALLATION_SCHEMA
    ):
        return _authenticate_lfm2_coupled_attention_group_qdeim_operator_receipt(
            receipt,
            layer_index=layer_index,
            body_recipe=body_recipe,
        )
    recipe = _authenticate_body_recipe_binding(
        receipt,
        body_recipe=body_recipe,
        component="qk",
        label=f"semantic attention layer {layer_index} operator receipt",
    )
    analytic_value = receipt.get("analytic_refold")
    bundle_value = (
        analytic_value.get("frozen_attention_six_tensor_bundle")
        if isinstance(analytic_value, Mapping)
        else None
    )
    output_writer_value = (
        analytic_value.get("output_writer")
        if isinstance(analytic_value, Mapping)
        else None
    )
    joint_value = (
        analytic_value.get("joint_value_output")
        if isinstance(analytic_value, Mapping)
        else None
    )
    value_output_gate_value = (
        analytic_value.get("value_output_candidate_gate")
        if isinstance(analytic_value, Mapping)
        else None
    )
    if not isinstance(analytic_value, Mapping):
        raise Lfm2SemanticPostfoldError(
            f"Attention layer {layer_index} omitted its analytic refold receipt."
        )
    analytic = dict(analytic_value)
    if not isinstance(bundle_value, Mapping):
        raise Lfm2SemanticPostfoldError(
            f"Attention layer {layer_index} omitted its six-tensor bundle."
        )
    bundle = dict(bundle_value)
    if (
        not isinstance(output_writer_value, Mapping)
        or not isinstance(joint_value, Mapping)
        or not isinstance(value_output_gate_value, Mapping)
    ):
        raise Lfm2SemanticPostfoldError(
            f"Attention layer {layer_index} omitted its sealed V/O decisions."
        )
    output_writer = _verify_sealed(
        output_writer_value,
        seal_name="gate_sha256",
        label=f"semantic attention layer {layer_index} output-writer gate",
    )
    joint = _verify_sealed(
        joint_value,
        seal_name="proposal_sha256",
        label=f"semantic attention layer {layer_index} joint-V/O proposal",
    )
    output_proposal_value = output_writer.get("proposal")
    if not isinstance(output_proposal_value, Mapping):
        raise Lfm2SemanticPostfoldError(
            f"Attention layer {layer_index} output writer omitted its proposal."
        )
    output_proposal = _verify_sealed(
        output_proposal_value,
        seal_name="proposal_sha256",
        label=f"semantic attention layer {layer_index} output-writer proposal",
    )
    value_output_gate = _verify_sealed(
        value_output_gate_value,
        seal_name="gate_sha256",
        label=f"semantic attention layer {layer_index} value/output candidate gate",
    )
    _authenticate_body_recipe_binding(
        value_output_gate,
        body_recipe=recipe,
        component="value_output",
        label=f"semantic attention layer {layer_index} value/output candidate gate",
    )
    if (
        analytic.get("body_recipe") != recipe.as_dict()
        or analytic.get("body_recipe_sha256") != recipe.recipe_sha256
        or analytic.get("body_recipe_id") != recipe.recipe_id
        or analytic.get("recipe_qk_lane") != recipe.component_lanes["qk"]
        or analytic.get("recipe_value_output_lane")
        != recipe.component_lanes["value_output"]
    ):
        raise Lfm2SemanticPostfoldError(
            f"Attention layer {layer_index} analytic Q/K/V/O recipe binding changed."
        )
    changed_keys = receipt.get("changed_state_keys")
    installed = receipt.get("installed")
    before_sha256 = receipt.get("model_state_before_sha256")
    after_sha256 = receipt.get("model_state_after_sha256")
    transport_evidence = receipt.get("transport_evidence_sha256")
    allowed_state_keys = {
        f"model.layers.{layer_index}.self_attn.{role}"
        for role in _ATTENTION_SIX_TENSOR_ROLES
    }
    expected_qk_sha = canonical_sha256(
        {
            "q": bundle.get("q_proj.weight"),
            "k": bundle.get("k_proj.weight"),
        }
    )
    expected_qkvo_sha = canonical_sha256(
        {
            "q": bundle.get("q_proj.weight"),
            "k": bundle.get("k_proj.weight"),
            "v": bundle.get("v_proj.weight"),
            "o": bundle.get("out_proj.weight"),
        }
    )
    candidate_reports = value_output_gate.get("candidate_reports")
    validation_gate = value_output_gate.get("validation")
    gate_schema = value_output_gate.get("schema")
    published_candidate = value_output_gate.get("published_candidate")
    if not isinstance(candidate_reports, Mapping) or not isinstance(
        validation_gate,
        Mapping,
    ):
        raise Lfm2SemanticPostfoldError(
            f"Attention layer {layer_index} omitted its V/O diagnostic partitions."
        )
    normal_gate = gate_schema == "neural-forge-lfm2-semantic-vo-candidate-gate-v1"
    gate_invalid = bool(
        not normal_gate
        or value_output_gate.get("published_value_weight_sha256")
        != bundle.get("v_proj.weight")
        or value_output_gate.get("published_output_weight_sha256")
        != bundle.get("out_proj.weight")
        or value_output_gate.get("numeric_metric_used_for_publication_or_rollback")
        is not False
        or value_output_gate.get("fit_only_analytic_solves") is not True
        or type(value_output_gate.get("fixed_value_weight_preserved")) is not bool
        or value_output_gate.get("selected_candidate_frozen_before_validation")
        is not True
        or value_output_gate.get("validation_triggered_refit_or_reselection")
        is not False
        or value_output_gate.get("alternate_candidate_selected_after_validation")
        is not False
        or value_output_gate.get("operation_audit")
        != dict(_ATTENTION_REFOLD_ZERO_OPERATIONS)
        or validation_gate.get("diagnostic_only") is not True
    )
    if normal_gate:
        expected_candidates = {
            "incumbent",
            "fixed_v_output_only",
            "joint_value_output",
        }
        expected_effect = {
            "incumbent": False,
            "fixed_v_output_only": output_proposal.get("changed"),
            "joint_value_output": joint.get("changed"),
        }
        eligible_count = sum(
            int(
                isinstance(candidate_reports.get(name), Mapping)
                and candidate_reports[name].get("eligible") is True
            )
            for name in ("fixed_v_output_only", "joint_value_output")
        )
        eligible_names = [
            name
            for name in ("fixed_v_output_only", "joint_value_output")
            if isinstance(candidate_reports.get(name), Mapping)
            and candidate_reports[name].get("eligible") is True
        ]

        def reported_fit_ordering_key(name: str) -> tuple[Any, ...] | None:
            report = candidate_reports.get(name)
            partitions = (
                report.get("partitions") if isinstance(report, Mapping) else None
            )
            fit = partitions.get("fit") if isinstance(partitions, Mapping) else None
            metrics = fit.get("candidate") if isinstance(fit, Mapping) else None
            proposal_sha256 = (
                report.get("proposal_sha256") if isinstance(report, Mapping) else None
            )
            fields = (
                "endpoint_relative_rmse",
                "endpoint_rmse",
                "endpoint_maximum_absolute_error",
                "endpoint_cosine_similarity_mean",
                "endpoint_cosine_similarity_worst",
            )
            if (
                not isinstance(metrics, Mapping)
                or not isinstance(proposal_sha256, str)
                or len(proposal_sha256) != 64
                or any(
                    isinstance(metrics.get(field), bool)
                    or not isinstance(metrics.get(field), (int, float))
                    or not math.isfinite(float(metrics[field]))
                    for field in fields
                )
            ):
                return None
            return (
                float(metrics[fields[0]]),
                float(metrics[fields[1]]),
                float(metrics[fields[2]]),
                -float(metrics[fields[3]]),
                -float(metrics[fields[4]]),
                proposal_sha256,
            )

        fit_ordering = {
            name: reported_fit_ordering_key(name) for name in eligible_names
        }
        fit_ordering_invalid = any(value is None for value in fit_ordering.values())
        fit_selected_candidate = (
            None
            if not eligible_names or fit_ordering_invalid
            else min(eligible_names, key=lambda name: fit_ordering[name])
        )
        selected_report = candidate_reports.get(published_candidate)
        expected_frozen_candidate_sha256 = (
            canonical_sha256(
                {
                    "candidate_name": published_candidate,
                    "value_weight_sha256": bundle.get("v_proj.weight"),
                    "output_weight_sha256": bundle.get("out_proj.weight"),
                    "proposal_sha256": selected_report.get("proposal_sha256"),
                }
            )
            if isinstance(selected_report, Mapping)
            else None
        )

        def invalid_candidate_report(name: str, report: Any) -> bool:
            if not isinstance(report, Mapping) or not isinstance(
                report.get("partitions"),
                Mapping,
            ):
                return True
            partitions = report["partitions"]
            structural = all(
                report.get(field) is True
                for field in ("finite", "dense_strided", "bfloat16", "exact_shape")
            )
            effectful = expected_effect[name]
            eligible = bool(structural and effectful)
            fit = partitions.get("fit")
            heldout = [
                partitions.get(name)
                for name in ("fit_check", "selection", "validation")
            ]
            return bool(
                set(partitions) != {"fit", "fit_check", "selection", "validation"}
                or type(effectful) is not bool
                or report.get("effectful") is not effectful
                or report.get("proposal_changed") is not effectful
                or report.get("eligible") is not eligible
                or report.get("eligible_after_fit_structural_check") is not eligible
                or report.get("eligible_after_fit_fit_check_selection") is not eligible
                or report.get("structurally_publishable") is not eligible
                or report.get("eligibility_basis")
                != _ATTENTION_CANDIDATE_ELIGIBILITY_BASIS
                or report.get("numeric_metric_used_for_eligibility") is not False
                or not isinstance(fit, Mapping)
                or fit.get("diagnostic_only")
                is not (not recipe.fit_metrics_order_candidates)
                or fit.get("used_for_candidate_ordering")
                is not (eligible and recipe.fit_metrics_order_candidates)
                or fit.get("used_for_structural_eligibility") is not False
                or fit.get("used_for_publication_threshold") is not False
                or fit.get("used_for_publication") is not False
                or any(
                    not isinstance(partition, Mapping)
                    or partition.get("diagnostic_only") is not True
                    or partition.get("used_for_candidate_ordering") is not False
                    or partition.get("used_for_structural_eligibility") is not False
                    or partition.get("used_for_publication_threshold") is not False
                    or partition.get("used_for_publication") is not False
                    for partition in heldout
                )
            )

        gate_invalid = bool(
            gate_invalid
            or set(candidate_reports) != expected_candidates
            or published_candidate not in expected_candidates
            or value_output_gate.get("selected_candidate") != published_candidate
            or value_output_gate.get("frozen_candidate_sha256")
            != expected_frozen_candidate_sha256
            or not isinstance(candidate_reports.get("fixed_v_output_only"), Mapping)
            or candidate_reports["fixed_v_output_only"].get("proposal_sha256")
            != output_proposal.get("proposal_sha256")
            or not isinstance(candidate_reports.get("joint_value_output"), Mapping)
            or candidate_reports["joint_value_output"].get("proposal_sha256")
            != joint.get("proposal_sha256")
            or value_output_gate.get("structurally_eligible_candidate_count")
            != eligible_count
            or value_output_gate.get("candidate_eligibility_basis")
            != _ATTENTION_CANDIDATE_ELIGIBILITY_BASIS
            or value_output_gate.get("structural_safety_is_only_candidate_veto")
            is not True
            or value_output_gate.get("fit_metrics_used_for_candidate_ordering")
            is not recipe.fit_metrics_order_candidates
            or value_output_gate.get("fit_ordering_metrics")
            != [
                "endpoint_relative_rmse",
                "endpoint_rmse",
                "endpoint_maximum_absolute_error",
                "negative_endpoint_cosine_similarity_mean",
                "negative_endpoint_cosine_similarity_worst",
            ]
            or value_output_gate.get("exact_fit_tie_breaker") != "proposal_sha256"
            or value_output_gate.get("fit_metrics_used_for_structural_eligibility")
            is not False
            or value_output_gate.get(
                "fit_metric_threshold_used_for_candidate_selection"
            )
            is not False
            or value_output_gate.get("numeric_metric_used_for_candidate_selection")
            is not (recipe.fit_metrics_order_candidates and eligible_count > 1)
            or value_output_gate.get("incumbent_noop_fallback_used")
            is not (eligible_count == 0)
            or value_output_gate.get("incumbent_noop_fallback_reason")
            != (
                "all-analytic-proposals-structurally-invalid-or-identical"
                if eligible_count == 0
                else None
            )
            or value_output_gate.get(
                "selected_candidate_frozen_before_heldout_diagnostics"
            )
            is not True
            or (published_candidate == "incumbent") is not (eligible_count == 0)
            or (recipe.fit_metrics_order_candidates and fit_ordering_invalid)
            or (
                recipe.component_lanes["value_output"]
                == "fit_ordered_structural_frontier"
                and eligible_count > 0
                and published_candidate != fit_selected_candidate
            )
            or (
                recipe.component_lanes["value_output"]
                == "joint_then_fixed_structural_priority"
                and published_candidate
                != (
                    "joint_value_output"
                    if "joint_value_output" in eligible_names
                    else (
                        "fixed_v_output_only"
                        if "fixed_v_output_only" in eligible_names
                        else "incumbent"
                    )
                )
            )
            or (
                recipe.component_lanes["value_output"]
                == "fixed_writer_then_joint_residual_priority"
                and published_candidate
                != (
                    "fixed_v_output_only"
                    if "fixed_v_output_only" in eligible_names
                    else (
                        "joint_value_output"
                        if "joint_value_output" in eligible_names
                        else "incumbent"
                    )
                )
            )
            or not is_supported_lfm2_semantic_component_equivalence_inventory(
                "value_output",
                recipe.component_lanes["value_output"],
                value_output_gate.get(
                    "component_selection_policy_equivalent_recipe_ids"
                ),
                value_output_gate.get(
                    "component_selection_policy_equivalent_to_other_recipe"
                ),
                materialized_recipe_id=recipe.recipe_id,
            )
            or (
                published_candidate != "incumbent"
                and candidate_reports[published_candidate].get("eligible") is not True
            )
            or value_output_gate.get("heldout_metrics_are_diagnostic_only") is not True
            or value_output_gate.get("joint_value_output_publication_allowed")
            is not True
            or value_output_gate.get("selected_correction_rejected_on_validation")
            is not False
            or value_output_gate.get("validation_publication_veto_applied") is not False
            or value_output_gate.get("rollback_authorized") is not False
            or validation_gate.get("used_for_publication") is not False
            or output_writer.get("accepted")
            is not (published_candidate == "fixed_v_output_only")
            or any(
                invalid_candidate_report(name, candidate_reports.get(name))
                for name in expected_candidates
            )
        )
    if gate_invalid:
        raise Lfm2SemanticPostfoldError(
            f"Attention layer {layer_index} V/O gate permits behavioral publication."
        )
    structurally_eligible_count = receipt.get("structurally_eligible_candidate_count")
    expected_status = (
        "installed-frozen-semantic-attention-candidate"
        if installed
        else _ATTENTION_STRUCTURAL_NOOP_STATUS
    )
    expected_selected_candidate = (
        "frozen-semantic-attention-candidate" if installed else "incumbent-noop"
    )
    joint_published = published_candidate == "joint_value_output"
    invalid = bool(
        receipt.get("schema") != LFM2_SEMANTIC_ATTENTION_INSTALLATION_SCHEMA
        or receipt.get("layer_index") != layer_index
        or type(installed) is not bool
        or receipt.get("status") != expected_status
        or type(structurally_eligible_count) is not int
        or structurally_eligible_count < 0
        or installed is not (structurally_eligible_count > 0)
        or receipt.get("selected_candidate") != expected_selected_candidate
        or receipt.get("selected_candidate_state_identical_to_incumbent") is installed
        or receipt.get("incumbent_noop_fallback_used") is installed
        or receipt.get("incumbent_noop_fallback_reason")
        != (
            None
            if installed
            else "all-analytic-proposals-structurally-invalid-or-identical"
        )
        or receipt.get("candidate_eligibility_basis")
        != _ATTENTION_CANDIDATE_ELIGIBILITY_BASIS
        or receipt.get("structural_safety_is_only_candidate_veto") is not True
        or receipt.get("fit_metrics_used_for_candidate_ordering")
        is not recipe.fit_metrics_order_candidates
        or receipt.get("fit_metrics_used_for_structural_eligibility") is not False
        or receipt.get("incumbent_selected_by_heldout_gates") is not False
        or receipt.get("validation_publication_veto_applied") is not False
        or receipt.get("rollback_authorized") is not False
        or receipt.get("numeric_metrics_have_final_intelligence_authority") is not False
        or receipt.get("validation_rejected_candidate") is not None
        or not isinstance(before_sha256, str)
        or not isinstance(after_sha256, str)
        or _digest(before_sha256, "attention model state before") != before_sha256
        or _digest(after_sha256, "attention model state after") != after_sha256
        or (before_sha256 != after_sha256) is not installed
        or not isinstance(transport_evidence, list)
        or not transport_evidence
        or any(
            not isinstance(value, str)
            or _digest(value, "attention transport evidence") != value
            for value in transport_evidence
        )
        or receipt.get("install_count") != int(installed)
        or not isinstance(changed_keys, list)
        or any(not isinstance(name, str) for name in changed_keys)
        or changed_keys != sorted(set(changed_keys))
        or bool(changed_keys) != installed
        or any(name not in allowed_state_keys for name in changed_keys)
        or receipt.get("parameter_element_count_before")
        != receipt.get("parameter_element_count_after")
        or receipt.get("parameter_count_changed") is not False
        or receipt.get("target_topology_sha256_before")
        != receipt.get("target_topology_sha256_after")
        or receipt.get("topology_changed") is not False
        or receipt.get("parameter_identities_preserved") is not True
        or receipt.get("parameter_storage_preserved") is not True
        or receipt.get("gradient_state_preserved") is not True
        or receipt.get("exact_noop_state_proven") is installed
        or receipt.get("heldout_metrics_are_diagnostic_only") is not True
        or receipt.get("numeric_metric_used_for_publication_or_rollback") is not False
        or receipt.get("candidate_frozen_before_validation") is not True
        or receipt.get("validation_triggered_refit") is not False
        or receipt.get("analytic_cached_evidence_only") is not True
        or receipt.get("operation_audit") != dict(_ZERO_OPERATIONS)
        or analytic.get("schema") != "neural-forge-lfm2-semantic-attention-refold-v1"
        or set(bundle) != set(_ATTENTION_SIX_TENSOR_ROLES)
        or any(
            not isinstance(bundle.get(role), str)
            or len(str(bundle.get(role))) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(bundle.get(role))
            )
            for role in _ATTENTION_SIX_TENSOR_ROLES
        )
        or analytic.get("frozen_attention_six_tensor_bundle_roles")
        != list(_ATTENTION_SIX_TENSOR_ROLES)
        or analytic.get("frozen_attention_six_tensor_bundle_sha256")
        != canonical_sha256(bundle)
        or analytic.get("frozen_qk_sha256") != expected_qk_sha
        or analytic.get("frozen_qkvo_sha256") != expected_qkvo_sha
        or type(analytic.get("qk_compiler_accepted")) is not bool
        or type(analytic.get("qk_compiler_changed")) is not bool
        or not isinstance(analytic.get("fixed_value_source"), str)
        or not analytic.get("fixed_value_source")
        or not isinstance(analytic.get("output_writer_source"), str)
        or not analytic.get("output_writer_source")
        or analytic.get("heldout_metrics_are_diagnostic_only") is not True
        or analytic.get("numeric_metric_used_for_publication_or_rollback") is not False
        or analytic.get("joint_value_output_candidate_diagnostic_only") is not False
        or analytic.get("joint_value_output_publication_allowed") is not True
        or analytic.get("joint_value_output_candidate_published") is not joint_published
        or analytic.get("complete_qkvo_bundle_installed_transactionally")
        is not installed
        or analytic.get("complete_qkvo_bundle_transaction_verified") is not True
        or analytic.get("authenticated_incumbent_qkvo_retained_without_installation")
        is installed
        or analytic.get("projection_parameter_copy_calls") != len(changed_keys)
        or analytic.get("all_semantically_exact_endpoint_rows_used_for_attention_fit")
        is not True
        or analytic.get("terminal_endpoint_only_supervision") is not False
        or analytic.get(
            "analytic_fp64_target_construction_distinct_from_bfloat16_runtime_replay"
        )
        is not True
        or analytic.get("runtime_replay_numeric_contract")
        != "bfloat16-linears-fp32-qk-softmax-bfloat16-probability-context"
        or analytic.get("cross_head_dimension_bridge_deferred") is not True
        or analytic.get("refold_operation_audit")
        != dict(_ATTENTION_REFOLD_ZERO_OPERATIONS)
        or output_writer.get("heldout_metrics_are_diagnostic_only") is not True
        or output_writer.get("numeric_metric_used_for_publication") is not False
        or type(output_writer.get("accepted")) is not bool
        or output_proposal.get("value_weight_fixed") is not True
        or output_proposal.get("fit_improved_is_diagnostic_only") is not True
        or output_proposal.get("numeric_metric_used_for_step_size_selection")
        is not False
        or output_proposal.get("numeric_metric_used_for_publication") is not False
        or (
            published_candidate == "fixed_v_output_only"
            and output_proposal.get("output_weight_sha256")
            != bundle.get("out_proj.weight")
        )
        or (
            joint_published
            and (
                joint.get("value_weight_sha256") != bundle.get("v_proj.weight")
                or joint.get("output_weight_sha256") != bundle.get("out_proj.weight")
            )
        )
        or joint.get("diagnostic_only") is not False
        or joint.get("publication_allowed") is not True
        or joint.get("numeric_metric_used_for_publication") is not False
        or type(analytic.get("fresh_output_writer_refit_performed")) is not bool
        or output_writer.get("accepted")
        is not analytic.get("fresh_output_writer_refit_performed")
    )
    if invalid:
        raise Lfm2SemanticPostfoldError(
            f"Attention layer {layer_index} persisted transaction is incomplete."
        )
    changed_projection = [
        name
        for name in changed_keys
        if name.rsplit(".", 2)[-2] in {"q_proj", "k_proj", "v_proj", "out_proj"}
    ]
    changed_norm = [
        name
        for name in changed_keys
        if name.rsplit(".", 2)[-2] in {"q_layernorm", "k_layernorm"}
    ]
    return MappingProxyType(
        {
            "layer_index": layer_index,
            "operator_receipt_sha256": receipt["result_sha256"],
            "model_state_before_sha256": before_sha256,
            "model_state_after_sha256": after_sha256,
            "transport_evidence_sha256": list(transport_evidence),
            "parameter_element_count": receipt["parameter_element_count_before"],
            "installed": installed,
            "changed_state_keys": list(changed_keys),
            "changed_projection_state_keys": changed_projection,
            "changed_norm_state_keys": changed_norm,
            "six_tensor_bundle": bundle,
            "six_tensor_bundle_sha256": analytic[
                "frozen_attention_six_tensor_bundle_sha256"
            ],
            "qk_bundle_sha256": analytic["frozen_qk_sha256"],
            "qkvo_bundle_sha256": analytic["frozen_qkvo_sha256"],
            "output_writer_gate_sha256": output_writer["gate_sha256"],
            "joint_value_output_proposal_sha256": joint["proposal_sha256"],
            "value_output_candidate_gate_sha256": value_output_gate["gate_sha256"],
            "body_recipe": recipe.as_dict(),
            "body_recipe_sha256": recipe.recipe_sha256,
            "body_recipe_id": recipe.recipe_id,
            "recipe_qk_lane": recipe.component_lanes["qk"],
            "recipe_value_output_lane": recipe.component_lanes["value_output"],
            "recipe_bound_publication_with_diagnostic_heldout_authenticated": True,
            "fit_only_publication_with_diagnostic_heldout_authenticated": (
                recipe.fit_metrics_order_candidates
            ),
            "numeric_behavior_gate_used": False,
        }
    )


def _authenticate_lfm2_coupled_attention_stack_root_binding(
    stack: Mapping[str, Any],
    operator: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Bind one authenticated coupled operator to its persisted stack roots."""

    layer_index = operator.get("layer_index")
    direct_by_layer = stack.get("direct_attention_source_receipt_sha256")
    attention_layer_indices = stack.get("attention_layer_indices")
    cache_key = stack.get("cache_key")
    cache_session_sha256 = stack.get("cache_selective_session_sha256")
    cache_session_value = stack.get("cache_selective_session")
    if not isinstance(cache_session_value, Mapping):
        raise Lfm2SemanticPostfoldError(
            "Coupled attention layer stack omitted its selective cache session."
        )
    cache_session = _verify_sealed(
        cache_session_value,
        seal_name="session_sha256",
        label="coupled attention layer-stack selective cache session",
    )
    bridge_artifact_sha256 = stack.get("bridge_artifact_sha256")
    cache_parent_fingerprint = stack.get("cache_parent_fingerprint")
    parent_checkpoint_session_sha256 = stack.get("parent_checkpoint_session_sha256")
    parent_checkpoint_session_value = stack.get("parent_checkpoint_session")
    if not isinstance(parent_checkpoint_session_value, Mapping):
        raise Lfm2SemanticPostfoldError(
            "Coupled attention layer stack omitted its parent checkpoint session."
        )
    parent_checkpoint_session = _verify_sealed(
        parent_checkpoint_session_value,
        seal_name="session_sha256",
        label="coupled attention layer-stack parent checkpoint session",
    )
    parent_identity_value = parent_checkpoint_session.get(
        "parent_identity_reconstruction"
    )
    if not isinstance(parent_identity_value, Mapping):
        raise Lfm2SemanticPostfoldError(
            "Coupled attention parent session omitted exact identity reconstruction."
        )
    parent_identity = _verify_sealed(
        parent_identity_value,
        seal_name="result_sha256",
        label="coupled attention exact parent identity reconstruction",
    )
    cache_contract = cache_session.get("authentication_contract")
    cache_batches = cache_session.get("ordered_authenticated_batches")
    expected_cache_contract_keys = {
        "committed_local_cache_only",
        "manifest_completion_and_layout_verified",
        "trace_file_sha256_passes_per_session",
        "record_file_sha256_passes_per_session",
        "record_self_hash_verified",
        "safetensors_metadata_verified",
        "complete_tensor_key_inventory_verified",
        "selected_tensor_descriptors_verified_per_load",
        "stable_file_identity_checked_before_and_after_load",
        "authenticated_trace_guard_held_open",
        "safetensors_mapping_held_open",
        "safetensors_mapping_opened_per_selection_and_closed",
        "unselected_tensor_payloads_loaded",
        "parent_forward_calls",
        "training_steps",
        "toctou_boundary",
    }
    expected_cache_session_keys = {
        "schema",
        "cache_key",
        "manifest_sha256",
        "completion_sha256",
        "batch_count",
        "ordered_authenticated_batches",
        "authentication_contract",
        "session_sha256",
    }
    expected_cache_batch_keys = {
        "batch_index",
        "input_sha256",
        "trace_file_sha256",
        "record_file_sha256",
        "record_sha256",
        "tensor_count",
        "logical_bytes",
    }
    cache_session_complete = bool(
        set(cache_session) == expected_cache_session_keys
        and _is_digest(cache_session.get("manifest_sha256"))
        and _is_digest(cache_session.get("completion_sha256"))
        and isinstance(cache_batches, list)
        and cache_batches
        and cache_session.get("batch_count") == len(cache_batches)
        and all(
            isinstance(batch, Mapping)
            and set(batch) == expected_cache_batch_keys
            and batch.get("batch_index") == ordinal
            and all(
                _is_digest(batch.get(name))
                for name in (
                    "input_sha256",
                    "trace_file_sha256",
                    "record_file_sha256",
                    "record_sha256",
                )
            )
            and isinstance(batch.get("tensor_count"), int)
            and not isinstance(batch.get("tensor_count"), bool)
            and batch.get("tensor_count") > 0
            and isinstance(batch.get("logical_bytes"), int)
            and not isinstance(batch.get("logical_bytes"), bool)
            and batch.get("logical_bytes") > 0
            for ordinal, batch in enumerate(cache_batches)
        )
        and isinstance(cache_contract, Mapping)
        and set(cache_contract) == expected_cache_contract_keys
        and all(
            cache_contract.get(name) is True
            for name in (
                "committed_local_cache_only",
                "manifest_completion_and_layout_verified",
                "record_self_hash_verified",
                "safetensors_metadata_verified",
                "complete_tensor_key_inventory_verified",
                "selected_tensor_descriptors_verified_per_load",
                "stable_file_identity_checked_before_and_after_load",
                "authenticated_trace_guard_held_open",
                "safetensors_mapping_opened_per_selection_and_closed",
            )
        )
        and cache_contract.get("trace_file_sha256_passes_per_session") == 1
        and cache_contract.get("record_file_sha256_passes_per_session") == 1
        and cache_contract.get("safetensors_mapping_held_open") is False
        and cache_contract.get("unselected_tensor_payloads_loaded") is False
        and cache_contract.get("parent_forward_calls") == 0
        and cache_contract.get("training_steps") == 0
        and isinstance(cache_contract.get("toctou_boundary"), str)
        and bool(cache_contract.get("toctou_boundary"))
    )
    expected_parent_session_keys = {
        "schema",
        "parent_model_identity_sha256",
        "checkpoint_manifest_sha256",
        "open_receipt_sha256",
        "config_file_sha256",
        "normalized_config_sha256",
        "norm_epsilon",
        "loaded_tensor_sha256",
        "parent_identity_reconstruction_sha256",
        "parent_identity_reconstruction",
        "exact_parent_identity_authenticated",
        "checkpoint_files_rehashed_after_materialization",
        "shards_hashed_once",
        "shards_opened_once",
        "parent_model_objects_constructed",
        "parent_forward_calls",
        "network_access_performed",
        "session_sha256",
    }
    expected_parent_identity_keys = {
        "schema",
        "cache_parent_identity_sha256",
        "cache_parent_record",
        "checkpoint_manifest_sha256",
        "config_file_sha256",
        "normalized_config_sha256",
        "checkpoint_tensor_inventory_sha256",
        "checkpoint_tensor_count",
        "checkpoint_tensor_records",
        "reconstructed_state_dict_sha256",
        "reconstructed_state_tensor_count",
        "runtime_buffer_records",
        "reconstructed_runtime_buffer_sha256",
        "reconstructed_runtime_buffer_count",
        "reconstructed_parent_identity",
        "reconstructed_parent_identity_sha256",
        "exact_parent_identity_authenticated",
        "parent_model_objects_constructed",
        "parent_forward_calls",
        "network_access_performed",
        "result_sha256",
    }
    loaded_tensor_sha256 = parent_checkpoint_session.get("loaded_tensor_sha256")
    checkpoint_tensor_records = parent_identity.get("checkpoint_tensor_records")
    runtime_buffer_records = parent_identity.get("runtime_buffer_records")
    reconstructed_parent = parent_identity.get("reconstructed_parent_identity")
    parent_session_complete = bool(
        set(parent_checkpoint_session) == expected_parent_session_keys
        and set(parent_identity) == expected_parent_identity_keys
        and all(
            _is_digest(parent_checkpoint_session.get(name))
            for name in (
                "parent_model_identity_sha256",
                "checkpoint_manifest_sha256",
                "open_receipt_sha256",
                "config_file_sha256",
                "normalized_config_sha256",
                "parent_identity_reconstruction_sha256",
            )
        )
        and isinstance(parent_checkpoint_session.get("norm_epsilon"), (int, float))
        and not isinstance(parent_checkpoint_session.get("norm_epsilon"), bool)
        and math.isfinite(float(parent_checkpoint_session["norm_epsilon"]))
        and float(parent_checkpoint_session["norm_epsilon"]) > 0.0
        and isinstance(loaded_tensor_sha256, Mapping)
        and loaded_tensor_sha256
        and all(
            isinstance(name, str) and bool(name) and _is_digest(digest)
            for name, digest in loaded_tensor_sha256.items()
        )
        and parent_checkpoint_session.get("parent_identity_reconstruction_sha256")
        == parent_identity.get("result_sha256")
        and parent_identity.get("schema")
        == "neural-forge-direct-parent-exact-identity-reconstruction-v1"
        and parent_identity.get("cache_parent_identity_sha256")
        == parent_checkpoint_session.get("parent_model_identity_sha256")
        and parent_identity.get("checkpoint_manifest_sha256")
        == parent_checkpoint_session.get("checkpoint_manifest_sha256")
        and parent_identity.get("config_file_sha256")
        == parent_checkpoint_session.get("config_file_sha256")
        and parent_identity.get("normalized_config_sha256")
        == parent_checkpoint_session.get("normalized_config_sha256")
        and all(
            _is_digest(parent_identity.get(name))
            for name in (
                "checkpoint_tensor_inventory_sha256",
                "reconstructed_state_dict_sha256",
                "reconstructed_runtime_buffer_sha256",
                "reconstructed_parent_identity_sha256",
            )
        )
        and isinstance(checkpoint_tensor_records, list)
        and parent_identity.get("checkpoint_tensor_count")
        == len(checkpoint_tensor_records)
        and isinstance(runtime_buffer_records, list)
        and parent_identity.get("reconstructed_runtime_buffer_count")
        == len(runtime_buffer_records)
        and isinstance(reconstructed_parent, Mapping)
        and canonical_sha256(reconstructed_parent)
        == parent_identity.get("reconstructed_parent_identity_sha256")
        and parent_identity.get("reconstructed_parent_identity_sha256")
        == parent_checkpoint_session.get("parent_model_identity_sha256")
        and parent_identity.get("exact_parent_identity_authenticated") is True
        and parent_identity.get("parent_model_objects_constructed") == 0
        and parent_identity.get("parent_forward_calls") == 0
        and parent_identity.get("network_access_performed") is False
    )
    expected_direct_sha256 = (
        direct_by_layer.get(str(layer_index))
        if isinstance(direct_by_layer, Mapping)
        else None
    )
    if (
        operator.get("coupled_attention_group_qdeim") is not True
        or isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or not isinstance(attention_layer_indices, list)
        or layer_index not in attention_layer_indices
        or not isinstance(cache_key, str)
        or not cache_key
        or _digest(cache_session_sha256, "layer-stack selective cache session")
        != cache_session_sha256
        or cache_session.get("schema")
        != "neural-forge-lfm2-circuit-trace-selective-session-v1"
        or cache_session.get("session_sha256") != cache_session_sha256
        or cache_session.get("cache_key") != cache_key
        or cache_session.get("manifest_sha256") != operator.get("cache_manifest_sha256")
        or cache_session_complete is not True
        or isinstance(cache_session.get("batch_count"), bool)
        or not isinstance(cache_session.get("batch_count"), int)
        or cache_session.get("batch_count") <= 0
        or not isinstance(cache_session.get("authentication_contract"), Mapping)
        or cache_session["authentication_contract"].get("parent_forward_calls") != 0
        or cache_session["authentication_contract"].get("training_steps") != 0
        or _digest(bridge_artifact_sha256, "layer-stack tokenizer bridge")
        != bridge_artifact_sha256
        or _digest(cache_parent_fingerprint, "layer-stack cache parent")
        != cache_parent_fingerprint
        or _digest(
            parent_checkpoint_session_sha256,
            "layer-stack parent checkpoint session",
        )
        != parent_checkpoint_session_sha256
        or parent_checkpoint_session.get("schema")
        != "neural-forge-direct-parent-session-v2"
        or parent_checkpoint_session.get("session_sha256")
        != parent_checkpoint_session_sha256
        or parent_checkpoint_session.get("parent_model_identity_sha256")
        != cache_parent_fingerprint
        or parent_session_complete is not True
        or parent_checkpoint_session.get(
            "checkpoint_files_rehashed_after_materialization"
        )
        is not True
        or parent_checkpoint_session.get("shards_hashed_once") is not True
        or parent_checkpoint_session.get("shards_opened_once") is not True
        or parent_checkpoint_session.get("exact_parent_identity_authenticated")
        is not True
        or parent_checkpoint_session.get("parent_model_objects_constructed") != 0
        or parent_checkpoint_session.get("parent_forward_calls") != 0
        or parent_checkpoint_session.get("network_access_performed") is not False
        or _digest(expected_direct_sha256, "layer-stack direct attention source")
        != expected_direct_sha256
        or operator.get("cache_key") != cache_key
        or operator.get("cache_selective_session_sha256") != cache_session_sha256
        or operator.get("bridge_artifact_sha256") != bridge_artifact_sha256
        or operator.get("cache_parent_fingerprint") != cache_parent_fingerprint
        or operator.get("source_parent_identity_sha256") != cache_parent_fingerprint
        or operator.get("direct_parent_session_open_receipt_sha256")
        != parent_checkpoint_session.get("open_receipt_sha256")
        or operator.get("direct_attention_source_receipt_sha256")
        != expected_direct_sha256
    ):
        raise Lfm2SemanticPostfoldError(
            f"Coupled attention Q-DEIM layer {layer_index} is not bound to "
            "the persisted layer-stack cache, bridge, parent, and direct source roots."
        )
    return MappingProxyType(
        {
            "layer_index": layer_index,
            "cache_key": cache_key,
            "cache_selective_session_sha256": cache_session_sha256,
            "cache_manifest_sha256": cache_session["manifest_sha256"],
            "bridge_artifact_sha256": bridge_artifact_sha256,
            "cache_parent_fingerprint": cache_parent_fingerprint,
            "parent_checkpoint_session_sha256": (parent_checkpoint_session_sha256),
            "parent_checkpoint_open_receipt_sha256": (
                parent_checkpoint_session["open_receipt_sha256"]
            ),
            "direct_attention_source_receipt_sha256": expected_direct_sha256,
            "cache_capability_sha256": operator["cache_capability_sha256"],
            "candidate_adapter_result_sha256": operator[
                "candidate_adapter_result_sha256"
            ],
            "candidate_compiler_result_sha256": operator[
                "candidate_compiler_result_sha256"
            ],
            "direct_compiler_lineage_sha256": operator[
                "direct_compiler_lineage_sha256"
            ],
        }
    )


def _authenticated_layer_stack_operator_topology(
    stack: Mapping[str, Any],
    layer_values: Any,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    """Recover the sealed dynamic operator topology without a depth constant."""

    layer_count = stack.get("layer_count")
    recorded_kinds = stack.get("operator_layer_kinds")
    recorded_attention = stack.get("attention_layer_indices")
    recorded_short_conv = stack.get("short_conv_layer_indices")
    terminal_layer_index = stack.get("terminal_layer_index")
    if (
        isinstance(layer_count, bool)
        or not isinstance(layer_count, int)
        or layer_count <= 0
        or not isinstance(layer_values, list)
        or len(layer_values) != layer_count
        or not isinstance(recorded_kinds, list)
        or len(recorded_kinds) != layer_count
        or any(kind not in {"full_attention", "short_conv"} for kind in recorded_kinds)
        or not isinstance(recorded_attention, list)
        or not isinstance(recorded_short_conv, list)
        or terminal_layer_index != layer_count - 1
        or stack.get(
            "dynamic_layer_count_derived_from_authenticated_cache_and_child_topology"
        )
        is not True
    ):
        raise Lfm2SemanticPostfoldError(
            "Semantic layer stack has no authenticated dynamic operator topology."
        )
    if any(not isinstance(value, Mapping) for value in layer_values):
        raise Lfm2SemanticPostfoldError(
            "Semantic layer stack contains a non-mapping layer transaction."
        )
    kinds = tuple(str(kind) for kind in recorded_kinds)
    nested_kinds = tuple(str(value.get("operator_kind")) for value in layer_values)
    attention = tuple(
        index for index, kind in enumerate(kinds) if kind == "full_attention"
    )
    short_conv = tuple(
        index for index, kind in enumerate(kinds) if kind == "short_conv"
    )
    if (
        not attention
        or nested_kinds != kinds
        or recorded_attention != list(attention)
        or recorded_short_conv != list(short_conv)
        or set(attention).intersection(short_conv)
        or set(attention).union(short_conv) != set(range(layer_count))
    ):
        raise Lfm2SemanticPostfoldError(
            "Semantic layer stack dynamic operator indices disagree with its "
            "sealed per-layer transactions."
        )
    return kinds, attention, short_conv


def authenticate_lfm2_semantic_layer_stack_attention_receipts(
    value: Mapping[str, Any],
    *,
    body_recipe: str | Lfm2SemanticBodyRecipe | None = None,
) -> Mapping[str, Any]:
    """Authenticate every persisted body component against one complete recipe."""

    stack = _verify_sealed(
        value,
        seal_name="result_sha256",
        label="semantic layer-stack receipt",
    )
    recipe = _authenticate_body_recipe_binding(
        stack,
        body_recipe=body_recipe,
        label="semantic layer-stack receipt",
    )
    layers = stack.get("layer_installations")
    layer_sha256 = stack.get("layer_installation_sha256")
    (
        operator_layer_kinds,
        attention_layer_indices,
        short_conv_layer_indices,
    ) = _authenticated_layer_stack_operator_topology(stack, layers)
    layer_count = len(operator_layer_kinds)
    terminal_layer_index = layer_count - 1
    if (
        stack.get("schema") != "neural-forge-lfm2-semantic-layer-stack-refold-v1"
        or not isinstance(layers, list)
        or not isinstance(layer_sha256, list)
        or len(layer_sha256) != layer_count
    ):
        raise Lfm2SemanticPostfoldError(
            "Semantic layer stack omitted its topology-derived layer transactions."
        )
    stack_before_sha256 = stack.get("model_state_before_sha256")
    stack_after_sha256 = stack.get("model_state_after_sha256")
    stack_changed_keys = stack.get("changed_state_keys")
    parameter_element_count = stack.get("parameter_element_count")
    invariant_failures = _semantic_layer_stack_state_invariant_failures(stack)
    if invariant_failures:
        raise Lfm2SemanticPostfoldError(
            "Semantic layer-stack state or parameter invariants changed: "
            + ", ".join(invariant_failures)
            + "."
        )
    global_residual_transition = _authenticate_semantic_global_residual_transition(
        stack,
        recipe,
    )
    causal_stack_after_sha256 = (
        stack_after_sha256
        if global_residual_transition is None
        else global_residual_transition["causal_layer_stack_model_state_after_sha256"]
    )
    causal_stack_changed_keys = (
        stack_changed_keys
        if global_residual_transition is None
        else global_residual_transition["causal_layer_stack_changed_state_keys"]
    )
    authenticated = []
    coupled_root_bindings: list[Mapping[str, Any]] = []
    verified_layers: list[Mapping[str, Any]] = []
    operator_installed_by_layer: list[bool] = []
    mlp_installed_by_layer: list[bool] = []
    expected_layer_before_sha256 = stack_before_sha256
    for layer_index, layer_value in enumerate(layers):
        layer = _verify_sealed(
            layer_value,
            seal_name="result_sha256",
            label=f"semantic layer {layer_index} transaction",
        )
        _authenticate_body_recipe_binding(
            layer,
            body_recipe=recipe,
            label=f"semantic layer {layer_index} transaction",
        )
        if (
            layer.get("schema") != "neural-forge-lfm2-semantic-layer-transaction-v1"
            or layer.get("layer_index") != layer_index
            or layer_sha256[layer_index] != layer["result_sha256"]
        ):
            raise Lfm2SemanticPostfoldError(
                f"Semantic layer {layer_index} transaction binding changed."
            )
        expected_operator_kind = operator_layer_kinds[layer_index]
        is_attention = expected_operator_kind == "full_attention"
        layer_before_sha256 = layer.get("model_state_before_operator_sha256")
        layer_after_operator_sha256 = layer.get("model_state_after_operator_sha256")
        layer_after_mlp_sha256 = layer.get("model_state_after_mlp_sha256")
        operator_installed = layer.get("operator_installed")
        mlp_installed = layer.get("mlp_installed")
        if (
            layer.get("operator_kind") != expected_operator_kind
            or not isinstance(layer_before_sha256, str)
            or not isinstance(layer_after_operator_sha256, str)
            or not isinstance(layer_after_mlp_sha256, str)
            or _digest(layer_before_sha256, "semantic layer state before operator")
            != layer_before_sha256
            or _digest(
                layer_after_operator_sha256,
                "semantic layer state after operator",
            )
            != layer_after_operator_sha256
            or _digest(layer_after_mlp_sha256, "semantic layer state after MLP")
            != layer_after_mlp_sha256
            or layer_before_sha256 != expected_layer_before_sha256
            or type(operator_installed) is not bool
            or layer.get("operator_decision_count") != 1
            or layer.get("operator_install_count") != int(operator_installed)
            or layer.get("operator_parameter_bytes_changed") is not operator_installed
            or (layer_before_sha256 != layer_after_operator_sha256)
            is not operator_installed
            or type(mlp_installed) is not bool
            or layer.get("mlp_decision_count") != 1
            or layer.get("mlp_install_count") != int(mlp_installed)
            or layer.get("mlp_parameter_bytes_changed") is not mlp_installed
            or (layer_after_operator_sha256 != layer_after_mlp_sha256)
            is not mlp_installed
        ):
            raise Lfm2SemanticPostfoldError(
                f"Semantic layer {layer_index} causal state chain changed."
            )
        operator_installed_by_layer.append(operator_installed)
        mlp_installed_by_layer.append(mlp_installed)
        expected_layer_before_sha256 = layer_after_mlp_sha256
        verified_layers.append(layer)
        operator_frontier_value = layer.get(
            "operator_complete_candidate_frontier_receipt"
        )
        mlp_frontier_value = layer.get("mlp_complete_candidate_frontier_receipt")
        if not isinstance(operator_frontier_value, Mapping) or not isinstance(
            mlp_frontier_value,
            Mapping,
        ):
            raise Lfm2SemanticPostfoldError(
                f"Semantic layer {layer_index} omitted its complete body frontiers."
            )
        operator_frontier = _verify_sealed(
            operator_frontier_value,
            seal_name="result_sha256",
            label=f"semantic layer {layer_index} operator frontier",
        )
        mlp_frontier = _verify_sealed(
            mlp_frontier_value,
            seal_name="result_sha256",
            label=f"semantic layer {layer_index} MLP frontier",
        )
        _authenticate_body_recipe_binding(
            operator_frontier,
            body_recipe=recipe,
            component="qk" if is_attention else "short_conv",
            label=f"semantic layer {layer_index} operator frontier",
        )
        _authenticate_body_recipe_binding(
            mlp_frontier,
            body_recipe=recipe,
            component="mlp",
            label=f"semantic layer {layer_index} MLP frontier",
        )
        if (
            operator_frontier.get("result_sha256")
            != layer.get("operator_decision_sha256")
            or operator_frontier.get("result_sha256")
            != layer.get("operator_installation_sha256")
            or mlp_frontier.get("result_sha256") != layer.get("mlp_decision_sha256")
            or mlp_frontier.get("result_sha256") != layer.get("mlp_installation_sha256")
            or operator_frontier.get("installed") is not operator_installed
            or mlp_frontier.get("installed") is not mlp_installed
        ):
            raise Lfm2SemanticPostfoldError(
                f"Semantic layer {layer_index} body frontier bindings disagree."
            )
        decision_value = layer.get("attention_operator_decision_receipt")
        installation_value = layer.get("attention_operator_installation_receipt")
        if not is_attention:
            if decision_value is not None or installation_value is not None:
                raise Lfm2SemanticPostfoldError(
                    f"Non-attention layer {layer_index} persisted attention data."
                )
            continue
        if not isinstance(decision_value, Mapping) or not isinstance(
            installation_value,
            Mapping,
        ):
            raise Lfm2SemanticPostfoldError(
                f"Attention layer {layer_index} omitted its full sealed receipts."
            )
        decision = _authenticate_lfm2_semantic_attention_operator_receipt(
            decision_value,
            layer_index=layer_index,
            body_recipe=recipe,
        )
        installation = _authenticate_lfm2_semantic_attention_operator_receipt(
            installation_value,
            layer_index=layer_index,
            body_recipe=recipe,
        )
        if (
            dict(decision_value) != dict(installation_value)
            or dict(decision_value) != dict(operator_frontier_value)
            or layer.get("operator_kind") != "full_attention"
            or layer.get("operator_decision_sha256")
            != decision["operator_receipt_sha256"]
            or layer.get("operator_installation_sha256")
            != installation["operator_receipt_sha256"]
            or layer.get("operator_installed") is not decision["installed"]
            or layer.get("operator_install_count") != int(decision["installed"])
            or layer.get("operator_parameter_bytes_changed")
            is not decision["installed"]
            or layer.get("model_state_before_operator_sha256")
            != decision["model_state_before_sha256"]
            or layer.get("model_state_after_operator_sha256")
            != decision["model_state_after_sha256"]
            or layer.get("operator_evidence_sha256")
            != decision["transport_evidence_sha256"]
            or decision["parameter_element_count"] != parameter_element_count
            or (
                layer.get("model_state_before_operator_sha256")
                != layer.get("model_state_after_operator_sha256")
            )
            is not decision["installed"]
        ):
            raise Lfm2SemanticPostfoldError(
                f"Attention layer {layer_index} receipt bindings disagree."
            )
        if decision.get("coupled_attention_group_qdeim") is True:
            coupled_root_bindings.append(
                _authenticate_lfm2_coupled_attention_stack_root_binding(
                    stack,
                    decision,
                )
            )
        authenticated.append(dict(decision))
    if expected_layer_before_sha256 != causal_stack_after_sha256:
        raise Lfm2SemanticPostfoldError(
            "Semantic layer-stack final causal state binding changed."
        )
    authenticated_by_layer = {item["layer_index"]: item for item in authenticated}
    owned_changed_keys: set[str] = set()
    changed_key_bindings_invalid = False
    for layer_index, layer in enumerate(verified_layers):
        is_attention = operator_layer_kinds[layer_index] == "full_attention"
        operator_prefix = (
            f"model.layers.{layer_index}.self_attn."
            if is_attention
            else f"model.layers.{layer_index}.conv."
        )
        mlp_prefix = f"model.layers.{layer_index}.feed_forward."
        operator_changed_keys = {
            name
            for name in causal_stack_changed_keys
            if name.startswith(operator_prefix)
        }
        mlp_changed_keys = {
            name for name in causal_stack_changed_keys if name.startswith(mlp_prefix)
        }
        if (
            bool(operator_changed_keys) is not operator_installed_by_layer[layer_index]
            or bool(mlp_changed_keys) is not mlp_installed_by_layer[layer_index]
            or (
                is_attention
                and operator_changed_keys
                != set(authenticated_by_layer[layer_index]["changed_state_keys"])
            )
        ):
            changed_key_bindings_invalid = True
        owned_changed_keys.update(operator_changed_keys)
        owned_changed_keys.update(mlp_changed_keys)
    short_conv_install_count = sum(
        int(operator_installed_by_layer[index]) for index in short_conv_layer_indices
    )
    if (
        changed_key_bindings_invalid
        or owned_changed_keys != set(causal_stack_changed_keys)
        or stack.get("short_conv_operator_decision_count")
        != len(short_conv_layer_indices)
        or stack.get("short_conv_operator_install_count") != short_conv_install_count
        or stack.get("operator_parameter_mutation_count")
        != sum(int(value) for value in operator_installed_by_layer)
        or stack.get("replay_fresh_mlp_evidence_compile_count") != layer_count
        or stack.get("nonterminal_mlp_writer_decision_count") != terminal_layer_index
        or stack.get("nonterminal_mlp_writer_install_count")
        != sum(int(value) for value in mlp_installed_by_layer[:terminal_layer_index])
        or stack.get("final_mlp_producer_install_count")
        != int(mlp_installed_by_layer[terminal_layer_index])
        or stack.get("mlp_parameter_mutation_count")
        != sum(int(value) for value in mlp_installed_by_layer)
    ):
        raise Lfm2SemanticPostfoldError(
            "Semantic layer-stack changed-key or parameter mutation bindings changed."
        )
    if (
        tuple(item["layer_index"] for item in authenticated) != attention_layer_indices
        or stack.get("attention_operator_decision_count") != len(authenticated)
        or stack.get("attention_operator_install_count")
        != sum(int(item["installed"]) for item in authenticated)
    ):
        raise Lfm2SemanticPostfoldError(
            "Semantic layer-stack attention receipt coverage is incomplete."
        )
    changed_projection = [
        name for item in authenticated for name in item["changed_projection_state_keys"]
    ]
    changed_norm = [
        name for item in authenticated for name in item["changed_norm_state_keys"]
    ]
    body = {
        "schema": "neural-forge-lfm2-semantic-attention-receipt-persistence-v1",
        "body_recipe": recipe.as_dict(),
        "body_recipe_sha256": recipe.recipe_sha256,
        "body_recipe_id": recipe.recipe_id,
        "complete_body_recipe_candidate_ids": list(LFM2_SEMANTIC_BODY_RECIPE_IDS),
        "complete_body_recipe_candidate_count": len(LFM2_SEMANTIC_BODY_RECIPE_IDS),
        "materialized_body_recipe_id": recipe.recipe_id,
        "semantic_layer_stack_result_sha256": stack["result_sha256"],
        "model_state_before_sha256": stack_before_sha256,
        "model_state_after_sha256": stack_after_sha256,
        "layer_stack_changed_state_keys_sha256": canonical_sha256(stack_changed_keys),
        "layer_stack_changed_state_key_count": len(stack_changed_keys),
        "parameter_element_count": parameter_element_count,
        "attention_layer_indices": list(attention_layer_indices),
        "attention_receipt_count": len(authenticated),
        "attention_install_count": sum(
            int(item["installed"]) for item in authenticated
        ),
        "complete_six_tensor_bundle_count": len(authenticated),
        "changed_projection_state_key_count": len(changed_projection),
        "changed_norm_state_key_count": len(changed_norm),
        "changed_projection_state_keys": changed_projection,
        "changed_norm_state_keys": changed_norm,
        "per_layer": authenticated,
        "coupled_attention_layer_indices": [
            item["layer_index"] for item in coupled_root_bindings
        ],
        "coupled_attention_root_binding_count": len(coupled_root_bindings),
        "coupled_attention_persisted_roots": [
            dict(item) for item in coupled_root_bindings
        ],
        "all_coupled_attention_stack_roots_authenticated": True,
        "full_operator_receipts_embedded": True,
        "full_installation_receipts_embedded": True,
        "all_six_tensor_transactions_authenticated": True,
        "all_short_conv_receipts_recipe_authenticated": True,
        "all_mlp_receipts_recipe_authenticated": True,
        "all_component_frontiers_recipe_authenticated": True,
        "causal_model_state_chain_authenticated": True,
        "changed_state_keys_authenticated": True,
        **(
            {
                "global_residual_closure_transition_authenticated": True,
                "global_residual_closure_transition": dict(global_residual_transition),
                "local_then_global_two_transition_chain_authenticated": True,
            }
            if global_residual_transition is not None
            else {}
        ),
        "parameter_invariants_authenticated": True,
        "heldout_metrics_are_diagnostic_only": True,
        "numeric_behavior_gate_used": False,
        "numeric_metrics_have_final_intelligence_authority": False,
        "one_authentication_receipt_covers_one_materialized_body_recipe": True,
        "other_body_recipes_require_sealed_foundation_replay": True,
    }
    return MappingProxyType(_sealed(body, "result_sha256"))


def _operator_layer_kinds(
    cache: Lfm2CircuitTraceCache,
    target_model: Any,
) -> tuple[str, ...]:
    topology = cache.manifest.get("layer_topology", {}).get("layers")
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    if not isinstance(topology, list) or not topology or len(topology) != len(layers):
        raise Lfm2SemanticPostfoldError(
            "Cache and target model do not expose one matching nonempty LFM2 "
            "layer topology."
        )
    kinds = []
    for index, value in enumerate(topology):
        contract = value.get("contract") if isinstance(value, Mapping) else None
        kind = contract.get("kind") if isinstance(contract, Mapping) else None
        if kind not in {"full_attention", "short_conv"}:
            raise Lfm2SemanticPostfoldError(
                f"Cache layer {index} has no supported LFM2 operator contract."
            )
        child_is_attention = getattr(layers[index], "is_attention_layer", None)
        if not isinstance(child_is_attention, bool) or child_is_attention != (
            kind == "full_attention"
        ):
            raise Lfm2SemanticPostfoldError(
                "Parent-cache and saved-child operator topology differs at layer "
                f"{index}."
            )
        kinds.append(str(kind))
    if "full_attention" not in kinds:
        raise Lfm2SemanticPostfoldError(
            "Parent-cache topology contains no attention operator."
        )
    child_types = tuple(getattr(target_model.config, "layer_types", ()))
    normalized_child_types = tuple(
        "short_conv" if value in {"conv", "short_conv"} else str(value)
        for value in child_types
    )
    if child_types and normalized_child_types != tuple(kinds):
        raise Lfm2SemanticPostfoldError(
            "Saved child layer types differ from the complete cache topology."
        )
    return tuple(kinds)


def _attention_layer_indices(
    cache: Lfm2CircuitTraceCache,
    target_model: Any,
) -> tuple[int, ...]:
    return tuple(
        index
        for index, kind in enumerate(_operator_layer_kinds(cache, target_model))
        if kind == "full_attention"
    )


def _target_rope(target_model: Any) -> tuple[Any, float]:
    torch = _torch()
    rotary = getattr(getattr(target_model, "model", None), "rotary_emb", None)
    inv_freq = getattr(rotary, "inv_freq", None)
    scaling = getattr(rotary, "attention_scaling", 1.0)
    if (
        not isinstance(inv_freq, torch.Tensor)
        or inv_freq.ndim != 1
        or inv_freq.numel() == 0
        or not bool(torch.isfinite(inv_freq).all().item())
        or isinstance(scaling, bool)
        or not isinstance(scaling, (int, float))
        or not math.isfinite(float(scaling))
        or float(scaling) <= 0.0
    ):
        raise Lfm2SemanticPostfoldError(
            "Saved child exposes no finite target RoPE contract."
        )
    return (
        inv_freq.detach().to(device="cpu", dtype=torch.float64).contiguous().clone(),
        float(scaling),
    )


def _input_tensor_names(reference: Any) -> tuple[str, ...]:
    inventory = set(reference.ordered_tensor_names)
    record = reference.record
    capture = record.get("capture") if isinstance(record, Mapping) else None
    decode_steps = capture.get("decode_steps") if isinstance(capture, Mapping) else None
    if (
        isinstance(decode_steps, bool)
        or not isinstance(decode_steps, int)
        or decode_steps <= 0
    ):
        raise Lfm2SemanticPostfoldError(
            "Complete cache input evidence has no forced decode horizon."
        )
    names = {
        "input.input_ids",
        "input.attention_mask",
        "input.forced_decode_ids",
        *(f"decode.{step:03d}.forced_token_ids" for step in range(decode_steps)),
    }
    if "input.position_ids" in inventory:
        names.add("input.position_ids")
    missing = names.difference(inventory)
    if missing:
        raise Lfm2SemanticPostfoldError(
            f"Complete cache input evidence is missing {sorted(missing)!r}."
        )
    return tuple(sorted(names))


def _semantic_alignments_from_session(
    *,
    cache: Lfm2CircuitTraceCache,
    session: Any,
    bridge: StrictTokenizerBridgeArtifact,
    donor_tokenizer: Any,
    target_tokenizer: Any,
) -> tuple[
    tuple[Lfm2SemanticSpanCpcfAlignment, ...],
    Mapping[str, Any],
]:
    manifest = cache.manifest
    manifest_sha256 = _digest(
        manifest.get("manifest_sha256"),
        "cache manifest SHA-256",
    )
    session_receipt = session.as_dict()
    alignments = []
    with bridge.authenticated_live_session(
        donor_tokenizer,
        target_tokenizer,
        selective_session_sha256=_digest(
            session_receipt.get("session_sha256"),
            "selective cache session SHA-256",
        ),
    ) as tokenizer_session:
        donor_sha = tokenizer_session.donor_tokenizer_semantics_sha256
        target_sha = tokenizer_session.target_tokenizer_semantics_sha256
        for reference in session.batch_references:
            payload = session.load_selected_tensors(
                reference.index,
                _input_tensor_names(reference),
            )
            try:
                selected_input = build_lfm2_semantic_span_selected_input_evidence(
                    reference,
                    payload,
                    session_receipt,
                    expected_cache_key=cache.cache_key,
                )
                semantic = build_lfm2_semantic_span_phase_bridge(
                    selected_input,
                    manifest,
                    donor_tokenizer,
                    target_tokenizer,
                    expected_source_cache_key=cache.cache_key,
                    expected_source_manifest_sha256=manifest_sha256,
                    expected_source_input_sha256=reference.input_sha256,
                    expected_donor_tokenizer_fingerprint=donor_sha,
                    expected_target_tokenizer_fingerprint=target_sha,
                    tokenizer_session=tokenizer_session,
                )
                prompt = len(semantic.source_prompt_token_ids)
                suffix_counts = {
                    phase: prompt if phase == "prefill" else 1
                    for phase in semantic.phase_order
                }
                alignments.append(
                    build_lfm2_semantic_span_cpcf_alignment(
                        semantic,
                        donor_tokenizer,
                        target_tokenizer,
                        parent_sampled_suffix_tokens=suffix_counts,
                        parent_history_suffix_tokens=suffix_counts,
                        tokenizer_session=tokenizer_session,
                    )
                )
            finally:
                session.release_selected_tensors(payload)
    return tuple(alignments), MappingProxyType(tokenizer_session.as_dict())


def _attention_tensor_names(
    reference: Any,
    *,
    layer_index: int,
    phase_order: Sequence[str],
) -> tuple[str, ...]:
    inventory = set(reference.ordered_tensor_names)
    names = set()
    for phase in phase_order:
        prefix = f"{phase}.layer.{layer_index:03d}.attention."
        names.update(
            {
                prefix + "q_normalized_pre_rope",
                prefix + "k_normalized_pre_rope",
                prefix + "q",
                prefix + "k",
                prefix + "v",
            }
        )
    names.add(f"{phase_order[-1]}.layer.{layer_index:03d}.attention.value_history")
    missing = tuple(sorted(names.difference(inventory)))
    if missing:
        raise Lfm2SemanticPostfoldError(
            "Complete causal cache lacks semantic attention tensors: "
            f"{list(missing[:6])}."
        )
    return tuple(sorted(names))


def _parent_residual_tensor_names(
    reference: Any,
    *,
    layer_index: int,
    phase_order: Sequence[str],
) -> tuple[str, ...]:
    inventory = set(reference.ordered_tensor_names)
    names = {
        f"{phase}.layer.{layer_index:03d}.residual.{boundary}"
        for phase in phase_order
        for boundary in ("pre_operator", "post_operator")
    }
    missing = tuple(sorted(names.difference(inventory)))
    if missing:
        raise Lfm2SemanticPostfoldError(
            "Complete causal cache lacks parent residual endpoints: "
            f"{list(missing[:6])}."
        )
    return tuple(sorted(names))


@dataclass(frozen=True, slots=True)
class _SemanticLayerReplay:
    batch_index: int
    raw_pre_attention_residual: Any = field(repr=False, compare=False)
    operator_input: Any = field(repr=False, compare=False)
    replay_sha256: str


def _layer_runtime(layer: Any) -> tuple[Any, Any]:
    torch = _torch()
    parameters = tuple(layer.parameters())
    if (
        not parameters
        or any(
            value.device.type == "meta"
            or value.dtype != torch.bfloat16
            or not torch.is_floating_point(value)
            for value in parameters
        )
        or len({value.device for value in parameters}) != 1
    ):
        raise Lfm2SemanticPostfoldError(
            "Every child macrocell must use one materialized BF16 device."
        )
    return parameters[0].device, parameters[0].dtype


def _additive_causal_mask(attention_mask: Any, residual: Any) -> Any:
    torch = _torch()
    sequence = int(attention_mask.shape[1])
    allowed = torch.tril(
        torch.ones(
            (sequence, sequence),
            device=residual.device,
            dtype=torch.bool,
        )
    )[None, None]
    active = attention_mask.to(
        device=residual.device,
        dtype=torch.bool,
    )[:, None, None, :]
    allowed = allowed & active
    minimum = torch.finfo(residual.dtype).min
    return torch.where(
        allowed,
        torch.zeros((), device=residual.device, dtype=residual.dtype),
        torch.full((), minimum, device=residual.device, dtype=residual.dtype),
    )


def _replay_child_to_attention_layer(
    target_model: Any,
    *,
    alignment: Lfm2SemanticSpanCpcfAlignment,
    layer_index: int,
) -> _SemanticLayerReplay:
    """Replay one final target prefix analytically; never call a model/layer."""

    torch = _torch()
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    if layer_index >= len(layers):
        raise Lfm2SemanticPostfoldError("Child replay layer lies outside topology.")
    final = alignment.phases[-1]
    ids = final.canonical_target_input_ids
    mask = final.canonical_target_attention_mask
    positions = final.canonical_target_position_ids
    embedding = getattr(target_model.get_input_embeddings(), "weight", None)
    if (
        not isinstance(embedding, torch.nn.Parameter)
        or embedding.dtype != torch.bfloat16
        or bool((ids < 0).any().item())
        or bool((ids >= int(embedding.shape[0])).any().item())
    ):
        raise Lfm2SemanticPostfoldError(
            "Target-native replay IDs exceed the frozen tied BF16 table."
        )
    with torch.no_grad():
        device_ids = ids.to(device=embedding.device, dtype=torch.long)
        residual = embedding.index_select(0, device_ids.reshape(-1)).reshape(
            1,
            int(device_ids.shape[1]),
            int(embedding.shape[1]),
        )
    replay_sha = []
    raw = None
    operator = None
    for index, layer in enumerate(layers[: layer_index + 1]):
        device, dtype = _layer_runtime(layer)
        residual = residual.to(device=device, dtype=dtype, non_blocking=False)
        if index == layer_index:
            raw = residual.detach().to(device="cpu").contiguous().clone()
        if bool(getattr(layer, "is_attention_layer", False)):
            replay = replay_lfm2_cpcf_macrocell(
                layer,
                residual,
                position_embeddings=_rope_from_target(
                    target_model,
                    residual,
                    positions,
                ),
                additive_causal_mask=_additive_causal_mask(mask, residual),
            )
        else:
            replay = replay_lfm2_cpcf_macrocell(
                layer,
                residual,
                padding_mask=mask.to(device=device, dtype=dtype),
            )
        replay_sha.append(replay.replay_sha256)
        if index == layer_index:
            operator = replay.operator_input
        residual = replay.post_layer_residual
    if raw is None or operator is None:
        raise Lfm2SemanticPostfoldError(
            "Target-native replay produced no attention endpoint."
        )
    body = {
        "schema": "neural-forge-lfm2-semantic-child-attention-replay-v1",
        "batch_index": alignment.source_batch_index,
        "layer_index": layer_index,
        "alignment_sha256": alignment.alignment_sha256,
        "final_phase_alignment_sha256": final.alignment_sha256,
        "raw_pre_attention_residual_sha256": _tensor_sha256(raw),
        "operator_input_sha256": _tensor_sha256(operator),
        "macrocell_replay_sha256": replay_sha,
        "model_forward_calls": 0,
        "layer_forward_calls": 0,
        "training_steps": 0,
    }
    return _SemanticLayerReplay(
        batch_index=alignment.source_batch_index,
        raw_pre_attention_residual=raw,
        operator_input=operator,
        replay_sha256=canonical_sha256(body),
    )


def _semantic_residual_chart(
    *,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    parent_pre_residual: Mapping[int, Any],
    child_replays: Mapping[int, _SemanticLayerReplay],
    producer_encoder: Any,
    layer_index: int,
) -> tuple[Any, Mapping[str, Any]]:
    """Fit one FIT-only orthogonal gauge around the saved interface chart."""

    torch = _torch()
    ordered = tuple(sorted(alignments, key=lambda value: value.source_batch_index))
    fit_count = max(1, len(ordered) // 2)
    if len(ordered) < 4:
        raise Lfm2SemanticPostfoldError(
            "Residual chart requires four trajectory-isolated cache batches."
        )
    encoder = producer_encoder.detach().to(device="cpu", dtype=torch.float64)
    fit_left = []
    fit_right = []
    check_left = []
    check_right = []
    fit_batches = {value.source_batch_index for value in ordered[:fit_count]}
    for alignment in ordered:
        batch = alignment.source_batch_index
        parent = parent_pre_residual[batch].to(torch.float64)
        replay = child_replays[batch].raw_pre_attention_residual.to(torch.float64)
        boundary = alignment.phases[-1].target_to_donor_boundary_positions
        targets = tuple(
            index for index, value in enumerate(boundary) if value is not None
        )
        donors = tuple(int(value) for value in boundary if value is not None)
        if donors != tuple(range(int(parent.shape[1]))):
            raise Lfm2SemanticPostfoldError(
                "Final semantic endpoints do not cover every donor residual row."
            )
        selected = replay.index_select(
            1,
            torch.tensor(targets, dtype=torch.long),
        )
        left = (parent[0] @ encoder).contiguous()
        right = selected[0].contiguous()
        destination_left = fit_left if batch in fit_batches else check_left
        destination_right = fit_right if batch in fit_batches else check_right
        # Equal total trajectory mass prevents long prompts dominating the chart.
        destination_left.append(left / math.sqrt(int(left.shape[0])))
        destination_right.append(right / math.sqrt(int(right.shape[0])))
    left_fit = torch.cat(fit_left, dim=0)
    right_fit = torch.cat(fit_right, dim=0)
    cross = left_fit.T @ right_fit
    left_vectors, singular, right_vectors_h = torch.linalg.svd(
        cross,
        full_matrices=False,
    )
    rotation = (left_vectors @ right_vectors_h).contiguous()
    if tuple(rotation.shape) != (
        int(encoder.shape[1]),
        int(encoder.shape[1]),
    ) or not bool(torch.isfinite(rotation).all().item()):
        raise Lfm2SemanticPostfoldError("FIT-only semantic residual gauge is invalid.")

    def metrics(left_rows: Sequence[Any], right_rows: Sequence[Any]) -> dict[str, Any]:
        left = torch.cat(tuple(left_rows), dim=0)
        right = torch.cat(tuple(right_rows), dim=0)
        prediction = left @ rotation
        difference = prediction - right
        denominator = right.square().mean().sqrt().clamp_min(1e-30)
        return {
            "rows": int(left.shape[0]),
            "relative_rmse": float(
                (difference.square().mean().sqrt() / denominator).item()
            ),
            "maximum_absolute_error": float(difference.abs().max().item()),
        }

    body = {
        "schema": LFM2_SEMANTIC_RESIDUAL_CHART_SCHEMA,
        "layer_index": layer_index,
        "fit_batch_indices": sorted(fit_batches),
        "heldout_batch_indices": sorted(
            value.source_batch_index for value in ordered[fit_count:]
        ),
        "base_producer_encoder_sha256": _tensor_sha256(encoder),
        "rotation_sha256": _tensor_sha256(rotation),
        "singular_values_sha256": _tensor_sha256(singular),
        "fit_metrics": metrics(fit_left, fit_right),
        "heldout_metrics": metrics(check_left, check_right),
        "method": "fit_only_orthogonal_procrustes_around_saved_interface_chart",
        "validation_rows_used_for_fit": 0,
        "numeric_parent_child_weight_match_objective": False,
        "model_forward_calls": 0,
        "training_steps": 0,
    }
    return (encoder @ rotation).contiguous(), MappingProxyType(
        _sealed(body, "chart_sha256")
    )


def _transport_evidence_for_layer(
    *,
    cache: Lfm2CircuitTraceCache,
    session: Any,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    layer_index: int,
    target_rope_inv_freq: Any,
    target_rope_attention_scaling: float,
    target_model: Any,
    bridge: StrictTokenizerBridgeArtifact,
    interface: TiedCausalInterfaceFold,
    compiler_lineage: Lfm2SemanticCompilerLayerLineage,
    parent_qk_tensors: Mapping[str, Any],
    parent_norm_epsilon: float,
) -> tuple[Lfm2SemanticAttentionTransportEvidence, ...]:
    torch = _torch()
    parent_pre: dict[int, Any] = {}
    parent_post: dict[int, Any] = {}
    child_replays = {
        alignment.source_batch_index: _replay_child_to_attention_layer(
            target_model,
            alignment=alignment,
            layer_index=layer_index,
        )
        for alignment in alignments
    }
    for alignment, reference in zip(
        alignments,
        session.batch_references,
        strict=True,
    ):
        payload = session.load_selected_tensors(
            reference.index,
            _parent_residual_tensor_names(
                reference,
                layer_index=layer_index,
                phase_order=alignment.phase_order,
            ),
        )
        try:
            parent_pre[reference.index] = torch.cat(
                tuple(
                    payload.tensors[
                        f"{phase}.layer.{layer_index:03d}.residual.pre_operator"
                    ]
                    for phase in alignment.phase_order
                ),
                dim=1,
            ).contiguous()
            parent_post[reference.index] = torch.cat(
                tuple(
                    payload.tensors[
                        f"{phase}.layer.{layer_index:03d}.residual.post_operator"
                    ]
                    for phase in alignment.phase_order
                ),
                dim=1,
            ).contiguous()
        finally:
            session.release_selected_tensors(payload)
    chart, chart_report = _semantic_residual_chart(
        alignments=alignments,
        parent_pre_residual=parent_pre,
        child_replays=child_replays,
        producer_encoder=interface.producer_encoder,
        layer_index=layer_index,
    )
    ordered_alignments = tuple(
        sorted(alignments, key=lambda value: value.source_batch_index)
    )
    parent_operator_rows = []
    child_operator_rows = []
    for alignment in ordered_alignments:
        batch_index = alignment.source_batch_index
        boundary = alignment.phases[-1].target_to_donor_boundary_positions
        target_positions = tuple(
            index for index, value in enumerate(boundary) if value is not None
        )
        donor_positions = tuple(int(value) for value in boundary if value is not None)
        parent_operator_rows.append(
            parent_pre[batch_index][0].index_select(
                0,
                torch.tensor(donor_positions, dtype=torch.long),
            )
        )
        child_operator_rows.append(
            child_replays[batch_index]
            .operator_input[0]
            .index_select(
                0,
                torch.tensor(target_positions, dtype=torch.long),
            )
        )
    target_layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    target_operator_norm_weight = getattr(
        getattr(target_layers[layer_index], "operator_norm", None),
        "weight",
        None,
    )
    target_norm_epsilon = getattr(
        getattr(target_model, "config", None),
        "norm_eps",
        None,
    )
    if (
        target_operator_norm_weight is None
        or not isinstance(target_norm_epsilon, (int, float))
        or not math.isfinite(float(target_norm_epsilon))
        or float(target_norm_epsilon) <= 0.0
    ):
        raise Lfm2SemanticPostfoldError(
            "Target attention layer has no operator_norm weight or norm epsilon."
        )
    normalized_input_decoder, normalized_input_decoder_receipt = (
        _fit_normalized_operator_input_decoder(
            parent_raw_rows=parent_operator_rows,
            child_normalized_rows=child_operator_rows,
            parent_operator_norm_weight=parent_qk_tensors["operator_norm"],
            target_operator_norm_weight=target_operator_norm_weight,
            parent_norm_epsilon=parent_norm_epsilon,
            target_norm_epsilon=float(target_norm_epsilon),
            residual_chart=chart,
            residual_chart_sha256=chart_report["chart_sha256"],
            fit_trajectory_count=max(1, len(ordered_alignments) // 2),
        )
    )
    (
        parent_query_weight_prior,
        parent_key_weight_prior,
        parent_value_weight_prior,
        parent_output_weight_prior,
        parent_query_norm_weight,
        parent_key_norm_weight,
        parent_weight_prior_receipt,
    ) = _fold_parent_qk_weight_priors(
        parent_query_weight=parent_qk_tensors["q"],
        parent_key_weight=parent_qk_tensors["k"],
        parent_value_weight=parent_qk_tensors["v"],
        parent_output_weight=parent_qk_tensors["o"],
        parent_query_norm_weight=parent_qk_tensors["q_norm"],
        parent_key_norm_weight=parent_qk_tensors["k_norm"],
        residual_chart=chart,
        query_groups=compiler_lineage.query_groups,
        query_group_weights=compiler_lineage.query_group_weights,
        key_value_groups=compiler_lineage.key_value_groups,
        key_value_group_weights=compiler_lineage.key_value_group_weights,
        head_dim=int(parent_qk_tensors["q_norm"].numel()),
        source_inventory_sha256=compiler_lineage.source_inventory.inventory_sha256,
        residual_chart_sha256=chart_report["chart_sha256"],
        normalized_input_decoder=normalized_input_decoder,
        normalized_input_decoder_receipt=normalized_input_decoder_receipt,
    )
    captured_state = fingerprint_lfm2_parent(target_model).identity_sha256
    result = []
    for alignment, reference in zip(
        alignments,
        session.batch_references,
        strict=True,
    ):
        phases = alignment.phase_order
        payload = session.load_selected_tensors(
            reference.index,
            _attention_tensor_names(
                reference,
                layer_index=layer_index,
                phase_order=phases,
            ),
        )
        try:
            tensors = payload.tensors
            prompt = alignment.phases[0].donor_full_prefix_tokens
            capability = inspect_lfm2_semantic_attention_cache_capability(
                tensors,
                phase_order=phases,
                layer_index=layer_index,
                donor_prompt_token_count=prompt,
            )
            capability.require_exact_history()
            history = assemble_lfm2_explicit_pre_rope_history(
                tensors,
                phase_order=phases,
                layer_index=layer_index,
                donor_prompt_token_count=prompt,
            )
            parent_query = torch.cat(
                tuple(
                    tensors[f"{phase}.layer.{layer_index:03d}.attention.q"]
                    for phase in phases
                ),
                dim=2,
            ).contiguous()
            parent_key = torch.cat(
                tuple(
                    tensors[f"{phase}.layer.{layer_index:03d}.attention.k"]
                    for phase in phases
                ),
                dim=2,
            ).contiguous()
            replay = child_replays[reference.index]
            for phase_index, phase_alignment in enumerate(alignment.phases):
                donor_tokens = phase_alignment.donor_full_prefix_tokens
                target_tokens = phase_alignment.target_full_prefix_tokens
                phase_order = alignment.phase_order[: phase_index + 1]
                phase_capability = Lfm2SemanticAttentionCacheCapability(
                    layer_index=layer_index,
                    phase_order=phase_order,
                    donor_prompt_token_count=prompt,
                    selected_route="explicit_pre_rope",
                    prefill_captured_token_count=prompt,
                    assembled_current_row_count=donor_tokens,
                    missing_enhanced_tensor_names=(),
                    missing_legacy_tensor_names=(),
                    missing_donor_prefix_positions=(),
                    blocking_reasons=(),
                    exact_history_supported=True,
                )
                phase_history = SemanticAttentionPreRopeHistory(
                    query_normalized_pre_rope=(
                        history.query_normalized_pre_rope[:, :, :donor_tokens]
                    ),
                    key_normalized_pre_rope=(
                        history.key_normalized_pre_rope[:, :, :donor_tokens]
                    ),
                    value=history.value[:, :, :donor_tokens],
                    source_capability_sha256=(phase_capability.capability_sha256),
                    source_tensor_names=tuple(
                        name
                        for name in history.source_tensor_names
                        if any(name.startswith(item + ".") for item in phase_order)
                    ),
                )
                transport = transport_lfm2_phase_probability_preserving_attention(
                    phase_alignment,
                    parent_query_post_rope=parent_query[:, :, :donor_tokens],
                    parent_key_post_rope=parent_key[:, :, :donor_tokens],
                    parent_value=history.value[:, :, :donor_tokens],
                    target_rope_inv_freq=target_rope_inv_freq,
                    target_rope_attention_scaling=target_rope_attention_scaling,
                    query_mode="semantic_endpoints",
                )
                if hasattr(
                    transport,
                    "target_to_donor_span_owner_positions",
                ) and hasattr(phase_history, "history_sha256"):
                    (
                        folded_parent_query_pre_rope,
                        folded_parent_key_pre_rope,
                        parent_pre_rope_prior_receipt,
                    ) = fold_semantic_attention_parent_pre_rope_priors(
                        phase_history,
                        transport,
                        query_groups=compiler_lineage.query_groups,
                        query_group_weights=compiler_lineage.query_group_weights,
                        key_value_groups=compiler_lineage.key_value_groups,
                        key_value_group_weights=(
                            compiler_lineage.key_value_group_weights
                        ),
                    )
                else:  # Narrow compatibility for typed-constructor test doubles.
                    folded_parent_query_pre_rope = None
                    folded_parent_key_pre_rope = None
                    parent_pre_rope_prior_receipt = None
                query_positions = torch.tensor(
                    transport.query_target_positions,
                    dtype=torch.long,
                )
                donor_positions = torch.tensor(
                    transport.query_donor_positions,
                    dtype=torch.long,
                )
                child_query = replay.operator_input[:, :target_tokens].index_select(
                    1,
                    query_positions,
                )
                child_key_value = replay.operator_input[:, :target_tokens]
                actual_pre = replay.raw_pre_attention_residual[
                    :, :target_tokens
                ].index_select(1, query_positions)
                parent_target = (
                    parent_post[reference.index][:, :donor_tokens]
                    .index_select(1, donor_positions)
                    .to(torch.float64)
                    @ chart
                ).to(torch.float32)
                weights = torch.ones(
                    (1, len(transport.query_target_positions)),
                    dtype=torch.float64,
                )
                actual_child = Lfm2SemanticAttentionReplayEndpoint(
                    batch_index=reference.index,
                    layer_index=layer_index,
                    alignment_sha256=phase_alignment.alignment_sha256,
                    cache_key=cache.cache_key,
                    bridge_artifact_sha256=bridge.artifact_sha256,
                    captured_child_model_state_sha256=captured_state,
                    target_query_inputs=child_query,
                    target_key_value_inputs=child_key_value,
                    actual_child_pre_attention_residual=actual_pre,
                    parent_post_attention_residual_target_space=parent_target,
                    query_row_sqrt_weights=weights,
                )
                result.append(
                    Lfm2SemanticAttentionTransportEvidence(
                        batch_index=reference.index,
                        layer_index=layer_index,
                        alignment_sha256=phase_alignment.alignment_sha256,
                        alignment=phase_alignment,
                        capability=phase_capability.to_evidence_dict(),
                        history=phase_history.to_evidence_dict(),
                        actual_child_replay=actual_child,
                        source_inventory=compiler_lineage.source_inventory,
                        query_groups=compiler_lineage.query_groups,
                        query_group_weights=(compiler_lineage.query_group_weights),
                        key_value_groups=compiler_lineage.key_value_groups,
                        key_value_group_weights=(
                            compiler_lineage.key_value_group_weights
                        ),
                        compiler_lineage_sha256=canonical_sha256(
                            {
                                "compiler_lineage_sha256": (
                                    compiler_lineage.compiler_lineage_sha256
                                ),
                                "semantic_residual_chart_sha256": (
                                    chart_report["chart_sha256"]
                                ),
                            }
                        ),
                        transport=transport,
                        folded_parent_query_pre_rope=(folded_parent_query_pre_rope),
                        folded_parent_key_pre_rope=folded_parent_key_pre_rope,
                        parent_pre_rope_prior_receipt=(parent_pre_rope_prior_receipt),
                        parent_query_weight_prior=parent_query_weight_prior,
                        parent_key_weight_prior=parent_key_weight_prior,
                        parent_value_weight_prior=parent_value_weight_prior,
                        parent_output_weight_prior=parent_output_weight_prior,
                        parent_query_norm_weight=parent_query_norm_weight,
                        parent_key_norm_weight=parent_key_norm_weight,
                        parent_weight_prior_receipt=parent_weight_prior_receipt,
                    )
                )
        finally:
            session.release_selected_tensors(payload)
    return tuple(result)


def _resolve_attention_installer(
    explicit: Callable[..., Mapping[str, Any]] | None,
) -> Callable[..., Mapping[str, Any]] | None:
    if explicit is not None:
        if not callable(explicit):
            raise TypeError("attention_installer must be callable.")
        return explicit
    from neural_forge import hf_lfm2_semantic_attention_refold_adapter as adapter

    value = getattr(adapter, SEMANTIC_ATTENTION_INSTALLER_CALLABLE, None)
    if callable(value):
        return value
    from neural_forge import hf_semantic_attention_lattice as lattice

    fallback = getattr(lattice, SEMANTIC_ATTENTION_INSTALLER_CALLABLE, None)
    return fallback if callable(fallback) else None


def _resolve_layer_stack_installer(
    explicit: Callable[..., Any] | None,
) -> Callable[..., Any] | None:
    if explicit is not None:
        if not callable(explicit):
            raise TypeError("semantic_layer_stack_compiler must be callable.")
        return explicit
    try:
        from neural_forge import hf_lfm2_semantic_layer_stack_refold_adapter as adapter
    except ImportError:
        return None
    value = getattr(adapter, SEMANTIC_LAYER_STACK_INSTALLER_CALLABLE, None)
    return value if callable(value) else None


def _final_w2_parameter(target_model: Any) -> Any:
    torch = _torch()
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    parameter = None
    if layers:
        parameter = getattr(
            getattr(getattr(layers[-1], "feed_forward", None), "w2", None),
            "weight",
            None,
        )
    if not isinstance(parameter, torch.nn.Parameter):
        raise Lfm2SemanticPostfoldError("Child final W2 is not a Parameter.")
    return parameter


def _validate_live_layer_stack_producer_coverage(
    target_model: Any,
    receipt: Mapping[str, Any],
) -> None:
    """Cross-check every nested coverage proof against the live folded child."""

    layer_values = receipt.get("layer_installations")
    operator_layer_kinds, _attention_indices, _short_conv_indices = (
        _authenticated_layer_stack_operator_topology(receipt, layer_values)
    )
    target_layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    if len(target_layers) != len(operator_layer_kinds):
        raise Lfm2SemanticPostfoldError(
            "Layer-stack receipt depth differs from the live folded child."
        )
    terminal_layer_index = len(operator_layer_kinds) - 1
    for layer_index, raw_layer in enumerate(layer_values):
        if not isinstance(raw_layer, Mapping):
            raise Lfm2SemanticPostfoldError(
                "Layer-stack producer coverage decision is not a mapping."
            )
        layer = _verify_sealed(
            raw_layer,
            seal_name="result_sha256",
            label=f"semantic layer {layer_index} transaction",
        )
        expected_operator_kind = operator_layer_kinds[layer_index]
        live_is_attention = getattr(
            target_layers[layer_index], "is_attention_layer", None
        )
        if (
            layer.get("layer_index") != layer_index
            or layer.get("operator_kind") != expected_operator_kind
            or not isinstance(live_is_attention, bool)
            or live_is_attention != (expected_operator_kind == "full_attention")
            or layer.get("operator_and_mlp_producer_chains_structurally_nonzero")
            is not True
        ):
            raise Lfm2SemanticPostfoldError(
                f"Layer {layer_index} structural producer receipt is inconsistent."
            )
        for lane, producer_kind, defer_down_weight in (
            ("operator", expected_operator_kind, False),
            ("mlp", "mlp", layer_index == terminal_layer_index),
        ):
            decision_value = layer.get(f"{lane}_producer_coverage_decision")
            if not isinstance(decision_value, Mapping):
                raise Lfm2SemanticPostfoldError(
                    f"Layer {layer_index} omitted {lane} producer coverage."
                )
            decision = _verify_sealed(
                decision_value,
                seal_name="result_sha256",
                label=f"layer {layer_index} {lane} producer coverage decision",
            )
            before_value = decision.get("coverage_before")
            after_value = decision.get("coverage_after")
            if not isinstance(before_value, Mapping) or not isinstance(
                after_value, Mapping
            ):
                raise Lfm2SemanticPostfoldError(
                    f"Layer {layer_index} {lane} coverage omitted before/after proof."
                )
            before_coverage = _verify_sealed(
                before_value,
                seal_name="result_sha256",
                label=f"layer {layer_index} {lane} pre-decision coverage",
            )
            after_coverage = _verify_sealed(
                after_value,
                seal_name="result_sha256",
                label=f"layer {layer_index} {lane} post-decision coverage",
            )
            installed = decision.get("installed_complete_structurally_nonzero_chain")
            retained = decision.get(
                "authenticated_nonzero_folded_seed_retained_by_noop"
            )
            expected_deferred = ["feed_forward.w2.weight"] if defer_down_weight else []
            if (
                decision.get("layer_index") != layer_index
                or decision.get("producer_kind") != producer_kind
                or decision.get("deferred_roles") != expected_deferred
                or decision.get("coverage_before_sha256")
                != before_coverage.get("result_sha256")
                or decision.get("coverage_after_sha256")
                != after_coverage.get("result_sha256")
                or decision.get(
                    "all_required_roles_structurally_nonzero_after_decision"
                )
                is not True
                or decision.get(
                    "blank_or_partially_blank_required_role_count_after_decision"
                )
                != 0
                or type(installed) is not bool
                or type(retained) is not bool
                or installed == retained
                or (
                    decision.get("cold_start_or_partial_seed_at_entry")
                    and not installed
                )
                or (
                    retained
                    and (
                        before_coverage.get("all_required_roles_structurally_nonzero")
                        is not True
                        or before_coverage.get("required_role_state_sha256")
                        != after_coverage.get("required_role_state_sha256")
                    )
                )
                or layer.get(f"{lane}_installed") is not installed
            ):
                raise Lfm2SemanticPostfoldError(
                    f"Layer {layer_index} {lane} coverage decision is internally false."
                )
            live = _lfm2_dense_producer_chain_coverage(
                target_model,
                layer_index=layer_index,
                producer_kind=producer_kind,
                defer_down_weight=defer_down_weight,
            )
            if live.get("result_sha256") != after_coverage.get("result_sha256"):
                raise Lfm2SemanticPostfoldError(
                    f"Layer {layer_index} live {lane} producer bytes differ from "
                    "the authenticated structural coverage decision."
                )


def compile_install_lfm2_semantic_layer_stack_stage(
    *,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    target_model: Any,
    interface: TiedCausalInterfaceFold,
    compiler_lineage: Mapping[int, Lfm2SemanticCompilerLayerLineage],
    parent_model_directory: str | os.PathLike[str],
    donor_tokenizer: Any,
    target_tokenizer: Any,
    attention_installer: Callable[..., Mapping[str, Any]] | None = None,
    semantic_layer_stack_compiler: Callable[..., Any] | None = None,
    fold_compute_device: str = "cpu",
    global_residual_closure_version: str = "producer_aware_v2a",
    correction_scale_frontier: Sequence[float] = (1.0,),
    correction_scale: float = 1.0,
    global_residual_backend: str = "auto",
    global_residual_basis_rank: int | None = None,
    authenticated_cache_session: Any | None = None,
    body_recipe: str | Lfm2SemanticBodyRecipe = "fit_guided",
    interface_anchor_selection: Mapping[str, Any] | None = None,
    mapped_parent_tied_table: Any | None = None,
    anchor_transport_witness: TiedResidualAnchorTransportWitness | None = None,
) -> Lfm2SemanticLayerStackInstallation:
    """Require causal operator->MLP layer order before terminal publication."""

    recipe = resolve_lfm2_semantic_body_recipe(body_recipe)
    closure_version = str(global_residual_closure_version).strip().lower()
    closure_policy = Lfm2GlobalResidualClosurePolicy(
        backend=str(global_residual_backend).strip().lower()
    )
    scale_policy = Lfm2ProducerAwareCorrectionScalePolicy(
        materialization_frontier=tuple(
            float(value) for value in correction_scale_frontier
        ),
        materialization_scale=float(correction_scale),
    )
    if global_residual_basis_rank is not None and (
        isinstance(global_residual_basis_rank, bool)
        or not isinstance(global_residual_basis_rank, int)
        or global_residual_basis_rank <= 0
    ):
        raise Lfm2SemanticPostfoldError(
            "global_residual_basis_rank must be a positive integer or None."
        )
    if closure_version not in {
        "producer_aware_v2a",
        "output_writer_v1_diagnostic",
    }:
        raise Lfm2SemanticPostfoldError(
            "global_residual_closure_version must be producer_aware_v2a or "
            "output_writer_v1_diagnostic."
        )
    anchor_selection = (
        None
        if interface_anchor_selection is None
        else _authenticate_foundation_anchor_selection(interface_anchor_selection)
    )
    anchor_interface_report = None
    anchor_transport_authentication = None
    if anchor_selection is not None:
        if not isinstance(interface, TiedCausalInterfaceFold):
            raise Lfm2SemanticPostfoldError(
                "Selected anchor body input is not a tied interface."
            )
        anchor_interface_report = _verify_sealed(
            _json_owned_receipt_value(interface.report),
            seal_name="report_sha256",
            label="selected anchor residual-aware body interface",
        )
        residual_aware = (
            anchor_interface_report.get("mode")
            == TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE
        )
        if (
            anchor_interface_report.get("schema") != TIED_RESIDUAL_CONGRUENCE_SCHEMA
            or anchor_interface_report.get("source_interface_report_sha256")
            != anchor_selection["selected_interface_report_sha256"]
            or (
                residual_aware
                and (
                    anchor_interface_report.get(
                        "anchor_base_target_weight_anchor_sha256"
                    )
                    != anchor_selection["anchor_base_target_weight_sha256"]
                    or anchor_interface_report.get(
                        "anchor_source_target_residual_anchor_sha256"
                    )
                    != anchor_selection["selected_residual_weight_sha256"]
                )
            )
            or (
                not residual_aware
                and anchor_interface_report.get("mode")
                != TIED_RESIDUAL_CURRENT_SELF_DUAL_MODE
            )
        ):
            raise Lfm2SemanticPostfoldError(
                "Selected anchor does not bind the residual-aware body interface."
            )
        if residual_aware:
            try:
                anchor_transport_authentication = (
                    authenticate_tied_residual_congruence_interface(
                        interface,
                        mapped_parent_tied_table=mapped_parent_tied_table,
                        anchor_transport_witness=anchor_transport_witness,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise Lfm2SemanticPostfoldError(
                    "Selected anchor residual transport failed exact algebraic "
                    "authentication before body folding."
                ) from exc
            if (
                anchor_transport_authentication.get(
                    "anchor_residual_formula_recomputed_exact"
                )
                is not True
            ):
                raise Lfm2SemanticPostfoldError(
                    "Selected anchor residual was not exactly transported into the "
                    "body interface."
                )
    compiler = _resolve_layer_stack_installer(semantic_layer_stack_compiler)
    if compiler is None:
        topology = cache.manifest.get("layer_topology", {}).get("layers", [])
        if not isinstance(topology, list):
            topology = []
        attention_count = sum(
            isinstance(value, Mapping)
            and isinstance(value.get("contract"), Mapping)
            and value["contract"].get("kind") == "full_attention"
            for value in topology
        )
        report = _sealed(
            {
                "schema": LFM2_SEMANTIC_POSTFOLD_FAILURE_SCHEMA,
                "status": "blocked-no-checkpoint-published",
                "stage": "semantic_layer_stack",
                "reason": "complete-causal-layer-stack-installer-unavailable",
                "blocking_reasons": [
                    "long-horizon-short-conv-refold-missing",
                    "replay-fresh-per-layer-mlp-refold-missing",
                    "attention-cannot-skip-intervening-conv-and-mlp-installs",
                ],
                "required_causal_layer_count": len(topology),
                "required_attention_operator_count": attention_count,
                "required_short_conv_operator_count": (len(topology) - attention_count),
                "required_replay_fresh_mlp_count": len(topology),
                "body_recipe": recipe.as_dict(),
                "body_recipe_sha256": recipe.recipe_sha256,
                "body_recipe_id": recipe.recipe_id,
                "missing_callable": (
                    "neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter."
                    + SEMANTIC_LAYER_STACK_INSTALLER_CALLABLE
                ),
                "available_partial_attention_callable": (
                    "neural_forge.hf_lfm2_semantic_attention_refold_adapter."
                    + SEMANTIC_ATTENTION_INSTALLER_CALLABLE
                ),
                "fused_terminal_refold_called": False,
                "checkpoint_save_called": False,
                "operation_audit": dict(_ZERO_OPERATIONS),
            },
            "failure_sha256",
        )
        raise Lfm2SemanticPostfoldBlockedError(
            "A complete causal operator/MLP layer-stack installer is required",
            report,
        )
    before = fingerprint_lfm2_parent(target_model).identity_sha256
    parameter_count = _parameter_count(target_model)
    final_w2_before = _tensor_sha256(_final_w2_parameter(target_model))
    result = compiler(
        cache=cache,
        bridge=bridge,
        target_model=target_model,
        interface=interface,
        compiler_lineage=compiler_lineage,
        parent_model_directory=parent_model_directory,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        attention_installer=_resolve_attention_installer(attention_installer),
        body_recipe=recipe,
        global_residual_closure_version=closure_version,
        correction_scale_frontier=scale_policy.materialization_frontier,
        correction_scale=scale_policy.materialization_scale,
        global_residual_backend=closure_policy.backend,
        **(
            {"global_residual_basis_rank": global_residual_basis_rank}
            if global_residual_basis_rank is not None
            else {}
        ),
        **(
            {"authenticated_cache_session": authenticated_cache_session}
            if authenticated_cache_session is not None
            else {}
        ),
        **(
            {"fold_compute_device": fold_compute_device}
            if fold_compute_device != "cpu"
            else {}
        ),
        **(
            {
                "mapped_parent_tied_table": mapped_parent_tied_table,
                "anchor_transport_witness": anchor_transport_witness,
            }
            if anchor_transport_authentication is not None
            else {}
        ),
    )
    if not isinstance(result, Lfm2SemanticLayerStackInstallation):
        raise Lfm2SemanticPostfoldError(
            "Semantic layer-stack compiler returned no typed deferred-W2 result."
        )
    verified = _verify_sealed(
        result.receipt,
        seal_name="result_sha256",
        label="semantic layer-stack installation",
    )
    after = fingerprint_lfm2_parent(target_model).identity_sha256
    final_w2_after = _tensor_sha256(_final_w2_parameter(target_model))
    global_residual_transition = _authenticate_semantic_global_residual_transition(
        verified,
        recipe,
    )
    expected_operator_layer_kinds = _operator_layer_kinds(cache, target_model)
    (
        receipt_operator_layer_kinds,
        attention_layer_indices,
        short_conv_layer_indices,
    ) = _authenticated_layer_stack_operator_topology(
        verified,
        verified.get("layer_installations"),
    )
    layer_count = len(expected_operator_layer_kinds)
    terminal_layer_index = layer_count - 1
    producer_chain_count = 2 * layer_count
    if (
        verified.get("body_recipe") != recipe.as_dict()
        or verified.get("body_recipe_sha256") != recipe.recipe_sha256
        or verified.get("body_recipe_id") != recipe.recipe_id
        or verified.get("complete_causal_layer_stack_adjudicated") is not True
        or (
            fold_compute_device != "cpu"
            and (
                verified.get("fold_compute_device") != fold_compute_device
                or verified.get("all_mlp_layers_proved_requested_fold_compute_device")
                is not True
                or verified.get(
                    "all_short_conv_layers_proved_requested_fold_compute_device"
                )
                is not True
            )
        )
        or receipt_operator_layer_kinds != expected_operator_layer_kinds
        or verified.get("all_attention_operator_decisions_completed") is not True
        or verified.get("attention_operator_decision_count")
        != len(attention_layer_indices)
        or verified.get("attention_operator_install_count", -1)
        + verified.get("attention_operator_incumbent_noop_count", -1)
        != len(attention_layer_indices)
        or verified.get("all_short_conv_operator_decisions_completed") is not True
        or verified.get("short_conv_operator_decision_count")
        != len(short_conv_layer_indices)
        or verified.get("short_conv_operator_install_count", -1)
        + verified.get("short_conv_operator_incumbent_noop_count", -1)
        != len(short_conv_layer_indices)
        or verified.get("all_replay_fresh_mlp_evidence_compiled") is not True
        or verified.get("replay_fresh_mlp_evidence_compile_count") != layer_count
        or verified.get("all_nonterminal_mlp_writer_decisions_completed") is not True
        or verified.get("nonterminal_mlp_writer_decision_count") != terminal_layer_index
        or verified.get("nonterminal_mlp_writer_install_count", -1)
        + verified.get("nonterminal_mlp_writer_incumbent_noop_count", -1)
        != terminal_layer_index
        or verified.get("final_mlp_producer_install_count", -1)
        + verified.get("final_mlp_producer_incumbent_noop_count", -1)
        != 1
        or verified.get("complete_nonzero_structural_producer_coverage") is not True
        or verified.get("all_producer_chains_installed_or_retained_nonzero_seed")
        is not True
        or verified.get("attention_producer_structural_coverage_count")
        != len(attention_layer_indices)
        or verified.get("short_conv_producer_structural_coverage_count")
        != len(short_conv_layer_indices)
        or verified.get("nonterminal_mlp_producer_structural_coverage_count")
        != terminal_layer_index
        or verified.get("final_mlp_producer_structural_coverage_count") != 1
        or verified.get("blank_or_partially_blank_producer_chain_count") != 0
        or verified.get("installed_complete_producer_chain_count", -1)
        + verified.get("authenticated_nonzero_folded_seed_retained_noop_count", -1)
        != producer_chain_count
        or verified.get("authenticated_nonzero_folded_seed_retained_noop_count", -1)
        != (
            verified.get("attention_operator_incumbent_noop_count", -2)
            + verified.get("short_conv_operator_incumbent_noop_count", -2)
            + verified.get("nonterminal_mlp_writer_incumbent_noop_count", -2)
            + verified.get("final_mlp_producer_incumbent_noop_count", -2)
        )
        or verified.get("whole_stage_contains_meaningful_parameter_changes") is not True
        or verified.get("operator_parameter_mutation_count", -1)
        != verified.get("attention_operator_install_count", -2)
        + verified.get("short_conv_operator_install_count", -2)
        or verified.get("mlp_parameter_mutation_count", -1)
        != verified.get("nonterminal_mlp_writer_install_count", -2)
        + verified.get("final_mlp_producer_install_count", -2)
        or verified.get("operator_parameter_mutation_count", 0)
        + verified.get("mlp_parameter_mutation_count", 0)
        <= 0
        or verified.get("final_layer_mlp_w2_deferred_to_fused_terminal") is not True
        or verified.get("final_layer_mlp_w2_install_count") != 0
        or verified.get("causal_install_order")
        != "operator_then_replay_fresh_mlp_then_next_layer"
        or verified.get("model_state_before_sha256") != before
        or verified.get("model_state_after_sha256") != after
        or before == after
        or not verified.get("changed_state_keys")
        or verified.get("parameter_element_count") != parameter_count
        or _parameter_count(target_model) != parameter_count
        or verified.get("parameter_count_changed") is not False
        or final_w2_after != final_w2_before
        or verified.get("deferred_final_w2_sha256") != final_w2_before
        or verified.get("operation_audit") != dict(_ZERO_OPERATIONS)
        or (
            global_residual_transition is not None
            and (
                global_residual_transition.get("global_residual_closure_version")
                != closure_version
                or global_residual_transition.get("global_residual_backend")
                != closure_policy.backend
                or global_residual_transition.get("correction_scale_frontier")
                != list(scale_policy.materialization_frontier)
                or global_residual_transition.get("correction_scale")
                != scale_policy.materialization_scale
            )
        )
    ):
        raise Lfm2SemanticPostfoldError(
            "Semantic layer-stack compiler returned an incomplete receipt."
        )
    _validate_live_layer_stack_producer_coverage(target_model, verified)
    authenticate_lfm2_semantic_layer_stack_attention_receipts(
        verified,
        body_recipe=recipe,
    )
    if anchor_selection is not None:
        tied_lineage = verified.get("tied_interface_lineage")
        if (
            "interface_anchor_selection" in verified
            or "interface_anchor_selection_sha256" in verified
            or not isinstance(tied_lineage, Mapping)
            or tied_lineage.get("interface_report_sha256")
            != anchor_interface_report["report_sha256"]
            or tied_lineage.get("target_weight_sha256")
            != anchor_interface_report["target_weight_sha256"]
            or tied_lineage.get("producer_encoder_sha256")
            != anchor_interface_report["producer_encoder_sha256"]
            or (
                anchor_transport_authentication is not None
                and tied_lineage.get("tied_residual_congruence_authentication_sha256")
                != anchor_transport_authentication["result_sha256"]
            )
        ):
            raise Lfm2SemanticPostfoldError(
                "Layer-stack compiler preempted or changed the runner-owned anchor "
                "interface binding."
            )
        augmented = dict(verified)
        augmented.pop("result_sha256")
        augmented.update(
            {
                "interface_anchor_selection": dict(anchor_selection),
                "interface_anchor_selection_sha256": anchor_selection[
                    "selection_sha256"
                ],
                "selected_anchor_candidate_index": anchor_selection[
                    "selected_anchor_candidate_index"
                ],
                "selected_anchor_candidate_recipe_sha256": anchor_selection[
                    "selected_anchor_candidate_recipe_sha256"
                ],
                "selected_anchor_target_weight_sha256": anchor_selection[
                    "selected_target_weight_sha256"
                ],
                "anchor_selection_bound_before_body_and_terminal_folding": True,
                "anchor_body_interface_report_sha256": anchor_interface_report[
                    "report_sha256"
                ],
                "anchor_body_target_weight_sha256": anchor_interface_report[
                    "target_weight_sha256"
                ],
                "anchor_body_producer_encoder_sha256": anchor_interface_report[
                    "producer_encoder_sha256"
                ],
                **(
                    {
                        "anchor_residual_transport_authentication": dict(
                            anchor_transport_authentication
                        ),
                        "anchor_residual_transport_authentication_sha256": (
                            anchor_transport_authentication["result_sha256"]
                        ),
                    }
                    if anchor_transport_authentication is not None
                    else {}
                ),
            }
        )
        verified = _sealed(augmented, "result_sha256")
    return Lfm2SemanticLayerStackInstallation(
        receipt=verified,
        final_layer_mlp_evidence=result.final_layer_mlp_evidence,
        final_layer_mlp_down_weight_prior=(result.final_layer_mlp_down_weight_prior),
    )


def compile_install_lfm2_semantic_attention_stage(
    *,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    target_model: Any,
    interface: TiedCausalInterfaceFold,
    compiler_lineage: Mapping[int, Lfm2SemanticCompilerLayerLineage],
    parent_model_directory: str | os.PathLike[str],
    donor_tokenizer: Any,
    target_tokenizer: Any,
    attention_installer: Callable[..., Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Audit and install every exact long-horizon semantic attention layer."""

    if not isinstance(cache, Lfm2CircuitTraceCache):
        raise TypeError("cache must be Lfm2CircuitTraceCache.")
    if not isinstance(bridge, StrictTokenizerBridgeArtifact):
        raise TypeError("bridge must be StrictTokenizerBridgeArtifact.")
    saved_tokenizer = getattr(
        target_model,
        "_neural_forge_quarantine_tokenizer",
        target_tokenizer,
    )
    if saved_tokenizer is not target_tokenizer and fingerprint_hf_tokenizer_semantics(
        saved_tokenizer
    ) != fingerprint_hf_tokenizer_semantics(target_tokenizer):
        raise Lfm2SemanticPostfoldError(
            "Saved child tokenizer differs from the semantic target tokenizer."
        )
    layers = _attention_layer_indices(cache, target_model)
    if set(compiler_lineage) != set(layers):
        raise Lfm2SemanticPostfoldError(
            "c61 compiler lineage does not cover every child attention layer."
        )
    inv_freq, attention_scaling = _target_rope(target_model)
    installer = _resolve_attention_installer(attention_installer)
    blocked_report = None
    layer_receipts = []
    state_before = fingerprint_lfm2_parent(target_model).identity_sha256
    parameter_count = _parameter_count(target_model)
    with cache.selective_authenticated_session() as session:
        alignments, tokenizer_session_receipt = _semantic_alignments_from_session(
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
        if not refinement["probability_preserving_attention_transport_go"]:
            blocked_report = {
                "reason": "semantic-span-attention-transport-not-exact",
                "blocking_reasons": list(refinement["blocking_reasons"]),
                "missing_callable": None,
            }
        elif installer is None:
            blocked_report = {
                "reason": "analytic-attention-weight-installer-unavailable",
                "blocking_reasons": [
                    "semantic-lattice-produces-teacher-tensors-but-no-qkvo-compiler"
                ],
                "missing_callable": (
                    "neural_forge.hf_lfm2_semantic_attention_refold_adapter."
                    + SEMANTIC_ATTENTION_INSTALLER_CALLABLE
                ),
            }
        else:
            parent_norm_epsilon = _local_lfm2_norm_epsilon(parent_model_directory)
            for layer_index in layers:
                _observed_inventory, parent_qk_tensors = (
                    _load_lfm2_parent_qkvo_from_local_safetensors(
                        parent_model_directory,
                        layer_index=layer_index,
                        expected_inventory=(
                            compiler_lineage[layer_index].source_inventory
                        ),
                    )
                )
                evidence = _transport_evidence_for_layer(
                    cache=cache,
                    session=session,
                    alignments=alignments,
                    layer_index=layer_index,
                    target_rope_inv_freq=inv_freq,
                    target_rope_attention_scaling=attention_scaling,
                    target_model=target_model,
                    bridge=bridge,
                    interface=interface,
                    compiler_lineage=compiler_lineage[layer_index],
                    parent_qk_tensors=parent_qk_tensors,
                    parent_norm_epsilon=parent_norm_epsilon,
                )
                before_hashes = _state_tensor_hashes(target_model)
                before = fingerprint_lfm2_parent(target_model).identity_sha256
                observed_count = _parameter_count(target_model)
                if observed_count != parameter_count:
                    raise Lfm2SemanticPostfoldError(
                        "Child parameter count changed before attention installation."
                    )
                candidate_receipt = installer(
                    target_model=target_model,
                    layer_index=layer_index,
                    evidence=evidence,
                    cache_key=cache.cache_key,
                    bridge_artifact_sha256=bridge.artifact_sha256,
                )
                after_hashes = _state_tensor_hashes(target_model)
                after = fingerprint_lfm2_parent(target_model).identity_sha256
                changed = _changed_state_keys(before_hashes, after_hashes)
                if _parameter_count(target_model) != parameter_count:
                    raise Lfm2SemanticPostfoldError(
                        "Attention installer changed child parameter count."
                    )
                layer_receipts.append(
                    _validate_attention_installation_receipt(
                        candidate_receipt,
                        layer_index=layer_index,
                        evidence=evidence,
                        before_sha256=before,
                        after_sha256=after,
                        changed_keys=changed,
                        parameter_count=parameter_count,
                    )
                )
        session.verify()
        session_receipt = session.as_dict()
    alignment_receipts = [value.to_evidence_dict() for value in alignments]
    common = {
        "schema": LFM2_SEMANTIC_ATTENTION_STAGE_SCHEMA,
        "cache_key": cache.cache_key,
        "cache_parent_fingerprint": cache.parent_fingerprint,
        "cache_selective_session_sha256": session_receipt["session_sha256"],
        "tokenizer_session_receipt_sha256": tokenizer_session_receipt["receipt_sha256"],
        "tokenizer_session": dict(tokenizer_session_receipt),
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "attention_layer_indices": list(layers),
        "semantic_alignment_sha256": [value.alignment_sha256 for value in alignments],
        "semantic_alignments": alignment_receipts,
        "refinement_audit": refinement,
        "target_rope_inv_freq_sha256": _tensor_sha256(inv_freq),
        "target_rope_attention_scaling": attention_scaling,
        "model_state_before_sha256": state_before,
        "parameter_element_count": parameter_count,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    if blocked_report is not None:
        failure = _sealed(
            {
                "schema": LFM2_SEMANTIC_POSTFOLD_FAILURE_SCHEMA,
                "status": "blocked-no-checkpoint-published",
                "stage": "semantic_attention",
                **blocked_report,
                "semantic_attention_audit": common,
                "terminal_refold_called": False,
                "checkpoint_save_called": False,
                "operation_audit": dict(_ZERO_OPERATIONS),
            },
            "failure_sha256",
        )
        missing = blocked_report["missing_callable"]
        detail = (
            f"; missing exact callable {missing}"
            if missing is not None
            else f"; blockers={blocked_report['blocking_reasons']}"
        )
        raise Lfm2SemanticPostfoldBlockedError(
            "Semantic attention postfold cannot publish" + detail,
            failure,
        )
    state_after = fingerprint_lfm2_parent(target_model).identity_sha256
    if (
        len(layer_receipts) != len(layers)
        or state_after == state_before
        or _parameter_count(target_model) != parameter_count
    ):
        raise Lfm2SemanticPostfoldError(
            "Semantic attention stage did not install every required layer."
        )
    return MappingProxyType(
        _sealed(
            {
                **common,
                "status": "installed",
                "model_state_after_sha256": state_after,
                "layer_installation_receipts": layer_receipts,
                "all_attention_layers_installed": True,
                "parameter_count_changed": False,
            },
            "result_sha256",
        )
    )


def _replay_lexical_rows(
    target_model: Any,
    *,
    token_ids: Sequence[int],
    position_origin: int,
    row_positions: Sequence[int],
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Replay bridge-known intermediate prefixes and expose final-W2 inputs."""

    torch = _torch()
    ids = torch.tensor([tuple(token_ids)], dtype=torch.long)
    mask = torch.ones_like(ids)
    positions = torch.arange(
        position_origin,
        position_origin + int(ids.shape[1]),
        dtype=torch.long,
    ).unsqueeze(0)
    selected = tuple(int(value) for value in row_positions)
    if (
        not selected
        or any(value < 0 or value >= int(ids.shape[1]) for value in selected)
        or len(set(selected)) != len(selected)
    ):
        raise Lfm2SemanticPostfoldError(
            "Lexical continuation replay rows are empty, repeated, or out of range."
        )
    embedding = getattr(target_model.get_input_embeddings(), "weight", None)
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    if (
        not isinstance(embedding, torch.nn.Parameter)
        or embedding.dtype != torch.bfloat16
        or not layers
        or bool((ids >= int(embedding.shape[0])).any().item())
    ):
        raise Lfm2SemanticPostfoldError(
            "Lexical continuation replay has no frozen dense BF16 child."
        )
    with torch.no_grad():
        device_ids = ids.to(device=embedding.device)
        residual = embedding.index_select(0, device_ids.reshape(-1)).reshape(
            1,
            int(ids.shape[1]),
            int(embedding.shape[1]),
        )
    replay_sha = []
    final_mlp = None
    final_pre = None
    for layer_index, layer in enumerate(layers):
        device, dtype = _layer_runtime(layer)
        residual = residual.to(device=device, dtype=dtype, non_blocking=False)
        if bool(getattr(layer, "is_attention_layer", False)):
            replay = replay_lfm2_cpcf_macrocell(
                layer,
                residual,
                position_embeddings=_rope_from_target(
                    target_model,
                    residual,
                    positions,
                ),
                additive_causal_mask=_additive_causal_mask(mask, residual),
            )
        else:
            replay = replay_lfm2_cpcf_macrocell(
                layer,
                residual,
                padding_mask=mask.to(device=device, dtype=dtype),
            )
        replay_sha.append(replay.replay_sha256)
        if layer_index == len(layers) - 1:
            index = torch.tensor(selected, dtype=torch.long)
            final_mlp = replay.mlp_input.index_select(1, index)[0]
            final_pre = replay.post_operator_residual.index_select(1, index)[0]
        residual = replay.post_layer_residual
    if final_mlp is None or final_pre is None:
        raise Lfm2SemanticPostfoldError(
            "Lexical continuation replay produced no final-W2 endpoints."
        )
    body = {
        "schema": "neural-forge-lfm2-semantic-lexical-replay-v1",
        "token_ids_sha256": canonical_sha256(list(token_ids)),
        "position_origin": position_origin,
        "row_positions": list(selected),
        "mlp_input_sha256": _tensor_sha256(final_mlp),
        "pre_mlp_residual_sha256": _tensor_sha256(final_pre),
        "macrocell_replay_sha256": replay_sha,
        "model_forward_calls": 0,
        "layer_forward_calls": 0,
        "training_steps": 0,
    }
    return final_mlp, final_pre, MappingProxyType(_sealed(body, "replay_sha256"))


def _append_only_lexical_supervision_coordinates(
    semantic: Any,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[str, ...], tuple[int, ...]]:
    """Return every bridge-inserted continuation query/target pair exactly once."""

    phases = tuple(semantic.phases)
    if len(phases) < 2:
        raise Lfm2SemanticPostfoldError(
            "Lexical span completion requires prefill plus decode phases."
        )
    batch_index = int(semantic.source_batch_index)
    previous = phases[0]
    row_positions = []
    desired = []
    row_ids = []
    multiplicities = []
    for phase in phases[1:]:
        replacement = tuple(int(value) for value in phase.target_replacement_token_ids)
        multiplicities.append(len(replacement))
        if (
            phase.target_replaced_suffix_token_count != 0
            or phase.target_rewrite_start != previous.target_full_prefix_length
            or not replacement
            or phase.target_full_prefix_length
            != previous.target_full_prefix_length + len(replacement)
        ):
            raise Lfm2SemanticPostfoldBlockedError(
                "Lexical completion currently requires append-only spans",
                _sealed(
                    {
                        "schema": LFM2_SEMANTIC_POSTFOLD_FAILURE_SCHEMA,
                        "status": "blocked-no-checkpoint-published",
                        "stage": "lexical_span_completion",
                        "reason": "non-append-only-semantic-transition",
                        "batch_index": batch_index,
                        "phase": phase.phase,
                        "operation_audit": dict(_ZERO_OPERATIONS),
                    },
                    "failure_sha256",
                ),
            )
        for offset in range(1, len(replacement)):
            row_positions.append(previous.target_full_prefix_length + offset - 1)
            desired.append(replacement[offset])
            row_ids.append(
                f"lexical:batch-{batch_index:06d}:{phase.phase}:subtoken-{offset:02d}"
            )
        previous = phase
    return (
        tuple(row_positions),
        tuple(desired),
        tuple(row_ids),
        tuple(multiplicities),
    )


def _lexical_teacher_rows(
    target_tied_weight: Any,
    desired_token_ids: Sequence[int],
) -> tuple[Any, Mapping[str, Any]]:
    """Select the best finite hidden teacher from a fixed closed-form frontier."""

    torch = _torch()
    if (
        not isinstance(target_tied_weight, torch.Tensor)
        or target_tied_weight.layout != torch.strided
        or target_tied_weight.device.type == "meta"
        or target_tied_weight.ndim != 2
        or int(target_tied_weight.shape[0]) <= 0
        or int(target_tied_weight.shape[1]) <= 0
        or not torch.is_floating_point(target_tied_weight)
    ):
        raise Lfm2SemanticPostfoldError(
            "Lexical teacher selection requires a materialized floating tied table."
        )
    table = target_tied_weight.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(table).all().item()):
        raise Lfm2SemanticPostfoldError(
            "Lexical teacher selection refuses a nonfinite tied table."
        )
    desired = tuple(int(value) for value in desired_token_ids)
    if not desired or min(desired) < 0 or max(desired) >= int(table.shape[0]):
        raise Lfm2SemanticPostfoldError(
            "Lexical desired token IDs exceed the installed tied table."
        )
    unique = tuple(sorted(set(desired)))
    indices = torch.tensor(unique, dtype=torch.long)
    rows = table.index_select(0, indices)
    gram = table.T @ table
    scale = float(gram.diagonal().mean().clamp_min(1e-30).item())
    candidates = [("tied_row", rows)]
    for ridge in (1e-8, 1e-6, 1e-4, 1e-2):
        inverse = torch.linalg.solve(
            gram + (ridge * scale) * torch.eye(int(gram.shape[0]), dtype=torch.float64),
            torch.eye(int(gram.shape[0]), dtype=torch.float64),
        )
        candidates.append((f"ridge_whitened_{ridge:.0e}", rows @ inverse))
    selected: dict[int, tuple[tuple[int, int, float, int], Any, str]] = {}
    finite_frontier_counts: dict[str, int] = {}
    vocabulary_size = int(table.shape[0])
    for candidate_index, (method, candidate64) in enumerate(candidates):
        candidate = candidate64.to(torch.bfloat16)
        finite_frontier_counts[method] = 0
        for start in range(0, len(unique), 32):
            stop = min(len(unique), start + 32)
            logits = candidate[start:stop].to(torch.float64) @ table.T
            winners = logits.argmax(dim=1)
            target_indices = torch.tensor(
                unique[start:stop],
                dtype=torch.long,
            )
            target_logits = logits.gather(1, target_indices[:, None]).squeeze(1)
            ranks = 1 + (logits > target_logits[:, None]).sum(dim=1)
            if vocabulary_size == 1:
                margins = torch.zeros_like(target_logits)
            else:
                competing = logits.clone()
                competing[
                    torch.arange(stop - start, dtype=torch.long),
                    target_indices,
                ] = -torch.inf
                margins = target_logits - competing.max(dim=1).values
            finite_rows = torch.isfinite(candidate[start:stop]).all(
                dim=1
            ) & torch.isfinite(logits).all(dim=1)
            finite_frontier_counts[method] += int(finite_rows.sum().item())
            for relative, winner in enumerate(winners.tolist()):
                if not bool(finite_rows[relative].item()):
                    continue
                offset = start + relative
                token = unique[offset]
                exact = token == winner
                rank = int(ranks[relative].item())
                margin = float(margins[relative].item())
                selection_key = (
                    0 if exact else 1,
                    rank,
                    -margin,
                    candidate_index,
                )
                incumbent = selected.get(token)
                if incumbent is None or selection_key < incumbent[0]:
                    selected[token] = (
                        selection_key,
                        candidate[offset].clone(),
                        method,
                    )
    missing_finite = tuple(token for token in unique if token not in selected)
    if missing_finite:
        raise Lfm2SemanticPostfoldError(
            "Closed-form lexical teacher frontier has no finite witness for "
            f"token IDs {list(missing_finite)}."
        )
    unique_teacher = torch.stack(tuple(selected[value][1] for value in unique), dim=0)
    unique_logits = unique_teacher.to(torch.float64) @ table.T
    if not bool(torch.isfinite(unique_teacher).all().item()) or not bool(
        torch.isfinite(unique_logits).all().item()
    ):
        raise Lfm2SemanticPostfoldError(
            "Selected lexical teacher frontier produced nonfinite output."
        )
    unique_indices = torch.tensor(unique, dtype=torch.long)
    unique_winners = unique_logits.argmax(dim=1)
    unique_targets = unique_logits.gather(1, unique_indices[:, None]).squeeze(1)
    unique_ranks = 1 + (unique_logits > unique_targets[:, None]).sum(dim=1)
    if vocabulary_size == 1:
        unique_margins = torch.zeros_like(unique_targets)
    else:
        unique_competing = unique_logits.clone()
        unique_competing[
            torch.arange(len(unique), dtype=torch.long),
            unique_indices,
        ] = -torch.inf
        unique_margins = unique_targets - unique_competing.max(dim=1).values
    selected_method = {token: selected[token][2] for token in unique}
    diagnostics = []
    for index, token in enumerate(unique):
        winner = int(unique_winners[index].item())
        diagnostics.append(
            {
                "target_token_id": token,
                "selected_method": selected_method[token],
                "winner_token_id": winner,
                "exact_target_top1": winner == token,
                "target_rank": int(unique_ranks[index].item()),
                "target_margin_over_best_competitor": float(
                    unique_margins[index].item()
                ),
            }
        )
    approximate = tuple(
        value["target_token_id"]
        for value in diagnostics
        if value["exact_target_top1"] is not True
    )
    exact = tuple(token for token in unique if token not in approximate)
    teacher = torch.stack(tuple(selected[value][1] for value in desired), dim=0)
    desired_exact_count = sum(value in exact for value in desired)
    methods = {
        name: sum(selected_method[value] == name for value in unique)
        for name, _candidate in candidates
    }
    body = {
        "schema": "neural-forge-lfm2-semantic-lexical-teacher-v1",
        "selection_contract": (
            "best-finite-closed-form-witness-exact-top1-preferred-v1"
        ),
        "desired_token_ids_sha256": canonical_sha256(list(desired)),
        "desired_row_count": len(desired),
        "unique_desired_token_count": len(unique),
        "teacher_rows_sha256": _tensor_sha256(teacher),
        "frontier": [name for name, _candidate in candidates],
        "finite_frontier_unique_token_counts": finite_frontier_counts,
        "selected_unique_token_counts": methods,
        "exact_unique_desired_token_count": len(exact),
        "approximate_unique_desired_token_count": len(approximate),
        "exact_desired_row_count": desired_exact_count,
        "approximate_desired_row_count": len(desired) - desired_exact_count,
        "exact_target_token_ids": list(exact),
        "missing_exact_top1_token_ids": list(approximate),
        "selected_unique_token_diagnostics": diagnostics,
        "maximum_selected_target_rank": max(
            int(value["target_rank"]) for value in diagnostics
        ),
        "minimum_selected_target_margin": min(
            float(value["target_margin_over_best_competitor"]) for value in diagnostics
        ),
        "teacher_rows_all_finite": True,
        "all_teacher_rows_exact_target_top1": not approximate,
        "gradient_search_used": False,
        "training_steps": 0,
    }
    return teacher, MappingProxyType(_sealed(body, "teacher_sha256"))


def _global_tokenizer_expansion_inventory(
    bridge: StrictTokenizerBridgeArtifact,
    donor_tokenizer: Any,
    target_tokenizer: Any,
) -> Mapping[str, Any]:
    """Audit global one-donor-token lexical expansion separately from traces."""

    donor_capacity = bridge.transport_profile.donor_row_capacity
    records = []
    multiplicities: dict[int, int] = {}
    exact_roundtrips = 0
    for donor_id in range(donor_capacity):
        text = _decode_exact(donor_tokenizer, (donor_id,), "donor")
        target_ids = _encode_exact(target_tokenizer, text, "target")
        multiplicity = len(target_ids)
        multiplicities[multiplicity] = multiplicities.get(multiplicity, 0) + 1
        roundtrip = _decode_exact(target_tokenizer, target_ids, "target") == text
        exact_roundtrips += int(roundtrip)
        records.append(
            {
                "donor_token_id": donor_id,
                "target_token_ids": list(target_ids),
                "exact_roundtrip": roundtrip,
            }
        )
    body = {
        "schema": "neural-forge-global-tokenizer-lexical-expansion-inventory-v1",
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "donor_tokenizer_semantics_sha256": (bridge.donor_tokenizer_semantics_sha256),
        "target_tokenizer_semantics_sha256": (bridge.target_tokenizer_semantics_sha256),
        "donor_token_count": donor_capacity,
        "exact_roundtrip_count": exact_roundtrips,
        "multiplicity_counts": {
            str(key): value for key, value in sorted(multiplicities.items())
        },
        "multi_token_expansion_count": sum(
            value for key, value in multiplicities.items() if key > 1
        ),
        "maximum_target_multiplicity": max(multiplicities),
        "mapping_sha256": canonical_sha256(records),
        "scope": "standalone_donor_token_text_not_contextual_trace_coverage",
        "training_steps": 0,
    }
    return MappingProxyType(_sealed(body, "inventory_sha256"))


def _convert_parent_terminal_evidence(
    evidence: Lfm2CpcfTerminalActionEvidence,
    producer_encoder: Any,
) -> Lfm2CpcfTerminalActionEvidence:
    torch = _torch()
    target = (
        evidence.parent_final_norm_output.to(torch.float64)
        @ producer_encoder.detach().to(device="cpu", dtype=torch.float64)
    ).to(torch.bfloat16)
    return Lfm2CpcfTerminalActionEvidence(
        split=evidence.split,
        row_ids=tuple("parent-endpoint:" + value for value in evidence.row_ids),
        mlp_input=evidence.mlp_input,
        pre_mlp_residual=evidence.pre_mlp_residual,
        parent_final_norm_output=target,
        sample_weights=evidence.sample_weights,
    )


def _merge_terminal_evidence(
    lanes: Mapping[str, Lfm2CpcfTerminalActionEvidence],
) -> Lfm2CpcfTerminalActionEvidence:
    torch = _torch()
    ordered = tuple((name, lanes[name]) for name in sorted(lanes))
    if len(ordered) < 2 or len({value.split for _name, value in ordered}) != 1:
        raise Lfm2SemanticPostfoldError(
            "Fused terminal evidence must contain aligned independent lanes."
        )
    mass = 1.0 / len(ordered)
    return Lfm2CpcfTerminalActionEvidence(
        split=ordered[0][1].split,
        row_ids=tuple(
            f"{name}:{row}" for name, value in ordered for row in value.row_ids
        ),
        mlp_input=torch.cat(
            tuple(value.mlp_input for _name, value in ordered),
            dim=0,
        ),
        pre_mlp_residual=torch.cat(
            tuple(value.pre_mlp_residual for _name, value in ordered),
            dim=0,
        ),
        parent_final_norm_output=torch.cat(
            tuple(value.parent_final_norm_output for _name, value in ordered),
            dim=0,
        ),
        sample_weights=torch.cat(
            tuple(value.sample_weights * mass for _name, value in ordered),
            dim=0,
        ),
    )


def _terminal_result_tensor_sha256(value: Any) -> str:
    """Independently reproduce the terminal-result tensor authentication."""

    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2SemanticPostfoldError(
            "Cannot authenticate a non-materialized terminal tensor."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
        "stride": [int(size) for size in tensor.stride()],
    }
    digest = hashlib.sha256(
        json.dumps(
            descriptor,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _reported_terminal_fit_key(record: Mapping[str, Any]) -> tuple[Any, ...] | None:
    metrics = record.get("fit_metrics")
    oracle = metrics.get("oracle") if isinstance(metrics, Mapping) else None
    parent = metrics.get("mapped_parent") if isinstance(metrics, Mapping) else None
    if not isinstance(oracle, Mapping) or not isinstance(parent, Mapping):
        return None

    def number(container: Mapping[str, Any], name: str) -> float | None:
        value = container.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        result = float(value)
        return result if math.isfinite(result) else None

    values = (
        number(oracle, "centered_logit_relative_rmse"),
        number(oracle, "top1_agreement_fraction"),
        number(parent, "centered_logit_relative_rmse"),
        number(parent, "top1_agreement_fraction"),
        number(parent, "reference_winner_positive_margin_fraction"),
        number(metrics, "maximum_post_layer_rms_ratio_to_incumbent"),
        number(record, "alpha"),
        number(record, "gauss_newton_damping"),
        number(record, "ridge"),
    )
    state_sha256 = record.get("state_sha256")
    if any(value is None for value in values) or not isinstance(state_sha256, str):
        return None
    return (
        values[0],
        -values[1],
        values[2],
        -values[3],
        -values[4],
        values[5],
        values[6],
        values[7],
        values[8],
        state_sha256,
    )


def _authenticated_exact_zero_terminal_plant(
    report: Mapping[str, Any],
    *,
    incumbent: Any,
    selected: Any,
    published: Any,
    policy: Lfm2CpcfTerminalActionPolicy,
    seed: Lfm2CpcfTerminalActionSeed | None,
    evidence: Mapping[str, Lfm2CpcfTerminalActionEvidence] | None,
) -> bool:
    """Recompute a cold publication from its exact local seed and evidence."""

    torch = _torch()
    required_splits = ("fit", "fit_check", "selection", "validation")
    if (
        int(torch.count_nonzero(incumbent).item()) != 0
        or not isinstance(policy, Lfm2CpcfTerminalActionPolicy)
        or not policy.verify_authentication()
        or not isinstance(seed, Lfm2CpcfTerminalActionSeed)
        or not seed.verify_authentication()
        or not isinstance(evidence, Mapping)
        or set(evidence) != set(required_splits)
        or not all(
            isinstance(evidence[split], Lfm2CpcfTerminalActionEvidence)
            and evidence[split].split == split
            and evidence[split].verify_authentication()
            for split in required_splits
        )
        or not torch.equal(seed.down_weight, incumbent)
    ):
        return False
    output_hashes = {
        "incumbent": _terminal_result_tensor_sha256(incumbent),
        "selected": _terminal_result_tensor_sha256(selected),
        "published": _terminal_result_tensor_sha256(published),
    }
    expected_evidence_sha256 = {
        split: evidence[split].authentication_sha256 for split in required_splits
    }
    expected_rows = {split: evidence[split].rows for split in required_splits}
    expected_row_ids_sha256 = {
        split: canonical_sha256(list(evidence[split].row_ids))
        for split in required_splits
    }
    coverage_value = report.get("selected_writer_structural_coverage")
    validation = report.get("validation")
    split_roles = report.get("split_roles")
    frontier = report.get("frontier")
    selected_metrics = report.get("selected_metrics_before_validation")
    incumbent_metrics = report.get("incumbent_metrics")
    if (
        not isinstance(coverage_value, Mapping)
        or not isinstance(validation, Mapping)
        or not isinstance(split_roles, Mapping)
        or not isinstance(frontier, list)
        or not isinstance(selected_metrics, Mapping)
        or not isinstance(incumbent_metrics, Mapping)
    ):
        return False
    coverage = dict(coverage_value)
    coverage_sha256 = coverage.pop("coverage_sha256", None)
    nonzero = selected.ne(0)
    all_finite = bool(torch.isfinite(selected).all().item())
    all_nonzero = bool(nonzero.all().item())
    all_outputs_covered = bool(nonzero.any(dim=1).all().item())
    all_sources_covered = bool(nonzero.any(dim=0).all().item())
    expected_count = 1 + (
        len(policy.gauss_newton_damping_frontier)
        * len(policy.ridge_frontier)
        * (len(policy.alpha_frontier) - 1)
    )
    expected_ids = {"incumbent"}
    expected_ids.update(
        f"gn-{damping_index:02d}-ridge-{ridge_index:02d}-alpha-{alpha_index:02d}"
        for damping_index, _damping in enumerate(policy.gauss_newton_damping_frontier)
        for ridge_index, _ridge in enumerate(policy.ridge_frontier)
        for alpha_index, _alpha in enumerate(policy.alpha_frontier[1:], start=1)
    )
    if not all(isinstance(value, Mapping) for value in frontier):
        return False
    records = {str(value.get("candidate_id")): value for value in frontier}
    selected_id = report.get("selected_candidate_id")
    selected_frontier = records.get(str(selected_id))
    incumbent_frontier = records.get("incumbent")
    if (
        len(frontier) != expected_count
        or len(records) != expected_count
        or set(records) != expected_ids
        or report.get("frontier_count") != expected_count
        or report.get("expected_frontier_count") != expected_count
        or not isinstance(selected_frontier, Mapping)
        or not isinstance(incumbent_frontier, Mapping)
    ):
        return False

    try:
        prepared = {
            split: _prepare_split(evidence[split], seed) for split in required_splits
        }
        recomputed_incumbent = {
            split: _evaluate_split(
                prepared[split],
                seed,
                incumbent,
                metric_chunk_rows=policy.metric_chunk_rows,
            )
            for split in required_splits
        }
        recomputed_selected = {
            split: _evaluate_split(
                prepared[split],
                seed,
                selected,
                metric_chunk_rows=policy.metric_chunk_rows,
            )
            for split in required_splits
        }
    except Exception:
        return False
    recomputed_fit_gates = _metric_gates(
        recomputed_selected["fit"],
        recomputed_incumbent["fit"],
        policy,
        require_selection_improvement=False,
    )
    recomputed_fit_check_gates = _metric_gates(
        recomputed_selected["fit_check"],
        recomputed_incumbent["fit_check"],
        policy,
        require_selection_improvement=False,
    )
    recomputed_selection_gates = _metric_gates(
        recomputed_selected["selection"],
        recomputed_incumbent["selection"],
        policy,
        require_selection_improvement=True,
    )
    recomputed_validation_gates = _metric_gates(
        recomputed_selected["validation"],
        recomputed_incumbent["validation"],
        policy,
        require_selection_improvement=False,
    )
    selected_fit_key = (
        recomputed_selected["fit"].oracle.centered_logit_relative_rmse,
        -recomputed_selected["fit"].oracle.top1_agreement_fraction,
        recomputed_selected["fit"].mapped_parent.centered_logit_relative_rmse,
        -recomputed_selected["fit"].mapped_parent.top1_agreement_fraction,
        -recomputed_selected[
            "fit"
        ].mapped_parent.reference_winner_positive_margin_fraction,
        recomputed_selected["fit"].maximum_post_layer_rms_ratio_to_incumbent,
        float(selected_frontier.get("alpha")),
        float(selected_frontier.get("gauss_newton_damping")),
        float(selected_frontier.get("ridge")),
        output_hashes["selected"],
    )
    incumbent_fit_key = (
        recomputed_incumbent["fit"].oracle.centered_logit_relative_rmse,
        -recomputed_incumbent["fit"].oracle.top1_agreement_fraction,
        recomputed_incumbent["fit"].mapped_parent.centered_logit_relative_rmse,
        -recomputed_incumbent["fit"].mapped_parent.top1_agreement_fraction,
        -recomputed_incumbent[
            "fit"
        ].mapped_parent.reference_winner_positive_margin_fraction,
        recomputed_incumbent["fit"].maximum_post_layer_rms_ratio_to_incumbent,
        float(incumbent_frontier.get("alpha")),
        float(incumbent_frontier.get("gauss_newton_damping")),
        float(incumbent_frontier.get("ridge")),
        output_hashes["incumbent"],
    )
    reported_keys = {
        candidate_id: _reported_terminal_fit_key(value)
        for candidate_id, value in records.items()
    }
    eligible_keys = []
    frontier_structural_authentication_valid = True
    structurally_eligible_fit_metric_gate_miss_count = 0
    for candidate_id, record in records.items():
        if candidate_id == "incumbent":
            continue
        key = reported_keys[candidate_id]
        candidate_coverage_value = record.get("structural_coverage")
        fit_gates = record.get("fit_gates")
        if (
            key is None
            or not isinstance(candidate_coverage_value, Mapping)
            or not isinstance(fit_gates, Mapping)
            or not fit_gates
            or not isinstance(record.get("state_sha256"), str)
            or any(type(value) is not bool for value in fit_gates.values())
        ):
            frontier_structural_authentication_valid = False
            continue
        candidate_coverage = dict(candidate_coverage_value)
        candidate_coverage_sha256 = candidate_coverage.pop(
            "coverage_sha256",
            None,
        )
        coverage_authenticated = bool(
            candidate_coverage_sha256 == canonical_sha256(candidate_coverage)
            and record.get("structural_coverage_sha256") == candidate_coverage_sha256
            and candidate_coverage_value.get("tensor_sha256")
            == record.get("state_sha256")
        )
        if not coverage_authenticated:
            frontier_structural_authentication_valid = False
            continue
        coverage_complete = bool(
            candidate_coverage_value.get("complete_dense_writer_coverage") is True
            and candidate_coverage_value.get("complete_dense_writer_structural_safety")
            is True
            and candidate_coverage_value.get("owns_exact_storage_allocation") is True
        )
        effectful_nonincumbent = (
            record.get("state_sha256") != output_hashes["incumbent"]
        )
        structurally_eligible = bool(
            coverage_authenticated and coverage_complete and effectful_nonincumbent
        )
        fit_metric_gates_passed = all(value is True for value in fit_gates.values())
        if (
            record.get("candidate_ranking_split") != "fit"
            or record.get("relative_to_zero_weight_delta_gate_applied") is not False
            or record.get("weight_norm_gate_passed") is not coverage_complete
            or record.get("effectful_nonincumbent") is not effectful_nonincumbent
            or record.get("structurally_eligible") is not structurally_eligible
            or record.get("structural_safety_passed") is not coverage_complete
            or record.get("eligible") is not structurally_eligible
            or record.get("candidate_eligibility_basis")
            != "finite_dense_shape_storage_coverage_and_nonincumbent_effect"
            or record.get("fit_metric_gates_passed") is not fit_metric_gates_passed
            or record.get("weight_delta_threshold_used_for_eligibility") is not False
            or record.get("fit_thresholds_used_for_eligibility") is not False
            or record.get("fit_check_thresholds_used_for_eligibility") is not False
            or record.get("selection_thresholds_used_for_eligibility") is not False
            or record.get("numeric_thresholds_used_for_publication") is not False
            or record.get("numeric_metrics_used_for_candidate_ordering") is not True
            or record.get("numeric_metrics_have_final_intelligence_authority")
            is not False
            or record.get("fit_metric_thresholds_used_for_candidate_eligibility")
            is not False
            or record.get("fit_metrics_used_for_candidate_ordering") is not True
            or record.get("fit_metrics_used_for_pre_review_candidate_selection")
            is not True
            or record.get("fit_metrics_used_for_structural_eligibility") is not False
            or record.get("fit_metric_thresholds_used_for_publication") is not False
            or record.get("fit_metrics_have_final_intelligence_authority") is not False
        ):
            frontier_structural_authentication_valid = False
            continue
        if structurally_eligible:
            eligible_keys.append((key, candidate_id))
            structurally_eligible_fit_metric_gate_miss_count += int(
                not fit_metric_gates_passed
            )
    unevaluated_marker = {"not_evaluated_before_fit_selection_freeze": True}
    nonselected_heldout_clean = all(
        value.get("fit_check_metrics") is None
        and value.get("selection_metrics") is None
        and value.get("fit_check_gates") == unevaluated_marker
        and value.get("selection_gates") == unevaluated_marker
        for candidate_id, value in records.items()
        if candidate_id not in {"incumbent", selected_id}
    )
    return bool(
        report.get("schema") == LFM2_CPCF_TERMINAL_ACTION_RESULT_SCHEMA
        and report.get("method") == _TERMINAL_ACTION_RESULT_METHOD
        and report.get("policy_sha256") == policy.authentication_sha256
        and report.get("seed_sha256") == seed.authentication_sha256
        and report.get("incumbent_seed_writer_sha256") == output_hashes["incumbent"]
        and report.get("split_evidence_sha256") == expected_evidence_sha256
        and report.get("split_rows") == expected_rows
        and report.get("split_row_ids_sha256") == expected_row_ids_sha256
        and report.get("output_state_sha256") == output_hashes
        and report.get("operation_audit") == dict(_TERMINAL_ZERO_OPERATION_AUDIT)
        and report.get("exact_zero_cold_start") is True
        and report.get("incumbent_all_elements_zero") is True
        and report.get("incumbent_nonzero_element_count") == 0
        and report.get("incumbent_noop_publication_forbidden") is True
        and report.get("exact_zero_candidate_eligibility_schema")
        == LFM2_CPCF_TERMINAL_ACTION_EXACT_ZERO_ELIGIBILITY_SCHEMA
        and report.get("structural_nonveto_schema")
        == LFM2_CPCF_TERMINAL_ACTION_STRUCTURAL_NONVETO_SCHEMA
        and report.get("candidate_selection_evidence_scope")
        == "finite-dense-structural-eligibility-plus-fit-ordering"
        and report.get("candidate_eligibility_basis")
        == "finite_dense_shape_storage_coverage_and_nonincumbent_effect"
        and report.get("structural_safety_is_only_candidate_veto") is True
        and report.get("numeric_metrics_used_for_candidate_ordering") is True
        and report.get("numeric_metrics_used_for_candidate_eligibility") is False
        and report.get("numeric_thresholds_used_for_candidate_eligibility") is False
        and report.get("numeric_thresholds_used_for_publication") is False
        and report.get("weight_delta_threshold_used_for_candidate_eligibility") is False
        and report.get("fit_thresholds_used_for_candidate_eligibility") is False
        and report.get("fit_check_thresholds_used_for_candidate_eligibility") is False
        and report.get("selection_thresholds_used_for_candidate_eligibility") is False
        and report.get("validation_thresholds_used_for_publication") is False
        and report.get("validation_publication_veto_applied") is False
        and report.get("rollback_authorized") is False
        and report.get("numeric_metrics_have_final_intelligence_authority") is False
        and report.get("behavioral_intelligence_review_outside_terminal_compiler")
        is True
        and report.get("fit_metrics_role")
        == "ordering-among-structurally-eligible-not-threshold-gating"
        and report.get("fit_metrics_used_for_candidate_ordering") is True
        and report.get("fit_metrics_used_for_pre_review_candidate_selection") is True
        and report.get("fit_metrics_used_for_structural_eligibility") is False
        and report.get("fit_metrics_have_final_intelligence_authority") is False
        and report.get("fit_metric_thresholds_used_for_candidate_eligibility") is False
        and report.get("fit_metric_thresholds_used_for_publication") is False
        and report.get("fit_metric_gate_miss_can_exclude_structural_candidate") is False
        and report.get("structurally_eligible_frontier_count") == len(eligible_keys)
        and report.get("structurally_eligible_fit_metric_gate_miss_count")
        == structurally_eligible_fit_metric_gate_miss_count
        and report.get("relative_to_zero_weight_delta_gate_applied") is False
        and report.get("heldout_evidence_role")
        == "diagnostic-only-after-frozen-fit-selection"
        and report.get("selected_before_any_heldout_publication_diagnostic") is True
        and report.get("heldout_candidate_evaluation_count_before_fit_freeze") == 0
        and report.get("heldout_candidate_evaluation_count_after_fit_freeze") == 6
        and report.get("heldout_candidate_count_evaluated_after_fit_freeze") == 2
        and report.get("selected_nonincumbent") is (selected_id != "incumbent")
        and selected_id != "incumbent"
        and report.get("accepted") is True
        and report.get("published_is_incumbent") is False
        and report.get("rollback_performed") is False
        and report.get("rollback_exact") is True
        and report.get("source_seed_writer_unchanged") is True
        and report.get("parameter_identity_or_topology_mutation") is False
        and report.get("parameter_count_before") == int(incumbent.numel())
        and report.get("parameter_count_after") == int(published.numel())
        and torch.equal(published, selected)
        and not torch.equal(published, incumbent)
        and coverage_sha256 == canonical_sha256(coverage)
        and report.get("selected_writer_structural_coverage_sha256") == coverage_sha256
        and coverage.get("tensor_sha256") == output_hashes["selected"]
        and coverage.get("shape") == [int(size) for size in selected.shape]
        and coverage.get("dtype") == "bfloat16"
        and coverage.get("element_count") == int(selected.numel())
        and coverage.get("all_elements_finite") is all_finite
        and coverage.get("all_elements_nonzero") is all_nonzero
        and coverage.get("all_target_outputs_covered") is all_outputs_covered
        and coverage.get("all_source_features_covered") is all_sources_covered
        and coverage.get("complete_dense_writer_coverage") is True
        and coverage.get("complete_dense_writer_structural_safety") is True
        and coverage.get("owns_exact_storage_allocation") is True
        and all_finite
        and int(torch.count_nonzero(selected).item()) > 0
        and all_outputs_covered
        and all_sources_covered
        and selected_frontier.get("state_sha256") == output_hashes["selected"]
        and incumbent_frontier.get("state_sha256") == output_hashes["incumbent"]
        and selected_frontier.get("structural_coverage") == coverage_value
        and selected_frontier.get("structural_coverage_sha256") == coverage_sha256
        and selected_frontier.get("candidate_ranking_split") == "fit"
        and selected_frontier.get("relative_to_zero_weight_delta_gate_applied") is False
        and selected_frontier.get("fit_metrics") == recomputed_selected["fit"].as_dict()
        and selected_frontier.get("fit_gates") == recomputed_fit_gates
        and selected_frontier.get("fit_check_metrics")
        == recomputed_selected["fit_check"].as_dict()
        and selected_frontier.get("fit_check_gates") == recomputed_fit_check_gates
        and selected_frontier.get("selection_metrics")
        == recomputed_selected["selection"].as_dict()
        and selected_frontier.get("selection_gates") == recomputed_selection_gates
        and selected_frontier.get(
            "heldout_diagnostics_evaluated_after_fit_selection_freeze"
        )
        is True
        and selected_frontier.get("frozen_candidate_state_sha256")
        == output_hashes["selected"]
        and tuple(selected_frontier.get("selection_key", ())) == selected_fit_key
        and tuple(incumbent_frontier.get("selection_key", ())) == incumbent_fit_key
        and tuple(report.get("selected_selection_key", ())) == selected_fit_key
        and report.get("selected_state_sha256") == output_hashes["selected"]
        and all(
            key is not None
            and tuple(records[candidate_id].get("selection_key", ())) == key
            for candidate_id, key in reported_keys.items()
        )
        and frontier_structural_authentication_valid
        and selected_frontier.get("eligible") is True
        and bool(eligible_keys)
        and min(eligible_keys)[1] == selected_id
        and nonselected_heldout_clean
        and incumbent_metrics
        == {split: recomputed_incumbent[split].as_dict() for split in required_splits}
        and selected_metrics
        == {
            split: recomputed_selected[split].as_dict()
            for split in ("fit", "fit_check", "selection")
        }
        and validation.get("metrics") == recomputed_selected["validation"].as_dict()
        and validation.get("gates") == recomputed_validation_gates
        and validation.get("passed") is _all_passed(recomputed_validation_gates)
        and validation.get("heldout_diagnostic_only") is True
        and validation.get("publication_gate_applied") is False
        and validation.get("used_only_for_frozen_candidate_acceptance") is False
        and validation.get("used_for_candidate_ranking") is False
        and validation.get("used_to_change_selected_candidate_id") is False
        and validation.get("accessed_only_after_selected_id_and_state_sha256_frozen")
        is True
        and validation.get("frozen_candidate_state_sha256") == output_hashes["selected"]
        and split_roles.get("candidate_eligibility_uses_only_finite_dense_structure")
        is True
        and split_roles.get("candidate_eligibility_uses_only_structural_safety") is True
        and split_roles.get("fit_used_for_candidate_eligibility") is False
        and split_roles.get("only_fit_rows_used_for_candidate_ordering") is True
        and split_roles.get("fit_metrics_used_for_pre_review_candidate_selection")
        is True
        and split_roles.get("fit_metrics_have_final_intelligence_authority") is False
        and split_roles.get("fit_metric_thresholds_used_for_publication") is False
        and split_roles.get("fit_check_diagnostic_only") is True
        and split_roles.get("selection_diagnostic_only") is True
        and split_roles.get("validation_diagnostic_only") is True
        and report.get("selected_writer_all_elements_finite") is True
        and report.get("selected_writer_all_elements_nonzero") is all_nonzero
        and report.get("selected_writer_all_source_features_covered") is True
        and report.get("published_writer_is_structurally_safe_effectful_nonincumbent")
        is True
        and report.get("published_writer_is_finite_dense_nonzero_structural_plant")
        is True
        and report.get("nonfinite_candidate_or_replay_is_hard_failure") is True
        and report.get("zero_training_pruning_quantization") is True
    )


def _terminal_publication_decision(
    terminal: Lfm2CpcfTerminalActionResidualResult,
    *,
    expected_seed: Lfm2CpcfTerminalActionSeed | None = None,
    expected_evidence: Mapping[str, Lfm2CpcfTerminalActionEvidence] | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    """Authenticate an externally reviewed terminal publication."""

    torch = _torch()
    if not isinstance(terminal, Lfm2CpcfTerminalActionResidualResult):
        raise TypeError("terminal must be a typed terminal-action result.")
    if not terminal.verify_authentication():
        raise Lfm2SemanticPostfoldError(
            "Fused terminal result changed after authentication."
        )
    report = terminal.as_dict()
    if (
        report.get("final_selection_status")
        == "deferred-to-manual-whole-response-review"
        and report.get("final_selection_performed") is False
        and report.get("accepted") is False
        and report.get("published_is_incumbent") is True
    ):
        raise Lfm2SemanticPostfoldError(
            "Terminal frontier is review-pending; no metric-ordered candidate may "
            "be installed before manual whole-response review."
        )
    operation_audit = report.get("operation_audit")
    if not isinstance(operation_audit, Mapping) or operation_audit != dict(
        _TERMINAL_ZERO_OPERATION_AUDIT
    ):
        raise Lfm2SemanticPostfoldError(
            "Fused terminal result does not prove a zero-operation compilation."
        )
    published = terminal.published_down_weight
    incumbent = terminal.incumbent_down_weight
    selected = terminal.selected_down_weight
    terminal_action_selected_sha256 = _terminal_action_tensor_sha256(
        selected,
        "terminal.selected_down_weight",
    )
    terminal_action_published_sha256 = _terminal_action_tensor_sha256(
        published,
        "terminal.published_down_weight",
    )
    semantic_selected_sha256 = _tensor_sha256(selected)
    semantic_published_sha256 = _tensor_sha256(published)
    semantic_incumbent_sha256 = _tensor_sha256(incumbent)
    published_is_incumbent = bool(torch.equal(published, incumbent))
    published_is_selected = bool(torch.equal(published, selected))
    incumbent_is_exact_zero = bool(torch.count_nonzero(incumbent).item() == 0)
    validation = report.get("validation")
    validation_diagnostics_passed = bool(
        isinstance(validation, Mapping) and validation.get("passed") is True
    )
    common_valid = bool(
        report.get("source_seed_writer_unchanged") is True
        and report.get("parameter_identity_or_topology_mutation") is False
        and report.get("parameter_count_before") == report.get("parameter_count_after")
        and report.get("rollback_exact") is True
        and report.get("structural_nonveto_schema")
        == LFM2_CPCF_TERMINAL_ACTION_STRUCTURAL_NONVETO_SCHEMA
        and report.get("candidate_eligibility_basis")
        == "finite_dense_shape_storage_coverage_and_nonincumbent_effect"
        and report.get("structural_safety_is_only_candidate_veto") is True
        and report.get("numeric_metrics_used_for_candidate_ordering") is True
        and report.get("numeric_metrics_used_for_candidate_eligibility") is False
        and report.get("numeric_thresholds_used_for_candidate_eligibility") is False
        and report.get("numeric_thresholds_used_for_publication") is False
        and report.get("weight_delta_threshold_used_for_candidate_eligibility") is False
        and report.get("fit_thresholds_used_for_candidate_eligibility") is False
        and report.get("fit_check_thresholds_used_for_candidate_eligibility") is False
        and report.get("selection_thresholds_used_for_candidate_eligibility") is False
        and report.get("validation_thresholds_used_for_publication") is False
        and report.get("validation_publication_veto_applied") is False
        and report.get("rollback_authorized") is False
        and report.get("numeric_metrics_have_final_intelligence_authority") is False
        and report.get("behavioral_intelligence_review_outside_terminal_compiler")
        is True
        and report.get("published_writer_is_structurally_safe_effectful_nonincumbent")
        is True
        and isinstance(validation, Mapping)
        and validation.get("heldout_diagnostic_only") is True
        and validation.get("publication_gate_applied") is False
        and validation.get("used_only_for_frozen_candidate_acceptance") is False
    )
    exact_zero_cold_start = _authenticated_exact_zero_terminal_plant(
        report,
        incumbent=incumbent,
        selected=selected,
        published=published,
        policy=terminal.policy,
        seed=expected_seed,
        evidence=expected_evidence,
    )
    changed_winner = bool(
        common_valid
        and terminal.accepted
        and terminal.changed
        and terminal.selected_candidate_id != "incumbent"
        and report.get("selected_nonincumbent") is True
        and report.get("published_is_incumbent") is False
        and report.get("rollback_performed") is False
        and published_is_selected
        and not published_is_incumbent
        and (not incumbent_is_exact_zero or exact_zero_cold_start)
    )
    if not changed_winner:
        raise Lfm2SemanticPostfoldError(
            "Fused terminal published neither an authenticated structural-nonveto "
            "changed winner nor a valid cold-start plant."
        )
    reason = (
        "exact-zero-structural-fit-ordered-plant"
        if exact_zero_cold_start
        else "accepted-nonincumbent-mutation"
    )
    decision = {
        "publication_decision": reason,
        "terminal_publication_decision_accepted": True,
        "terminal_nonincumbent_candidate_accepted": True,
        "terminal_incumbent_retained": False,
        "terminal_writer_changed": True,
        "structural_candidate_eligibility_authenticated": True,
        "heldout_metrics_diagnostic_only": True,
        "validation_diagnostics_passed": validation_diagnostics_passed,
        "validation_publication_gate_applied": False,
        "numeric_metrics_have_final_intelligence_authority": False,
        "terminal_action_result_sha256": terminal.result_sha256,
        "terminal_action_selected_w2_sha256": terminal_action_selected_sha256,
        "terminal_action_published_w2_sha256": terminal_action_published_sha256,
        "semantic_selected_w2_sha256": semantic_selected_sha256,
        "semantic_published_w2_sha256": semantic_published_sha256,
        "semantic_incumbent_w2_sha256": semantic_incumbent_sha256,
        "published_w2_sha256": semantic_published_sha256,
        "selected_w2_sha256": semantic_selected_sha256,
        "incumbent_w2_sha256": semantic_incumbent_sha256,
        "published_is_selected": published_is_selected,
        "published_is_incumbent": published_is_incumbent,
    }
    if exact_zero_cold_start:
        decision.update(
            {
                "terminal_exact_zero_cold_start": True,
                "incumbent_noop_publication_forbidden": True,
                "structural_candidate_eligibility_authenticated": True,
                "fit_metric_ordering_frozen_before_heldout_diagnostics": True,
                "fit_metrics_used_for_candidate_ordering": True,
                "fit_metrics_used_for_pre_review_candidate_selection": True,
                "fit_metrics_used_for_structural_eligibility": False,
                "fit_metric_thresholds_used_for_candidate_eligibility": False,
                "fit_metric_thresholds_used_for_publication": False,
                "fit_metrics_have_final_intelligence_authority": False,
                "heldout_metrics_diagnostic_only": True,
                "published_w2_complete_dense_structural_plant": True,
            }
        )
    return published, MappingProxyType(decision)


def _install_final_w2_once(
    target_model: Any,
    *,
    published_down_weight: Any,
    terminal_publication_decision: Mapping[str, Any],
) -> Mapping[str, Any]:
    torch = _torch()
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    parameter = getattr(
        getattr(getattr(layers[-1], "feed_forward", None), "w2", None),
        "weight",
        None,
    )
    if not isinstance(parameter, torch.nn.Parameter):
        raise Lfm2SemanticPostfoldError("Child final W2 is not a Parameter.")
    if (
        not isinstance(published_down_weight, torch.Tensor)
        or published_down_weight.layout != torch.strided
        or published_down_weight.device.type == "meta"
        or tuple(published_down_weight.shape) != tuple(parameter.shape)
        or published_down_weight.requires_grad
        or published_down_weight.grad_fn is not None
        or not bool(torch.isfinite(published_down_weight).all().item())
    ):
        raise Lfm2SemanticPostfoldError(
            "Published final W2 is not a finite detached dense topology match."
        )
    if not isinstance(terminal_publication_decision, Mapping):
        raise Lfm2SemanticPostfoldError(
            "Final W2 installation omitted its terminal publication decision."
        )
    terminal_action_result_sha256 = _digest(
        terminal_publication_decision.get("terminal_action_result_sha256"),
        "terminal-action result",
    )
    terminal_action_selected_sha256 = _digest(
        terminal_publication_decision.get("terminal_action_selected_w2_sha256"),
        "terminal-action selected W2",
    )
    terminal_action_published_sha256 = _digest(
        terminal_publication_decision.get("terminal_action_published_w2_sha256"),
        "terminal-action published W2",
    )
    semantic_selected_sha256 = _digest(
        terminal_publication_decision.get("semantic_selected_w2_sha256"),
        "semantic selected W2",
    )
    semantic_published_sha256 = _digest(
        terminal_publication_decision.get("semantic_published_w2_sha256"),
        "semantic published W2",
    )
    observed_semantic_published_sha256 = _tensor_sha256(published_down_weight)
    observed_terminal_action_published_sha256 = _terminal_action_tensor_sha256(
        published_down_weight,
        "published_down_weight",
    )
    if (
        terminal_action_selected_sha256 != terminal_action_published_sha256
        or terminal_action_published_sha256 != observed_terminal_action_published_sha256
        or semantic_selected_sha256 != semantic_published_sha256
        or semantic_published_sha256 != observed_semantic_published_sha256
        or terminal_publication_decision.get("selected_w2_sha256")
        != semantic_selected_sha256
        or terminal_publication_decision.get("published_w2_sha256")
        != semantic_published_sha256
    ):
        raise Lfm2SemanticPostfoldError(
            "Final W2 publication decision is not bound to the installed writer."
        )
    parameter_name = f"model.layers.{len(layers) - 1}.feed_forward.w2.weight"
    before_state = fingerprint_lfm2_parent(target_model).identity_sha256
    before_hashes = _state_tensor_hashes(target_model)
    before_count = _parameter_count(target_model)
    before_runtime = _parameter_runtime_snapshot(target_model)
    incumbent = parameter.detach().to(device="cpu").contiguous().clone()
    incumbent_is_exact_zero = bool(torch.count_nonzero(incumbent).item() == 0)
    published = (
        published_down_weight.detach()
        .to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        .contiguous()
    )
    published_coverage = _dense_weight_structural_coverage(
        published,
        parameter_name=parameter_name,
        require_parameter=False,
    )
    if published_coverage.get("structurally_nonzero") is not True:
        raise Lfm2SemanticPostfoldError(
            "Fused terminal refused to publish a blank or structurally partial W2."
        )
    install = not torch.equal(parameter.detach(), published)
    if incumbent_is_exact_zero and not install:
        raise Lfm2SemanticPostfoldError(
            "Fused terminal forbids incumbent/noop publication for an exact-zero "
            "W2 cold start."
        )
    try:
        if install:
            with torch.no_grad():
                parameter.copy_(published)
        after_state = fingerprint_lfm2_parent(target_model).identity_sha256
        after_hashes = _state_tensor_hashes(target_model)
        after_runtime = _parameter_runtime_snapshot(target_model)
        runtime_proof = _parameter_runtime_proof(before_runtime, after_runtime)
        changed_state_keys = _changed_state_keys(before_hashes, after_hashes)
        runtime_preserved = all(
            runtime_proof[name] is True
            for name in (
                "parameter_topology_preserved",
                "parameter_identities_preserved",
                "parameter_storage_preserved",
                "gradient_state_preserved",
            )
        )
        mutation_valid = bool(
            install
            and after_state != before_state
            and changed_state_keys == (parameter_name,)
        )
        noop_valid = bool(
            not install and after_state == before_state and not changed_state_keys
        )
        if (
            not runtime_preserved
            or _parameter_count(target_model) != before_count
            or not torch.equal(parameter.detach(), published)
            or _tensor_sha256(parameter) != semantic_published_sha256
            or _terminal_action_tensor_sha256(
                parameter,
                "installed_final_w2",
            )
            != terminal_action_published_sha256
            or not (mutation_valid or noop_valid)
        ):
            raise Lfm2SemanticPostfoldError(
                "Fused terminal publication changed topology, identity, storage, "
                "gradients, or unexpected state bytes."
            )
    except BaseException:
        with torch.no_grad():
            parameter.copy_(incumbent.to(device=parameter.device))
        rollback_runtime = _parameter_runtime_proof(
            before_runtime,
            _parameter_runtime_snapshot(target_model),
        )
        if (
            fingerprint_lfm2_parent(target_model).identity_sha256 != before_state
            or _state_tensor_hashes(target_model) != before_hashes
            or not all(
                rollback_runtime[name] is True
                for name in (
                    "parameter_topology_preserved",
                    "parameter_identities_preserved",
                    "parameter_storage_preserved",
                    "gradient_state_preserved",
                )
            )
        ):
            raise Lfm2SemanticPostfoldError("Fused terminal rollback was not exact.")
        raise
    body = {
        "schema": "neural-forge-lfm2-fused-semantic-terminal-install-v1",
        "status": (
            "installed-accepted-terminal-writer"
            if install
            else "authenticated-published-incumbent-noop"
        ),
        "parameter_name": parameter_name,
        "model_state_before_sha256": before_state,
        "model_state_after_sha256": after_state,
        "terminal_action_result_sha256": terminal_action_result_sha256,
        "terminal_action_selected_w2_sha256": terminal_action_selected_sha256,
        "terminal_action_published_w2_sha256": terminal_action_published_sha256,
        "semantic_selected_w2_sha256": semantic_selected_sha256,
        "semantic_published_w2_sha256": semantic_published_sha256,
        "incumbent_w2_sha256": _tensor_sha256(incumbent),
        "published_w2_sha256": _tensor_sha256(parameter),
        "published_w2_structural_coverage": dict(published_coverage),
        "published_w2_structural_coverage_sha256": published_coverage["result_sha256"],
        "published_w2_structurally_nonzero": True,
        "installed_w2_sha256": _tensor_sha256(parameter) if install else None,
        "changed_state_keys": list(changed_state_keys),
        **dict(runtime_proof),
        "parameter_identity_preserved": runtime_proof["parameter_identities_preserved"],
        "parameter_count_changed": False,
        "decision_count": 1,
        "installed": install,
        "install_count": int(install),
        "mutation_count": int(install),
        "incumbent_retention_count": int(not install),
        "exact_zero_copy_noop": not install,
        "parameter_copy_calls": int(install),
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    if incumbent_is_exact_zero:
        body.update(
            {
                "exact_zero_cold_start": True,
                "incumbent_all_elements_zero": True,
                "incumbent_noop_publication_forbidden": True,
                "published_w2_all_elements_finite": True,
                "published_w2_all_elements_nonzero": bool(published.ne(0).all().item()),
                "published_w2_all_output_units_covered": True,
                "published_w2_all_input_units_covered": True,
                "transactional_dense_cold_start_plant": True,
            }
        )
    return MappingProxyType(_sealed(body, "result_sha256"))


def _authenticate_fused_terminal_tied_interface(
    *,
    target_model: Any,
    interface: TiedCausalInterfaceFold,
    mapped_parent_tied_table: Any,
    layer_receipt: Mapping[str, Any],
    anchor_transport_witness: TiedResidualAnchorTransportWitness | None = None,
) -> Mapping[str, Any]:
    """Bind terminal evidence to the live, layer-authenticated congruence."""

    torch = _torch()
    if not isinstance(interface, TiedCausalInterfaceFold):
        raise TypeError("interface must be TiedCausalInterfaceFold.")
    if interface.report.get("schema") != TIED_RESIDUAL_CONGRUENCE_SCHEMA:
        raise Lfm2SemanticPostfoldError(
            "Fused terminal requires an explicit tied residual-congruence schema."
        )
    try:
        authentication = authenticate_tied_residual_congruence_interface(
            interface,
            mapped_parent_tied_table=mapped_parent_tied_table,
            anchor_transport_witness=anchor_transport_witness,
        )
    except (TypeError, ValueError) as exc:
        raise Lfm2SemanticPostfoldError(
            "Fused terminal rejected the tied residual-congruence lineage."
        ) from exc
    lineage = layer_receipt.get("tied_interface_lineage")
    residual_transport = (
        authentication.get("strong_base_plus_anchor_residual_claimed") is True
    )
    publication_authentication = layer_receipt.get(
        "anchor_residual_transport_authentication"
    )
    publication_authentication_sha256 = layer_receipt.get(
        "anchor_residual_transport_authentication_sha256"
    )
    if residual_transport and (
        not isinstance(lineage, Mapping)
        or not isinstance(publication_authentication, Mapping)
        or dict(publication_authentication) != dict(authentication)
        or publication_authentication_sha256 != authentication["result_sha256"]
        or lineage.get("tied_residual_congruence_authentication_sha256")
        != authentication["result_sha256"]
    ):
        raise Lfm2SemanticPostfoldError(
            "Fresh anchor-residual authentication differs from the publication-time "
            "layer-stack proof."
        )
    input_weight = getattr(target_model.get_input_embeddings(), "weight", None)
    output_weight = getattr(target_model.get_output_embeddings(), "weight", None)
    if (
        not isinstance(lineage, Mapping)
        or lineage.get("interface_report_schema") != TIED_RESIDUAL_CONGRUENCE_SCHEMA
        or lineage.get("interface_report_sha256")
        != authentication["interface_report_sha256"]
        or lineage.get("producer_encoder_sha256")
        != authentication["producer_encoder_sha256"]
        or lineage.get("target_weight_sha256") != authentication["target_weight_sha256"]
        or lineage.get("tied_residual_congruence_authenticated") is not True
        or lineage.get("global_residual_gauge_required") is not True
        or not isinstance(input_weight, torch.nn.Parameter)
        or output_weight is not input_weight
        or not torch.equal(
            interface.target_weight.to(device="cpu", dtype=input_weight.dtype),
            input_weight.detach().to(device="cpu"),
        )
        or (
            authentication.get("strong_congruence_claimed") is True
            and authentication.get("strong_congruence_recomputed_exact") is not True
        )
        or (
            authentication.get("strong_base_plus_anchor_residual_claimed") is True
            and authentication.get("anchor_residual_formula_recomputed_exact")
            is not True
        )
    ):
        raise Lfm2SemanticPostfoldError(
            "Fused terminal interface differs from the frozen layer-stack "
            "congruence or live tied child."
        )
    body = {
        "schema": "neural-forge-lfm2-fused-terminal-tied-interface-v1",
        "interface_authentication_sha256": authentication["result_sha256"],
        "interface_report_sha256": authentication["interface_report_sha256"],
        "interface_report_schema": TIED_RESIDUAL_CONGRUENCE_SCHEMA,
        "mode": authentication["mode"],
        "producer_encoder_sha256": authentication["producer_encoder_sha256"],
        "target_weight_sha256": authentication["target_weight_sha256"],
        "layer_stack_interface_lineage_sha256": canonical_sha256(dict(lineage)),
        "strong_congruence_recomputed_exact": authentication[
            "strong_congruence_recomputed_exact"
        ],
        "global_residual_gauge_frozen_through_terminal": True,
        "live_native_tied_table_matches_interface": True,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    return MappingProxyType(_sealed(body, "result_sha256"))


def _terminal_heldout_diagnostic(
    published_metrics: Any,
    incumbent_metrics: Any,
    policy: Lfm2CpcfTerminalActionPolicy,
) -> dict[str, Any]:
    """Seal numerical heldout evidence without granting it publication authority."""

    gates = _metric_gates(
        published_metrics,
        incumbent_metrics,
        policy,
        require_selection_improvement=False,
    )
    body = {
        "incumbent": incumbent_metrics.as_dict(),
        "published": published_metrics.as_dict(),
        "gates": gates,
        "passed": _all_passed(gates),
        "role": "diagnostic-only-after-frozen-terminal-publication-decision",
        "publication_gate_applied": False,
    }
    return _sealed(body, "diagnostic_sha256")


def _build_terminal_frontier_export_context(
    *,
    terminal: Lfm2CpcfTerminalActionResidualResult,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    parent_final_norm: Lfm2LocalParentFinalRmsNormLoad,
    mapped_parent_tied_table: Any,
    layer_receipt: Mapping[str, Any],
    terminal_interface_authentication: Mapping[str, Any],
    terminal_cache_compatibility: Mapping[str, Any],
    terminal_cache_payload_binding: Mapping[str, Any],
    terminal_composite_lineage: Lfm2LongHorizonTerminalCompositeLineage,
    terminal_cache_session_sha256: str,
    same_live_cache_session_as_layer_stack: bool,
    cache_content_identity_matches_layer_stack: bool,
    preterminal_resume_mode: bool,
    body_recipe: str | Lfm2SemanticBodyRecipe | None = None,
) -> Mapping[str, Any]:
    """Seal the live provenance a frontier publisher cannot reconstruct."""

    if not isinstance(terminal, Lfm2CpcfTerminalActionResidualResult):
        raise TypeError("terminal must be a typed terminal-action result.")
    if not terminal.verify_authentication():
        raise Lfm2SemanticPostfoldError(
            "Terminal frontier changed before export-context construction."
        )
    if any(
        type(value) is not bool
        for value in (
            same_live_cache_session_as_layer_stack,
            cache_content_identity_matches_layer_stack,
            preterminal_resume_mode,
        )
    ):
        raise TypeError("Terminal frontier cache-session facts must be bools.")
    if not cache_content_identity_matches_layer_stack or (
        preterminal_resume_mode and same_live_cache_session_as_layer_stack
    ):
        raise Lfm2SemanticPostfoldError(
            "Terminal frontier export context has contradictory cache-session facts."
        )
    compatibility = _verify_sealed(
        terminal_cache_compatibility,
        seal_name="authentication_sha256",
        label="terminal frontier cache compatibility",
    )
    payload_binding = _verify_sealed(
        terminal_cache_payload_binding,
        seal_name="authentication_sha256",
        label="terminal frontier cache payload binding",
    )
    interface_authentication = _verify_sealed(
        terminal_interface_authentication,
        seal_name="result_sha256",
        label="terminal frontier tied-interface authentication",
    )
    try:
        composite_lineage = authenticate_lfm2_long_horizon_terminal_composite_lineage(
            terminal_composite_lineage.as_dict()
        )
    except (TypeError, ValueError) as exc:
        raise Lfm2SemanticPostfoldError(
            "Terminal frontier composite lineage authentication failed."
        ) from exc
    composite_inputs = composite_lineage.document["compiler_inputs"]
    compatibility_sha256 = _digest(
        compatibility.get("authentication_sha256"),
        "terminal cache compatibility authentication",
    )
    payload_binding_sha256 = _digest(
        payload_binding.get("authentication_sha256"),
        "terminal cache payload binding authentication",
    )
    layer_result_sha256 = _digest(
        layer_receipt.get("result_sha256"),
        "semantic layer-stack result",
    )
    recipe = _authenticate_body_recipe_binding(
        layer_receipt,
        body_recipe=body_recipe,
        label="terminal frontier semantic layer-stack receipt",
    )
    anchor_selection_value = layer_receipt.get("interface_anchor_selection")
    anchor_selection = (
        None
        if anchor_selection_value is None
        else _authenticate_foundation_anchor_selection(anchor_selection_value)
    )
    anchor_selection_sha256 = (
        None if anchor_selection is None else anchor_selection["selection_sha256"]
    )
    if layer_receipt.get("interface_anchor_selection_sha256") != (
        anchor_selection_sha256
    ):
        raise Lfm2SemanticPostfoldError(
            "Terminal frontier layer receipt has a mismatched anchor selection."
        )
    interface_result_sha256 = _digest(
        interface_authentication.get("result_sha256"),
        "terminal tied-interface authentication",
    )
    terminal_seed_sha256 = _digest(
        terminal.report.get("seed_sha256"),
        "terminal seed authentication",
    )
    terminal_incumbent_sha256 = _terminal_action_tensor_sha256(
        terminal.incumbent_down_weight,
        "terminal.incumbent_down_weight",
    )
    terminal_session_sha256 = _digest(
        terminal_cache_session_sha256,
        "terminal cache session",
    )
    mapped_table_sha256 = _tensor_sha256(mapped_parent_tied_table)
    terminal_prior_authentication_sha256 = terminal.report.get(
        "down_weight_prior_authentication_sha256"
    )
    terminal_prior_tensor_sha256 = terminal.report.get(
        "down_weight_prior_tensor_sha256"
    )
    layer_prior_authentication_sha256 = layer_receipt.get(
        "final_layer_mlp_down_weight_prior_authentication_sha256"
    )
    layer_prior_tensor_sha256 = layer_receipt.get(
        "final_layer_mlp_down_weight_prior_tensor_sha256"
    )
    prior_bound = terminal_prior_authentication_sha256 is not None
    if prior_bound:
        _digest(
            terminal_prior_authentication_sha256,
            "terminal down-weight prior authentication",
        )
        _digest(
            terminal_prior_tensor_sha256,
            "terminal down-weight prior tensor",
        )
        if (
            terminal_prior_authentication_sha256 != layer_prior_authentication_sha256
            or terminal_prior_tensor_sha256 != layer_prior_tensor_sha256
            or terminal.report.get(
                "frontier_centered_on_authenticated_down_weight_prior"
            )
            is not True
        ):
            raise Lfm2SemanticPostfoldError(
                "Terminal frontier prior differs from the layer-stack Q-DEIM prior."
            )
    elif layer_prior_authentication_sha256 is not None:
        raise Lfm2SemanticPostfoldError(
            "Layer-stack Q-DEIM prior was dropped before terminal frontier export."
        )
    if (
        terminal.report.get("incumbent_seed_writer_sha256") != terminal_incumbent_sha256
        or compatibility.get("schema")
        != LFM2_LONG_HORIZON_TERMINAL_CACHE_COMPATIBILITY_SCHEMA
        or compatibility.get("cache_key") != cache.cache_key
        or payload_binding.get("compatibility_authentication_sha256")
        != compatibility_sha256
        or payload_binding.get("payload_session_sha256") != terminal_session_sha256
        or composite_inputs.get("cache_key") != cache.cache_key
        or composite_inputs.get("cache_session_sha256") != terminal_session_sha256
        or composite_inputs.get("cache_compatibility_sha256") != compatibility_sha256
        or composite_inputs.get("cache_payload_binding_sha256")
        != payload_binding_sha256
        or not isinstance(composite_inputs.get("source_lineage"), Mapping)
        or composite_inputs["source_lineage"].get("bridge_artifact_sha256")
        != bridge.artifact_sha256
        or interface_authentication.get("schema")
        != "neural-forge-lfm2-fused-terminal-tied-interface-v1"
    ):
        raise Lfm2SemanticPostfoldError(
            "Terminal frontier live facts differ from their authenticated sources."
        )
    body = {
        "schema": LFM2_SEMANTIC_TERMINAL_FRONTIER_EXPORT_CONTEXT_SCHEMA,
        "terminal_result_sha256": terminal.result_sha256,
        "terminal_seed_sha256": terminal_seed_sha256,
        "terminal_incumbent_down_weight_terminal_action_sha256": (
            terminal_incumbent_sha256
        ),
        "terminal_cache_compatibility_sha256": compatibility_sha256,
        "terminal_cache_payload_binding_sha256": payload_binding_sha256,
        "terminal_cache_session_sha256": terminal_session_sha256,
        "terminal_source_lineage_sha256": composite_inputs["source_lineage_sha256"],
        "terminal_composite_lineage": composite_lineage.as_dict(),
        "terminal_composite_lineage_sha256": (composite_lineage.authentication_sha256),
        "terminal_compiler_binding_sha256": (composite_lineage.compiler_binding_sha256),
        "layer_stack_cache_session_sha256": _digest(
            layer_receipt.get("cache_selective_session_sha256"),
            "layer-stack cache session",
        ),
        "cache_key": _digest(cache.cache_key, "semantic cache key"),
        "cache_parent_fingerprint": _digest(
            cache.parent_fingerprint,
            "semantic cache parent fingerprint",
        ),
        "bridge_artifact_sha256": _digest(
            bridge.artifact_sha256,
            "tokenizer bridge artifact",
        ),
        "parent_final_norm_receipt_sha256": _digest(
            parent_final_norm.result_sha256,
            "parent final norm receipt",
        ),
        "mapped_parent_tied_table_sha256": mapped_table_sha256,
        "mapped_parent_tied_table_interface_report_sha256": _digest(
            interface_authentication.get("interface_report_sha256"),
            "mapped tied-table interface report",
        ),
        "semantic_layer_stack_result_sha256": layer_result_sha256,
        "body_recipe": recipe.as_dict(),
        "body_recipe_sha256": recipe.recipe_sha256,
        "body_recipe_id": recipe.recipe_id,
        "interface_anchor_selection_sha256": anchor_selection_sha256,
        **(
            {"interface_anchor_selection": dict(anchor_selection)}
            if anchor_selection is not None
            else {}
        ),
        "tied_interface_authentication_sha256": interface_result_sha256,
        "preterminal_resume_mode": preterminal_resume_mode,
        "same_live_cache_session_as_layer_stack": (
            same_live_cache_session_as_layer_stack
        ),
        "cache_content_identity_matches_layer_stack": (
            cache_content_identity_matches_layer_stack
        ),
        "frontier_export_occurs_before_final_w2_installation": True,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    if prior_bound:
        body.update(
            {
                "down_weight_prior_authentication_sha256": (
                    terminal_prior_authentication_sha256
                ),
                "down_weight_prior_tensor_sha256": terminal_prior_tensor_sha256,
                "down_weight_prior_bound_from_layer_stack_into_terminal": True,
                "pure_down_weight_prior_frontier_candidate_id": (
                    "qdeim-prior-alpha-00"
                ),
            }
        )
    sealed = _sealed(body, "context_sha256")
    issued_facts = {name: sealed[name] for name in _TERMINAL_FRONTIER_LIVE_FACT_NAMES}
    return _CompilerIssuedTerminalFrontierExportContext(
        sealed,
        issued_facts,
        issuer=_TERMINAL_FRONTIER_EXPORT_CONTEXT_ISSUER,
    )


def _authenticate_terminal_frontier_export_context(
    terminal: Lfm2CpcfTerminalActionResidualResult,
    context: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(context, _CompilerIssuedTerminalFrontierExportContext):
        raise Lfm2SemanticPostfoldError(
            "Terminal frontier export context is not compiler-issued."
        )
    issued_fact_tuple, issued_fact_sha256 = context._authenticate_issued_facts(
        _TERMINAL_FRONTIER_EXPORT_CONTEXT_ISSUER
    )
    issued_facts = dict(issued_fact_tuple)
    authenticated = _verify_sealed(
        context,
        seal_name="context_sha256",
        label="terminal frontier export context",
    )
    composite_value = authenticated.get("terminal_composite_lineage")
    try:
        composite = authenticate_lfm2_long_horizon_terminal_composite_lineage(
            composite_value
        )
    except (TypeError, ValueError) as exc:
        raise Lfm2SemanticPostfoldError(
            "Terminal frontier context has invalid composite lineage."
        ) from exc
    composite_inputs = composite.document["compiler_inputs"]
    recipe = _authenticate_body_recipe_binding(
        authenticated,
        label="terminal frontier export context",
    )
    anchor_selection_value = authenticated.get("interface_anchor_selection")
    anchor_selection = (
        None
        if anchor_selection_value is None
        else _authenticate_foundation_anchor_selection(anchor_selection_value)
    )
    terminal_prior_authentication_sha256 = terminal.report.get(
        "down_weight_prior_authentication_sha256"
    )
    terminal_prior_tensor_sha256 = terminal.report.get(
        "down_weight_prior_tensor_sha256"
    )
    prior_context_valid = (
        (
            authenticated.get("down_weight_prior_authentication_sha256")
            == terminal_prior_authentication_sha256
            and authenticated.get("down_weight_prior_tensor_sha256")
            == terminal_prior_tensor_sha256
            and authenticated.get(
                "down_weight_prior_bound_from_layer_stack_into_terminal"
            )
            is True
            and authenticated.get("pure_down_weight_prior_frontier_candidate_id")
            == "qdeim-prior-alpha-00"
        )
        if terminal_prior_authentication_sha256 is not None
        else (
            "down_weight_prior_authentication_sha256" not in authenticated
            and "down_weight_prior_tensor_sha256" not in authenticated
        )
    )
    if (
        authenticated.get("schema")
        != LFM2_SEMANTIC_TERMINAL_FRONTIER_EXPORT_CONTEXT_SCHEMA
        or authenticated.get("terminal_result_sha256") != terminal.result_sha256
        or authenticated.get("terminal_seed_sha256")
        != terminal.report.get("seed_sha256")
        or authenticated.get("terminal_incumbent_down_weight_terminal_action_sha256")
        != _terminal_action_tensor_sha256(
            terminal.incumbent_down_weight,
            "terminal.incumbent_down_weight",
        )
        or authenticated.get("terminal_composite_lineage_sha256")
        != composite.authentication_sha256
        or authenticated.get("terminal_compiler_binding_sha256")
        != composite.compiler_binding_sha256
        or authenticated.get("terminal_source_lineage_sha256")
        != composite_inputs.get("source_lineage_sha256")
        or authenticated.get("body_recipe_sha256") != recipe.recipe_sha256
        or authenticated.get("interface_anchor_selection_sha256")
        != (None if anchor_selection is None else anchor_selection["selection_sha256"])
        or authenticated.get("terminal_cache_session_sha256")
        != composite_inputs.get("cache_session_sha256")
        or authenticated.get("terminal_cache_compatibility_sha256")
        != composite_inputs.get("cache_compatibility_sha256")
        or authenticated.get("terminal_cache_payload_binding_sha256")
        != composite_inputs.get("cache_payload_binding_sha256")
        or authenticated.get("frontier_export_occurs_before_final_w2_installation")
        is not True
        or prior_context_valid is not True
        or authenticated.get("cache_content_identity_matches_layer_stack") is not True
        or type(authenticated.get("same_live_cache_session_as_layer_stack")) is not bool
        or type(authenticated.get("preterminal_resume_mode")) is not bool
        or (
            authenticated.get("preterminal_resume_mode") is True
            and authenticated.get("same_live_cache_session_as_layer_stack") is True
        )
        or authenticated.get("operation_audit") != dict(_ZERO_OPERATIONS)
        or any(
            authenticated.get(name) != issued_facts.get(name)
            for name in _TERMINAL_FRONTIER_LIVE_FACT_NAMES
        )
        or issued_fact_sha256
        != canonical_sha256([list(value) for value in issued_fact_tuple])
    ):
        raise Lfm2SemanticPostfoldError(
            "Terminal frontier export context does not authenticate the live result."
        )
    return context


def _export_authenticated_terminal_frontier(
    terminal: Lfm2CpcfTerminalActionResidualResult,
    context: Mapping[str, Any],
    exporter: Callable[
        [Lfm2CpcfTerminalActionResidualResult, Mapping[str, Any]],
        Any,
    ]
    | None,
) -> dict[str, Any] | None:
    """Expose the full private bank once, before any final-W2 installation."""

    if not isinstance(terminal, Lfm2CpcfTerminalActionResidualResult):
        raise TypeError("terminal must be a typed terminal-action result.")
    if not terminal.verify_authentication():
        raise Lfm2SemanticPostfoldError(
            "Fused terminal frontier changed before its export seam."
        )
    authenticated_context = _authenticate_terminal_frontier_export_context(
        terminal,
        context,
    )
    if exporter is None:
        return None
    if not callable(exporter):
        raise TypeError("terminal_frontier_exporter must be callable.")
    exported = exporter(terminal, authenticated_context)
    if not terminal.verify_authentication():
        raise Lfm2SemanticPostfoldError(
            "Terminal frontier exporter mutated the authenticated result."
        )
    _authenticate_terminal_frontier_export_context(terminal, authenticated_context)
    as_dict = getattr(exported, "as_dict", None)
    raw_export = as_dict() if callable(as_dict) else exported
    if not isinstance(raw_export, Mapping) or not raw_export:
        raise Lfm2SemanticPostfoldError(
            "Terminal frontier exporter returned no authenticated reference."
        )
    try:
        return json.loads(
            json.dumps(
                dict(raw_export),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise Lfm2SemanticPostfoldError(
            "Terminal frontier export reference is not canonical JSON."
        ) from exc


def _terminal_frontier_export_receipt_fields(
    reference: Mapping[str, Any] | None,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    authenticated_context = dict(
        _verify_sealed(
            context,
            seal_name="context_sha256",
            label="terminal frontier export receipt context",
        )
    )
    normalized = None if reference is None else dict(reference)
    return {
        "terminal_frontier_export_context": authenticated_context,
        "terminal_frontier_export_context_sha256": authenticated_context[
            "context_sha256"
        ],
        "terminal_frontier_export": normalized,
        "terminal_frontier_export_sha256": (
            None if normalized is None else canonical_sha256(normalized)
        ),
        "terminal_frontier_exported_before_w2_installation": (normalized is not None),
    }


def compile_install_lfm2_fused_semantic_terminal_refold(
    *,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    target_model: Any,
    layer_stack: Lfm2SemanticLayerStackInstallation,
    interface: TiedCausalInterfaceFold,
    source_selected_tied_table: Any,
    source_lineage: Lfm2LongHorizonTerminalSourceLineage,
    parent_final_norm: Lfm2LocalParentFinalRmsNormLoad,
    donor_tokenizer: Any,
    target_tokenizer: Any,
    body_recipe: str | Lfm2SemanticBodyRecipe | None = None,
    split_plan: Lfm2LongHorizonTrajectorySplitPlan | None = None,
    policy: Lfm2CpcfTerminalActionPolicy | None = None,
    terminal_cache_compatibility: Mapping[str, Any] | None = None,
    authenticated_cache_session: Any | None = None,
    preterminal_resume_mode: bool = False,
    terminal_frontier_exporter: Callable[
        [Lfm2CpcfTerminalActionResidualResult, Mapping[str, Any]],
        Any,
    ]
    | None = None,
    anchor_transport_witness: TiedResidualAnchorTransportWitness | None = None,
) -> Mapping[str, Any]:
    """Jointly compile final-MLP, parent, and lexical lanes into one final W2."""

    torch = _torch()
    if type(preterminal_resume_mode) is not bool:
        raise TypeError("preterminal_resume_mode must be bool.")
    if preterminal_resume_mode and authenticated_cache_session is not None:
        raise Lfm2SemanticPostfoldError(
            "Preterminal resume must open and authenticate a distinct terminal "
            "cache session internally."
        )
    if not preterminal_resume_mode and authenticated_cache_session is None:
        raise Lfm2SemanticPostfoldError(
            "Continuous fused terminal compilation requires the live cache "
            "session used by the semantic layer stack."
        )
    current_terminal_cache_compatibility = preflight_lfm2_long_horizon_terminal_cache(
        cache
    )
    if terminal_cache_compatibility is not None and (
        not isinstance(terminal_cache_compatibility, Mapping)
        or dict(terminal_cache_compatibility) != current_terminal_cache_compatibility
    ):
        raise Lfm2SemanticPostfoldError(
            "Early terminal-cache compatibility receipt differs from the "
            "current sealed cache records."
        )
    if not isinstance(layer_stack, Lfm2SemanticLayerStackInstallation):
        raise TypeError("layer_stack must authenticate deferred final-MLP evidence.")
    if not isinstance(parent_final_norm, Lfm2LocalParentFinalRmsNormLoad):
        raise TypeError("parent_final_norm must be an authenticated direct-local load.")
    if not isinstance(source_lineage, Lfm2LongHorizonTerminalSourceLineage):
        raise TypeError("source_lineage must be Lfm2LongHorizonTerminalSourceLineage.")
    if not source_lineage.direct_local_checkpoint_source_authenticated:
        raise Lfm2SemanticPostfoldError(
            "Fused semantic terminal requires direct-local checkpoint source lineage."
        )
    if (
        parent_final_norm.final_norm.cache_key != cache.cache_key
        or parent_final_norm.final_norm.parent_fingerprint != cache.parent_fingerprint
        or parent_final_norm.final_norm.layer_topology_fingerprint
        != cache.layer_topology_fingerprint
    ):
        raise Lfm2SemanticPostfoldError(
            "Direct-local parent final RMSNorm is not bound to the semantic "
            "evidence cache."
        )
    layer_receipt = layer_stack.receipt
    recipe = _authenticate_body_recipe_binding(
        layer_receipt,
        body_recipe=body_recipe,
        label="fused terminal semantic layer-stack receipt",
    )
    layer_anchor_selection_value = layer_receipt.get("interface_anchor_selection")
    layer_anchor_selection = (
        None
        if layer_anchor_selection_value is None
        else _authenticate_foundation_anchor_selection(layer_anchor_selection_value)
    )
    if layer_receipt.get("interface_anchor_selection_sha256") != (
        None
        if layer_anchor_selection is None
        else layer_anchor_selection["selection_sha256"]
    ):
        raise Lfm2SemanticPostfoldError(
            "Fused terminal layer receipt has a mismatched anchor selection."
        )
    terminal_interface_authentication = _authenticate_fused_terminal_tied_interface(
        target_model=target_model,
        interface=interface,
        mapped_parent_tied_table=source_selected_tied_table,
        layer_receipt=layer_receipt,
        anchor_transport_witness=anchor_transport_witness,
    )
    source_lineage.verify_for(
        cache=cache,
        bridge=bridge,
        parent_final_norm=parent_final_norm,
        interface=interface,
        source_selected_tied_table=source_selected_tied_table,
        anchor_transport_witness=anchor_transport_witness,
    )
    layer_installations = layer_receipt.get("layer_installations")
    operator_layer_kinds, _, _ = _authenticated_layer_stack_operator_topology(
        layer_receipt,
        layer_installations,
    )
    terminal_layer_index = len(operator_layer_kinds) - 1
    current_state = fingerprint_lfm2_parent(target_model).identity_sha256
    current_w2 = _tensor_sha256(_final_w2_parameter(target_model))
    final_mlp_producer_coverage_before_terminal = _lfm2_dense_producer_chain_coverage(
        target_model,
        layer_index=terminal_layer_index,
        producer_kind="mlp",
        defer_down_weight=True,
    )
    final_layer_installation = (
        layer_installations[-1]
        if isinstance(layer_installations, list) and layer_installations
        else None
    )
    recorded_final_mlp_coverage = (
        final_layer_installation.get("mlp_producer_coverage_decision")
        if isinstance(final_layer_installation, Mapping)
        else None
    )
    if (
        layer_receipt.get("model_state_after_sha256") != current_state
        or layer_receipt.get("deferred_final_w2_sha256") != current_w2
        or layer_receipt.get("complete_nonzero_structural_producer_coverage")
        is not True
        or final_mlp_producer_coverage_before_terminal.get(
            "all_required_roles_structurally_nonzero"
        )
        is not True
        or final_mlp_producer_coverage_before_terminal.get(
            "blank_or_partially_blank_required_role_count"
        )
        != 0
        or not isinstance(recorded_final_mlp_coverage, Mapping)
        or recorded_final_mlp_coverage.get("coverage_after_sha256")
        != final_mlp_producer_coverage_before_terminal.get("result_sha256")
        or layer_receipt.get("whole_stage_contains_meaningful_parameter_changes")
        is not True
        or layer_receipt.get("model_state_before_sha256") == current_state
        or not layer_receipt.get("changed_state_keys")
    ):
        raise Lfm2SemanticPostfoldError(
            "Child changed after deferred final-MLP evidence was captured or the "
            "layer stack proved no meaningful mutation."
        )
    selected_policy = policy or Lfm2CpcfTerminalActionPolicy()
    plan = split_plan or Lfm2LongHorizonTrajectorySplitPlan.deterministic_round_robin(
        cache
    )
    plan.verify_for(cache)
    terminal_session_context = (
        cache.selective_authenticated_session()
        if authenticated_cache_session is None
        else nullcontext(authenticated_cache_session)
    )
    with terminal_session_context as terminal_cache_session:
        seed_parent, _down, layers, _epsilon = _target_runtime_seed(
            target_model,
            interface,
            source_selected_tied_table,
        )
        before_replay = fingerprint_lfm2_parent(target_model).identity_sha256
        cached, semantic_bridges, cache_receipt = _collect_cached_semantic_trajectories(
            cache,
            bridge,
            donor_tokenizer,
            target_tokenizer,
            parent_final_norm.final_norm,
            source_selected_tied_table,
            source_lineage,
            plan,
            authenticated_session=terminal_cache_session,
        )
        terminal_cache_session.verify()
    terminal_cache_payload_binding = (
        _bind_terminal_cache_compatibility_to_payload_session(
            current_terminal_cache_compatibility,
            cache_receipt,
        )
    )
    terminal_composite_lineage = _build_terminal_composite_lineage(
        cache=cache,
        source_lineage=source_lineage,
        compatibility=current_terminal_cache_compatibility,
        payload_binding=terminal_cache_payload_binding,
        cache_receipt=cache_receipt,
        cached=cached,
    )
    terminal_cache_tensor_payload_load_audit = _summarize_terminal_cache_payload_loads(
        cache_receipt
    )
    if (
        terminal_cache_tensor_payload_load_audit.get(
            "every_batch_loaded_full_tensor_inventory_once"
        )
        is not True
        or terminal_cache_tensor_payload_load_audit.get(
            "full_tensor_inventory_payload_load_count"
        )
        != cache.batch_count
    ):
        raise Lfm2SemanticPostfoldError(
            "Semantic terminal collection did not load each complete sealed tensor "
            "inventory exactly once."
        )
    layer_stack_cache_session_sha256 = layer_receipt.get(
        "cache_selective_session_sha256"
    )
    terminal_cache_session_sha256 = cache_receipt["session"]["session_sha256"]
    cache_content_identity_matches_layer_stack = (
        layer_stack_cache_session_sha256 == terminal_cache_session_sha256
    )
    same_live_cache_session_as_layer_stack = not preterminal_resume_mode
    if not cache_content_identity_matches_layer_stack:
        raise Lfm2SemanticPostfoldError(
            "Terminal cache content identity differs from the session "
            "authenticated by the layer-stack receipt."
        )
    if not all(
        value.parent_final_norm_endpoint.get("runtime_replay_to_cached_output_verified")
        is True
        and value.parent_final_norm_endpoint.get(
            "authenticated_cached_output_used_as_compiler_target"
        )
        is True
        for value in cached
    ):
        raise Lfm2SemanticPostfoldError(
            "Semantic postfold requires every cached pre-final residual to "
            "authenticate the direct-local parent RMSNorm output."
        )
    parent_final_norm_endpoint_diagnostics = [
        dict(value.parent_final_norm_endpoint) for value in cached
    ]
    runtime_norm_envelope_diagnostics = [
        value.get("runtime_replay_within_legacy_numeric_envelope")
        for value in parent_final_norm_endpoint_diagnostics
        if value.get("cached_output_endpoint_available") is True
    ]
    all_runtime_norm_replays_within_legacy_envelope = (
        all(value is True for value in runtime_norm_envelope_diagnostics)
        if runtime_norm_envelope_diagnostics
        else None
    )
    parent_raw, _horizons, parent_replays, parent_rope_calls = _build_split_evidence(
        cached,
        None,
        target_model,
        layers,
    )
    parent = {
        value.split: _convert_parent_terminal_evidence(
            value,
            interface.producer_encoder,
        )
        for value in parent_raw
    }
    lexical_rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("fit", "fit_check", "selection", "validation")
    }
    observed = []
    all_desired = []
    pending = []
    for semantic in semantic_bridges:
        batch_index = semantic.source_batch_index
        split = plan.assignments[batch_index]
        (
            row_positions,
            desired,
            row_ids,
            transition_multiplicities,
        ) = _append_only_lexical_supervision_coordinates(semantic)
        if not row_positions:
            raise Lfm2SemanticPostfoldError(
                f"Cache batch {batch_index} has no inserted lexical rows."
            )
        mlp, pre, replay = _replay_lexical_rows(
            target_model,
            token_ids=semantic.phases[-1].target_full_prefix_token_ids,
            position_origin=semantic.target_position_origin,
            row_positions=row_positions,
        )
        pending.append((split, row_ids, mlp, pre, tuple(desired)))
        all_desired.extend(desired)
        observed.append(
            {
                "batch_index": batch_index,
                "split": split,
                "parent_decode_steps": len(semantic.phases) - 1,
                "target_decode_tokens": (
                    semantic.phases[-1].target_full_prefix_length
                    - semantic.phases[0].target_full_prefix_length
                ),
                "inserted_continuation_rows": len(row_positions),
                "maximum_donor_span_multiplicity": max(transition_multiplicities),
                "transition_multiplicity_counts": {
                    str(value): transition_multiplicities.count(value)
                    for value in sorted(set(transition_multiplicities))
                },
                "lexical_row_ids_sha256": canonical_sha256(row_ids),
                "desired_token_ids_sha256": canonical_sha256(desired),
                "replay_sha256": replay["replay_sha256"],
            }
        )
    teachers, teacher_report = _lexical_teacher_rows(
        seed_parent.target_tied_weight,
        all_desired,
    )
    cursor = 0
    for split, row_ids, mlp, pre, desired in pending:
        count = len(desired)
        lexical_rows[split].append(
            {
                "row_ids": tuple(row_ids),
                "mlp": mlp,
                "pre": pre,
                "teacher": teachers[cursor : cursor + count],
            }
        )
        cursor += count
    lexical = {}
    for split, values in lexical_rows.items():
        if not values:
            raise Lfm2SemanticPostfoldError(
                f"Lexical split {split!r} has no whole-trajectory rows."
            )
        rows = sum(len(value["row_ids"]) for value in values)
        lexical[split] = Lfm2CpcfTerminalActionEvidence(
            split=split,
            row_ids=tuple(row for value in values for row in value["row_ids"]),
            mlp_input=torch.cat(tuple(value["mlp"] for value in values), dim=0),
            pre_mlp_residual=torch.cat(
                tuple(value["pre"] for value in values),
                dim=0,
            ),
            parent_final_norm_output=torch.cat(
                tuple(value["teacher"] for value in values),
                dim=0,
            ),
            sample_weights=torch.full((rows,), 1.0 / rows, dtype=torch.float64),
        )
    hidden = seed_parent.target_hidden_size
    identity = torch.eye(hidden, dtype=torch.float64)
    seed = Lfm2CpcfTerminalActionSeed(
        gate_weight=seed_parent.gate_weight,
        up_weight=seed_parent.up_weight,
        down_weight=seed_parent.down_weight,
        final_norm_weight=seed_parent.final_norm_weight,
        producer_encoder=identity,
        target_tied_weight=seed_parent.target_tied_weight,
        source_tied_table=seed_parent.target_tied_weight,
        norm_epsilon=seed_parent.norm_epsilon,
    )
    final_mlp = dict(layer_stack.final_layer_mlp_evidence)
    fused = {
        split: _merge_terminal_evidence(
            {
                "final_layer_mlp": final_mlp[split],
                "lexical": lexical[split],
                "parent_endpoint": parent[split],
            }
        )
        for split in lexical
    }
    terminal = compile_lfm2_cpcf_terminal_action_residual(
        seed=seed,
        fit_evidence=fused["fit"],
        fit_check_evidence=fused["fit_check"],
        selection_evidence=fused["selection"],
        validation_evidence=fused["validation"],
        policy=selected_policy,
        down_weight_prior=layer_stack.final_layer_mlp_down_weight_prior,
    )
    terminal_frontier_export_context = _build_terminal_frontier_export_context(
        terminal=terminal,
        cache=cache,
        bridge=bridge,
        parent_final_norm=parent_final_norm,
        mapped_parent_tied_table=source_selected_tied_table,
        layer_receipt=layer_receipt,
        terminal_interface_authentication=terminal_interface_authentication,
        terminal_cache_compatibility=current_terminal_cache_compatibility,
        terminal_cache_payload_binding=terminal_cache_payload_binding,
        terminal_composite_lineage=terminal_composite_lineage,
        terminal_cache_session_sha256=terminal_cache_session_sha256,
        same_live_cache_session_as_layer_stack=(same_live_cache_session_as_layer_stack),
        cache_content_identity_matches_layer_stack=(
            cache_content_identity_matches_layer_stack
        ),
        preterminal_resume_mode=preterminal_resume_mode,
        body_recipe=recipe,
    )
    terminal_frontier_export = _export_authenticated_terminal_frontier(
        terminal,
        terminal_frontier_export_context,
        terminal_frontier_exporter,
    )
    terminal_cache_io_audit = _sealed(
        {
            "metadata_preflight_trace_file_sha256_passes": 0,
            "metadata_preflight_trace_payload_bytes_read": 0,
            "fused_terminal_selective_session_constructions": (
                0 if authenticated_cache_session is not None else 1
            ),
            "end_to_end_selective_session_constructions": (
                1 if same_live_cache_session_as_layer_stack else 2
            ),
            "consuming_trace_file_sha256_passes_per_batch": 1,
            "consuming_trace_file_sha256_passes_total": cache.batch_count,
            "same_consuming_session_reused_by_semantic_collector": True,
            "same_consuming_session_reused_from_layer_stack": (
                same_live_cache_session_as_layer_stack
            ),
            "same_live_cache_session_as_layer_stack": (
                same_live_cache_session_as_layer_stack
            ),
            "cache_content_identity_matches_layer_stack": (
                cache_content_identity_matches_layer_stack
            ),
            "preterminal_resume_mode": preterminal_resume_mode,
            "additional_full_trace_file_sha256_passes_performed_by_fused_terminal": 0,
            "selected_tensor_payload_loads_total": (
                terminal_cache_tensor_payload_load_audit[
                    "selected_tensor_payload_load_count"
                ]
            ),
            "selected_tensor_payload_logical_bytes_total": (
                terminal_cache_tensor_payload_load_audit[
                    "selected_tensor_payload_logical_bytes_total"
                ]
            ),
            "full_tensor_inventory_payload_loads_per_batch": 1,
            "full_tensor_inventory_payload_loads_total": (
                terminal_cache_tensor_payload_load_audit[
                    "full_tensor_inventory_payload_load_count"
                ]
            ),
            "full_tensor_inventory_payload_logical_bytes_total": (
                terminal_cache_tensor_payload_load_audit[
                    "full_tensor_inventory_payload_logical_bytes_total"
                ]
            ),
        },
        "result_sha256",
    )
    if terminal_frontier_export is not None:
        terminal_action_selected_sha256 = _terminal_action_tensor_sha256(
            terminal.selected_down_weight,
            "terminal.selected_down_weight",
        )
        terminal_action_published_sha256 = _terminal_action_tensor_sha256(
            terminal.published_down_weight,
            "terminal.published_down_weight",
        )
        semantic_selected_sha256 = _tensor_sha256(terminal.selected_down_weight)
        semantic_published_sha256 = _tensor_sha256(terminal.published_down_weight)
        deferred_body = {
            "schema": LFM2_SEMANTIC_TERMINAL_REVIEW_PENDING_SCHEMA,
            "cache_key": cache.cache_key,
            "cache_parent_fingerprint": cache.parent_fingerprint,
            "bridge_artifact_sha256": bridge.artifact_sha256,
            "semantic_layer_stack_result_sha256": layer_receipt["result_sha256"],
            "interface_anchor_selection_sha256": (
                None
                if layer_anchor_selection is None
                else layer_anchor_selection["selection_sha256"]
            ),
            **(
                {"interface_anchor_selection": dict(layer_anchor_selection)}
                if layer_anchor_selection is not None
                else {}
            ),
            "tied_interface_authentication": dict(terminal_interface_authentication),
            "tied_interface_authentication_sha256": (
                terminal_interface_authentication["result_sha256"]
            ),
            "parent_cache_authentication": cache_receipt,
            "terminal_cache_compatibility": current_terminal_cache_compatibility,
            "terminal_cache_compatibility_sha256": (
                current_terminal_cache_compatibility["authentication_sha256"]
            ),
            "terminal_cache_payload_binding": terminal_cache_payload_binding,
            "terminal_source_lineage": source_lineage.as_dict(),
            "terminal_source_lineage_sha256": source_lineage.authentication_sha256,
            "terminal_composite_lineage": terminal_composite_lineage.as_dict(),
            "terminal_composite_lineage_sha256": (
                terminal_composite_lineage.authentication_sha256
            ),
            "terminal_compiler_binding_sha256": (
                terminal_composite_lineage.compiler_binding_sha256
            ),
            "terminal_cache_tensor_payload_load_audit": (
                terminal_cache_tensor_payload_load_audit
            ),
            "cached_full_prefix_payload_verification": [
                dict(value.cached_full_prefix_payload) for value in cached
            ],
            "cached_full_prefix_payload_verification_sha256": [
                value.cached_full_prefix_payload["authentication_sha256"]
                for value in cached
            ],
            "terminal_cache_io_audit": terminal_cache_io_audit,
            "parent_action_projection_sha256": [
                value.parent_action_projection["authentication_sha256"]
                for value in cached
            ],
            "parent_action_projection_diagnostics": [
                dict(value.parent_action_projection) for value in cached
            ],
            "parent_action_projection_numeric_threshold_applied": False,
            "parent_action_projection_metrics_diagnostic_only": True,
            "terminal_result": terminal.as_dict(),
            "terminal_action_result_sha256": terminal.result_sha256,
            "terminal_action_selected_w2_sha256": (terminal_action_selected_sha256),
            "terminal_action_published_w2_sha256": (terminal_action_published_sha256),
            "semantic_selected_w2_sha256": semantic_selected_sha256,
            "semantic_published_w2_sha256": semantic_published_sha256,
            **_terminal_frontier_export_receipt_fields(
                terminal_frontier_export,
                terminal_frontier_export_context,
            ),
            "final_selection_status": "deferred-to-manual-whole-response-review",
            "terminal_review_pending_schema": (
                LFM2_CPCF_TERMINAL_ACTION_REVIEW_PENDING_SCHEMA
            ),
            "final_selected_candidate_id": None,
            "final_selection_performed": False,
            "automatic_candidate_selection_performed": False,
            "automatic_candidate_acceptance_performed": False,
            "candidate_published_before_manual_review": False,
            "manual_whole_response_review_required": True,
            "diagnostic_order_has_final_selection_authority": False,
            "terminal_writer_install_count": 0,
            "terminal_writer_mutation_count": 0,
            "accepted": False,
            "terminal_publication_decision_accepted": False,
            "operation_audit": dict(_ZERO_OPERATIONS),
        }
        terminal_prior = layer_stack.final_layer_mlp_down_weight_prior
        if terminal_prior is not None:
            deferred_body.update(
                {
                    "final_layer_mlp_down_weight_prior_authentication_sha256": (
                        terminal_prior.authentication_sha256
                    ),
                    "final_layer_mlp_down_weight_prior_tensor_sha256": (
                        _terminal_action_tensor_sha256(
                            terminal_prior.down_weight,
                            "review_pending.final_layer_mlp_down_weight_prior",
                        )
                    ),
                    "final_layer_mlp_down_weight_prior_used_as_frontier_center": (True),
                    "pure_down_weight_prior_candidate_id": ("qdeim-prior-alpha-00"),
                    "final_layer_mlp_down_weight_prior_install_count": 0,
                }
            )
        return _issue_fused_semantic_terminal_receipt(
            _sealed(deferred_body, "result_sha256")
        )
    published_down_weight, publication_decision = _terminal_publication_decision(
        terminal,
        expected_seed=seed,
        expected_evidence=fused,
    )
    exact_zero_cold_start = bool(
        publication_decision.get("terminal_exact_zero_cold_start") is True
    )
    fused_heldout_gates = {}
    for split in ("selection", "validation"):
        prepared = _prepare_split(fused[split], seed)
        incumbent_metrics = _evaluate_split(
            prepared,
            seed,
            seed.down_weight,
            metric_chunk_rows=selected_policy.metric_chunk_rows,
        )
        published_metrics = _evaluate_split(
            prepared,
            seed,
            published_down_weight,
            metric_chunk_rows=selected_policy.metric_chunk_rows,
        )
        heldout_report = _terminal_heldout_diagnostic(
            published_metrics,
            incumbent_metrics,
            selected_policy,
        )
        fused_heldout_gates[split] = heldout_report
    lane_gates = {}
    for lane_name, banks in (
        ("parent_endpoint", parent),
        ("lexical", lexical),
        ("final_layer_mlp", final_mlp),
    ):
        lane_gates[lane_name] = {}
        for split in ("selection", "validation"):
            prepared = _prepare_split(banks[split], seed)
            incumbent_metrics = _evaluate_split(
                prepared,
                seed,
                seed.down_weight,
                metric_chunk_rows=selected_policy.metric_chunk_rows,
            )
            published_metrics = _evaluate_split(
                prepared,
                seed,
                published_down_weight,
                metric_chunk_rows=selected_policy.metric_chunk_rows,
            )
            heldout_report = _terminal_heldout_diagnostic(
                published_metrics,
                incumbent_metrics,
                selected_policy,
            )
            lane_gates[lane_name][split] = heldout_report
    if fingerprint_lfm2_parent(target_model).identity_sha256 != before_replay:
        raise Lfm2SemanticPostfoldError(
            "Fused terminal evidence replay mutated the child before installation."
        )
    global_inventory = _global_tokenizer_expansion_inventory(
        bridge,
        donor_tokenizer,
        target_tokenizer,
    )
    installation = _install_final_w2_once(
        target_model,
        published_down_weight=published_down_weight,
        terminal_publication_decision=publication_decision,
    )
    final_mlp_full_coverage_after_terminal = _lfm2_dense_producer_chain_coverage(
        target_model,
        layer_index=terminal_layer_index,
        producer_kind="mlp",
        defer_down_weight=False,
    )
    if (
        final_mlp_full_coverage_after_terminal.get(
            "all_required_roles_structurally_nonzero"
        )
        is not True
        or final_mlp_full_coverage_after_terminal.get(
            "blank_or_partially_blank_required_role_count"
        )
        != 0
    ):
        raise Lfm2SemanticPostfoldError(
            "Fused terminal did not leave a complete nonzero final MLP chain."
        )
    observed_counts = {value["inserted_continuation_rows"] for value in observed}
    decode_counts = {value["parent_decode_steps"] for value in observed}
    target_counts = {value["target_decode_tokens"] for value in observed}
    maximum_multiplicity = max(
        value["maximum_donor_span_multiplicity"] for value in observed
    )
    terminal_cache_io_audit = _sealed(
        {
            "metadata_preflight_trace_file_sha256_passes": 0,
            "metadata_preflight_trace_payload_bytes_read": 0,
            "fused_terminal_selective_session_constructions": (
                0 if authenticated_cache_session is not None else 1
            ),
            "end_to_end_selective_session_constructions": (
                1 if same_live_cache_session_as_layer_stack else 2
            ),
            "consuming_trace_file_sha256_passes_per_batch": 1,
            "consuming_trace_file_sha256_passes_total": cache.batch_count,
            "same_consuming_session_reused_by_semantic_collector": True,
            "same_consuming_session_reused_from_layer_stack": (
                same_live_cache_session_as_layer_stack
            ),
            "same_live_cache_session_as_layer_stack": (
                same_live_cache_session_as_layer_stack
            ),
            "cache_content_identity_matches_layer_stack": (
                cache_content_identity_matches_layer_stack
            ),
            "preterminal_resume_mode": preterminal_resume_mode,
            "additional_full_trace_file_sha256_passes_performed_by_fused_terminal": 0,
            "selected_tensor_payload_loads_total": (
                terminal_cache_tensor_payload_load_audit[
                    "selected_tensor_payload_load_count"
                ]
            ),
            "selected_tensor_payload_logical_bytes_total": (
                terminal_cache_tensor_payload_load_audit[
                    "selected_tensor_payload_logical_bytes_total"
                ]
            ),
            "full_tensor_inventory_payload_loads_per_batch": 1,
            "full_tensor_inventory_payload_loads_total": (
                terminal_cache_tensor_payload_load_audit[
                    "full_tensor_inventory_payload_load_count"
                ]
            ),
            "full_tensor_inventory_payload_logical_bytes_total": (
                terminal_cache_tensor_payload_load_audit[
                    "full_tensor_inventory_payload_logical_bytes_total"
                ]
            ),
        },
        "result_sha256",
    )
    body = {
        "schema": LFM2_SEMANTIC_LEXICAL_COMPLETION_SCHEMA,
        "cache_key": cache.cache_key,
        "cache_parent_fingerprint": cache.parent_fingerprint,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "semantic_layer_stack_result_sha256": layer_receipt["result_sha256"],
        "interface_anchor_selection_sha256": (
            None
            if layer_anchor_selection is None
            else layer_anchor_selection["selection_sha256"]
        ),
        **(
            {"interface_anchor_selection": dict(layer_anchor_selection)}
            if layer_anchor_selection is not None
            else {}
        ),
        "tied_interface_authentication": dict(terminal_interface_authentication),
        "tied_interface_authentication_sha256": (
            terminal_interface_authentication["result_sha256"]
        ),
        "split_plan": plan.as_dict(),
        "parent_cache_authentication": cache_receipt,
        "terminal_cache_compatibility": current_terminal_cache_compatibility,
        "terminal_cache_compatibility_sha256": (
            current_terminal_cache_compatibility["authentication_sha256"]
        ),
        "terminal_cache_payload_binding": terminal_cache_payload_binding,
        "terminal_source_lineage": source_lineage.as_dict(),
        "terminal_source_lineage_sha256": source_lineage.authentication_sha256,
        "terminal_composite_lineage": terminal_composite_lineage.as_dict(),
        "terminal_composite_lineage_sha256": (
            terminal_composite_lineage.authentication_sha256
        ),
        "terminal_compiler_binding_sha256": (
            terminal_composite_lineage.compiler_binding_sha256
        ),
        "terminal_cache_tensor_payload_load_audit": (
            terminal_cache_tensor_payload_load_audit
        ),
        "cached_full_prefix_payload_verification": [
            dict(value.cached_full_prefix_payload) for value in cached
        ],
        "cached_full_prefix_payload_verification_sha256": [
            value.cached_full_prefix_payload["authentication_sha256"]
            for value in cached
        ],
        "terminal_cache_io_audit": terminal_cache_io_audit,
        "parent_final_norm_direct_local_source": parent_final_norm.as_dict(),
        "parent_final_norm_provenance": {
            "direct_local_full_state_identity_reconstructed": True,
            "reconstructed_parent_fingerprint": (
                parent_final_norm.final_norm.parent_fingerprint
            ),
            "matches_authenticated_cache_parent_fingerprint": (
                parent_final_norm.final_norm.parent_fingerprint
                == cache.parent_fingerprint
            ),
            "cached_final_residual_to_norm_input_exact": True,
            "runtime_equivalent_norm_replay_within_legacy_envelope_at_every_phase": (
                all_runtime_norm_replays_within_legacy_envelope
            ),
            "runtime_norm_numeric_diagnostics_are_admission_authority": False,
            "authenticated_cached_norm_output_used_as_terminal_target": True,
            "parent_model_objects_constructed": 0,
            "parent_model_forward_calls": 0,
        },
        "parent_final_norm_endpoint_sha256": [
            value.parent_final_norm_endpoint["authentication_sha256"]
            for value in cached
        ],
        "parent_final_norm_endpoint_diagnostics": (
            parent_final_norm_endpoint_diagnostics
        ),
        "parent_action_projection_sha256": [
            value.parent_action_projection["authentication_sha256"] for value in cached
        ],
        "parent_action_projection_diagnostics": [
            dict(value.parent_action_projection) for value in cached
        ],
        "parent_action_projection_numeric_threshold_applied": False,
        "parent_action_projection_metrics_diagnostic_only": True,
        "parent_endpoint_evidence_sha256": {
            split: value.authentication_sha256 for split, value in parent.items()
        },
        "lexical_evidence_sha256": {
            split: value.authentication_sha256 for split, value in lexical.items()
        },
        "deferred_final_layer_mlp_evidence_sha256": {
            split: value.authentication_sha256 for split, value in final_mlp.items()
        },
        "fused_evidence_sha256": {
            split: value.authentication_sha256 for split, value in fused.items()
        },
        "observed_trace_coverage": {
            "batch_count": len(observed),
            "per_batch": observed,
            "uniform_inserted_continuation_rows": (
                next(iter(observed_counts)) if len(observed_counts) == 1 else None
            ),
            "uniform_parent_decode_steps": (
                next(iter(decode_counts)) if len(decode_counts) == 1 else None
            ),
            "uniform_target_decode_tokens": (
                next(iter(target_counts)) if len(target_counts) == 1 else None
            ),
            "maximum_donor_span_multiplicity": maximum_multiplicity,
            "total_inserted_continuation_rows": sum(observed_counts)
            if len(observed) == 1
            else sum(value["inserted_continuation_rows"] for value in observed),
            "all_inserted_rows_supervised": True,
        },
        "global_tokenizer_expansion_inventory": dict(global_inventory),
        "observed_trace_is_not_global_lexical_proof": True,
        "lexical_teacher": dict(teacher_report),
        "terminal_result": terminal.as_dict(),
        "terminal_action_result_sha256": publication_decision[
            "terminal_action_result_sha256"
        ],
        "terminal_action_selected_w2_sha256": publication_decision[
            "terminal_action_selected_w2_sha256"
        ],
        "terminal_action_published_w2_sha256": publication_decision[
            "terminal_action_published_w2_sha256"
        ],
        "semantic_selected_w2_sha256": publication_decision[
            "semantic_selected_w2_sha256"
        ],
        "semantic_published_w2_sha256": installation["semantic_published_w2_sha256"],
        **_terminal_frontier_export_receipt_fields(
            terminal_frontier_export,
            terminal_frontier_export_context,
        ),
        "terminal_publication_decision": dict(publication_decision),
        "fused_heldout_diagnostics": fused_heldout_gates,
        "separate_heldout_lane_diagnostics": lane_gates,
        "installation": dict(installation),
        "final_mlp_producer_coverage_before_terminal": dict(
            final_mlp_producer_coverage_before_terminal
        ),
        "final_mlp_full_coverage_after_terminal": dict(
            final_mlp_full_coverage_after_terminal
        ),
        "complete_nonzero_final_mlp_structural_coverage": True,
        "terminal_writer_decision_count": 1,
        "terminal_writer_install_count": installation["install_count"],
        "terminal_writer_mutation_count": installation["mutation_count"],
        "terminal_writer_incumbent_retention_count": installation[
            "incumbent_retention_count"
        ],
        "one_frozen_final_w2_publication_decision": True,
        "one_final_w2_install": installation["install_count"] == 1,
        "joint_terminal_evidence_compiled_and_heldout_diagnosed": True,
        "final_layer_mlp_parent_endpoint_and_lexical_rows_jointly_fitted": (
            publication_decision["terminal_nonincumbent_candidate_accepted"]
        ),
        "prior_final_layer_mlp_w2_install_count": 0,
        "parent_trajectory_replay_receipts": list(parent_replays),
        "parent_attention_rope_materializations": parent_rope_calls,
        "accepted": True,
        "terminal_publication_decision_accepted": True,
        "terminal_nonincumbent_candidate_accepted": publication_decision[
            "terminal_nonincumbent_candidate_accepted"
        ],
        "terminal_incumbent_retained": publication_decision[
            "terminal_incumbent_retained"
        ],
        "terminal_heldout_metrics_diagnostic_only": True,
        "terminal_heldout_publication_gate_applied": False,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    terminal_prior = layer_stack.final_layer_mlp_down_weight_prior
    if terminal_prior is not None:
        body.update(
            {
                "final_layer_mlp_down_weight_prior_authentication_sha256": (
                    terminal_prior.authentication_sha256
                ),
                "final_layer_mlp_down_weight_prior_tensor_sha256": (
                    _terminal_action_tensor_sha256(
                        terminal_prior.down_weight,
                        "fused.final_layer_mlp_down_weight_prior",
                    )
                ),
                "final_layer_mlp_down_weight_prior_used_as_frontier_center": True,
                "pure_down_weight_prior_candidate_in_terminal_frontier": True,
                "pure_down_weight_prior_candidate_id": "qdeim-prior-alpha-00",
                "final_layer_mlp_down_weight_prior_install_count": 0,
            }
        )
    if exact_zero_cold_start:
        body.update(
            {
                "terminal_exact_zero_cold_start": True,
                "terminal_incumbent_noop_publication_forbidden": True,
                "terminal_structural_candidate_eligibility_authenticated": True,
                "terminal_fit_metric_ordering_frozen_before_heldout_diagnostics": (
                    True
                ),
                "terminal_fit_metrics_used_for_candidate_ordering": True,
                "terminal_fit_metrics_used_for_pre_review_candidate_selection": True,
                "terminal_fit_metrics_used_for_structural_eligibility": False,
                "terminal_fit_metric_thresholds_used_for_candidate_eligibility": (
                    False
                ),
                "terminal_fit_metric_thresholds_used_for_publication": False,
                "terminal_fit_metrics_have_final_intelligence_authority": False,
                "terminal_dense_structural_plant_installed_transactionally": True,
                "zero_training_pruning_quantization": True,
            }
        )
    return _issue_fused_semantic_terminal_receipt(_sealed(body, "result_sha256"))


def _authenticate_fused_terminal_cache_evidence(
    receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    compatibility_value = receipt.get("terminal_cache_compatibility")
    binding_value = receipt.get("terminal_cache_payload_binding")
    raw_values = receipt.get("cached_full_prefix_payload_verification")
    raw_sha256 = receipt.get("cached_full_prefix_payload_verification_sha256")
    cache_authentication = receipt.get("parent_cache_authentication")
    io_value = receipt.get("terminal_cache_io_audit")
    payload_load_value = receipt.get("terminal_cache_tensor_payload_load_audit")
    source_lineage_value = receipt.get("terminal_source_lineage")
    composite_lineage_value = receipt.get("terminal_composite_lineage")
    if (
        not all(
            isinstance(value, Mapping)
            for value in (
                compatibility_value,
                binding_value,
                cache_authentication,
                io_value,
                payload_load_value,
                source_lineage_value,
                composite_lineage_value,
            )
        )
        or not isinstance(raw_values, list)
        or not isinstance(raw_sha256, list)
    ):
        raise Lfm2SemanticPostfoldError(
            "Fused terminal omitted cache compatibility or raw-logit evidence."
        )
    compatibility = _verify_sealed(
        compatibility_value,
        seal_name="authentication_sha256",
        label="fused terminal cache compatibility",
    )
    binding = _verify_sealed(
        binding_value,
        seal_name="authentication_sha256",
        label="fused terminal cache payload binding",
    )
    io_audit = _verify_sealed(
        io_value,
        seal_name="result_sha256",
        label="fused terminal cache I/O audit",
    )
    payload_load_audit = _verify_sealed(
        payload_load_value,
        seal_name="authentication_sha256",
        label="fused terminal tensor-payload load audit",
    )
    try:
        source_lineage = Lfm2LongHorizonTerminalSourceLineage(source_lineage_value)
        composite_lineage = authenticate_lfm2_long_horizon_terminal_composite_lineage(
            composite_lineage_value,
            expected_source_lineage_sha256=(source_lineage.authentication_sha256),
        )
    except (TypeError, ValueError) as exc:
        raise Lfm2SemanticPostfoldError(
            "Fused terminal exact source/composite lineage is invalid."
        ) from exc
    composite_inputs = composite_lineage.document["compiler_inputs"]
    try:
        recomputed_payload_load_audit = _summarize_terminal_cache_payload_loads(
            cache_authentication
        )
    except (TypeError, ValueError) as exc:
        raise Lfm2SemanticPostfoldError(
            "Fused terminal tensor-payload load audit could not be rederived."
        ) from exc
    session_value = cache_authentication.get("session")
    if not isinstance(session_value, Mapping):
        raise Lfm2SemanticPostfoldError(
            "Fused terminal cache authentication has no session receipt."
        )
    session = _verify_sealed(
        session_value,
        seal_name="session_sha256",
        label="fused terminal selective cache session",
    )
    batch_count = compatibility.get("batch_count")
    batch_compatibility = compatibility.get("batch_compatibility")
    session_batches = session.get("ordered_authenticated_batches")
    preflight_io = compatibility.get("preflight_io_contract")
    cached_mode = compatibility.get("decode_trace_mode") == "cached_verified"
    same_pipeline_session = io_audit.get(
        "same_consuming_session_reused_from_layer_stack"
    )
    resume_mode = io_audit.get("preterminal_resume_mode")
    if (
        compatibility.get("schema")
        != LFM2_LONG_HORIZON_TERMINAL_CACHE_COMPATIBILITY_SCHEMA
        or compatibility.get("cache_key") != receipt.get("cache_key")
        or source_lineage.direct_local_checkpoint_source_authenticated is not True
        or source_lineage.document.get("cache_key") != receipt.get("cache_key")
        or source_lineage.document.get("cache_parent_fingerprint")
        != receipt.get("cache_parent_fingerprint")
        or source_lineage.document.get("bridge_artifact_sha256")
        != receipt.get("bridge_artifact_sha256")
        or receipt.get("terminal_source_lineage_sha256")
        != source_lineage.authentication_sha256
        or receipt.get("terminal_composite_lineage_sha256")
        != composite_lineage.authentication_sha256
        or receipt.get("terminal_compiler_binding_sha256")
        != composite_lineage.compiler_binding_sha256
        or composite_inputs.get("cache_key") != receipt.get("cache_key")
        or composite_inputs.get("cache_session_sha256") != session.get("session_sha256")
        or composite_inputs.get("cache_compatibility_sha256")
        != compatibility.get("authentication_sha256")
        or composite_inputs.get("cache_payload_binding_sha256")
        != binding.get("authentication_sha256")
        or receipt.get("parent_action_projection_sha256")
        != composite_inputs.get("parent_action_projection_sha256")
        or receipt.get("parent_action_projection_diagnostics")
        != composite_inputs.get("parent_action_projections")
        or receipt.get("parent_action_projection_numeric_threshold_applied")
        is not False
        or receipt.get("parent_action_projection_metrics_diagnostic_only") is not True
        or receipt.get("terminal_cache_compatibility_sha256")
        != compatibility.get("authentication_sha256")
        or type(batch_count) is not int
        or batch_count <= 0
        or not isinstance(batch_compatibility, list)
        or len(batch_compatibility) != batch_count
        or len(raw_values) != batch_count
        or len(raw_sha256) != batch_count
        or compatibility.get("terminal_trajectory_compatibility_verified") is not True
        or compatibility.get("forced_decode_tokens_are_caller_supplied") is not True
        or compatibility.get("top1_exactness_is_diagnostic_only") is not True
        or compatibility.get("top1_exactness_is_publication_gate") is not False
        or type(compatibility.get("all_top1_disagreements_numeric_envelope_verified"))
        is not bool
        or compatibility.get("full_prefix_recache_required") is not False
        or (
            cached_mode
            and type(compatibility.get("all_cached_logits_numerically_verified"))
            is not bool
        )
        or not isinstance(preflight_io, Mapping)
        or preflight_io.get("trace_file_sha256_passes") != 0
        or preflight_io.get("trace_payload_bytes_read") != 0
        or preflight_io.get("trace_tensor_payloads_materialized") != 0
        or not isinstance(session_batches, list)
        or len(session_batches) != batch_count
        or session.get("manifest_sha256") != compatibility.get("manifest_sha256")
        or session.get("completion_sha256") != compatibility.get("completion_sha256")
        or [value.get("record_sha256") for value in session_batches]
        != compatibility.get("ordered_batch_record_sha256")
        or [value.get("record_file_sha256") for value in session_batches]
        != compatibility.get("ordered_batch_record_file_sha256")
        or binding.get("compatibility_authentication_sha256")
        != compatibility.get("authentication_sha256")
        or binding.get("payload_session_sha256") != session.get("session_sha256")
        or binding.get("manifest_completion_and_ordered_records_exact") is not True
        or binding.get(
            "large_trace_file_sha256_authenticated_once_by_consuming_session"
        )
        is not True
        or io_audit.get("metadata_preflight_trace_file_sha256_passes") != 0
        or io_audit.get("metadata_preflight_trace_payload_bytes_read") != 0
        or io_audit.get("consuming_trace_file_sha256_passes_per_batch") != 1
        or io_audit.get("consuming_trace_file_sha256_passes_total") != batch_count
        or io_audit.get("same_consuming_session_reused_by_semantic_collector")
        is not True
        or type(same_pipeline_session) is not bool
        or io_audit.get("same_live_cache_session_as_layer_stack")
        is not same_pipeline_session
        or io_audit.get("cache_content_identity_matches_layer_stack") is not True
        or type(resume_mode) is not bool
        or (resume_mode and same_pipeline_session)
        or io_audit.get("fused_terminal_selective_session_constructions")
        != (0 if same_pipeline_session else 1)
        or io_audit.get("end_to_end_selective_session_constructions")
        != (1 if same_pipeline_session else 2)
        or io_audit.get(
            "additional_full_trace_file_sha256_passes_performed_by_fused_terminal"
        )
        != 0
        or dict(payload_load_audit) != recomputed_payload_load_audit
        or payload_load_audit.get("selected_tensor_payload_load_count") != batch_count
        or payload_load_audit.get("full_tensor_inventory_payload_load_count")
        != batch_count
        or payload_load_audit.get("every_batch_loaded_full_tensor_inventory_once")
        is not True
        or io_audit.get("selected_tensor_payload_loads_total") != batch_count
        or io_audit.get("selected_tensor_payload_logical_bytes_total")
        != payload_load_audit.get("selected_tensor_payload_logical_bytes_total")
        or io_audit.get("full_tensor_inventory_payload_loads_per_batch") != 1
        or io_audit.get("full_tensor_inventory_payload_loads_total") != batch_count
        or io_audit.get("full_tensor_inventory_payload_logical_bytes_total")
        != payload_load_audit.get("full_tensor_inventory_payload_logical_bytes_total")
    ):
        raise Lfm2SemanticPostfoldError(
            "Fused terminal cache compatibility/session binding is invalid."
        )
    authenticated_raw = []
    total_raw_pairs = 0
    total_raw_rows = 0
    total_raw_disagreements = 0
    for index, (raw_value, expected_sha, expected_batch) in enumerate(
        zip(raw_values, raw_sha256, batch_compatibility, strict=True)
    ):
        if not isinstance(raw_value, Mapping) or not isinstance(
            expected_batch,
            Mapping,
        ):
            raise Lfm2SemanticPostfoldError(
                "Fused terminal raw-logit batch evidence is malformed."
            )
        raw = _verify_sealed(
            raw_value,
            seal_name="authentication_sha256",
            label=f"fused terminal raw-logit batch {index}",
        )
        raw_pair_count = raw.get("raw_cached_full_prefix_pair_count")
        raw_row_count = raw.get("raw_logit_rows_verified")
        raw_disagreement_count = raw.get("top1_disagreement_count")
        if (
            raw.get("authentication_sha256") != expected_sha
            or raw.get("batch_index") != index
            or expected_batch.get("batch_index") != index
            or raw.get("decode_trace_mode") != expected_batch.get("decode_trace_mode")
            or raw_disagreement_count != expected_batch.get("top1_disagreement_count")
            or type(raw_pair_count) is not int
            or raw_pair_count < 0
            or type(raw_row_count) is not int
            or raw_row_count < 0
            or type(raw_disagreement_count) is not int
            or raw_disagreement_count < 0
            or type(raw.get("all_logits_numerically_close")) is not bool
            or type(raw.get("all_top1_disagreements_within_numeric_error_envelope"))
            is not bool
            or (
                cached_mode
                and raw.get("all_logits_numerically_close")
                is not expected_batch.get("cached_logits_numerically_verified")
            )
            or raw.get("all_top1_disagreements_within_numeric_error_envelope")
            is not expected_batch.get(
                "all_top1_disagreements_numeric_envelope_verified"
            )
            or raw.get("top1_exactness_is_diagnostic_only") is not True
        ):
            raise Lfm2SemanticPostfoldError(
                f"Fused terminal raw-logit batch {index} contradicts its sealed "
                "metadata preflight."
            )
        phase_values = raw.get("phase_verification")
        phase_sha256 = raw.get("phase_authentication_sha256")
        if raw.get("decode_trace_mode") == "cached_verified":
            expected_phase_count = expected_batch.get("verification_phase_count")
            expected_phase_order = expected_batch.get("verification_phase_order")
            if (
                type(expected_phase_count) is not int
                or expected_phase_count <= 0
                or not isinstance(expected_phase_order, list)
                or len(expected_phase_order) != expected_phase_count
                or not isinstance(phase_values, list)
                or not isinstance(phase_sha256, list)
                or len(phase_values) != expected_phase_count
                or len(phase_sha256) != len(phase_values)
                or raw_pair_count != len(phase_values)
            ):
                raise Lfm2SemanticPostfoldError(
                    f"Fused terminal raw-logit batch {index} omitted phase "
                    "verification receipts."
                )
            observed_disagreements = 0
            observed_rows = 0
            observed_all_close = True
            observed_all_envelope = True
            for phase_index, (phase_value, phase_sha) in enumerate(
                zip(phase_values, phase_sha256, strict=True)
            ):
                if not isinstance(phase_value, Mapping):
                    raise Lfm2SemanticPostfoldError(
                        "Fused terminal raw-logit phase receipt is malformed."
                    )
                phase = _verify_sealed(
                    phase_value,
                    seal_name="authentication_sha256",
                    label=(
                        f"fused terminal raw-logit batch {index} phase {phase_index}"
                    ),
                )
                forced_sha = phase.get("forced_token_sha256")
                phase_name = phase.get("phase")
                phase_rows = phase.get("row_count")
                phase_disagreements = phase.get("top1_disagreement_count")
                phase_exact = phase.get("top1_exact_agreement")
                maximum_error = phase.get("maximum_absolute_error")
                if (
                    phase.get("authentication_sha256") != phase_sha
                    or phase_name != expected_phase_order[phase_index]
                    or type(phase_rows) is not int
                    or phase_rows <= 0
                    or type(phase_disagreements) is not int
                    or phase_disagreements < 0
                    or type(phase_exact) is not bool
                    or phase_exact is not (phase_disagreements == 0)
                    or isinstance(maximum_error, bool)
                    or not isinstance(maximum_error, (int, float))
                    or not math.isfinite(float(maximum_error))
                    or float(maximum_error) < 0.0
                    or type(phase.get("numerically_close")) is not bool
                    or type(
                        phase.get(
                            "all_top1_disagreements_within_numeric_error_envelope"
                        )
                    )
                    is not bool
                    or (phase_index == 0 and forced_sha is not None)
                    or (
                        phase_index > 0
                        and (not isinstance(forced_sha, str) or len(forced_sha) != 64)
                    )
                ):
                    raise Lfm2SemanticPostfoldError(
                        "Fused terminal raw-logit phase receipt is inconsistent."
                    )
                _digest(
                    phase.get("cached_logits_sha256"),
                    "fused terminal cached-logit phase SHA-256",
                )
                _digest(
                    phase.get("full_prefix_logits_sha256"),
                    "fused terminal full-prefix-logit phase SHA-256",
                )
                observed_rows += phase_rows
                observed_disagreements += phase_disagreements
                observed_all_close &= phase["numerically_close"] is True
                observed_all_envelope &= (
                    phase["all_top1_disagreements_within_numeric_error_envelope"]
                    is True
                )
            if (
                observed_rows != raw_row_count
                or observed_disagreements != raw_disagreement_count
                or observed_all_close is not raw.get("all_logits_numerically_close")
                or observed_all_envelope
                is not raw.get("all_top1_disagreements_within_numeric_error_envelope")
            ):
                raise Lfm2SemanticPostfoldError(
                    "Fused terminal raw-logit phase receipt is inconsistent with "
                    "its batch aggregate."
                )
        elif (
            raw_pair_count != 0
            or raw_row_count != 0
            or raw_disagreement_count != 0
            or phase_values is not None
            or phase_sha256 is not None
        ):
            raise Lfm2SemanticPostfoldError(
                "Canonical terminal cache receipt claims cached logit pairs."
            )
        total_raw_pairs += raw_pair_count
        total_raw_rows += raw_row_count
        total_raw_disagreements += raw_disagreement_count
        authenticated_raw.append(raw["authentication_sha256"])
    all_raw_close = all(
        value.get("all_logits_numerically_close") is True
        for value in raw_values
        if isinstance(value, Mapping)
    )
    all_raw_envelope = all(
        value.get("all_top1_disagreements_within_numeric_error_envelope") is True
        for value in raw_values
        if isinstance(value, Mapping)
    )
    if (
        total_raw_disagreements != compatibility.get("cached_top1_disagreement_count")
        or (
            cached_mode
            and all_raw_close
            is not compatibility.get("all_cached_logits_numerically_verified")
        )
        or all_raw_envelope
        is not compatibility.get("all_top1_disagreements_numeric_envelope_verified")
    ):
        raise Lfm2SemanticPostfoldError(
            "Fused terminal raw-logit diagnostics contradict the cache "
            "compatibility aggregate."
        )
    body = {
        "terminal_source_lineage_sha256": source_lineage.authentication_sha256,
        "terminal_composite_lineage_sha256": (composite_lineage.authentication_sha256),
        "terminal_compiler_binding_sha256": (composite_lineage.compiler_binding_sha256),
        "terminal_cache_compatibility_sha256": compatibility["authentication_sha256"],
        "terminal_cache_payload_binding_sha256": binding["authentication_sha256"],
        "terminal_cache_session_sha256": session["session_sha256"],
        "terminal_cache_io_audit_sha256": io_audit["result_sha256"],
        "terminal_cache_tensor_payload_load_audit_sha256": payload_load_audit[
            "authentication_sha256"
        ],
        "raw_logit_batch_authentication_sha256": authenticated_raw,
        "batch_count": batch_count,
        "raw_cached_full_prefix_pair_count": total_raw_pairs,
        "raw_logit_rows_verified": total_raw_rows,
        "top1_disagreement_count": total_raw_disagreements,
        "all_logits_numerically_close": all_raw_close,
        "all_top1_disagreements_within_numeric_error_envelope": all_raw_envelope,
        "numeric_diagnostics_are_admission_authority": False,
        "exact_cache_lineage_is_admission_authority": True,
        "every_raw_cached_full_prefix_row_rederived": True,
    }
    return MappingProxyType(_sealed(body, "result_sha256"))


def _validate_review_pending_fused_terminal_receipt(
    receipt: Mapping[str, Any],
    *,
    layer_receipt: Mapping[str, Any],
    current_model_state_sha256: str,
    cache_evidence: Mapping[str, Any],
    terminal_context: Mapping[str, Any],
    layer_anchor_selection: Mapping[str, Any] | None,
    receipt_anchor_selection: Mapping[str, Any] | None,
    context_anchor_selection: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Authenticate the no-install boundary emitted before human review."""

    terminal_result_value = receipt.get("terminal_result")
    if not isinstance(terminal_result_value, Mapping):
        raise Lfm2SemanticPostfoldError(
            "Review-pending fused terminal omitted its terminal result."
        )
    terminal_result = _verify_sealed(
        terminal_result_value,
        seal_name="result_sha256",
        label="review-pending terminal-action result",
    )
    output_hashes = terminal_result.get("output_state_sha256")
    validation = terminal_result.get("validation")
    frontier = terminal_result.get("frontier")
    export_reference = receipt.get("terminal_frontier_export")
    if not isinstance(frontier, list) or not all(
        isinstance(value, Mapping) for value in frontier
    ):
        raise Lfm2SemanticPostfoldError(
            "Review-pending fused terminal omitted its complete frontier."
        )
    candidate_records = frontier[1:]
    candidate_ids = [str(value.get("candidate_id")) for value in candidate_records]
    eligible_ids = [
        str(value.get("candidate_id"))
        for value in candidate_records
        if value.get("structurally_eligible") is True
    ]
    diagnostic_order = terminal_result.get("diagnostic_order_candidate_ids")
    split_roles = terminal_result.get("split_roles")
    records_by_id = {
        str(value.get("candidate_id")): value for value in candidate_records
    }

    def structurally_authenticates_candidate(record: Mapping[str, Any]) -> bool:
        coverage_value = record.get("structural_coverage")
        if not isinstance(coverage_value, Mapping):
            return False
        coverage = dict(coverage_value)
        coverage_sha256 = coverage.pop("coverage_sha256", None)
        structural_safety = bool(
            coverage_sha256 == canonical_sha256(coverage)
            and record.get("structural_coverage_sha256") == coverage_sha256
            and coverage_value.get("tensor_sha256") == record.get("state_sha256")
            and coverage_value.get("all_elements_finite") is True
            and coverage_value.get("owns_exact_storage_allocation") is True
            and coverage_value.get("complete_dense_writer_structural_safety") is True
        )
        effectful = record.get("effectful_nonincumbent") is True
        eligible = bool(structural_safety and effectful)
        return bool(
            record.get("structural_safety_passed") is structural_safety
            and record.get("weight_norm_gate_passed") is structural_safety
            and record.get("structurally_eligible") is eligible
            and record.get("eligible") is eligible
            and record.get("candidate_eligibility_basis")
            == "finite_dense_shape_storage_coverage_and_nonincumbent_effect"
            and record.get("weight_delta_threshold_used_for_eligibility") is False
            and record.get("fit_thresholds_used_for_eligibility") is False
            and record.get("fit_check_thresholds_used_for_eligibility") is False
            and record.get("selection_thresholds_used_for_eligibility") is False
            and record.get("numeric_thresholds_used_for_publication") is False
            and record.get("numeric_metrics_used_for_final_selection") is False
            and record.get("numeric_metrics_have_final_intelligence_authority") is False
        )

    candidate_contract_valid = all(
        structurally_authenticates_candidate(record) for record in candidate_records
    )
    diagnostic_lead_id = terminal_result.get("diagnostic_lead_candidate_id")
    diagnostic_lead_record = records_by_id.get(str(diagnostic_lead_id))
    anchors_match = (
        (None if layer_anchor_selection is None else dict(layer_anchor_selection))
        == (
            None if receipt_anchor_selection is None else dict(receipt_anchor_selection)
        )
        == (
            None if context_anchor_selection is None else dict(context_anchor_selection)
        )
    )
    incumbent_sha256 = (
        output_hashes.get("incumbent") if isinstance(output_hashes, Mapping) else None
    )
    invalid = bool(
        receipt.get("schema") != LFM2_SEMANTIC_TERMINAL_REVIEW_PENDING_SCHEMA
        or receipt.get("semantic_layer_stack_result_sha256")
        != layer_receipt.get("result_sha256")
        or receipt.get("final_selection_status")
        != "deferred-to-manual-whole-response-review"
        or receipt.get("terminal_review_pending_schema")
        != LFM2_CPCF_TERMINAL_ACTION_REVIEW_PENDING_SCHEMA
        or receipt.get("final_selected_candidate_id") is not None
        or receipt.get("final_selection_performed") is not False
        or receipt.get("automatic_candidate_selection_performed") is not False
        or receipt.get("automatic_candidate_acceptance_performed") is not False
        or receipt.get("candidate_published_before_manual_review") is not False
        or receipt.get("manual_whole_response_review_required") is not True
        or receipt.get("diagnostic_order_has_final_selection_authority") is not False
        or receipt.get("terminal_writer_install_count") != 0
        or receipt.get("terminal_writer_mutation_count") != 0
        or receipt.get("accepted") is not False
        or receipt.get("terminal_publication_decision_accepted") is not False
        or receipt.get("operation_audit") != dict(_ZERO_OPERATIONS)
        or layer_receipt.get("model_state_after_sha256") != current_model_state_sha256
        or layer_receipt.get("whole_stage_contains_meaningful_parameter_changes")
        is not True
        or layer_receipt.get("model_state_before_sha256") == current_model_state_sha256
        or terminal_context.get("schema")
        != LFM2_SEMANTIC_TERMINAL_FRONTIER_EXPORT_CONTEXT_SCHEMA
        or terminal_context.get("semantic_layer_stack_result_sha256")
        != layer_receipt.get("result_sha256")
        or terminal_context.get("terminal_result_sha256")
        != terminal_result.get("result_sha256")
        or receipt.get("terminal_frontier_export_context_sha256")
        != terminal_context.get("context_sha256")
        or terminal_context.get("terminal_cache_compatibility_sha256")
        != cache_evidence.get("terminal_cache_compatibility_sha256")
        or terminal_context.get("terminal_cache_payload_binding_sha256")
        != cache_evidence.get("terminal_cache_payload_binding_sha256")
        or terminal_context.get("terminal_cache_session_sha256")
        != cache_evidence.get("terminal_cache_session_sha256")
        or not anchors_match
        or receipt.get("interface_anchor_selection_sha256")
        != (
            None
            if layer_anchor_selection is None
            else layer_anchor_selection["selection_sha256"]
        )
        or terminal_result.get("schema") != LFM2_CPCF_TERMINAL_ACTION_RESULT_SCHEMA
        or terminal_result.get("method") != _TERMINAL_ACTION_RESULT_METHOD
        or terminal_result.get("review_pending_schema")
        != LFM2_CPCF_TERMINAL_ACTION_REVIEW_PENDING_SCHEMA
        or terminal_result.get("final_selection_status")
        != "deferred-to-manual-whole-response-review"
        or terminal_result.get("final_selected_candidate_id") is not None
        or terminal_result.get("final_selection_performed") is not False
        or terminal_result.get("automatic_candidate_selection_performed") is not False
        or terminal_result.get("automatic_candidate_acceptance_performed") is not False
        or terminal_result.get("candidate_published_before_manual_review") is not False
        or terminal_result.get("manual_whole_response_review_required") is not True
        or terminal_result.get("diagnostic_order_is_provisional") is not True
        or terminal_result.get("diagnostic_order_has_final_selection_authority")
        is not False
        or terminal_result.get("structural_nonveto_schema")
        != LFM2_CPCF_TERMINAL_ACTION_STRUCTURAL_NONVETO_SCHEMA
        or terminal_result.get("candidate_eligibility_basis")
        != "finite_dense_shape_storage_coverage_and_nonincumbent_effect"
        or terminal_result.get("structural_safety_is_only_candidate_veto") is not True
        or terminal_result.get("numeric_metrics_used_for_diagnostic_ordering")
        is not True
        or terminal_result.get("numeric_metrics_used_for_final_selection") is not False
        or terminal_result.get("numeric_metrics_used_for_candidate_eligibility")
        is not False
        or terminal_result.get("numeric_thresholds_used_for_candidate_eligibility")
        is not False
        or terminal_result.get("numeric_thresholds_used_for_publication") is not False
        or terminal_result.get("validation_publication_veto_applied") is not False
        or terminal_result.get("rollback_authorized") is not False
        or terminal_result.get("numeric_metrics_have_final_intelligence_authority")
        is not False
        or terminal_result.get(
            "behavioral_intelligence_review_outside_terminal_compiler"
        )
        is not True
        or terminal_result.get("selected_candidate_id") != "incumbent"
        or terminal_result.get("accepted") is not False
        or terminal_result.get("selected_nonincumbent") is not False
        or terminal_result.get("published_is_incumbent") is not True
        or terminal_result.get("rollback_performed") is not False
        or terminal_result.get("rollback_exact") is not True
        or terminal_result.get("source_seed_writer_unchanged") is not True
        or terminal_result.get("parameter_identity_or_topology_mutation") is not False
        or terminal_result.get("parameter_count_before")
        != terminal_result.get("parameter_count_after")
        or terminal_result.get("published_writer_is_incumbent_pending_manual_review")
        is not True
        or terminal_result.get(
            "published_writer_is_structurally_safe_effectful_nonincumbent"
        )
        is not False
        or terminal_result.get(
            "numeric_diagnostic_miss_can_exclude_structural_candidate"
        )
        is not False
        or not isinstance(output_hashes, Mapping)
        or terminal_result.get("selected_state_sha256") != incumbent_sha256
        or output_hashes.get("selected") != incumbent_sha256
        or output_hashes.get("published") != incumbent_sha256
        or receipt.get("terminal_action_result_sha256")
        != terminal_result.get("result_sha256")
        or receipt.get("terminal_action_selected_w2_sha256") != incumbent_sha256
        or receipt.get("terminal_action_published_w2_sha256") != incumbent_sha256
        or receipt.get("semantic_selected_w2_sha256")
        != receipt.get("semantic_published_w2_sha256")
        or not isinstance(validation, Mapping)
        or validation.get("heldout_diagnostic_only") is not True
        or validation.get("publication_gate_applied") is not False
        or validation.get("used_for_final_selection") is not False
        or validation.get("used_for_candidate_acceptance") is not False
        or validation.get("used_for_fit") is not False
        or validation.get("used_for_diagnostic_ordering") is not False
        or not isinstance(split_roles, Mapping)
        or split_roles.get("candidate_eligibility_uses_only_structural_safety")
        is not True
        or split_roles.get("numeric_metrics_used_for_diagnostic_ordering") is not True
        or split_roles.get("numeric_metrics_used_for_final_selection") is not False
        or split_roles.get("manual_whole_response_review_required") is not True
        or split_roles.get("validation_diagnostic_only") is not True
        or split_roles.get("validation_publication_gate_applied") is not False
        or len(frontier) < 2
        or terminal_result.get("frontier_count") != len(frontier)
        or terminal_result.get("expected_frontier_count") != len(frontier)
        or frontier[0].get("candidate_id") != "incumbent"
        or frontier[0].get("eligible") is not False
        or frontier[0].get("structurally_eligible") is not False
        or frontier[0].get("effectful_nonincumbent") is not False
        or len(candidate_ids) != len(set(candidate_ids))
        or not candidate_contract_valid
        or not eligible_ids
        or not isinstance(diagnostic_order, list)
        or set(diagnostic_order) != set(eligible_ids)
        or len(diagnostic_order) != len(eligible_ids)
        or terminal_result.get("diagnostic_order_candidate_count") != len(eligible_ids)
        or diagnostic_lead_id != diagnostic_order[0]
        or not isinstance(diagnostic_lead_record, Mapping)
        or diagnostic_lead_record.get("structurally_eligible") is not True
        or diagnostic_lead_record.get("state_sha256")
        != terminal_result.get("diagnostic_lead_state_sha256")
        or terminal_result.get("diagnostic_lead_has_final_selection_authority")
        is not False
        or not isinstance(export_reference, Mapping)
        or not export_reference
        or receipt.get("terminal_frontier_export_sha256")
        != canonical_sha256(export_reference)
        or receipt.get("terminal_frontier_exported_before_w2_installation") is not True
        or terminal_result.get("operation_audit")
        != dict(_TERMINAL_ZERO_OPERATION_AUDIT)
    )
    if invalid:
        raise Lfm2SemanticPostfoldError(
            "Review-pending fused semantic terminal receipt is incomplete or false."
        )
    return MappingProxyType(dict(receipt))


def _validate_fused_semantic_terminal_receipt(
    receipt: Mapping[str, Any],
    *,
    layer_receipt: Mapping[str, Any],
    current_model_state_sha256: str,
) -> Mapping[str, Any]:
    receipt = _authenticate_fused_semantic_terminal_receipt_lineage(receipt)
    cache_evidence = _authenticate_fused_terminal_cache_evidence(receipt)
    recipe = _authenticate_body_recipe_binding(
        layer_receipt,
        label="fused terminal semantic layer-stack receipt",
    )
    layer_anchor_value = layer_receipt.get("interface_anchor_selection")
    layer_anchor_selection = (
        None
        if layer_anchor_value is None
        else _authenticate_foundation_anchor_selection(layer_anchor_value)
    )
    receipt_anchor_value = receipt.get("interface_anchor_selection")
    receipt_anchor_selection = (
        None
        if receipt_anchor_value is None
        else _authenticate_foundation_anchor_selection(receipt_anchor_value)
    )
    terminal_context_value = receipt.get("terminal_frontier_export_context")
    if not isinstance(terminal_context_value, Mapping):
        raise Lfm2SemanticPostfoldError(
            "Fused semantic terminal omitted its recipe-bound export context."
        )
    terminal_context = _verify_sealed(
        terminal_context_value,
        seal_name="context_sha256",
        label="fused semantic terminal export context",
    )
    _authenticate_body_recipe_binding(
        terminal_context,
        body_recipe=recipe,
        label="fused semantic terminal export context",
    )
    context_anchor_value = terminal_context.get("interface_anchor_selection")
    context_anchor_selection = (
        None
        if context_anchor_value is None
        else _authenticate_foundation_anchor_selection(context_anchor_value)
    )
    if receipt.get("schema") == LFM2_SEMANTIC_TERMINAL_REVIEW_PENDING_SCHEMA:
        return _validate_review_pending_fused_terminal_receipt(
            receipt,
            layer_receipt=layer_receipt,
            current_model_state_sha256=current_model_state_sha256,
            cache_evidence=cache_evidence,
            terminal_context=terminal_context,
            layer_anchor_selection=layer_anchor_selection,
            receipt_anchor_selection=receipt_anchor_selection,
            context_anchor_selection=context_anchor_selection,
        )
    layer_count = layer_receipt.get("layer_count")
    if (
        isinstance(layer_count, bool)
        or not isinstance(layer_count, int)
        or layer_count <= 0
    ):
        raise Lfm2SemanticPostfoldError(
            "Fused semantic terminal has no authenticated dynamic layer count."
        )
    terminal_layer_index = layer_count - 1
    installation = receipt.get("installation")
    if not isinstance(installation, Mapping):
        raise Lfm2SemanticPostfoldError(
            "Fused semantic terminal returned no installation decision."
        )
    coverage_before_value = receipt.get("final_mlp_producer_coverage_before_terminal")
    coverage_after_value = receipt.get("final_mlp_full_coverage_after_terminal")
    published_coverage_value = installation.get("published_w2_structural_coverage")
    if not all(
        isinstance(value, Mapping)
        for value in (
            coverage_before_value,
            coverage_after_value,
            published_coverage_value,
        )
    ):
        raise Lfm2SemanticPostfoldError(
            "Fused semantic terminal omitted structural producer coverage."
        )
    coverage_before = _verify_sealed(
        coverage_before_value,
        seal_name="result_sha256",
        label="preterminal final-MLP producer coverage",
    )
    coverage_after = _verify_sealed(
        coverage_after_value,
        seal_name="result_sha256",
        label="published final-MLP producer coverage",
    )
    published_coverage = _verify_sealed(
        published_coverage_value,
        seal_name="result_sha256",
        label="published final-W2 structural coverage",
    )
    before_roles = coverage_before.get("role_receipts")
    after_roles = coverage_after.get("role_receipts")
    w1_w3_unchanged = bool(
        isinstance(before_roles, Mapping)
        and isinstance(after_roles, Mapping)
        and all(
            isinstance(before_roles.get(role), Mapping)
            and isinstance(after_roles.get(role), Mapping)
            and before_roles[role].get("tensor_sha256")
            == after_roles[role].get("tensor_sha256")
            for role in ("feed_forward.w1.weight", "feed_forward.w3.weight")
        )
    )
    counts = {
        "decision": receipt.get("terminal_writer_decision_count"),
        "install": receipt.get("terminal_writer_install_count"),
        "mutation": receipt.get("terminal_writer_mutation_count"),
        "retention": receipt.get("terminal_writer_incumbent_retention_count"),
    }
    if any(type(value) is not int for value in counts.values()):
        raise Lfm2SemanticPostfoldError(
            "Fused semantic terminal decision counts are not exact integers."
        )
    operation_audit = receipt.get("operation_audit")
    installation_exact_zero = installation.get("exact_zero_cold_start") is True
    terminal_result_value = receipt.get("terminal_result")
    terminal_decision = receipt.get("terminal_publication_decision")
    if not isinstance(terminal_result_value, Mapping):
        raise Lfm2SemanticPostfoldError(
            "Fused semantic terminal omitted its authenticated terminal result."
        )
    terminal_result = _verify_sealed(
        terminal_result_value,
        seal_name="result_sha256",
        label="terminal-action result",
    )
    terminal_context_invalid = bool(
        terminal_context.get("schema")
        != LFM2_SEMANTIC_TERMINAL_FRONTIER_EXPORT_CONTEXT_SCHEMA
        or terminal_context.get("semantic_layer_stack_result_sha256")
        != layer_receipt.get("result_sha256")
        or terminal_context.get("terminal_result_sha256")
        != terminal_result.get("result_sha256")
        or receipt.get("terminal_frontier_export_context_sha256")
        != terminal_context.get("context_sha256")
        or terminal_context.get("terminal_cache_compatibility_sha256")
        != cache_evidence.get("terminal_cache_compatibility_sha256")
        or terminal_context.get("terminal_cache_payload_binding_sha256")
        != cache_evidence.get("terminal_cache_payload_binding_sha256")
        or terminal_context.get("terminal_cache_session_sha256")
        != cache_evidence.get("terminal_cache_session_sha256")
        or (None if layer_anchor_selection is None else dict(layer_anchor_selection))
        != (
            None if receipt_anchor_selection is None else dict(receipt_anchor_selection)
        )
        or (None if layer_anchor_selection is None else dict(layer_anchor_selection))
        != (
            None if context_anchor_selection is None else dict(context_anchor_selection)
        )
        or receipt.get("interface_anchor_selection_sha256")
        != (
            None
            if layer_anchor_selection is None
            else layer_anchor_selection["selection_sha256"]
        )
        or terminal_context.get("interface_anchor_selection_sha256")
        != receipt.get("interface_anchor_selection_sha256")
    )
    terminal_validation = terminal_result.get("validation")
    terminal_output_hashes = terminal_result.get("output_state_sha256")
    terminal_frontier = terminal_result.get("frontier")
    terminal_selected_id = terminal_result.get("selected_candidate_id")
    terminal_selected_record = None
    if isinstance(terminal_frontier, list):
        terminal_selected_record = next(
            (
                value
                for value in terminal_frontier
                if isinstance(value, Mapping)
                and value.get("candidate_id") == terminal_selected_id
            ),
            None,
        )
    terminal_selected_coverage = terminal_result.get(
        "selected_writer_structural_coverage"
    )
    terminal_decision_values = (
        terminal_decision if isinstance(terminal_decision, Mapping) else {}
    )
    terminal_output_values = (
        terminal_output_hashes if isinstance(terminal_output_hashes, Mapping) else {}
    )
    # Terminal-action digests include that compiler's tensor descriptor, while
    # semantic digests use this module's descriptor.  Bind each domain end to
    # end, but never compare values across the two domains.
    receipt_cross_binding_invalid = bool(
        terminal_result.get("result_sha256")
        != terminal_decision_values.get("terminal_action_result_sha256")
        or terminal_decision_values.get("terminal_action_result_sha256")
        != installation.get("terminal_action_result_sha256")
        or receipt.get("terminal_action_result_sha256")
        != terminal_decision_values.get("terminal_action_result_sha256")
        or terminal_output_values.get("selected")
        != terminal_decision_values.get("terminal_action_selected_w2_sha256")
        or terminal_decision_values.get("terminal_action_selected_w2_sha256")
        != installation.get("terminal_action_selected_w2_sha256")
        or receipt.get("terminal_action_selected_w2_sha256")
        != terminal_decision_values.get("terminal_action_selected_w2_sha256")
        or terminal_output_values.get("published")
        != terminal_decision_values.get("terminal_action_published_w2_sha256")
        or terminal_decision_values.get("terminal_action_published_w2_sha256")
        != installation.get("terminal_action_published_w2_sha256")
        or receipt.get("terminal_action_published_w2_sha256")
        != terminal_decision_values.get("terminal_action_published_w2_sha256")
        or terminal_decision_values.get("semantic_selected_w2_sha256")
        != terminal_decision_values.get("selected_w2_sha256")
        or terminal_decision_values.get("semantic_selected_w2_sha256")
        != installation.get("semantic_selected_w2_sha256")
        or receipt.get("semantic_selected_w2_sha256")
        != terminal_decision_values.get("semantic_selected_w2_sha256")
        or terminal_decision_values.get("semantic_published_w2_sha256")
        != terminal_decision_values.get("published_w2_sha256")
        or terminal_decision_values.get("semantic_published_w2_sha256")
        != installation.get("semantic_published_w2_sha256")
        or installation.get("semantic_published_w2_sha256")
        != installation.get("published_w2_sha256")
        or receipt.get("semantic_published_w2_sha256")
        != installation.get("semantic_published_w2_sha256")
    )
    terminal_result_invalid = bool(
        receipt_cross_binding_invalid
        or terminal_result.get("schema") != LFM2_CPCF_TERMINAL_ACTION_RESULT_SCHEMA
        or terminal_result.get("method") != _TERMINAL_ACTION_RESULT_METHOD
        or terminal_result.get("structural_nonveto_schema")
        != LFM2_CPCF_TERMINAL_ACTION_STRUCTURAL_NONVETO_SCHEMA
        or terminal_result.get("candidate_eligibility_basis")
        != "finite_dense_shape_storage_coverage_and_nonincumbent_effect"
        or terminal_result.get("structural_safety_is_only_candidate_veto") is not True
        or terminal_result.get("numeric_metrics_used_for_candidate_eligibility")
        is not False
        or terminal_result.get("numeric_thresholds_used_for_candidate_eligibility")
        is not False
        or terminal_result.get("numeric_thresholds_used_for_publication") is not False
        or terminal_result.get("validation_publication_veto_applied") is not False
        or terminal_result.get("rollback_authorized") is not False
        or terminal_result.get("numeric_metrics_have_final_intelligence_authority")
        is not False
        or terminal_result.get(
            "behavioral_intelligence_review_outside_terminal_compiler"
        )
        is not True
        or terminal_result.get(
            "published_writer_is_structurally_safe_effectful_nonincumbent"
        )
        is not True
        or terminal_result.get("accepted") is not True
        or terminal_result.get("selected_nonincumbent") is not True
        or not isinstance(terminal_selected_id, str)
        or not terminal_selected_id
        or terminal_selected_id == "incumbent"
        or terminal_result.get("published_is_incumbent") is not False
        or terminal_result.get("rollback_performed") is not False
        or terminal_result.get("rollback_exact") is not True
        or terminal_result.get("source_seed_writer_unchanged") is not True
        or terminal_result.get("parameter_identity_or_topology_mutation") is not False
        or terminal_result.get("parameter_count_before")
        != terminal_result.get("parameter_count_after")
        or not isinstance(terminal_output_hashes, Mapping)
        or terminal_output_hashes.get("selected")
        != terminal_output_hashes.get("published")
        or terminal_output_hashes.get("selected")
        == terminal_output_hashes.get("incumbent")
        or not isinstance(terminal_selected_record, Mapping)
        or terminal_selected_record.get("state_sha256")
        != terminal_output_hashes.get("selected")
        or terminal_selected_record.get("eligible") is not True
        or terminal_selected_record.get("structurally_eligible") is not True
        or terminal_selected_record.get("effectful_nonincumbent") is not True
        or terminal_selected_record.get("structural_safety_passed") is not True
        or not isinstance(terminal_selected_coverage, Mapping)
        or terminal_selected_coverage.get("tensor_sha256")
        != terminal_output_hashes.get("selected")
        or terminal_selected_coverage.get("owns_exact_storage_allocation") is not True
        or terminal_selected_coverage.get("complete_dense_writer_structural_safety")
        is not True
        or not isinstance(terminal_validation, Mapping)
        or type(terminal_validation.get("passed")) is not bool
        or terminal_validation.get("heldout_diagnostic_only") is not True
        or terminal_validation.get("publication_gate_applied") is not False
        or terminal_validation.get("used_only_for_frozen_candidate_acceptance")
        is not False
        or terminal_result.get("operation_audit")
        != dict(_TERMINAL_ZERO_OPERATION_AUDIT)
        or not isinstance(terminal_decision, Mapping)
        or terminal_decision.get("terminal_publication_decision_accepted") is not True
        or terminal_decision.get("terminal_nonincumbent_candidate_accepted") is not True
        or terminal_decision.get("terminal_incumbent_retained") is not False
        or terminal_decision.get("terminal_writer_changed") is not True
        or terminal_decision.get("structural_candidate_eligibility_authenticated")
        is not True
        or terminal_decision.get("heldout_metrics_diagnostic_only") is not True
        or terminal_decision.get("validation_publication_gate_applied") is not False
        or terminal_decision.get("numeric_metrics_have_final_intelligence_authority")
        is not False
        or terminal_decision.get("published_is_selected") is not True
        or terminal_decision.get("published_is_incumbent") is not False
        or terminal_decision.get("published_w2_sha256")
        != terminal_decision.get("selected_w2_sha256")
        or terminal_decision.get("published_w2_sha256")
        != installation.get("published_w2_sha256")
    )
    fused_heldout = receipt.get("fused_heldout_diagnostics")
    separate_heldout = receipt.get("separate_heldout_lane_diagnostics")
    heldout_rows = (
        list(fused_heldout.values()) if isinstance(fused_heldout, Mapping) else []
    )
    if isinstance(separate_heldout, Mapping):
        heldout_rows.extend(
            value
            for lane in separate_heldout.values()
            if isinstance(lane, Mapping)
            for value in lane.values()
        )
    heldout_diagnostics_invalid = len(heldout_rows) != 8
    if not heldout_diagnostics_invalid:
        for index, value in enumerate(heldout_rows):
            if not isinstance(value, Mapping):
                heldout_diagnostics_invalid = True
                break
            diagnostic = _verify_sealed(
                value,
                seal_name="diagnostic_sha256",
                label=f"terminal heldout diagnostic {index}",
            )
            if (
                diagnostic.get("role")
                != "diagnostic-only-after-frozen-terminal-publication-decision"
                or diagnostic.get("publication_gate_applied") is not False
                or type(diagnostic.get("passed")) is not bool
            ):
                heldout_diagnostics_invalid = True
                break
    cold_receipt_invalid = False
    if installation_exact_zero:
        terminal_operations = terminal_result.get("operation_audit")
        cold_receipt_invalid = bool(
            receipt.get("terminal_exact_zero_cold_start") is not True
            or receipt.get("terminal_incumbent_noop_publication_forbidden") is not True
            or receipt.get("terminal_structural_candidate_eligibility_authenticated")
            is not True
            or receipt.get(
                "terminal_fit_metric_ordering_frozen_before_heldout_diagnostics"
            )
            is not True
            or receipt.get("terminal_fit_metrics_used_for_candidate_ordering")
            is not True
            or receipt.get(
                "terminal_fit_metrics_used_for_pre_review_candidate_selection"
            )
            is not True
            or receipt.get("terminal_fit_metrics_used_for_structural_eligibility")
            is not False
            or receipt.get(
                "terminal_fit_metric_thresholds_used_for_candidate_eligibility"
            )
            is not False
            or receipt.get("terminal_fit_metric_thresholds_used_for_publication")
            is not False
            or receipt.get("terminal_fit_metrics_have_final_intelligence_authority")
            is not False
            or receipt.get("terminal_heldout_metrics_diagnostic_only") is not True
            or receipt.get("terminal_heldout_publication_gate_applied") is not False
            or receipt.get("terminal_dense_structural_plant_installed_transactionally")
            is not True
            or receipt.get("zero_training_pruning_quantization") is not True
            or not isinstance(terminal_decision, Mapping)
            or terminal_decision.get("terminal_exact_zero_cold_start") is not True
            or terminal_decision.get("incumbent_noop_publication_forbidden") is not True
            or terminal_decision.get("structural_candidate_eligibility_authenticated")
            is not True
            or terminal_decision.get(
                "fit_metric_ordering_frozen_before_heldout_diagnostics"
            )
            is not True
            or terminal_decision.get("fit_metrics_used_for_candidate_ordering")
            is not True
            or terminal_decision.get(
                "fit_metrics_used_for_pre_review_candidate_selection"
            )
            is not True
            or terminal_decision.get("fit_metrics_used_for_structural_eligibility")
            is not False
            or terminal_decision.get(
                "fit_metric_thresholds_used_for_candidate_eligibility"
            )
            is not False
            or terminal_decision.get("fit_metric_thresholds_used_for_publication")
            is not False
            or terminal_decision.get("fit_metrics_have_final_intelligence_authority")
            is not False
            or terminal_decision.get("heldout_metrics_diagnostic_only") is not True
            or terminal_decision.get("published_w2_complete_dense_structural_plant")
            is not True
            or terminal_result.get("exact_zero_cold_start") is not True
            or terminal_result.get("incumbent_all_elements_zero") is not True
            or terminal_result.get("incumbent_noop_publication_forbidden") is not True
            or terminal_result.get("exact_zero_candidate_eligibility_schema")
            != LFM2_CPCF_TERMINAL_ACTION_EXACT_ZERO_ELIGIBILITY_SCHEMA
            or terminal_result.get("candidate_selection_evidence_scope")
            != "finite-dense-structural-eligibility-plus-fit-ordering"
            or terminal_result.get("candidate_eligibility_basis")
            != "finite_dense_shape_storage_coverage_and_nonincumbent_effect"
            or terminal_result.get("structural_nonveto_schema")
            != LFM2_CPCF_TERMINAL_ACTION_STRUCTURAL_NONVETO_SCHEMA
            or terminal_result.get("structural_safety_is_only_candidate_veto")
            is not True
            or terminal_result.get("numeric_metrics_used_for_candidate_eligibility")
            is not False
            or terminal_result.get("numeric_thresholds_used_for_candidate_eligibility")
            is not False
            or terminal_result.get("numeric_thresholds_used_for_publication")
            is not False
            or terminal_result.get("validation_publication_veto_applied") is not False
            or terminal_result.get("rollback_authorized") is not False
            or terminal_result.get(
                "behavioral_intelligence_review_outside_terminal_compiler"
            )
            is not True
            or terminal_result.get("fit_metrics_role")
            != "ordering-among-structurally-eligible-not-threshold-gating"
            or terminal_result.get("fit_metrics_used_for_candidate_ordering")
            is not True
            or terminal_result.get(
                "fit_metrics_used_for_pre_review_candidate_selection"
            )
            is not True
            or terminal_result.get("fit_metrics_used_for_structural_eligibility")
            is not False
            or terminal_result.get("fit_metrics_have_final_intelligence_authority")
            is not False
            or terminal_result.get(
                "fit_metric_thresholds_used_for_candidate_eligibility"
            )
            is not False
            or terminal_result.get("fit_metric_thresholds_used_for_publication")
            is not False
            or terminal_result.get("relative_to_zero_weight_delta_gate_applied")
            is not False
            or terminal_result.get("selected_before_any_heldout_publication_diagnostic")
            is not True
            or terminal_result.get(
                "published_writer_is_finite_dense_nonzero_structural_plant"
            )
            is not True
            or terminal_result.get("nonfinite_candidate_or_replay_is_hard_failure")
            is not True
            or terminal_result.get("zero_training_pruning_quantization") is not True
            or not isinstance(terminal_validation, Mapping)
            or terminal_validation.get("heldout_diagnostic_only") is not True
            or terminal_validation.get("publication_gate_applied") is not False
            or terminal_validation.get("used_only_for_frozen_candidate_acceptance")
            is not False
            or not isinstance(terminal_operations, Mapping)
            or not terminal_operations
            or any(value != 0 for value in terminal_operations.values())
            or installation.get("incumbent_all_elements_zero") is not True
            or installation.get("incumbent_noop_publication_forbidden") is not True
            or installation.get("published_w2_all_elements_finite") is not True
            or installation.get("published_w2_all_output_units_covered") is not True
            or installation.get("published_w2_all_input_units_covered") is not True
            or installation.get("transactional_dense_cold_start_plant") is not True
        )
    else:
        cold_receipt_invalid = bool(
            receipt.get("terminal_exact_zero_cold_start") is True
            or (
                isinstance(terminal_result_value, Mapping)
                and terminal_result_value.get("exact_zero_cold_start") is True
            )
            or (
                isinstance(terminal_decision, Mapping)
                and terminal_decision.get("terminal_exact_zero_cold_start") is True
            )
        )
    invalid = bool(
        receipt.get("schema") != LFM2_SEMANTIC_LEXICAL_COMPLETION_SCHEMA
        or terminal_context_invalid
        or receipt.get("semantic_layer_stack_result_sha256")
        != layer_receipt.get("result_sha256")
        or receipt.get("accepted") is not True
        or receipt.get("terminal_publication_decision_accepted") is not True
        or receipt.get("terminal_heldout_metrics_diagnostic_only") is not True
        or receipt.get("terminal_heldout_publication_gate_applied") is not False
        or receipt.get("terminal_nonincumbent_candidate_accepted") is not True
        or receipt.get("terminal_incumbent_retained") is not False
        or terminal_result_invalid
        or heldout_diagnostics_invalid
        or counts["decision"] != 1
        or counts["install"] != 1
        or counts["mutation"] != 1
        or counts["retention"] != 0
        or installation.get("decision_count") != 1
        or installation.get("install_count") != counts["install"]
        or installation.get("mutation_count") != counts["mutation"]
        or installation.get("incumbent_retention_count") != counts["retention"]
        or installation.get("parameter_topology_preserved") is not True
        or installation.get("parameter_identities_preserved") is not True
        or installation.get("parameter_storage_preserved") is not True
        or installation.get("gradient_state_preserved") is not True
        or installation.get("parameter_count_changed") is not False
        or installation.get("published_w2_structurally_nonzero") is not True
        or installation.get("published_w2_structural_coverage_sha256")
        != published_coverage.get("result_sha256")
        or published_coverage.get("structurally_nonzero") is not True
        or published_coverage.get("tensor_sha256")
        != installation.get("published_w2_sha256")
        or receipt.get("complete_nonzero_final_mlp_structural_coverage") is not True
        or coverage_before.get("producer_kind") != "mlp"
        or coverage_before.get("layer_index") != terminal_layer_index
        or coverage_before.get("deferred_roles") != ["feed_forward.w2.weight"]
        or coverage_before.get("all_required_roles_structurally_nonzero") is not True
        or coverage_after.get("producer_kind") != "mlp"
        or coverage_after.get("layer_index") != terminal_layer_index
        or coverage_after.get("deferred_roles") != []
        or coverage_after.get("all_required_roles_structurally_nonzero") is not True
        or coverage_after.get("blank_or_partially_blank_required_role_count") != 0
        or not w1_w3_unchanged
        or not isinstance(after_roles, Mapping)
        or not isinstance(after_roles.get("feed_forward.w2.weight"), Mapping)
        or after_roles["feed_forward.w2.weight"].get("tensor_sha256")
        != published_coverage.get("tensor_sha256")
        or installation.get("model_state_before_sha256")
        != layer_receipt.get("model_state_after_sha256")
        or installation.get("model_state_after_sha256") != current_model_state_sha256
        or not isinstance(operation_audit, Mapping)
        or operation_audit != dict(_ZERO_OPERATIONS)
        or layer_receipt.get("whole_stage_contains_meaningful_parameter_changes")
        is not True
        or layer_receipt.get("model_state_before_sha256") == current_model_state_sha256
        or cold_receipt_invalid
    )
    invalid = bool(
        invalid
        or installation.get("installed") is not True
        or installation.get("exact_zero_copy_noop") is not False
        or installation.get("parameter_copy_calls") != 1
        or not installation.get("changed_state_keys")
        or installation.get("model_state_before_sha256")
        == installation.get("model_state_after_sha256")
    )
    if invalid:
        raise Lfm2SemanticPostfoldError(
            "Fused semantic terminal receipt is incomplete or internally false."
        )
    return MappingProxyType(dict(receipt))


def _ffn_width(model: Any) -> int:
    widths = set()
    for layer in tuple(getattr(getattr(model, "model", None), "layers", ())):
        feed_forward = getattr(layer, "feed_forward", None)
        w1 = getattr(getattr(feed_forward, "w1", None), "weight", None)
        w2 = getattr(getattr(feed_forward, "w2", None), "weight", None)
        w3 = getattr(getattr(feed_forward, "w3", None), "weight", None)
        if w1 is None or w2 is None or w3 is None:
            continue
        if int(w1.shape[0]) != int(w3.shape[0]) or int(w2.shape[1]) != int(w1.shape[0]):
            raise Lfm2SemanticPostfoldError("Child FFN shapes are inconsistent.")
        widths.add(int(w1.shape[0]))
    if len(widths) != 1:
        raise Lfm2SemanticPostfoldError("Child does not expose one dense FFN width.")
    return next(iter(widths))


def _safe_remove_staging(path: Path, parent: Path) -> None:
    try:
        resolved = path.resolve()
        root = parent.resolve()
    except OSError:
        return
    if (
        resolved.parent == root
        and resolved.name.startswith(".semantic-postfold-")
        and resolved.is_dir()
        and not resolved.is_symlink()
    ):
        shutil.rmtree(resolved)


def _safe_remove_failed_publication(path: Path, expected: Path) -> None:
    """Remove only the exact new output created by this failed transaction."""

    try:
        resolved = path.resolve()
        intended = expected.resolve()
    except OSError:
        return
    if (
        resolved == intended
        and resolved.parent == intended.parent
        and resolved.is_dir()
        and not resolved.is_symlink()
    ):
        shutil.rmtree(resolved)


def publish_lfm2_semantic_postfold_checkpoint(
    *,
    target_model: Any,
    target_tokenizer: Any,
    output_directory: str | os.PathLike[str],
    context_binding_sha256: str,
    interface_anchor_selection_sha256: str | None = None,
    device: str,
    dtype: str = "bfloat16",
) -> tuple[Any, Mapping[str, Any]]:
    """Atomically save and twice reload one dense parent-free checkpoint."""

    output = _new_output_path(output_directory)
    context = _digest(context_binding_sha256, "postfold context binding")
    anchor_selection_sha256 = (
        None
        if interface_anchor_selection_sha256 is None
        else _digest(
            interface_anchor_selection_sha256,
            "checkpoint interface anchor selection",
        )
    )
    model_state = fingerprint_model_state(target_model)
    inventory = independent_parameter_inventory(target_model)
    inventory_sha = _digest(
        inventory.get("inventory_sha256"),
        "postfold parameter inventory",
    )
    tokenizer_sha = fingerprint_hf_tokenizer_semantics(target_tokenizer)
    width = _ffn_width(target_model)
    saver = Lfm2CpcfV5QuarantineSaver(
        target_tokenizer=target_tokenizer,
        context_binding_sha256=context,
        expected_ffn_width=width,
        expected_model_state_sha256=model_state,
        expected_parameter_inventory_sha256=inventory_sha,
        expected_tokenizer_semantics_sha256=tokenizer_sha,
    )
    loader = Lfm2CpcfV5StrictLocalLoader(
        context_binding_sha256=context,
        device=device,
        dtype=dtype,
        expected_ffn_width=width,
        expected_model_state_sha256=model_state,
        expected_parameter_inventory_sha256=inventory_sha,
        expected_tokenizer_semantics_sha256=tokenizer_sha,
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=".semantic-postfold-",
            dir=str(output.parent),
        )
    ).resolve()
    publication_moved = False
    try:
        saver(target_model, staging)
        staged = loader(staging)
        if fingerprint_model_state(staged) != model_state:
            raise Lfm2SemanticPostfoldError(
                "Staged checkpoint reload differs from the postfold child."
            )
        staged_metadata = _input_artifact_metadata(
            staging,
            expected_artifact_sha256=_read_json(
                staging / LFM2_CPCF_V5_ARTIFACT_METADATA,
                "staged artifact metadata",
            )["artifact_sha256"],
        )
        staged_inventory = canonical_standalone_fold_artifact_inventory(staging)
        staged_file_roles = {
            value["path"]: value["artifact_role"] for value in staged_inventory
        }
        staged_artifact_tree = scan_standalone_fold_artifact_tree(
            staging,
            expected_file_roles=staged_file_roles,
        )
        os.replace(staging, output)
        publication_moved = True
        reloaded = loader(output)
        if fingerprint_model_state(reloaded) != model_state or _parameter_count(
            reloaded
        ) != _parameter_count(target_model):
            raise Lfm2SemanticPostfoldError(
                "Published checkpoint reload differs from the postfold child."
            )
        metadata = _input_artifact_metadata(
            output,
            expected_artifact_sha256=_read_json(
                output / LFM2_CPCF_V5_ARTIFACT_METADATA,
                "published artifact metadata",
            )["artifact_sha256"],
        )
        artifact_inventory = canonical_standalone_fold_artifact_inventory(output)
        artifact_file_roles = {
            value["path"]: value["artifact_role"] for value in artifact_inventory
        }
        artifact_tree = scan_standalone_fold_artifact_tree(
            output,
            expected_file_roles=artifact_file_roles,
        )
        if (
            metadata != staged_metadata
            or artifact_inventory != staged_inventory
            or artifact_tree != staged_artifact_tree
        ):
            raise Lfm2SemanticPostfoldError(
                "Published checkpoint inventory differs from the staged child."
            )
    except BaseException:
        if staging.exists():
            _safe_remove_staging(staging, output.parent)
        if publication_moved and output.exists():
            _safe_remove_failed_publication(output, output)
        raise
    artifact_tree_sha256 = canonical_sha256(artifact_tree)
    artifact_inventory_sha256 = canonical_sha256(artifact_inventory)
    body = {
        "schema": LFM2_SEMANTIC_POSTFOLD_CHECKPOINT_SCHEMA,
        "output_directory": str(output),
        "context_binding_sha256": context,
        "interface_anchor_selection_sha256": anchor_selection_sha256,
        "artifact_sha256": metadata["artifact_sha256"],
        "model_state_sha256": model_state,
        "parameter_inventory_sha256": inventory_sha,
        "tokenizer_semantics_sha256": tokenizer_sha,
        "parameter_element_count": _parameter_count(reloaded),
        "materialized_ffn_width": width,
        "staged_reload_verified": True,
        "published_reload_verified": True,
        "standalone_dense_safetensors": True,
        "parent_model_required_for_reload": False,
        "cache_required_for_reload": False,
        "network_access_performed": False,
        "artifact_tree": [dict(value) for value in artifact_tree],
        "artifact_tree_sha256": artifact_tree_sha256,
        "artifact_file_count": len(artifact_tree),
        "artifact_total_bytes": sum(
            int(value["size_bytes"]) for value in artifact_tree
        ),
        "complete_artifact_tree_hashed": True,
        "artifact_inventory_schema": STANDALONE_FOLD_ARTIFACT_INVENTORY_SCHEMA,
        "artifact_inventory": [dict(value) for value in artifact_inventory],
        "artifact_inventory_sha256": artifact_inventory_sha256,
        "canonical_typed_artifact_inventory_authenticated": True,
        "staged_artifact_tree_purity_verified_before_publication": True,
        "unallowlisted_artifact_files_present": False,
        "forbidden_named_or_referenced_sidecars_present": False,
        "standalone_directory_purity_authenticated": True,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    return reloaded, MappingProxyType(_sealed(body, "result_sha256"))


def publish_preterminal_capsule_then_compile_fused_terminal(
    *,
    capsule_publisher: Callable[[], Any] | None,
    fused_terminal_compiler: Callable[..., Mapping[str, Any]],
    terminal_arguments: Mapping[str, Any],
    terminal_frontier_exporter_factory: Callable[[Any], Callable[..., Any]]
    | None = None,
) -> tuple[Any | None, Mapping[str, Any]]:
    """Publish a durable seam first, then call the fallible terminal compiler."""

    if capsule_publisher is not None and not callable(capsule_publisher):
        raise TypeError("capsule_publisher must be callable or None.")
    if not callable(fused_terminal_compiler):
        raise TypeError("fused_terminal_compiler must be callable.")
    if not isinstance(terminal_arguments, Mapping):
        raise TypeError("terminal_arguments must be a mapping.")
    if terminal_frontier_exporter_factory is not None and not callable(
        terminal_frontier_exporter_factory
    ):
        raise TypeError("terminal_frontier_exporter_factory must be callable or None.")
    if terminal_frontier_exporter_factory is not None and capsule_publisher is None:
        raise Lfm2SemanticPostfoldError(
            "A terminal frontier exporter factory requires a preterminal capsule "
            "publisher."
        )
    arguments = dict(terminal_arguments)
    if (
        terminal_frontier_exporter_factory is not None
        and "terminal_frontier_exporter" in arguments
    ):
        raise Lfm2SemanticPostfoldError(
            "A direct terminal frontier exporter and capsule-aware exporter factory "
            "are mutually exclusive."
        )
    capsule_reference = None if capsule_publisher is None else capsule_publisher()
    if terminal_frontier_exporter_factory is not None:
        if capsule_reference is None:
            raise Lfm2SemanticPostfoldError(
                "The preterminal capsule publisher returned no reference for the "
                "terminal frontier exporter factory."
            )
        exporter = terminal_frontier_exporter_factory(capsule_reference)
        if not callable(exporter):
            raise Lfm2SemanticPostfoldError(
                "The terminal frontier exporter factory returned no callable exporter."
            )
        arguments["terminal_frontier_exporter"] = exporter
    terminal = fused_terminal_compiler(**arguments)
    if not isinstance(terminal, Mapping):
        raise Lfm2SemanticPostfoldError(
            "Fused terminal compiler returned no receipt mapping."
        )
    return capsule_reference, terminal


def run_lfm2_semantic_postfold(
    *,
    input_checkpoint_directory: str | os.PathLike[str],
    expected_input_artifact_sha256: str,
    output_checkpoint_directory: str | os.PathLike[str],
    cache_directory: str | os.PathLike[str],
    expected_cache_key: str,
    bridge_directory: str | os.PathLike[str],
    expected_bridge_artifact_sha256: str,
    donor_tokenizer: Any,
    target_tokenizer: Any,
    parent_model_directory: str | os.PathLike[str],
    compiler_diagnostic_json: str | os.PathLike[str],
    expected_compiler_diagnostic_sha256: str,
    device: str,
    dtype: str = "bfloat16",
    fold_compute_device: str = "cpu",
    global_residual_closure_version: str = "producer_aware_v2a",
    correction_scale_frontier: Sequence[float] = (1.0,),
    correction_scale: float = 1.0,
    global_residual_backend: str = "auto",
    global_residual_basis_rank: int | None = None,
    body_recipe: str | Lfm2SemanticBodyRecipe = "fit_guided",
    parent_final_norm_tensor_key: str = "model.embedding_norm.weight",
    parent_norm_epsilon_config_key: str = "norm_eps",
    parent_tied_tensor_key: str = "model.embed_tokens.weight",
    tied_interface_mode: str = TIED_RESIDUAL_STRONG_CONGRUENCE_MODE,
    interface_statistics_directory: str | os.PathLike[str] | None = None,
    expected_interface_statistics_sha256: str | None = None,
    interface_frontier_companion_json: str | os.PathLike[str] | None = None,
    expected_interface_frontier_companion_file_sha256: str | None = None,
    expected_interface_frontier_companion_sha256: str | None = None,
    anchor_candidate_index: int | None = None,
    attention_installer: Callable[..., Mapping[str, Any]] | None = None,
    semantic_layer_stack_compiler: Callable[..., Any] | None = None,
    preterminal_capsule_directory: str | os.PathLike[str] | None = None,
    fused_terminal_compiler: Callable[..., Mapping[str, Any]] = (
        compile_install_lfm2_fused_semantic_terminal_refold
    ),
    terminal_frontier_exporter: Callable[
        [Lfm2CpcfTerminalActionResidualResult, Mapping[str, Any]],
        Any,
    ]
    | None = None,
    terminal_frontier_exporter_factory: Callable[[Any], Callable[..., Any]]
    | None = None,
    split_plan: Lfm2LongHorizonTrajectorySplitPlan | None = None,
) -> Mapping[str, Any]:
    """Execute the complete saved-child semantic postfold transaction."""

    recipe = resolve_lfm2_semantic_body_recipe(body_recipe)
    closure_version = str(global_residual_closure_version).strip().lower()
    closure_policy = Lfm2GlobalResidualClosurePolicy(
        backend=str(global_residual_backend).strip().lower()
    )
    scale_policy = Lfm2ProducerAwareCorrectionScalePolicy(
        materialization_frontier=tuple(
            float(value) for value in correction_scale_frontier
        ),
        materialization_scale=float(correction_scale),
    )
    if global_residual_basis_rank is not None and (
        isinstance(global_residual_basis_rank, bool)
        or not isinstance(global_residual_basis_rank, int)
        or global_residual_basis_rank <= 0
    ):
        raise Lfm2SemanticPostfoldError(
            "global_residual_basis_rank must be a positive integer or None."
        )
    if closure_version not in {
        "producer_aware_v2a",
        "output_writer_v1_diagnostic",
    }:
        raise Lfm2SemanticPostfoldError(
            "global_residual_closure_version must be producer_aware_v2a or "
            "output_writer_v1_diagnostic."
        )
    if terminal_frontier_exporter is not None and not callable(
        terminal_frontier_exporter
    ):
        raise TypeError("terminal_frontier_exporter must be callable or None.")
    if terminal_frontier_exporter_factory is not None and not callable(
        terminal_frontier_exporter_factory
    ):
        raise TypeError("terminal_frontier_exporter_factory must be callable or None.")
    if (
        terminal_frontier_exporter is not None
        and terminal_frontier_exporter_factory is not None
    ):
        raise Lfm2SemanticPostfoldError(
            "A direct terminal frontier exporter and capsule-aware exporter factory "
            "are mutually exclusive."
        )
    if (
        terminal_frontier_exporter_factory is not None
        and preterminal_capsule_directory is None
    ):
        raise Lfm2SemanticPostfoldError(
            "A terminal frontier exporter factory requires a preterminal capsule "
            "directory."
        )
    statistics_requested = interface_statistics_directory is not None
    if statistics_requested != (expected_interface_statistics_sha256 is not None):
        raise Lfm2SemanticPostfoldError(
            "Interface statistics directory and external SHA-256 pin must be "
            "supplied together."
        )
    expected_statistics_sha256 = (
        _digest(
            expected_interface_statistics_sha256,
            "expected interface statistics SHA-256",
        )
        if statistics_requested
        else None
    )
    companion_values = (
        interface_frontier_companion_json,
        expected_interface_frontier_companion_file_sha256,
        expected_interface_frontier_companion_sha256,
        anchor_candidate_index,
    )
    companion_requested = any(value is not None for value in companion_values)
    if companion_requested and any(value is None for value in companion_values):
        raise Lfm2SemanticPostfoldError(
            "Interface frontier companion JSON, file pin, canonical pin, and anchor "
            "candidate index must be supplied as one complete set."
        )
    if companion_requested and not statistics_requested:
        raise Lfm2SemanticPostfoldError(
            "An interface frontier companion requires pinned interface statistics."
        )
    if anchor_candidate_index is not None and (
        isinstance(anchor_candidate_index, bool)
        or not isinstance(anchor_candidate_index, int)
        or anchor_candidate_index < 0
    ):
        raise Lfm2SemanticPostfoldError(
            "Anchor candidate index must be a nonnegative integer."
        )
    if (
        tied_interface_mode == TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE
        and not companion_requested
    ):
        raise Lfm2SemanticPostfoldError(
            "Strong-base anchor-residual mode requires a pinned frontier selection."
        )
    expected_companion_file_sha256 = (
        _digest(
            expected_interface_frontier_companion_file_sha256,
            "expected interface frontier companion file SHA-256",
        )
        if companion_requested
        else None
    )
    expected_companion_sha256 = (
        _digest(
            expected_interface_frontier_companion_sha256,
            "expected interface frontier companion SHA-256",
        )
        if companion_requested
        else None
    )
    output, validated_capsule_output = _validate_preterminal_capsule_path_disjointness(
        output_checkpoint_directory=output_checkpoint_directory,
        preterminal_capsule_directory=preterminal_capsule_directory,
        input_checkpoint_directory=input_checkpoint_directory,
        cache_directory=cache_directory,
        bridge_directory=bridge_directory,
        parent_model_directory=parent_model_directory,
        compiler_diagnostic_json=compiler_diagnostic_json,
        interface_statistics_directory=interface_statistics_directory,
        interface_frontier_companion_json=interface_frontier_companion_json,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
    )
    preterminal_capsule_directory = validated_capsule_output
    foundation_ancestry = _claims_semantic_foundation_schema(compiler_diagnostic_json)
    if statistics_requested and not foundation_ancestry:
        raise Lfm2SemanticPostfoldError(
            "Compact interface statistics are supported only for "
            "semantic-foundation ancestry."
        )
    output = _new_output_path(output)
    if preterminal_capsule_directory is not None:
        preterminal_capsule_directory = _new_output_path(preterminal_capsule_directory)
    # Record compatibility is authenticated before any child is loaded or
    # mutated.  Each trace file receives one whole-file SHA pass when the pipeline
    # selective session opens immediately before the first target mutation.
    cache = open_lfm2_circuit_trace_cache(cache_directory, verify=False)
    if cache.cache_key != _digest(expected_cache_key, "expected cache key"):
        raise Lfm2SemanticPostfoldError("Complete causal cache identity changed.")
    terminal_cache_compatibility = preflight_lfm2_long_horizon_terminal_cache(cache)
    bridge = load_strict_tokenizer_bridge_artifact(
        bridge_directory,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        expected_artifact_sha256=expected_bridge_artifact_sha256,
    )
    target_model, saved_tokenizer, input_metadata = load_lfm2_cpcf_v5_standalone_child(
        input_checkpoint_directory,
        expected_artifact_sha256=expected_input_artifact_sha256,
        device=device,
        dtype=dtype,
    )
    if fingerprint_hf_tokenizer_semantics(
        saved_tokenizer
    ) != fingerprint_hf_tokenizer_semantics(target_tokenizer):
        raise Lfm2SemanticPostfoldError(
            "Saved child tokenizer differs from the target bridge package."
        )
    parent_norm = load_lfm2_parent_final_rmsnorm_from_local_safetensors(
        cache,
        parent_model_directory,
        tensor_key=parent_final_norm_tensor_key,
        epsilon_config_key=parent_norm_epsilon_config_key,
    )
    loaded_source_table = load_lfm2_mapped_parent_tied_table_from_local_safetensors(
        parent_model_directory,
        cache=cache,
        bridge=bridge,
        authenticated_parent_final_norm=parent_norm,
        tensor_key=parent_tied_tensor_key,
        include_source_lineage_report=True,
    )
    if len(loaded_source_table) == 3:
        source_table, source_table_report, source_table_lineage_report = (
            loaded_source_table
        )
    else:  # Compatibility for injected test loaders; the real loader returns three.
        source_table, source_table_report = loaded_source_table
        source_table_lineage_report = None
    lineage = load_lfm2_semantic_compiler_lineage(
        compiler_diagnostic_json,
        expected_diagnostic_sha256=expected_compiler_diagnostic_sha256,
        cache=cache,
        bridge=bridge,
        input_metadata=input_metadata,
        **(
            {"parent_model_directory": parent_model_directory}
            if foundation_ancestry
            else {}
        ),
    )
    if foundation_ancestry:
        try:
            interface, interface_rehydration = (
                rehydrate_lfm2_semantic_foundation_interface(
                    target_model,
                    source_table,
                    semantic_cache=cache,
                    bridge=bridge,
                    target_tokenizer=target_tokenizer,
                    parent_model_directory=parent_model_directory,
                    input_metadata=input_metadata,
                    foundation_receipt_json=compiler_diagnostic_json,
                    expected_foundation_receipt_sha256=(
                        expected_compiler_diagnostic_sha256
                    ),
                    **(
                        {
                            "interface_statistics_directory": (
                                interface_statistics_directory
                            ),
                            "expected_interface_statistics_sha256": (
                                expected_statistics_sha256
                            ),
                            **(
                                {
                                    "interface_frontier_companion_json": (
                                        interface_frontier_companion_json
                                    ),
                                    "expected_interface_frontier_companion_file_sha256": (
                                        expected_companion_file_sha256
                                    ),
                                    "expected_interface_frontier_companion_sha256": (
                                        expected_companion_sha256
                                    ),
                                    "anchor_candidate_index": anchor_candidate_index,
                                }
                                if companion_requested
                                else {}
                            ),
                        }
                        if statistics_requested
                        else {}
                    ),
                )
            )
        except Lfm2SemanticFoundationError as exc:
            raise Lfm2SemanticPostfoldError(
                "Semantic-foundation interface rehydration failed."
            ) from exc
    else:
        interface, interface_rehydration = rehydrate_lfm2_c61_tied_interface(
            target_model,
            source_table,
            semantic_cache=cache,
            bridge=bridge,
            target_tokenizer=target_tokenizer,
            compiler_diagnostic_json=compiler_diagnostic_json,
            expected_compiler_diagnostic_sha256=(expected_compiler_diagnostic_sha256),
        )
    interface_statistics_replay = None
    interface_anchor_selection = None
    interface_anchor_base_target = None
    interface_anchor_residual = None
    if statistics_requested:
        verified_rehydration = _verify_sealed(
            interface_rehydration,
            seal_name="result_sha256",
            label="compact interface-statistics rehydration",
        )
        artifact_sha256 = _digest(
            verified_rehydration.get("interface_statistics_artifact_sha256"),
            "rehydrated interface statistics artifact",
        )
        manifest_sha256 = _digest(
            verified_rehydration.get("interface_statistics_manifest_file_sha256"),
            "rehydrated interface statistics manifest",
        )
        rehydration_schema = verified_rehydration.get("schema")
        if companion_requested:
            selection_value = verified_rehydration.get("interface_anchor_selection")
            if not isinstance(selection_value, Mapping):
                raise Lfm2SemanticPostfoldError(
                    "Companion rehydration omitted its selected anchor binding."
                )
            interface_anchor_selection = _authenticate_foundation_anchor_selection(
                selection_value,
                selected_interface=interface,
            )
            residual_materializer = getattr(
                interface_rehydration,
                "materialize_selected_anchor_residual",
                None,
            )
            base_materializer = getattr(
                interface_rehydration,
                "materialize_anchor_base_target",
                None,
            )
            if not callable(residual_materializer) or not callable(base_materializer):
                raise Lfm2SemanticPostfoldError(
                    "Companion rehydration discarded its anchor base or residual."
                )
            interface_anchor_base_target = base_materializer()
            interface_anchor_residual = residual_materializer()
            if (
                interface_anchor_selection["selected_anchor_candidate_index"]
                != anchor_candidate_index
                or _foundation_lineage_tensor_sha256(interface_anchor_base_target)
                != interface_anchor_selection[
                    "anchor_base_target_weight_foundation_sha256"
                ]
                or _foundation_lineage_tensor_sha256(interface_anchor_residual)
                != interface_anchor_selection[
                    "selected_residual_weight_foundation_sha256"
                ]
            ):
                raise Lfm2SemanticPostfoldError(
                    "Returned anchor candidate or residual differs from the request."
                )
            replay_sha256 = None
        else:
            replay_sha256 = _digest(
                verified_rehydration.get("interface_statistics_replay_sha256"),
                "interface statistics replay",
            )
        if (
            rehydration_schema
            != (
                "neural-forge-lfm2-semantic-foundation-interface-rehydration-v3"
                if companion_requested
                else "neural-forge-lfm2-semantic-foundation-interface-rehydration-v2"
            )
            or artifact_sha256 != expected_statistics_sha256
            or (
                companion_requested
                and (
                    verified_rehydration.get("interface_frontier_companion_file_sha256")
                    != expected_companion_file_sha256
                    or verified_rehydration.get("interface_frontier_companion_sha256")
                    != expected_companion_sha256
                    or verified_rehydration.get("interface_anchor_selection_sha256")
                    != interface_anchor_selection["selection_sha256"]
                )
            )
            or verified_rehydration.get("compiler_cache_open_calls") != 0
            or verified_rehydration.get("trace_safetensor_open_calls") != 0
            or verified_rehydration.get("selective_provider_open_calls") != 0
        ):
            raise Lfm2SemanticPostfoldError(
                "Compact interface-statistics replay did not authenticate its "
                "artifact identity or zero-open evidence boundary."
            )
        statistics_body = {
            "schema": (
                "neural-forge-lfm2-semantic-postfold-interface-statistics-replay-v1"
            ),
            "artifact_sha256": artifact_sha256,
            "manifest_file_sha256": manifest_sha256,
            **(
                {
                    "frontier_companion_file_sha256": (expected_companion_file_sha256),
                    "frontier_companion_sha256": expected_companion_sha256,
                    "interface_anchor_selection_sha256": (
                        interface_anchor_selection["selection_sha256"]
                    ),
                    "anchor_candidate_index": anchor_candidate_index,
                    "migration_companion_consumed": True,
                }
                if companion_requested
                else {"replay_sha256": replay_sha256}
            ),
            "interface_rehydration_sha256": verified_rehydration["result_sha256"],
            "semantic_evidence_cache_key": cache.cache_key,
            "semantic_evidence_cache_open_calls": 1,
            "compiler_cache_open_calls": 0,
            "trace_safetensor_open_calls": 0,
            "selective_provider_open_calls": 0,
        }
        interface_statistics_replay = MappingProxyType(
            {
                **statistics_body,
                "binding_sha256": canonical_sha256(statistics_body),
            }
        )
    rehydrated_interface_report = dict(interface.report)
    selected_anchor_source_interface = (
        interface if interface_anchor_selection is not None else None
    )
    preterminal_capsule_reference = None
    # The single whole-file SHA authentication pass happens before the first
    # target mutation (the tied residual gauge).  This exact session remains open
    # while the layer stack reads selected tensors and the terminal collector
    # later reads every tensor payload in each sealed batch inventory.
    with cache.selective_authenticated_session() as pipeline_cache_session:
        anchor_selection_installation = None
        if interface_anchor_selection is not None:
            anchor_selection_installation = (
                install_lfm2_semantic_foundation_anchor_selection(
                    target_model,
                    interface,
                    interface_anchor_selection,
                )
            )
        effective_tied_interface_mode = (
            TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE
            if interface_anchor_selection is not None
            and tied_interface_mode == TIED_RESIDUAL_STRONG_CONGRUENCE_MODE
            else tied_interface_mode
        )
        anchor_transport_witness = (
            TiedResidualAnchorTransportWitness.from_source_interface(
                selected_anchor_source_interface,
                anchor_base_target_weight=interface_anchor_base_target,
                anchor_source_residual_weight=interface_anchor_residual,
            )
            if effective_tied_interface_mode
            == TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE
            else None
        )
        interface, tied_residual_interface_gauge = (
            install_lfm2_tied_residual_congruence_gauge(
                target_model,
                interface,
                source_table,
                mode=effective_tied_interface_mode,
                **(
                    {
                        "anchor_base_target_weight": interface_anchor_base_target,
                        "anchor_source_residual_weight": interface_anchor_residual,
                    }
                    if effective_tied_interface_mode
                    == TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE
                    else {}
                ),
            )
        )
        terminal_source_lineage = build_lfm2_long_horizon_terminal_source_lineage(
            cache=cache,
            bridge=bridge,
            parent_final_norm=parent_norm,
            interface=interface,
            source_selected_tied_table=source_table,
            direct_local_mapped_source_report=source_table_lineage_report,
            anchor_transport_witness=anchor_transport_witness,
        )
        layer_stack = compile_install_lfm2_semantic_layer_stack_stage(
            cache=cache,
            bridge=bridge,
            target_model=target_model,
            interface=interface,
            compiler_lineage=lineage,
            parent_model_directory=parent_model_directory,
            donor_tokenizer=donor_tokenizer,
            target_tokenizer=target_tokenizer,
            attention_installer=attention_installer,
            semantic_layer_stack_compiler=semantic_layer_stack_compiler,
            fold_compute_device=fold_compute_device,
            global_residual_closure_version=closure_version,
            correction_scale_frontier=scale_policy.materialization_frontier,
            correction_scale=scale_policy.materialization_scale,
            global_residual_backend=closure_policy.backend,
            global_residual_basis_rank=global_residual_basis_rank,
            authenticated_cache_session=pipeline_cache_session,
            body_recipe=recipe,
            interface_anchor_selection=interface_anchor_selection,
            **(
                {
                    "mapped_parent_tied_table": source_table,
                    "anchor_transport_witness": anchor_transport_witness,
                }
                if effective_tied_interface_mode
                == TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE
                else {}
            ),
        )
        if interface_anchor_selection is not None and (
            layer_stack.receipt.get("interface_anchor_selection_sha256")
            != interface_anchor_selection["selection_sha256"]
            or layer_stack.receipt.get("interface_anchor_selection")
            != dict(interface_anchor_selection)
        ):
            raise Lfm2SemanticPostfoldError(
                "Semantic layer stack dropped or changed the selected anchor binding."
            )
        attention_receipt_authentication = (
            authenticate_lfm2_semantic_layer_stack_attention_receipts(
                layer_stack.receipt,
                body_recipe=recipe,
            )
        )
        capsule_publisher = None
        if preterminal_capsule_directory is not None:
            from neural_forge.hf_lfm2_semantic_preterminal_capsule import (
                Lfm2SemanticPreterminalCapsuleBindings,
                publish_lfm2_semantic_preterminal_capsule,
            )

            capsule_bindings = Lfm2SemanticPreterminalCapsuleBindings(
                input_artifact_sha256=input_metadata["artifact_sha256"],
                compiler_diagnostic_sha256=_digest(
                    expected_compiler_diagnostic_sha256,
                    "expected compiler diagnostic SHA-256",
                ),
                compiler_cache_key=interface_rehydration["compiler_cache_key"],
                semantic_cache_key=cache.cache_key,
                semantic_cache_manifest_sha256=cache.manifest["manifest_sha256"],
                cache_parent_fingerprint=cache.parent_fingerprint,
                bridge_artifact_sha256=bridge.artifact_sha256,
                mapped_parent_tied_table_receipt_sha256=(
                    source_table_report["result_sha256"]
                ),
                parent_final_norm_receipt_sha256=parent_norm.result_sha256,
                interface_rehydration_sha256=interface_rehydration["result_sha256"],
                tied_residual_interface_gauge_sha256=(
                    tied_residual_interface_gauge["result_sha256"]
                ),
                cache_selective_session_sha256=layer_stack.receipt[
                    "cache_selective_session_sha256"
                ],
                interface_statistics_artifact_sha256=expected_statistics_sha256,
            )

            def capsule_publisher() -> Any:
                return publish_lfm2_semantic_preterminal_capsule(
                    target_model=target_model,
                    target_tokenizer=target_tokenizer,
                    layer_stack=layer_stack,
                    interface=interface,
                    mapped_parent_tied_table=source_table,
                    anchor_transport_witness=anchor_transport_witness,
                    attention_receipt_authentication=(attention_receipt_authentication),
                    bindings=capsule_bindings,
                    output_directory=preterminal_capsule_directory,
                )

        preterminal_capsule_reference, terminal = (
            publish_preterminal_capsule_then_compile_fused_terminal(
                capsule_publisher=capsule_publisher,
                fused_terminal_compiler=fused_terminal_compiler,
                terminal_frontier_exporter_factory=(terminal_frontier_exporter_factory),
                terminal_arguments={
                    "cache": cache,
                    "bridge": bridge,
                    "target_model": target_model,
                    "layer_stack": layer_stack,
                    "interface": interface,
                    "source_selected_tied_table": source_table,
                    "source_lineage": terminal_source_lineage,
                    "parent_final_norm": parent_norm,
                    "split_plan": split_plan,
                    "donor_tokenizer": donor_tokenizer,
                    "target_tokenizer": target_tokenizer,
                    "terminal_cache_compatibility": terminal_cache_compatibility,
                    "authenticated_cache_session": pipeline_cache_session,
                    "body_recipe": recipe,
                    **(
                        {"terminal_frontier_exporter": terminal_frontier_exporter}
                        if terminal_frontier_exporter is not None
                        else {}
                    ),
                    **(
                        {
                            "anchor_transport_witness": anchor_transport_witness,
                        }
                        if effective_tied_interface_mode
                        == TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE
                        else {}
                    ),
                },
            )
        )
        state_after_terminal = fingerprint_lfm2_parent(target_model).identity_sha256
        terminal_receipt = dict(
            _validate_fused_semantic_terminal_receipt(
                terminal,
                layer_receipt=layer_stack.receipt,
                current_model_state_sha256=state_after_terminal,
            )
        )
        terminal_review_pending = bool(
            terminal_receipt.get("schema")
            == LFM2_SEMANTIC_TERMINAL_REVIEW_PENDING_SCHEMA
        )
        pipeline_cache_session.verify()
        pipeline_cache_session_receipt = pipeline_cache_session.as_dict()
        terminal_payload_binding = terminal_receipt.get(
            "terminal_cache_payload_binding"
        )
        if (
            layer_stack.receipt.get("cache_selective_session_sha256")
            != pipeline_cache_session_receipt["session_sha256"]
            or not isinstance(terminal_payload_binding, Mapping)
            or terminal_payload_binding.get("payload_session_sha256")
            != pipeline_cache_session_receipt["session_sha256"]
        ):
            raise Lfm2SemanticPostfoldError(
                "Layer stack and fused terminal did not reuse the same "
                "authenticated pipeline cache session."
            )
        terminal_tensor_payload_audit = terminal_receipt.get(
            "terminal_cache_tensor_payload_load_audit"
        )
        if not isinstance(terminal_tensor_payload_audit, Mapping):
            terminal_tensor_payload_audit = {}
        pipeline_cache_session_reuse = _sealed(
            {
                "pipeline_cache_session_sha256": (
                    pipeline_cache_session_receipt["session_sha256"]
                ),
                "layer_stack_cache_session_sha256": layer_stack.receipt[
                    "cache_selective_session_sha256"
                ],
                "terminal_payload_session_sha256": terminal_payload_binding[
                    "payload_session_sha256"
                ],
                "same_pipeline_session_reused_by_layer_stack_and_terminal": True,
                "cache_open_verify_payload_scan_performed": False,
                "metadata_preflight_preceded_pipeline_session": True,
                "pipeline_session_opened_before_first_target_mutation": True,
                "trace_file_sha256_passes_per_batch": 1,
                "additional_full_trace_file_sha256_passes_in_pipeline": 0,
                "terminal_full_tensor_inventory_payload_loads_total": (
                    terminal_tensor_payload_audit.get(
                        "full_tensor_inventory_payload_load_count",
                        0,
                    )
                ),
                "terminal_full_tensor_inventory_payload_logical_bytes_total": (
                    terminal_tensor_payload_audit.get(
                        "full_tensor_inventory_payload_logical_bytes_total",
                        0,
                    )
                ),
            },
            "result_sha256",
        )
    global_residual_controls = {
        "global_residual_closure_version": closure_version,
        "global_residual_backend": closure_policy.backend,
        "global_residual_closure_policy_sha256": (closure_policy.authentication_sha256),
        "global_residual_correction_scale_frontier": list(
            scale_policy.materialization_frontier
        ),
        "global_residual_correction_scale": scale_policy.materialization_scale,
        "global_residual_correction_scale_policy_sha256": (
            scale_policy.authentication_sha256
        ),
        "global_residual_frontier_member_selected_as_winner": False,
        "global_residual_numeric_or_manual_quality_gate": False,
        **(
            {
                "global_residual_materialization_compute_device": (
                    layer_stack.receipt[
                        "global_residual_materialization_compute_device"
                    ]
                ),
                "global_residual_materialization_compute_execution_receipt_sha256": (
                    layer_stack.receipt[
                        "global_residual_materialization_compute_execution_receipt_sha256"
                    ]
                ),
            }
            if isinstance(
                layer_stack.receipt.get(
                    "global_residual_materialization_compute_execution_receipt_sha256"
                ),
                str,
            )
            else {}
        ),
    }
    context_body = {
        "schema": "neural-forge-lfm2-semantic-postfold-context-v1",
        "input_artifact_sha256": input_metadata["artifact_sha256"],
        "compiler_ancestry_schema": (
            LFM2_SEMANTIC_FOUNDATION_SCHEMA
            if foundation_ancestry
            else "neural-forge-lfm2-cpcf-v5-selective-blank-fold-run-v1"
        ),
        "compiler_cache_key": interface_rehydration["compiler_cache_key"],
        "semantic_evidence_cache_key": cache.cache_key,
        "terminal_cache_compatibility_sha256": terminal_cache_compatibility[
            "authentication_sha256"
        ],
        "pipeline_cache_session_reuse_sha256": pipeline_cache_session_reuse[
            "result_sha256"
        ],
        "body_recipe": recipe.as_dict(),
        "body_recipe_sha256": recipe.recipe_sha256,
        "body_recipe_id": recipe.recipe_id,
        "complete_body_recipe_candidate_ids": list(LFM2_SEMANTIC_BODY_RECIPE_IDS),
        "materialized_body_recipe_id": recipe.recipe_id,
        "fold_compute_device": fold_compute_device,
        **global_residual_controls,
        "common_parent_fingerprint": cache.parent_fingerprint,
        "compiler_and_semantic_cache_identities_bound_separately": True,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "mapped_parent_tied_table_sha256": source_table_report["result_sha256"],
        "rehydrated_interface_report_sha256": rehydrated_interface_report[
            "report_sha256"
        ],
        "active_tied_interface_report_sha256": interface.report["report_sha256"],
        "interface_rehydration_sha256": interface_rehydration["result_sha256"],
        **(
            {
                "interface_anchor_selection": dict(interface_anchor_selection),
                "interface_anchor_selection_sha256": interface_anchor_selection[
                    "selection_sha256"
                ],
                "interface_anchor_selection_installation_sha256": (
                    anchor_selection_installation["result_sha256"]
                ),
            }
            if interface_anchor_selection is not None
            else {}
        ),
        "requested_tied_interface_mode": tied_interface_mode,
        "tied_interface_mode": effective_tied_interface_mode,
        "tied_residual_interface_gauge_sha256": (
            tied_residual_interface_gauge["result_sha256"]
        ),
        "self_dual_tied_gauge_sha256": tied_residual_interface_gauge["result_sha256"],
        "compiler_diagnostic_sha256": _digest(
            expected_compiler_diagnostic_sha256,
            "expected compiler diagnostic SHA-256",
        ),
        "semantic_layer_stack_stage_sha256": layer_stack.receipt["result_sha256"],
        "attention_receipt_authentication_sha256": (
            attention_receipt_authentication["result_sha256"]
        ),
        **(
            {"preterminal_capsule": (preterminal_capsule_reference.as_dict())}
            if preterminal_capsule_reference is not None
            else {}
        ),
        "fused_semantic_terminal_refold_sha256": terminal_receipt["result_sha256"],
        "complete_nonzero_structural_producer_coverage": layer_stack.receipt[
            "complete_nonzero_structural_producer_coverage"
        ],
        "complete_nonzero_final_mlp_structural_coverage": terminal_receipt.get(
            "complete_nonzero_final_mlp_structural_coverage",
            False,
        ),
        "terminal_review_pending": terminal_review_pending,
        "terminal_writer_install_count": terminal_receipt.get(
            "terminal_writer_install_count",
            0,
        ),
        **(
            {"interface_statistics_replay": dict(interface_statistics_replay)}
            if interface_statistics_replay is not None
            else {}
        ),
    }
    context_sha = canonical_sha256(context_body)
    if terminal_review_pending:
        if (
            state_after_terminal != layer_stack.receipt.get("model_state_after_sha256")
            or terminal_receipt.get("terminal_writer_install_count") != 0
            or terminal_receipt.get("terminal_writer_mutation_count") != 0
            or terminal_receipt.get("candidate_published_before_manual_review")
            is not False
            or terminal_receipt.get("terminal_frontier_export") is None
            or output.exists()
        ):
            raise Lfm2SemanticPostfoldError(
                "Review-pending top-level orchestration changed or published the "
                "child before whole-response review."
            )
        pending_body = {
            "schema": LFM2_SEMANTIC_POSTFOLD_SCHEMA,
            "status": "terminal-frontier-exported-review-pending",
            "terminal_review_pending_schema": (
                LFM2_CPCF_TERMINAL_ACTION_REVIEW_PENDING_SCHEMA
            ),
            "final_selection_status": "deferred-to-manual-whole-response-review",
            "final_selected_candidate_id": None,
            "final_selection_performed": False,
            "automatic_candidate_selection_performed": False,
            "automatic_candidate_acceptance_performed": False,
            "candidate_published_before_manual_review": False,
            "manual_whole_response_review_required": True,
            "diagnostic_order_has_final_selection_authority": False,
            "checkpoint_published": False,
            "requested_output_checkpoint_directory": str(output),
            "input_child": {
                "checkpoint_directory": str(
                    _ordinary_directory(
                        input_checkpoint_directory,
                        "input checkpoint",
                    )
                ),
                "artifact_sha256": input_metadata["artifact_sha256"],
                "model_state_sha256": input_metadata["model_state_sha256"],
            },
            "compiler_ancestry_schema": context_body["compiler_ancestry_schema"],
            "compiler_cache_key": interface_rehydration["compiler_cache_key"],
            "semantic_evidence_cache_key": cache.cache_key,
            "body_recipe": recipe.as_dict(),
            "body_recipe_sha256": recipe.recipe_sha256,
            "body_recipe_id": recipe.recipe_id,
            "complete_body_recipe_candidate_ids": list(LFM2_SEMANTIC_BODY_RECIPE_IDS),
            "complete_body_recipe_candidate_count": len(LFM2_SEMANTIC_BODY_RECIPE_IDS),
            "materialized_body_recipe_id": recipe.recipe_id,
            **global_residual_controls,
            "one_checkpoint_contains_one_materialized_body_recipe": False,
            "one_frontier_bank_contains_one_body_recipe": True,
            "other_body_recipes_require_sealed_foundation_replay": True,
            "terminal_cache_compatibility": dict(terminal_cache_compatibility),
            "pipeline_cache_session_reuse": dict(pipeline_cache_session_reuse),
            "cache_parent_fingerprint": cache.parent_fingerprint,
            "bridge_artifact_sha256": bridge.artifact_sha256,
            "mapped_parent_tied_table": dict(source_table_report),
            "rehydrated_interface": rehydrated_interface_report,
            "active_tied_interface": dict(interface.report),
            "interface_rehydration": dict(interface_rehydration),
            **(
                {"interface_statistics_replay": dict(interface_statistics_replay)}
                if interface_statistics_replay is not None
                else {}
            ),
            "requested_tied_interface_mode": tied_interface_mode,
            "tied_interface_mode": effective_tied_interface_mode,
            "tied_residual_interface_gauge": dict(tied_residual_interface_gauge),
            "self_dual_tied_gauge": dict(tied_residual_interface_gauge),
            "compiler_lineage": {
                str(index): dict(value.report) for index, value in lineage.items()
            },
            "semantic_layer_stack": dict(layer_stack.receipt),
            "attention_receipt_authentication": dict(attention_receipt_authentication),
            **(
                {"preterminal_capsule": preterminal_capsule_reference.as_dict()}
                if preterminal_capsule_reference is not None
                else {}
            ),
            "fused_semantic_terminal": terminal_receipt,
            "terminal_frontier_export": terminal_receipt["terminal_frontier_export"],
            "terminal_publication_decision_accepted": False,
            "terminal_nonincumbent_candidate_accepted": False,
            "terminal_incumbent_retained": True,
            "terminal_writer_install_count": 0,
            "terminal_writer_mutation_count": 0,
            "current_child_model_state_sha256": state_after_terminal,
            "context_binding": {**context_body, "context_sha256": context_sha},
            "training_replaced_by": (
                "sealed-cache semantic lattice transport plus closed-form linear solves"
            ),
            "operation_audit": dict(_ZERO_OPERATIONS),
            "claims": {
                "complete_causal_layer_stack_adjudicated": True,
                "complete_nonzero_structural_producer_coverage": (
                    layer_stack.receipt["complete_nonzero_structural_producer_coverage"]
                ),
                "complete_nonzero_final_mlp_structural_coverage": False,
                "attention_operator_decision_count": layer_stack.receipt.get(
                    "attention_operator_decision_count",
                    0,
                ),
                "attention_correction_install_count": layer_stack.receipt.get(
                    "attention_operator_install_count",
                    0,
                ),
                "short_conv_operator_decision_count": layer_stack.receipt.get(
                    "short_conv_operator_decision_count",
                    0,
                ),
                "short_conv_correction_install_count": layer_stack.receipt.get(
                    "short_conv_operator_install_count",
                    0,
                ),
                "terminal_writer_install_count": 0,
                "terminal_writer_mutation_count": 0,
                "checkpoint_published": False,
                "training_performed": False,
                "optimizer_steps": 0,
                "backward_calls": 0,
                "parameters_pruned": 0,
                "parameters_quantized": 0,
                "parameters_sparsified": 0,
                "numeric_metrics_have_final_intelligence_authority": False,
                "manual_whole_response_review_required": True,
            },
        }
        return MappingProxyType(_sealed(pending_body, "result_sha256"))
    reloaded, checkpoint = publish_lfm2_semantic_postfold_checkpoint(
        target_model=target_model,
        target_tokenizer=target_tokenizer,
        output_directory=output,
        context_binding_sha256=context_sha,
        interface_anchor_selection_sha256=(
            None
            if interface_anchor_selection is None
            else interface_anchor_selection["selection_sha256"]
        ),
        device=device,
        dtype=dtype,
    )
    state_after_reload = fingerprint_lfm2_parent(reloaded).identity_sha256
    body = {
        "schema": LFM2_SEMANTIC_POSTFOLD_SCHEMA,
        "status": "standalone-semantic-postfold-saved-and-reloaded",
        "input_child": {
            "checkpoint_directory": str(
                _ordinary_directory(input_checkpoint_directory, "input checkpoint")
            ),
            "artifact_sha256": input_metadata["artifact_sha256"],
            "model_state_sha256": input_metadata["model_state_sha256"],
        },
        "compiler_ancestry_schema": context_body["compiler_ancestry_schema"],
        "compiler_cache_key": interface_rehydration["compiler_cache_key"],
        "semantic_evidence_cache_key": cache.cache_key,
        "body_recipe": recipe.as_dict(),
        "body_recipe_sha256": recipe.recipe_sha256,
        "body_recipe_id": recipe.recipe_id,
        "complete_body_recipe_candidate_ids": list(LFM2_SEMANTIC_BODY_RECIPE_IDS),
        "complete_body_recipe_candidate_count": len(LFM2_SEMANTIC_BODY_RECIPE_IDS),
        "materialized_body_recipe_id": recipe.recipe_id,
        **global_residual_controls,
        "one_checkpoint_contains_one_materialized_body_recipe": True,
        "other_body_recipes_require_sealed_foundation_replay": True,
        "terminal_cache_compatibility": dict(terminal_cache_compatibility),
        "pipeline_cache_session_reuse": dict(pipeline_cache_session_reuse),
        "cache_parent_fingerprint": cache.parent_fingerprint,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "mapped_parent_tied_table": dict(source_table_report),
        "rehydrated_interface": rehydrated_interface_report,
        "active_tied_interface": dict(interface.report),
        "interface_rehydration": dict(interface_rehydration),
        **(
            {
                "interface_anchor_selection": dict(interface_anchor_selection),
                "interface_anchor_selection_installation": dict(
                    anchor_selection_installation
                ),
            }
            if interface_anchor_selection is not None
            else {}
        ),
        **(
            {"interface_statistics_replay": dict(interface_statistics_replay)}
            if interface_statistics_replay is not None
            else {}
        ),
        "requested_tied_interface_mode": tied_interface_mode,
        "tied_interface_mode": effective_tied_interface_mode,
        "tied_residual_interface_gauge": dict(tied_residual_interface_gauge),
        "self_dual_tied_gauge": dict(tied_residual_interface_gauge),
        "compiler_lineage": {
            str(index): dict(value.report) for index, value in lineage.items()
        },
        "semantic_layer_stack": dict(layer_stack.receipt),
        "attention_receipt_authentication": dict(attention_receipt_authentication),
        **(
            {"preterminal_capsule": (preterminal_capsule_reference.as_dict())}
            if preterminal_capsule_reference is not None
            else {}
        ),
        "fused_semantic_terminal": terminal_receipt,
        "terminal_publication_decision_accepted": True,
        "terminal_nonincumbent_candidate_accepted": terminal_receipt[
            "terminal_nonincumbent_candidate_accepted"
        ],
        "terminal_incumbent_retained": terminal_receipt["terminal_incumbent_retained"],
        "checkpoint": dict(checkpoint),
        "fresh_reload_parent_free_state_sha256": state_after_reload,
        "context_binding": {**context_body, "context_sha256": context_sha},
        "training_replaced_by": (
            "sealed-cache semantic lattice transport plus closed-form linear solves"
        ),
        "operation_audit": dict(_ZERO_OPERATIONS),
        "claims": {
            "complete_causal_layer_stack_adjudicated": True,
            "complete_nonzero_structural_producer_coverage": (
                layer_stack.receipt["complete_nonzero_structural_producer_coverage"]
            ),
            "blank_or_partially_blank_producer_chain_count": (
                layer_stack.receipt["blank_or_partially_blank_producer_chain_count"]
            ),
            "complete_nonzero_final_mlp_structural_coverage": (
                terminal_receipt["complete_nonzero_final_mlp_structural_coverage"]
            ),
            "complete_causal_layer_stack_installed": layer_stack.receipt[
                "complete_causal_layer_stack_installed"
            ],
            "attention_operator_decision_count": layer_stack.receipt[
                "attention_operator_decision_count"
            ],
            "attention_correction_install_count": layer_stack.receipt[
                "attention_operator_install_count"
            ],
            "attention_correction_installed": (
                layer_stack.receipt["attention_operator_install_count"] > 0
            ),
            "attention_full_receipt_persistence_authenticated": True,
            "attention_full_receipt_count": attention_receipt_authentication[
                "attention_receipt_count"
            ],
            "attention_complete_six_tensor_bundle_count": (
                attention_receipt_authentication["complete_six_tensor_bundle_count"]
            ),
            "short_conv_operator_decision_count": layer_stack.receipt[
                "short_conv_operator_decision_count"
            ],
            "short_conv_correction_install_count": layer_stack.receipt[
                "short_conv_operator_install_count"
            ],
            "short_conv_correction_installed": (
                layer_stack.receipt["short_conv_operator_install_count"] > 0
            ),
            "replay_fresh_mlp_evidence_compile_count": layer_stack.receipt[
                "replay_fresh_mlp_evidence_compile_count"
            ],
            "nonterminal_mlp_writer_decision_count": layer_stack.receipt[
                "nonterminal_mlp_writer_decision_count"
            ],
            "replay_fresh_mlp_correction_install_count": layer_stack.receipt[
                "nonterminal_mlp_writer_install_count"
            ],
            "replay_fresh_mlp_corrections_installed": (
                layer_stack.receipt["nonterminal_mlp_writer_install_count"] > 0
            ),
            "final_mlp_producer_install_count": layer_stack.receipt[
                "final_mlp_producer_install_count"
            ],
            "terminal_writer_decision_count": terminal_receipt[
                "terminal_writer_decision_count"
            ],
            "terminal_writer_install_count": terminal_receipt[
                "terminal_writer_install_count"
            ],
            "fused_parent_endpoint_and_lexical_w2_installed_once": (
                terminal_receipt["terminal_writer_install_count"] == 1
            ),
            "terminal_w2_heldout_diagnosed_without_publication_gate": True,
            "body_recipe_authenticated_across_attention_short_conv_and_mlp": True,
            "numeric_metrics_have_final_intelligence_authority": False,
            "all_body_recipe_ids_require_sealed_foundation_replay_for_manual_review": (
                True
            ),
            "overall_postfold_contains_meaningful_parameter_changes": True,
            "standalone_child": True,
            "parameter_count_compaction_changed_by_postfold": False,
            "training_performed": False,
            "pruning_performed": False,
            "quantization_performed": False,
        },
    }
    return MappingProxyType(_sealed(body, "result_sha256"))


__all__ = [
    "authenticate_lfm2_semantic_layer_stack_attention_receipts",
    "LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_LAYER_INSTALLATION_SCHEMA",
    "LFM2_SEMANTIC_ATTENTION_INSTALLATION_SCHEMA",
    "LFM2_SEMANTIC_ATTENTION_STAGE_SCHEMA",
    "LFM2_SEMANTIC_COMPILER_LINEAGE_SCHEMA",
    "LFM2_SEMANTIC_LEXICAL_COMPLETION_SCHEMA",
    "LFM2_SEMANTIC_POSTFOLD_CHECKPOINT_SCHEMA",
    "LFM2_SEMANTIC_POSTFOLD_FAILURE_SCHEMA",
    "LFM2_SEMANTIC_POSTFOLD_SCHEMA",
    "LFM2_SEMANTIC_RESIDUAL_CHART_SCHEMA",
    "LFM2_SEMANTIC_TERMINAL_REVIEW_PENDING_SCHEMA",
    "Lfm2SemanticAttentionTransportEvidence",
    "Lfm2SemanticCompilerLayerLineage",
    "Lfm2SemanticLayerStackInstallation",
    "Lfm2SemanticPostfoldBlockedError",
    "Lfm2SemanticPostfoldError",
    "SEMANTIC_ATTENTION_INSTALLER_CALLABLE",
    "SEMANTIC_LAYER_STACK_INSTALLER_CALLABLE",
    "TIED_RESIDUAL_CONGRUENCE_MODES",
    "TIED_RESIDUAL_STRONG_CONGRUENCE_MODE",
    "compile_install_lfm2_semantic_attention_stage",
    "compile_install_lfm2_fused_semantic_terminal_refold",
    "compile_install_lfm2_semantic_layer_stack_stage",
    "install_lfm2_self_dual_tied_interface_gauge",
    "install_lfm2_tied_residual_congruence_gauge",
    "load_lfm2_cpcf_v5_standalone_child",
    "load_lfm2_mapped_parent_tied_table_from_local_safetensors",
    "load_lfm2_parent_qkvo_inventory_from_local_safetensors",
    "load_lfm2_semantic_compiler_lineage",
    "publish_lfm2_semantic_postfold_checkpoint",
    "publish_preterminal_capsule_then_compile_fused_terminal",
    "rehydrate_lfm2_c61_tied_interface",
    "reconstruct_lfm2_saved_tied_interface",
    "run_lfm2_semantic_postfold",
    "seal_lfm2_semantic_attention_installation_receipt",
]
