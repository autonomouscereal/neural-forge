#!/usr/bin/env python3
"""Fold one fresh exact-4x untied LFM2 blank into decoded behavior.

This is the first end-to-end transaction which combines the strongest
independently proven no-training lanes in one child coordinate:

* a catalog-selected dense H776/I700 untied blank under the exact quarter
  parameter budget;
* contextual token composition before the H1024 -> H776 interface solve;
* coupled residual-attention, short-convolution, replay-fresh MLP, and global
  residual closure folding;
* the authenticated deferred final-W2 prior; and
* a causal token head solved from the final child's own residual coordinates.

The parent is never executed.  Its reusable cache is read once.  There is no
training, autograd, optimizer, pruning, quantization, sparsity, parent sidecar,
or runtime weight sharing.  Diagnostic numbers are retained for navigation;
decoded whole responses are the behavioral evidence and no numeric or binary
score can reject, delete, or promote the child.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from experiments.review_lfm2_semantic_frontier import (  # noqa: E402
    _load_parent_envelope,
    load_review_prompts,
    review_frontier,
)
from experiments.run_lfm2_residual_attention_fold import (  # noqa: E402
    load_local_tokenizer,
)
from neural_forge.analytic_execution_profile import (  # noqa: E402
    resolve_analytic_execution_profile,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_fold_executor import (  # noqa: E402
    fingerprint_model_runtime_state,
    fingerprint_model_state,
)
from neural_forge.dynamic_lfm2_adapter import (  # noqa: E402
    Lfm2DenseBudgetedSelectedTarget,
    Lfm2TargetSynthesisAdapter,
    validate_lfm2_dense_budgeted_child,
)
from neural_forge.dynamic_lfm2_targets import Lfm2Topology  # noqa: E402
from neural_forge.dynamic_model_inspector import (  # noqa: E402
    default_inspector_registry,
    inspect_model_circuits,
)
from neural_forge.dynamic_parameter_frontier import (  # noqa: E402
    DenseParameterBudgetRequest,
    DenseTargetTopologyCatalogRegistry,
    Lfm2DenseTargetTopologyCatalog,
    select_lfm2_untied_rebalance_candidate,
    synthesize_dense_parameter_frontier,
)
from neural_forge.dynamic_target_synthesis import (  # noqa: E402
    DynamicTargetAdapterRegistry,
    DynamicTargetSynthesisResult,
    DynamicTargetSynthesizer,
    plan_budgeted_dynamic_target,
    serialized_inference_state_inventory,
    verify_compiler_authenticated_target,
)
from neural_forge.dynamic_weight_complete_basis import (  # noqa: E402
    compile_role_balanced_operator_geometry,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    fingerprint_lfm2_parent,
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import (  # noqa: E402
    install_lfm2_cpcf_neutral_norms,
)
from neural_forge.hf_lfm2_cpcf_selective_evidence import (  # noqa: E402
    migrate_lfm2_cpcf_selective_evidence_provider_from_legacy_catalog,
)
from neural_forge.hf_lfm2_direct_local import (  # noqa: E402
    Lfm2DirectLocalCheckpointIO,
    load_lfm2_direct_local_model,
)
from neural_forge.hf_lfm2_exact4x_composite_interface import (  # noqa: E402
    LFM2_INDEPENDENT_STRUCTURAL_INTERFACE_INSTALL_SCHEMA,
    install_lfm2_independent_causal_output_head,
    install_lfm2_independent_structural_interface,
)
from neural_forge.hf_lfm2_interface_frontier import (  # noqa: E402
    DEFAULT_INTERFACE_CONTEXTS,
    LFM2_UNSUPPORTED_LOGIT_POLICY_ADAPTIVE_STRONGEST_SURVIVING_BACKGROUND_SEAL,
    LFM2_UNSUPPORTED_LOGIT_POLICY_MEASURE_DRIFT,
    collect_lfm2_child_causal_residual_evidence,
    collect_lfm2_terminal_cache_evidence,
    compile_lfm2_causal_residual_token_bridge,
    compile_lfm2_causal_trie_evidence,
    compile_lfm2_contextual_input_table,
    compile_lfm2_interface_sequence_plan,
    fingerprint_lfm2_causal_body_inventory,
)
from neural_forge.lfm2_exact4x_contract import (  # noqa: E402
    CAUSAL_HEAD_RECIPE,
    EXPECTED_CHILD_TOPOLOGY,
    EXPECTED_CHILD_UNIQUE_PARAMETERS,
    EXPECTED_PARENT_UNIQUE_PARAMETERS,
    PARENT_MODEL_ROLE,
)
from neural_forge.lfm2_native_exact4x_target import (  # noqa: E402
    LFM2_NATIVE_EXACT4X_LAYER_TYPES,
    LFM2_NATIVE_EXACT4X_TARGET_CLASS_ID,
    derive_lfm2_native_exact4x_target_contract,
)
from neural_forge.lfm2_circuit_balanced_coordinate_recipe import (  # noqa: E402
    LFM2_CIRCUIT_BALANCED_COORDINATE_EVIDENCE_SCHEMA,
    LFM2_CIRCUIT_BALANCED_COORDINATE_RECIPE,
    build_lfm2_circuit_balanced_coordinate_evidence,
)
from neural_forge.lfm2_body_global_coordinate_completion import (  # noqa: E402
    LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_RECIPES,
)
from neural_forge.hf_lfm2_normalized_input_replay_capsule import (  # noqa: E402
    prepare_lfm2_normalized_input_replay_capture_root,
)
from neural_forge.hf_lfm2_postfold_reclosure_seed_archive import (  # noqa: E402
    write_lfm2_postfold_reclosure_seed_archive,
)
from neural_forge.hf_lfm2_pre_causal_body_resume_capsule import (  # noqa: E402
    Lfm2PreCausalBodyResumeBindings,
    publish_lfm2_pre_causal_body_resume_capsule,
)
from neural_forge.hf_lfm2_semantic_body_recipe import (  # noqa: E402
    resolve_lfm2_semantic_body_recipe,
)
from neural_forge.hf_lfm2_semantic_foundation import (  # noqa: E402
    build_lfm2_semantic_foundation_attention_lineage,
    fit_lfm2_semantic_foundation_interface_from_provider,
    inventory_lfm2_semantic_foundation_parent,
    read_lfm2_semantic_foundation_catalog_preparation_metadata,
)
from neural_forge.hf_lfm2_semantic_postfold import (  # noqa: E402
    LFM2_GLOBAL_RESIDUAL_CLOSURE_AUTO,
    LFM2_GLOBAL_RESIDUAL_CLOSURE_REQUESTS,
    LFM2_LIVE_CHILD_PREFIX_RECAPTURE_DISABLED,
    LFM2_LIVE_CHILD_PREFIX_RECAPTURE_MODES,
    compile_install_lfm2_semantic_layer_stack_stage,
    install_lfm2_deferred_final_w2_prior,
    normalize_lfm2_semantic_foundation_attention_lineage,
)
from neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter import (  # noqa: E402
    finalize_lfm2_moving_gauge_after_deferred_w2,
)
from neural_forge.hf_lfm2_semantic_short_conv_refold_adapter import (  # noqa: E402
    LFM2_SEMANTIC_SHORT_CONV_COORDINATE_COMPLETE_FHVR_FULL_COVERAGE_CONSTRUCTION,
    LFM2_SEMANTIC_SHORT_CONV_CONSTRUCTION_IDS,
    resolve_lfm2_semantic_short_conv_construction,
)
from neural_forge.hf_lfm2_structural_attention_candidate_archive import (  # noqa: E402
    write_lfm2_structural_attention_candidate_archive,
)
from neural_forge.hf_lfm2_weight_complete_basis import (  # noqa: E402
    inventory_lfm2_weight_complete_operators,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    fingerprint_hf_tokenizer_semantics,
    load_strict_tokenizer_bridge_artifact,
    strict_tokenizer_bridge_target_package_fingerprint,
)


RUN_SCHEMA = "neural-forge-lfm2-exact4x-composite-fold-run-v1"
FRONTIER_SCHEMA = "neural-forge-lfm2-semantic-terminal-frontier-run-v1"
FAILURE_SCHEMA = "neural-forge-lfm2-exact4x-composite-fold-failure-v1"
FAILURE_EXCEPTION_CHAIN_SCHEMA = (
    "neural-forge-lfm2-exact4x-composite-fold-exception-chain-v1"
)
FAILURE_EXCEPTION_CHAIN_MAXIMUM_DEPTH = 16
FAILURE_EXCEPTION_MESSAGE_MAXIMUM_CHARACTERS = 4_096
FAILURE_EXCEPTION_TRACEBACK_MAXIMUM_FRAMES = 32
STRUCTURAL_BODY_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-exact4x-structural-body-checkpoint-v1"
)
BODY_FOLD_REALIZATION_SCHEMA = (
    "neural-forge-lfm2-exact4x-body-fold-realization-diagnostics-v1"
)
MOVING_GAUGE_BODY_FOLD_REALIZATION_SCHEMA = (
    "neural-forge-lfm2-exact4x-moving-gauge-body-fold-realization-v1"
)
GLOBAL_RESIDUAL_CLOSURE_SELECTION_SCHEMA = (
    "neural-forge-lfm2-exact4x-global-residual-closure-selection-v1"
)
BODY_GLOBAL_PARENT_GEOMETRY_SCHEMA = (
    "neural-forge-lfm2-exact4x-body-global-parent-geometry-v1"
)
POSTFOLD_RECLOSURE_SEED_ARCHIVE_PUBLICATION_SCHEMA = (
    "neural-forge-lfm2-exact4x-postfold-reclosure-seed-publication-v1"
)
class Lfm2Exact4xCompositeFoldError(RuntimeError):
    """One externally pinned composite-fold invariant failed."""


def _sealed_result_receipt_body(
    value: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return one canonically authenticated result-receipt body."""

    if not isinstance(value, Mapping):
        return None
    body = dict(value)
    supplied = body.pop("result_sha256", None)
    try:
        expected = canonical_sha256(body)
    except (TypeError, ValueError, OverflowError):
        return None
    return body if supplied == expected else None


def _nonzero_lowercase_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value != "0" * 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _causal_structural_install_foundation_lineage_valid(
    causal_bridge_receipt: Mapping[str, Any],
    structural_install_receipt: Mapping[str, Any],
    coordinate_evidence_receipt: Mapping[str, Any],
    *,
    target_vocabulary_size: int,
) -> bool:
    """Bind causal logits to the materialized foundation across dtype domains.

    The semantic-foundation interface is analytically owned in FP64, while the
    exact child installs that table in its requested materialization dtype.
    Raw tensor hashes therefore cannot be equal across those two legitimate
    representations.  The structural-install receipt is the authenticated
    transition between them and is the authority for the causal baseline.
    """

    if (
        not isinstance(causal_bridge_receipt, Mapping)
        or isinstance(target_vocabulary_size, bool)
        or not isinstance(target_vocabulary_size, int)
        or target_vocabulary_size <= 0
    ):
        return False
    structural = _sealed_result_receipt_body(structural_install_receipt)
    coordinate = _sealed_result_receipt_body(coordinate_evidence_receipt)
    if structural is None or coordinate is None:
        return False
    foundation = coordinate.get("foundation_interface")
    token_bridge = coordinate.get("token_bridge")
    causal_contract = coordinate.get("causal_output_contract")
    if not all(
        isinstance(value, Mapping)
        for value in (foundation, token_bridge, causal_contract)
    ):
        return False
    installed_baseline = structural.get("installed_input_table_sha256")
    return (
        structural.get("schema")
        == LFM2_INDEPENDENT_STRUCTURAL_INTERFACE_INSTALL_SCHEMA
        and coordinate.get("schema")
        == LFM2_CIRCUIT_BALANCED_COORDINATE_EVIDENCE_SCHEMA
        and coordinate.get("recipe") == LFM2_CIRCUIT_BALANCED_COORDINATE_RECIPE
        and _nonzero_lowercase_sha256(
            foundation.get("structural_target_table_sha256")
        )
        and _nonzero_lowercase_sha256(foundation.get("report_sha256"))
        and _nonzero_lowercase_sha256(installed_baseline)
        and structural.get("foundation_active_interface_report_sha256")
        == foundation.get("report_sha256")
        and foundation.get("structural_target_table_preserved") is True
        and foundation.get("target_vocabulary_row_count") == target_vocabulary_size
        and token_bridge.get("full_target_vocabulary_row_coverage") is True
        and token_bridge.get("full_target_vocabulary_row_count")
        == target_vocabulary_size
        and causal_contract.get("full_structural_table_is_foundation") is True
        and causal_contract.get("partial_causal_trie_may_supply_residual_only")
        is True
        and causal_contract.get("partial_causal_trie_may_replace_structural_table")
        is False
        and causal_contract.get("unsupported_logit_policy_required")
        == causal_bridge_receipt.get("unsupported_logit_policy")
        and causal_contract.get(
            "fully_unsupported_rows_must_equal_structural_baseline"
        )
        is True
        and structural.get("input_output_values_equal_at_structural_install")
        is True
        and structural.get("installed_output_baseline_sha256")
        == installed_baseline
        and causal_bridge_receipt.get("structural_output_table_sha256")
        == installed_baseline
        and causal_bridge_receipt.get("target_row_count")
        == target_vocabulary_size
    )


def _bounded_exception_message(exc: BaseException) -> dict[str, Any]:
    try:
        value = str(exc)
    except BaseException:
        value = "<exception message unavailable>"
    original_characters = len(value)
    truncated = original_characters > FAILURE_EXCEPTION_MESSAGE_MAXIMUM_CHARACTERS
    return {
        "message": value[:FAILURE_EXCEPTION_MESSAGE_MAXIMUM_CHARACTERS],
        "message_character_count": min(
            original_characters,
            FAILURE_EXCEPTION_MESSAGE_MAXIMUM_CHARACTERS,
        ),
        "message_truncated": truncated,
    }


def _bounded_exception_traceback(exc: BaseException) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    current = exc.__traceback__
    while current is not None:
        code = current.tb_frame.f_code
        source = Path(code.co_filename)
        try:
            source_name = source.resolve().relative_to(_REPOSITORY_ROOT).as_posix()
            source_scope = "repository-relative"
        except (OSError, ValueError):
            source_name = source.name
            source_scope = "external-basename"
        frames.append(
            {
                "source": source_name,
                "source_scope": source_scope,
                "line": current.tb_lineno,
                "function": code.co_name,
            }
        )
        current = current.tb_next
    truncated = len(frames) > FAILURE_EXCEPTION_TRACEBACK_MAXIMUM_FRAMES
    retained = frames[-FAILURE_EXCEPTION_TRACEBACK_MAXIMUM_FRAMES:]
    return {
        "traceback_frame_count": len(retained),
        "traceback_frames": retained,
        "traceback_frames_truncated": truncated,
        "traceback_source_lines_serialized": False,
        "traceback_locals_serialized": False,
    }


def _exception_chain_receipt(exc: BaseException) -> dict[str, Any]:
    """Seal bounded causes and frame locations without source lines or locals."""

    if not isinstance(exc, BaseException):
        raise TypeError("exc must be a BaseException.")
    entries: list[dict[str, Any]] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    relationship = "root"
    cycle_detected = False
    while current is not None and len(entries) < FAILURE_EXCEPTION_CHAIN_MAXIMUM_DEPTH:
        identity = id(current)
        if identity in seen:
            cycle_detected = True
            break
        seen.add(identity)
        exception_type = type(current)
        entries.append(
            {
                "depth": len(entries),
                "relationship": relationship,
                "exception_type": exception_type.__name__,
                "exception_qualified_type": (
                    f"{exception_type.__module__}.{exception_type.__qualname__}"
                ),
                **_bounded_exception_message(current),
                **_bounded_exception_traceback(current),
            }
        )
        explicit_cause = current.__cause__
        implicit_context = current.__context__
        if explicit_cause is not None:
            current = explicit_cause
            relationship = "cause"
        elif implicit_context is not None and not current.__suppress_context__:
            current = implicit_context
            relationship = "context"
        else:
            current = None
    truncated = current is not None and not cycle_detected
    body = {
        "schema": FAILURE_EXCEPTION_CHAIN_SCHEMA,
        "maximum_depth": FAILURE_EXCEPTION_CHAIN_MAXIMUM_DEPTH,
        "entry_count": len(entries),
        "entries": entries,
        "truncated": truncated,
        "cycle_detected": cycle_detected,
        "traceback_source_lines_serialized": False,
        "traceback_locals_serialized": False,
    }
    return {**body, "chain_sha256": canonical_sha256(body)}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise Lfm2Exact4xCompositeFoldError(
            f"{label} must be a lowercase SHA-256 digest."
        )
    return normalized


def _ordinary_directory(value: Path, label: str) -> Path:
    if "://" in os.fspath(value):
        raise Lfm2Exact4xCompositeFoldError(f"{label} must be local.")
    path = value.expanduser().absolute()
    if path.is_symlink() or not path.is_dir():
        raise Lfm2Exact4xCompositeFoldError(f"{label} is missing: {path}.")
    return path


def _ordinary_file(value: Path, label: str) -> Path:
    if "://" in os.fspath(value):
        raise Lfm2Exact4xCompositeFoldError(f"{label} must be local.")
    path = value.expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise Lfm2Exact4xCompositeFoldError(f"{label} is missing: {path}.")
    return path


def _new_run_directory(value: Path) -> Path:
    if "://" in os.fspath(value):
        raise Lfm2Exact4xCompositeFoldError("run_directory must be local.")
    path = value.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise Lfm2Exact4xCompositeFoldError("run_directory must not already exist.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()
    return path


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Lfm2Exact4xCompositeFoldError(
            f"Refusing to replace existing evidence: {path}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    path.write_text(payload + "\n", encoding="ascii")


def _persist_semantic_layer_stack_pre_auth_receipt(
    run_root: Path,
    receipt: Mapping[str, Any],
) -> None:
    if not receipt:
        return
    body = dict(receipt)
    result_sha256 = body.pop("result_sha256", None)
    if result_sha256 != canonical_sha256(body):
        raise Lfm2Exact4xCompositeFoldError(
            "Semantic layer-stack pre-auth receipt is not canonically sealed."
        )
    _write_json(run_root / "semantic-layer-stack-pre-auth.json", receipt)


@contextmanager
def _semantic_layer_stack_pre_auth_capture(
    run_root: Path,
) -> Iterator[dict[str, Any]]:
    pre_auth_receipt: dict[str, Any] = {}
    try:
        yield pre_auth_receipt
    except BaseException as original_exc:
        try:
            _persist_semantic_layer_stack_pre_auth_receipt(
                run_root,
                pre_auth_receipt,
            )
        except BaseException as persistence_exc:
            original_exc.add_note(
                "Could not persist semantic-layer-stack-pre-auth.json: "
                f"{type(persistence_exc).__name__}: {persistence_exc}"
            )
        raise
    _persist_semantic_layer_stack_pre_auth_receipt(run_root, pre_auth_receipt)


def _emit(stage: str, **values: Any) -> None:
    print(
        json.dumps(
            {"stage": stage, **values},
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _unique_parameter_count(model: Any) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def _child_torch_dtype(value: str) -> Any:
    normalized = str(value).strip().lower()
    try:
        return {
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[normalized]
    except KeyError as exc:
        raise Lfm2Exact4xCompositeFoldError(
            "child_dtype must be bfloat16 or float32."
        ) from exc


def _model_parameter_dtype_name(model: Any) -> str:
    names = {
        str(parameter.dtype).removeprefix("torch.") for parameter in model.parameters()
    }
    if len(names) != 1:
        raise Lfm2Exact4xCompositeFoldError(
            "The exact-4x child must have one uniform parameter dtype."
        )
    return next(iter(names))


def _parameter_storage_addresses(model: Any) -> set[int]:
    return {
        int(parameter.untyped_storage().data_ptr()) for parameter in model.parameters()
    }


def _artifact_inventory(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise Lfm2Exact4xCompositeFoldError(
                "Standalone checkpoint contains a symlink."
            )
        if path.is_file():
            rows.append(
                {
                    "relative_path": path.relative_to(directory).as_posix(),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _file_sha256(path),
                }
            )
    if not rows:
        raise Lfm2Exact4xCompositeFoldError("Standalone checkpoint is empty.")
    return rows


def _target_topology(model: Any) -> dict[str, Any]:
    config = model.config
    return {
        "vocab_size": int(config.vocab_size),
        "hidden_size": int(config.hidden_size),
        "intermediate_size": int(config.intermediate_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "head_dim": int(config.head_dim),
        "tie_word_embeddings": bool(config.tie_word_embeddings),
        "layer_types": [str(value) for value in config.layer_types],
    }


def _body_fold_realization_receipt(
    layer_stack_receipt: Mapping[str, Any],
    *,
    child_dtype: str,
    requested_representability_level: str,
) -> dict[str, Any]:
    """Bind direction and functional-effect evidence without scoring quality."""

    closure = layer_stack_receipt.get("global_residual_closure_receipt")
    if not isinstance(closure, Mapping):
        raise Lfm2Exact4xCompositeFoldError(
            "Semantic layer stack lacks its residual-closure receipt."
        )
    dense = closure.get("dense_transaction_receipt")
    linearized = closure.get("linearized_residual_diagnostics")
    if not isinstance(dense, Mapping) or not isinstance(linearized, Mapping):
        raise Lfm2Exact4xCompositeFoldError(
            "Residual closure lacks dense or functional diagnostics."
        )
    representability = dense.get("target_dtype_representability")
    compute = dense.get("compute_execution_receipt")
    if not isinstance(representability, Mapping) or not isinstance(compute, Mapping):
        raise Lfm2Exact4xCompositeFoldError(
            "Residual closure lacks target-dtype realization evidence."
        )
    normalized_dtype = str(child_dtype).strip().lower()
    normalized_level = str(requested_representability_level).strip().lower()
    selected_level = str(representability.get("selected_level", "")).strip().lower()
    selected_multiplier = representability.get("selected_ray_multiplier")
    parameter_diagnostics = representability.get("parameter_diagnostics")
    if (
        selected_level != normalized_level
        or type(selected_multiplier) not in (int, float)
        or not isinstance(parameter_diagnostics, list)
        or not parameter_diagnostics
        or {str(value.get("target_dtype")) for value in parameter_diagnostics}
        != {normalized_dtype}
        or compute.get("compute_dtype") != "float64"
    ):
        raise Lfm2Exact4xCompositeFoldError(
            "Body realization disagrees with its requested dtype/analytic path."
        )
    if normalized_dtype == "float32" and (
        selected_level != "requested" or float(selected_multiplier) != 1.0
    ):
        raise Lfm2Exact4xCompositeFoldError(
            "The FP32 directional path must retain the requested unit ray."
        )
    requested_before = linearized.get("weighted_residual_squared_before")
    requested_after = linearized.get(
        "weighted_residual_squared_at_requested_analytic_ray"
    )
    installed_projected_after = linearized.get(
        "weighted_residual_squared_at_installed_delta_projected_ray"
    )
    diagnostic_scalars = (
        requested_before,
        requested_after,
        representability.get("requested_analytic_dense_delta_l2_norm"),
        representability.get("installed_target_dtype_dense_delta_l2_norm"),
    )
    if any(
        type(value) not in (int, float) or not math.isfinite(float(value))
        for value in diagnostic_scalars
    ):
        raise Lfm2Exact4xCompositeFoldError(
            "Body realization lacks finite direction/function diagnostics."
        )
    optional_diagnostics = (
        installed_projected_after,
        representability.get("installed_to_requested_analytic_direction_cosine"),
        representability.get("installed_to_requested_analytic_delta_l2_ratio"),
        representability.get("installed_to_requested_analytic_projection_scale"),
    )
    if any(
        value is not None
        and (type(value) not in (int, float) or not math.isfinite(float(value)))
        for value in optional_diagnostics
    ):
        raise Lfm2Exact4xCompositeFoldError(
            "Body realization contains a non-finite optional direction diagnostic."
        )
    body = {
        "schema": BODY_FOLD_REALIZATION_SCHEMA,
        "semantic_layer_stack_sha256": layer_stack_receipt.get("result_sha256"),
        "global_residual_closure_sha256": closure.get("report_sha256"),
        "dense_transaction_sha256": dense.get("report_sha256"),
        "linearized_residual_diagnostics_sha256": linearized.get("report_sha256"),
        "child_materialization_dtype": normalized_dtype,
        "dense_absorption_analytic_compute_dtype": compute["compute_dtype"],
        "causal_head_analytic_compute_dtype": "float32",
        "requested_representability_level": normalized_level,
        "selected_representability_level": selected_level,
        "requested_analytic_ray_multiplier": representability.get(
            "requested_analytic_ray_multiplier"
        ),
        "materialized_ray_multiplier": float(selected_multiplier),
        "requested_unit_ray_preserved_without_ulp_escalation": bool(
            selected_level == "requested" and float(selected_multiplier) == 1.0
        ),
        "target_dtype_ulp_escalation_applied": bool(
            representability.get("target_dtype_ulp_escalation_applied")
        ),
        "requested_analytic_delta_l2_norm": representability.get(
            "requested_analytic_dense_delta_l2_norm"
        ),
        "installed_target_dtype_delta_l2_norm": representability.get(
            "installed_target_dtype_dense_delta_l2_norm"
        ),
        "installed_to_requested_direction_cosine": representability.get(
            "installed_to_requested_analytic_direction_cosine"
        ),
        "installed_to_requested_delta_l2_ratio": representability.get(
            "installed_to_requested_analytic_delta_l2_ratio"
        ),
        "installed_delta_projected_ray_scale": representability.get(
            "installed_to_requested_analytic_projection_scale"
        ),
        "requested_functional_weighted_residual_squared_before": requested_before,
        "requested_functional_weighted_residual_squared_after": requested_after,
        "requested_analytic_functional_effect_weighted_residual_reduction": (
            linearized.get(
                "requested_analytic_functional_effect_weighted_residual_reduction"
            )
        ),
        "installed_projected_ray_functional_weighted_residual_squared_after": (
            installed_projected_after
        ),
        "installed_projected_ray_functional_effect_weighted_residual_reduction": (
            linearized.get(
                "installed_delta_projected_ray_functional_effect_weighted_"
                "residual_reduction"
            )
        ),
        "installed_functional_effect_is_projected_ray_diagnostic": True,
        "off_ray_error_is_reported_by_direction_cosine": True,
        "physical_target_dtype_effect_observed": bool(
            int(representability.get("selected_changed_coordinate_count", 0)) > 0
        ),
        "seed_identical_projection_claims_realized_effect": False,
        "direction_and_function_diagnostics_are_advisory_only": True,
        "diagnostic_numbers_have_accept_reject_delete_stop_authority": False,
        "decoded_whole_responses_are_primary_intelligence_evidence": True,
        "float32_storage_is_not_a_quantization_claim": normalized_dtype == "float32",
        "operation_audit": {
            "training_steps": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "parent_trace_recaptures": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "parameters_sparsified": 0,
            "runtime_shared_parameters": 0,
        },
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _global_residual_closure_selection_receipt(
    layer_stack_receipt: Mapping[str, Any],
    *,
    requested_closure_version: str,
) -> dict[str, Any]:
    """Bind the caller request to the engine selected by stack capabilities."""

    stack = dict(layer_stack_receipt)
    stack_sha256 = stack.pop("result_sha256", None)
    requested = str(requested_closure_version).strip().lower()
    selected = layer_stack_receipt.get("global_residual_closure_version")
    if (
        stack_sha256 != canonical_sha256(stack)
        or requested not in LFM2_GLOBAL_RESIDUAL_CLOSURE_REQUESTS
        or layer_stack_receipt.get("global_residual_requested_closure_version")
        != requested
        or layer_stack_receipt.get("global_residual_selected_closure_version")
        != selected
        or layer_stack_receipt.get(
            "global_residual_closure_selection_authenticated"
        )
        is not True
        or not isinstance(selected, str)
        or selected not in LFM2_GLOBAL_RESIDUAL_CLOSURE_REQUESTS
        or selected == LFM2_GLOBAL_RESIDUAL_CLOSURE_AUTO
    ):
        raise Lfm2Exact4xCompositeFoldError(
            "The semantic stack did not authenticate the requested and selected "
            "global residual closure modes."
        )
    moving_selected = selected == "moving_gauge_v3"
    if moving_selected != (
        layer_stack_receipt.get("global_residual_moving_gauge_v3") is True
    ) or moving_selected != (
        layer_stack_receipt.get(
            "global_residual_deferred_physical_finalizer_required"
        )
        is True
    ):
        raise Lfm2Exact4xCompositeFoldError(
            "The selected residual engine disagrees with its physical finalizer "
            "capability."
        )
    body = {
        "schema": GLOBAL_RESIDUAL_CLOSURE_SELECTION_SCHEMA,
        "semantic_layer_stack_sha256": stack_sha256,
        "requested_closure_version": requested,
        "selected_closure_version": selected,
        "requested_auto_selection": requested == LFM2_GLOBAL_RESIDUAL_CLOSURE_AUTO,
        "selection_authenticated_by_stack_capabilities": True,
        "moving_gauge_selected": moving_selected,
        "post_w2_physical_finalizer_required": moving_selected,
        "numeric_or_behavioral_metric_selected_engine": False,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _moving_gauge_body_fold_realization_receipt(
    layer_stack_receipt: Mapping[str, Any],
    closure_selection_receipt: Mapping[str, Any],
    finalization_receipt: Mapping[str, Any],
    *,
    child_dtype: str,
    requested_representability_level: str,
    dynamic_model_state_after_sha256: str,
    lfm2_identity_after_sha256: str,
) -> dict[str, Any]:
    """Bind the physical post-W2 moving closure used by the final child body."""

    finalization = dict(finalization_receipt)
    finalization_sha256 = finalization.pop("result_sha256", None)
    selection = dict(closure_selection_receipt)
    selection_sha256 = selection.pop("receipt_sha256", None)
    normalized_dtype = str(child_dtype).strip().lower()
    normalized_level = str(requested_representability_level).strip().lower()
    changed = finalization_receipt.get("changed_state_keys")
    if (
        finalization_sha256 != canonical_sha256(finalization)
        or selection_sha256 != canonical_sha256(selection)
        or closure_selection_receipt.get("selected_closure_version")
        != "moving_gauge_v3"
        or closure_selection_receipt.get("post_w2_physical_finalizer_required")
        is not True
        or finalization_receipt.get("semantic_layer_stack_result_sha256")
        != layer_stack_receipt.get("result_sha256")
        or finalization_receipt.get("deferred_finalization_plan_sha256")
        != layer_stack_receipt.get(
            "moving_gauge_deferred_finalization_plan_sha256"
        )
        or finalization_receipt.get(
            "final_w2_was_installed_before_moving_publication"
        )
        is not True
        or finalization_receipt.get("final_w2_changed_by_moving_finalizer")
        is not False
        or finalization_receipt.get(
            "actual_final_recapture_includes_corrected_final_w2"
        )
        is not True
        or finalization_receipt.get("model_state_after_sha256")
        != lfm2_identity_after_sha256
        or not isinstance(changed, list)
        or not changed
        or any(
            finalization_receipt.get(name) != 0
            for name in (
                "autograd_grad_calls",
                "backward_calls",
                "optimizer_steps",
                "training_steps",
                "parameters_pruned",
                "parameters_quantized",
                "parent_model_forward_calls",
                "parent_trace_recaptures",
            )
        )
    ):
        raise Lfm2Exact4xCompositeFoldError(
            "Moving-gauge body realization is not an authenticated effectful "
            "post-W2 physical transaction."
        )
    body = {
        "schema": MOVING_GAUGE_BODY_FOLD_REALIZATION_SCHEMA,
        "semantic_layer_stack_sha256": layer_stack_receipt["result_sha256"],
        "global_residual_closure_selection_sha256": selection_sha256,
        "moving_gauge_post_w2_finalization_sha256": finalization_sha256,
        "global_residual_closure_sha256": layer_stack_receipt[
            "global_residual_closure_receipt_sha256"
        ],
        "child_materialization_dtype": normalized_dtype,
        "requested_representability_level": normalized_level,
        "selected_closure_version": "moving_gauge_v3",
        "physical_target_dtype_effect_observed": True,
        "changed_state_key_count": len(changed),
        "final_child_body_dynamic_model_state_sha256": (
            dynamic_model_state_after_sha256
        ),
        "final_child_body_lfm2_identity_sha256": lfm2_identity_after_sha256,
        "actual_final_physical_recapture_sha256": finalization_receipt[
            "actual_final_physical_recapture_sha256"
        ],
        "actual_final_recapture_includes_corrected_final_w2": True,
        "causal_head_must_be_solved_after_final_physical_body": True,
        "direction_and_function_diagnostics_are_advisory_only": True,
        "diagnostic_numbers_have_accept_reject_delete_stop_authority": False,
        "decoded_whole_responses_are_primary_intelligence_evidence": True,
        "float32_storage_is_not_a_quantization_claim": normalized_dtype == "float32",
        "operation_audit": {
            "training_steps": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "parent_trace_recaptures": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "parameters_sparsified": 0,
            "runtime_shared_parameters": 0,
        },
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _frontier_receipt(
    *,
    run_directory: Path,
    candidate_id: str,
    checkpoint: Path,
    model_state_sha256: str,
    checkpoint_inventory: Sequence[Mapping[str, Any]],
    lfm2_identity_sha256: str,
    lfm2_config_sha256: str,
    materialization_dtype: str,
) -> dict[str, Any]:
    body = {
        "schema": FRONTIER_SCHEMA,
        "status": "complete",
        "run_directory": str(run_directory),
        "candidate_ids": [candidate_id],
        "materializations": [
            {
                "candidate_id": candidate_id,
                "candidate_state_sha256": model_state_sha256,
                "model_state_sha256": model_state_sha256,
                "checkpoint_path": str(checkpoint),
                "checkpoint_inventory": [dict(value) for value in checkpoint_inventory],
                "checkpoint_inventory_sha256": canonical_sha256(
                    [dict(value) for value in checkpoint_inventory]
                ),
                "lfm2_identity_sha256": lfm2_identity_sha256,
                "lfm2_config_sha256": lfm2_config_sha256,
                "materialization_dtype": materialization_dtype,
                "unique_parameters": EXPECTED_CHILD_UNIQUE_PARAMETERS,
                "standalone_fresh_reload_verified": True,
            }
        ],
        "all_retained_candidates_materialized": True,
        "candidate_metric_threshold_used_for_materialization": False,
        "decoded_behavior_is_primary_evidence": True,
        "interface_isolation_control": False,
        "final_four_x_topology": True,
        "automatic_behavior_acceptance_or_rejection_performed": False,
        "binary_manual_verdict_required": False,
    }
    return {**body, "run_sha256": canonical_sha256(body)}


def run(
    *,
    run_directory: Path,
    parent_model_directory: Path,
    donor_tokenizer_directory: Path,
    target_tokenizer_directory: Path,
    bridge_directory: Path,
    expected_bridge_artifact_sha256: str,
    compiler_cache_directory: Path,
    expected_compiler_cache_key: str,
    semantic_cache_directory: Path,
    expected_semantic_cache_key: str,
    trajectory_corpus: Path,
    selective_catalog_store_directory: Path,
    selective_catalog_receipt_json: Path,
    expected_selective_catalog_receipt_sha256: str,
    expected_parent_identity_sha256: str,
    expected_parent_unique_parameters: int,
    prompts_json: Path,
    expected_prompts_file_sha256: str,
    parent_envelope: Path,
    expected_parent_envelope_sha256: str,
    expected_parent_envelope_file_sha256: str,
    body_recipe: str,
    expected_body_recipe_sha256: str,
    short_conv_construction: str,
    blank_seed: int,
    model_device: str,
    fold_compute_device: str,
    analytic_solve_device: str,
    global_residual_backend: str,
    global_residual_basis_rank: int | None,
    correction_scale: float,
    target_dtype_representability_level: str,
    child_dtype: str,
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
    global_residual_closure_version: str = LFM2_GLOBAL_RESIDUAL_CLOSURE_AUTO,
    coordinate_completion_recipe: str = LFM2_CIRCUIT_BALANCED_COORDINATE_RECIPE,
    normalized_input_correction_rho: float = 0.25,
    normalized_input_request_capture_directory: Path | None = None,
    live_child_prefix_recapture_mode: str = (
        LFM2_LIVE_CHILD_PREFIX_RECAPTURE_DISABLED
    ),
    existing_target: DynamicTargetSynthesisResult | None = None,
    existing_parent_model: Any | None = None,
    defer_publication_to_caller: bool = False,
    existing_donor_tokenizer: Any | None = None,
    existing_target_tokenizer: Any | None = None,
    existing_bridge: Any | None = None,
    existing_compiler_cache: Any | None = None,
    existing_semantic_cache: Any | None = None,
) -> dict[str, Any]:
    if (existing_target is None) != (existing_parent_model is None):
        raise Lfm2Exact4xCompositeFoldError(
            "existing_target and existing_parent_model must be supplied together."
        )
    if not isinstance(defer_publication_to_caller, bool):
        raise Lfm2Exact4xCompositeFoldError(
            "defer_publication_to_caller must be boolean."
        )
    if defer_publication_to_caller and existing_target is None:
        raise Lfm2Exact4xCompositeFoldError(
            "Deferred publication is only valid for an existing compiler blank."
        )
    existing_compiler_blank_used = existing_target is not None
    preopened_resources = (
        existing_donor_tokenizer,
        existing_target_tokenizer,
        existing_bridge,
        existing_compiler_cache,
        existing_semantic_cache,
    )
    if any(value is not None for value in preopened_resources) and not all(
        value is not None for value in preopened_resources
    ):
        raise Lfm2Exact4xCompositeFoldError(
            "Preopened token and cache resources must be supplied as one bundle."
        )
    preopened_semantic_resources_used = all(
        value is not None for value in preopened_resources
    )
    run_root = _new_run_directory(run_directory)
    normalized_input_capture_root = (
        None
        if normalized_input_request_capture_directory is None
        else prepare_lfm2_normalized_input_replay_capture_root(
            normalized_input_request_capture_directory
        )
    )
    child_torch_dtype = _child_torch_dtype(child_dtype)
    child_dtype_name = str(child_torch_dtype).removeprefix("torch.")
    normalized_representability_level = (
        str(target_dtype_representability_level).strip().lower()
    )
    if child_dtype_name == "float32" and normalized_representability_level != (
        "requested"
    ):
        raise Lfm2Exact4xCompositeFoldError(
            "FP32 body folding requires requested directional realization, not "
            "an effect/mass ULP escalation."
        )
    parent_path = _ordinary_directory(parent_model_directory, "parent model")
    donor_path = _ordinary_directory(donor_tokenizer_directory, "donor tokenizer")
    target_path = _ordinary_directory(target_tokenizer_directory, "target tokenizer")
    bridge_path = _ordinary_directory(bridge_directory, "tokenizer bridge")
    compiler_cache_path = _ordinary_directory(
        compiler_cache_directory,
        "compiler cache",
    )
    semantic_cache_path = _ordinary_directory(
        semantic_cache_directory,
        "semantic cache",
    )
    trajectory_path = _ordinary_file(trajectory_corpus, "trajectory corpus")
    catalog_store = _ordinary_directory(
        selective_catalog_store_directory,
        "selective catalog store",
    )
    catalog_receipt_path = _ordinary_file(
        selective_catalog_receipt_json,
        "selective catalog receipt",
    )
    prompts_path = _ordinary_file(prompts_json, "whole-response prompts")
    envelope_path = _ordinary_file(parent_envelope, "parent response envelope")

    expected_parent = _digest(
        expected_parent_identity_sha256,
        "expected parent identity",
    )
    expected_bridge = _digest(
        expected_bridge_artifact_sha256,
        "expected bridge artifact",
    )
    expected_compiler_cache = _digest(
        expected_compiler_cache_key,
        "expected compiler cache key",
    )
    expected_semantic_cache = _digest(
        expected_semantic_cache_key,
        "expected semantic cache key",
    )
    expected_catalog = _digest(
        expected_selective_catalog_receipt_sha256,
        "expected selective catalog receipt",
    )
    expected_prompts_file = _digest(
        expected_prompts_file_sha256,
        "expected prompts file",
    )
    expected_envelope = _digest(
        expected_parent_envelope_sha256,
        "expected parent envelope",
    )
    expected_envelope_file = _digest(
        expected_parent_envelope_file_sha256,
        "expected parent envelope file",
    )
    if _file_sha256(prompts_path) != expected_prompts_file:
        raise Lfm2Exact4xCompositeFoldError("Whole-response prompt bytes changed.")
    if _file_sha256(envelope_path) != expected_envelope_file:
        raise Lfm2Exact4xCompositeFoldError("Parent envelope bytes changed.")
    if (
        expected_parent_unique_parameters != EXPECTED_PARENT_UNIQUE_PARAMETERS
        or not torch.cuda.is_available()
        or not model_device.startswith("cuda:")
        or not fold_compute_device.startswith("cuda:")
        or not analytic_solve_device.startswith("cuda:")
    ):
        raise Lfm2Exact4xCompositeFoldError(
            "The production proof requires its exact parent count and explicit CUDA."
        )
    fold_execution_profile = resolve_analytic_execution_profile(
        fold_compute_device,
        require_cuda=True,
        allow_cpu_fallback=False,
    )
    analytic_execution_profile = resolve_analytic_execution_profile(
        analytic_solve_device,
        require_cuda=True,
        allow_cpu_fallback=False,
    )
    recipe = resolve_lfm2_semantic_body_recipe(body_recipe)
    if recipe.recipe_sha256 != _digest(
        expected_body_recipe_sha256,
        "expected body recipe",
    ):
        raise Lfm2Exact4xCompositeFoldError("Body recipe differs from its pin.")
    requested_global_residual_closure_version = (
        str(global_residual_closure_version).strip().lower()
    )
    if requested_global_residual_closure_version not in (
        LFM2_GLOBAL_RESIDUAL_CLOSURE_REQUESTS
    ):
        raise Lfm2Exact4xCompositeFoldError(
            "Unknown global residual closure request."
        )
    if short_conv_construction not in LFM2_SEMANTIC_SHORT_CONV_CONSTRUCTION_IDS:
        raise Lfm2Exact4xCompositeFoldError(
            "Unknown semantic short-convolution construction."
        )
    resolved_short_conv_construction = (
        resolve_lfm2_semantic_short_conv_construction(short_conv_construction)
    )
    if (
        not resolved_short_conv_construction.uses_body_global_coordinate_completion
        or coordinate_completion_recipe
        not in LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_RECIPES
    ):
        raise Lfm2Exact4xCompositeFoldError(
            "The exact-4x body-global proof requires an authenticated short-conv "
            "construction with coordinate-completion capability and one supported "
            "completion recipe."
        )

    if preopened_semantic_resources_used:
        _emit("reuse-resolver-authenticated-local-assets")
        donor_tokenizer = existing_donor_tokenizer
        target_tokenizer = existing_target_tokenizer
        bridge = existing_bridge
        compiler_cache = existing_compiler_cache
        semantic_cache = existing_semantic_cache
        verify_bridge = getattr(bridge, "verify_files", None)
        if callable(verify_bridge):
            verify_bridge()
    else:
        _emit("load-authenticated-local-assets")
        donor_tokenizer = load_local_tokenizer(donor_path)
        target_tokenizer = load_local_tokenizer(target_path)
        bridge = load_strict_tokenizer_bridge_artifact(
            bridge_path,
            donor_tokenizer=donor_tokenizer,
            target_tokenizer=target_tokenizer,
            expected_artifact_sha256=expected_bridge,
        )
        compiler_cache = open_lfm2_circuit_trace_cache(
            compiler_cache_path,
            # The 85 GB compiler cache is consumed through its separately sealed
            # selective catalog below.  A throwaway full-payload scan here would
            # reread every unselected tensor before the authenticated provider
            # opens and verifies the exact material it uses.
            verify=False,
        )
        semantic_cache = open_lfm2_circuit_trace_cache(
            semantic_cache_path,
            # Terminal collection opens a selective authenticated session which
            # hashes the committed files and verifies every selected tensor.
            verify=False,
        )
    if (
        compiler_cache.cache_key != expected_compiler_cache
        or semantic_cache.cache_key != expected_semantic_cache
        or bridge.artifact_sha256 != expected_bridge
        or compiler_cache.parent_fingerprint != expected_parent
        or semantic_cache.parent_fingerprint != expected_parent
        or fingerprint_hf_tokenizer_semantics(donor_tokenizer)
        != bridge.donor_tokenizer_semantics_sha256
        or fingerprint_hf_tokenizer_semantics(target_tokenizer)
        != bridge.target_tokenizer_semantics_sha256
    ):
        raise Lfm2Exact4xCompositeFoldError(
            "Caches, bridge, tokenizers, or common parent identity diverged."
        )
    catalog_preparation = read_lfm2_semantic_foundation_catalog_preparation_metadata(
        catalog_receipt_path,
        expected_receipt_sha256=expected_catalog,
        cache=compiler_cache,
        catalog_store_directory=catalog_store,
    )

    prompt_rows = [dict(value) for value in load_review_prompts(prompts_path)]
    prompt_set_sha256 = canonical_sha256(prompt_rows)
    parent_behavior = _load_parent_envelope(
        envelope_path,
        expected_envelope_sha256=expected_envelope,
        expected_file_sha256=expected_envelope_file,
        expected_prompt_set_sha256=prompt_set_sha256,
        expected_prompt_ids=[str(value["id"]) for value in prompt_rows],
    )
    if (
        parent_behavior.get("parent_identity_sha256") != expected_parent
        or parent_behavior.get("parent_unique_parameters")
        != EXPECTED_PARENT_UNIQUE_PARAMETERS
        or parent_behavior.get("maximum_new_tokens") != maximum_new_tokens
    ):
        raise Lfm2Exact4xCompositeFoldError(
            "Cached parent behavior differs from this proof request."
        )

    _emit("collect-cache-only-terminal-and-causal-evidence")
    terminal = collect_lfm2_terminal_cache_evidence(
        semantic_cache,
        expected_cache_key=expected_semantic_cache,
        expected_parent_identity_sha256=expected_parent,
        bridge=bridge,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
    )
    terminal_receipt = terminal.as_dict()
    _write_json(run_root / "terminal-cache-evidence.json", terminal_receipt)
    causal_trie = compile_lfm2_causal_trie_evidence(
        terminal.donor_logits,
        terminal.causal_root_records,
        bridge,
        donor_tokenizer,
        target_tokenizer,
        parent_terminal_evidence=terminal_receipt,
        maximum_donor_candidates_per_root=64,
        maximum_internal_prefixes_per_root=8,
        maximum_total_internal_prefixes=512,
    )
    trie_receipt = causal_trie.as_dict()
    _write_json(run_root / "causal-token-trie-evidence.json", trie_receipt)
    sequence_plan = compile_lfm2_interface_sequence_plan(
        bridge,
        donor_tokenizer,
        target_tokenizer,
        contexts=DEFAULT_INTERFACE_CONTEXTS,
    )
    _write_json(run_root / "contextual-sequence-plan.json", sequence_plan)

    if existing_parent_model is None:
        _emit("load-parent-direct-local-without-execution")
        parent = load_lfm2_direct_local_model(
            parent_path,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        parent.requires_grad_(False)
        parent.eval()
        parent_model_loads_inside_kernel = 1
    else:
        _emit("reuse-compiler-parent-without-execution")
        parent = existing_parent_model
        parent_model_loads_inside_kernel = 0
        if bool(getattr(parent, "training", True)) or any(
            parameter.requires_grad for parameter in parent.parameters()
        ):
            raise Lfm2Exact4xCompositeFoldError(
                "Existing compiler parent must already be frozen in evaluation mode."
            )
    parent_state_before = fingerprint_model_state(parent)
    parent_storage = _parameter_storage_addresses(parent)
    parent_input = parent.get_input_embeddings().weight
    if (
        fingerprint_lfm2_parent(parent).identity_sha256 != expected_parent
        or _unique_parameter_count(parent) != EXPECTED_PARENT_UNIQUE_PARAMETERS
        or parent.get_output_embeddings().weight is not parent_input
        or tuple(parent_input.shape)
        != (bridge.transport_profile.donor_row_capacity, 1_024)
    ):
        raise Lfm2Exact4xCompositeFoldError(
            "Direct-local parent differs from the sealed cache and bridge."
        )

    _emit("plan-and-materialize-exact-untied-four-x-blank")
    source_parent_name = (
        PARENT_MODEL_ROLE
        if existing_target is None
        else str(existing_target.adapter_payload.source_parent)
    )
    parent_graph = inspect_model_circuits(
        parent,
        model_name=source_parent_name,
        tokenizer_fingerprint=bridge.donor_tokenizer_semantics_sha256,
    )
    native_target_contract = None
    if existing_target is None:
        # Reproducibility boundary: the standalone CLI retains the immutable
        # H776/I700 planner and exact count used by the original proof.
        catalogs = DenseTargetTopologyCatalogRegistry(
            (
                Lfm2DenseTargetTopologyCatalog(
                    tokenizer_bridge_artifact=bridge,
                    target_token_interface_mode="untied",
                ),
            )
        )
        budget_request = DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            compression_ratio=4,
            target_vocabulary_size=bridge.transport_profile.target_row_capacity,
            maximum_candidates=8,
        )
        frontier = synthesize_dense_parameter_frontier(
            parent_graph,
            budget_request,
            catalogs=catalogs,
        )
        selection = select_lfm2_untied_rebalance_candidate(frontier)
        adapter = Lfm2TargetSynthesisAdapter(
            terminal_topology=Lfm2Topology(1, 1, 32),
            evidence_fingerprint=semantic_cache.cache_key,
            device=torch.device(model_device),
            dtype=child_torch_dtype,
            tokenizer_bridge_artifact=bridge,
        )
        adapter_registry = DynamicTargetAdapterRegistry((adapter,))
        plan = plan_budgeted_dynamic_target(
            parent_graph,
            budget_request,
            child_name="lfm2-exact4x-contextual-residual-attention-child",
            selected_candidate_rank=selection.selected_candidate_rank,
            tokenizer_fingerprint=bridge.target_tokenizer_semantics_sha256,
            adapter_id=adapter.adapter_id,
            blank_seed=blank_seed,
            catalogs=catalogs,
        )
        if (
            plan.frontier.fingerprint != frontier.fingerprint
            or plan.target_request.exact_unique_parameters
            != EXPECTED_CHILD_UNIQUE_PARAMETERS
        ):
            raise Lfm2Exact4xCompositeFoldError(
                "Repeated dynamic planning changed the exact selected topology."
            )
        target = DynamicTargetSynthesizer(
            inspectors=default_inspector_registry(),
            adapters=adapter_registry,
        ).synthesize_planned(
            plan=plan,
            parent_models={source_parent_name: parent},
            parent_tokenizer_fingerprints={
                source_parent_name: bridge.donor_tokenizer_semantics_sha256
            },
        )
        target_model_materializations_inside_kernel = 1
        target_unique_parameters = EXPECTED_CHILD_UNIQUE_PARAMETERS
        expected_target_topology = {
            **EXPECTED_CHILD_TOPOLOGY,
            "layer_types": list(LFM2_NATIVE_EXACT4X_LAYER_TYPES),
        }
    else:
        target = existing_target
        verify_compiler_authenticated_target(target)
        selected_target = target.adapter_payload
        if not isinstance(selected_target, Lfm2DenseBudgetedSelectedTarget):
            raise Lfm2Exact4xCompositeFoldError(
                "Existing compiler blank is not a dense LFM2 selection."
            )
        validate_lfm2_dense_budgeted_child(
            selected_target,
            target.model,
            target.graph,
        )
        native_target_contract = derive_lfm2_native_exact4x_target_contract(
            target,
            expected_parent_unique_parameters=EXPECTED_PARENT_UNIQUE_PARAMETERS,
            expected_bridge_artifact_sha256=expected_bridge,
        )
        if parent_graph.fingerprint != native_target_contract.source_graph_fingerprint:
            raise Lfm2Exact4xCompositeFoldError(
                "Existing compiler target is bound to a different source graph."
            )
        if target.request.blank_seed != blank_seed:
            raise Lfm2Exact4xCompositeFoldError(
                "Existing compiler blank seed differs from the fold request."
            )
        target_model_materializations_inside_kernel = 0
        target_unique_parameters = native_target_contract.target_unique_parameters
        expected_target_topology = native_target_contract.topology_dict
    child = target.model
    if existing_target is None:
        child.requires_grad_(False)
        child.eval()
    elif bool(getattr(child, "training", True)) or any(
        parameter.requires_grad for parameter in child.parameters()
    ):
        raise Lfm2Exact4xCompositeFoldError(
            "Compiler blank must already be frozen in evaluation mode."
        )
    blank_state = fingerprint_model_state(child)
    topology = _target_topology(child)
    if (
        _unique_parameter_count(child) != target_unique_parameters
        or topology != expected_target_topology
        or target.blank_state_fingerprint != blank_state
        or target.blank_provenance is None
        or target.blank_provenance.compiler_created_blank is not True
        or target.blank_provenance.all_parameters_zero is not True
        or _model_parameter_dtype_name(child) != child_dtype_name
        or child.get_input_embeddings().weight is child.get_output_embeddings().weight
        or not _parameter_storage_addresses(child).isdisjoint(parent_storage)
    ):
        raise Lfm2Exact4xCompositeFoldError(
            "Dynamic target is not the exact dense independent zero blank."
        )
    _write_json(run_root / "blank-target.json", target.as_dict())

    _emit("compose-token-context-before-hidden-width-fold")
    contextual_parent_rows = compile_lfm2_contextual_input_table(
        parent_input,
        sequence_plan,
        strict_anchor_ppm=500_000,
        preserve_anchor_norm=True,
    )
    _emit("authenticate-once-and-migrate-selective-catalog-provider")
    selective_provider = (
        migrate_lfm2_cpcf_selective_evidence_provider_from_legacy_catalog(
            compiler_cache,
            artifact_store=catalog_store,
            expected_legacy_binding_sha256=_digest(
                catalog_preparation.get("binding_sha256"),
                "legacy catalog binding SHA-256",
            ),
            expected_legacy_manifest_sha256=_digest(
                catalog_preparation.get("artifact_manifest_sha256"),
                "legacy catalog manifest SHA-256",
            ),
            expected_legacy_catalog_sha256=_digest(
                catalog_preparation.get("artifact_catalog_sha256"),
                "legacy catalog artifact SHA-256",
            ),
        )
    )
    try:
        provider_artifact_receipt = selective_provider.catalog_artifact_receipt
        if provider_artifact_receipt is None:
            raise Lfm2Exact4xCompositeFoldError(
                "Provider migration did not publish or load an immutable catalog."
            )
        provider_migration_body = {
            "schema": "neural-forge-lfm2-exact4x-provider-migration-v1",
            "legacy_preparation_receipt_sha256": catalog_preparation[
                "preparation_receipt_sha256"
            ],
            "legacy_binding_sha256": catalog_preparation["binding_sha256"],
            "legacy_manifest_sha256": catalog_preparation["artifact_manifest_sha256"],
            "legacy_catalog_artifact_sha256": catalog_preparation[
                "artifact_catalog_sha256"
            ],
            "legacy_selective_provider_sha256": catalog_preparation[
                "selective_provider_sha256"
            ],
            "current_selective_source_sha256": selective_provider.source_sha256,
            "current_selective_provider_sha256": selective_provider.provider_sha256,
            "current_selective_catalog_sha256": (
                selective_provider.catalog.catalog_sha256
            ),
            "current_catalog_artifact": dict(provider_artifact_receipt),
            "one_authenticated_cache_session_reused": True,
            "cold_tensor_catalog_rebuild_performed": False,
            "parent_model_forward_calls": 0,
            "training_steps": 0,
        }
        provider_migration_receipt = {
            **provider_migration_body,
            "migration_sha256": canonical_sha256(provider_migration_body),
        }
        _write_json(
            run_root / "selective-catalog-provider-migration.json",
            provider_migration_receipt,
        )
        interface_solve = fit_lfm2_semantic_foundation_interface_from_provider(
            compiler_cache=compiler_cache,
            selective_provider=selective_provider,
            bridge=bridge,
            target_tokenizer=target_tokenizer,
            trajectory_corpus=trajectory_path,
            source_selected_tied_table=contextual_parent_rows,
            target_model=child,
        )
    finally:
        selective_provider.close()
    structural_install = install_lfm2_independent_structural_interface(
        target,
        interface_solve,
        donor_tied_table=parent_input,
        sequence_plan=sequence_plan,
        strict_anchor_ppm=500_000,
        preserve_anchor_norm=True,
    )
    structural_receipt = structural_install.as_dict()
    neutral_norms = dict(install_lfm2_cpcf_neutral_norms(child))
    _write_json(run_root / "structural-interface-install.json", structural_receipt)
    _write_json(run_root / "neutral-norm-install.json", neutral_norms)

    _emit("build-fresh-direct-local-attention-lineage")
    parent_inventory = inventory_lfm2_semantic_foundation_parent(
        parent_path,
        expected_parent_identity_sha256=expected_parent,
    )
    raw_lineage = build_lfm2_semantic_foundation_attention_lineage(
        parent_inventory=parent_inventory,
        target_model=child,
    )
    typed_lineage = normalize_lfm2_semantic_foundation_attention_lineage(
        raw_lineage,
        parent_inventory=parent_inventory,
        cache=semantic_cache,
        bridge=bridge,
        target_model=child,
    )
    _write_json(
        run_root / "attention-lineage.json",
        {
            "raw": dict(raw_lineage),
            "typed": {
                str(index): dict(value.report) for index, value in typed_lineage.items()
            },
        },
    )

    circuit_balanced_coordinate_evidence = None
    if coordinate_completion_recipe == LFM2_CIRCUIT_BALANCED_COORDINATE_RECIPE:
        _emit("build-cache-only-circuit-balanced-vocabulary-margin-evidence")
        circuit_balanced_coordinate_evidence = (
            build_lfm2_circuit_balanced_coordinate_evidence(
                parent,
                semantic_cache,
                int(child.config.hidden_size),
                foundation_interface=interface_solve.interface,
                tokenizer_bridge=bridge,
                circuit_graph=parent_graph,
            )
        )
        _write_json(
            run_root / "circuit-balanced-coordinate-evidence.json",
            circuit_balanced_coordinate_evidence.as_dict(),
        )

    _emit("inventory-parent-residual-facing-weight-geometry-once")
    if circuit_balanced_coordinate_evidence is None:
        weight_complete_inventory = inventory_lfm2_weight_complete_operators(parent)
    else:
        weight_complete_inventory = (
            circuit_balanced_coordinate_evidence.balanced_basis.inventory
        )
    weight_complete_manifest = weight_complete_inventory.as_dict()
    weight_complete_inventory_binding = {
        "inventory_sha256": weight_complete_manifest["inventory_sha256"],
        "operator_manifest_sha256": weight_complete_manifest["hashes"][
            "operator_manifest_sha256"
        ],
        "operator_count": weight_complete_manifest["operator_count"],
        "source_width": weight_complete_manifest["source_width"],
    }
    _emit("compile-parent-role-balanced-operator-geometry-once")
    parent_operator_geometry = compile_role_balanced_operator_geometry(
        weight_complete_inventory.operators,
        compute_device=fold_execution_profile,
    )
    parent_geometry_report = parent_operator_geometry.as_dict()
    parent_geometry_hashes = parent_geometry_report["hashes"]
    if (
        parent_geometry_report.get("execution_profile_sha256")
        != fold_execution_profile.result_sha256
        or parent_geometry_report.get("operator_count")
        != weight_complete_inventory_binding["operator_count"]
        or parent_geometry_report.get("source_width")
        != weight_complete_inventory_binding["source_width"]
        or len(parent_operator_geometry.ordered_operator_sha256s)
        != weight_complete_inventory_binding["operator_count"]
    ):
        raise Lfm2Exact4xCompositeFoldError(
            "Pre-aggregated parent geometry diverged from its single inventory."
        )
    parent_geometry_receipt_body = {
        "schema": BODY_GLOBAL_PARENT_GEOMETRY_SCHEMA,
        "status": "parent-operator-geometry-preaggregated-before-body-fold",
        "parent_identity_sha256": expected_parent,
        "weight_complete_inventory": weight_complete_inventory_binding,
        "parent_geometry_sha256": parent_operator_geometry.geometry_sha256,
        "parent_operator_sha256s": list(
            parent_operator_geometry.ordered_operator_sha256s
        ),
        "parent_raw_gram_sha256": parent_geometry_hashes["raw_gram_sha256"],
        "parent_balanced_gram_sha256": parent_geometry_hashes["balanced_gram_sha256"],
        "fold_execution_profile_sha256": fold_execution_profile.result_sha256,
        "analytic_execution_profile_sha256": (analytic_execution_profile.result_sha256),
        "fold_requested_device": fold_execution_profile.requested_device,
        "fold_resolved_device": fold_execution_profile.resolved_device,
        "analytic_requested_device": analytic_execution_profile.requested_device,
        "analytic_resolved_device": analytic_execution_profile.resolved_device,
        "cuda_required": True,
        "cpu_fallback_permitted": False,
        "parent_operator_inventory_count": 1,
        "parent_geometry_compile_count": 1,
        "parent_operator_reconstruction_after_geometry_compile": False,
        "model_forward_calls": 0,
        "trace_capture_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "autograd_used": False,
    }
    parent_geometry_receipt = {
        **parent_geometry_receipt_body,
        "receipt_sha256": canonical_sha256(parent_geometry_receipt_body),
    }
    del weight_complete_manifest
    del weight_complete_inventory
    gc.collect()
    _write_json(
        run_root / "body-global-parent-geometry.json",
        parent_geometry_receipt,
    )
    _emit("fold-residual-attention-conv-mlp-body")
    structural_attention_candidates = {}
    with _semantic_layer_stack_pre_auth_capture(run_root) as pre_auth_receipt_sink:
        layer_stack = compile_install_lfm2_semantic_layer_stack_stage(
            cache=semantic_cache,
            bridge=bridge,
            target_model=child,
            interface=interface_solve.interface,
            compiler_lineage=typed_lineage,
            parent_model_directory=parent_path,
            donor_tokenizer=donor_tokenizer,
            target_tokenizer=target_tokenizer,
            fold_compute_device=fold_compute_device,
            global_residual_closure_version=(
                requested_global_residual_closure_version
            ),
            correction_scale_frontier=(correction_scale,),
            correction_scale=correction_scale,
            global_residual_backend=global_residual_backend,
            global_residual_basis_rank=global_residual_basis_rank,
            target_dtype_representability_level=(
                target_dtype_representability_level
            ),
            normalized_input_correction_rho=normalized_input_correction_rho,
            **(
                {
                    "normalized_input_request_capture_directory": (
                        normalized_input_capture_root
                    )
                }
                if normalized_input_capture_root is not None
                else {}
            ),
            body_recipe=recipe,
            short_conv_construction=short_conv_construction,
            interface_runtime_mode="independent_output",
            coordinate_completion_parent_geometry=parent_operator_geometry,
            coordinate_completion_least_squares_execution_profile=(
                analytic_execution_profile
            ),
            coordinate_completion_circuit_balanced_evidence=(
                circuit_balanced_coordinate_evidence
            ),
            coordinate_completion_recipe=coordinate_completion_recipe,
            compiled_attention_candidate_sink=structural_attention_candidates,
            pre_auth_layer_stack_receipt_sink=pre_auth_receipt_sink,
            live_child_prefix_recapture_mode=(
                live_child_prefix_recapture_mode
            ),
        )
    if layer_stack.final_layer_mlp_down_weight_prior is None:
        raise Lfm2Exact4xCompositeFoldError(
            "Residual-aware body did not produce its deferred final W2 prior."
        )
    closure_selection = _global_residual_closure_selection_receipt(
        layer_stack.receipt,
        requested_closure_version=requested_global_residual_closure_version,
    )
    selected_global_residual_closure_version = closure_selection[
        "selected_closure_version"
    ]
    moving_gauge_finalization_plan = layer_stack.moving_gauge_finalization_plan
    if (moving_gauge_finalization_plan is not None) != bool(
        closure_selection["post_w2_physical_finalizer_required"]
    ):
        raise Lfm2Exact4xCompositeFoldError(
            "The selected residual engine lost its typed finalization plan."
        )
    _write_json(run_root / "semantic-layer-stack.json", layer_stack.receipt)
    _write_json(
        run_root / "global-residual-closure-selection.json",
        closure_selection,
    )
    deferred_final_w2_seed = (
        child.model.layers[-1]
        .feed_forward.w2.weight.detach()
        .to(device="cpu")
        .contiguous()
        .clone()
    )
    _emit("persist-pristine-structural-attention-candidates")
    structural_attention_candidate_archive = dict(
        write_lfm2_structural_attention_candidate_archive(
            run_root / "structural-attention-candidates",
            structural_attention_candidates,
            semantic_layer_stack_receipt=layer_stack.receipt,
        )
    )
    body_fold_realization = None
    if moving_gauge_finalization_plan is None:
        body_fold_realization = _body_fold_realization_receipt(
            layer_stack.receipt,
            child_dtype=child_dtype_name,
            requested_representability_level=normalized_representability_level,
        )
        _write_json(
            run_root / "body-fold-realization.json",
            body_fold_realization,
        )

    _emit("install-authenticated-deferred-final-w2")
    w2_install = dict(
        install_lfm2_deferred_final_w2_prior(
            child,
            layer_stack.final_layer_mlp_down_weight_prior,
            layer_stack_receipt=layer_stack.receipt,
        )
    )
    moving_gauge_finalization = None
    if moving_gauge_finalization_plan is not None:
        moving_gauge_finalization = dict(
            finalize_lfm2_moving_gauge_after_deferred_w2(
                child,
                moving_gauge_finalization_plan,
                layer_stack_receipt=layer_stack.receipt,
                deferred_final_w2_installation_receipt=w2_install,
            )
        )
    _write_json(run_root / "deferred-final-w2-install.json", w2_install)
    if moving_gauge_finalization is not None:
        _write_json(
            run_root / "moving-gauge-post-w2-finalization.json",
            moving_gauge_finalization,
        )
        body_fold_realization = _moving_gauge_body_fold_realization_receipt(
            layer_stack.receipt,
            closure_selection,
            moving_gauge_finalization,
            child_dtype=child_dtype_name,
            requested_representability_level=normalized_representability_level,
            dynamic_model_state_after_sha256=fingerprint_model_state(child),
            lfm2_identity_after_sha256=(
                fingerprint_lfm2_parent(child).identity_sha256
            ),
        )
        _write_json(
            run_root / "body-fold-realization.json",
            body_fold_realization,
        )
    if body_fold_realization is None:
        raise Lfm2Exact4xCompositeFoldError(
            "The finalized child body omitted its physical realization receipt."
        )
    _emit("persist-postfold-reclosure-seeds")
    reclosure_archive_state_before = fingerprint_model_state(child)
    reclosure_archive_parameter_count_before = _unique_parameter_count(child)
    postfold_reclosure_seed_archive = dict(
        write_lfm2_postfold_reclosure_seed_archive(
            run_root / "postfold-reclosure-seeds",
            mapped_parent_tied_table=contextual_parent_rows,
            producer_encoder=interface_solve.interface.producer_encoder,
            deferred_final_w2_seed=deferred_final_w2_seed,
            foundation_interface_solve_receipt=interface_solve.report,
            semantic_layer_stack_receipt=layer_stack.receipt,
            deferred_final_w2_installation_receipt=w2_install,
        )
    )
    reclosure_archive_state_after = fingerprint_model_state(child)
    reclosure_archive_parameter_count_after = _unique_parameter_count(child)
    if (
        reclosure_archive_state_after != reclosure_archive_state_before
        or reclosure_archive_parameter_count_after
        != reclosure_archive_parameter_count_before
        or postfold_reclosure_seed_archive.get("fresh_reload_verified") is not True
    ):
        raise Lfm2Exact4xCompositeFoldError(
            "Postfold reclosure seed publication changed the live child state."
        )
    reclosure_publication_body = {
        "schema": POSTFOLD_RECLOSURE_SEED_ARCHIVE_PUBLICATION_SCHEMA,
        "status": "postfold-reclosure-seeds-saved-and-freshly-reloaded",
        "archive_manifest_sha256": postfold_reclosure_seed_archive[
            "manifest_sha256"
        ],
        "foundation_interface_solve_result_sha256": interface_solve.report[
            "result_sha256"
        ],
        "semantic_layer_stack_result_sha256": layer_stack.receipt["result_sha256"],
        "deferred_final_w2_installation_result_sha256": w2_install["result_sha256"],
        "child_model_state_before_sha256": reclosure_archive_state_before,
        "child_model_state_after_sha256": reclosure_archive_state_after,
        "child_model_state_unchanged": True,
        "parameter_element_count_before": reclosure_archive_parameter_count_before,
        "parameter_element_count_after": reclosure_archive_parameter_count_after,
        "parameter_count_unchanged": True,
        "archive_fresh_reload_verified": True,
        "parent_model_forward_calls": 0,
        "child_model_forward_calls": 0,
        "parent_trace_recaptures": 0,
        "training_steps": 0,
    }
    postfold_reclosure_seed_publication = {
        **reclosure_publication_body,
        "receipt_sha256": canonical_sha256(reclosure_publication_body),
    }
    _write_json(
        run_root / "postfold-reclosure-seed-publication.json",
        postfold_reclosure_seed_publication,
    )
    del deferred_final_w2_seed
    body_state = fingerprint_model_state(child)
    body_identity_sha256 = fingerprint_lfm2_parent(child).identity_sha256
    body_inventory_sha256 = fingerprint_lfm2_causal_body_inventory(child)
    input_table_sha256 = structural_receipt["installed_input_table_sha256"]
    if child.get_output_embeddings().weight is child.get_input_embeddings().weight:
        raise Lfm2Exact4xCompositeFoldError(
            "Body transaction unexpectedly retied the token interface."
        )

    structural_body_receipt_body = {
        "schema": STRUCTURAL_BODY_RECEIPT_SCHEMA,
        "parent_identity_sha256": expected_parent,
        "target_unique_parameters": target_unique_parameters,
        "target_topology": topology,
        "child_parameter_dtype": child_dtype_name,
        "body_fold_realization_sha256": body_fold_realization["receipt_sha256"],
        "global_residual_closure_selection_sha256": closure_selection[
            "receipt_sha256"
        ],
        "global_residual_requested_closure_version": (
            requested_global_residual_closure_version
        ),
        "global_residual_selected_closure_version": (
            selected_global_residual_closure_version
        ),
        "structural_interface_install_sha256": structural_receipt["result_sha256"],
        "semantic_layer_stack_sha256": layer_stack.receipt["result_sha256"],
        "structural_attention_candidate_archive_manifest_sha256": (
            structural_attention_candidate_archive["manifest_sha256"]
        ),
        "structural_attention_candidate_archive_fresh_reload_verified": (
            structural_attention_candidate_archive["fresh_reload_verified"]
        ),
        "postfold_reclosure_seed_archive_manifest_sha256": (
            postfold_reclosure_seed_archive["manifest_sha256"]
        ),
        "postfold_reclosure_seed_archive_fresh_reload_verified": (
            postfold_reclosure_seed_archive["fresh_reload_verified"]
        ),
        "postfold_reclosure_seed_publication_sha256": (
            postfold_reclosure_seed_publication["receipt_sha256"]
        ),
        "postfold_reclosure_seed_archive_model_state_unchanged": True,
        "deferred_final_w2_install_sha256": w2_install["result_sha256"],
        **(
            {
                "moving_gauge_post_w2_finalization_sha256": (
                    moving_gauge_finalization["result_sha256"]
                ),
                "moving_gauge_finalization_preceded_body_fingerprint": True,
                "causal_head_solve_required_after_moving_gauge_finalization": (
                    True
                ),
            }
            if moving_gauge_finalization is not None
            else {}
        ),
        "body_model_state_sha256": body_state,
        "body_lfm2_identity_sha256": body_identity_sha256,
        "causal_body_inventory_sha256": body_inventory_sha256,
        "installed_input_table_sha256": input_table_sha256,
        "installed_output_baseline_sha256": structural_receipt[
            "installed_output_baseline_sha256"
        ],
        "input_and_output_parameters_distinct": True,
        "input_and_output_values_equal": True,
        "causal_output_head_installed": False,
        "operation_audit": {
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "child_forward_calls": 0,
        },
    }
    if native_target_contract is not None:
        structural_body_receipt_body.update(
            {
                "native_target_class_id": LFM2_NATIVE_EXACT4X_TARGET_CLASS_ID,
                "native_target_contract_sha256": (
                    native_target_contract.fingerprint
                ),
                "component_allocation_receipt_sha256": (
                    native_target_contract.component_allocation_receipt_sha256
                ),
                "component_allocation_topology_sha256": (
                    native_target_contract.component_allocation_topology_sha256
                ),
            }
        )
    checkpoint_io = Lfm2DirectLocalCheckpointIO(
        tokenizer_bridge=bridge,
        device=torch.device(model_device),
        dtype=child_torch_dtype,
    )
    if defer_publication_to_caller:
        _emit("defer-structural-body-publication-to-outer-compiler")
        structural_body_receipt_body.update(
            {
                "status": "in-place-structural-body-awaiting-outer-publication",
                "checkpoint_path": None,
                "checkpoint_inventory": [],
                "checkpoint_inventory_sha256": canonical_sha256([]),
                "checkpoint_reload_dtype": None,
                "evaluation_dtype": None,
                "standalone_fresh_reload_verified": False,
                "publication_deferred_to_outer_compiler": True,
                "verification_reload_materializations": 0,
            }
        )
    else:
        _emit("save-and-reload-structural-body-base")
        base_checkpoint = run_root / "structural-body-checkpoint"
        base_checkpoint.mkdir()
        checkpoint_io.save(child, base_checkpoint)
        base_reload = checkpoint_io.load(base_checkpoint)
        if (
            fingerprint_model_state(base_reload) != body_state
            or fingerprint_lfm2_parent(base_reload).identity_sha256
            != body_identity_sha256
            or _unique_parameter_count(base_reload)
            != target_unique_parameters
            or _model_parameter_dtype_name(base_reload) != child_dtype_name
            or base_reload.get_input_embeddings().weight
            is base_reload.get_output_embeddings().weight
            or not torch.equal(
                base_reload.get_input_embeddings().weight.detach(),
                base_reload.get_output_embeddings().weight.detach(),
            )
            or fingerprint_lfm2_causal_body_inventory(base_reload)
            != body_inventory_sha256
        ):
            raise Lfm2Exact4xCompositeFoldError(
                "Fresh structural-body reload changed the exact child coordinate."
            )
        base_checkpoint_inventory = _artifact_inventory(base_checkpoint)
        structural_body_receipt_body.update(
            {
                "status": "standalone-structural-body-saved-and-freshly-reloaded",
                "checkpoint_path": str(base_checkpoint),
                "checkpoint_inventory": base_checkpoint_inventory,
                "checkpoint_inventory_sha256": canonical_sha256(
                    base_checkpoint_inventory
                ),
                "checkpoint_reload_dtype": child_dtype_name,
                "evaluation_dtype": child_dtype_name,
                "standalone_fresh_reload_verified": True,
                "publication_deferred_to_outer_compiler": False,
                "verification_reload_materializations": 1,
            }
        )
        del base_reload
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    structural_body_receipt = {
        **structural_body_receipt_body,
        "receipt_sha256": canonical_sha256(structural_body_receipt_body),
    }
    _write_json(
        run_root / "structural-body-receipt.json",
        structural_body_receipt,
    )

    pre_causal_body_resume_reference_receipt = None
    pre_causal_capsule_verification_reload_materializations = 0
    if defer_publication_to_caller:
        if native_target_contract is None or moving_gauge_finalization is None:
            raise Lfm2Exact4xCompositeFoldError(
                "Deferred exact-4x production must expose its native target and "
                "post-moving-gauge coordinate before causal compilation."
            )
        _emit("persist-authenticated-post-body-pre-causal-resume")
        target_state_inventory = dict(
            serialized_inference_state_inventory(child)
        )
        pre_causal_bindings = Lfm2PreCausalBodyResumeBindings(
            child_name=target.graph.model_name,
            target_unique_parameters=target_unique_parameters,
            target_logical_parameters=target.graph.logical_parameter_count,
            parent_identity_sha256=expected_parent,
            target_synthesis_fingerprint=target.fingerprint,
            target_topology_fingerprint=target.topology_fingerprint,
            target_configuration_sha256=target.graph.config_fingerprint,
            target_state_inventory_sha256=canonical_sha256(
                target_state_inventory
            ),
            native_target_contract_sha256=native_target_contract.fingerprint,
            child_graph_fingerprint=target.graph.fingerprint,
            child_structural_topology_fingerprint=(
                target.graph.structural_topology_fingerprint
            ),
            bridge_artifact_sha256=bridge.artifact_sha256,
            bridge_profile_sha256=bridge.profile_fingerprint,
            target_tokenizer_semantics_sha256=(
                bridge.target_tokenizer_semantics_sha256
            ),
            target_tokenizer_package_fingerprint=(
                strict_tokenizer_bridge_target_package_fingerprint(bridge)
            ),
            body_fold_realization_sha256=body_fold_realization[
                "receipt_sha256"
            ],
            global_residual_closure_selection_sha256=closure_selection[
                "receipt_sha256"
            ],
            structural_interface_install_sha256=structural_receipt[
                "result_sha256"
            ],
            semantic_layer_stack_sha256=layer_stack.receipt[
                "result_sha256"
            ],
            structural_attention_candidate_archive_manifest_sha256=(
                structural_attention_candidate_archive["manifest_sha256"]
            ),
            postfold_reclosure_seed_archive_manifest_sha256=(
                postfold_reclosure_seed_archive["manifest_sha256"]
            ),
            postfold_reclosure_seed_publication_sha256=(
                postfold_reclosure_seed_publication["receipt_sha256"]
            ),
            deferred_final_w2_install_sha256=w2_install["result_sha256"],
            moving_gauge_post_w2_finalization_sha256=(
                moving_gauge_finalization["result_sha256"]
            ),
            body_model_state_sha256=body_state,
            body_runtime_state_sha256=fingerprint_model_runtime_state(child),
            body_lfm2_identity_sha256=body_identity_sha256,
            causal_body_inventory_sha256=body_inventory_sha256,
            installed_input_table_sha256=input_table_sha256,
            installed_output_baseline_sha256=structural_receipt[
                "installed_output_baseline_sha256"
            ],
        )
        pre_causal_reference = publish_lfm2_pre_causal_body_resume_capsule(
            model=child,
            checkpoint_io=checkpoint_io,
            bindings=pre_causal_bindings,
            output_directory=run_root / "pre-causal-body-resume",
        )
        pre_causal_capsule_verification_reload_materializations = 1
        reference_body = {
            "schema": "neural-forge-lfm2-pre-causal-body-resume-reference-v1",
            "status": "authenticated-nonpromotable-pre-causal-resume-retained",
            "reference": pre_causal_reference.as_dict(),
            "bindings_sha256": pre_causal_bindings.bindings_sha256,
            "body_model_state_sha256": body_state,
            "causal_body_inventory_sha256": body_inventory_sha256,
            "causal_output_head_installed": False,
            "parent_sidecar_parameters": 0,
            "training_steps": 0,
        }
        pre_causal_body_resume_reference_receipt = {
            **reference_body,
            "reference_sha256": canonical_sha256(reference_body),
        }
        _write_json(
            run_root / "pre-causal-body-resume-reference.json",
            pre_causal_body_resume_reference_receipt,
        )

    _emit("replay-final-child-coordinate-and-solve-causal-output")
    child_evidence = collect_lfm2_child_causal_residual_evidence(
        child,
        causal_trie,
        expected_input_table_sha256=input_table_sha256,
        expected_body_inventory_sha256=body_inventory_sha256,
        device=model_device,
        evidence_role="exact4x-final-body-before-independent-causal-head",
    )
    child_evidence_receipt = child_evidence.as_dict()
    _write_json(
        run_root / "child-causal-residual-evidence.json",
        child_evidence_receipt,
    )
    feature_rows = child_evidence.hidden_rows.to(dtype=torch.float32)
    gram_diagonal_mean = float(feature_rows.square().sum(dim=1).mean())
    if not gram_diagonal_mean > 0.0:
        raise Lfm2Exact4xCompositeFoldError(
            "Final child causal features have zero analytic ridge scale."
        )
    target_special_ids = tuple(
        sorted(
            {
                int(value)
                for value in getattr(target_tokenizer, "all_special_ids", ())
                if 0 <= int(value) < int(topology["vocab_size"])
            }
        )
    )
    structural_table = (
        child.get_input_embeddings()
        .weight.detach()
        .to(device="cpu")
        .contiguous()
        .clone()
    )
    causal_bridge = compile_lfm2_causal_residual_token_bridge(
        structural_table,
        child_evidence,
        causal_trie,
        parent_terminal_evidence=terminal_receipt,
        sequence_plan_sha256=sequence_plan["plan_sha256"],
        ridge=gram_diagonal_mean * float(CAUSAL_HEAD_RECIPE["ridge_fraction"]),
        residual_scale_ppm=int(CAUSAL_HEAD_RECIPE["residual_scale_ppm"]),
        protected_target_ids=(
            target_special_ids if bool(CAUSAL_HEAD_RECIPE["protect_specials"]) else ()
        ),
        row_norm_calibration_ppm=int(CAUSAL_HEAD_RECIPE["row_norm_calibration_ppm"]),
        solve_device=analytic_solve_device,
        unsupported_logit_policy=(
            LFM2_UNSUPPORTED_LOGIT_POLICY_ADAPTIVE_STRONGEST_SURVIVING_BACKGROUND_SEAL
            if coordinate_completion_recipe
            == LFM2_CIRCUIT_BALANCED_COORDINATE_RECIPE
            else LFM2_UNSUPPORTED_LOGIT_POLICY_MEASURE_DRIFT
        ),
    )
    causal_bridge_receipt = causal_bridge.as_dict()
    _write_json(
        run_root / "causal-output-bridge-pre-install.json",
        causal_bridge_receipt,
    )
    if coordinate_completion_recipe == LFM2_CIRCUIT_BALANCED_COORDINATE_RECIPE:
        if circuit_balanced_coordinate_evidence is None:
            raise Lfm2Exact4xCompositeFoldError(
                "Circuit-balanced recipe lost its full-vocabulary evidence."
            )
        coordinate_evidence_receipt = (
            circuit_balanced_coordinate_evidence.as_dict()
        )
        if (
            causal_bridge_receipt.get("unsupported_logit_policy")
            != LFM2_UNSUPPORTED_LOGIT_POLICY_ADAPTIVE_STRONGEST_SURVIVING_BACKGROUND_SEAL
            or causal_bridge_receipt.get("vocabulary_row_coverage_complete")
            is not True
            or causal_bridge_receipt.get(
                "all_vocabulary_rows_have_structural_or_contextual_authority"
            )
            is not True
            or causal_bridge_receipt.get(
                "fully_unsupported_output_rows_equal_structural_baseline"
            )
            is not True
            or causal_bridge_receipt.get(
                "fully_unsupported_output_rows_copied_from_structural_baseline"
            )
            is not True
            or causal_bridge_receipt.get(
                "fully_unsupported_output_row_logit_drift_max_abs"
            )
            != 0.0
            or causal_bridge_receipt.get("adaptive_background_policy_selected")
            is not True
            or causal_bridge_receipt.get(
                "target_dtype_candidate_accounting_complete"
            )
            is not True
            or causal_bridge_receipt.get(
                "adaptive_background_retained_every_nonzero_candidate"
            )
            is not True
            or causal_bridge_receipt.get(
                "target_dtype_background_reverted_output_row_count"
            )
            != 0
            or causal_bridge_receipt.get("target_row_count")
            != bridge.target_vocabulary_size
            or not _causal_structural_install_foundation_lineage_valid(
                causal_bridge_receipt,
                structural_receipt,
                coordinate_evidence_receipt,
                target_vocabulary_size=bridge.target_vocabulary_size,
            )
        ):
            raise Lfm2Exact4xCompositeFoldError(
                "Causal output is not a sealed residual over the complete "
                "semantic-foundation vocabulary table."
            )
    causal_install = install_lfm2_independent_causal_output_head(
        structural_install,
        causal_bridge,
        child_evidence,
        expected_input_table_sha256=input_table_sha256,
        expected_body_inventory_sha256=body_inventory_sha256,
    )
    causal_install_receipt = causal_install.as_dict()
    _write_json(run_root / "causal-output-install.json", causal_install_receipt)

    candidate_id = str(CAUSAL_HEAD_RECIPE["candidate_id"])
    installed_state = fingerprint_model_state(child)
    installed_lfm2_fingerprint = fingerprint_lfm2_parent(child)
    blank_target_receipt = target.as_dict()
    interface_solve_receipt = dict(interface_solve.report)
    layer_stack_receipt = dict(layer_stack.receipt)
    if fingerprint_model_state(parent) != parent_state_before:
        raise Lfm2Exact4xCompositeFoldError(
            "Parent changed during the no-execution composite fold."
        )
    if defer_publication_to_caller:
        _emit("defer-complete-child-publication-to-outer-compiler")
        candidate_checkpoint = None
        candidate_inventory: list[dict[str, Any]] = []
        frontier_path = None
        dossier_path = None
        frontier_sha256 = None
        dossier_sha256 = None
        standalone_fresh_reload_verified = False
        decoded_whole_responses_retained = False
        complete_child_verification_reload_materializations = 0
    else:
        _emit("save-reload-and-decode-complete-child")
        if checkpoint_io is None:
            raise Lfm2Exact4xCompositeFoldError(
                "Standalone publication lost its checkpoint implementation."
            )
        candidate_checkpoint = run_root / "candidates" / candidate_id / "checkpoint"
        candidate_checkpoint.mkdir(parents=True)
        checkpoint_io.save(child, candidate_checkpoint)
        reloaded = checkpoint_io.load(candidate_checkpoint)
        reload_state = fingerprint_model_state(reloaded)
        reload_lfm2_fingerprint = fingerprint_lfm2_parent(reloaded)
        if (
            reload_state != installed_state
            or reload_lfm2_fingerprint.as_dict()
            != installed_lfm2_fingerprint.as_dict()
            or _unique_parameter_count(reloaded)
            != target_unique_parameters
            or _model_parameter_dtype_name(reloaded) != child_dtype_name
            or reloaded.get_input_embeddings().weight
            is reloaded.get_output_embeddings().weight
            or not _parameter_storage_addresses(reloaded).isdisjoint(parent_storage)
            or fingerprint_lfm2_causal_body_inventory(reloaded)
            != body_inventory_sha256
        ):
            raise Lfm2Exact4xCompositeFoldError(
                "Fresh complete-child reload changed topology, body, or state bytes."
            )
        del reloaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        candidate_inventory = _artifact_inventory(candidate_checkpoint)
        frontier = _frontier_receipt(
            run_directory=run_root,
            candidate_id=candidate_id,
            checkpoint=candidate_checkpoint,
            model_state_sha256=installed_state,
            checkpoint_inventory=candidate_inventory,
            lfm2_identity_sha256=reload_lfm2_fingerprint.identity_sha256,
            lfm2_config_sha256=reload_lfm2_fingerprint.config_sha256,
            materialization_dtype=child_dtype_name,
        )
        frontier_path = run_root / "frontier-run.json"
        _write_json(frontier_path, frontier)
        dossier_path = run_root / "whole-response-dossier.json"
        dossier = review_frontier(
            frontier_run_receipt=frontier_path,
            expected_frontier_run_sha256=frontier["run_sha256"],
            expected_frontier_run_file_sha256=_file_sha256(frontier_path),
            prompts_json=prompts_path,
            output_json=dossier_path,
            candidate_ids=(candidate_id,),
            device=model_device,
            dtype=child_dtype_name,
            maximum_prompt_tokens=maximum_prompt_tokens,
            maximum_new_tokens=maximum_new_tokens,
            parent_envelope=envelope_path,
            expected_parent_envelope_sha256=expected_envelope,
            expected_parent_envelope_file_sha256=expected_envelope_file,
        )
        reviews = dossier.get("candidate_reviews")
        if (
            dossier.get("reviewed_candidate_count") != 1
            or not isinstance(reviews, list)
            or len(reviews) != 1
            or reviews[0].get("candidate_id") != candidate_id
            or len(reviews[0].get("responses", [])) != len(prompt_rows)
        ):
            raise Lfm2Exact4xCompositeFoldError(
                "Whole-response dossier did not retain every decoded response."
            )
        frontier_sha256 = frontier["run_sha256"]
        dossier_sha256 = dossier["dossier_sha256"]
        standalone_fresh_reload_verified = True
        decoded_whole_responses_retained = True
        complete_child_verification_reload_materializations = 2
    verification_reload_materializations = (
        pre_causal_capsule_verification_reload_materializations
        + complete_child_verification_reload_materializations
    )
    del (
        causal_install,
        causal_bridge,
        child_evidence,
        layer_stack,
        structural_install,
        interface_solve,
        contextual_parent_rows,
        child,
        target,
        parent,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    body = {
        "schema": RUN_SCHEMA,
        "status": (
            "in-place-exact4x-composite-child-installed"
            if defer_publication_to_caller
            else "standalone-exact4x-composite-child-decoded"
        ),
        "run_directory": str(run_root),
        "parent_model_directory": str(parent_path),
        "parent_identity_sha256": expected_parent,
        "parent_unique_parameters": EXPECTED_PARENT_UNIQUE_PARAMETERS,
        "target_unique_parameters": target_unique_parameters,
        "realized_parameter_compaction_ratio": (
            EXPECTED_PARENT_UNIQUE_PARAMETERS / target_unique_parameters
        ),
        "target_topology": topology,
        "child_parameter_dtype": child_dtype_name,
        "checkpoint_reload_dtype": (
            None if defer_publication_to_caller else child_dtype_name
        ),
        "whole_response_evaluation_dtype": (
            None if defer_publication_to_caller else child_dtype_name
        ),
        "float32_storage_is_not_quantization": child_dtype_name == "float32",
        "target_is_dense_untied_blank_architecture": True,
        "existing_compiler_blank_used": existing_compiler_blank_used,
        "publication_deferred_to_outer_compiler": defer_publication_to_caller,
        "blank_target": blank_target_receipt,
        "blank_model_state_sha256": blank_state,
        "compiler_cache_directory": str(compiler_cache_path),
        "compiler_cache_key": compiler_cache.cache_key,
        "semantic_cache_directory": str(semantic_cache_path),
        "semantic_cache_key": semantic_cache.cache_key,
        "sealed_parent_cache_reused_without_recapture": True,
        "preopened_semantic_resources_reused": (
            preopened_semantic_resources_used
        ),
        "bridge_directory": str(bridge_path),
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "sequence_plan_sha256": sequence_plan["plan_sha256"],
        "terminal_cache_evidence_sha256": terminal_receipt["evidence_sha256"],
        "causal_trie_evidence_sha256": trie_receipt["result_sha256"],
        "selective_catalog_provider_migration": provider_migration_receipt,
        "foundation_interface_solve": interface_solve_receipt,
        "structural_interface_install": structural_receipt,
        "neutral_norm_install": neutral_norms,
        "attention_lineage_sha256": canonical_sha256(dict(raw_lineage)),
        "body_recipe": recipe.as_dict(),
        "body_recipe_sha256": recipe.recipe_sha256,
        "global_residual_closure_version": (
            selected_global_residual_closure_version
        ),
        "global_residual_requested_closure_version": (
            requested_global_residual_closure_version
        ),
        "global_residual_selected_closure_version": (
            selected_global_residual_closure_version
        ),
        "global_residual_closure_selection": closure_selection,
        "short_conv_construction": short_conv_construction,
        "body_global_coordinate_completion_recipe": (coordinate_completion_recipe),
        "circuit_balanced_coordinate_evidence": (
            None
            if circuit_balanced_coordinate_evidence is None
            else circuit_balanced_coordinate_evidence.as_dict()
        ),
        "circuit_balanced_coordinate_evidence_sha256": (
            None
            if circuit_balanced_coordinate_evidence is None
            else circuit_balanced_coordinate_evidence.result_sha256
        ),
        "circuit_balanced_coordinate_recipe_selected": (
            coordinate_completion_recipe
            == LFM2_CIRCUIT_BALANCED_COORDINATE_RECIPE
        ),
        "foundation_structural_vocabulary_table_preserved_before_causal_residual": (
            circuit_balanced_coordinate_evidence is not None
        ),
        "causal_trie_authority": (
            "residual-only-over-full-structural-foundation"
            if circuit_balanced_coordinate_evidence is not None
            else "legacy-configured"
        ),
        "normalized_input_correction_rho": normalized_input_correction_rho,
        "live_child_prefix_recapture_mode": live_child_prefix_recapture_mode,
        "live_child_prefix_recapture": {
            "body_stack_mode": layer_stack_receipt[
                "live_child_prefix_recapture_mode"
            ],
            "body_stack_recapture_count": layer_stack_receipt[
                "live_child_full_prefix_recapture_count"
            ],
            "body_stack_recaptures": layer_stack_receipt[
                "live_child_full_prefix_recaptures"
            ],
            "output_interface_recap_after_final_body_install": (
                child_evidence_receipt
            ),
            "output_interface_recap_uses_public_child_full_prefix": (
                child_evidence_receipt["full_prefix_use_cache_false"]
            ),
            "output_interface_recap_child_forward_calls": (
                child_evidence_receipt["operation_audit"]["child_forward_calls"]
            ),
            "output_interface_recap_parent_forward_calls": 0,
            "output_interface_recap_parent_trace_recaptures": 0,
            "output_interface_recap_parent_cache_regenerations": 0,
            "output_interface_recap_training_steps": 0,
            "output_interface_recap_supplies_causal_output_head_fit": True,
        },
        "body_global_parent_geometry_receipt": parent_geometry_receipt,
        "body_global_parent_geometry_receipt_sha256": (
            parent_geometry_receipt["receipt_sha256"]
        ),
        "body_global_parent_geometry_sha256": (
            parent_operator_geometry.geometry_sha256
        ),
        "fold_execution_profile": fold_execution_profile.as_dict(),
        "fold_execution_profile_sha256": fold_execution_profile.result_sha256,
        "analytic_execution_profile": analytic_execution_profile.as_dict(),
        "analytic_execution_profile_sha256": (analytic_execution_profile.result_sha256),
        "semantic_layer_stack": layer_stack_receipt,
        "structural_attention_candidate_archive": (
            structural_attention_candidate_archive
        ),
        "postfold_reclosure_seed_archive_manifest_sha256": (
            postfold_reclosure_seed_archive["manifest_sha256"]
        ),
        "postfold_reclosure_seed_archive": postfold_reclosure_seed_archive,
        "postfold_reclosure_seed_publication": (
            postfold_reclosure_seed_publication
        ),
        "body_fold_realization": body_fold_realization,
        "deferred_final_w2_install": w2_install,
        "moving_gauge_post_w2_finalization": moving_gauge_finalization,
        "structural_body_model_state_sha256": body_state,
        "structural_body_lfm2_identity_sha256": body_identity_sha256,
        "causal_body_inventory_sha256": body_inventory_sha256,
        "structural_body_checkpoint_receipt": structural_body_receipt,
        "pre_causal_body_resume_reference": (
            pre_causal_body_resume_reference_receipt
        ),
        "child_causal_residual_evidence_sha256": child_evidence_receipt[
            "evidence_sha256"
        ],
        "causal_head_recipe": dict(CAUSAL_HEAD_RECIPE),
        "causal_token_bridge": causal_bridge_receipt,
        "causal_output_install": causal_install_receipt,
        "moving_gauge_finalization_preceded_causal_head_install": (
            moving_gauge_finalization is not None
        ),
        "finalized_body_to_causal_head_lineage": {
            "moving_gauge_post_w2_finalization_sha256": (
                None
                if moving_gauge_finalization is None
                else moving_gauge_finalization["result_sha256"]
            ),
            "child_causal_residual_evidence_sha256": child_evidence_receipt[
                "evidence_sha256"
            ],
            "causal_output_install_sha256": causal_install_receipt[
                "result_sha256"
            ],
            "causal_head_installed_after_finalized_body": True,
        },
        "complete_child_model_state_sha256": installed_state,
        "candidate_id": candidate_id,
        "candidate_checkpoint": (
            None if candidate_checkpoint is None else str(candidate_checkpoint)
        ),
        "candidate_checkpoint_inventory": candidate_inventory,
        "candidate_checkpoint_inventory_sha256": canonical_sha256(candidate_inventory),
        "standalone_fresh_reload_verified": standalone_fresh_reload_verified,
        "frontier_run_receipt": (
            None if frontier_path is None else str(frontier_path)
        ),
        "frontier_run_sha256": frontier_sha256,
        "whole_response_dossier": (
            None if dossier_path is None else str(dossier_path)
        ),
        "whole_response_dossier_sha256": dossier_sha256,
        "prompt_set_sha256": prompt_set_sha256,
        "prompt_count": len(prompt_rows),
        "maximum_prompt_tokens": maximum_prompt_tokens,
        "maximum_new_tokens": maximum_new_tokens,
        "behavior_evidence_policy": {
            "decoded_whole_responses_are_primary_evidence": True,
            "decoded_whole_responses_retained_in_this_stage": (
                decoded_whole_responses_retained
            ),
            "decoded_review_deferred_to_outer_publication_gate": (
                defer_publication_to_caller
            ),
            "diagnostic_numbers_may_guide_navigation": True,
            "diagnostic_numbers_have_decision_authority": False,
            "automatic_behavior_acceptance_or_rejection_performed": False,
            "automatic_behavior_verdict_performed": False,
            "manual_intelligence_review_required": True,
            "binary_manual_verdict_required": False,
            "ninety_percent_intelligence_retention_is_objective_not_cutoff": True,
        },
        "operation_audit": {
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "parent_model_loads_inside_kernel": parent_model_loads_inside_kernel,
            "target_model_materializations_inside_kernel": (
                target_model_materializations_inside_kernel
            ),
            "verification_reload_materializations": (
                verification_reload_materializations
            ),
            "pre_causal_capsule_verification_reload_materializations": (
                pre_causal_capsule_verification_reload_materializations
            ),
            "complete_child_verification_reload_materializations": (
                complete_child_verification_reload_materializations
            ),
            "compiler_cache_authenticated_sessions": 1,
            "compiler_cache_cold_catalog_tensor_sweeps": 0,
            "compiler_cache_reopens_during_interface_fit": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "parameters_sparsified": 0,
            "external_adapter_parameters": 0,
            "parent_sidecar_parameters": 0,
            "runtime_shared_parent_parameters": 0,
            "network_access": False,
        },
    }
    if native_target_contract is not None:
        body.update(
            {
                "native_target_class_id": LFM2_NATIVE_EXACT4X_TARGET_CLASS_ID,
                "native_target_contract_sha256": (
                    native_target_contract.fingerprint
                ),
                "native_target_contract": native_target_contract.as_dict(),
                "component_allocation_receipt_sha256": (
                    native_target_contract.component_allocation_receipt_sha256
                ),
                "component_allocation_topology_sha256": (
                    native_target_contract.component_allocation_topology_sha256
                ),
            }
        )
    result = {**body, "result_sha256": canonical_sha256(body)}
    _write_json(run_root / "run-receipt.json", result)
    _emit(
        "complete",
        result_sha256=result["result_sha256"],
        dossier_sha256=dossier_sha256,
    )
    return result


def run_existing_target(
    *,
    target: DynamicTargetSynthesisResult,
    parent_model: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the exact-4x kernel in one already-synthesized compiler blank.

    Complete-child publication and decoded review remain owned by the outer
    automatic compiler.  This path creates no target blank, but it does retain
    and freshly reload exactly one non-promotable pre-causal body capsule.
    """

    if "existing_target" in kwargs or "existing_parent_model" in kwargs:
        raise Lfm2Exact4xCompositeFoldError(
            "Existing target bindings must use the typed wrapper arguments."
        )
    if "defer_publication_to_caller" in kwargs:
        raise Lfm2Exact4xCompositeFoldError(
            "run_existing_target owns the outer-publication boundary."
        )
    return run(
        existing_target=target,
        existing_parent_model=parent_model,
        defer_publication_to_caller=True,
        **kwargs,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--parent-model-directory", required=True, type=Path)
    parser.add_argument("--donor-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--target-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--bridge-directory", required=True, type=Path)
    parser.add_argument("--expected-bridge-artifact-sha256", required=True)
    parser.add_argument("--compiler-cache-directory", required=True, type=Path)
    parser.add_argument("--expected-compiler-cache-key", required=True)
    parser.add_argument("--semantic-cache-directory", required=True, type=Path)
    parser.add_argument("--expected-semantic-cache-key", required=True)
    parser.add_argument("--trajectory-corpus", required=True, type=Path)
    parser.add_argument(
        "--selective-catalog-store-directory",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--selective-catalog-receipt-json",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--expected-selective-catalog-receipt-sha256",
        required=True,
    )
    parser.add_argument("--expected-parent-identity-sha256", required=True)
    parser.add_argument(
        "--expected-parent-unique-parameters",
        type=int,
        default=EXPECTED_PARENT_UNIQUE_PARAMETERS,
    )
    parser.add_argument("--prompts-json", required=True, type=Path)
    parser.add_argument("--expected-prompts-file-sha256", required=True)
    parser.add_argument("--parent-envelope", required=True, type=Path)
    parser.add_argument("--expected-parent-envelope-sha256", required=True)
    parser.add_argument(
        "--expected-parent-envelope-file-sha256",
        required=True,
    )
    parser.add_argument(
        "--body-recipe",
        default=(
            "nonlinear_qdeim_coupled_attention_source_local_residual_fit_corrected"
        ),
    )
    parser.add_argument("--expected-body-recipe-sha256", required=True)
    parser.add_argument(
        "--short-conv-construction",
        choices=LFM2_SEMANTIC_SHORT_CONV_CONSTRUCTION_IDS,
        default=(
            LFM2_SEMANTIC_SHORT_CONV_COORDINATE_COMPLETE_FHVR_FULL_COVERAGE_CONSTRUCTION
        ),
    )
    parser.add_argument(
        "--coordinate-completion-recipe",
        choices=LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_RECIPES,
        default=LFM2_CIRCUIT_BALANCED_COORDINATE_RECIPE,
    )
    parser.add_argument("--blank-seed", type=int, default=0)
    parser.add_argument("--model-device", default="cuda:0")
    parser.add_argument("--fold-compute-device", default="cuda:0")
    parser.add_argument("--analytic-solve-device", default="cuda:0")
    parser.add_argument("--global-residual-backend", default="cuda")
    parser.add_argument(
        "--global-residual-closure-version",
        choices=LFM2_GLOBAL_RESIDUAL_CLOSURE_REQUESTS,
        default=LFM2_GLOBAL_RESIDUAL_CLOSURE_AUTO,
    )
    parser.add_argument("--global-residual-basis-rank", type=int, default=128)
    parser.add_argument("--correction-scale", type=float, default=1.0)
    parser.add_argument(
        "--normalized-input-correction-rho",
        type=float,
        default=0.25,
        help=(
            "Frobenius cap on each normalized MLP FIT-rowspace correction "
            "relative to the authenticated structural prior."
        ),
    )
    parser.add_argument(
        "--normalized-input-request-capture-directory",
        type=Path,
        help=(
            "Optional new local root for one authenticated CPU-replay capsule "
            "per normalized MLP compiler request."
        ),
    )
    parser.add_argument(
        "--live-child-prefix-recapture-mode",
        choices=LFM2_LIVE_CHILD_PREFIX_RECAPTURE_MODES,
        default=LFM2_LIVE_CHILD_PREFIX_RECAPTURE_DISABLED,
        help=(
            "Opt in to a current-child-only, full-depth, full-prefix analytic "
            "recapture after every effectful body component install."
        ),
    )
    parser.add_argument(
        "--child-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help=(
            "Dense child/checkpoint/evaluation dtype. The FP32 path must use "
            "--target-dtype-representability-level requested."
        ),
    )
    parser.add_argument(
        "--target-dtype-representability-level",
        default="first_effectful",
    )
    parser.add_argument("--maximum-prompt-tokens", type=int, default=1024)
    parser.add_argument("--maximum-new-tokens", type=int, default=256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run(
            run_directory=args.run_directory,
            parent_model_directory=args.parent_model_directory,
            donor_tokenizer_directory=args.donor_tokenizer_directory,
            target_tokenizer_directory=args.target_tokenizer_directory,
            bridge_directory=args.bridge_directory,
            expected_bridge_artifact_sha256=(args.expected_bridge_artifact_sha256),
            compiler_cache_directory=args.compiler_cache_directory,
            expected_compiler_cache_key=args.expected_compiler_cache_key,
            semantic_cache_directory=args.semantic_cache_directory,
            expected_semantic_cache_key=args.expected_semantic_cache_key,
            trajectory_corpus=args.trajectory_corpus,
            selective_catalog_store_directory=(args.selective_catalog_store_directory),
            selective_catalog_receipt_json=args.selective_catalog_receipt_json,
            expected_selective_catalog_receipt_sha256=(
                args.expected_selective_catalog_receipt_sha256
            ),
            expected_parent_identity_sha256=(args.expected_parent_identity_sha256),
            expected_parent_unique_parameters=(args.expected_parent_unique_parameters),
            prompts_json=args.prompts_json,
            expected_prompts_file_sha256=args.expected_prompts_file_sha256,
            parent_envelope=args.parent_envelope,
            expected_parent_envelope_sha256=(args.expected_parent_envelope_sha256),
            expected_parent_envelope_file_sha256=(
                args.expected_parent_envelope_file_sha256
            ),
            body_recipe=args.body_recipe,
            expected_body_recipe_sha256=args.expected_body_recipe_sha256,
            short_conv_construction=args.short_conv_construction,
            blank_seed=args.blank_seed,
            model_device=args.model_device,
            fold_compute_device=args.fold_compute_device,
            analytic_solve_device=args.analytic_solve_device,
            global_residual_backend=args.global_residual_backend,
            global_residual_closure_version=(
                args.global_residual_closure_version
            ),
            global_residual_basis_rank=args.global_residual_basis_rank,
            correction_scale=args.correction_scale,
            normalized_input_correction_rho=(args.normalized_input_correction_rho),
            normalized_input_request_capture_directory=(
                args.normalized_input_request_capture_directory
            ),
            live_child_prefix_recapture_mode=(
                args.live_child_prefix_recapture_mode
            ),
            target_dtype_representability_level=(
                args.target_dtype_representability_level
            ),
            child_dtype=args.child_dtype,
            maximum_prompt_tokens=args.maximum_prompt_tokens,
            maximum_new_tokens=args.maximum_new_tokens,
            coordinate_completion_recipe=args.coordinate_completion_recipe,
        )
    except BaseException as exc:
        exception_chain = _exception_chain_receipt(exc)
        failure_body = {
            "schema": FAILURE_SCHEMA,
            "status": "failed-without-behavior-promotion",
            "exception_type": type(exc).__name__,
            "message": exception_chain["entries"][0]["message"],
            "exception_chain": exception_chain,
            "automatic_acceptance_or_rejection_performed": False,
        }
        failure = {
            **failure_body,
            "failure_sha256": canonical_sha256(failure_body),
        }
        run_root = args.run_directory.expanduser().absolute()
        if run_root.is_dir() and not run_root.is_symlink():
            path = run_root / "failure-receipt.json"
            if not path.exists() and not path.is_symlink():
                _write_json(path, failure)
        print(json.dumps(failure, allow_nan=False, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
