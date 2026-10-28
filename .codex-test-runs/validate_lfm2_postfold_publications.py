#!/usr/bin/env python3
"""Authenticate three structural LFM2 postfold publications from any run root.

This is deliberately a structural/provenance validator.  It authenticates the
exact dense parameter budget, no-training/no-pruning/no-quantization contract,
standalone artifact trees, selected residual anchor, and hybrid tied-interface
proof continuity.  It does not read or threshold any behavioral metric; decoded
whole-response intelligence is reviewed separately by humans.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if _SOURCE_ROOT.is_dir() and str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

POSTFOLD_SCHEMA = "neural-forge-lfm2-semantic-postfold-v1"
POSTFOLD_STATUS = "standalone-semantic-postfold-saved-and-reloaded"
CHECKPOINT_SCHEMA = "neural-forge-lfm2-semantic-postfold-checkpoint-v1"
ARTIFACT_METADATA_SCHEMA = "neural-forge-lfm2-cpcf-v5-standalone-candidate-v1"
ANCHOR_SELECTION_SCHEMA = "neural-forge-lfm2-semantic-foundation-anchor-selection-v1"
ANCHOR_RESIDUAL_SCHEMA = "neural-forge-tied-causal-anchor-residual-v1"
BASE_INTERFACE_SCHEMA = "neural-forge-tied-causal-interface-fold-v1"
CONGRUENCE_SCHEMA = "neural-forge-tied-residual-congruence-candidate-v1"
CONGRUENCE_AUTH_SCHEMA = "neural-forge-tied-residual-congruence-authentication-v2"
POLAR_CERTIFICATE_SCHEMA = "neural-forge-source-bound-polar-certificate-v1"
FUSED_INTERFACE_SCHEMA = "neural-forge-lfm2-fused-terminal-tied-interface-v1"
TERMINAL_SCHEMA = "neural-forge-lfm2-semantic-lexical-completion-v1"
CAPSULE_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-semantic-preterminal-capsule-receipt-v1"
)
CAPSULE_MANIFEST_SCHEMA = (
    "neural-forge-lfm2-semantic-preterminal-capsule-manifest-v1"
)
CAPSULE_BINDINGS_SCHEMA = (
    "neural-forge-lfm2-semantic-preterminal-capsule-bindings-v1"
)
CAPSULE_CONTEXT_SCHEMA = (
    "neural-forge-lfm2-semantic-preterminal-capsule-context-v1"
)
HYBRID_MODE = "strong-congruence-plus-anchor-residual"

PARENT_UNIQUE_PARAMETERS = 229_693_184
CHILD_UNIQUE_PARAMETERS = 57_405_368
SELECTED_ANCHOR_INDEX = 7
RECIPES = (
    "fit_guided",
    "full_strength_structural",
    "residual_writer",
)
NATIVE_TIED_ALIAS_NAMES = frozenset(
    {"lm_head.weight", "model.embed_tokens.weight"}
)

POSTFOLD_ZERO_OPERATIONS = {
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

CAPSULE_ZERO_OPERATIONS = {
    "parent_model_objects_constructed": 0,
    "parent_model_forward_calls": 0,
    "child_model_forward_calls": 0,
    "model_generate_calls": 0,
    "trace_recaptures": 0,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "training_steps": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
    "network_access_performed": False,
    "pickle_files_written": 0,
}

CAPSULE_NONPROMOTABLE = {
    "status": "quarantined-preterminal-resume-only",
    "terminal_complete": False,
    "promotion_authorized": False,
    "accepted_artifact": False,
    "behavior_capability_proven": False,
    "training_performed": False,
    "optimizer_steps": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
}


class ReceiptValidationError(RuntimeError):
    """One publication differs from the committed postfold contract."""


def canonical_sha256(value: Any) -> str:
    """Match ``dynamic_circuit_ir.canonical_sha256`` without importing Torch."""

    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _scan_standalone_checkpoint(
    root: Path,
) -> tuple[tuple[dict[str, str], ...], tuple[dict[str, Any], ...]]:
    """Use the committed scanner, imported lazily so ``--help`` is lightweight."""

    from neural_forge.dynamic_fold_artifact import (
        canonical_standalone_fold_artifact_inventory,
        scan_standalone_fold_artifact_tree,
    )

    inventory = canonical_standalone_fold_artifact_inventory(root)
    roles = {row["path"]: row["artifact_role"] for row in inventory}
    tree = scan_standalone_fold_artifact_tree(root, expected_file_roles=roles)
    return inventory, tree


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptValidationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be one JSON object")
    return dict(value)


def _sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def _verify_seal(value: Any, field: str, label: str) -> dict[str, Any]:
    document = _mapping(value, label)
    supplied = document.pop(field, None)
    _sha256(supplied, f"{label}.{field}")
    _require(
        supplied == canonical_sha256(document),
        f"{label} self-seal is invalid",
    )
    return {**document, field: supplied}


def _ordinary_file(path: Path, label: str) -> Path:
    expanded = path.expanduser().absolute()
    _require(not expanded.is_symlink(), f"{label} cannot be a symbolic link")
    try:
        resolved = expanded.resolve(strict=True)
    except OSError as exc:
        raise ReceiptValidationError(f"{label} is missing: {expanded}") from exc
    _require(resolved.is_file(), f"{label} must be an ordinary file")
    return resolved


def _ordinary_directory(path: Path, label: str) -> Path:
    expanded = path.expanduser().absolute()
    _require(not expanded.is_symlink(), f"{label} cannot be a symbolic link")
    try:
        resolved = expanded.resolve(strict=True)
    except OSError as exc:
        raise ReceiptValidationError(f"{label} is missing: {expanded}") from exc
    _require(resolved.is_dir(), f"{label} must be an ordinary directory")
    return resolved


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    source = _ordinary_file(path, label)
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReceiptValidationError(f"{label} is not canonical JSON") from exc
    document = _mapping(value, label)
    canonical = (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    _require(raw == canonical, f"{label} bytes are not canonical JSON plus newline")
    return document, raw


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_zero_audit(value: Any, expected: Mapping[str, Any], label: str) -> None:
    audit = _mapping(value, label)
    _require(audit == dict(expected), f"{label} is not the exact committed zero audit")


def _validate_recipe(receipt: Mapping[str, Any], expected_recipe: str, label: str) -> None:
    recipe = _mapping(receipt.get("body_recipe"), f"{label}.body_recipe")
    recipe_sha = recipe.pop("recipe_sha256", None)
    _sha256(recipe_sha, f"{label}.body_recipe.recipe_sha256")
    _require(
        recipe_sha == canonical_sha256(recipe),
        f"{label} body-recipe seal is invalid",
    )
    _require(recipe.get("recipe_id") == expected_recipe, f"{label} recipe ID changed")
    _require(
        receipt.get("body_recipe_id") == expected_recipe
        and receipt.get("materialized_body_recipe_id") == expected_recipe
        and receipt.get("body_recipe_sha256") == recipe_sha,
        f"{label} materialized recipe binding changed",
    )
    _require(
        recipe.get("numeric_thresholds_can_gate_or_delete_candidates") is False
        and recipe.get("numeric_metrics_have_final_intelligence_authority") is False
        and recipe.get("whole_response_manual_intelligence_review_is_authoritative")
        is True,
        f"{label} incorrectly grants numeric metrics intelligence authority",
    )


def _validate_parameter_inventory(value: Any, label: str) -> dict[str, Any]:
    inventory = _mapping(value, label)
    supplied = inventory.pop("inventory_sha256", None)
    _sha256(supplied, f"{label}.inventory_sha256")
    seal_body = {**inventory, "checkpoint_byte_size_used": False}
    _require(
        supplied == canonical_sha256(seal_body),
        f"{label} seal is invalid after restoring its documented omitted field",
    )
    aliases = inventory.get("alias_groups")
    _require(isinstance(aliases, list), f"{label}.alias_groups must be a list")
    _require(
        inventory.get("unique_parameters") == CHILD_UNIQUE_PARAMETERS
        and inventory.get("parameter_iterator_unique_numel")
        == CHILD_UNIQUE_PARAMETERS
        and inventory.get("independent_methods_agree") is True,
        f"{label} does not prove exactly {CHILD_UNIQUE_PARAMETERS:,} dense parameters",
    )
    _require(
        inventory.get("quantization_adjustment_used") is False
        and inventory.get("pruning_mask_adjustment_used") is False,
        f"{label} uses quantization or pruning adjustment in parameter accounting",
    )
    _require(
        inventory.get("alias_group_count") == 1 and len(aliases) == 1,
        f"{label} has non-native parameter sharing",
    )
    alias = _mapping(aliases[0], f"{label}.alias_groups[0]")
    _require(
        frozenset(alias.get("names", ())) == NATIVE_TIED_ALIAS_NAMES,
        f"{label} contains an alias other than the native tied embedding/LM head",
    )
    return {**inventory, "inventory_sha256": supplied}


def _validate_checkpoint(
    value: Any,
    *,
    expected_directory: Path,
    context_sha256: str,
    anchor_selection_sha256: str,
    label: str,
) -> dict[str, Any]:
    checkpoint = _verify_seal(value, "result_sha256", label)
    _require(checkpoint.get("schema") == CHECKPOINT_SCHEMA, f"{label} schema changed")
    output = _ordinary_directory(Path(str(checkpoint.get("output_directory"))), label)
    _require(output == expected_directory, f"{label} output path differs from lane")
    _require(
        checkpoint.get("context_binding_sha256") == context_sha256
        and checkpoint.get("interface_anchor_selection_sha256")
        == anchor_selection_sha256,
        f"{label} context or anchor binding changed",
    )
    _require(
        checkpoint.get("parameter_element_count") == CHILD_UNIQUE_PARAMETERS,
        f"{label} parameter count is not exact",
    )
    for name, expected in {
        "staged_reload_verified": True,
        "published_reload_verified": True,
        "standalone_dense_safetensors": True,
        "parent_model_required_for_reload": False,
        "cache_required_for_reload": False,
        "network_access_performed": False,
        "complete_artifact_tree_hashed": True,
        "canonical_typed_artifact_inventory_authenticated": True,
        "staged_artifact_tree_purity_verified_before_publication": True,
        "unallowlisted_artifact_files_present": False,
        "forbidden_named_or_referenced_sidecars_present": False,
        "standalone_directory_purity_authenticated": True,
    }.items():
        _require(checkpoint.get(name) is expected, f"{label}.{name} changed")
    _require_exact_zero_audit(
        checkpoint.get("operation_audit"), POSTFOLD_ZERO_OPERATIONS, f"{label}.operation_audit"
    )

    inventory, tree = _scan_standalone_checkpoint(output)
    _require(
        checkpoint.get("artifact_inventory") == list(inventory)
        and checkpoint.get("artifact_inventory_sha256") == canonical_sha256(inventory),
        f"{label} typed artifact inventory differs from fresh scan",
    )
    _require(
        checkpoint.get("artifact_tree") == list(tree)
        and checkpoint.get("artifact_tree_sha256") == canonical_sha256(tree)
        and checkpoint.get("artifact_file_count") == len(tree)
        and checkpoint.get("artifact_total_bytes")
        == sum(int(row["size_bytes"]) for row in tree),
        f"{label} artifact byte tree differs from fresh scan",
    )

    metadata, _raw = _read_json(
        output / "neural_forge_lfm2_artifact.json",
        f"{label}.artifact_metadata",
    )
    artifact_sha = metadata.pop("artifact_sha256", None)
    _sha256(artifact_sha, f"{label}.artifact_metadata.artifact_sha256")
    _require(
        metadata.get("schema") == ARTIFACT_METADATA_SCHEMA
        and artifact_sha == canonical_sha256(metadata)
        and checkpoint.get("artifact_sha256") == artifact_sha,
        f"{label} artifact metadata is not authentic",
    )
    _require(
        metadata.get("model_state_sha256") == checkpoint.get("model_state_sha256")
        and metadata.get("parameter_inventory_sha256")
        == checkpoint.get("parameter_inventory_sha256")
        and metadata.get("tokenizer_semantics_sha256")
        == checkpoint.get("tokenizer_semantics_sha256"),
        f"{label} metadata identity differs from publication receipt",
    )
    for name, expected in {
        "dense_safetensors_only": True,
        "target_tokenizer_saved_with_child": True,
        "local_reload_only": True,
        "network_access": False,
        "hub_loader_used": False,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }.items():
        _require(metadata.get(name) == expected, f"{label}.metadata.{name} changed")
    _require(
        isinstance(metadata.get("dense_safetensor_files"), list)
        and bool(metadata["dense_safetensor_files"]),
        f"{label} has no dense safetensor payload",
    )
    parameter_inventory = _validate_parameter_inventory(
        metadata.get("parameter_inventory"), f"{label}.parameter_inventory"
    )
    _require(
        parameter_inventory["inventory_sha256"]
        == checkpoint.get("parameter_inventory_sha256"),
        f"{label} parameter inventory seal differs from receipt",
    )
    return checkpoint


def _validate_anchor_selection(value: Any, label: str) -> dict[str, Any]:
    selection = _verify_seal(value, "selection_sha256", label)
    recipe = _mapping(
        selection.get("selected_anchor_candidate_recipe"),
        f"{label}.selected_anchor_candidate_recipe",
    )
    recipe_sha = recipe.pop("recipe_sha256", None)
    _sha256(recipe_sha, f"{label}.selected_anchor_candidate_recipe.recipe_sha256")
    _require(recipe_sha == canonical_sha256(recipe), f"{label} recipe seal is invalid")
    _require(
        selection.get("schema") == ANCHOR_SELECTION_SCHEMA
        and selection.get("selected_anchor_candidate_index") == SELECTED_ANCHOR_INDEX
        and recipe.get("candidate_index") == SELECTED_ANCHOR_INDEX
        and selection.get("selected_anchor_candidate_recipe_sha256") == recipe_sha,
        f"{label} does not select authenticated candidate {SELECTED_ANCHOR_INDEX}",
    )
    _require(
        selection.get("selected_anchor_exactly_equals_base_plus_residual") is True
        and selection.get("selected_candidate_lazily_materialized") is True
        and selection.get("dense_anchor_candidate_checkpoints_persisted") == 0
        and selection.get("numeric_metrics_have_final_intelligence_authority") is False,
        f"{label} residual selection or numeric-authority contract changed",
    )
    return selection


def _validate_selected_anchor_source_binding(
    receipt: Mapping[str, Any],
    selection: Mapping[str, Any],
    label: str,
) -> str:
    """Return the selected anchor producer hash in the interface hash domain.

    The anchor frontier seals tensors with its descriptor-prefixed hash, while
    the base tied-interface and polar-certificate paths use the older
    dtype/NUL/shape/NUL/bytes hash.  The selected anchor report authenticates
    that its producer is fixed from its sealed base-interface lineage.  Follow
    that explicit lineage instead of comparing two intentionally different
    digests of the same producer bytes.
    """

    anchor = _verify_seal(
        receipt.get("rehydrated_interface"),
        "report_sha256",
        f"{label}.selected_anchor_report",
    )
    recipe = _mapping(
        selection.get("selected_anchor_candidate_recipe"),
        f"{label}.selected_anchor_recipe",
    )
    lineage = _mapping(
        anchor.get("base_interface_lineage"),
        f"{label}.base_interface_lineage",
    )
    base_report = _verify_seal(
        lineage.get("report"),
        "report_sha256",
        f"{label}.base_interface_report",
    )
    _require(
        anchor.get("schema") == ANCHOR_RESIDUAL_SCHEMA
        and anchor.get("report_sha256")
        == selection.get("selected_interface_report_sha256")
        and anchor.get("selected_candidate_index") == SELECTED_ANCHOR_INDEX
        and anchor.get("producer_encoder_fixed") is True
        and anchor.get("producer_encoder_sha256")
        == selection.get("selected_producer_encoder_sha256")
        and anchor.get("producer_encoder_sha256")
        == recipe.get("producer_encoder_sha256")
        and anchor.get("base_interface_report_sha256")
        == base_report.get("report_sha256")
        and lineage.get("report_sha256") == base_report.get("report_sha256")
        and lineage.get("producer_encoder_sha256")
        == base_report.get("producer_encoder_sha256")
        and lineage.get("tensor_identity_verified") is True
        and base_report.get("schema") == BASE_INTERFACE_SCHEMA,
        f"{label} does not authenticate a fixed producer from its base interface",
    )
    return _sha256(
        base_report.get("producer_encoder_sha256"),
        f"{label}.base_interface_producer_encoder_sha256",
    )


def _validate_source_bound_polar_certificate(
    interface: Mapping[str, Any],
    *,
    source_encoder_sha256: str,
    label: str,
) -> dict[str, Any]:
    certificate = _verify_seal(
        interface.get("anchor_source_bound_polar_certificate"),
        "result_sha256",
        label,
    )
    expected_fields = {
        "schema",
        "source_encoder_sha256",
        "producer_basis_sha256",
        "right_stretch_sha256",
        "source_width",
        "target_width",
        "fp64_scalar_operation_envelope",
        "fp64_gamma_relation_bound",
        "orthonormal_columns_within_derived_bound",
        "source_reconstruction_within_derived_bound",
        "right_stretch_symmetric_within_derived_bound",
        "right_stretch_positive_definite_above_derived_floor",
        "source_bound_unique_full_column_polar_factor_authenticated",
        "fresh_svd_reexecution_required",
        "empirical_tolerance_used",
        "result_sha256",
    }
    _require(
        set(certificate) == expected_fields,
        f"{label} fields differ from the committed polar-certificate schema",
    )
    source_width = certificate.get("source_width")
    target_width = certificate.get("target_width")
    _require(
        type(source_width) is int
        and source_width > 0
        and type(target_width) is int
        and target_width > 0,
        f"{label} widths must be positive integers",
    )
    operation_count = target_width * (2 * source_width + 2 * target_width + 2)
    scaled = operation_count * sys.float_info.epsilon
    expected_gamma = scaled / (1.0 - scaled)
    _require(
        certificate.get("schema") == POLAR_CERTIFICATE_SCHEMA
        and certificate.get("source_encoder_sha256")
        == source_encoder_sha256
        and certificate.get("producer_basis_sha256")
        == interface.get("producer_encoder_sha256")
        and certificate.get("source_width") == interface.get("source_width")
        and certificate.get("target_width") == interface.get("target_width")
        and certificate.get("fp64_scalar_operation_envelope") == operation_count
        and certificate.get("fp64_gamma_relation_bound") == expected_gamma,
        f"{label} is not bound to the selected source encoder and active basis",
    )
    _sha256(certificate.get("right_stretch_sha256"), f"{label}.right_stretch_sha256")
    for name, expected in {
        "orthonormal_columns_within_derived_bound": True,
        "source_reconstruction_within_derived_bound": True,
        "right_stretch_symmetric_within_derived_bound": True,
        "right_stretch_positive_definite_above_derived_floor": True,
        "source_bound_unique_full_column_polar_factor_authenticated": True,
        "fresh_svd_reexecution_required": False,
        "empirical_tolerance_used": False,
    }.items():
        _require(certificate.get(name) is expected, f"{label}.{name} changed")
    _require(
        interface.get("anchor_source_bound_polar_certificate_sha256")
        == certificate.get("result_sha256"),
        f"{label} seal differs from the active interface binding",
    )
    return certificate


def _validate_hybrid_continuity(
    receipt: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    capsule_manifest: Mapping[str, Any] | None,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    interface = _verify_seal(
        receipt.get("active_tied_interface"), "report_sha256", f"{label}.active_interface"
    )
    _require(
        interface.get("schema") == CONGRUENCE_SCHEMA
        and interface.get("mode") == HYBRID_MODE
        and interface.get("current_self_dual_interface_report_authentication_role")
        == (
            "compile-time-provenance-only;source-bound-polar-certificate-is-"
            "fresh-authentication"
        )
        and interface.get("selected_target_is_strong_base_plus_anchor_residual") is True
        and interface.get("anchor_residual_transport_applied") is True
        and interface.get("anchor_residual_added_after_strong_base") is True
        and interface.get("anchor_residual_transport_formula")
        == "delta_selfdual=delta_source@(B.T@E);target=(S@B)+delta_selfdual"
        and interface.get("strong_base_producer_encoder_sha256")
        == interface.get("producer_encoder_sha256")
        and interface.get("fp64_anchor_residual_formula_exact") is True
        and interface.get("fp64_selected_formula_exact") is True,
        f"{label} active interface is not the exact hybrid formula",
    )
    _require_exact_zero_audit(
        interface.get("operation_audit"), POSTFOLD_ZERO_OPERATIONS, f"{label}.interface.audit"
    )
    _require(
        interface.get("source_interface_report_sha256")
        == selection.get("selected_interface_report_sha256")
        and interface.get("anchor_base_target_weight_anchor_sha256")
        == selection.get("anchor_base_target_weight_sha256")
        and interface.get("anchor_source_target_residual_anchor_sha256")
        == selection.get("selected_residual_weight_sha256"),
        f"{label} hybrid interface is not bound to selected anchor {SELECTED_ANCHOR_INDEX}",
    )
    source_encoder_sha256 = _validate_selected_anchor_source_binding(
        receipt,
        selection,
        f"{label}.selected_anchor_source",
    )
    polar_certificate = _validate_source_bound_polar_certificate(
        interface,
        source_encoder_sha256=source_encoder_sha256,
        label=f"{label}.source_bound_polar_certificate",
    )

    layer = _verify_seal(
        receipt.get("semantic_layer_stack"), "result_sha256", f"{label}.semantic_layer_stack"
    )
    _require_exact_zero_audit(
        layer.get("operation_audit"),
        POSTFOLD_ZERO_OPERATIONS,
        f"{label}.semantic_layer_stack.audit",
    )
    _require(
        layer.get("interface_anchor_selection") == dict(selection)
        and layer.get("interface_anchor_selection_sha256")
        == selection.get("selection_sha256")
        and layer.get("selected_anchor_candidate_index") == SELECTED_ANCHOR_INDEX
        and layer.get("anchor_selection_bound_before_body_and_terminal_folding") is True,
        f"{label} layer stack lost the selected anchor binding",
    )
    auth = _verify_seal(
        layer.get("anchor_residual_transport_authentication"),
        "result_sha256",
        f"{label}.anchor_residual_authentication",
    )
    expected_auth_fields = {
        "schema",
        "interface_report_sha256",
        "mode",
        "producer_encoder_sha256",
        "target_weight_sha256",
        "producer_basis_orthogonality_authenticated",
        "mapped_parent_tied_table_supplied",
        "strong_congruence_recomputed_exact",
        "fp64_congruence_recomputed_exact",
        "bf16_congruence_recomputed_exact",
        "strong_base_recomputed_for_anchor_residual",
        "anchor_residual_formula_recomputed_exact",
        "anchor_source_bound_polar_authenticated",
        "anchor_source_bound_polar_certificate_sha256",
        "raw_linalg_diagnostics_in_authentication_identity",
        "strong_congruence_claimed",
        "strong_base_plus_anchor_residual_claimed",
        "current_selfdual_control_claimed",
        "anchor_residual_transport_applied",
        "operation_audit",
        "result_sha256",
    }
    _require(
        set(auth) == expected_auth_fields,
        f"{label} fields differ from the stable v2 authentication identity",
    )
    _require(
        layer.get("anchor_residual_transport_authentication_sha256")
        == auth.get("result_sha256")
        and auth.get("schema") == CONGRUENCE_AUTH_SCHEMA
        and auth.get("mode") == HYBRID_MODE
        and auth.get("interface_report_sha256") == interface.get("report_sha256")
        and auth.get("producer_encoder_sha256") == interface.get("producer_encoder_sha256")
        and auth.get("target_weight_sha256") == interface.get("target_weight_sha256")
        and auth.get("mapped_parent_tied_table_supplied") is True
        and auth.get("strong_base_recomputed_for_anchor_residual") is True
        and auth.get("anchor_residual_formula_recomputed_exact") is True
        and auth.get("anchor_source_bound_polar_authenticated") is True
        and auth.get("anchor_source_bound_polar_certificate_sha256")
        == polar_certificate.get("result_sha256")
        and auth.get("raw_linalg_diagnostics_in_authentication_identity") is False
        and auth.get("strong_base_plus_anchor_residual_claimed") is True
        and auth.get("anchor_residual_transport_applied") is True,
        f"{label} hybrid formula was not independently recomputed exactly",
    )
    _require(
        not {
            "producer_basis_orthogonality_spectral_error_recomputed",
            "recomputed_congruence_relative_error",
            "recomputed_congruence_maximum_absolute_error",
        }.intersection(auth),
        f"{label} authentication identity contains thread-unstable diagnostics",
    )
    _require_exact_zero_audit(
        auth.get("operation_audit"), POSTFOLD_ZERO_OPERATIONS, f"{label}.anchor_auth.audit"
    )
    lineage = _mapping(layer.get("tied_interface_lineage"), f"{label}.interface_lineage")
    _require(
        lineage.get("interface_report_schema") == CONGRUENCE_SCHEMA
        and lineage.get("interface_report_sha256") == interface.get("report_sha256")
        and lineage.get("producer_encoder_sha256") == interface.get("producer_encoder_sha256")
        and lineage.get("target_weight_sha256") == interface.get("target_weight_sha256")
        and lineage.get("tied_residual_congruence_authenticated") is True
        and lineage.get("tied_residual_congruence_authentication_sha256")
        == auth.get("result_sha256"),
        f"{label} layer lineage differs from exact hybrid authentication",
    )

    terminal = _verify_seal(
        receipt.get("fused_semantic_terminal"), "result_sha256", f"{label}.terminal"
    )
    _require(
        terminal.get("schema") == TERMINAL_SCHEMA
        and terminal.get("semantic_layer_stack_result_sha256")
        == layer.get("result_sha256")
        and terminal.get("interface_anchor_selection") == dict(selection)
        and terminal.get("interface_anchor_selection_sha256")
        == selection.get("selection_sha256"),
        f"{label} fused terminal lost layer/anchor continuity",
    )
    terminal_auth = _verify_seal(
        terminal.get("tied_interface_authentication"),
        "result_sha256",
        f"{label}.terminal_interface_authentication",
    )
    _require(
        terminal.get("tied_interface_authentication_sha256")
        == terminal_auth.get("result_sha256")
        and terminal_auth.get("schema") == FUSED_INTERFACE_SCHEMA
        and terminal_auth.get("mode") == HYBRID_MODE
        and terminal_auth.get("interface_authentication_sha256")
        == auth.get("result_sha256")
        and terminal_auth.get("interface_report_sha256") == interface.get("report_sha256")
        and terminal_auth.get("producer_encoder_sha256") == interface.get("producer_encoder_sha256")
        and terminal_auth.get("target_weight_sha256") == interface.get("target_weight_sha256")
        and terminal_auth.get("strong_congruence_recomputed_exact")
        is auth.get("strong_congruence_recomputed_exact")
        and terminal_auth.get("global_residual_gauge_frozen_through_terminal") is True
        and terminal_auth.get("live_native_tied_table_matches_interface") is True,
        f"{label} terminal authentication differs from layer publication proof",
    )
    _require_exact_zero_audit(
        terminal_auth.get("operation_audit"),
        POSTFOLD_ZERO_OPERATIONS,
        f"{label}.terminal_interface_authentication.audit",
    )
    _require_exact_zero_audit(
        terminal.get("operation_audit"), POSTFOLD_ZERO_OPERATIONS, f"{label}.terminal.audit"
    )

    if capsule_manifest is not None:
        _require(
            capsule_manifest.get("semantic_layer_stack_receipt") == dict(layer)
            and capsule_manifest.get("active_interface_report") == dict(interface)
            and capsule_manifest.get("active_interface_authentication") == dict(auth),
            f"{label} capsule does not pin the exact publication-time hybrid proof",
        )
    return layer, auth, terminal


def _validate_capsule(
    value: Any,
    *,
    expected_directory: Path,
    label: str,
) -> dict[str, Any]:
    reference = _mapping(value, label)
    _require(
        set(reference)
        == {
            "path",
            "receipt_sha256",
            "receipt_file_sha256",
            "manifest_sha256",
            "bindings_sha256",
            "context_binding_sha256",
        },
        f"{label} fields differ from committed capsule reference schema",
    )
    root = _ordinary_directory(Path(str(reference.get("path"))), label)
    _require(root == expected_directory, f"{label} path differs from lane")
    for name in (
        "receipt_sha256",
        "receipt_file_sha256",
        "manifest_sha256",
        "bindings_sha256",
        "context_binding_sha256",
    ):
        _sha256(reference.get(name), f"{label}.{name}")

    capsule_receipt_path = root / "capsule-receipt.json"
    capsule_receipt, raw = _read_json(capsule_receipt_path, f"{label}.receipt")
    _require(
        hashlib.sha256(raw).hexdigest() == reference["receipt_file_sha256"],
        f"{label} receipt file hash differs from reference",
    )
    capsule_receipt = _verify_seal(
        capsule_receipt, "receipt_sha256", f"{label}.receipt"
    )
    _require(
        capsule_receipt.get("schema") == CAPSULE_RECEIPT_SCHEMA
        and capsule_receipt.get("receipt_sha256") == reference["receipt_sha256"]
        and capsule_receipt.get("manifest_sha256") == reference["manifest_sha256"]
        and capsule_receipt.get("bindings_sha256") == reference["bindings_sha256"]
        and capsule_receipt.get("context_binding_sha256")
        == reference["context_binding_sha256"],
        f"{label} receipt differs from external reference",
    )
    for name, expected in CAPSULE_NONPROMOTABLE.items():
        _require(capsule_receipt.get(name) == expected, f"{label}.receipt.{name} changed")
    _require_exact_zero_audit(
        capsule_receipt.get("operation_audit"),
        CAPSULE_ZERO_OPERATIONS,
        f"{label}.receipt.operation_audit",
    )

    manifest_record = _mapping(
        capsule_receipt.get("manifest_file"), f"{label}.receipt.manifest_file"
    )
    manifest, manifest_raw = _read_json(
        root / "capsule-manifest.json", f"{label}.manifest"
    )
    _require(
        manifest_record.get("path") == "capsule-manifest.json"
        and manifest_record.get("size_bytes") == len(manifest_raw)
        and manifest_record.get("sha256") == hashlib.sha256(manifest_raw).hexdigest(),
        f"{label} manifest file differs from capsule receipt",
    )
    manifest = _verify_seal(manifest, "manifest_sha256", f"{label}.manifest")
    _require(
        manifest.get("schema") == CAPSULE_MANIFEST_SCHEMA
        and manifest.get("manifest_sha256") == reference["manifest_sha256"],
        f"{label} manifest schema or seal changed",
    )
    for name, expected in CAPSULE_NONPROMOTABLE.items():
        _require(manifest.get(name) == expected, f"{label}.manifest.{name} changed")
    _require_exact_zero_audit(
        manifest.get("operation_audit"),
        CAPSULE_ZERO_OPERATIONS,
        f"{label}.manifest.operation_audit",
    )
    bindings = _mapping(manifest.get("bindings"), f"{label}.manifest.bindings")
    _require(
        bindings.get("schema") == CAPSULE_BINDINGS_SCHEMA
        and canonical_sha256(bindings) == reference["bindings_sha256"]
        and manifest.get("bindings_sha256") == reference["bindings_sha256"],
        f"{label} bindings changed",
    )
    context = _mapping(
        manifest.get("context_binding"), f"{label}.manifest.context_binding"
    )
    _require(
        context.get("schema") == CAPSULE_CONTEXT_SCHEMA
        and canonical_sha256(context) == reference["context_binding_sha256"]
        and manifest.get("context_binding_sha256")
        == reference["context_binding_sha256"],
        f"{label} context binding changed",
    )
    _require(
        manifest.get("anchor_residual_transport_inputs_persisted") is True
        and manifest.get("payload_tensor_count") == 22
        and isinstance(manifest.get("anchor_source_interface_report"), Mapping)
        and isinstance(manifest.get("anchor_transport_witness"), Mapping)
        and manifest.get("exact_four_typed_evidence_banks") is True
        and manifest.get("active_interface_tensors_persisted") is True
        and manifest.get("dense_strict_local_checkpoint") is True
        and manifest.get("no_pickle_files") is True,
        f"{label} did not persist the complete hybrid preterminal seam",
    )
    model = _mapping(manifest.get("model"), f"{label}.manifest.model")
    _require(
        model.get("parameter_element_count") == CHILD_UNIQUE_PARAMETERS
        and model.get("final_w2_exact_zero") is True,
        f"{label} model binding is not the exact preterminal child",
    )
    _validate_parameter_inventory(
        model.get("parameter_inventory"), f"{label}.manifest.model.parameter_inventory"
    )

    payload = _mapping(manifest.get("seam_payload"), f"{label}.manifest.seam_payload")
    payload_path = _ordinary_file(root / "preterminal-seam.safetensors", f"{label}.payload")
    _require(
        payload.get("path") == "preterminal-seam.safetensors"
        and payload.get("size_bytes") == payload_path.stat().st_size
        and payload.get("sha256") == _file_sha256(payload_path),
        f"{label} seam payload differs from manifest",
    )

    checkpoint_record = _mapping(
        manifest.get("checkpoint"), f"{label}.manifest.checkpoint"
    )
    checkpoint_root = _ordinary_directory(root / "checkpoint", f"{label}.checkpoint")
    inventory, tree = _scan_standalone_checkpoint(checkpoint_root)
    _require(
        checkpoint_record.get("artifact_inventory") == list(inventory)
        and checkpoint_record.get("artifact_inventory_sha256")
        == canonical_sha256(inventory)
        and checkpoint_record.get("artifact_tree") == list(tree)
        and checkpoint_record.get("artifact_tree_sha256") == canonical_sha256(tree),
        f"{label} checkpoint byte tree differs from manifest",
    )
    artifact_metadata = _mapping(
        checkpoint_record.get("artifact_metadata"), f"{label}.checkpoint.metadata"
    )
    _require(
        artifact_metadata.get("artifact_sha256")
        == capsule_receipt.get("checkpoint_artifact_sha256")
        and checkpoint_record.get("artifact_tree_sha256")
        == capsule_receipt.get("checkpoint_artifact_tree_sha256"),
        f"{label} checkpoint hashes differ from capsule receipt",
    )
    for name, expected in {
        "dense_safetensors_only": True,
        "network_access": False,
        "hub_loader_used": False,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }.items():
        _require(
            artifact_metadata.get(name) == expected,
            f"{label}.checkpoint.metadata.{name} changed",
        )
    _validate_parameter_inventory(
        artifact_metadata.get("parameter_inventory"),
        f"{label}.checkpoint.metadata.parameter_inventory",
    )
    return manifest


def _validate_lane(base: Path, recipe: str) -> dict[str, Any]:
    lane = _ordinary_directory(base / recipe, f"{recipe} lane")
    receipt_path = lane / "postfold-receipt.json"
    receipt, raw = _read_json(receipt_path, f"{recipe} postfold receipt")
    label = f"{recipe} postfold receipt"
    receipt = _verify_seal(receipt, "result_sha256", label)
    _require(
        receipt.get("schema") == POSTFOLD_SCHEMA
        and receipt.get("status") == POSTFOLD_STATUS,
        f"{label} is not a successful publication",
    )
    _validate_recipe(receipt, recipe, label)
    _require(
        receipt.get("complete_body_recipe_candidate_ids") == list(RECIPES)
        and receipt.get("complete_body_recipe_candidate_count") == len(RECIPES)
        and receipt.get("one_checkpoint_contains_one_materialized_body_recipe") is True,
        f"{label} complete recipe frontier changed",
    )
    _require_exact_zero_audit(
        receipt.get("operation_audit"), POSTFOLD_ZERO_OPERATIONS, f"{label}.operation_audit"
    )
    claims = _mapping(receipt.get("claims"), f"{label}.claims")
    _require(
        claims.get("standalone_child") is True
        and claims.get("parameter_count_compaction_changed_by_postfold") is False
        and claims.get("training_performed") is False
        and claims.get("pruning_performed") is False
        and claims.get("quantization_performed") is False
        and claims.get("numeric_metrics_have_final_intelligence_authority") is False,
        f"{label} claims training, pruning, quantization, or numeric authority",
    )
    _require(
        receipt.get("requested_tied_interface_mode") == HYBRID_MODE
        and receipt.get("tied_interface_mode") == HYBRID_MODE,
        f"{label} did not execute hybrid residual-attention folding",
    )
    selection = _validate_anchor_selection(
        receipt.get("interface_anchor_selection"), f"{label}.anchor_selection"
    )

    capsule_reference = _mapping(
        receipt.get("preterminal_capsule"), f"{label}.preterminal_capsule"
    )
    context = _verify_seal(
        receipt.get("context_binding"), "context_sha256", f"{label}.context_binding"
    )
    _require(
        context.get("body_recipe_id") == recipe
        and context.get("materialized_body_recipe_id") == recipe
        and context.get("interface_anchor_selection") == dict(selection)
        and context.get("interface_anchor_selection_sha256")
        == selection.get("selection_sha256")
        and context.get("preterminal_capsule") == capsule_reference,
        f"{label} context lost recipe, anchor, or capsule continuity",
    )

    checkpoint = _validate_checkpoint(
        receipt.get("checkpoint"),
        expected_directory=_ordinary_directory(lane / "checkpoint", f"{recipe} checkpoint"),
        context_sha256=context["context_sha256"],
        anchor_selection_sha256=selection["selection_sha256"],
        label=f"{label}.checkpoint",
    )
    capsule_manifest = _validate_capsule(
        capsule_reference,
        expected_directory=_ordinary_directory(
            lane / "preterminal-capsule", f"{recipe} preterminal capsule"
        ),
        label=f"{label}.preterminal_capsule",
    )
    layer, auth, terminal = _validate_hybrid_continuity(
        receipt,
        selection=selection,
        capsule_manifest=capsule_manifest,
        label=label,
    )
    installation = _verify_seal(
        terminal.get("installation"),
        "result_sha256",
        f"{label}.terminal.installation",
    )
    fresh_reload_identity = _sha256(
        receipt.get("fresh_reload_parent_free_state_sha256"),
        f"{label}.fresh_reload_parent_free_state_sha256",
    )
    _require(
        installation.get("schema")
        == "neural-forge-lfm2-fused-semantic-terminal-install-v1"
        and fresh_reload_identity == installation.get("model_state_after_sha256"),
        f"{label} fresh semantic identity differs from terminal installation state",
    )
    _require(
        context.get("semantic_layer_stack_stage_sha256") == layer.get("result_sha256")
        and context.get("fused_semantic_terminal_refold_sha256")
        == terminal.get("result_sha256")
        and context.get("active_tied_interface_report_sha256")
        == receipt["active_tied_interface"]["report_sha256"],
        f"{label} context hashes differ from layer/interface/terminal",
    )
    _require(
        capsule_manifest["context_binding"].get(
            "semantic_layer_stack_result_sha256"
        )
        == layer.get("result_sha256")
        and capsule_manifest["context_binding"].get(
            "interface_authentication_sha256"
        )
        == auth.get("result_sha256"),
        f"{label} capsule context differs from exact hybrid authentication",
    )
    return {
        "recipe_id": recipe,
        "receipt_path": str(receipt_path.resolve()),
        "receipt_file_sha256": hashlib.sha256(raw).hexdigest(),
        "receipt_sha256": receipt["result_sha256"],
        "checkpoint_path": checkpoint["output_directory"],
        "checkpoint_receipt_sha256": checkpoint["result_sha256"],
        "checkpoint_artifact_sha256": checkpoint["artifact_sha256"],
        "checkpoint_tensor_state_sha256": checkpoint["model_state_sha256"],
        "fresh_reload_semantic_identity_sha256": fresh_reload_identity,
        "preterminal_capsule_path": capsule_reference["path"],
        "preterminal_capsule_receipt_sha256": capsule_reference["receipt_sha256"],
        "preterminal_capsule_manifest_sha256": capsule_reference["manifest_sha256"],
        "anchor_candidate_index": SELECTED_ANCHOR_INDEX,
        "hybrid_authentication_schema": CONGRUENCE_AUTH_SCHEMA,
        "hybrid_interface_authentication_sha256": auth["result_sha256"],
        "source_bound_polar_certificate_sha256": receipt["active_tied_interface"][
            "anchor_source_bound_polar_certificate_sha256"
        ],
        "child_unique_parameters": CHILD_UNIQUE_PARAMETERS,
    }


def validate(base_directory: Path) -> dict[str, Any]:
    base = _ordinary_directory(base_directory, "postfold base directory")
    lanes = [_validate_lane(base, recipe) for recipe in RECIPES]
    _require(
        {lane["recipe_id"] for lane in lanes} == set(RECIPES),
        "the three publications do not contain exactly the committed recipe lanes",
    )
    for field in (
        "receipt_sha256",
        "checkpoint_path",
        "preterminal_capsule_path",
    ):
        _require(
            len({lane[field] for lane in lanes}) == len(RECIPES),
            f"the three lanes unexpectedly share {field}",
        )
    return {
        "schema": "neural-forge-lfm2-postfold-structural-validation-v2",
        "status": "three-structurally-authentic-postfold-publications",
        "base_directory": str(base),
        "parent_unique_parameters": PARENT_UNIQUE_PARAMETERS,
        "child_unique_parameters": CHILD_UNIQUE_PARAMETERS,
        "exact_compression_ratio_numerator": PARENT_UNIQUE_PARAMETERS,
        "exact_compression_ratio_denominator": CHILD_UNIQUE_PARAMETERS,
        "compression_ratio_diagnostic": (
            PARENT_UNIQUE_PARAMETERS / CHILD_UNIQUE_PARAMETERS
        ),
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "sparse_or_factorized_storage_used": False,
        "non_native_weight_sharing_used": False,
        "parent_or_cache_sidecar_present": False,
        "numeric_behavior_or_intelligence_metric_used_as_acceptance_gate": False,
        "decoded_whole_response_manual_intelligence_review_still_required": True,
        "lanes": lanes,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-directory",
        type=Path,
        required=True,
        help=(
            "Directory containing fit_guided/, full_strength_structural/, and "
            "residual_writer/ postfold publications."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate(args.base_directory)
    except (OSError, ValueError, TypeError, ReceiptValidationError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
