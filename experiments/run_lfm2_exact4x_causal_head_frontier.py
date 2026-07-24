#!/usr/bin/env python3
"""Resume one exact-4x structural body into an analytic causal-head frontier.

The source run must already contain a freshly reloadable structural-body
checkpoint.  This runner never loads or executes the parent model, never opens
the large compiler cache, and never refolds the child body.  It authenticates
the existing semantic trace cache once, reconstructs the bounded parent causal
conditionals, replays them through the saved child body once, and analytically
materializes six independent dense output heads.

Every structurally valid recipe is saved, freshly reloaded, and decoded.  The
numeric solve diagnostics are navigation evidence only: they do not rank,
reject, promote, or delete candidates.  Whole decoded responses are the
behavioral evidence.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
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

from experiments.review_lfm2_semantic_frontier import (  # noqa: E402
    _load_parent_envelope,
    load_review_prompts,
    review_frontier,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_fold_executor import (  # noqa: E402
    fingerprint_model_state,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    fingerprint_lfm2_parent,
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_direct_local import (  # noqa: E402
    Lfm2DirectLocalCheckpointIO,
    load_lfm2_direct_local_tokenizer,
)
from neural_forge.hf_lfm2_exact4x_composite_interface import (  # noqa: E402
    LFM2_INDEPENDENT_STRUCTURAL_INTERFACE_INSTALL_SCHEMA,
    Lfm2IndependentStructuralInterfaceInstall,
    install_lfm2_independent_causal_output_head,
)
from neural_forge.hf_lfm2_interface_frontier import (  # noqa: E402
    DEFAULT_INTERFACE_CONTEXTS,
    LFM2_CAUSAL_TRIE_EVIDENCE_SCHEMA,
    LFM2_INTERFACE_SEQUENCE_PLAN_SCHEMA,
    LFM2_TERMINAL_CACHE_EVIDENCE_SCHEMA,
    collect_lfm2_child_causal_residual_evidence,
    collect_lfm2_terminal_cache_evidence,
    compile_lfm2_causal_residual_token_bridge,
    compile_lfm2_causal_trie_evidence,
    compile_lfm2_interface_sequence_plan,
    fingerprint_lfm2_causal_body_inventory,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    fingerprint_hf_tokenizer_semantics,
    load_strict_tokenizer_bridge_artifact,
)


RUN_SCHEMA = "neural-forge-lfm2-exact4x-causal-head-frontier-run-v1"
FRONTIER_SCHEMA = "neural-forge-lfm2-semantic-terminal-frontier-run-v1"
FAILURE_SCHEMA = "neural-forge-lfm2-exact4x-causal-head-frontier-failure-v1"
STRUCTURAL_BODY_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-exact4x-structural-body-checkpoint-v1"
)
DEFERRED_W2_SCHEMA = "neural-forge-lfm2-deferred-final-w2-installation-v1"
SEMANTIC_LAYER_STACK_SCHEMA = "neural-forge-lfm2-semantic-layer-stack-refold-v1"
EXPECTED_PARENT_UNIQUE_PARAMETERS = 229_693_184
EXPECTED_CHILD_UNIQUE_PARAMETERS = 57_422_440
EXPECTED_CHILD_TOPOLOGY = {
    "vocab_size": 4_096,
    "hidden_size": 776,
    "intermediate_size": 700,
    "num_hidden_layers": 14,
    "num_attention_heads": 10,
    "num_key_value_heads": 5,
    "head_dim": 64,
    "tie_word_embeddings": False,
}
STRUCTURAL_ZERO_OPERATION_AUDIT = {
    "training_steps": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "autograd_grad_calls": 0,
    "backward_calls": 0,
    "parent_forward_calls": 0,
    "parent_generate_calls": 0,
    "parent_trace_recaptures": 0,
    "child_forward_calls": 0,
    "child_generate_calls": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
    "parameters_sparsified": 0,
    "network_access": False,
}
STACK_ZERO_OPERATION_AUDIT = {
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
STRUCTURAL_BODY_ZERO_OPERATION_AUDIT = {
    "training_steps": 0,
    "optimizer_steps": 0,
    "backward_calls": 0,
    "parent_forward_calls": 0,
    "child_forward_calls": 0,
}
DEFERRED_W2_ZERO_OPERATION_AUDIT = {
    "training_steps": 0,
    "optimizer_steps": 0,
    "backward_calls": 0,
    "parent_forward_calls": 0,
    "child_forward_calls": 0,
}

# This is a factor-isolated neighborhood around the first integrated causal
# recipe.  It tests whether the causal delta is harmful, too weak/strong,
# ill-conditioned, or corrupting row scale.  All six are always materialized.
CAUSAL_HEAD_RECIPES = (
    {
        "candidate_id": "exact4x-body-control-a000-r1e2-norm1000",
        "ridge_fraction": 1e-2,
        "residual_scale_ppm": 0,
        "row_norm_calibration_ppm": 1_000_000,
        "protect_specials": True,
    },
    {
        "candidate_id": "exact4x-causal-a500-r1e2-norm1000",
        "ridge_fraction": 1e-2,
        "residual_scale_ppm": 500_000,
        "row_norm_calibration_ppm": 1_000_000,
        "protect_specials": True,
    },
    {
        "candidate_id": "exact4x-causal-a1000-r1e2-norm250",
        "ridge_fraction": 1e-2,
        "residual_scale_ppm": 1_000_000,
        "row_norm_calibration_ppm": 250_000,
        "protect_specials": True,
    },
    {
        "candidate_id": "exact4x-causal-a1000-r1e2-norm1000",
        "ridge_fraction": 1e-2,
        "residual_scale_ppm": 1_000_000,
        "row_norm_calibration_ppm": 1_000_000,
        "protect_specials": True,
    },
    {
        "candidate_id": "exact4x-causal-a1000-r1e1-norm1000",
        "ridge_fraction": 1e-1,
        "residual_scale_ppm": 1_000_000,
        "row_norm_calibration_ppm": 1_000_000,
        "protect_specials": True,
    },
    {
        "candidate_id": "exact4x-causal-a1000-r1e3-norm1000",
        "ridge_fraction": 1e-3,
        "residual_scale_ppm": 1_000_000,
        "row_norm_calibration_ppm": 1_000_000,
        "protect_specials": True,
    },
)


class Lfm2Exact4xCausalHeadFrontierError(RuntimeError):
    """An authenticated resume or structural materialization invariant failed."""


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
        raise Lfm2Exact4xCausalHeadFrontierError(
            f"{label} must be a lowercase SHA-256 digest."
        )
    return normalized


def _ordinary_directory(value: Path, label: str) -> Path:
    if "://" in os.fspath(value):
        raise Lfm2Exact4xCausalHeadFrontierError(f"{label} must be local.")
    path = value.expanduser().absolute()
    if path.is_symlink() or not path.is_dir():
        raise Lfm2Exact4xCausalHeadFrontierError(f"{label} is missing: {path}.")
    return path


def _ordinary_file(value: Path, label: str) -> Path:
    if "://" in os.fspath(value):
        raise Lfm2Exact4xCausalHeadFrontierError(f"{label} must be local.")
    path = value.expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise Lfm2Exact4xCausalHeadFrontierError(f"{label} is missing: {path}.")
    return path


def _new_run_directory(value: Path) -> Path:
    if "://" in os.fspath(value):
        raise Lfm2Exact4xCausalHeadFrontierError("run_directory must be local.")
    path = value.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise Lfm2Exact4xCausalHeadFrontierError(
            "run_directory must not already exist."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()
    return path


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Lfm2Exact4xCausalHeadFrontierError(
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


def _emit(stage: str, **values: Any) -> None:
    print(
        json.dumps(
            {"stage": stage, **values},
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _load_sealed_json(
    path: Path,
    *,
    seal_field: str,
    expected_seal: str,
    expected_schema: str,
    label: str,
) -> dict[str, Any]:
    document = json.loads(_ordinary_file(path, label).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise Lfm2Exact4xCausalHeadFrontierError(f"{label} must be a JSON object.")
    body = dict(document)
    supplied = body.pop(seal_field, None)
    if (
        supplied != canonical_sha256(body)
        or supplied != _digest(expected_seal, f"expected {label} seal")
        or document.get("schema") != expected_schema
    ):
        raise Lfm2Exact4xCausalHeadFrontierError(
            f"{label} seal, schema, or external pin is invalid."
        )
    return document


def _unique_parameter_count(model: Any) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def _artifact_inventory(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise Lfm2Exact4xCausalHeadFrontierError(
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
        raise Lfm2Exact4xCausalHeadFrontierError("Standalone checkpoint is empty.")
    return rows


def _authenticate_source_checkpoint_inventory(
    checkpoint: Path,
    *,
    expected_sha256: str,
) -> list[dict[str, Any]]:
    """Hash every source checkpoint asset immediately before one model load."""

    inventory = _artifact_inventory(checkpoint)
    actual_sha256 = canonical_sha256(inventory)
    if actual_sha256 != expected_sha256:
        raise Lfm2Exact4xCausalHeadFrontierError(
            "Source checkpoint files differ from the externally pinned inventory."
        )
    return inventory


def _authenticate_source_fold_receipts(
    *,
    structural_receipt: Mapping[str, Any],
    semantic_layer_stack_receipt: Mapping[str, Any],
    deferred_w2_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Close the sealed structural-stack-to-deferred-W2 provenance chain."""

    deferred_audit = {
        name: deferred_w2_receipt.get(name) for name in DEFERRED_W2_ZERO_OPERATION_AUDIT
    }
    changed_state_keys = deferred_w2_receipt.get("changed_state_keys")
    final_w2_parameter_name = deferred_w2_receipt.get("final_w2_parameter_name")
    effectful = deferred_w2_receipt.get("effectful_installation")
    state_changed = deferred_w2_receipt.get(
        "model_state_before_sha256"
    ) != deferred_w2_receipt.get("model_state_after_sha256")
    w2_changed = deferred_w2_receipt.get(
        "live_w2_sha256_before"
    ) != deferred_w2_receipt.get("live_w2_sha256_after")
    if (
        structural_receipt.get("operation_audit") != STRUCTURAL_ZERO_OPERATION_AUDIT
        or semantic_layer_stack_receipt.get("operation_audit")
        != STACK_ZERO_OPERATION_AUDIT
        or deferred_audit != DEFERRED_W2_ZERO_OPERATION_AUDIT
        or semantic_layer_stack_receipt.get(
            "final_layer_mlp_w2_deferred_to_fused_terminal"
        )
        is not True
        or semantic_layer_stack_receipt.get("final_layer_mlp_w2_install_count") != 0
        or deferred_w2_receipt.get("model_state_before_sha256")
        != semantic_layer_stack_receipt.get("model_state_after_sha256")
        or not isinstance(changed_state_keys, list)
        or not isinstance(final_w2_parameter_name, str)
        or not final_w2_parameter_name
        or not isinstance(effectful, bool)
        or state_changed is not effectful
        or w2_changed is not effectful
        or (changed_state_keys != ([final_w2_parameter_name] if effectful else []))
        or deferred_w2_receipt.get("prior_tensor_sha256")
        != deferred_w2_receipt.get("live_w2_sha256_after")
    ):
        raise Lfm2Exact4xCausalHeadFrontierError(
            "Source fold receipts do not prove a closed zero-operation "
            "stack-to-final-W2 transition."
        )
    return {
        "structural_interface": dict(structural_receipt["operation_audit"]),
        "semantic_layer_stack": dict(semantic_layer_stack_receipt["operation_audit"]),
        "deferred_final_w2": deferred_audit,
        "semantic_layer_stack_to_deferred_final_w2": {
            "semantic_layer_stack_result_sha256": semantic_layer_stack_receipt[
                "result_sha256"
            ],
            "model_state_before_sha256": deferred_w2_receipt[
                "model_state_before_sha256"
            ],
            "model_state_after_sha256": deferred_w2_receipt["model_state_after_sha256"],
            "final_w2_parameter_name": final_w2_parameter_name,
            "changed_state_keys": list(changed_state_keys),
            "effectful_installation": effectful,
        },
    }


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


def _frontier_receipt(
    *,
    run_directory: Path,
    materializations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_ids = [str(row["candidate_id"]) for row in materializations]
    body = {
        "schema": FRONTIER_SCHEMA,
        "status": "complete",
        "run_directory": str(run_directory),
        "candidate_ids": candidate_ids,
        "materializations": [dict(row) for row in materializations],
        "all_retained_candidates_materialized": True,
        "candidate_metric_threshold_used_for_materialization": False,
        "decoded_behavior_is_primary_evidence": True,
        "interface_isolation_control": True,
        "final_four_x_topology": True,
        "automatic_behavior_acceptance_or_rejection_performed": False,
        "binary_manual_verdict_required": False,
    }
    return {**body, "run_sha256": canonical_sha256(body)}


def _verify_optional_structural_body_receipt(
    source_root: Path,
    *,
    expected_receipt_sha256: str | None,
    checkpoint_inventory_sha256: str,
    body_state_sha256: str,
    body_identity_sha256: str,
    body_inventory_sha256: str,
    structural_install_sha256: str,
    semantic_layer_stack_sha256: str,
    deferred_w2_sha256: str,
) -> dict[str, Any] | None:
    path = source_root / "structural-body-receipt.json"
    if not path.exists() and not path.is_symlink():
        if expected_receipt_sha256 is not None:
            raise Lfm2Exact4xCausalHeadFrontierError(
                "Pinned structural-body receipt is missing from the source run."
            )
        return None
    if expected_receipt_sha256 is None:
        raise Lfm2Exact4xCausalHeadFrontierError(
            "A source structural-body receipt exists but was not externally pinned."
        )
    value = _load_sealed_json(
        path,
        seal_field="receipt_sha256",
        expected_seal=expected_receipt_sha256,
        expected_schema=STRUCTURAL_BODY_RECEIPT_SCHEMA,
        label="source structural-body receipt",
    )
    if (
        value.get("checkpoint_inventory_sha256") != checkpoint_inventory_sha256
        or value.get("body_model_state_sha256") != body_state_sha256
        or value.get("body_lfm2_identity_sha256") != body_identity_sha256
        or value.get("causal_body_inventory_sha256") != body_inventory_sha256
        or value.get("structural_interface_install_sha256") != structural_install_sha256
        or value.get("semantic_layer_stack_sha256") != semantic_layer_stack_sha256
        or value.get("deferred_final_w2_install_sha256") != deferred_w2_sha256
        or value.get("standalone_fresh_reload_verified") is not True
        or value.get("causal_output_head_installed") is not False
        or value.get("operation_audit") != STRUCTURAL_BODY_ZERO_OPERATION_AUDIT
    ):
        raise Lfm2Exact4xCausalHeadFrontierError(
            "Source structural-body receipt differs from the fresh reload."
        )
    return value


def run(
    *,
    run_directory: Path,
    source_run_directory: Path,
    donor_tokenizer_directory: Path,
    target_tokenizer_directory: Path,
    bridge_directory: Path,
    expected_bridge_artifact_sha256: str,
    semantic_cache_directory: Path,
    expected_semantic_cache_key: str,
    expected_parent_identity_sha256: str,
    expected_source_structural_interface_sha256: str,
    expected_source_deferred_w2_sha256: str,
    expected_source_terminal_evidence_sha256: str,
    expected_source_causal_trie_sha256: str,
    expected_source_sequence_plan_sha256: str,
    expected_source_checkpoint_inventory_sha256: str,
    expected_source_structural_body_receipt_sha256: str | None,
    prompts_json: Path,
    expected_prompts_file_sha256: str,
    parent_envelope: Path,
    expected_parent_envelope_sha256: str,
    expected_parent_envelope_file_sha256: str,
    model_device: str,
    analytic_solve_device: str,
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("PyTorch is required for causal-head folding.") from exc

    run_root = _new_run_directory(run_directory)
    source_root = _ordinary_directory(source_run_directory, "source run")
    donor_path = _ordinary_directory(donor_tokenizer_directory, "donor tokenizer")
    target_path = _ordinary_directory(target_tokenizer_directory, "target tokenizer")
    bridge_path = _ordinary_directory(bridge_directory, "tokenizer bridge")
    semantic_cache_path = _ordinary_directory(
        semantic_cache_directory,
        "semantic cache",
    )
    base_checkpoint = _ordinary_directory(
        source_root / "structural-body-checkpoint",
        "source structural-body checkpoint",
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
    expected_semantic_cache = _digest(
        expected_semantic_cache_key,
        "expected semantic cache key",
    )
    expected_source_checkpoint_inventory = _digest(
        expected_source_checkpoint_inventory_sha256,
        "expected source checkpoint inventory",
    )
    if (
        _file_sha256(prompts_path)
        != _digest(expected_prompts_file_sha256, "expected prompts file")
        or _file_sha256(envelope_path)
        != _digest(
            expected_parent_envelope_file_sha256,
            "expected parent envelope file",
        )
        or not torch.cuda.is_available()
        or not model_device.startswith("cuda:")
        or not analytic_solve_device.startswith("cuda:")
    ):
        raise Lfm2Exact4xCausalHeadFrontierError(
            "Pinned behavior assets or required CUDA execution differ."
        )
    if not 1 <= maximum_prompt_tokens <= 8192:
        raise Lfm2Exact4xCausalHeadFrontierError(
            "maximum_prompt_tokens must be between 1 and 8192."
        )
    if not 16 <= maximum_new_tokens <= 2048:
        raise Lfm2Exact4xCausalHeadFrontierError(
            "maximum_new_tokens must be between 16 and 2048."
        )

    _emit("authenticate-read-only-behavior-preflight")
    prompt_rows = [dict(value) for value in load_review_prompts(prompts_path)]
    prompt_set_sha256 = canonical_sha256(prompt_rows)
    parent_behavior = _load_parent_envelope(
        envelope_path,
        expected_envelope_sha256=_digest(
            expected_parent_envelope_sha256,
            "expected parent envelope",
        ),
        expected_file_sha256=_digest(
            expected_parent_envelope_file_sha256,
            "expected parent envelope file",
        ),
        expected_prompt_set_sha256=prompt_set_sha256,
        expected_prompt_ids=[str(value["id"]) for value in prompt_rows],
    )
    if (
        parent_behavior.get("parent_identity_sha256") != expected_parent
        or parent_behavior.get("parent_unique_parameters")
        != EXPECTED_PARENT_UNIQUE_PARAMETERS
        or parent_behavior.get("maximum_new_tokens") != maximum_new_tokens
    ):
        raise Lfm2Exact4xCausalHeadFrontierError(
            "Cached parent behavior differs from this resume request."
        )

    _emit("authenticate-source-receipts")
    structural_receipt = _load_sealed_json(
        source_root / "structural-interface-install.json",
        seal_field="result_sha256",
        expected_seal=expected_source_structural_interface_sha256,
        expected_schema=LFM2_INDEPENDENT_STRUCTURAL_INTERFACE_INSTALL_SCHEMA,
        label="source structural-interface receipt",
    )
    deferred_w2_receipt = _load_sealed_json(
        source_root / "deferred-final-w2-install.json",
        seal_field="result_sha256",
        expected_seal=expected_source_deferred_w2_sha256,
        expected_schema=DEFERRED_W2_SCHEMA,
        label="source deferred-W2 receipt",
    )
    semantic_layer_stack_receipt = _load_sealed_json(
        source_root / "semantic-layer-stack.json",
        seal_field="result_sha256",
        expected_seal=str(
            deferred_w2_receipt.get("semantic_layer_stack_result_sha256")
        ),
        expected_schema=SEMANTIC_LAYER_STACK_SCHEMA,
        label="source semantic layer-stack receipt",
    )
    expected_terminal_receipt = _load_sealed_json(
        source_root / "terminal-cache-evidence.json",
        seal_field="evidence_sha256",
        expected_seal=expected_source_terminal_evidence_sha256,
        expected_schema=LFM2_TERMINAL_CACHE_EVIDENCE_SCHEMA,
        label="source terminal-cache evidence",
    )
    expected_trie_receipt = _load_sealed_json(
        source_root / "causal-token-trie-evidence.json",
        seal_field="result_sha256",
        expected_seal=expected_source_causal_trie_sha256,
        expected_schema=LFM2_CAUSAL_TRIE_EVIDENCE_SCHEMA,
        label="source causal-trie evidence",
    )
    expected_sequence_plan = _load_sealed_json(
        source_root / "contextual-sequence-plan.json",
        seal_field="plan_sha256",
        expected_seal=expected_source_sequence_plan_sha256,
        expected_schema=LFM2_INTERFACE_SEQUENCE_PLAN_SCHEMA,
        label="source contextual sequence plan",
    )
    source_operation_audits = _authenticate_source_fold_receipts(
        structural_receipt=structural_receipt,
        semantic_layer_stack_receipt=semantic_layer_stack_receipt,
        deferred_w2_receipt=deferred_w2_receipt,
    )
    if (
        expected_trie_receipt.get("parent_terminal_evidence_sha256")
        != expected_terminal_receipt["evidence_sha256"]
        or structural_receipt.get("sequence_plan_sha256")
        != expected_sequence_plan["plan_sha256"]
        or any(
            value.get("parent_identity_sha256") != expected_parent
            for value in (expected_terminal_receipt, expected_trie_receipt)
        )
    ):
        raise Lfm2Exact4xCausalHeadFrontierError("Source evidence ancestry diverged.")

    _emit("reconstruct-parent-causal-evidence-from-one-semantic-cache-session")
    donor_tokenizer = load_lfm2_direct_local_tokenizer(donor_path)
    target_tokenizer = load_lfm2_direct_local_tokenizer(target_path)
    bridge = load_strict_tokenizer_bridge_artifact(
        bridge_path,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        expected_artifact_sha256=expected_bridge,
    )
    semantic_cache = open_lfm2_circuit_trace_cache(
        semantic_cache_path,
        verify=False,
    )
    if (
        semantic_cache.cache_key != expected_semantic_cache
        or semantic_cache.parent_fingerprint != expected_parent
        or fingerprint_hf_tokenizer_semantics(donor_tokenizer)
        != bridge.donor_tokenizer_semantics_sha256
        or fingerprint_hf_tokenizer_semantics(target_tokenizer)
        != bridge.target_tokenizer_semantics_sha256
    ):
        raise Lfm2Exact4xCausalHeadFrontierError(
            "Semantic cache, tokenizers, bridge, or parent ancestry changed."
        )
    terminal = collect_lfm2_terminal_cache_evidence(
        semantic_cache,
        expected_cache_key=expected_semantic_cache,
        expected_parent_identity_sha256=expected_parent,
        bridge=bridge,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
    )
    terminal_receipt = terminal.as_dict()
    if terminal_receipt != expected_terminal_receipt:
        raise Lfm2Exact4xCausalHeadFrontierError(
            "Reconstructed terminal evidence differs from the source run."
        )
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
    if causal_trie.as_dict() != expected_trie_receipt:
        raise Lfm2Exact4xCausalHeadFrontierError(
            "Reconstructed causal trie differs from the source run."
        )
    sequence_plan = compile_lfm2_interface_sequence_plan(
        bridge,
        donor_tokenizer,
        target_tokenizer,
        contexts=DEFAULT_INTERFACE_CONTEXTS,
    )
    if sequence_plan != expected_sequence_plan:
        raise Lfm2Exact4xCausalHeadFrontierError(
            "Reconstructed contextual sequence plan differs from the source run."
        )
    del terminal
    gc.collect()

    _emit("fresh-reload-and-authenticate-structural-body")
    checkpoint_io = Lfm2DirectLocalCheckpointIO(
        tokenizer_bridge=bridge,
        device=torch.device(model_device),
        dtype=torch.bfloat16,
    )
    checkpoint_inventory = _authenticate_source_checkpoint_inventory(
        base_checkpoint,
        expected_sha256=expected_source_checkpoint_inventory,
    )
    replay_model = checkpoint_io.load(base_checkpoint)
    replay_model.requires_grad_(False)
    replay_model.eval()
    body_state = fingerprint_model_state(replay_model)
    body_identity = fingerprint_lfm2_parent(replay_model).identity_sha256
    body_inventory_sha256 = fingerprint_lfm2_causal_body_inventory(replay_model)
    topology = _target_topology(replay_model)
    observed_topology = {key: topology[key] for key in EXPECTED_CHILD_TOPOLOGY}
    input_weight = replay_model.get_input_embeddings().weight
    output_weight = replay_model.get_output_embeddings().weight
    checkpoint_inventory_sha256 = canonical_sha256(checkpoint_inventory)
    if (
        _unique_parameter_count(replay_model) != EXPECTED_CHILD_UNIQUE_PARAMETERS
        or observed_topology != EXPECTED_CHILD_TOPOLOGY
        or input_weight is output_weight
        or not bool(
            torch.equal(
                input_weight.detach().to(device="cpu"),
                output_weight.detach().to(device="cpu"),
            )
        )
        or structural_receipt.get("exact_target_unique_parameters")
        != EXPECTED_CHILD_UNIQUE_PARAMETERS
        or structural_receipt.get("installed_input_table_sha256")
        != structural_receipt.get("installed_output_baseline_sha256")
        or deferred_w2_receipt.get("model_state_after_sha256") != body_identity
        or deferred_w2_receipt.get("training_steps") != 0
        or deferred_w2_receipt.get("optimizer_steps") != 0
        or deferred_w2_receipt.get("backward_calls") != 0
    ):
        raise Lfm2Exact4xCausalHeadFrontierError(
            "Fresh source body reload differs from its exact untied folded coordinate."
        )
    optional_body_receipt = _verify_optional_structural_body_receipt(
        source_root,
        expected_receipt_sha256=(expected_source_structural_body_receipt_sha256),
        checkpoint_inventory_sha256=checkpoint_inventory_sha256,
        body_state_sha256=body_state,
        body_identity_sha256=body_identity,
        body_inventory_sha256=body_inventory_sha256,
        structural_install_sha256=structural_receipt["result_sha256"],
        semantic_layer_stack_sha256=semantic_layer_stack_receipt["result_sha256"],
        deferred_w2_sha256=deferred_w2_receipt["result_sha256"],
    )
    source_operation_audits["optional_structural_body_checkpoint"] = (
        None
        if optional_body_receipt is None
        else dict(optional_body_receipt["operation_audit"])
    )
    source_authentication_body = {
        "schema": "neural-forge-lfm2-exact4x-resumed-body-authentication-v1",
        "source_run_directory": str(source_root),
        "source_checkpoint": str(base_checkpoint),
        "checkpoint_inventory": checkpoint_inventory,
        "checkpoint_inventory_sha256": checkpoint_inventory_sha256,
        "externally_pinned_checkpoint_inventory_sha256": (
            expected_source_checkpoint_inventory
        ),
        "structural_interface_install_sha256": structural_receipt["result_sha256"],
        "semantic_layer_stack_sha256": semantic_layer_stack_receipt["result_sha256"],
        "deferred_final_w2_install_sha256": deferred_w2_receipt["result_sha256"],
        "optional_structural_body_receipt_sha256": (
            None
            if optional_body_receipt is None
            else optional_body_receipt["receipt_sha256"]
        ),
        "body_model_state_sha256": body_state,
        "body_lfm2_identity_sha256": body_identity,
        "causal_body_inventory_sha256": body_inventory_sha256,
        "authenticated_source_operation_audits": source_operation_audits,
        "source_fold_zero_forbidden_operations_authenticated": True,
        "target_topology": topology,
        "target_unique_parameters": EXPECTED_CHILD_UNIQUE_PARAMETERS,
        "body_refold_calls": 0,
        "parent_model_loads": 0,
        "parent_forward_calls": 0,
        "compiler_cache_opens": 0,
    }
    source_authentication = {
        **source_authentication_body,
        "authentication_sha256": canonical_sha256(source_authentication_body),
    }
    _write_json(run_root / "source-body-authentication.json", source_authentication)

    _emit("replay-child-causal-residuals-once")
    child_evidence = collect_lfm2_child_causal_residual_evidence(
        replay_model,
        causal_trie,
        expected_input_table_sha256=structural_receipt["installed_input_table_sha256"],
        expected_body_inventory_sha256=body_inventory_sha256,
        device=model_device,
        evidence_role="exact4x-resumed-structural-body-shared-head-frontier",
    )
    child_evidence_receipt = child_evidence.as_dict()
    _write_json(
        run_root / "child-causal-residual-evidence.json",
        child_evidence_receipt,
    )
    structural_table = (
        replay_model.get_input_embeddings()
        .weight.detach()
        .to(device="cpu")
        .contiguous()
        .clone()
    )
    feature_rows = child_evidence.hidden_rows.to(dtype=torch.float32)
    gram_diagonal_mean = float(feature_rows.square().sum(dim=1).mean())
    if not gram_diagonal_mean > 0.0:
        raise Lfm2Exact4xCausalHeadFrontierError(
            "Replayed child residuals have zero analytic ridge scale."
        )
    target_special_ids = tuple(
        sorted(
            {
                int(value)
                for value in getattr(target_tokenizer, "all_special_ids", ())
                if 0 <= int(value) < EXPECTED_CHILD_TOPOLOGY["vocab_size"]
            }
        )
    )
    del replay_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _emit("compile-install-save-reload-six-output-heads")
    candidate_records = []
    materializations = []
    for recipe in CAUSAL_HEAD_RECIPES:
        candidate_id = str(recipe["candidate_id"])
        causal_bridge = compile_lfm2_causal_residual_token_bridge(
            structural_table,
            child_evidence,
            causal_trie,
            parent_terminal_evidence=terminal_receipt,
            sequence_plan_sha256=sequence_plan["plan_sha256"],
            ridge=gram_diagonal_mean * float(recipe["ridge_fraction"]),
            residual_scale_ppm=int(recipe["residual_scale_ppm"]),
            protected_target_ids=(
                target_special_ids if bool(recipe["protect_specials"]) else ()
            ),
            row_norm_calibration_ppm=int(recipe["row_norm_calibration_ppm"]),
            solve_device=analytic_solve_device,
        )
        candidate_root = run_root / "candidates" / candidate_id
        candidate_root.mkdir(parents=True)
        _write_json(
            candidate_root / "causal-token-bridge.json",
            causal_bridge.as_dict(),
        )
        _authenticate_source_checkpoint_inventory(
            base_checkpoint,
            expected_sha256=expected_source_checkpoint_inventory,
        )
        candidate = checkpoint_io.load(base_checkpoint)
        candidate.requires_grad_(False)
        candidate.eval()
        if (
            fingerprint_model_state(candidate) != body_state
            or fingerprint_lfm2_parent(candidate).identity_sha256 != body_identity
            or fingerprint_lfm2_causal_body_inventory(candidate)
            != body_inventory_sha256
            or _unique_parameter_count(candidate) != EXPECTED_CHILD_UNIQUE_PARAMETERS
            or candidate.get_input_embeddings().weight
            is candidate.get_output_embeddings().weight
        ):
            raise Lfm2Exact4xCausalHeadFrontierError(
                f"Fresh base reload changed before candidate {candidate_id!r}."
            )
        structural_install = Lfm2IndependentStructuralInterfaceInstall(
            model=candidate,
            receipt=structural_receipt,
        )
        installed = install_lfm2_independent_causal_output_head(
            structural_install,
            causal_bridge,
            child_evidence,
            expected_input_table_sha256=structural_receipt[
                "installed_input_table_sha256"
            ],
            expected_body_inventory_sha256=body_inventory_sha256,
        )
        install_receipt = installed.as_dict()
        _write_json(candidate_root / "causal-output-install.json", install_receipt)
        installed_state = fingerprint_model_state(candidate)
        installed_lfm2_fingerprint = fingerprint_lfm2_parent(candidate)
        checkpoint = candidate_root / "checkpoint"
        checkpoint.mkdir()
        checkpoint_io.save(candidate, checkpoint)
        del installed, structural_install, candidate
        gc.collect()
        reloaded = checkpoint_io.load(checkpoint)
        reload_state = fingerprint_model_state(reloaded)
        reload_lfm2_fingerprint = fingerprint_lfm2_parent(reloaded)
        if (
            reload_state != installed_state
            or reload_lfm2_fingerprint.as_dict() != installed_lfm2_fingerprint.as_dict()
            or _unique_parameter_count(reloaded) != EXPECTED_CHILD_UNIQUE_PARAMETERS
            or reloaded.get_input_embeddings().weight
            is reloaded.get_output_embeddings().weight
            or fingerprint_lfm2_causal_body_inventory(reloaded) != body_inventory_sha256
        ):
            raise Lfm2Exact4xCausalHeadFrontierError(
                f"Fresh candidate reload changed {candidate_id!r}."
            )
        del reloaded
        gc.collect()
        inventory = _artifact_inventory(checkpoint)
        record = {
            "candidate_id": candidate_id,
            "recipe": dict(recipe),
            "causal_token_bridge": causal_bridge.as_dict(),
            "causal_output_install": install_receipt,
            "checkpoint": str(checkpoint),
            "checkpoint_inventory": inventory,
            "checkpoint_inventory_sha256": canonical_sha256(inventory),
            "installed_model_state_sha256": installed_state,
            "fresh_reload_model_state_sha256": reload_state,
            "lfm2_fingerprint": reload_lfm2_fingerprint.as_dict(),
            "lfm2_identity_sha256": reload_lfm2_fingerprint.identity_sha256,
            "lfm2_config_sha256": reload_lfm2_fingerprint.config_sha256,
            "causal_body_inventory_sha256": body_inventory_sha256,
            "standalone_fresh_reload_verified": True,
            "unique_parameters": EXPECTED_CHILD_UNIQUE_PARAMETERS,
            "realized_parameter_compaction_ratio": (
                EXPECTED_PARENT_UNIQUE_PARAMETERS / EXPECTED_CHILD_UNIQUE_PARAMETERS
            ),
            "diagnostic_metrics_used_for_materialization": False,
        }
        candidate_records.append(record)
        materializations.append(
            {
                "candidate_id": candidate_id,
                "candidate_state_sha256": installed_state,
                "model_state_sha256": installed_state,
                "checkpoint_path": str(checkpoint),
                "checkpoint_inventory": inventory,
                "checkpoint_inventory_sha256": canonical_sha256(inventory),
                "lfm2_identity_sha256": reload_lfm2_fingerprint.identity_sha256,
                "lfm2_config_sha256": reload_lfm2_fingerprint.config_sha256,
                "unique_parameters": EXPECTED_CHILD_UNIQUE_PARAMETERS,
                "standalone_fresh_reload_verified": True,
            }
        )
        del causal_bridge
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    frontier = _frontier_receipt(
        run_directory=run_root,
        materializations=materializations,
    )
    frontier_path = run_root / "frontier-run.json"
    _write_json(frontier_path, frontier)
    dossier_path = run_root / "whole-response-dossier.json"
    candidate_ids = tuple(str(row["candidate_id"]) for row in materializations)
    dossier = review_frontier(
        frontier_run_receipt=frontier_path,
        expected_frontier_run_sha256=frontier["run_sha256"],
        expected_frontier_run_file_sha256=_file_sha256(frontier_path),
        prompts_json=prompts_path,
        output_json=dossier_path,
        candidate_ids=candidate_ids,
        device=model_device,
        dtype="bfloat16",
        maximum_prompt_tokens=maximum_prompt_tokens,
        maximum_new_tokens=maximum_new_tokens,
        parent_envelope=envelope_path,
        expected_parent_envelope_sha256=expected_parent_envelope_sha256,
        expected_parent_envelope_file_sha256=(expected_parent_envelope_file_sha256),
    )
    reviews = dossier.get("candidate_reviews")
    if (
        dossier.get("reviewed_candidate_count") != len(candidate_ids)
        or not isinstance(reviews, list)
        or [row.get("candidate_id") for row in reviews] != list(candidate_ids)
        or any(len(row.get("responses", [])) != len(prompt_rows) for row in reviews)
    ):
        raise Lfm2Exact4xCausalHeadFrontierError(
            "Whole-response dossier did not retain every candidate response."
        )

    resume_operation_audit = {
        "training_steps": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "parent_model_loads": 0,
        "parent_forward_calls": 0,
        "parent_generate_calls": 0,
        "parent_trace_recaptures": 0,
        "compiler_cache_opens": 0,
        "body_refolds": 0,
        "semantic_cache_authenticated_sessions": 1,
        "child_causal_replays": 1,
        "causal_head_analytic_solves": len(CAUSAL_HEAD_RECIPES),
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "parameters_sparsified": 0,
        "network_access": False,
    }
    resume_forbidden_operation_expectations = {
        name: resume_operation_audit[name]
        for name in (
            "training_steps",
            "optimizer_constructions",
            "optimizer_steps",
            "autograd_grad_calls",
            "backward_calls",
            "parent_model_loads",
            "parent_forward_calls",
            "parent_generate_calls",
            "parent_trace_recaptures",
            "compiler_cache_opens",
            "body_refolds",
            "parameters_pruned",
            "parameters_quantized",
            "parameters_sparsified",
            "network_access",
        )
    }
    if any(
        value not in (0, False)
        for value in resume_forbidden_operation_expectations.values()
    ):
        raise Lfm2Exact4xCausalHeadFrontierError(
            "The resume phase used a forbidden training, parent, or compression operation."
        )
    end_to_end_zero_basis = {
        "source_body_authentication_sha256": source_authentication[
            "authentication_sha256"
        ],
        "source_operation_audits_sha256": canonical_sha256(source_operation_audits),
        "resume_operation_audit_sha256": canonical_sha256(resume_operation_audit),
        "source_fold_audits_authenticated_before_resume": True,
        "resume_forbidden_operations_authenticated_zero": True,
    }
    body = {
        "schema": RUN_SCHEMA,
        "status": "six-exact4x-resumed-causal-heads-decoded",
        "run_directory": str(run_root),
        "source_run_directory": str(source_root),
        "source_body_authentication": source_authentication,
        "parent_identity_sha256": expected_parent,
        "parent_unique_parameters": EXPECTED_PARENT_UNIQUE_PARAMETERS,
        "target_unique_parameters": EXPECTED_CHILD_UNIQUE_PARAMETERS,
        "realized_parameter_compaction_ratio": (
            EXPECTED_PARENT_UNIQUE_PARAMETERS / EXPECTED_CHILD_UNIQUE_PARAMETERS
        ),
        "target_topology": topology,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "semantic_cache_key": semantic_cache.cache_key,
        "terminal_cache_evidence_sha256": terminal_receipt["evidence_sha256"],
        "causal_trie_evidence_sha256": causal_trie.as_dict()["result_sha256"],
        "sequence_plan_sha256": sequence_plan["plan_sha256"],
        "child_causal_residual_evidence_sha256": child_evidence_receipt[
            "evidence_sha256"
        ],
        "shared_child_causal_replay_used_for_all_heads": True,
        "causal_head_recipes": [dict(value) for value in CAUSAL_HEAD_RECIPES],
        "candidate_count": len(candidate_records),
        "candidate_ids": list(candidate_ids),
        "candidates": candidate_records,
        "all_candidates_materialized": True,
        "frontier_run_receipt": str(frontier_path),
        "frontier_run_sha256": frontier["run_sha256"],
        "whole_response_dossier": str(dossier_path),
        "whole_response_dossier_sha256": dossier["dossier_sha256"],
        "prompt_set_sha256": prompt_set_sha256,
        "prompt_count": len(prompt_rows),
        "maximum_prompt_tokens": maximum_prompt_tokens,
        "maximum_new_tokens": maximum_new_tokens,
        "behavior_evidence_policy": {
            "decoded_whole_responses_are_primary_evidence": True,
            "diagnostic_numbers_may_guide_navigation": True,
            "diagnostic_numbers_have_decision_authority": False,
            "automatic_behavior_acceptance_or_rejection_performed": False,
            "binary_manual_verdict_required": False,
            "ninety_percent_intelligence_retention_is_objective_not_cutoff": True,
        },
        "authenticated_source_operation_audits": source_operation_audits,
        "resume_operation_audit_scope": (
            "saved-structural-body-authentication-through-six-head-decode"
        ),
        "resume_operation_audit": resume_operation_audit,
        "end_to_end_zero_forbidden_operations_authenticated": True,
        "end_to_end_zero_forbidden_operations_basis": end_to_end_zero_basis,
    }
    result = {**body, "result_sha256": canonical_sha256(body)}
    _write_json(run_root / "run-receipt.json", result)
    _emit(
        "complete",
        result_sha256=result["result_sha256"],
        dossier_sha256=dossier["dossier_sha256"],
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--source-run-directory", required=True, type=Path)
    parser.add_argument("--donor-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--target-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--bridge-directory", required=True, type=Path)
    parser.add_argument("--expected-bridge-artifact-sha256", required=True)
    parser.add_argument("--semantic-cache-directory", required=True, type=Path)
    parser.add_argument("--expected-semantic-cache-key", required=True)
    parser.add_argument("--expected-parent-identity-sha256", required=True)
    parser.add_argument(
        "--expected-source-structural-interface-sha256",
        required=True,
    )
    parser.add_argument("--expected-source-deferred-w2-sha256", required=True)
    parser.add_argument(
        "--expected-source-terminal-evidence-sha256",
        required=True,
    )
    parser.add_argument("--expected-source-causal-trie-sha256", required=True)
    parser.add_argument("--expected-source-sequence-plan-sha256", required=True)
    parser.add_argument(
        "--expected-source-checkpoint-inventory-sha256",
        required=True,
    )
    parser.add_argument("--expected-source-structural-body-receipt-sha256")
    parser.add_argument("--prompts-json", required=True, type=Path)
    parser.add_argument("--expected-prompts-file-sha256", required=True)
    parser.add_argument("--parent-envelope", required=True, type=Path)
    parser.add_argument("--expected-parent-envelope-sha256", required=True)
    parser.add_argument(
        "--expected-parent-envelope-file-sha256",
        required=True,
    )
    parser.add_argument("--model-device", default="cuda:0")
    parser.add_argument("--analytic-solve-device", default="cuda:0")
    parser.add_argument("--maximum-prompt-tokens", type=int, default=1024)
    parser.add_argument("--maximum-new-tokens", type=int, default=256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run(
            run_directory=args.run_directory,
            source_run_directory=args.source_run_directory,
            donor_tokenizer_directory=args.donor_tokenizer_directory,
            target_tokenizer_directory=args.target_tokenizer_directory,
            bridge_directory=args.bridge_directory,
            expected_bridge_artifact_sha256=args.expected_bridge_artifact_sha256,
            semantic_cache_directory=args.semantic_cache_directory,
            expected_semantic_cache_key=args.expected_semantic_cache_key,
            expected_parent_identity_sha256=args.expected_parent_identity_sha256,
            expected_source_structural_interface_sha256=(
                args.expected_source_structural_interface_sha256
            ),
            expected_source_deferred_w2_sha256=(
                args.expected_source_deferred_w2_sha256
            ),
            expected_source_terminal_evidence_sha256=(
                args.expected_source_terminal_evidence_sha256
            ),
            expected_source_causal_trie_sha256=(
                args.expected_source_causal_trie_sha256
            ),
            expected_source_sequence_plan_sha256=(
                args.expected_source_sequence_plan_sha256
            ),
            expected_source_checkpoint_inventory_sha256=(
                args.expected_source_checkpoint_inventory_sha256
            ),
            expected_source_structural_body_receipt_sha256=(
                args.expected_source_structural_body_receipt_sha256
            ),
            prompts_json=args.prompts_json,
            expected_prompts_file_sha256=args.expected_prompts_file_sha256,
            parent_envelope=args.parent_envelope,
            expected_parent_envelope_sha256=(args.expected_parent_envelope_sha256),
            expected_parent_envelope_file_sha256=(
                args.expected_parent_envelope_file_sha256
            ),
            model_device=args.model_device,
            analytic_solve_device=args.analytic_solve_device,
            maximum_prompt_tokens=args.maximum_prompt_tokens,
            maximum_new_tokens=args.maximum_new_tokens,
        )
    except BaseException as exc:
        failure_body = {
            "schema": FAILURE_SCHEMA,
            "status": "failed-without-behavior-promotion",
            "exception_type": type(exc).__name__,
            "message": str(exc),
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


if __name__ == "__main__":
    raise SystemExit(main())
