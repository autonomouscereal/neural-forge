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

import torch  # noqa: E402

from experiments.review_lfm2_semantic_frontier import (  # noqa: E402
    _load_parent_envelope,
    load_review_prompts,
    review_frontier,
)
from experiments.run_lfm2_residual_attention_fold import (  # noqa: E402
    load_local_tokenizer,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_fold_executor import (  # noqa: E402
    fingerprint_model_state,
)
from neural_forge.dynamic_lfm2_adapter import (  # noqa: E402
    Lfm2TargetSynthesisAdapter,
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
    DynamicTargetSynthesizer,
    plan_budgeted_dynamic_target,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    fingerprint_lfm2_parent,
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import (  # noqa: E402
    install_lfm2_cpcf_neutral_norms,
)
from neural_forge.hf_lfm2_direct_local import (  # noqa: E402
    Lfm2DirectLocalCheckpointIO,
    load_lfm2_direct_local_model,
)
from neural_forge.hf_lfm2_exact4x_composite_interface import (  # noqa: E402
    install_lfm2_independent_causal_output_head,
    install_lfm2_independent_structural_interface,
)
from neural_forge.hf_lfm2_interface_frontier import (  # noqa: E402
    DEFAULT_INTERFACE_CONTEXTS,
    collect_lfm2_child_causal_residual_evidence,
    collect_lfm2_terminal_cache_evidence,
    compile_lfm2_causal_residual_token_bridge,
    compile_lfm2_causal_trie_evidence,
    compile_lfm2_contextual_input_table,
    compile_lfm2_interface_sequence_plan,
    fingerprint_lfm2_causal_body_inventory,
)
from neural_forge.hf_lfm2_semantic_body_recipe import (  # noqa: E402
    resolve_lfm2_semantic_body_recipe,
)
from neural_forge.hf_lfm2_semantic_foundation import (  # noqa: E402
    build_lfm2_semantic_foundation_attention_lineage,
    fit_lfm2_semantic_foundation_interface,
    inventory_lfm2_semantic_foundation_parent,
    read_lfm2_semantic_foundation_catalog_preparation,
)
from neural_forge.hf_lfm2_semantic_postfold import (  # noqa: E402
    compile_install_lfm2_semantic_layer_stack_stage,
    install_lfm2_deferred_final_w2_prior,
    normalize_lfm2_semantic_foundation_attention_lineage,
)
from neural_forge.hf_lfm2_semantic_short_conv_refold_adapter import (  # noqa: E402
    LFM2_SEMANTIC_SHORT_CONV_CONSTRUCTION_IDS,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    fingerprint_hf_tokenizer_semantics,
    load_strict_tokenizer_bridge_artifact,
)


RUN_SCHEMA = "neural-forge-lfm2-exact4x-composite-fold-run-v1"
FRONTIER_SCHEMA = "neural-forge-lfm2-semantic-terminal-frontier-run-v1"
FAILURE_SCHEMA = "neural-forge-lfm2-exact4x-composite-fold-failure-v1"
EXPECTED_PARENT_UNIQUE_PARAMETERS = 229_693_184
EXPECTED_CHILD_UNIQUE_PARAMETERS = 57_422_440
PARENT_MODEL_ROLE = "parent"
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
CAUSAL_HEAD_RECIPE = {
    "candidate_id": "exact4x-context-residual-attention-causal-head-a1000-r1e2",
    "ridge_fraction": 1e-2,
    "residual_scale_ppm": 1_000_000,
    "row_norm_calibration_ppm": 250_000,
    "protect_specials": True,
}


class Lfm2Exact4xCompositeFoldError(RuntimeError):
    """One externally pinned composite-fold invariant failed."""


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
        raise Lfm2Exact4xCompositeFoldError(
            "run_directory must not already exist."
        )
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


def _parameter_storage_addresses(model: Any) -> set[int]:
    return {
        int(parameter.untyped_storage().data_ptr())
        for parameter in model.parameters()
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


def _frontier_receipt(
    *,
    run_directory: Path,
    candidate_id: str,
    checkpoint: Path,
    model_state_sha256: str,
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
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
) -> dict[str, Any]:
    run_root = _new_run_directory(run_directory)
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
    recipe = resolve_lfm2_semantic_body_recipe(body_recipe)
    if recipe.recipe_sha256 != _digest(
        expected_body_recipe_sha256,
        "expected body recipe",
    ):
        raise Lfm2Exact4xCompositeFoldError("Body recipe differs from its pin.")
    if short_conv_construction not in LFM2_SEMANTIC_SHORT_CONV_CONSTRUCTION_IDS:
        raise Lfm2Exact4xCompositeFoldError(
            "Unknown semantic short-convolution construction."
        )

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
    catalog_preparation = read_lfm2_semantic_foundation_catalog_preparation(
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

    _emit("load-parent-direct-local-without-execution")
    parent = load_lfm2_direct_local_model(
        parent_path,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    parent.requires_grad_(False)
    parent.eval()
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
    parent_graph = inspect_model_circuits(
        parent,
        model_name=PARENT_MODEL_ROLE,
        tokenizer_fingerprint=bridge.donor_tokenizer_semantics_sha256,
    )
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
        dtype=torch.bfloat16,
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
        parent_models={PARENT_MODEL_ROLE: parent},
        parent_tokenizer_fingerprints={
            PARENT_MODEL_ROLE: bridge.donor_tokenizer_semantics_sha256
        },
    )
    child = target.model
    child.requires_grad_(False)
    child.eval()
    blank_state = fingerprint_model_state(child)
    topology = _target_topology(child)
    observed_topology = {
        key: topology[key] for key in EXPECTED_CHILD_TOPOLOGY
    }
    if (
        _unique_parameter_count(child) != EXPECTED_CHILD_UNIQUE_PARAMETERS
        or observed_topology != EXPECTED_CHILD_TOPOLOGY
        or target.blank_state_fingerprint != blank_state
        or target.blank_provenance is None
        or target.blank_provenance.compiler_created_blank is not True
        or target.blank_provenance.all_parameters_zero is not True
        or child.get_input_embeddings().weight
        is child.get_output_embeddings().weight
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
    interface_solve = fit_lfm2_semantic_foundation_interface(
        compiler_cache=compiler_cache,
        bridge=bridge,
        target_tokenizer=target_tokenizer,
        trajectory_corpus=trajectory_path,
        catalog_preparation=catalog_preparation,
        source_selected_tied_table=contextual_parent_rows,
        target_model=child,
    )
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
                str(index): dict(value.report)
                for index, value in typed_lineage.items()
            },
        },
    )

    _emit("fold-residual-attention-conv-mlp-body")
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
        global_residual_closure_version="producer_aware_v2a",
        correction_scale_frontier=(correction_scale,),
        correction_scale=correction_scale,
        global_residual_backend=global_residual_backend,
        global_residual_basis_rank=global_residual_basis_rank,
        target_dtype_representability_level=(
            target_dtype_representability_level
        ),
        body_recipe=recipe,
        short_conv_construction=short_conv_construction,
        interface_runtime_mode="independent_output",
    )
    if layer_stack.final_layer_mlp_down_weight_prior is None:
        raise Lfm2Exact4xCompositeFoldError(
            "Residual-aware body did not produce its deferred final W2 prior."
        )
    _write_json(run_root / "semantic-layer-stack.json", layer_stack.receipt)

    _emit("install-authenticated-deferred-final-w2")
    w2_install = dict(
        install_lfm2_deferred_final_w2_prior(
            child,
            layer_stack.final_layer_mlp_down_weight_prior,
            layer_stack_receipt=layer_stack.receipt,
        )
    )
    _write_json(run_root / "deferred-final-w2-install.json", w2_install)
    body_state = fingerprint_model_state(child)
    body_inventory_sha256 = fingerprint_lfm2_causal_body_inventory(child)
    input_table_sha256 = structural_receipt["installed_input_table_sha256"]
    if child.get_output_embeddings().weight is child.get_input_embeddings().weight:
        raise Lfm2Exact4xCompositeFoldError(
            "Body transaction unexpectedly retied the token interface."
        )

    _emit("save-and-reload-structural-body-base")
    checkpoint_io = Lfm2DirectLocalCheckpointIO(
        tokenizer_bridge=bridge,
        device=torch.device(model_device),
        dtype=torch.bfloat16,
    )
    base_checkpoint = run_root / "structural-body-checkpoint"
    base_checkpoint.mkdir()
    checkpoint_io.save(child, base_checkpoint)
    base_reload = checkpoint_io.load(base_checkpoint)
    if (
        fingerprint_model_state(base_reload) != body_state
        or _unique_parameter_count(base_reload) != EXPECTED_CHILD_UNIQUE_PARAMETERS
        or base_reload.get_input_embeddings().weight
        is base_reload.get_output_embeddings().weight
        or fingerprint_lfm2_causal_body_inventory(base_reload)
        != body_inventory_sha256
    ):
        raise Lfm2Exact4xCompositeFoldError(
            "Fresh structural-body reload changed the exact child coordinate."
        )
    del base_reload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
                if 0 <= int(value) < EXPECTED_CHILD_TOPOLOGY["vocab_size"]
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
            target_special_ids
            if bool(CAUSAL_HEAD_RECIPE["protect_specials"])
            else ()
        ),
        row_norm_calibration_ppm=int(
            CAUSAL_HEAD_RECIPE["row_norm_calibration_ppm"]
        ),
        solve_device=analytic_solve_device,
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

    _emit("save-reload-and-decode-complete-child")
    candidate_id = str(CAUSAL_HEAD_RECIPE["candidate_id"])
    candidate_checkpoint = run_root / "candidates" / candidate_id / "checkpoint"
    candidate_checkpoint.mkdir(parents=True)
    installed_state = fingerprint_model_state(child)
    checkpoint_io.save(child, candidate_checkpoint)
    reloaded = checkpoint_io.load(candidate_checkpoint)
    reload_state = fingerprint_model_state(reloaded)
    if (
        reload_state != installed_state
        or _unique_parameter_count(reloaded) != EXPECTED_CHILD_UNIQUE_PARAMETERS
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
    blank_target_receipt = target.as_dict()
    interface_solve_receipt = dict(interface_solve.report)
    layer_stack_receipt = dict(layer_stack.receipt)
    causal_bridge_receipt = causal_bridge.as_dict()
    if fingerprint_model_state(parent) != parent_state_before:
        raise Lfm2Exact4xCompositeFoldError(
            "Parent changed during the no-execution composite fold."
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

    frontier = _frontier_receipt(
        run_directory=run_root,
        candidate_id=candidate_id,
        checkpoint=candidate_checkpoint,
        model_state_sha256=installed_state,
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
        dtype="bfloat16",
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
    body = {
        "schema": RUN_SCHEMA,
        "status": "standalone-exact4x-composite-child-decoded",
        "run_directory": str(run_root),
        "parent_model_directory": str(parent_path),
        "parent_identity_sha256": expected_parent,
        "parent_unique_parameters": EXPECTED_PARENT_UNIQUE_PARAMETERS,
        "target_unique_parameters": EXPECTED_CHILD_UNIQUE_PARAMETERS,
        "realized_parameter_compaction_ratio": (
            EXPECTED_PARENT_UNIQUE_PARAMETERS / EXPECTED_CHILD_UNIQUE_PARAMETERS
        ),
        "target_topology": topology,
        "target_is_dense_untied_blank_architecture": True,
        "blank_target": blank_target_receipt,
        "blank_model_state_sha256": blank_state,
        "compiler_cache_directory": str(compiler_cache_path),
        "compiler_cache_key": compiler_cache.cache_key,
        "semantic_cache_directory": str(semantic_cache_path),
        "semantic_cache_key": semantic_cache.cache_key,
        "sealed_parent_cache_reused_without_recapture": True,
        "bridge_directory": str(bridge_path),
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "sequence_plan_sha256": sequence_plan["plan_sha256"],
        "terminal_cache_evidence_sha256": terminal_receipt["evidence_sha256"],
        "causal_trie_evidence_sha256": trie_receipt["result_sha256"],
        "foundation_interface_solve": interface_solve_receipt,
        "structural_interface_install": structural_receipt,
        "neutral_norm_install": neutral_norms,
        "attention_lineage_sha256": canonical_sha256(dict(raw_lineage)),
        "body_recipe": recipe.as_dict(),
        "body_recipe_sha256": recipe.recipe_sha256,
        "short_conv_construction": short_conv_construction,
        "semantic_layer_stack": layer_stack_receipt,
        "deferred_final_w2_install": w2_install,
        "structural_body_model_state_sha256": body_state,
        "causal_body_inventory_sha256": body_inventory_sha256,
        "child_causal_residual_evidence_sha256": child_evidence_receipt[
            "evidence_sha256"
        ],
        "causal_head_recipe": dict(CAUSAL_HEAD_RECIPE),
        "causal_token_bridge": causal_bridge_receipt,
        "causal_output_install": causal_install_receipt,
        "complete_child_model_state_sha256": installed_state,
        "candidate_id": candidate_id,
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_checkpoint_inventory": candidate_inventory,
        "candidate_checkpoint_inventory_sha256": canonical_sha256(
            candidate_inventory
        ),
        "standalone_fresh_reload_verified": True,
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
        "operation_audit": {
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "parameters_sparsified": 0,
            "external_adapter_parameters": 0,
            "parent_sidecar_parameters": 0,
            "runtime_shared_parent_parameters": 0,
            "network_access": False,
        },
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
        default="nonlinear_qdeim_coupled_attention_fit_corrected",
    )
    parser.add_argument("--expected-body-recipe-sha256", required=True)
    parser.add_argument(
        "--short-conv-construction",
        choices=LFM2_SEMANTIC_SHORT_CONV_CONSTRUCTION_IDS,
        default="full-source-structural-v1",
    )
    parser.add_argument("--blank-seed", type=int, default=0)
    parser.add_argument("--model-device", default="cuda:0")
    parser.add_argument("--fold-compute-device", default="cuda:0")
    parser.add_argument("--analytic-solve-device", default="cuda:0")
    parser.add_argument("--global-residual-backend", default="cuda")
    parser.add_argument("--global-residual-basis-rank", type=int, default=128)
    parser.add_argument("--correction-scale", type=float, default=1.0)
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
            expected_bridge_artifact_sha256=(
                args.expected_bridge_artifact_sha256
            ),
            compiler_cache_directory=args.compiler_cache_directory,
            expected_compiler_cache_key=args.expected_compiler_cache_key,
            semantic_cache_directory=args.semantic_cache_directory,
            expected_semantic_cache_key=args.expected_semantic_cache_key,
            trajectory_corpus=args.trajectory_corpus,
            selective_catalog_store_directory=(
                args.selective_catalog_store_directory
            ),
            selective_catalog_receipt_json=args.selective_catalog_receipt_json,
            expected_selective_catalog_receipt_sha256=(
                args.expected_selective_catalog_receipt_sha256
            ),
            expected_parent_identity_sha256=(
                args.expected_parent_identity_sha256
            ),
            expected_parent_unique_parameters=(
                args.expected_parent_unique_parameters
            ),
            prompts_json=args.prompts_json,
            expected_prompts_file_sha256=args.expected_prompts_file_sha256,
            parent_envelope=args.parent_envelope,
            expected_parent_envelope_sha256=(
                args.expected_parent_envelope_sha256
            ),
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
            global_residual_basis_rank=args.global_residual_basis_rank,
            correction_scale=args.correction_scale,
            target_dtype_representability_level=(
                args.target_dtype_representability_level
            ),
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


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
