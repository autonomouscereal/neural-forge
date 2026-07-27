#!/usr/bin/env python3
"""Reclose one pinned exact-4x LFM2 body through causal Q/K/O folding.

This is the production continuation of a successful exact-4x composite fold.
It loads only the saved child, authenticated local tokenizers/bridge, and the
immutable parent-evidence cache.  The installed terminal W2 is first recovered
as a prior and the archived deferred seed is restored.  The child then passes
through causal target-lattice Q/K, causal output projection folding, a fresh
source-local whole-stack reclosure, and exactly one corrected terminal-W2
publication.  Every serialized transition is freshly reloaded before it can
be consumed by the next compiler stage.

No parent model is constructed.  There is no network access, trace recapture,
training, autograd, optimizer, pruning, quantization, sparsity, parameter
sharing, or parent runtime sidecar.  Numerical diagnostics are retained as
evidence only.  Every decoded response is retained for manual intelligence
review and no numeric or manual score gates publication of the artifact.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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

from experiments import run_lfm2_exact4x_causal_head_frontier as frontier  # noqa: E402
from experiments import run_lfm2_exact4x_composite_fold as composite  # noqa: E402
from experiments import run_lfm2_exact4x_target_lattice_qk_treatment as qk_runner  # noqa: E402
from experiments.review_lfm2_semantic_frontier import (  # noqa: E402
    _load_parent_envelope,
    load_review_prompts,
    review_frontier,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_fold_executor import (  # noqa: E402
    fingerprint_model_state,
    fingerprint_model_state_entries,
)
from neural_forge.hf_lfm2_causal_post_qk_output_fold import (  # noqa: E402
    compile_install_lfm2_causal_post_qk_output_fold,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    fingerprint_lfm2_parent,
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_direct_local import (  # noqa: E402
    Lfm2DirectLocalCheckpointIO,
    load_lfm2_direct_local_tokenizer,
)
from neural_forge.hf_lfm2_interface_frontier import (  # noqa: E402
    fingerprint_lfm2_causal_body_inventory,
)
from neural_forge.hf_lfm2_post_causal_o_global_reclosure import (  # noqa: E402
    compile_install_lfm2_post_causal_o_source_local_global_reclosure,
)
from neural_forge.hf_lfm2_postfold_reclosure_seed_archive import (  # noqa: E402
    load_lfm2_postfold_reclosure_seed_archive,
)
from neural_forge.hf_lfm2_postfold_terminal_w2_reclosure import (  # noqa: E402
    install_lfm2_post_reclosure_terminal_w2,
    rebase_lfm2_installed_terminal_w2_for_output_reclosure,
)
from neural_forge.hf_lfm2_saved_body_target_lattice_qk import (  # noqa: E402
    LFM2_SAVED_BODY_TARGET_LATTICE_QK_SCHEMA,
    authenticate_lfm2_saved_attention_lineage,
    compile_install_lfm2_saved_body_target_lattice_qk_closure,
    extract_lfm2_qdeim_head_selections,
)
from neural_forge.hf_lfm2_structural_attention_candidate_archive import (  # noqa: E402
    load_lfm2_structural_attention_candidate_archive,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    fingerprint_hf_tokenizer_semantics,
    load_strict_tokenizer_bridge_artifact,
)


RUN_SCHEMA = "neural-forge-lfm2-exact4x-causal-reclosure-run-v1"
FAILURE_SCHEMA = "neural-forge-lfm2-exact4x-causal-reclosure-failure-v1"
PIPELINE_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-exact4x-causal-reclosure-stage-sequence-v1"
)
PIPELINE_STEP_SCHEMA = "neural-forge-lfm2-exact4x-causal-reclosure-stage-step-v1"
MANUAL_REVIEW_SCHEMA = "neural-forge-lfm2-exact4x-manual-review-pending-v1"
QK_PROVENANCE_SCHEMA = frontier.QK_TREATED_BODY_PROVENANCE_SCHEMA
EXPECTED_PARENT_UNIQUE_PARAMETERS = composite.EXPECTED_PARENT_UNIQUE_PARAMETERS
EXPECTED_CHILD_UNIQUE_PARAMETERS = composite.EXPECTED_CHILD_UNIQUE_PARAMETERS
EXPECTED_CHILD_TOPOLOGY = dict(composite.EXPECTED_CHILD_TOPOLOGY)
FINAL_CANDIDATE_ID = "exact4x-causal-qk-o-global-terminal-w2"

PIPELINE_STAGE_ORDER = (
    "authenticate-load-pinned-composite-source",
    "fresh-load-postfold-reclosure-seeds",
    "rebase-installed-terminal-w2-to-archived-seed",
    "save-fresh-reload-rebased-source",
    "compile-causal-target-lattice-qk",
    "save-fresh-reload-seal-qk-v2",
    "fresh-load-structural-attention-candidates",
    "compile-causal-post-qk-output-projections",
    "compile-post-o-source-local-global-reclosure",
    "install-corrected-terminal-w2",
    "save-fresh-reload-standalone-child",
    "decode-retain-whole-responses-and-manual-review",
)

FORBIDDEN_OPERATION_AUDIT = {
    "training_steps": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "autograd_grad_calls": 0,
    "backward_calls": 0,
    "parent_model_objects_constructed": 0,
    "parent_model_loads": 0,
    "parent_model_forward_calls": 0,
    "parent_model_generate_calls": 0,
    "parent_trace_recaptures": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
    "parameters_sparsified": 0,
    "parameter_sharing_used": False,
    "parent_runtime_sidecar_used": False,
    "network_access": False,
}


class Lfm2Exact4xCausalReclosureError(RuntimeError):
    """A local artifact, causal transition, or fresh reload failed closed."""


@dataclass(frozen=True, slots=True)
class _StageOutcome:
    value: Any
    evidence: Mapping[str, Any]


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2Exact4xCausalReclosureError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _emit(stage: str, **values: Any) -> None:
    print(
        json.dumps({"stage": stage, **values}, allow_nan=False, sort_keys=True),
        flush=True,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sealed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    body = dict(value)
    if field in body:
        raise Lfm2Exact4xCausalReclosureError(
            f"Cannot seal evidence which already contains {field!r}."
        )
    return {**body, field: canonical_sha256(body)}


def _verify_embedded_seal(
    value: Any,
    *,
    seal_field: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2Exact4xCausalReclosureError(f"{label} is not one mapping.")
    document = dict(value)
    body = dict(document)
    supplied = body.pop(seal_field, None)
    if supplied != canonical_sha256(body):
        raise Lfm2Exact4xCausalReclosureError(
            f"{label} is stale or not canonically sealed."
        )
    return document


def _load_pinned_run_receipt(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_result_sha256: str,
) -> dict[str, Any]:
    source = frontier._ordinary_file(path, "source composite run receipt")
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != _digest(
        expected_file_sha256,
        "expected source composite run file",
    ):
        raise Lfm2Exact4xCausalReclosureError(
            "Source composite run receipt differs from its external file pin."
        )
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2Exact4xCausalReclosureError(
            "Source composite run receipt is not valid UTF-8 JSON."
        ) from exc
    receipt = _verify_embedded_seal(
        value,
        seal_field="result_sha256",
        label="source composite run receipt",
    )
    if (
        receipt.get("schema") != composite.RUN_SCHEMA
        or receipt.get("result_sha256")
        != _digest(
            expected_result_sha256,
            "expected source composite run result",
        )
        or receipt.get("status") != "standalone-exact4x-composite-child-decoded"
    ):
        raise Lfm2Exact4xCausalReclosureError(
            "Source run is not the externally pinned successful exact-4x composite."
        )
    return receipt


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Lfm2Exact4xCausalReclosureError(
            f"Refusing to replace existing evidence: {path}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )


def _run_ordered_stage_machine(
    handlers: Mapping[str, Callable[[Mapping[str, Any]], _StageOutcome]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute every required stage once, in the immutable causal order."""

    if set(handlers) != set(PIPELINE_STAGE_ORDER):
        missing = sorted(set(PIPELINE_STAGE_ORDER) - set(handlers))
        extra = sorted(set(handlers) - set(PIPELINE_STAGE_ORDER))
        raise Lfm2Exact4xCausalReclosureError(
            f"Pipeline handler inventory changed; missing={missing}, extra={extra}."
        )
    state: dict[str, Any] = {}
    steps: list[dict[str, Any]] = []
    for sequence_number, stage in enumerate(PIPELINE_STAGE_ORDER):
        _emit(stage, sequence_number=sequence_number)
        outcome = handlers[stage](state)
        if not isinstance(outcome, _StageOutcome) or not isinstance(
            outcome.evidence,
            Mapping,
        ):
            raise Lfm2Exact4xCausalReclosureError(
                f"Pipeline stage {stage!r} returned no typed evidence."
            )
        evidence = dict(outcome.evidence)
        evidence_sha256 = canonical_sha256(evidence)
        step = _sealed(
            {
                "schema": PIPELINE_STEP_SCHEMA,
                "stage": stage,
                "sequence_number": sequence_number,
                "completed_predecessors": list(
                    PIPELINE_STAGE_ORDER[:sequence_number]
                ),
                "evidence_sha256": evidence_sha256,
            },
            "result_sha256",
        )
        state[stage] = outcome.value
        steps.append(step)
    body = {
        "schema": PIPELINE_RECEIPT_SCHEMA,
        "stage_order": list(PIPELINE_STAGE_ORDER),
        "stage_count": len(PIPELINE_STAGE_ORDER),
        "stage_steps": steps,
        "stage_step_sha256": [value["result_sha256"] for value in steps],
        "every_required_stage_executed_once": True,
        "terminal_w2_published_only_after_qk_o_and_global_reclosure": True,
        "behavior_review_occurs_only_after_final_fresh_reload": True,
    }
    return state, _sealed(body, "result_sha256")


def _uniform_fp32(model: Any) -> None:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - required runtime
        raise RuntimeError("PyTorch is required for exact-4x reclosure.") from exc
    dtypes = {parameter.dtype for parameter in model.parameters()}
    if dtypes != {torch.float32}:
        raise Lfm2Exact4xCausalReclosureError(
            "The exact-4x reclosure pipeline requires one uniform FP32 target."
        )


def _parameter_count(model: Any) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def _expected_topology_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value.get(name) for name in EXPECTED_CHILD_TOPOLOGY}


def _assert_dense_unshared_exact4x(model: Any) -> None:
    _uniform_fp32(model)
    named = tuple(model.named_parameters(remove_duplicate=False))
    if (
        _parameter_count(model) != EXPECTED_CHILD_UNIQUE_PARAMETERS
        or _expected_topology_projection(frontier._target_topology(model))
        != EXPECTED_CHILD_TOPOLOGY
        or len({id(parameter) for _, parameter in named}) != len(named)
        or len({int(parameter.data_ptr()) for _, parameter in named}) != len(named)
        or model.get_input_embeddings().weight is model.get_output_embeddings().weight
    ):
        raise Lfm2Exact4xCausalReclosureError(
            "Live target is not the dense, unshared, untied exact-4x FP32 child."
        )


def _authenticate_coordinate_completion_and_direction(
    *,
    semantic_stack: Mapping[str, Any],
    body_realization: Mapping[str, Any],
) -> dict[str, Any]:
    wrapper = _verify_embedded_seal(
        semantic_stack.get("body_global_coordinate_completion_receipt"),
        seal_field="result_sha256",
        label="body-global coordinate-completion wrapper",
    )
    candidate = _verify_embedded_seal(
        semantic_stack.get("body_global_coordinate_completion_candidate_receipt"),
        seal_field="result_sha256",
        label="body-global coordinate-completion candidate",
    )
    required_rank = candidate.get("required_full_rank")
    if (
        semantic_stack.get("body_global_coordinate_completion_enabled") is not True
        or semantic_stack.get("body_global_coordinate_completion_compile_count") != 1
        or semantic_stack.get("body_global_completion_shared_across_every_layer")
        is not True
        or semantic_stack.get(
            "body_global_completion_shared_across_attention_mlp_and_short_conv"
        )
        is not True
        or wrapper.get("selected_candidate_result_sha256")
        != candidate["result_sha256"]
        or isinstance(required_rank, bool)
        or not isinstance(required_rank, int)
        or required_rank <= 0
        or candidate.get("structural_rank") != required_rank
        or candidate.get("materialized_rank") != required_rank
        or candidate.get("full_rank_materialized") is not True
        or body_realization.get("child_materialization_dtype") != "float32"
        or body_realization.get("selected_representability_level") != "requested"
        or body_realization.get(
            "requested_unit_ray_preserved_without_ulp_escalation"
        )
        is not True
        or body_realization.get("materialized_ray_multiplier") != 1.0
        or body_realization.get("target_dtype_ulp_escalation_applied") is not False
    ):
        raise Lfm2Exact4xCausalReclosureError(
            "Source body lacks full-rank completion or unit-ray FP32 materialization."
        )
    return {
        "coordinate_completion_sha256": wrapper["result_sha256"],
        "coordinate_completion_candidate_sha256": candidate["result_sha256"],
        "required_full_rank": required_rank,
        "full_rank_materialized": True,
        "body_realization_sha256": body_realization["receipt_sha256"],
        "requested_unit_ray_preserved": True,
    }


def _assert_qk_treatment(
    *,
    treatment: Mapping[str, Any],
    before_entries: Mapping[str, str],
    target_model: Any,
) -> tuple[list[str], str, str, str]:
    after_entries = dict(fingerprint_model_state_entries(target_model))
    changed = qk_runner._changed_entries(before_entries, after_entries)
    state = fingerprint_model_state(target_model)
    identity = fingerprint_lfm2_parent(target_model).identity_sha256
    inventory = fingerprint_lfm2_causal_body_inventory(target_model)
    if (
        treatment.get("schema") != LFM2_SAVED_BODY_TARGET_LATTICE_QK_SCHEMA
        or treatment.get("model_state_after_sha256") != identity
        or treatment.get("changed_state_keys") != changed
        or treatment.get("actual_changed_state_keys_are_qk_only") is not True
        or treatment.get("all_non_qk_tensors_byte_identical") is not True
        or treatment.get("parameter_dtype") != "float32"
        or treatment.get("checkpoint_dtype_preserved") is not True
        or treatment.get("parameter_count_changed") is not False
        or not changed
    ):
        raise Lfm2Exact4xCausalReclosureError(
            "Causal target-lattice Q/K treatment is incomplete or changed non-Q/K state."
        )
    _assert_dense_unshared_exact4x(target_model)
    return changed, state, identity, inventory


def _final_w2_sha256(model: Any, parameter_name: str) -> str:
    from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (
        lfm2_cpcf_terminal_action_tensor_sha256,
    )

    named = dict(model.named_parameters())
    if parameter_name not in named:
        raise Lfm2Exact4xCausalReclosureError(
            "Fresh checkpoint lost the authenticated terminal W2 parameter."
        )
    return lfm2_cpcf_terminal_action_tensor_sha256(named[parameter_name])


def _save_and_fresh_reload(
    *,
    model: Any,
    checkpoint: Path,
    checkpoint_io: Any,
) -> tuple[Any, dict[str, Any]]:
    checkpoint.mkdir(parents=True)
    _assert_dense_unshared_exact4x(model)
    state_before = fingerprint_model_state(model)
    entries_before = dict(fingerprint_model_state_entries(model))
    identity_before = fingerprint_lfm2_parent(model)
    body_before = fingerprint_lfm2_causal_body_inventory(model)
    checkpoint_io.save(model, checkpoint)
    inventory = frontier._artifact_inventory(checkpoint)
    inventory_sha256 = canonical_sha256(inventory)
    frontier._authenticate_source_checkpoint_inventory(
        checkpoint,
        expected_sha256=inventory_sha256,
    )
    reloaded = checkpoint_io.load(checkpoint)
    reloaded.requires_grad_(False)
    reloaded.eval()
    _assert_dense_unshared_exact4x(reloaded)
    if (
        fingerprint_model_state(reloaded) != state_before
        or dict(fingerprint_model_state_entries(reloaded)) != entries_before
        or fingerprint_lfm2_parent(reloaded).as_dict() != identity_before.as_dict()
        or fingerprint_lfm2_causal_body_inventory(reloaded) != body_before
    ):
        raise Lfm2Exact4xCausalReclosureError(
            "Standalone checkpoint fresh reload changed target state or topology."
        )
    receipt = {
        "checkpoint": str(checkpoint),
        "checkpoint_inventory": inventory,
        "checkpoint_inventory_sha256": inventory_sha256,
        "model_state_sha256": state_before,
        "lfm2_identity_sha256": identity_before.identity_sha256,
        "lfm2_config_sha256": identity_before.config_sha256,
        "causal_body_inventory_sha256": body_before,
        "parameter_dtype": "float32",
        "parameter_element_count": EXPECTED_CHILD_UNIQUE_PARAMETERS,
        "standalone_fresh_reload_verified": True,
    }
    return reloaded, receipt


def _manual_review_artifact(
    *,
    dossier: Mapping[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    reviews = dossier.get("candidate_reviews")
    if not isinstance(reviews, list) or len(reviews) != 1:
        raise Lfm2Exact4xCausalReclosureError(
            "Whole-response dossier has no single retained candidate review."
        )
    responses = reviews[0].get("responses")
    if not isinstance(responses, list):
        raise Lfm2Exact4xCausalReclosureError(
            "Whole-response dossier omitted raw responses."
        )
    body = {
        "schema": MANUAL_REVIEW_SCHEMA,
        "status": "pending-human-whole-response-intelligence-review",
        "candidate_id": candidate_id,
        "whole_response_dossier_sha256": dossier.get("dossier_sha256"),
        "response_count": len(responses),
        "review_items": [
            {
                "response_index": index,
                "raw_response": dict(response),
                "coherent": None,
                "useful": None,
                "parent_relative_intelligence_retained": None,
                "reviewer_notes": None,
            }
            for index, response in enumerate(responses)
        ],
        "numeric_metric_threshold_used_as_gate": False,
        "manual_score_used_as_compilation_gate": False,
        "automatic_acceptance_or_rejection_performed": False,
        "whole_decoded_behavior_is_primary_intelligence_evidence": True,
    }
    return _sealed(body, "review_artifact_sha256")


def run(
    *,
    run_directory: Path,
    source_run_directory: Path,
    expected_source_run_result_sha256: str,
    expected_source_run_file_sha256: str,
    donor_tokenizer_directory: Path,
    target_tokenizer_directory: Path,
    bridge_directory: Path,
    expected_bridge_artifact_sha256: str,
    semantic_cache_directory: Path,
    expected_semantic_cache_key: str,
    expected_parent_identity_sha256: str,
    expected_attention_lineage_file_sha256: str,
    prompts_json: Path,
    expected_prompts_file_sha256: str,
    parent_envelope: Path,
    expected_parent_envelope_sha256: str,
    expected_parent_envelope_file_sha256: str,
    model_device: str,
    fold_compute_device: str,
    o_ridge: float,
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - required runtime
        raise RuntimeError("PyTorch is required for exact-4x reclosure.") from exc

    run_root = frontier._new_run_directory(run_directory)
    source_root = frontier._ordinary_directory(source_run_directory, "source run")
    donor_path = frontier._ordinary_directory(
        donor_tokenizer_directory,
        "donor tokenizer",
    )
    target_path = frontier._ordinary_directory(
        target_tokenizer_directory,
        "target tokenizer",
    )
    bridge_path = frontier._ordinary_directory(bridge_directory, "tokenizer bridge")
    cache_path = frontier._ordinary_directory(
        semantic_cache_directory,
        "semantic cache",
    )
    prompts_path = frontier._ordinary_file(prompts_json, "review prompts")
    envelope_path = frontier._ordinary_file(parent_envelope, "parent envelope")
    expected_parent = _digest(
        expected_parent_identity_sha256,
        "expected parent identity",
    )
    expected_bridge = _digest(
        expected_bridge_artifact_sha256,
        "expected bridge artifact",
    )
    expected_cache = _digest(
        expected_semantic_cache_key,
        "expected semantic cache key",
    )
    expected_prompts = _digest(
        expected_prompts_file_sha256,
        "expected prompts file",
    )
    if _file_sha256(prompts_path) != expected_prompts:
        raise Lfm2Exact4xCausalReclosureError(
            "Review prompts differ from their external file pin."
        )
    prompt_rows = [dict(value) for value in load_review_prompts(prompts_path)]
    if not prompt_rows or maximum_prompt_tokens <= 0 or maximum_new_tokens <= 0:
        raise Lfm2Exact4xCausalReclosureError(
            "Whole-response evaluation must retain nonempty uncapped review work."
        )
    _load_parent_envelope(
        envelope_path,
        expected_envelope_sha256=_digest(
            expected_parent_envelope_sha256,
            "expected parent envelope",
        ),
        expected_file_sha256=_digest(
            expected_parent_envelope_file_sha256,
            "expected parent envelope file",
        ),
        expected_prompt_set_sha256=canonical_sha256(prompt_rows),
        expected_prompt_ids=[str(row["id"]) for row in prompt_rows],
    )

    donor_tokenizer = load_lfm2_direct_local_tokenizer(donor_path)
    target_tokenizer = load_lfm2_direct_local_tokenizer(target_path)
    bridge = load_strict_tokenizer_bridge_artifact(
        bridge_path,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        expected_artifact_sha256=expected_bridge,
    )
    cache = open_lfm2_circuit_trace_cache(cache_path, verify=False)
    if (
        cache.cache_key != expected_cache
        or cache.parent_fingerprint != expected_parent
        or fingerprint_hf_tokenizer_semantics(donor_tokenizer)
        != bridge.donor_tokenizer_semantics_sha256
        or fingerprint_hf_tokenizer_semantics(target_tokenizer)
        != bridge.target_tokenizer_semantics_sha256
    ):
        raise Lfm2Exact4xCausalReclosureError(
            "Cache, bridge, tokenizer, or parent ancestry changed."
        )
    checkpoint_io = Lfm2DirectLocalCheckpointIO(
        tokenizer_bridge=bridge,
        device=torch.device(model_device),
        dtype=torch.float32,
    )

    shared: dict[str, Any] = {}

    def source_stage(_: Mapping[str, Any]) -> _StageOutcome:
        source_run = _load_pinned_run_receipt(
            source_root / "run-receipt.json",
            expected_file_sha256=expected_source_run_file_sha256,
            expected_result_sha256=expected_source_run_result_sha256,
        )
        structural_body = _verify_embedded_seal(
            source_run.get("structural_body_checkpoint_receipt"),
            seal_field="receipt_sha256",
            label="embedded structural-body checkpoint receipt",
        )
        structural_body_file = frontier._load_sealed_json(
            source_root / "structural-body-receipt.json",
            seal_field="receipt_sha256",
            expected_seal=structural_body["receipt_sha256"],
            expected_schema=composite.STRUCTURAL_BODY_RECEIPT_SCHEMA,
            label="source structural-body receipt",
        )
        if structural_body_file != structural_body:
            raise Lfm2Exact4xCausalReclosureError(
                "Embedded and filesystem structural-body receipts disagree."
            )
        structural = frontier._load_sealed_json(
            source_root / "structural-interface-install.json",
            seal_field="result_sha256",
            expected_seal=source_run["structural_interface_install"][
                "result_sha256"
            ],
            expected_schema=frontier.LFM2_INDEPENDENT_STRUCTURAL_INTERFACE_INSTALL_SCHEMA,
            label="source structural-interface receipt",
        )
        stack = frontier._load_sealed_json(
            source_root / "semantic-layer-stack.json",
            seal_field="result_sha256",
            expected_seal=source_run["semantic_layer_stack"]["result_sha256"],
            expected_schema=frontier.SEMANTIC_LAYER_STACK_SCHEMA,
            label="source semantic layer-stack receipt",
        )
        deferred = frontier._load_sealed_json(
            source_root / "deferred-final-w2-install.json",
            seal_field="result_sha256",
            expected_seal=source_run["deferred_final_w2_install"]["result_sha256"],
            expected_schema=frontier.DEFERRED_W2_SCHEMA,
            label="source deferred final-W2 receipt",
        )
        frontier._authenticate_source_fold_receipts(
            structural_receipt=structural,
            semantic_layer_stack_receipt=stack,
            deferred_w2_receipt=deferred,
        )
        foundation = _verify_embedded_seal(
            source_run.get("foundation_interface_solve"),
            seal_field="result_sha256",
            label="source foundation-interface solve",
        )
        body_realization = frontier._load_sealed_json(
            source_root / "body-fold-realization.json",
            seal_field="receipt_sha256",
            expected_seal=structural_body["body_fold_realization_sha256"],
            expected_schema=composite.BODY_FOLD_REALIZATION_SCHEMA,
            label="source body-fold realization",
        )
        completion = _authenticate_coordinate_completion_and_direction(
            semantic_stack=stack,
            body_realization=body_realization,
        )
        if (
            source_run.get("parent_unique_parameters")
            != EXPECTED_PARENT_UNIQUE_PARAMETERS
            or source_run.get("target_unique_parameters")
            != EXPECTED_CHILD_UNIQUE_PARAMETERS
            or not isinstance(source_run.get("target_topology"), Mapping)
            or _expected_topology_projection(source_run["target_topology"])
            != EXPECTED_CHILD_TOPOLOGY
            or source_run.get("child_parameter_dtype") != "float32"
            or source_run.get("checkpoint_reload_dtype") != "float32"
            or source_run.get("target_is_dense_untied_blank_architecture")
            is not True
            or source_run.get("float32_storage_is_not_quantization") is not True
            or structural_body.get("postfold_reclosure_seed_archive_fresh_reload_verified")
            is not True
            or structural_body.get(
                "structural_attention_candidate_archive_fresh_reload_verified"
            )
            is not True
        ):
            raise Lfm2Exact4xCausalReclosureError(
                "Pinned source is not the dense FP32 exact-4x composite body."
            )
        checkpoint = frontier._ordinary_directory(
            source_root / "structural-body-checkpoint",
            "source structural-body checkpoint",
        )
        inventory = frontier._authenticate_source_checkpoint_inventory(
            checkpoint,
            expected_sha256=structural_body["checkpoint_inventory_sha256"],
        )
        model = checkpoint_io.load(checkpoint)
        model.requires_grad_(False)
        model.eval()
        _assert_dense_unshared_exact4x(model)
        if (
            fingerprint_model_state(model)
            != structural_body["body_model_state_sha256"]
            or fingerprint_lfm2_parent(model).identity_sha256
            != structural_body["body_lfm2_identity_sha256"]
            or fingerprint_lfm2_causal_body_inventory(model)
            != structural_body["causal_body_inventory_sha256"]
        ):
            raise Lfm2Exact4xCausalReclosureError(
                "Fresh source checkpoint differs from its structural-body receipt."
            )
        deferred_authentication = frontier._authenticate_loaded_deferred_final_w2(
            model,
            deferred,
        )
        attention_document = qk_runner._load_pinned_json(
            source_root / "attention-lineage.json",
            expected_file_sha256=expected_attention_lineage_file_sha256,
            label="source attention lineage",
        )
        compiler_lineage = authenticate_lfm2_saved_attention_lineage(
            attention_document
        )
        selections = extract_lfm2_qdeim_head_selections(stack)
        value = {
            "run": source_run,
            "structural_body": structural_body,
            "structural": structural,
            "stack": stack,
            "deferred": deferred,
            "foundation": foundation,
            "body_realization": body_realization,
            "model": model,
            "checkpoint_inventory": inventory,
            "deferred_authentication": deferred_authentication,
            "compiler_lineage": compiler_lineage,
            "qdeim_selections": selections,
        }
        shared["source"] = value
        return _StageOutcome(
            value=value,
            evidence={
                "source_run_result_sha256": source_run["result_sha256"],
                "source_checkpoint_inventory_sha256": structural_body[
                    "checkpoint_inventory_sha256"
                ],
                "source_body_identity_sha256": structural_body[
                    "body_lfm2_identity_sha256"
                ],
                "source_semantic_stack_sha256": stack["result_sha256"],
                "source_deferred_w2_sha256": deferred["result_sha256"],
                **completion,
            },
        )

    def seed_stage(state: Mapping[str, Any]) -> _StageOutcome:
        source = state[PIPELINE_STAGE_ORDER[0]]
        expected_manifest = source["structural_body"][
            "postfold_reclosure_seed_archive_manifest_sha256"
        ]
        archive, manifest = load_lfm2_postfold_reclosure_seed_archive(
            source_root / "postfold-reclosure-seeds",
            foundation_interface_solve_receipt=source["foundation"],
            semantic_layer_stack_receipt=source["stack"],
            deferred_final_w2_installation_receipt=source["deferred"],
            expected_manifest_sha256=expected_manifest,
        )
        if manifest.get("fresh_reload_verified") is not True:
            raise Lfm2Exact4xCausalReclosureError(
                "Postfold reclosure seeds were not freshly authenticated."
            )
        value = {"archive": archive, "manifest": manifest}
        shared["seeds"] = value
        return _StageOutcome(
            value=value,
            evidence={
                "manifest_sha256": manifest["manifest_sha256"],
                "tensor_file_sha256": manifest["tensor_file_sha256"],
                "fresh_reload_verified": True,
            },
        )

    def rebase_stage(state: Mapping[str, Any]) -> _StageOutcome:
        source = state[PIPELINE_STAGE_ORDER[0]]
        seeds = state[PIPELINE_STAGE_ORDER[1]]
        rebase = rebase_lfm2_installed_terminal_w2_for_output_reclosure(
            source["model"],
            semantic_layer_stack_receipt=source["stack"],
            deferred_final_w2_installation_receipt=source["deferred"],
            original_deferred_w2=seeds["archive"].deferred_final_w2_seed,
        )
        if not rebase.verify():
            raise Lfm2Exact4xCausalReclosureError(
                "Terminal-W2 rebase failed authentication."
            )
        _write_json(run_root / "terminal-w2-rebase.json", rebase.receipt)
        shared["rebase"] = rebase
        return _StageOutcome(
            value=rebase,
            evidence={
                "rebase_result_sha256": rebase.receipt["result_sha256"],
                "terminal_prior_authentication_sha256": (
                    rebase.terminal_prior.authentication_sha256
                ),
                "restored_live_w2_sha256": rebase.receipt[
                    "restored_live_w2_sha256"
                ],
                "model_state_after_sha256": rebase.receipt[
                    "model_state_after_sha256"
                ],
            },
        )

    def rebase_reload_stage(state: Mapping[str, Any]) -> _StageOutcome:
        source = state[PIPELINE_STAGE_ORDER[0]]
        rebase = state[PIPELINE_STAGE_ORDER[2]]
        model, receipt = _save_and_fresh_reload(
            model=source["model"],
            checkpoint=run_root / "rebased-source" / "checkpoint",
            checkpoint_io=checkpoint_io,
        )
        final_name = rebase.receipt["final_w2_parameter_name"]
        if (
            receipt["lfm2_identity_sha256"]
            != rebase.receipt["model_state_after_sha256"]
            or _final_w2_sha256(model, final_name)
            != rebase.receipt["restored_live_w2_sha256"]
        ):
            raise Lfm2Exact4xCausalReclosureError(
                "Fresh rebased checkpoint lost its deferred terminal-W2 seed."
            )
        source["model"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        value = {"model": model, "receipt": receipt}
        shared["rebased"] = value
        return _StageOutcome(value=value, evidence=receipt)

    def qk_stage(state: Mapping[str, Any]) -> _StageOutcome:
        source = state[PIPELINE_STAGE_ORDER[0]]
        rebased = state[PIPELINE_STAGE_ORDER[3]]
        model = rebased["model"]
        before_entries = dict(fingerprint_model_state_entries(model))
        treatment = dict(
            compile_install_lfm2_saved_body_target_lattice_qk_closure(
                cache=cache,
                bridge=bridge,
                target_model=model,
                compiler_lineage=source["compiler_lineage"],
                qdeim_head_selections=source["qdeim_selections"],
                donor_tokenizer=donor_tokenizer,
                target_tokenizer=target_tokenizer,
                authenticated_cache_session=shared["cache_session"],
            )
        )
        changed, model_state, identity, body_inventory = _assert_qk_treatment(
            treatment=treatment,
            before_entries=before_entries,
            target_model=model,
        )
        _write_json(run_root / "target-lattice-qk-treatment.json", treatment)
        value = {
            "model": model,
            "treatment": treatment,
            "changed_state_keys": changed,
            "model_state_sha256": model_state,
            "identity_sha256": identity,
            "body_inventory_sha256": body_inventory,
        }
        shared["qk"] = value
        return _StageOutcome(
            value=value,
            evidence={
                "treatment_result_sha256": treatment["result_sha256"],
                "changed_state_keys": changed,
                "model_state_after_sha256": identity,
                "parameter_dtype": "float32",
                "qk_only": True,
            },
        )

    def qk_reload_stage(state: Mapping[str, Any]) -> _StageOutcome:
        rebased = state[PIPELINE_STAGE_ORDER[3]]
        qk = state[PIPELINE_STAGE_ORDER[4]]
        model, receipt = _save_and_fresh_reload(
            model=qk["model"],
            checkpoint=run_root / "qk-body" / "checkpoint",
            checkpoint_io=checkpoint_io,
        )
        changed_paths = frontier._changed_checkpoint_paths(
            rebased["receipt"]["checkpoint_inventory"],
            receipt["checkpoint_inventory"],
        )
        if changed_paths != ["model.safetensors"]:
            raise Lfm2Exact4xCausalReclosureError(
                "Q/K checkpoint serialization changed non-weight assets."
            )
        source_basis_body = {
            "schema": "neural-forge-lfm2-exact4x-rebased-source-basis-v1",
            "source_run_result_sha256": shared["source"]["run"]["result_sha256"],
            "postfold_reclosure_seed_manifest_sha256": shared["seeds"][
                "manifest"
            ]["manifest_sha256"],
            "terminal_rebase_result_sha256": shared["rebase"].receipt[
                "result_sha256"
            ],
            "rebased_checkpoint_inventory_sha256": rebased["receipt"][
                "checkpoint_inventory_sha256"
            ],
            "rebased_body_model_state_sha256": rebased["receipt"][
                "model_state_sha256"
            ],
            "rebased_body_lfm2_identity_sha256": rebased["receipt"][
                "lfm2_identity_sha256"
            ],
            "standalone_fresh_reload_verified": True,
        }
        source_basis = _sealed(source_basis_body, "basis_sha256")
        provenance_body = {
            "schema": QK_PROVENANCE_SCHEMA,
            "status": "saved-reloaded-qk-only-treated-body",
            "source_body_authentication_basis": source_basis,
            "source_body_authentication_basis_sha256": source_basis["basis_sha256"],
            "source_structural_body_receipt_sha256": shared["source"][
                "structural_body"
            ]["receipt_sha256"],
            "source_checkpoint_inventory_sha256": rebased["receipt"][
                "checkpoint_inventory_sha256"
            ],
            "source_checkpoint_dtype": "float32",
            "source_body_model_state_sha256": rebased["receipt"][
                "model_state_sha256"
            ],
            "source_body_lfm2_identity_sha256": rebased["receipt"][
                "lfm2_identity_sha256"
            ],
            "source_causal_body_inventory_sha256": rebased["receipt"][
                "causal_body_inventory_sha256"
            ],
            "source_deferred_final_w2_authentication": {
                "terminal_rebase_result_sha256": shared["rebase"].receipt[
                    "result_sha256"
                ],
                "restored_live_w2_sha256": shared["rebase"].receipt[
                    "restored_live_w2_sha256"
                ],
            },
            "treated_checkpoint_inventory_sha256": receipt[
                "checkpoint_inventory_sha256"
            ],
            "treated_checkpoint_dtype": "float32",
            "checkpoint_dtype_preserved": True,
            "treated_body_model_state_sha256": receipt["model_state_sha256"],
            "treated_body_lfm2_identity_sha256": receipt["lfm2_identity_sha256"],
            "treated_causal_body_inventory_sha256": receipt[
                "causal_body_inventory_sha256"
            ],
            "changed_checkpoint_relative_paths": changed_paths,
            "changed_state_keys": qk["changed_state_keys"],
            "target_lattice_qk_treatment_sha256": qk["treatment"][
                "result_sha256"
            ],
            "terminal_w2_rebased_before_qk": True,
            "terminal_w2_remains_deferred": True,
            "causal_output_head_installed": False,
            "standalone_fresh_reload_verified": True,
            "operation_audit": dict(frontier.QK_TREATED_BODY_ZERO_OPERATION_AUDIT),
        }
        provenance = _sealed(provenance_body, "receipt_sha256")
        _write_json(
            run_root / "qk-body" / "treated-body-provenance.json",
            provenance,
        )
        qk["model"] = None
        rebased["model"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        value = {"model": model, "receipt": receipt, "provenance": provenance}
        shared["qk_reload"] = value
        return _StageOutcome(
            value=value,
            evidence={
                **receipt,
                "qk_fresh_reload_provenance_sha256": provenance["receipt_sha256"],
                "changed_checkpoint_relative_paths": changed_paths,
            },
        )

    def candidates_stage(state: Mapping[str, Any]) -> _StageOutcome:
        source = state[PIPELINE_STAGE_ORDER[0]]
        expected_manifest = source["structural_body"][
            "structural_attention_candidate_archive_manifest_sha256"
        ]
        candidates, manifest = load_lfm2_structural_attention_candidate_archive(
            source_root / "structural-attention-candidates",
            semantic_layer_stack_receipt=source["stack"],
            expected_manifest_sha256=expected_manifest,
        )
        if manifest.get("fresh_reload_verified") is not True:
            raise Lfm2Exact4xCausalReclosureError(
                "Structural attention candidates were not freshly authenticated."
            )
        value = {"candidates": candidates, "manifest": manifest}
        shared["candidates"] = value
        return _StageOutcome(
            value=value,
            evidence={
                "manifest_sha256": manifest["manifest_sha256"],
                "attention_layer_indices": manifest["attention_layer_indices"],
                "fresh_reload_verified": True,
            },
        )

    def causal_o_stage(state: Mapping[str, Any]) -> _StageOutcome:
        source = state[PIPELINE_STAGE_ORDER[0]]
        qk = state[PIPELINE_STAGE_ORDER[4]]
        qk_reload = state[PIPELINE_STAGE_ORDER[5]]
        candidates = state[PIPELINE_STAGE_ORDER[6]]
        installation = compile_install_lfm2_causal_post_qk_output_fold(
            cache=cache,
            bridge=bridge,
            donor_tokenizer=donor_tokenizer,
            target_tokenizer=target_tokenizer,
            target_model=qk_reload["model"],
            qk_treatment_receipt=qk["treatment"],
            qk_fresh_reload_provenance_receipt=qk_reload["provenance"],
            expected_qk_fresh_reload_provenance_sha256=qk_reload["provenance"][
                "receipt_sha256"
            ],
            semantic_layer_stack_receipt=source["stack"],
            expected_semantic_layer_stack_receipt_sha256=source["stack"][
                "result_sha256"
            ],
            structural_candidates=candidates["candidates"],
            structural_candidate_archive_manifest=candidates["manifest"],
            expected_structural_candidate_archive_manifest_sha256=candidates[
                "manifest"
            ]["manifest_sha256"],
            authenticated_cache_session=shared["cache_session"],
            ridge=o_ridge,
            fold_compute_device=fold_compute_device,
        )
        if not installation.verify():
            raise Lfm2Exact4xCausalReclosureError(
                "Causal post-Q/K O installation failed authentication."
            )
        direction_receipts = []
        for layer in installation.layer_receipts:
            direction = _verify_embedded_seal(
                layer.get("direction_materialization_receipt"),
                seal_field="result_sha256",
                label=f"layer {layer.get('layer_index')} O direction receipt",
            )
            if not (
                direction.get("direction_preserved") is True
                or direction.get(
                    "direction_preservation_not_applicable_to_zero_published_delta"
                )
                is True
            ):
                raise Lfm2Exact4xCausalReclosureError(
                    "An O materialization lost its analytic correction direction."
                )
            direction_receipts.append(direction["result_sha256"])
        _write_json(run_root / "causal-post-qk-o.json", installation.receipt)
        shared["causal_o"] = installation
        return _StageOutcome(
            value=installation,
            evidence={
                "causal_o_result_sha256": installation.receipt["result_sha256"],
                "direction_materialization_receipt_sha256": direction_receipts,
                "causal_downstream_replay_after_each_o_install": True,
                "qkv_byte_identical_before_and_after": True,
            },
        )

    def global_stage(state: Mapping[str, Any]) -> _StageOutcome:
        source = state[PIPELINE_STAGE_ORDER[0]]
        seeds = state[PIPELINE_STAGE_ORDER[1]]
        rebase = state[PIPELINE_STAGE_ORDER[2]]
        qk_reload = state[PIPELINE_STAGE_ORDER[5]]
        causal_o = state[PIPELINE_STAGE_ORDER[7]]
        manifest = seeds["manifest"]
        records = manifest["tensor_records"]
        installation = (
            compile_install_lfm2_post_causal_o_source_local_global_reclosure(
                cache=cache,
                bridge=bridge,
                donor_tokenizer=donor_tokenizer,
                target_tokenizer=target_tokenizer,
                target_model=qk_reload["model"],
                causal_o_installation=causal_o,
                expected_causal_o_receipt_sha256=causal_o.receipt[
                    "result_sha256"
                ],
                semantic_layer_stack_receipt=source["stack"],
                expected_semantic_layer_stack_receipt_sha256=source["stack"][
                    "result_sha256"
                ],
                mapped_parent_tied_table=seeds[
                    "archive"
                ].mapped_parent_tied_table,
                mapped_parent_tied_table_receipt=manifest,
                expected_mapped_parent_tied_table_sha256=records[
                    "mapped_parent_tied_table"
                ]["sha256"],
                expected_mapped_parent_tied_table_receipt_sha256=manifest[
                    "manifest_sha256"
                ],
                producer_encoder=seeds["archive"].producer_encoder,
                producer_encoder_receipt=manifest,
                expected_producer_encoder_sha256=records["producer_encoder"][
                    "sha256"
                ],
                expected_producer_encoder_receipt_sha256=manifest[
                    "manifest_sha256"
                ],
                terminal_rebase=rebase,
                expected_terminal_rebase_receipt_sha256=rebase.receipt[
                    "result_sha256"
                ],
                expected_terminal_prior_authentication_sha256=(
                    rebase.terminal_prior.authentication_sha256
                ),
                authenticated_cache_session=shared["cache_session"],
            )
        )
        if not installation.verify():
            raise Lfm2Exact4xCausalReclosureError(
                "Post-O source-local global reclosure failed authentication."
            )
        _write_json(
            run_root / "post-o-source-local-global-reclosure.json",
            installation.receipt,
        )
        _write_json(
            run_root / "post-o-source-local-global-installation.json",
            installation.installation.report,
        )
        shared["global"] = installation
        return _StageOutcome(
            value=installation,
            evidence={
                "global_reclosure_result_sha256": installation.receipt[
                    "result_sha256"
                ],
                "global_residual_evidence_sha256": installation.receipt[
                    "global_residual_evidence_sha256"
                ],
                "global_reclosure_report_sha256": installation.receipt[
                    "global_reclosure_report_sha256"
                ],
                "corrected_terminal_w2_prior_sha256": installation.receipt[
                    "corrected_terminal_w2_prior_sha256"
                ],
                "live_terminal_w2_unchanged_and_deferred": True,
            },
        )

    def terminal_stage(state: Mapping[str, Any]) -> _StageOutcome:
        rebase = state[PIPELINE_STAGE_ORDER[2]]
        qk_reload = state[PIPELINE_STAGE_ORDER[5]]
        global_installation = state[PIPELINE_STAGE_ORDER[8]]
        receipt = dict(
            install_lfm2_post_reclosure_terminal_w2(
                qk_reload["model"],
                rebase=rebase,
                corrected_terminal_prior=(
                    global_installation.corrected_terminal_w2_prior
                ),
                source_local_global_reclosure_report=(
                    global_installation.installation.report
                ),
            )
        )
        _assert_dense_unshared_exact4x(qk_reload["model"])
        _write_json(run_root / "corrected-terminal-w2-install.json", receipt)
        shared["terminal"] = receipt
        return _StageOutcome(
            value={"model": qk_reload["model"], "receipt": receipt},
            evidence={
                "terminal_install_result_sha256": receipt["result_sha256"],
                "final_w2_parameter_name": receipt["final_w2_parameter_name"],
                "live_w2_sha256_after": receipt["live_w2_sha256_after"],
                "installation_count": receipt["installation_count"],
            },
        )

    def final_reload_stage(state: Mapping[str, Any]) -> _StageOutcome:
        terminal = state[PIPELINE_STAGE_ORDER[9]]
        model, receipt = _save_and_fresh_reload(
            model=terminal["model"],
            checkpoint=(
                run_root
                / "candidates"
                / FINAL_CANDIDATE_ID
                / "checkpoint"
            ),
            checkpoint_io=checkpoint_io,
        )
        terminal_receipt = terminal["receipt"]
        if (
            receipt["lfm2_identity_sha256"]
            != terminal_receipt["model_state_after_sha256"]
            or _final_w2_sha256(
                model,
                terminal_receipt["final_w2_parameter_name"],
            )
            != terminal_receipt["live_w2_sha256_after"]
        ):
            raise Lfm2Exact4xCausalReclosureError(
                "Final standalone reload changed the corrected terminal W2."
            )
        terminal["model"] = None
        state[PIPELINE_STAGE_ORDER[5]]["model"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        value = {"model": model, "receipt": receipt}
        shared["final"] = value
        return _StageOutcome(value=value, evidence=receipt)

    def behavior_stage(state: Mapping[str, Any]) -> _StageOutcome:
        final = state[PIPELINE_STAGE_ORDER[10]]
        receipt = final["receipt"]
        final["model"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        frontier_receipt = composite._frontier_receipt(
            run_directory=run_root,
            candidate_id=FINAL_CANDIDATE_ID,
            checkpoint=Path(receipt["checkpoint"]),
            model_state_sha256=receipt["model_state_sha256"],
            checkpoint_inventory=receipt["checkpoint_inventory"],
            lfm2_identity_sha256=receipt["lfm2_identity_sha256"],
            lfm2_config_sha256=receipt["lfm2_config_sha256"],
            materialization_dtype="float32",
        )
        frontier_path = run_root / "frontier-run.json"
        _write_json(frontier_path, frontier_receipt)
        dossier_path = run_root / "whole-response-dossier.json"
        dossier = review_frontier(
            frontier_run_receipt=frontier_path,
            expected_frontier_run_sha256=frontier_receipt["run_sha256"],
            expected_frontier_run_file_sha256=_file_sha256(frontier_path),
            prompts_json=prompts_path,
            output_json=dossier_path,
            candidate_ids=(FINAL_CANDIDATE_ID,),
            device=model_device,
            dtype="float32",
            maximum_prompt_tokens=maximum_prompt_tokens,
            maximum_new_tokens=maximum_new_tokens,
            parent_envelope=envelope_path,
            expected_parent_envelope_sha256=expected_parent_envelope_sha256,
            expected_parent_envelope_file_sha256=(
                expected_parent_envelope_file_sha256
            ),
        )
        reviews = dossier.get("candidate_reviews")
        if (
            dossier.get("reviewed_candidate_count") != 1
            or not isinstance(reviews, list)
            or len(reviews) != 1
            or reviews[0].get("candidate_id") != FINAL_CANDIDATE_ID
            or len(reviews[0].get("responses", [])) != len(prompt_rows)
        ):
            raise Lfm2Exact4xCausalReclosureError(
                "Whole-response dossier did not retain every decoded response."
            )
        manual = _manual_review_artifact(
            dossier=dossier,
            candidate_id=FINAL_CANDIDATE_ID,
        )
        _write_json(run_root / "manual-review.json", manual)
        final["model"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        value = {
            "frontier": frontier_receipt,
            "dossier": dossier,
            "manual_review": manual,
        }
        shared["behavior"] = value
        return _StageOutcome(
            value=value,
            evidence={
                "frontier_run_sha256": frontier_receipt["run_sha256"],
                "whole_response_dossier_sha256": dossier["dossier_sha256"],
                "manual_review_artifact_sha256": manual[
                    "review_artifact_sha256"
                ],
                "response_count": manual["response_count"],
                "all_raw_outputs_retained": True,
                "numeric_or_manual_hard_gate_used": False,
            },
        )

    handlers = {
        PIPELINE_STAGE_ORDER[0]: source_stage,
        PIPELINE_STAGE_ORDER[1]: seed_stage,
        PIPELINE_STAGE_ORDER[2]: rebase_stage,
        PIPELINE_STAGE_ORDER[3]: rebase_reload_stage,
        PIPELINE_STAGE_ORDER[4]: qk_stage,
        PIPELINE_STAGE_ORDER[5]: qk_reload_stage,
        PIPELINE_STAGE_ORDER[6]: candidates_stage,
        PIPELINE_STAGE_ORDER[7]: causal_o_stage,
        PIPELINE_STAGE_ORDER[8]: global_stage,
        PIPELINE_STAGE_ORDER[9]: terminal_stage,
        PIPELINE_STAGE_ORDER[10]: final_reload_stage,
        PIPELINE_STAGE_ORDER[11]: behavior_stage,
    }
    with cache.selective_authenticated_session() as cache_session:
        shared["cache_session"] = cache_session
        state, sequence_receipt = _run_ordered_stage_machine(handlers)
    shared.pop("cache_session", None)
    _write_json(run_root / "pipeline-stage-sequence.json", sequence_receipt)

    source = state[PIPELINE_STAGE_ORDER[0]]
    final = state[PIPELINE_STAGE_ORDER[10]]
    behavior = state[PIPELINE_STAGE_ORDER[11]]
    realized_ratio = EXPECTED_PARENT_UNIQUE_PARAMETERS / EXPECTED_CHILD_UNIQUE_PARAMETERS
    if any(value not in (0, False) for value in FORBIDDEN_OPERATION_AUDIT.values()):
        raise AssertionError("Static forbidden-operation audit is not zero.")
    body = {
        "schema": RUN_SCHEMA,
        "status": "standalone-exact4x-causal-reclosed-child-decoded",
        "run_directory": str(run_root),
        "source_run_directory": str(source_root),
        "source_run_result_sha256": source["run"]["result_sha256"],
        "parent_identity_sha256": expected_parent,
        "parent_unique_parameters": EXPECTED_PARENT_UNIQUE_PARAMETERS,
        "target_unique_parameters": EXPECTED_CHILD_UNIQUE_PARAMETERS,
        "realized_parameter_compaction_ratio": realized_ratio,
        "target_topology": dict(EXPECTED_CHILD_TOPOLOGY),
        "target_parameter_dtype": "float32",
        "float32_storage_is_not_quantization": True,
        "dense_parameter_compaction_not_quantization": True,
        "target_is_dense_untied_blank_architecture": True,
        "parameter_sharing_used": False,
        "pipeline_stage_sequence_sha256": sequence_receipt["result_sha256"],
        "postfold_reclosure_seed_manifest_sha256": state[
            PIPELINE_STAGE_ORDER[1]
        ]["manifest"]["manifest_sha256"],
        "terminal_w2_rebase_sha256": state[PIPELINE_STAGE_ORDER[2]].receipt[
            "result_sha256"
        ],
        "qk_treatment_sha256": state[PIPELINE_STAGE_ORDER[4]]["treatment"][
            "result_sha256"
        ],
        "qk_fresh_reload_provenance_sha256": state[PIPELINE_STAGE_ORDER[5]][
            "provenance"
        ]["receipt_sha256"],
        "structural_candidate_archive_manifest_sha256": state[
            PIPELINE_STAGE_ORDER[6]
        ]["manifest"]["manifest_sha256"],
        "causal_o_result_sha256": state[PIPELINE_STAGE_ORDER[7]].receipt[
            "result_sha256"
        ],
        "post_o_global_reclosure_sha256": state[
            PIPELINE_STAGE_ORDER[8]
        ].receipt["result_sha256"],
        "corrected_terminal_w2_install_sha256": state[
            PIPELINE_STAGE_ORDER[9]
        ]["receipt"]["result_sha256"],
        "final_checkpoint": final["receipt"]["checkpoint"],
        "final_checkpoint_inventory_sha256": final["receipt"][
            "checkpoint_inventory_sha256"
        ],
        "standalone_fresh_reload_verified": True,
        "whole_response_dossier": str(run_root / "whole-response-dossier.json"),
        "whole_response_dossier_sha256": behavior["dossier"]["dossier_sha256"],
        "manual_review_artifact": str(run_root / "manual-review.json"),
        "manual_review_artifact_sha256": behavior["manual_review"][
            "review_artifact_sha256"
        ],
        "all_raw_outputs_retained": True,
        "whole_decoded_behavior_is_primary_intelligence_evidence": True,
        "numeric_metrics_have_final_intelligence_authority": False,
        "numeric_or_manual_score_used_as_hard_gate": False,
        "automatic_behavior_acceptance_or_rejection_performed": False,
        "single_authenticated_parent_cache_session": True,
        "sealed_parent_cache_reused_without_trace_recapture": True,
        "forbidden_operation_audit": dict(FORBIDDEN_OPERATION_AUDIT),
    }
    result = _sealed(body, "result_sha256")
    _write_json(run_root / "run-receipt.json", result)
    _emit("complete", result_sha256=result["result_sha256"])
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--source-run-directory", required=True, type=Path)
    parser.add_argument("--expected-source-run-result-sha256", required=True)
    parser.add_argument("--expected-source-run-file-sha256", required=True)
    parser.add_argument("--donor-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--target-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--bridge-directory", required=True, type=Path)
    parser.add_argument("--expected-bridge-artifact-sha256", required=True)
    parser.add_argument("--semantic-cache-directory", required=True, type=Path)
    parser.add_argument("--expected-semantic-cache-key", required=True)
    parser.add_argument("--expected-parent-identity-sha256", required=True)
    parser.add_argument("--expected-attention-lineage-file-sha256", required=True)
    parser.add_argument("--prompts-json", required=True, type=Path)
    parser.add_argument("--expected-prompts-file-sha256", required=True)
    parser.add_argument("--parent-envelope", required=True, type=Path)
    parser.add_argument("--expected-parent-envelope-sha256", required=True)
    parser.add_argument("--expected-parent-envelope-file-sha256", required=True)
    parser.add_argument("--model-device", default="cuda:0")
    parser.add_argument("--fold-compute-device", default="cuda:0")
    parser.add_argument("--o-ridge", type=float, default=1.0e-6)
    parser.add_argument("--maximum-prompt-tokens", type=int, default=1024)
    parser.add_argument("--maximum-new-tokens", type=int, default=256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run(**vars(args))
    except BaseException as exc:
        body = {
            "schema": FAILURE_SCHEMA,
            "status": "failed-without-behavior-promotion",
            "exception_type": type(exc).__name__,
            "message": str(exc)[:4096],
            "automatic_acceptance_or_rejection_performed": False,
        }
        failure = _sealed(body, "failure_sha256")
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
