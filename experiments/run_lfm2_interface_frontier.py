#!/usr/bin/env python3
"""Compile and review the exact-body LFM2 token-interface frontier.

The runner reuses one authenticated local parent cache and one sealed strict
tokenizer bridge.  It preserves a tied contextual-input control, dense untied
first-subtoken and full-decomposition structural controls, and three closed-
form cached-logit residual variants.  Every candidate is an independent dense
checkpoint, is freshly reloaded without a parent sidecar, and is retained for
whole-response review.  Numeric reconstruction diagnostics never select or
reject candidates.

This is an interface-isolation experiment, not the final >=4x topology.  The
body is copied exactly so lexical/interface damage can be separated from body
fold damage.  There is no training, autograd, optimizer, pruning, quantization,
sparsity, parent forward/generate, trace recapture, network access, adapter,
sidecar, or runtime parameter sharing.
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
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    fingerprint_lfm2_parent,
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_cpcf_blank_target import (  # noqa: E402
    plan_and_synthesize_lfm2_cpcf_blank,
)
from neural_forge.hf_lfm2_direct_local import (  # noqa: E402
    Lfm2DirectLocalCheckpointIO,
    load_lfm2_direct_local_model,
)
from neural_forge.hf_lfm2_interface_frontier import (  # noqa: E402
    DEFAULT_INTERFACE_CONTEXTS,
    Lfm2InterfaceRecipe,
    compile_lfm2_cached_logit_residual_head,
    compile_lfm2_interface_sequence_plan,
    compile_lfm2_inverse_bucket_output_table,
    compile_lfm2_inverse_output_plan,
    fold_lfm2_interface_candidate,
    target_unique_parameters_for_recipe,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    fingerprint_hf_tokenizer_semantics,
    load_strict_tokenizer_bridge_artifact,
)


RUN_SCHEMA = "neural-forge-lfm2-interface-frontier-run-v1"
TERMINAL_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-interface-terminal-cache-evidence-v1"
)
FRONTIER_SCHEMA = "neural-forge-lfm2-semantic-terminal-frontier-run-v1"
FAILURE_SCHEMA = "neural-forge-lfm2-interface-frontier-failure-v1"

STRUCTURAL_RECIPES = (
    Lfm2InterfaceRecipe(
        candidate_id="context-anchor500-norm-tied-control",
        input_strict_anchor_ppm=500_000,
        input_preserve_anchor_norm=True,
        output_mode="tied-to-input",
    ),
    Lfm2InterfaceRecipe(
        candidate_id="context-anchor500-inverse-first-bucket-untied",
        input_strict_anchor_ppm=500_000,
        input_preserve_anchor_norm=True,
        output_mode="inverse-bucket-centroid",
        output_strict_anchor_ppm=250_000,
        output_preserve_anchor_norm=True,
        output_bucket_weighting="row-norm",
    ),
    Lfm2InterfaceRecipe(
        candidate_id="context-anchor500-prefix-transition-proxy-untied",
        input_strict_anchor_ppm=500_000,
        input_preserve_anchor_norm=True,
        output_mode="prefix-transition-centroid",
        output_strict_anchor_ppm=250_000,
        output_preserve_anchor_norm=True,
        output_bucket_weighting="row-norm",
    ),
)

ANALYTIC_VARIANTS = (
    {
        "candidate_id": "cached-first-bucket-a250-r1e3-norm500-specials",
        "ridge_fraction": 1e-3,
        "residual_scale_ppm": 250_000,
        "row_norm_calibration_ppm": 500_000,
        "protect_specials": True,
    },
    {
        "candidate_id": "cached-first-bucket-a500-r1e2-norm250-specials",
        "ridge_fraction": 1e-2,
        "residual_scale_ppm": 500_000,
        "row_norm_calibration_ppm": 250_000,
        "protect_specials": True,
    },
    {
        "candidate_id": "cached-first-bucket-a1000-r1e1-raw",
        "ridge_fraction": 1e-1,
        "residual_scale_ppm": 1_000_000,
        "row_norm_calibration_ppm": 0,
        "protect_specials": False,
    },
)


class Lfm2InterfaceFrontierRunnerError(RuntimeError):
    """The authenticated interface-frontier transaction failed."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: str, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise Lfm2InterfaceFrontierRunnerError(
            f"{label} must be a lowercase SHA-256 digest."
        )
    return normalized


def _ordinary_directory(value: Path, label: str) -> Path:
    if "://" in os.fspath(value):
        raise Lfm2InterfaceFrontierRunnerError(f"{label} must be local.")
    path = value.expanduser().absolute()
    if path.is_symlink() or not path.is_dir():
        raise Lfm2InterfaceFrontierRunnerError(f"{label} is missing: {path}.")
    return path


def _ordinary_file(value: Path, label: str) -> Path:
    if "://" in os.fspath(value):
        raise Lfm2InterfaceFrontierRunnerError(f"{label} must be local.")
    path = value.expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise Lfm2InterfaceFrontierRunnerError(f"{label} is missing: {path}.")
    return path


def _new_run_directory(value: Path) -> Path:
    if "://" in os.fspath(value):
        raise Lfm2InterfaceFrontierRunnerError("run_directory must be local.")
    path = value.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise Lfm2InterfaceFrontierRunnerError(
            "run_directory must not already exist."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()
    return path


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Lfm2InterfaceFrontierRunnerError(
            f"Refusing to replace existing evidence: {path}."
        )
    payload = json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    path.write_text(payload + "\n", encoding="ascii")


def _unique_parameter_count(model: Any) -> int:
    return sum(int(value.numel()) for value in model.parameters())


def _parameter_storage_addresses(model: Any) -> set[int]:
    return {
        int(parameter.untyped_storage().data_ptr())
        for parameter in model.parameters()
    }


def _artifact_inventory(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise Lfm2InterfaceFrontierRunnerError(
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
        raise Lfm2InterfaceFrontierRunnerError("Standalone checkpoint is empty.")
    return rows


def _terminal_phase_pairs(reference: Any) -> tuple[tuple[str, str, str], ...]:
    names = set(reference.ordered_tensor_names)
    suffix = ".output.lm_head_input"
    pairs = []
    for hidden_name in sorted(name for name in names if name.endswith(suffix)):
        phase = hidden_name[: -len(suffix)]
        logits_name = f"{phase}.logits"
        if logits_name not in names:
            raise Lfm2InterfaceFrontierRunnerError(
                f"Cache phase {phase!r} lacks matching full logits."
            )
        pairs.append((phase, hidden_name, logits_name))
    if not pairs:
        raise Lfm2InterfaceFrontierRunnerError(
            "Cache batch contains no aligned terminal feature/logit rows."
        )
    return tuple(pairs)


def _collect_terminal_evidence(
    cache: Any,
    *,
    expected_cache_key: str,
    expected_parent_identity_sha256: str,
) -> tuple[Any, Any, dict[str, Any]]:
    if (
        cache.cache_key != expected_cache_key
        or cache.parent_fingerprint != expected_parent_identity_sha256
    ):
        raise Lfm2InterfaceFrontierRunnerError(
            "Trace cache ancestry differs from its external pins."
        )
    hidden_parts = []
    logits_parts = []
    selected_batches = []
    with cache.selective_authenticated_session() as session:
        for reference in session.batch_references:
            pairs = _terminal_phase_pairs(reference)
            names = tuple(
                sorted(
                    name
                    for _phase, hidden_name, logits_name in pairs
                    for name in (hidden_name, logits_name)
                )
            )
            payload = session.load_selected_tensors(reference.index, names)
            try:
                batch_rows = 0
                phases = []
                for phase, hidden_name, logits_name in pairs:
                    hidden = payload.tensors[hidden_name]
                    logits = payload.tensors[logits_name]
                    if (
                        hidden.ndim != 3
                        or logits.ndim != 3
                        or tuple(hidden.shape[:-1]) != tuple(logits.shape[:-1])
                        or int(hidden.shape[-1]) <= 0
                        or int(logits.shape[-1]) <= 0
                    ):
                        raise Lfm2InterfaceFrontierRunnerError(
                            f"Terminal phase {phase!r} is not row-aligned."
                        )
                    rows = int(hidden.shape[0] * hidden.shape[1])
                    hidden_parts.append(
                        hidden.detach().reshape(rows, int(hidden.shape[-1])).clone()
                    )
                    logits_parts.append(
                        logits.detach().reshape(rows, int(logits.shape[-1])).clone()
                    )
                    batch_rows += rows
                    phases.append(
                        {
                            "phase": phase,
                            "rows": rows,
                            "hidden_name": hidden_name,
                            "logits_name": logits_name,
                        }
                    )
                selected_batches.append(
                    {
                        "batch_index": reference.index,
                        "input_sha256": reference.input_sha256,
                        "selection_sha256": payload.selection_sha256,
                        "logical_bytes": payload.logical_bytes,
                        "terminal_rows": batch_rows,
                        "phases": phases,
                    }
                )
            finally:
                session.release_selected_tensors(payload)
        session.verify()
        session_receipt = session.as_dict()
    hidden_rows = torch.cat(hidden_parts, dim=0).contiguous()
    donor_logits = torch.cat(logits_parts, dim=0).contiguous()
    if (
        int(hidden_rows.shape[0]) != int(donor_logits.shape[0])
        or int(hidden_rows.shape[0])
        != sum(int(row["terminal_rows"]) for row in selected_batches)
        or not bool(torch.isfinite(hidden_rows).all())
        or not bool(torch.isfinite(donor_logits).all())
    ):
        raise Lfm2InterfaceFrontierRunnerError(
            "Selected terminal evidence is incomplete or nonfinite."
        )
    body = {
        "schema": TERMINAL_EVIDENCE_SCHEMA,
        "status": "complete-authenticated-terminal-cache-selection",
        "cache_key": expected_cache_key,
        "parent_identity_sha256": expected_parent_identity_sha256,
        "selective_session_sha256": session_receipt["session_sha256"],
        "manifest_sha256": session_receipt["manifest_sha256"],
        "completion_sha256": session_receipt["completion_sha256"],
        "batch_count": len(selected_batches),
        "selected_batches": selected_batches,
        "terminal_row_count": int(hidden_rows.shape[0]),
        "hidden_size": int(hidden_rows.shape[1]),
        "donor_vocabulary_size": int(donor_logits.shape[1]),
        "all_aligned_prefill_and_decode_rows_retained": True,
        "unpaired_cached_prefill_logits_excluded": True,
        "sealed_parent_cache_reused_without_recapture": True,
        "operation_audit": {
            "training_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "network_access": False,
        },
    }
    receipt = {**body, "evidence_sha256": canonical_sha256(body)}
    return hidden_rows, donor_logits, receipt


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
        "final_four_x_topology": False,
        "first_subtoken_and_prefix_transition_ablations_retained": True,
    }
    return {**body, "run_sha256": canonical_sha256(body)}


def _analytic_recipe(candidate_id: str) -> Lfm2InterfaceRecipe:
    return Lfm2InterfaceRecipe(
        candidate_id=candidate_id,
        input_strict_anchor_ppm=500_000,
        input_preserve_anchor_norm=True,
        output_mode="cached-logit-residual",
        output_strict_anchor_ppm=250_000,
        output_preserve_anchor_norm=True,
        output_bucket_weighting="row-norm",
    )


def run(
    *,
    run_directory: Path,
    parent_model_directory: Path,
    donor_tokenizer_directory: Path,
    target_tokenizer_directory: Path,
    bridge_directory: Path,
    expected_bridge_artifact_sha256: str,
    trace_cache_directory: Path,
    expected_cache_key: str,
    expected_parent_identity_sha256: str,
    expected_parent_unique_parameters: int,
    expected_base_tied_target_unique_parameters: int,
    blank_seed: int,
    maximum_candidates: int,
    selected_candidate_rank: int | None,
    prompts_json: Path,
    expected_prompts_file_sha256: str,
    parent_envelope: Path,
    expected_parent_envelope_sha256: str,
    expected_parent_envelope_file_sha256: str,
    review_device: str,
    analytic_solve_device: str,
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
) -> dict[str, Any]:
    run_root = _new_run_directory(run_directory)
    parent_path = _ordinary_directory(parent_model_directory, "parent model")
    donor_path = _ordinary_directory(donor_tokenizer_directory, "donor tokenizer")
    target_path = _ordinary_directory(target_tokenizer_directory, "target tokenizer")
    bridge_path = _ordinary_directory(bridge_directory, "tokenizer bridge")
    cache_path = _ordinary_directory(trace_cache_directory, "trace cache")
    prompts_path = _ordinary_file(prompts_json, "review prompts")
    envelope_path = _ordinary_file(parent_envelope, "parent envelope")
    expected_bridge = _digest(
        expected_bridge_artifact_sha256,
        "expected bridge artifact SHA-256",
    )
    expected_cache = _digest(expected_cache_key, "expected cache key")
    expected_parent = _digest(
        expected_parent_identity_sha256,
        "expected parent identity SHA-256",
    )
    expected_prompt_file = _digest(
        expected_prompts_file_sha256,
        "expected prompt file SHA-256",
    )
    expected_envelope = _digest(
        expected_parent_envelope_sha256,
        "expected parent envelope SHA-256",
    )
    expected_envelope_file = _digest(
        expected_parent_envelope_file_sha256,
        "expected parent envelope file SHA-256",
    )
    if _file_sha256(prompts_path) != expected_prompt_file:
        raise Lfm2InterfaceFrontierRunnerError("Review prompt bytes changed.")
    if _file_sha256(envelope_path) != expected_envelope_file:
        raise Lfm2InterfaceFrontierRunnerError("Parent envelope bytes changed.")
    if (
        isinstance(expected_parent_unique_parameters, bool)
        or expected_parent_unique_parameters <= 0
        or isinstance(expected_base_tied_target_unique_parameters, bool)
        or expected_base_tied_target_unique_parameters <= 0
        or expected_base_tied_target_unique_parameters
        >= expected_parent_unique_parameters
        or maximum_candidates <= 0
    ):
        raise Lfm2InterfaceFrontierRunnerError(
            "Expected parameter inventories or planner bound are invalid."
        )

    donor_tokenizer = load_local_tokenizer(donor_path)
    target_tokenizer = load_local_tokenizer(target_path)
    bridge = load_strict_tokenizer_bridge_artifact(
        bridge_path,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        expected_artifact_sha256=expected_bridge,
    )
    cache = open_lfm2_circuit_trace_cache(cache_path, verify=True)
    if (
        cache.cache_key != expected_cache
        or cache.parent_fingerprint != expected_parent
        or fingerprint_hf_tokenizer_semantics(donor_tokenizer)
        != bridge.donor_tokenizer_semantics_sha256
        or fingerprint_hf_tokenizer_semantics(target_tokenizer)
        != bridge.target_tokenizer_semantics_sha256
    ):
        raise Lfm2InterfaceFrontierRunnerError(
            "Cache, parent, bridge, or live tokenizer ancestry differs from pins."
        )

    prompt_rows = [dict(value) for value in load_review_prompts(prompts_path)]
    prompt_set_sha256 = canonical_sha256(prompt_rows)
    parent_evidence = _load_parent_envelope(
        envelope_path,
        expected_envelope_sha256=expected_envelope,
        expected_file_sha256=expected_envelope_file,
        expected_prompt_set_sha256=prompt_set_sha256,
        expected_prompt_ids=[str(value["id"]) for value in prompt_rows],
    )
    if (
        parent_evidence.get("parent_identity_sha256") != expected_parent
        or parent_evidence.get("parent_unique_parameters")
        != expected_parent_unique_parameters
        or parent_evidence.get("maximum_new_tokens") != maximum_new_tokens
    ):
        raise Lfm2InterfaceFrontierRunnerError(
            "Cached parent behavior differs from the interface request."
        )

    hidden_rows, donor_logits, terminal_evidence = _collect_terminal_evidence(
        cache,
        expected_cache_key=expected_cache,
        expected_parent_identity_sha256=expected_parent,
    )
    terminal_evidence_path = run_root / "terminal-cache-evidence.json"
    _write_json(terminal_evidence_path, terminal_evidence)

    parent = load_lfm2_direct_local_model(
        parent_path,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    parent.requires_grad_(False)
    parent.eval()
    if (
        fingerprint_lfm2_parent(parent).identity_sha256 != expected_parent
        or _unique_parameter_count(parent) != expected_parent_unique_parameters
        or int(parent.get_input_embeddings().weight.shape[0])
        != bridge.transport_profile.donor_row_capacity
        or int(parent.get_input_embeddings().weight.shape[1])
        != int(hidden_rows.shape[1])
    ):
        raise Lfm2InterfaceFrontierRunnerError(
            "Direct-local parent differs from sealed terminal evidence."
        )
    parent_state_before = fingerprint_model_state(parent)
    parent_storage = _parameter_storage_addresses(parent)
    sequence_plan = compile_lfm2_interface_sequence_plan(
        bridge,
        donor_tokenizer,
        target_tokenizer,
        contexts=DEFAULT_INTERFACE_CONTEXTS,
    )
    target_pad_id = getattr(target_tokenizer, "pad_token_id", None)
    if (
        isinstance(target_pad_id, bool)
        or not isinstance(target_pad_id, int)
        or not 0 <= target_pad_id < len(target_tokenizer)
    ):
        raise Lfm2InterfaceFrontierRunnerError(
            "The no-UNK inverse bridge requires an authenticated target pad row."
        )
    inverse_plan = compile_lfm2_inverse_output_plan(
        bridge,
        donor_tokenizer,
        target_tokenizer,
        fallback_target_id=target_pad_id,
    )
    mapping = tuple(
        int(value)
        for value in bridge.transport_profile.target_to_donor_row_indices
    )
    structural_output = compile_lfm2_inverse_bucket_output_table(
        parent.get_output_embeddings().weight,
        inverse_plan,
        mapping,
        strict_anchor_ppm=250_000,
        preserve_anchor_norm=True,
        weighting="row-norm",
    )
    gram_diagonal_mean = float(
        hidden_rows.to(dtype=torch.float32).square().sum(dim=1).mean()
    )
    if not gram_diagonal_mean > 0.0:
        raise Lfm2InterfaceFrontierRunnerError(
            "Terminal features have zero closed-form ridge scale."
        )
    target_special_ids = tuple(
        sorted(
            {
                int(value)
                for value in getattr(target_tokenizer, "all_special_ids", ())
                if 0 <= int(value) < int(structural_output.shape[0])
            }
        )
    )
    analytic_heads = {}
    analytic_recipes = []
    for variant in ANALYTIC_VARIANTS:
        candidate_id = str(variant["candidate_id"])
        head = compile_lfm2_cached_logit_residual_head(
            structural_output,
            hidden_rows,
            donor_logits,
            inverse_plan,
            source_evidence_sha256=terminal_evidence["evidence_sha256"],
            ridge=gram_diagonal_mean * float(variant["ridge_fraction"]),
            residual_scale_ppm=int(variant["residual_scale_ppm"]),
            protected_target_ids=(
                target_special_ids if bool(variant["protect_specials"]) else ()
            ),
            row_norm_calibration_ppm=int(
                variant["row_norm_calibration_ppm"]
            ),
            solve_device=analytic_solve_device,
        )
        analytic_heads[candidate_id] = head
        analytic_recipes.append(_analytic_recipe(candidate_id))
    del hidden_rows
    del donor_logits
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    recipes = (*STRUCTURAL_RECIPES, *analytic_recipes)
    io = Lfm2DirectLocalCheckpointIO(
        tokenizer_bridge=bridge,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    candidate_records = []
    materializations = []
    for recipe in recipes:
        blank = plan_and_synthesize_lfm2_cpcf_blank(
            parent,
            parent_name="lfm2-parent",
            parent_tokenizer_fingerprint=(
                bridge.donor_tokenizer_semantics_sha256
            ),
            evidence_fingerprint=cache.cache_key,
            child_name=recipe.candidate_id,
            target_unique_parameters=(
                expected_base_tied_target_unique_parameters
            ),
            target_vocabulary_size=(
                bridge.transport_profile.target_row_capacity
            ),
            child_tokenizer_fingerprint=(
                bridge.target_tokenizer_semantics_sha256
            ),
            tokenizer_bridge_artifact=bridge,
            minimum_hidden_size=32,
            maximum_candidates=maximum_candidates,
            selected_candidate_rank=selected_candidate_rank,
            blank_seed=blank_seed,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        target_parameters = target_unique_parameters_for_recipe(
            base_tied_target_unique_parameters=(
                expected_base_tied_target_unique_parameters
            ),
            target_vocabulary_size=(
                bridge.transport_profile.target_row_capacity
            ),
            hidden_size=int(parent.get_input_embeddings().weight.shape[1]),
            recipe=recipe,
        )
        folded = fold_lfm2_interface_candidate(
            parent,
            blank.model,
            bridge=bridge,
            donor_tokenizer=donor_tokenizer,
            target_tokenizer=target_tokenizer,
            blank_target_receipt=blank.as_dict(),
            recipe=recipe,
            maximum_target_unique_parameters=target_parameters,
            contexts=DEFAULT_INTERFACE_CONTEXTS,
            cached_logit_residual_head=analytic_heads.get(recipe.candidate_id),
            precompiled_sequence_plan=sequence_plan,
            precompiled_inverse_output_plan=inverse_plan,
            inverse_output_fallback_target_id=target_pad_id,
        )
        fold_receipt = folded.as_dict()
        if (
            fold_receipt["target_unique_parameters"] != target_parameters
            or fold_receipt["exact_parent_body_depth_width_preserved"] is not True
            or fold_receipt["interface_isolation_control"] is not True
            or fold_receipt["final_four_x_topology"] is not False
        ):
            raise Lfm2InterfaceFrontierRunnerError(
                f"Candidate {recipe.candidate_id!r} failed exact-body accounting."
            )
        candidate_root = run_root / "candidates" / recipe.candidate_id
        checkpoint = candidate_root / "checkpoint"
        checkpoint.mkdir(parents=True)
        installed_state = fingerprint_model_state(folded.model)
        io.save(folded.model, checkpoint)
        fold_receipt_path = candidate_root / "fold-receipt.json"
        _write_json(fold_receipt_path, fold_receipt)
        del blank
        del folded
        gc.collect()

        reloaded = io.load(checkpoint)
        reload_state = fingerprint_model_state(reloaded)
        reload_parameters = _unique_parameter_count(reloaded)
        input_weight = reloaded.get_input_embeddings().weight
        output_weight = reloaded.get_output_embeddings().weight
        observed_tied = input_weight is output_weight
        expected_tied = not recipe.untied_output
        if (
            reload_state != installed_state
            or reload_parameters != target_parameters
            or observed_tied != expected_tied
            or bool(getattr(reloaded.config, "tie_word_embeddings", False))
            != expected_tied
            or not _parameter_storage_addresses(reloaded).isdisjoint(
                parent_storage
            )
        ):
            raise Lfm2InterfaceFrontierRunnerError(
                f"Fresh reload failed for {recipe.candidate_id!r}."
            )
        del reloaded
        gc.collect()
        inventory = _artifact_inventory(checkpoint)
        record = {
            "candidate_id": recipe.candidate_id,
            "recipe": recipe.as_dict(),
            "fold_receipt": str(fold_receipt_path),
            "fold_receipt_sha256": fold_receipt["result_sha256"],
            "checkpoint": str(checkpoint),
            "checkpoint_artifact_inventory": inventory,
            "checkpoint_artifact_inventory_sha256": canonical_sha256(inventory),
            "installed_model_state_sha256": installed_state,
            "fresh_reload_model_state_sha256": reload_state,
            "standalone_fresh_reload_verified": True,
            "fresh_reload_input_output_tied": observed_tied,
            "parent_runtime_storage_disjoint": True,
            "unique_parameters": target_parameters,
            "realized_parameter_compaction_ratio": (
                expected_parent_unique_parameters / target_parameters
            ),
        }
        candidate_records.append(record)
        materializations.append(
            {
                "candidate_id": recipe.candidate_id,
                "candidate_state_sha256": installed_state,
                "model_state_sha256": installed_state,
                "checkpoint_path": str(checkpoint),
                "unique_parameters": target_parameters,
                "fold_receipt_sha256": fold_receipt["result_sha256"],
                "standalone_fresh_reload_verified": True,
            }
        )

    if fingerprint_model_state(parent) != parent_state_before:
        raise Lfm2InterfaceFrontierRunnerError(
            "Parent changed while compiling the interface frontier."
        )
    del parent
    gc.collect()
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
        device=review_device,
        dtype="bfloat16",
        maximum_prompt_tokens=maximum_prompt_tokens,
        maximum_new_tokens=maximum_new_tokens,
        parent_envelope=envelope_path,
        expected_parent_envelope_sha256=expected_envelope,
        expected_parent_envelope_file_sha256=expected_envelope_file,
    )
    reviews = dossier.get("candidate_reviews")
    if (
        dossier.get("reviewed_candidate_count") != len(candidate_ids)
        or not isinstance(reviews, list)
        or [row.get("candidate_id") for row in reviews] != list(candidate_ids)
        or any(len(row.get("responses", [])) != len(prompt_rows) for row in reviews)
    ):
        raise Lfm2InterfaceFrontierRunnerError(
            "Whole-response dossier did not preserve every candidate response."
        )

    body = {
        "schema": RUN_SCHEMA,
        "status": "standalone-exact-body-interface-frontier-reviewed",
        "run_directory": str(run_root),
        "diagnostic_role": "isolates-token-interface-from-body-folding",
        "interface_isolation_control": True,
        "final_four_x_topology": False,
        "parent_model_directory": str(parent_path),
        "parent_identity_sha256": expected_parent,
        "parent_unique_parameters": expected_parent_unique_parameters,
        "base_tied_target_unique_parameters": (
            expected_base_tied_target_unique_parameters
        ),
        "trace_cache_directory": str(cache_path),
        "trace_cache_key": expected_cache,
        "terminal_cache_evidence": str(terminal_evidence_path),
        "terminal_cache_evidence_sha256": terminal_evidence["evidence_sha256"],
        "sealed_parent_cache_reused_without_recapture": True,
        "bridge_directory": str(bridge_path),
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "sequence_plan_sha256": sequence_plan["plan_sha256"],
        "inverse_output_plan_sha256": inverse_plan["plan_sha256"],
        "inverse_output_fallback": {
            "policy": inverse_plan["fallback_policy"],
            "target_id": inverse_plan["fallback_target_id"],
            "target_surface": inverse_plan["fallback_target_surface"],
            "fallback_assignment_count": inverse_plan[
                "fallback_assignment_count"
            ],
            "valid_empty_decomposition_count": inverse_plan[
                "valid_empty_decomposition_count"
            ],
            "invalid_donor_capacity_row_count": inverse_plan[
                "invalid_donor_capacity_row_count"
            ],
        },
        "tokenizer_bridge_limitations": {
            "exact_autoregressive_tokenizer_bridge_claimed": False,
            "first_subtoken_mass_is_partial_bridge_only": True,
            "prefix_transition_candidate_is_structural_proxy": True,
            "cached_child_features_after_inserted_target_subtokens_used": False,
            "token_completion_transition_mixtures_resolved": False,
            "reason": (
                "sealed terminal cache has no per-branch child feature rows; "
                "a donor decomposition can end while another continues"
            ),
        },
        "candidate_count": len(candidate_records),
        "candidate_ids": list(candidate_ids),
        "candidates": candidate_records,
        "all_candidates_retained": True,
        "frontier_run_receipt": str(frontier_path),
        "frontier_run_sha256": frontier["run_sha256"],
        "whole_response_dossier": str(dossier_path),
        "whole_response_dossier_sha256": dossier["dossier_sha256"],
        "prompt_set_sha256": prompt_set_sha256,
        "prompt_count": len(prompt_rows),
        "maximum_prompt_tokens": maximum_prompt_tokens,
        "maximum_new_tokens": maximum_new_tokens,
        "decoded_whole_responses_are_primary_evidence": True,
        "numeric_or_binary_behavior_gate_used": False,
        "operation_audit": {
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "child_generate_calls": len(prompt_rows) * len(candidate_ids),
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
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--parent-model-directory", required=True, type=Path)
    parser.add_argument("--donor-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--target-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--bridge-directory", required=True, type=Path)
    parser.add_argument("--expected-bridge-artifact-sha256", required=True)
    parser.add_argument("--trace-cache-directory", required=True, type=Path)
    parser.add_argument("--expected-cache-key", required=True)
    parser.add_argument("--expected-parent-identity-sha256", required=True)
    parser.add_argument("--expected-parent-unique-parameters", required=True, type=int)
    parser.add_argument(
        "--expected-base-tied-target-unique-parameters",
        required=True,
        type=int,
    )
    parser.add_argument("--blank-seed", type=int, default=0)
    parser.add_argument("--maximum-candidates", type=int, default=64)
    parser.add_argument("--selected-candidate-rank", type=int)
    parser.add_argument("--prompts-json", required=True, type=Path)
    parser.add_argument("--expected-prompts-file-sha256", required=True)
    parser.add_argument("--parent-envelope", required=True, type=Path)
    parser.add_argument("--expected-parent-envelope-sha256", required=True)
    parser.add_argument("--expected-parent-envelope-file-sha256", required=True)
    parser.add_argument("--review-device", default="cuda:0")
    parser.add_argument("--analytic-solve-device", default="cuda:0")
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
            trace_cache_directory=args.trace_cache_directory,
            expected_cache_key=args.expected_cache_key,
            expected_parent_identity_sha256=(
                args.expected_parent_identity_sha256
            ),
            expected_parent_unique_parameters=(
                args.expected_parent_unique_parameters
            ),
            expected_base_tied_target_unique_parameters=(
                args.expected_base_tied_target_unique_parameters
            ),
            blank_seed=args.blank_seed,
            maximum_candidates=args.maximum_candidates,
            selected_candidate_rank=args.selected_candidate_rank,
            prompts_json=args.prompts_json,
            expected_prompts_file_sha256=args.expected_prompts_file_sha256,
            parent_envelope=args.parent_envelope,
            expected_parent_envelope_sha256=(
                args.expected_parent_envelope_sha256
            ),
            expected_parent_envelope_file_sha256=(
                args.expected_parent_envelope_file_sha256
            ),
            review_device=args.review_device,
            analytic_solve_device=args.analytic_solve_device,
            maximum_prompt_tokens=args.maximum_prompt_tokens,
            maximum_new_tokens=args.maximum_new_tokens,
        )
    except Exception as exc:
        run_root = args.run_directory.expanduser().absolute()
        if run_root.is_dir() and not run_root.is_symlink():
            failure_body = {
                "schema": FAILURE_SCHEMA,
                "status": "failed",
                "error_type": type(exc).__qualname__,
                "error": str(exc),
                "operation_audit": {
                    "training_steps": 0,
                    "optimizer_steps": 0,
                    "backward_calls": 0,
                    "parent_forward_calls": 0,
                    "parent_generate_calls": 0,
                    "parent_trace_recaptures": 0,
                    "parameters_pruned": 0,
                    "parameters_quantized": 0,
                    "network_access": False,
                },
            }
            failure = {
                **failure_body,
                "result_sha256": canonical_sha256(failure_body),
            }
            failure_path = run_root / "failure.json"
            if not failure_path.exists():
                _write_json(failure_path, failure)
        print(f"error: {type(exc).__qualname__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_directory": result["run_directory"],
                "candidate_count": result["candidate_count"],
                "whole_response_dossier": result["whole_response_dossier"],
                "result_sha256": result["result_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
