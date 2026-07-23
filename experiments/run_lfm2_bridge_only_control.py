#!/usr/bin/env python3
"""Materialize and review the exact same-body LFM2 v4096 bridge control.

The target is planner-synthesized as one independent dense zero blank.  Every
non-lexical tensor is transported identically from the local parent and the
one native tied embedding/readout table is row-selected through the sealed
strict tokenizer bridge.  The child is saved, freshly reloaded without the
parent, and evaluated through the existing 16-prompt whole-response review.

This diagnostic performs no training, autograd, optimizer, pruning,
quantization, sparsity, parent forward/generate, trace recapture, network
access, external adapter, or parent sidecar operation.
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
from neural_forge.hf_lfm2_bridge_only_control import (  # noqa: E402
    fold_lfm2_bridge_only_control,
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
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    fingerprint_hf_tokenizer_semantics,
    load_strict_tokenizer_bridge_artifact,
)


RUN_SCHEMA = "neural-forge-lfm2-bridge-only-control-run-v1"
FRONTIER_SCHEMA = "neural-forge-lfm2-semantic-terminal-frontier-run-v1"
FAILURE_SCHEMA = "neural-forge-lfm2-bridge-only-control-failure-v1"
CONTROL_CANDIDATE_ID = "same-body-v4096-identity-bridge"


class Lfm2BridgeOnlyControlRunnerError(RuntimeError):
    """The authenticated control transaction could not complete."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while piece := handle.read(1024 * 1024):
            digest.update(piece)
    return digest.hexdigest()


def _digest(value: str, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise Lfm2BridgeOnlyControlRunnerError(
            f"{label} must be a lowercase SHA-256 digest."
        )
    return normalized


def _ordinary_directory(value: Path, label: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2BridgeOnlyControlRunnerError(f"{label} must be local.")
    path = value.expanduser().absolute()
    if path.is_symlink() or not path.is_dir():
        raise Lfm2BridgeOnlyControlRunnerError(f"{label} is missing: {path}.")
    return path


def _ordinary_file(value: Path, label: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2BridgeOnlyControlRunnerError(f"{label} must be local.")
    path = value.expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise Lfm2BridgeOnlyControlRunnerError(f"{label} is missing: {path}.")
    return path


def _new_run_directory(value: Path) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2BridgeOnlyControlRunnerError("run_directory must be local.")
    path = value.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise Lfm2BridgeOnlyControlRunnerError(
            "run_directory must not already exist."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()
    return path


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Lfm2BridgeOnlyControlRunnerError(
            f"Refusing to replace existing evidence: {path}."
        )
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    path.write_text(payload + "\n", encoding="ascii")


def _unique_parameter_count(model: Any) -> int:
    return sum(int(value.numel()) for value in model.parameters())


def _artifact_inventory(directory: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise Lfm2BridgeOnlyControlRunnerError(
                "Standalone checkpoint contains a symlink."
            )
        if path.is_file():
            records.append(
                {
                    "relative_path": path.relative_to(directory).as_posix(),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _file_sha256(path),
                }
            )
    if not records:
        raise Lfm2BridgeOnlyControlRunnerError("Standalone checkpoint is empty.")
    return records


def _frontier_receipt(
    *,
    run_directory: Path,
    checkpoint: Path,
    model_state_sha256: str,
    unique_parameters: int,
    fold_receipt_sha256: str,
) -> dict[str, Any]:
    materialization = {
        "candidate_id": CONTROL_CANDIDATE_ID,
        "candidate_state_sha256": model_state_sha256,
        "model_state_sha256": model_state_sha256,
        "checkpoint_path": str(checkpoint),
        "unique_parameters": unique_parameters,
        "fold_receipt_sha256": fold_receipt_sha256,
        "standalone_fresh_reload_verified": True,
    }
    body = {
        "schema": FRONTIER_SCHEMA,
        "status": "complete",
        "run_directory": str(run_directory),
        "candidate_ids": [CONTROL_CANDIDATE_ID],
        "materializations": [materialization],
        "all_retained_candidates_materialized": True,
        "candidate_metric_threshold_used_for_materialization": False,
        "decoded_behavior_is_primary_evidence": True,
        "bridge_only_diagnostic_control": True,
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
    trace_cache_directory: Path,
    expected_cache_key: str,
    expected_parent_identity_sha256: str,
    expected_parent_unique_parameters: int,
    expected_target_unique_parameters: int,
    blank_seed: int,
    maximum_candidates: int,
    selected_candidate_rank: int | None,
    prompts_json: Path,
    expected_prompts_file_sha256: str,
    parent_envelope: Path,
    expected_parent_envelope_sha256: str,
    expected_parent_envelope_file_sha256: str,
    device: str,
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
) -> dict[str, Any]:
    run_root = _new_run_directory(run_directory)
    parent_path = _ordinary_directory(parent_model_directory, "parent model")
    donor_path = _ordinary_directory(donor_tokenizer_directory, "donor tokenizer")
    target_tokenizer_path = _ordinary_directory(
        target_tokenizer_directory, "target tokenizer"
    )
    bridge_path = _ordinary_directory(bridge_directory, "tokenizer bridge")
    cache_path = _ordinary_directory(trace_cache_directory, "trace cache")
    prompts_path = _ordinary_file(prompts_json, "review prompts")
    envelope_path = _ordinary_file(parent_envelope, "parent envelope")
    expected_bridge = _digest(
        expected_bridge_artifact_sha256, "expected bridge artifact SHA-256"
    )
    expected_cache = _digest(expected_cache_key, "expected cache key")
    expected_parent = _digest(
        expected_parent_identity_sha256, "expected parent identity SHA-256"
    )
    expected_prompt_file = _digest(
        expected_prompts_file_sha256, "expected prompt file SHA-256"
    )
    expected_envelope = _digest(
        expected_parent_envelope_sha256, "expected parent envelope SHA-256"
    )
    expected_envelope_file = _digest(
        expected_parent_envelope_file_sha256,
        "expected parent envelope file SHA-256",
    )
    if _file_sha256(prompts_path) != expected_prompt_file:
        raise Lfm2BridgeOnlyControlRunnerError("Review prompt bytes changed.")
    if _file_sha256(envelope_path) != expected_envelope_file:
        raise Lfm2BridgeOnlyControlRunnerError("Parent envelope bytes changed.")
    if (
        isinstance(expected_parent_unique_parameters, bool)
        or expected_parent_unique_parameters <= 0
        or isinstance(expected_target_unique_parameters, bool)
        or expected_target_unique_parameters <= 0
        or expected_target_unique_parameters >= expected_parent_unique_parameters
    ):
        raise Lfm2BridgeOnlyControlRunnerError(
            "Expected parameter inventories are invalid."
        )
    if maximum_candidates <= 0:
        raise Lfm2BridgeOnlyControlRunnerError(
            "maximum_candidates must be positive."
        )

    donor_tokenizer = load_local_tokenizer(donor_path)
    target_tokenizer = load_local_tokenizer(target_tokenizer_path)
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
        raise Lfm2BridgeOnlyControlRunnerError(
            "Cache, parent, or live tokenizer ancestry differs from its pins."
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
        raise Lfm2BridgeOnlyControlRunnerError(
            "Cached parent behavior differs from the control request."
        )

    parent = load_lfm2_direct_local_model(
        parent_path,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    parent.requires_grad_(False)
    parent.eval()
    parent_identity = fingerprint_lfm2_parent(parent).identity_sha256
    parent_parameters = _unique_parameter_count(parent)
    if (
        parent_identity != expected_parent
        or parent_parameters != expected_parent_unique_parameters
    ):
        raise Lfm2BridgeOnlyControlRunnerError(
            "Direct-local parent differs from the sealed cache."
        )

    blank = plan_and_synthesize_lfm2_cpcf_blank(
        parent,
        parent_name="lfm2-parent",
        parent_tokenizer_fingerprint=bridge.donor_tokenizer_semantics_sha256,
        evidence_fingerprint=cache.cache_key,
        child_name=CONTROL_CANDIDATE_ID,
        target_unique_parameters=expected_target_unique_parameters,
        target_vocabulary_size=bridge.transport_profile.target_row_capacity,
        child_tokenizer_fingerprint=bridge.target_tokenizer_semantics_sha256,
        tokenizer_bridge_artifact=bridge,
        minimum_hidden_size=32,
        maximum_candidates=maximum_candidates,
        selected_candidate_rank=selected_candidate_rank,
        blank_seed=blank_seed,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    blank_receipt = blank.as_dict()
    folded = fold_lfm2_bridge_only_control(
        parent,
        blank.model,
        bridge=bridge,
        blank_target_receipt=blank_receipt,
    )
    fold_receipt = folded.as_dict()
    if fold_receipt["target_unique_parameters"] != expected_target_unique_parameters:
        raise Lfm2BridgeOnlyControlRunnerError(
            "Folded control misses the exact target parameter count."
        )

    checkpoint = run_root / "checkpoint"
    checkpoint.mkdir()
    io = Lfm2DirectLocalCheckpointIO(
        tokenizer_bridge=bridge,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    installed_state_sha256 = fingerprint_model_state(folded.model)
    io.save(folded.model, checkpoint)
    del parent
    del blank
    del folded
    gc.collect()

    reloaded = io.load(checkpoint)
    reload_state_sha256 = fingerprint_model_state(reloaded)
    reload_parameters = _unique_parameter_count(reloaded)
    if (
        reload_state_sha256 != installed_state_sha256
        or reload_parameters != expected_target_unique_parameters
        or reloaded.get_input_embeddings().weight
        is not reloaded.get_output_embeddings().weight
    ):
        raise Lfm2BridgeOnlyControlRunnerError(
            "Fresh standalone reload differs from the installed child."
        )
    del reloaded
    gc.collect()

    fold_receipt_path = run_root / "fold-receipt.json"
    _write_json(fold_receipt_path, fold_receipt)
    artifact_inventory = _artifact_inventory(checkpoint)
    frontier = _frontier_receipt(
        run_directory=run_root,
        checkpoint=checkpoint,
        model_state_sha256=installed_state_sha256,
        unique_parameters=expected_target_unique_parameters,
        fold_receipt_sha256=fold_receipt["result_sha256"],
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
        candidate_ids=(CONTROL_CANDIDATE_ID,),
        device=device,
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
        or reviews[0].get("unique_parameters") != expected_target_unique_parameters
        or len(reviews[0].get("responses", [])) != len(prompt_rows)
    ):
        raise Lfm2BridgeOnlyControlRunnerError(
            "Whole-response review is incomplete or targets the wrong child."
        )

    body = {
        "schema": RUN_SCHEMA,
        "status": "standalone-bridge-only-child-reviewed",
        "run_directory": str(run_root),
        "diagnostic_role": (
            "isolates-v4096-token-bridge-and-native-tied-readout-from-body-folding"
        ),
        "parent_model_directory": str(parent_path),
        "parent_identity_sha256": expected_parent,
        "parent_unique_parameters": expected_parent_unique_parameters,
        "trace_cache_directory": str(cache_path),
        "trace_cache_key": cache.cache_key,
        "sealed_parent_cache_reused_without_recapture": True,
        "bridge_directory": str(bridge_path),
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "target_tokenizer_semantics_sha256": (
            bridge.target_tokenizer_semantics_sha256
        ),
        "blank_target": blank_receipt,
        "fold_receipt": str(fold_receipt_path),
        "fold_receipt_sha256": fold_receipt["result_sha256"],
        "checkpoint": str(checkpoint),
        "checkpoint_artifact_inventory": artifact_inventory,
        "checkpoint_artifact_inventory_sha256": canonical_sha256(artifact_inventory),
        "installed_model_state_sha256": installed_state_sha256,
        "fresh_reload_model_state_sha256": reload_state_sha256,
        "standalone_fresh_reload_verified": True,
        "parent_sidecar_required": False,
        "target_unique_parameters": expected_target_unique_parameters,
        "realized_parameter_compaction_ratio": (
            expected_parent_unique_parameters / expected_target_unique_parameters
        ),
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
            "child_generate_calls": len(prompt_rows),
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
    parser.add_argument("--expected-target-unique-parameters", required=True, type=int)
    parser.add_argument("--blank-seed", type=int, default=0)
    parser.add_argument("--maximum-candidates", type=int, default=64)
    parser.add_argument("--selected-candidate-rank", type=int)
    parser.add_argument("--prompts-json", required=True, type=Path)
    parser.add_argument("--expected-prompts-file-sha256", required=True)
    parser.add_argument("--parent-envelope", required=True, type=Path)
    parser.add_argument("--expected-parent-envelope-sha256", required=True)
    parser.add_argument("--expected-parent-envelope-file-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
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
            expected_parent_identity_sha256=(args.expected_parent_identity_sha256),
            expected_parent_unique_parameters=(args.expected_parent_unique_parameters),
            expected_target_unique_parameters=(args.expected_target_unique_parameters),
            blank_seed=args.blank_seed,
            maximum_candidates=args.maximum_candidates,
            selected_candidate_rank=args.selected_candidate_rank,
            prompts_json=args.prompts_json,
            expected_prompts_file_sha256=args.expected_prompts_file_sha256,
            parent_envelope=args.parent_envelope,
            expected_parent_envelope_sha256=(args.expected_parent_envelope_sha256),
            expected_parent_envelope_file_sha256=(
                args.expected_parent_envelope_file_sha256
            ),
            device=args.device,
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
                "target_unique_parameters": result["target_unique_parameters"],
                "realized_parameter_compaction_ratio": result[
                    "realized_parameter_compaction_ratio"
                ],
                "whole_response_dossier": result["whole_response_dossier"],
                "result_sha256": result["result_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
