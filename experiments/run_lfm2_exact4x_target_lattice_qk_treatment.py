#!/usr/bin/env python3
"""Treat one exact-4x saved body with causal target-lattice Q/K closure.

The original structural body is freshly authenticated and loaded locally.  A
single analytic pass then adjusts only Q/K, in causal layer order, using the
immutable semantic cache and the exact Q-DEIM head order already planted in
the child.  The treated body is saved and freshly reloaded before the existing
six-head frontier performs its normal full-response decode.  No parent model,
training, gradients, optimizer, pruning, quantization, or cache recapture is
used.
"""

from __future__ import annotations

import argparse
import gc
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

from experiments import run_lfm2_exact4x_causal_head_frontier as frontier  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_fold_executor import (  # noqa: E402
    fingerprint_model_state,
    fingerprint_model_state_entries,
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
from neural_forge.hf_lfm2_saved_body_target_lattice_qk import (  # noqa: E402
    LFM2_SAVED_BODY_TARGET_LATTICE_QK_SCHEMA,
    authenticate_lfm2_saved_attention_lineage,
    compile_install_lfm2_saved_body_target_lattice_qk_closure,
    extract_lfm2_qdeim_head_selections,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    fingerprint_hf_tokenizer_semantics,
    load_strict_tokenizer_bridge_artifact,
)


RUN_SCHEMA = "neural-forge-lfm2-exact4x-target-lattice-qk-treatment-run-v1"
FAILURE_SCHEMA = (
    "neural-forge-lfm2-exact4x-target-lattice-qk-treatment-failure-v1"
)


class Lfm2Exact4xTargetLatticeQkTreatmentError(RuntimeError):
    """The exact-4x saved-body Q/K treatment failed authentication."""


def _emit(stage: str, **values: Any) -> None:
    print(
        json.dumps({"stage": stage, **values}, allow_nan=False, sort_keys=True),
        flush=True,
    )


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(frontier._ordinary_file(path, label).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Lfm2Exact4xTargetLatticeQkTreatmentError(
            f"{label} must be a JSON object."
        )
    return value


def _changed_entries(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> list[str]:
    return sorted(
        name
        for name in set(before).union(after)
        if before.get(name) != after.get(name)
    )


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
    expected_source_structural_body_receipt_sha256: str,
    expected_attention_lineage_file_sha256: str,
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
        raise RuntimeError("PyTorch is required for exact-4x Q/K treatment.") from exc

    run_root = frontier._new_run_directory(run_directory)
    source_root = frontier._ordinary_directory(source_run_directory, "source run")
    baseline_checkpoint = frontier._ordinary_directory(
        source_root / "structural-body-checkpoint",
        "source structural-body checkpoint",
    )
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
    expected_parent = frontier._digest(
        expected_parent_identity_sha256,
        "expected parent identity",
    )
    expected_bridge = frontier._digest(
        expected_bridge_artifact_sha256,
        "expected bridge artifact",
    )
    expected_cache = frontier._digest(
        expected_semantic_cache_key,
        "expected semantic cache key",
    )

    _emit("authenticate-source-body-and-qdeim-lineage")
    structural_receipt = frontier._load_sealed_json(
        source_root / "structural-interface-install.json",
        seal_field="result_sha256",
        expected_seal=expected_source_structural_interface_sha256,
        expected_schema=frontier.LFM2_INDEPENDENT_STRUCTURAL_INTERFACE_INSTALL_SCHEMA,
        label="source structural-interface receipt",
    )
    deferred_receipt = frontier._load_sealed_json(
        source_root / "deferred-final-w2-install.json",
        seal_field="result_sha256",
        expected_seal=expected_source_deferred_w2_sha256,
        expected_schema=frontier.DEFERRED_W2_SCHEMA,
        label="source deferred-W2 receipt",
    )
    semantic_stack_receipt = frontier._load_sealed_json(
        source_root / "semantic-layer-stack.json",
        seal_field="result_sha256",
        expected_seal=str(deferred_receipt.get("semantic_layer_stack_result_sha256")),
        expected_schema=frontier.SEMANTIC_LAYER_STACK_SCHEMA,
        label="source semantic layer-stack receipt",
    )
    frontier._authenticate_source_fold_receipts(
        structural_receipt=structural_receipt,
        semantic_layer_stack_receipt=semantic_stack_receipt,
        deferred_w2_receipt=deferred_receipt,
    )
    attention_lineage_path = frontier._ordinary_file(
        source_root / "attention-lineage.json",
        "source attention lineage",
    )
    if frontier._file_sha256(attention_lineage_path) != frontier._digest(
        expected_attention_lineage_file_sha256,
        "expected attention-lineage file",
    ):
        raise Lfm2Exact4xTargetLatticeQkTreatmentError(
            "Saved attention-lineage file differs from its external pin."
        )
    attention_lineage_document = _load_json(
        attention_lineage_path,
        label="source attention lineage",
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
        raise Lfm2Exact4xTargetLatticeQkTreatmentError(
            "Semantic cache, tokenizers, bridge, or parent ancestry changed."
        )
    checkpoint_io = Lfm2DirectLocalCheckpointIO(
        tokenizer_bridge=bridge,
        device=torch.device(model_device),
        dtype=torch.bfloat16,
    )
    baseline_inventory = frontier._authenticate_source_checkpoint_inventory(
        baseline_checkpoint,
        expected_sha256=frontier._digest(
            expected_source_checkpoint_inventory_sha256,
            "expected source checkpoint inventory",
        ),
    )
    baseline = checkpoint_io.load(baseline_checkpoint)
    baseline.requires_grad_(False)
    baseline.eval()
    baseline_state = fingerprint_model_state(baseline)
    baseline_entries = dict(fingerprint_model_state_entries(baseline))
    baseline_fingerprint = fingerprint_lfm2_parent(baseline)
    baseline_body_inventory = fingerprint_lfm2_causal_body_inventory(baseline)
    baseline_topology = frontier._target_topology(baseline)
    observed_topology = {
        key: baseline_topology[key] for key in frontier.EXPECTED_CHILD_TOPOLOGY
    }
    baseline_inventory_sha256 = canonical_sha256(baseline_inventory)
    optional_body_receipt = frontier._verify_optional_structural_body_receipt(
        source_root,
        expected_receipt_sha256=(
            expected_source_structural_body_receipt_sha256
        ),
        checkpoint_inventory_sha256=baseline_inventory_sha256,
        body_state_sha256=baseline_state,
        body_identity_sha256=baseline_fingerprint.identity_sha256,
        body_inventory_sha256=baseline_body_inventory,
        structural_install_sha256=structural_receipt["result_sha256"],
        semantic_layer_stack_sha256=semantic_stack_receipt["result_sha256"],
        deferred_w2_sha256=deferred_receipt["result_sha256"],
    )
    if (
        optional_body_receipt is None
        or deferred_receipt.get("model_state_after_sha256") != baseline_state
        or frontier._unique_parameter_count(baseline)
        != frontier.EXPECTED_CHILD_UNIQUE_PARAMETERS
        or observed_topology != frontier.EXPECTED_CHILD_TOPOLOGY
        or baseline.get_input_embeddings().weight
        is baseline.get_output_embeddings().weight
        or not bool(
            torch.equal(
                baseline.get_input_embeddings().weight.detach().cpu(),
                baseline.get_output_embeddings().weight.detach().cpu(),
            )
        )
    ):
        raise Lfm2Exact4xTargetLatticeQkTreatmentError(
            "Fresh source body differs from the pinned exact-4x structural body."
        )
    compiler_lineage = authenticate_lfm2_saved_attention_lineage(
        attention_lineage_document
    )
    selections = extract_lfm2_qdeim_head_selections(semantic_stack_receipt)

    _emit("compile-install-causal-target-lattice-qk-only")
    treatment = dict(
        compile_install_lfm2_saved_body_target_lattice_qk_closure(
            cache=cache,
            bridge=bridge,
            target_model=baseline,
            compiler_lineage=compiler_lineage,
            qdeim_head_selections=selections,
            donor_tokenizer=donor_tokenizer,
            target_tokenizer=target_tokenizer,
        )
    )
    treated_state = fingerprint_model_state(baseline)
    treated_entries = dict(fingerprint_model_state_entries(baseline))
    treated_fingerprint = fingerprint_lfm2_parent(baseline)
    treated_body_inventory = fingerprint_lfm2_causal_body_inventory(baseline)
    changed_state_keys = _changed_entries(baseline_entries, treated_entries)
    if (
        treatment.get("schema") != LFM2_SAVED_BODY_TARGET_LATTICE_QK_SCHEMA
        or treatment.get("model_state_before_sha256")
        != baseline_fingerprint.identity_sha256
        or treatment.get("model_state_after_sha256")
        != treated_fingerprint.identity_sha256
        or treatment.get("changed_state_keys") != changed_state_keys
        or treatment.get("actual_changed_state_keys_are_qk_only") is not True
        or not changed_state_keys
    ):
        raise Lfm2Exact4xTargetLatticeQkTreatmentError(
            "Target-lattice treatment receipt differs from the live Q/K transition."
        )
    frontier._write_json(run_root / "target-lattice-qk-treatment.json", treatment)

    _emit("save-and-fresh-reload-standalone-treated-body")
    treated_root = run_root / "treated-body"
    treated_root.mkdir()
    treated_checkpoint = treated_root / "checkpoint"
    treated_checkpoint.mkdir()
    checkpoint_io.save(baseline, treated_checkpoint)
    del baseline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    treated_checkpoint_inventory = frontier._artifact_inventory(treated_checkpoint)
    treated_checkpoint_inventory_sha256 = canonical_sha256(
        treated_checkpoint_inventory
    )
    frontier._authenticate_source_checkpoint_inventory(
        treated_checkpoint,
        expected_sha256=treated_checkpoint_inventory_sha256,
    )
    reloaded = checkpoint_io.load(treated_checkpoint)
    reloaded.requires_grad_(False)
    reloaded.eval()
    reloaded_state = fingerprint_model_state(reloaded)
    reloaded_entries = dict(fingerprint_model_state_entries(reloaded))
    reloaded_fingerprint = fingerprint_lfm2_parent(reloaded)
    reloaded_body_inventory = fingerprint_lfm2_causal_body_inventory(reloaded)
    changed_checkpoint_paths = frontier._changed_checkpoint_paths(
        baseline_inventory,
        treated_checkpoint_inventory,
    )
    if (
        reloaded_state != treated_state
        or reloaded_entries != treated_entries
        or reloaded_fingerprint.as_dict() != treated_fingerprint.as_dict()
        or reloaded_body_inventory != treated_body_inventory
        or frontier._target_topology(reloaded) != baseline_topology
        or frontier._unique_parameter_count(reloaded)
        != frontier.EXPECTED_CHILD_UNIQUE_PARAMETERS
        or reloaded.get_input_embeddings().weight
        is reloaded.get_output_embeddings().weight
        or changed_checkpoint_paths != ["model.safetensors"]
    ):
        raise Lfm2Exact4xTargetLatticeQkTreatmentError(
            "Standalone treated-body fresh reload changed the Q/K-only result."
        )
    del reloaded
    gc.collect()
    provenance_body = {
        "schema": frontier.QK_TREATED_BODY_PROVENANCE_SCHEMA,
        "status": "saved-reloaded-qk-only-treated-body",
        "source_structural_body_receipt_sha256": optional_body_receipt[
            "receipt_sha256"
        ],
        "source_checkpoint_inventory_sha256": baseline_inventory_sha256,
        "source_body_model_state_sha256": baseline_state,
        "source_body_lfm2_identity_sha256": baseline_fingerprint.identity_sha256,
        "source_causal_body_inventory_sha256": baseline_body_inventory,
        "treated_checkpoint_inventory_sha256": (
            treated_checkpoint_inventory_sha256
        ),
        "treated_body_model_state_sha256": treated_state,
        "treated_body_lfm2_identity_sha256": treated_fingerprint.identity_sha256,
        "treated_causal_body_inventory_sha256": treated_body_inventory,
        "changed_checkpoint_relative_paths": changed_checkpoint_paths,
        "changed_state_keys": changed_state_keys,
        "target_lattice_qk_treatment_sha256": treatment["result_sha256"],
        "causal_output_head_installed": False,
        "standalone_fresh_reload_verified": True,
        "operation_audit": dict(frontier.QK_TREATED_BODY_ZERO_OPERATION_AUDIT),
    }
    provenance = {
        **provenance_body,
        "receipt_sha256": canonical_sha256(provenance_body),
    }
    frontier._write_json(treated_root / "treated-body-provenance.json", provenance)

    _emit("reuse-six-head-frontier-and-retain-all-raw-decoded-outputs")
    downstream = frontier.run(
        run_directory=run_root / "six-head-frontier",
        source_run_directory=source_root,
        donor_tokenizer_directory=donor_path,
        target_tokenizer_directory=target_path,
        bridge_directory=bridge_path,
        expected_bridge_artifact_sha256=expected_bridge,
        semantic_cache_directory=cache_path,
        expected_semantic_cache_key=expected_cache,
        expected_parent_identity_sha256=expected_parent,
        expected_source_structural_interface_sha256=(
            expected_source_structural_interface_sha256
        ),
        expected_source_deferred_w2_sha256=expected_source_deferred_w2_sha256,
        expected_source_terminal_evidence_sha256=(
            expected_source_terminal_evidence_sha256
        ),
        expected_source_causal_trie_sha256=expected_source_causal_trie_sha256,
        expected_source_sequence_plan_sha256=expected_source_sequence_plan_sha256,
        expected_source_checkpoint_inventory_sha256=(
            expected_source_checkpoint_inventory_sha256
        ),
        expected_source_structural_body_receipt_sha256=(
            expected_source_structural_body_receipt_sha256
        ),
        prompts_json=prompts_json,
        expected_prompts_file_sha256=expected_prompts_file_sha256,
        parent_envelope=parent_envelope,
        expected_parent_envelope_sha256=expected_parent_envelope_sha256,
        expected_parent_envelope_file_sha256=(
            expected_parent_envelope_file_sha256
        ),
        model_device=model_device,
        analytic_solve_device=analytic_solve_device,
        maximum_prompt_tokens=maximum_prompt_tokens,
        maximum_new_tokens=maximum_new_tokens,
        treated_body_directory=treated_root,
        expected_treated_body_provenance_sha256=provenance["receipt_sha256"],
    )
    body = {
        "schema": RUN_SCHEMA,
        "status": "qk-treated-body-saved-reloaded-and-six-head-frontier-decoded",
        "run_directory": str(run_root),
        "source_run_directory": str(source_root),
        "target_lattice_qk_treatment_sha256": treatment["result_sha256"],
        "treated_body_provenance_sha256": provenance["receipt_sha256"],
        "treated_checkpoint_inventory_sha256": treated_checkpoint_inventory_sha256,
        "changed_state_keys": changed_state_keys,
        "six_head_frontier_result_sha256": downstream["result_sha256"],
        "whole_response_dossier_sha256": downstream[
            "whole_response_dossier_sha256"
        ],
        "all_six_heads_materialized_and_decoded": True,
        "all_raw_outputs_retained": True,
        "numeric_metrics_have_final_intelligence_authority": False,
        "whole_decoded_behavior_is_primary_intelligence_evidence": True,
        "operation_audit": dict(frontier.QK_TREATED_BODY_ZERO_OPERATION_AUDIT),
    }
    result = {**body, "result_sha256": canonical_sha256(body)}
    frontier._write_json(run_root / "run-receipt.json", result)
    _emit("complete", result_sha256=result["result_sha256"])
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
    parser.add_argument("--expected-source-terminal-evidence-sha256", required=True)
    parser.add_argument("--expected-source-causal-trie-sha256", required=True)
    parser.add_argument("--expected-source-sequence-plan-sha256", required=True)
    parser.add_argument(
        "--expected-source-checkpoint-inventory-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-source-structural-body-receipt-sha256",
        required=True,
    )
    parser.add_argument("--expected-attention-lineage-file-sha256", required=True)
    parser.add_argument("--prompts-json", required=True, type=Path)
    parser.add_argument("--expected-prompts-file-sha256", required=True)
    parser.add_argument("--parent-envelope", required=True, type=Path)
    parser.add_argument("--expected-parent-envelope-sha256", required=True)
    parser.add_argument("--expected-parent-envelope-file-sha256", required=True)
    parser.add_argument("--model-device", default="cuda:0")
    parser.add_argument("--analytic-solve-device", default="cuda:0")
    parser.add_argument("--maximum-prompt-tokens", type=int, default=1024)
    parser.add_argument("--maximum-new-tokens", type=int, default=256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run(**vars(args))
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
                frontier._write_json(path, failure)
        print(json.dumps(failure, allow_nan=False, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
