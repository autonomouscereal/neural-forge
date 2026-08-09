"""End-to-end frozen-V5 re-score of the A9/A11 base dual-route frontier child.

This is an evaluation-only runtime for open cell (e) of the sealed stage-1
ceiling report.  It reuses, unmodified, the sealed 2026-07-30 capped-prompt
reuse runner (`experiments/rerun_lfm2_sparse_semantic_capped_prompts.py`) for
every authentication step and for the decode call itself.  The one thing it
changes is prompt selection: the sealed runner replays only the source
dossier's generation-limit hits, while the frozen V5 protocol grades the whole
16-case surface at 1,024 initial / 2,048 retry.  So this driver runs all 16
frozen-pack cases through the same authenticated child, route binding and
greedy decode.

It never loads a parent model, captures a parent trace, discovers supports,
solves coefficients, folds parameters, trains, prunes, or quantizes.  It writes
only under its own output root.

Before anything is loaded it verifies three digests and stops on any mismatch:

* the frozen protocol document (`PROTOCOL_SEAL.json` `protocol_sha256`), which
  the seal record asks the review runner to check the same way it checks the
  prompt pack;
* the frozen prompt pack file (`prompt_pack_sha256`);
* the protocol's normalized prompt-set digest, recomputed from the pack through
  the shared `load_review_prompts` normalizer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_SEALED_RUNNER_ROOT = Path(
    os.environ.get(
        "NF_CELL_E_SEALED_RUNNER_ROOT",
        "/home/cereal/model-surgery-lab/neural-forge-highcap-eb0a5460",
    )
).resolve()
for _entry in (str(_SEALED_RUNNER_ROOT), str(_SEALED_RUNNER_ROOT / "src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import torch  # noqa: E402

from experiments import (  # noqa: E402
    rerun_lfm2_sparse_semantic_capped_prompts as sealed,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_direct_local import (  # noqa: E402
    Lfm2DirectLocalCheckpointIO,
    fingerprint_lfm2_direct_local_state,
    load_lfm2_direct_local_tokenizer,
)
from neural_forge.hf_lfm2_semantic_step_token_fold import (  # noqa: E402
    bind_lfm2_separate_sparse_semantic_code_runtime,
    bind_lfm2_sparse_semantic_code_runtime,
)
from neural_forge.hf_lfm2_sparse_semantic_codebook import (  # noqa: E402
    load_lfm2_sparse_semantic_codebook,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    load_strict_tokenizer_bridge_artifact,
)
from neural_forge.lfm2_decoded_review import load_review_prompts  # noqa: E402

PLAN_SCHEMA = "neural-forge-lfm2-frozen-v5-rescore-plan-v1"
RESULT_SCHEMA = "neural-forge-lfm2-frozen-v5-rescore-result-v1"

FROZEN_PROTOCOL_SHA256 = (
    "e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b"
)
FROZEN_PROMPT_PACK_SHA256 = (
    "0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b"
)
FROZEN_NORMALIZED_PROMPT_SET_SHA256 = (
    "5615b9e4cee0727a9c35c0da08ac770541941d42077e9c50ac1d2a03fea9d741"
)
FROZEN_INITIAL_CAP = 1024
FROZEN_RETRY_CAP = 2048
FROZEN_CASE_COUNT = 16


class FrozenV5RescoreError(RuntimeError):
    """A frozen-protocol binding, plan input, or lane artifact is invalid."""


def verify_frozen_protocol_bindings(
    *,
    protocol_file: Path,
    prompts_file: Path,
) -> dict[str, Any]:
    """Stop before any model load unless all three frozen digests verify."""

    protocol_path = sealed._local_file(protocol_file, "frozen protocol document")
    protocol_digest = sealed._file_sha256(protocol_path)
    if protocol_digest != FROZEN_PROTOCOL_SHA256:
        raise FrozenV5RescoreError(
            "Frozen protocol digest mismatch: expected "
            f"{FROZEN_PROTOCOL_SHA256} but the supplied document hashes to "
            f"{protocol_digest}. Refusing to review against an unsealed protocol."
        )
    prompts_path = sealed._local_file(prompts_file, "frozen prompt pack")
    prompts_digest = sealed._file_sha256(prompts_path)
    if prompts_digest != FROZEN_PROMPT_PACK_SHA256:
        raise FrozenV5RescoreError(
            "Frozen prompt pack digest mismatch: expected "
            f"{FROZEN_PROMPT_PACK_SHA256} but got {prompts_digest}."
        )
    prompts = list(load_review_prompts(prompts_path))
    normalized_digest = canonical_sha256(prompts)
    if normalized_digest != FROZEN_NORMALIZED_PROMPT_SET_SHA256:
        raise FrozenV5RescoreError(
            "Normalized prompt-set digest mismatch: expected "
            f"{FROZEN_NORMALIZED_PROMPT_SET_SHA256} but got {normalized_digest}."
        )
    if len(prompts) != FROZEN_CASE_COUNT:
        raise FrozenV5RescoreError(
            f"Frozen surface must carry {FROZEN_CASE_COUNT} cases, saw {len(prompts)}."
        )
    body = {
        "schema": "neural-forge-frozen-v5-protocol-binding-v1",
        "protocol_file": str(protocol_path),
        "protocol_sha256": protocol_digest,
        "prompt_pack_file": str(prompts_path),
        "prompt_pack_sha256": prompts_digest,
        "normalized_prompt_set_sha256": normalized_digest,
        "case_count": len(prompts),
        "case_ids": [str(row["id"]) for row in prompts],
        "initial_new_token_cap": FROZEN_INITIAL_CAP,
        "retry_new_token_cap": FROZEN_RETRY_CAP,
        "decode_rule": "greedy-argmax-donor-logits",
        "verified_before_any_model_load": True,
    }
    return {**body, "binding_sha256": canonical_sha256(body)}


def load_rescore_plan(path: Path) -> dict[str, Any]:
    """Load and self-verify the frozen-V5 re-score plan."""

    source = sealed._local_file(path, "rescore plan")
    plan = sealed._read_json(source, "rescore plan")
    claimed = sealed._sha256(plan.get("plan_sha256"), "plan_sha256")
    body = dict(plan)
    body.pop("plan_sha256")
    if claimed != canonical_sha256(body):
        raise FrozenV5RescoreError("Rescore plan self-hash does not verify.")
    if plan.get("schema") != PLAN_SCHEMA:
        raise FrozenV5RescoreError("Rescore plan schema is unsupported.")
    policy = plan.get("policy")
    shared = plan.get("shared")
    lanes = plan.get("lanes")
    if not isinstance(policy, Mapping) or not isinstance(shared, Mapping):
        raise FrozenV5RescoreError("Rescore plan policy/shared sections are missing.")
    if not isinstance(lanes, list) or not lanes:
        raise FrozenV5RescoreError("Rescore plan must contain lanes.")
    if (
        policy.get("initial_high_cap") != FROZEN_INITIAL_CAP
        or policy.get("adaptive_maximum_cap") != FROZEN_RETRY_CAP
    ):
        raise FrozenV5RescoreError(
            "Rescore plan caps must be the frozen protocol's 1024/2048."
        )
    required_true = (
        "safety_is_separate_from_folding_quality",
        "cap_hit_is_censored_not_failed",
        "score_every_frozen_case",
        "reuse_saved_child_and_routes_only",
        "parent_trace_regeneration_forbidden",
    )
    if any(policy.get(name) is not True for name in required_true):
        raise FrozenV5RescoreError("Rescore plan evaluation policy is incomplete.")
    operation_contract = plan.get("operation_contract")
    expected_operation_contract = {
        "parent_model_load_calls": 0,
        "parent_forward_calls": 0,
        "parent_generate_calls": 0,
        "parent_trace_recaptures": 0,
        "fold_compilations": 0,
        "support_discovery_calls": 0,
        "coefficient_solve_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "pruned_parameters": 0,
        "quantized_parameters": 0,
        "network_access": False,
    }
    if (
        not isinstance(operation_contract, Mapping)
        or dict(operation_contract) != expected_operation_contract
    ):
        raise FrozenV5RescoreError(
            "Rescore plan permits work outside child-only evaluation."
        )
    lane_ids: set[str] = set()
    for lane in lanes:
        if not isinstance(lane, Mapping):
            raise FrozenV5RescoreError("Every lane must be an object.")
        lane_id = str(lane.get("id", "")).strip()
        if not lane_id or lane_id in lane_ids:
            raise FrozenV5RescoreError("Lane IDs are invalid.")
        lane_ids.add(lane_id)
    return plan


def run_lane(
    plan: Mapping[str, Any],
    lane: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    device_override: str | None = None,
) -> dict[str, Any]:
    """Decode all 16 frozen cases from one lane's persisted child and routes."""

    shared = plan["shared"]
    policy = plan["policy"]
    lane_id = str(lane["id"])
    print(
        json.dumps(
            {
                "stage": "lane-preflight-start",
                "lane_id": lane_id,
                "device_override": device_override,
                "protocol_binding_sha256": binding["binding_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    # The 256-cap source dossier is authenticated for provenance and to supply
    # the legacy per-case comparison rows; it no longer selects the prompts.
    source_dossier, source_capped_rows = sealed.authenticate_lane_source(
        {
            **lane,
            "source_cap": lane["source_cap"],
            "capped_prompt_ids": list(lane["legacy_capped_prompt_ids"]),
        }
    )
    prompts_path = sealed._local_file(shared["prompts_json"], "review prompt file")
    if sealed._file_sha256(prompts_path) != FROZEN_PROMPT_PACK_SHA256:
        raise FrozenV5RescoreError("Review prompt bytes changed.")
    prompts = list(load_review_prompts(prompts_path))
    selected_ids = [str(row["id"]) for row in prompts]
    if selected_ids != list(binding["case_ids"]):
        raise FrozenV5RescoreError("Frozen case order changed between checks.")

    checkpoint = sealed._local_directory(shared["checkpoint_directory"], "checkpoint")
    checkpoint_inventory_sha256 = sealed._checkpoint_inventory_sha256(checkpoint)
    if checkpoint_inventory_sha256 != sealed._sha256(
        shared["checkpoint_artifact_inventory_sha256"],
        "checkpoint artifact inventory SHA-256",
    ):
        raise FrozenV5RescoreError("Checkpoint artifact inventory changed.")
    donor_tokenizer = load_lfm2_direct_local_tokenizer(
        sealed._local_directory(
            shared["donor_tokenizer_directory"], "donor tokenizer directory"
        )
    )
    target_tokenizer = load_lfm2_direct_local_tokenizer(
        sealed._local_directory(
            shared["target_tokenizer_directory"], "target tokenizer directory"
        )
    )
    bridge = load_strict_tokenizer_bridge_artifact(
        sealed._local_directory(shared["bridge_directory"], "bridge"),
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        expected_artifact_sha256=shared["bridge_artifact_sha256"],
    )
    device = str(device_override or shared.get("runtime_device", "cuda:0"))
    runtime_dtype, dtype_receipt = sealed.resolve_runtime_dtype(
        str(shared.get("runtime_dtype", "fp32")), device
    )
    provenance_dtype = sealed.resolve_checkpoint_provenance_dtype(
        str(shared.get("checkpoint_provenance_dtype", "bf16"))
    )
    checkpoint_io = Lfm2DirectLocalCheckpointIO(
        tokenizer_bridge=bridge,
        device=torch.device(str(shared.get("checkpoint_provenance_device", "cpu"))),
        dtype=provenance_dtype,
    )
    child, load_receipt = checkpoint_io.load_authenticated_portable(
        checkpoint,
        runtime_device=torch.device(device),
        runtime_dtype=runtime_dtype,
    )
    child.requires_grad_(False)
    child.eval()
    physical_parameters = sum(int(parameter.numel()) for parameter in child.parameters())
    if physical_parameters != sealed._positive_int(
        shared["physical_child_parameter_count"], "physical child parameters"
    ):
        raise FrozenV5RescoreError("Loaded child parameter count changed.")
    child_state_sha256 = fingerprint_lfm2_direct_local_state(child)

    span_evidence = sealed._read_sealed_json(
        sealed._local_file(shared["span_evidence_file"], "span evidence"),
        seal_field="evidence_sha256",
        expected_sha256=shared["span_evidence_sha256"],
        name="span evidence",
    )
    codebook = load_lfm2_sparse_semantic_codebook(
        sealed._local_directory(lane["codebook_directory"], "sparse codebook"),
        expected_artifact_sha256=lane["codebook_artifact_sha256"],
    )
    output = lane.get("functional_output")
    output_coefficients = None
    output_authentication = None
    if output is not None:
        output_coefficients, output_authentication = (
            sealed.load_authenticated_functional_output_refit(
                coefficient_tensor_file=sealed._local_file(
                    output["coefficient_tensor_file"], "functional output coefficients"
                ),
                sealed_refit_receipt_file=sealed._local_file(
                    output["sealed_receipt_file"], "functional output receipt"
                ),
                expected_math_cache_key=sealed._sha256(
                    output["math_cache_key"], "functional output cache key"
                ),
                expected_result_sha256=sealed._sha256(
                    output["result_sha256"], "functional output result"
                ),
                selected_codebook=codebook,
            )
        )
    live_table = child.get_input_embeddings().weight
    if output_coefficients is None:
        routing = bind_lfm2_sparse_semantic_code_runtime(
            span_evidence,
            live_table,
            semantic_code_offsets=codebook.semantic_code_offsets,
            semantic_code_target_ids=codebook.semantic_code_target_ids,
            semantic_code_coefficients=codebook.semantic_code_coefficients,
            target_to_donor=codebook.target_to_donor,
        )
    else:
        routing = bind_lfm2_separate_sparse_semantic_code_runtime(
            span_evidence,
            live_table,
            input_semantic_code_offsets=codebook.semantic_code_offsets,
            input_semantic_code_target_ids=codebook.semantic_code_target_ids,
            input_semantic_code_coefficients=codebook.semantic_code_coefficients,
            output_semantic_code_offsets=codebook.semantic_code_offsets,
            output_semantic_code_target_ids=codebook.semantic_code_target_ids,
            output_semantic_code_coefficients=output_coefficients,
            target_to_donor=codebook.target_to_donor,
        )
    routing_receipt = sealed._plain_json(routing.as_dict())
    routing_contract_sha256 = canonical_sha256(routing_receipt)
    output_root = sealed._output_directory(lane["output_directory"])

    legacy_by_id = {
        str(row["id"]): row
        for row in source_dossier.get("comparisons", [])
        if isinstance(row, Mapping)
    }
    legacy_capped_ids = {str(row["id"]) for row in source_capped_rows}

    prompt_results = []
    for prompt in prompts:
        attempts = []
        for cap in (FROZEN_INITIAL_CAP, FROZEN_RETRY_CAP):
            if attempts and attempts[-1]["termination_evidence"].get(
                "censored_by_output_cap"
            ) is not True:
                break
            attempt = sealed._load_or_run_attempt(
                output_root=output_root,
                lane_id=lane_id,
                prompt=prompt,
                cap=cap,
                child=child,
                routing=routing,
                donor_tokenizer=donor_tokenizer,
                target_tokenizer=target_tokenizer,
                source_dossier_sha256=str(source_dossier["dossier_sha256"]),
                child_state_sha256=child_state_sha256,
                routing_contract_sha256=routing_contract_sha256,
                maximum_prompt_tokens=sealed._positive_int(
                    shared["maximum_prompt_tokens"], "maximum prompt tokens"
                ),
                maximum_embedding_rows_per_chunk=sealed._positive_int(
                    shared["maximum_embedding_rows_per_chunk"],
                    "maximum embedding rows per chunk",
                ),
                maximum_donor_rows_per_score_chunk=sealed._positive_int(
                    shared["maximum_donor_rows_per_score_chunk"],
                    "maximum donor rows per score chunk",
                ),
            )
            attempts.append(attempt)
        prompt_id = str(prompt["id"])
        prompt_results.append(
            {
                "id": prompt_id,
                "category": prompt.get("category"),
                "prompt": sealed._plain_json(prompt),
                "legacy_256_cap_comparison": sealed._plain_json(
                    legacy_by_id.get(prompt_id)
                ),
                "legacy_256_cap_was_censored": prompt_id in legacy_capped_ids,
                "attempts": attempts,
                "authoritative_frozen_attempt": attempts[-1],
                "manual_review_instruction": (
                    "Grade against the frozen V5 per-case anchor for this ID. "
                    "Record termination separately. A cap hit is censored, not a "
                    "failure. Safety is separate from intelligence."
                ),
            }
        )
    body = {
        "schema": RESULT_SCHEMA,
        "status": "complete-awaiting-manual-frozen-v5-review",
        "lane_id": lane_id,
        "lane_label": str(lane["label"]),
        "counted_compaction": lane["counted_compaction"],
        "body_contraction": lane["body_contraction"],
        "frozen_protocol_binding": sealed._plain_json(binding),
        "legacy_review_file": str(lane["legacy_review_file"]),
        "legacy_review_tally": sealed._plain_json(lane["legacy_review_tally"]),
        "source_dossier_sha256": source_dossier["dossier_sha256"],
        "source_cap": int(policy["source_cap"]),
        "initial_high_cap": FROZEN_INITIAL_CAP,
        "adaptive_maximum_cap": FROZEN_RETRY_CAP,
        "prompt_count": len(prompt_results),
        "prompt_ids": selected_ids,
        "prompt_results": prompt_results,
        "child_state_sha256": child_state_sha256,
        "physical_child_parameter_count": physical_parameters,
        "codebook_artifact_sha256": codebook.artifact_sha256,
        "routing_contract_sha256": routing_contract_sha256,
        "routing_receipt": routing_receipt,
        "functional_output_authentication": output_authentication,
        "checkpoint_portable_load_receipt": sealed._plain_json(load_receipt.as_dict()),
        "checkpoint_artifact_inventory_sha256": checkpoint_inventory_sha256,
        "runtime_dtype_receipt": sealed._plain_json(dtype_receipt),
        "evaluation_policy": sealed._plain_json(policy),
        "operation_audit": {
            "parent_model_load_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "fold_compilations": 0,
            "support_discovery_calls": 0,
            "coefficient_solve_calls": 0,
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "pruned_parameters": 0,
            "quantized_parameters": 0,
            "network_access": False,
            "every_frozen_case_decoded": True,
        },
        "automatic_acceptance_or_rejection_performed": False,
        "safety_is_separate_from_folding_quality": True,
        "generation_limit_hit_is_censored_not_failed": True,
    }
    result = {**body, "result_sha256": canonical_sha256(body)}
    sealed._atomic_write_json(output_root / "frozen-v5-rescore-dossier.json", result)
    print(
        json.dumps(
            {
                "stage": "lane-complete",
                "lane_id": lane_id,
                "prompt_count": len(prompt_results),
                "result_sha256": result["result_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--lane", action="append")
    parser.add_argument("--device")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = load_rescore_plan(args.manifest)
    binding = verify_frozen_protocol_bindings(
        protocol_file=args.protocol,
        prompts_file=Path(plan["shared"]["prompts_json"]),
    )
    print(json.dumps({"stage": "protocol-binding-verified", **binding}, sort_keys=True), flush=True)
    requested = None if not args.lane else set(args.lane)
    lanes = [
        lane for lane in plan["lanes"] if requested is None or str(lane["id"]) in requested
    ]
    if requested is not None and {str(lane["id"]) for lane in lanes} != requested:
        raise FrozenV5RescoreError("One or more requested lanes are unknown.")
    results = [
        run_lane(plan, lane, binding, device_override=args.device) for lane in lanes
    ]
    print(
        json.dumps(
            {
                "status": "complete",
                "protocol_binding_sha256": binding["binding_sha256"],
                "lanes": [
                    {
                        "id": row["lane_id"],
                        "prompt_count": row["prompt_count"],
                        "result_sha256": row["result_sha256"],
                    }
                    for row in results
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FrozenV5RescoreError",
    "load_rescore_plan",
    "main",
    "run_lane",
    "verify_frozen_protocol_bindings",
]
