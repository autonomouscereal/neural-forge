"""Emit RESCORE_RECEIPT.json: the auditable index for open cell (e).

Digests every file in this directory, restates the verdicts and the delta against
the legacy tally, and records the server-side provenance. Self-hashed the same way
the rest of the estate does it (canonical JSON, sorted keys, ASCII, no NaN).
"""

import hashlib
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
SELF = os.path.basename(__file__)
RECEIPT = "RESCORE_RECEIPT.json"


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(8 * 1024 ** 2):
            digest.update(block)
    return digest.hexdigest()


inventory = []
for name in sorted(os.listdir(BASE)):
    path = os.path.join(BASE, name)
    if not os.path.isfile(path) or name in (RECEIPT, SELF):
        continue
    inventory.append(
        {
            "relative_path": name,
            "size_bytes": os.path.getsize(path),
            "sha256": file_sha256(path),
        }
    )

body = {
    "schema": "neural-forge-frontier-frozen-rescore-receipt-v1",
    "status": "complete-frozen-v5-rescore-anchor-revised",
    "review_date_utc": "2026-08-08",
    "reviewer": "claude-opus-5",
    "cell": (
        "docs/STAGE1_CEILING_REPORT_2026-08-08.md §6.1(e) — end-to-end frozen-V5 "
        "re-score of the A9/A11 base dual-route frontier child"
    ),
    "repository_head_at_start": "4e6d0ed6",
    "repository_head_at_finish": "79c68eee",
    "concurrent_commits_during_this_cell": {
        "commits": ["fec82744", "79c68eee"],
        "author": "a concurrent agent on branch codex/lrc-blank-compression",
        "files_touched": [
            "docs/STAGE2_FIRST_MEASUREMENT_DESIGN_2026-08-08.md",
            "src/neural_forge/hf_lfm2_circuit_trace_row_selection.py",
            "src/neural_forge/hf_lfm2_sealed_evidence_set.py",
            "tests/test_hf_lfm2_sealed_evidence_set.py",
        ],
        "intersects_reviewed_inputs": False,
        "digests_rechecked_at_finish": {
            "V5_FROZEN_MANUAL_REVIEW_PROTOCOL.md": "e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b",
            "lfm2_semantic_whole_response_prompts_v1.json": "0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b",
            "rerun_lfm2_sparse_semantic_capped_prompts.py": "6b480f6eb7be5cb3c5e2697f223b87c05f376f3b5dca6c6ade1614db0cc723cf",
        },
    },
    "git_operations_performed_by_this_review": [],
    "frozen_protocol": {
        "seal_record": "reports/r12-certified-20260804/PROTOCOL_SEAL.json",
        "protocol_file": "reports/r12-certified-20260804/V5_FROZEN_MANUAL_REVIEW_PROTOCOL.md",
        "protocol_sha256": "e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b",
        "protocol_digest_verified_on_workstation": True,
        "protocol_digest_verified_on_decode_host_before_model_load": True,
        "prompt_pack_file": "research/lfm2_semantic_whole_response_prompts_v1.json",
        "prompt_pack_sha256": "0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b",
        "normalized_prompt_set_sha256": "5615b9e4cee0727a9c35c0da08ac770541941d42077e9c50ac1d2a03fea9d741",
        "protocol_binding_sha256": "aad175bb78c53324228d99e1442bd50b8d6c65092241810a1b12f840c8dc10aa",
        "initial_new_token_cap": 1024,
        "retry_new_token_cap": 2048,
        "decode_rule": "greedy-argmax-donor-logits",
        "named_runtime_deviation": (
            "The protocol names neural_forge.lfm2_carrier_review_cli as the runtime. "
            "That CLI authenticates a carrier child package and cannot decode a "
            "route-only bridge publication over an unmodified base table, so it was "
            "not used. The decode ran through the sealed reuse runner "
            "experiments/rerun_lfm2_sparse_semantic_capped_prompts.py, imported "
            "unmodified; equivalence is proved byte-for-byte against the 2026-07-30 "
            "high-cap attempts in FRONTIER_FROZEN_RESCORE.md section 3.2."
        ),
    },
    "artifact_under_review": {
        "publication_directory": (
            "/home/cereal/model-surgery-lab/artifacts/neural_forge_causal_support_20260730/"
            "a9-a11-base-dual-route/4439a2b7-b301539b"
        ),
        "artifact_sha256": "860bccfe6be46e543f67c8514f3f3ff7dc0d729eb82267f66a9ce1766e43d7bb",
        "completion_sha256": "55a550aaf48189f9989d799b49980f7d61b59134d4f4aa8edc4cfa7e3c84bfb3",
        "package_schema": "neural-forge-lfm2-carrier-reclosed-bridge-v1",
        "package_role": "proven-coherent-unchanged-table-a9-c-in-a11-c-out",
        "carries_body": False,
        "body_checkpoint": (
            "/home/cereal/model-surgery-lab/artifacts/lfm2_bridge_only_control_20260723/"
            "89cd45b0/same-body-v4096-p40-gpu1-a0001/checkpoint"
        ),
        "body_checkpoint_inventory_sha256": "41007288f9a1a7b9dcdf96b4d1d39a1eeec416c46fb928b5bb2cc35ea9cbf8b2",
    },
    "lanes": {
        "a9-k32-residual-omp": {
            "counted_compaction": "1.361447x",
            "body_contraction": "1.000x",
            "dossier_file_sha256": "5dbdac7ade6abb680dc57626f61e4c06bb16e081025751079a7ecae0029eb3f6",
            "result_sha256": "793091bd7e63de1567559268e81704330f769ff15089c6b34aba68c1f0b2bc21",
            "child_state_sha256": "54b9c5f7f484fd256a985aa33346f1bf4f601bec5f1978c287341d9e27f36e85",
            "frozen_useful": 3,
            "frozen_partial": 5,
            "frozen_observed_miss": 7,
            "frozen_useful_or_partial_of_15": 8,
            "frozen_safety_record": "concerning",
            "termination_complete": 14,
            "termination_censored": 2,
            "termination_runtime_error": 0,
            "legacy_published_useful_or_partial_of_16": 10,
            "legacy_ex_safety_useful_or_partial_of_15": 9,
            "grades_revised": ["explanation-seasons"],
            "sensitivity_interval_of_15": [6, 9],
        },
        "a11-k32-split-cout": {
            "counted_compaction": "1.346018x",
            "body_contraction": "1.000x",
            "dossier_file_sha256": "4d3c053b94f5febfe2942f92ca8a66992935f2aee716144748c7781ab6e2ab3f",
            "result_sha256": "d034042fd620e059ecb493bb3b82e817e47393f0d3898e0ca7d5f4e2ccb5f350",
            "child_state_sha256": "54b9c5f7f484fd256a985aa33346f1bf4f601bec5f1978c287341d9e27f36e85",
            "frozen_useful": 2,
            "frozen_partial": 7,
            "frozen_observed_miss": 6,
            "frozen_useful_or_partial_of_15": 9,
            "frozen_safety_record": "concerning",
            "termination_complete": 16,
            "termination_censored": 0,
            "termination_runtime_error": 0,
            "legacy_published_useful_or_partial_of_16": 12,
            "legacy_ex_safety_useful_or_partial_of_15": 11,
            "grades_revised": [
                "summarization-priorities",
                "writing-concise-email",
                "creative-continuity",
            ],
            "sensitivity_interval_of_15": [9, 11],
        },
    },
    "outcome": {
        "verdict": "REVISED",
        "legacy_published_band": "10-12 of 16",
        "frozen_band": "8-9 of 15",
        "suite_conclusion_both_lanes": "promising but incomplete",
        "parent_comparison_both_lanes": "parent-not-demonstrated",
        "structural_claims_of_ceiling_report_affected": False,
        "consequence": (
            "Ceiling report section 2.4 bullet 1 becomes a correction rather than a "
            "caveat. Seven statements in docs/STAGE1_CEILING_REPORT_2026-08-08.md now "
            "carry a superseded number or a closed open cell; they are enumerated in "
            "FRONTIER_FROZEN_RESCORE.md section 9.1. The report was not edited by this "
            "review."
        ),
    },
    "equivalence_proof": {
        "overlapping_attempts_with_sealed_20260730_highcap": 15,
        "decoded_responses_byte_identical": 15,
        "attempt_records_hash_identical": 1,
        "explained_difference": (
            "routing_session_receipt.runtime_prepared_semantic_code_cache_hit_count and "
            "…use_count are per-process monotone counters; the two runs reach the shared "
            "prompts at different ordinals. Response text, token ids, termination "
            "evidence and diagnostics match on all 15."
        ),
    },
    "confounds_recorded": {
        "cap_changed": "256 -> 1024 initial / 2048 retry; 13 legacy censored rows became 2",
        "numeric_lane_changed": (
            "legacy decode ran bfloat16 on Tesla P40 SM 6.1 (emulated BF16, the caveat "
            "recorded in the A11 review); frozen decode ran float32 on RTX 3090 SM 8.6"
        ),
        "legacy_eos_rows_decoding_differently": 8,
        "legacy_eos_rows_total": 19,
        "grading_protocol_changed": (
            "pre-V5 BEHAVIOR_EVIDENCE_DECISION_POLICY_2026-07-23 (no per-case anchors) "
            "-> frozen V5 (16 per-case anchors)"
        ),
        "grades_revised_of_30": 4,
    },
    "server_provenance": {
        "host": "ai (192.168.50.222)",
        "working_directory": "/home/cereal/nf-cell-e/",
        "gpu": "GPU-2ce6a95c-7b1f-644a-3808-d781732a0582 (RTX 3090, index 4)",
        "sealed_runner_root": "/home/cereal/model-surgery-lab/neural-forge-highcap-eb0a5460",
        "sealed_runner_sha256": "6b480f6eb7be5cb3c5e2697f223b87c05f376f3b5dca6c6ade1614db0cc723cf",
        "plan_sha256": "a82c753885bdf31ec6e69336d70584230af7f4cae0d445060ac0ce61e862761a",
        "sealed_artifact_directories_modified": 0,
        "sealed_artifact_directories_checked": [
            "artifacts/neural_forge_causal_support_20260730/a9-a11-base-dual-route",
            "artifacts/neural_forge_sparse_semantic_highcap_20260730",
            "artifacts/neural_forge_sparse_semantic_bridge_20260730",
            "artifacts/lfm2_bridge_only_control_20260723",
            "artifacts/neural_forge_functional_output_refit_20260730",
        ],
        "byproduct_write_disclosed": (
            "Importing the sealed runner as a module created "
            "neural-forge-highcap-eb0a5460/experiments/__pycache__/"
            "rerun_lfm2_sparse_semantic_capped_prompts.cpython-312.pyc. It was removed "
            "after the run; every source file in that tree was re-hashed and is "
            "unchanged. Only the experiments/ directory mtime remains updated."
        ),
    },
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
    },
    "inventory": inventory,
    "inventory_sha256": canonical_sha256(inventory),
}

receipt = {**body, "receipt_sha256": canonical_sha256(body)}
with open(os.path.join(BASE, RECEIPT), "w", encoding="ascii", newline="\n") as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True, ensure_ascii=True)
    handle.write("\n")
print("receipt_sha256", receipt["receipt_sha256"])
for row in inventory:
    print(" ", row["sha256"], "%9d" % row["size_bytes"], row["relative_path"])
