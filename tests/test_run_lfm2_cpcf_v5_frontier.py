from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments import run_lfm2_cpcf_v5_frontier as supervisor
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_lfm2_cpcf_v5_frontier import (
    Lfm2CpcfV5FrontierLifecycle,
    authorize_lfm2_cpcf_v5_reviewed_rerun,
    plan_lfm2_cpcf_v5_frontier,
)
from tests.test_dynamic_lfm2_cpcf_v5_frontier import (
    _approved,
    _real_geometry_graph,
    _resume,
    _sealed_bridge,
)


EXPECTED_TARGETS = {
    4: (57_405_368, 2),
    10: (22_966_464, 0),
    25: (9_183_872, 0),
    50: (4_592_768, 0),
    100: (2_291_840, 0),
    250: (916_928, 0),
}


def _digest(label: str) -> str:
    return canonical_sha256({"test": label})


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for name in (
        "parent",
        "donor_tokenizer",
        "target_tokenizer",
        "cache",
    ):
        path = tmp_path / name
        path.mkdir()
        paths[name] = path
    (paths["donor_tokenizer"] / "tokenizer.json").write_text(
        '{"fixture":"donor-tokenizer"}\n',
        encoding="ascii",
    )
    corpus = tmp_path / "trajectories.json"
    corpus.write_text('{"fixture":"cached-only"}\n', encoding="ascii")
    envelope = tmp_path / "parent-envelope.json"
    envelope.write_text('{"fixture":"parent-envelope"}\n', encoding="ascii")
    paths["corpus"] = corpus
    paths["envelope"] = envelope
    return paths


def _evidence(
    *,
    cache_sha256: str,
    corpus: Path,
    envelope: Path,
) -> supervisor.FrontierEvidenceBindings:
    return supervisor.FrontierEvidenceBindings(
        parent_model_identity_sha256=_digest("parent-identity"),
        donor_tokenizer_package_sha256=_digest(
            "donor-tokenizer-package"
        ),
        cache_sha256=cache_sha256,
        corpus_sha256=_file_sha(corpus),
        parent_envelope_file_sha256=_file_sha(envelope),
        parent_envelope_sha256=_digest("parent-envelope-seal"),
        parent_envelope_document_sha256=_digest(
            "parent-envelope-document"
        ),
        holdout_manifest_sha256=_digest("holdout-manifest"),
        holdout_manifest_document_sha256=_digest(
            "holdout-manifest-document"
        ),
        holdout_case_count=64,
    )


def _write_fake_quarantine_receipt(
    *,
    lane: supervisor.FrontierSupervisorLane,
    evidence: supervisor.FrontierEvidenceBindings,
    config: supervisor.FrontierSupervisorConfig,
    compiler_receipt: str,
    source_binding: str,
    target_state: str,
) -> tuple[str, dict[str, Any]]:
    import neural_forge.dynamic_candidate_capture as capture
    import neural_forge.hf_lfm2_cpcf_v5_quarantine as backend

    root = lane.paths.quarantine_directory
    root.mkdir()
    candidate = root / capture.CANDIDATE_DIRECTORY_NAME
    candidate.mkdir()
    (candidate / "weights.bin").write_bytes(b"dense-materialized-fixture")
    parameter_inventory_sha = _digest("parameter-inventory")
    parameter_inventory = {
        "method": "unique_materialized_parameter_storage_offset_numel_identity",
        "unique_parameters": (
            lane.catalog_lane.exact_compression.target_unique_parameters
        ),
        "raw_named_parameter_occurrences": 1,
        "raw_named_parameter_numel": (
            lane.catalog_lane.exact_compression.target_unique_parameters
        ),
        "parameter_iterator_unique_numel": (
            lane.catalog_lane.exact_compression.target_unique_parameters
        ),
        "independent_methods_agree": True,
        "alias_group_count": 0,
        "alias_groups": [],
        "quantization_adjustment_used": False,
        "pruning_mask_adjustment_used": False,
        "inventory_sha256": parameter_inventory_sha,
    }
    metadata_body = {
        "schema": backend.LFM2_CPCF_V5_QUARANTINE_ARTIFACT_SCHEMA,
        "backend_schema": backend.LFM2_CPCF_V5_QUARANTINE_BACKEND_SCHEMA,
        "context_binding_sha256": _digest("context-binding"),
        "model_state_sha256": target_state,
        "parameter_inventory": parameter_inventory,
        "parameter_inventory_sha256": parameter_inventory_sha,
        "materialized_ffn_width": 1,
        "serialized_ffn_aliases": {},
        "tokenizer_package_sha256": _digest("tokenizer-package"),
        "tokenizer_package_files": [],
        "tokenizer_semantics_sha256": _digest("tokenizer-semantics"),
        "tokenizer_special_token_ids": {},
        "dense_safetensor_files": [],
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
    }
    supervisor._atomic_new_json(
        candidate / backend.LFM2_CPCF_V5_ARTIFACT_METADATA,
        {
            **metadata_body,
            "artifact_sha256": canonical_sha256(metadata_body),
        },
    )
    inventory = capture._candidate_inventory(candidate)

    callable_identities = {}
    callable_bindings = {}
    for role in ("saver", "loader", "evaluator"):
        reachable = {"fixture": role}
        identity_body = {
            "schema": (
                "neural-forge-dynamic-candidate-callable-binding-v1"
            ),
            "role": role,
            "implementation_sha256": _digest(f"{role}-implementation"),
            "resource_sha256": _digest(f"{role}-resource"),
            "resource_identity_explicitly_declared": True,
            "reachable_manifest_sha256": canonical_sha256(reachable),
            "reachable_manifest": reachable,
        }
        binding = canonical_sha256(identity_body)
        callable_bindings[role] = binding
        callable_identities[role] = {
            **identity_body,
            "binding_sha256": binding,
        }

    output_sha = _digest("candidate-output")
    evaluation_sha = _digest("candidate-evaluation")
    prior_audit = capture.QuarantineZeroOperationAudit().as_dict()
    observed_audit = {**prior_audit, "protected_parent_count": 1}
    request = {
        "schema": capture.QUARANTINE_REQUEST_SCHEMA,
        "plan_sha256": lane.catalog_lane.target_plan_fingerprint,
        "parent_sha256": evidence.parent_model_identity_sha256,
        "cache_sha256": evidence.cache_sha256,
        "bridge_sha256": config.expected_bridge_artifact_sha256,
        "corpus_sha256": evidence.corpus_sha256,
        "compiler_sha256": compiler_receipt,
        "source_sha256": source_binding,
        "saver_sha256": callable_bindings["saver"],
        "loader_sha256": callable_bindings["loader"],
        "evaluator_sha256": callable_bindings["evaluator"],
        "compiler_binding_semantics": (
            "authenticated-compiler-run-receipt-bound-to-source"
        ),
        "expected_model_state_sha256": target_state,
        "expected_output_sha256": None,
        "pre_reviewed_candidate_fingerprint_sha256": None,
        "prior_audit": prior_audit,
    }
    reload_body = {
        "schema": capture.QUARANTINE_RELOAD_SCHEMA,
        "source": "same-filesystem-staged-immutable-bytes",
        "final_destination_reload_required": True,
        "artifact_inventory_sha256": inventory["inventory_sha256"],
        "loader_sha256": callable_bindings["loader"],
        "loader_path_challenge_performed": True,
        "loader_path_challenge_exception_type": "FixtureChallenge",
        "loader_returned_disjoint_model": True,
        "in_memory_model_state_sha256": target_state,
        "fresh_reload_model_state_sha256": target_state,
        "state_reproduced_after_reload": True,
        "in_memory_output_sha256": output_sha,
        "fresh_reload_output_sha256": output_sha,
        "output_reproduced_after_reload": True,
        "in_memory_evaluation_sha256": evaluation_sha,
        "fresh_reload_evaluation_sha256": evaluation_sha,
        "evaluation_reproduced_after_reload": True,
        "in_memory_execution": {
            "module_forward_calls": 1,
            "root_forward_calls": 1,
            "inference_entrypoint_calls": 1,
        },
        "fresh_reload_execution": {
            "module_forward_calls": 1,
            "root_forward_calls": 1,
            "inference_entrypoint_calls": 1,
        },
        "standalone": True,
        "local_path_only": True,
        "promotable": False,
        "publication_authorized": False,
    }
    reload_receipt = {
        **reload_body,
        "receipt_sha256": canonical_sha256(reload_body),
    }
    path_identity = {
        "device": 0,
        "inode": 0,
        "link_count": 1,
        "mode": 0,
    }
    body = {
        "schema": capture.QUARANTINE_CAPTURE_SCHEMA,
        "status": "quarantined-unreviewed-candidate",
        "lifecycle": "immutable-quarantine",
        "capture_request": request,
        "bindings": {
            "plan_sha256": lane.catalog_lane.target_plan_fingerprint,
            "parent_sha256": evidence.parent_model_identity_sha256,
            "cache_sha256": evidence.cache_sha256,
            "bridge_sha256": config.expected_bridge_artifact_sha256,
            "corpus_sha256": evidence.corpus_sha256,
            "compiler_sha256": compiler_receipt,
            "compiler_binding_semantics": (
                "authenticated-compiler-run-receipt-bound-to-source"
            ),
            "source_sha256": source_binding,
            "model_state_sha256": target_state,
            "output_sha256": output_sha,
        },
        "callable_identities": callable_identities,
        "candidate_artifact": {
            "path": capture.CANDIDATE_DIRECTORY_NAME,
            "inventory": inventory,
            "inventory_sha256": inventory["inventory_sha256"],
            "standalone": True,
            "strict_sidecar_scan_passed": True,
            "foreign_entries_present": False,
        },
        "model_state": {
            "in_memory_sha256": target_state,
            "fresh_reload_sha256": target_state,
            "byte_identical_after_reload": True,
            "in_memory_inventory": {},
            "fresh_reload_inventory": {},
            "dense_materialized": True,
            "quantized": False,
            "pruned": False,
        },
        "evaluations": {
            "in_memory_output_sha256": output_sha,
            "fresh_reload_output_sha256": output_sha,
            "in_memory_evaluation_sha256": evaluation_sha,
            "fresh_reload_evaluation_sha256": evaluation_sha,
            "output_reproduced_after_reload": True,
            "evaluation_reproduced_after_reload": True,
            "corpus_sha256": evidence.corpus_sha256,
            "evaluator_sha256": callable_bindings["evaluator"],
        },
        "standalone_reload_receipt": reload_receipt,
        "audit": {
            "prior_fold": prior_audit,
            "capture_observed": observed_audit,
            "combined_clean": True,
            "historical_fold_scope_is_caller_rooted": True,
            "capture_scope_is_runtime_observed": True,
        },
        "filesystem_identity": {
            "destination_path_sha256": _digest("destination-path"),
            "destination_parent": path_identity,
            "quarantine_root": path_identity,
            "candidate_directory": path_identity,
            "receipt_file": path_identity,
            "candidate_tree_sha256": _digest("candidate-tree"),
        },
        "complete_artifact_inventory": True,
        "standalone_candidate": True,
        "published_compilation": False,
        "accepted_compilation": False,
        "behavior_capability_proven": False,
        "promotable": False,
        "publication_authorized": False,
        "requires_reviewed_rerun": True,
        "pre_reviewed_candidate_fingerprint_supplied": False,
        "no_parent_sidecar": True,
        "no_cache_sidecar": True,
        "no_coordinator_sidecar": True,
        "no_resume_sidecar": True,
        "receipt_outside_candidate_artifact": True,
        "root_entry_allowlist": [
            capture.CANDIDATE_DIRECTORY_NAME,
            capture.QUARANTINE_RECEIPT_FILENAME,
        ],
    }
    receipt = capture.QuarantineCaptureReceipt(body).as_dict()
    supervisor._atomic_new_json(
        root / capture.QUARANTINE_RECEIPT_FILENAME,
        receipt,
    )
    verification_body = {
        "schema": capture.QUARANTINE_VERIFICATION_SCHEMA,
        "capture_receipt_sha256": receipt["receipt_sha256"],
        "destination_path_sha256": _digest("destination-path"),
        "model_state_sha256": target_state,
        "output_sha256": output_sha,
        "evaluation_sha256": evaluation_sha,
        "artifact_inventory_sha256": inventory["inventory_sha256"],
        "loader_sha256": callable_bindings["loader"],
        "evaluator_sha256": callable_bindings["evaluator"],
        "loader_path_challenge_performed": True,
        "loader_path_challenge_exception_type": "FixtureChallenge",
        "fresh_final_destination_reload_performed": True,
        "fresh_reload_execution": {
            "module_forward_calls": 1,
            "root_forward_calls": 1,
            "inference_entrypoint_calls": 1,
        },
        "capture_observed_audit": observed_audit,
        "promotable": False,
        "publication_authorized": False,
        "requires_reviewed_rerun": True,
    }
    verification = {
        **verification_body,
        "verification_sha256": canonical_sha256(verification_body),
    }
    return receipt["receipt_sha256"], verification


@pytest.fixture(scope="module")
def frontier_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("v5-frontier-supervisor")
    bridge = _sealed_bridge(root / "bridge")
    graph = _real_geometry_graph(
        tokenizer_fingerprint=bridge.donor_tokenizer_semantics_sha256
    )
    schedule = plan_lfm2_cpcf_v5_frontier(
        parent_graph=graph,
        expected_parent_graph_fingerprint=graph.fingerprint,
        tokenizer_bridge=bridge,
        expected_bridge_artifact_sha256=bridge.artifact_sha256,
        maximum_candidates_per_ratio=8,
        maximum_parallel_lanes=1,
    )
    return graph, schedule, bridge.root_dir


def _config(
    tmp_path: Path,
    *,
    graph: Any,
    schedule: Any,
    inputs: dict[str, Path],
    lifecycle: Lfm2CpcfV5FrontierLifecycle = (
        Lfm2CpcfV5FrontierLifecycle.CAPTURE_QUARANTINE
    ),
    reviews: dict[str, Any] | None = None,
    resumes: dict[str, Any] | None = None,
    ratios: tuple[int, ...] = supervisor.DECLARATIVE_RATIOS,
    bridge_directory: Path | None = None,
    attempt_id: str = "attempt-0001",
) -> supervisor.FrontierSupervisorConfig:
    return supervisor.FrontierSupervisorConfig(
        parent_model_directory=inputs["parent"],
        donor_tokenizer_directory=inputs["donor_tokenizer"],
        target_tokenizer_directory=inputs["target_tokenizer"],
        bridge_directory=(
            tmp_path / "bridge"
            if bridge_directory is None
            else bridge_directory
        ),
        trace_cache_directory=inputs["cache"],
        trajectory_corpus=inputs["corpus"],
        parent_envelope=inputs["envelope"],
        work_directory=tmp_path / "runs",
        expected_parent_graph_fingerprint=graph.fingerprint,
        expected_bridge_artifact_sha256=schedule.bridge_artifact_sha256,
        expected_cache_sha256=_digest("cache"),
        expected_parent_envelope_file_sha256=_file_sha(
            inputs["envelope"]
        ),
        lifecycle=lifecycle,
        compression_ratios=ratios,
        approved_review_bindings={} if reviews is None else reviews,
        resume_references={} if resumes is None else resumes,
        runtime_limits=supervisor.FrontierRuntimeLimits(
            maximum_process_rss_gib=64,
            maximum_cuda_gib_per_device=20,
            cpu_threads=16,
            gpu_selectors=("GPU-test-3090-a", "GPU-test-3090-b"),
        ),
        attempt_id=attempt_id,
    )


def test_plan_builds_all_exact_real_targets_as_isolated_blocked_processes(
    tmp_path: Path,
    frontier_fixture,
) -> None:
    graph, schedule, bridge_directory = frontier_fixture
    inputs = _inputs(tmp_path)
    config = _config(
        tmp_path,
        graph=graph,
        schedule=schedule,
        inputs=inputs,
        bridge_directory=bridge_directory,
    )
    evidence = _evidence(
        cache_sha256=config.expected_cache_sha256,
        corpus=inputs["corpus"],
        envelope=inputs["envelope"],
    )

    plan = supervisor.build_frontier_supervisor_plan(
        schedule,
        config,
        evidence,
    )
    report = plan.as_dict()

    assert tuple(lane.ratio for lane in plan.lanes) == (
        4,
        10,
        25,
        50,
        100,
        250,
    )
    assert report["maximum_parallel_ratio_processes"] == 1
    assert report["input_bindings"][
        "donor_tokenizer_package_sha256"
    ] == evidence.donor_tokenizer_package_sha256
    assert report["catalog_schedule"][
        "donor_tokenizer_semantics_sha256"
    ] == schedule.donor_tokenizer_semantics_sha256
    assert (
        evidence.donor_tokenizer_package_sha256
        != schedule.donor_tokenizer_semantics_sha256
    )
    assert report["parent_trace_generation_calls"] == 0
    assert report["training_steps"] == 0
    assert report["parameters_pruned"] == 0
    assert report["parameters_quantized"] == 0
    assert report["structural_diagnostics_can_authorize_promotion"] is False
    assert report["publication_authorized"] is False
    assert report["status"] == "capture_quarantine_processes_ready"

    all_paths: set[str] = set()
    for lane in plan.lanes:
        expected_target, expected_rank = EXPECTED_TARGETS[lane.ratio]
        assert (
            lane.catalog_lane.exact_compression.target_unique_parameters
            == expected_target
        )
        assert lane.catalog_lane.selected_candidate_rank == expected_rank
        assert lane.execution_ready is True
        assert lane.execution_blockers == ()

        manifest = dict(lane.command_manifest)
        worker_command = manifest["command"]
        command = manifest["diagnostic_runner_command"]
        assert "--internal-worker-manifest" in worker_command
        assert "--compression-ratio" in command
        assert "--target-unique-parameters" not in command
        ratio_index = command.index("--compression-ratio")
        assert command[ratio_index + 1] == str(lane.ratio)
        assert "--selected-candidate-rank" not in command
        plan_index = command.index("--expected-target-plan-fingerprint")
        assert command[plan_index + 1] == (
            lane.catalog_lane.target_plan_fingerprint
        )
        assert "--resume-store-directory" in command
        cuda_cap_index = command.index(
            "--maximum-cuda-gib-per-device"
        )
        assert command[cuda_cap_index + 1] == "20.0"
        assert "--train" not in command
        assert "--prune" not in command
        assert "--quantize" not in command
        assert manifest["environment_overrides"]["CUDA_VISIBLE_DEVICES"] == (
            "GPU-test-3090-a,GPU-test-3090-b"
        )
        assert manifest["runtime_limits"]["logical_gpu_indices_inside_fold"] == [
            0,
            1,
        ]

        capture = manifest["quarantine_capture_template"]
        assert capture["quarantine_capture_request_constructed"] is False
        assert capture["unresolved_live_callable_bindings"] == {
            "compiler_sha256": None,
            "source_sha256": None,
            "saver_sha256": None,
            "loader_sha256": None,
            "evaluator_sha256": None,
        }
        assert "source_sha256" not in (
            capture["resolved_plan_and_input_bindings"]
        )
        assert len(capture["prelaunch_source_bundle_sha256"]) == 64
        assert capture["required_reuse"][
            "per_case_manifest_generation_limits_required"
        ] is True
        assert capture["required_reuse"][
            "arbitrary_global_64_token_cap_forbidden"
        ] is True

        paths = lane.paths.as_dict()
        path_values = {
            value for value in paths.values() if isinstance(value, str)
        }
        assert not (path_values & all_paths)
        all_paths.update(path_values)


def test_single_ratio_launches_exactly_one_fixed_worker_and_authenticates(
    tmp_path: Path,
    frontier_fixture,
) -> None:
    graph, schedule, bridge_directory = frontier_fixture
    inputs = _inputs(tmp_path)
    config = _config(
        tmp_path,
        graph=graph,
        schedule=schedule,
        inputs=inputs,
        bridge_directory=bridge_directory,
    )
    evidence = _evidence(
        cache_sha256=config.expected_cache_sha256,
        corpus=inputs["corpus"],
        envelope=inputs["envelope"],
    )
    plan = supervisor.build_frontier_supervisor_plan(
        schedule,
        config,
        evidence,
    )
    calls = []

    def fake_worker(*args, **kwargs):
        calls.append((args, kwargs))
        lane = plan.lane_for_ratio(100)
        compiler_receipt = _digest("compiler-receipt")
        selective_source = _digest("selective-source")
        compiler_implementation = _digest("compiler-implementation")
        runner_source = _file_sha(supervisor._DIAGNOSTIC_RUNNER)
        target_state = _digest("target-state")
        guard_identity = _digest("fold-only-runtime-guard")
        source_binding = supervisor._sha(
            {
                "selective_source_sha256": selective_source,
                "compiler_implementation_sha256": compiler_implementation,
                "runner_source_sha256": runner_source,
                "fold_only_runtime_guard_identity_sha256": guard_identity,
            }
        )
        capture_receipt, final_reload_verification = (
            _write_fake_quarantine_receipt(
            lane=lane,
            evidence=evidence,
            config=config,
            compiler_receipt=compiler_receipt,
            source_binding=source_binding,
            target_state=target_state,
            )
        )
        compiled = {
            "result_sha256": compiler_receipt,
            "selective_source_sha256": selective_source,
            "compiler_implementation": {
                "implementation_sha256": compiler_implementation,
            },
        }
        guard_body = {
            "schema": (
                "neural-forge-lfm2-cpcf-v5-compiler-runtime-guard-v1"
            ),
            "guard_implementation": {
                "identity_sha256": guard_identity,
                "source_sha256": plan.source_identities[
                    "fold_only_runtime_guard"
                ],
                "training_api_firewall": True,
                "common_pruning_api_firewall": True,
                "common_quantization_api_firewall": True,
            },
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "autograd_grad_calls": 0,
            "training_steps": 0,
            "pruning_operations": 0,
            "quantization_operations": 0,
            "clean": True,
        }
        runtime_guard = {
            **guard_body,
            "report_sha256": supervisor._sha(guard_body),
        }
        consumer_guard_body = {
            "schema": (
                "neural-forge-lfm2-cpcf-v5-consumer-runtime-fold-only-guard-v1"
            ),
            "guard_implementation": dict(
                guard_body["guard_implementation"]
            ),
            "compiler_runtime_guard_sha256": runtime_guard[
                "report_sha256"
            ],
            "capture_verification_sha256": (
                final_reload_verification["verification_sha256"]
            ),
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "autograd_grad_calls": 0,
            "training_steps": 0,
            "pruning_operations": 0,
            "quantization_operations": 0,
            "clean": True,
        }
        consumer_guard = {
            **consumer_guard_body,
            "report_sha256": supervisor._sha(consumer_guard_body),
        }
        diagnostic_body = {
            "schema": (
                "neural-forge-lfm2-cpcf-v5-selective-blank-fold-run-v1"
            ),
            "inputs": {
                "parent_fingerprint": (
                    evidence.parent_model_identity_sha256
                ),
                "cache_key": evidence.cache_sha256,
                "bridge_artifact_sha256": (
                    config.expected_bridge_artifact_sha256
                ),
                "trajectory_corpus_sha256": evidence.corpus_sha256,
            },
            "compiled_model_fold": compiled,
            "compiler_runtime_fold_only_guard": runtime_guard,
            "resume_progress": {
                "resume_requested": False,
                "requested_admission_sha256": None,
                "invocation_head": None,
            },
        }
        supervisor._atomic_new_json(
            lane.paths.diagnostic_json,
            {
                **diagnostic_body,
                "receipt_sha256": supervisor._sha(diagnostic_body),
            },
        )
        receipt_body = {
            "schema": supervisor.EXECUTION_SCHEMA,
            "status": "captured_quarantine_requires_reviewed_rerun",
            "ratio": 100,
            "target_plan_fingerprint": (
                lane.catalog_lane.target_plan_fingerprint
            ),
            "target_unique_parameters": (
                lane.catalog_lane.exact_compression.target_unique_parameters
            ),
            "selected_candidate_fingerprint": (
                lane.catalog_lane.selected_candidate.fingerprint
            ),
            "selected_candidate_rank": (
                lane.catalog_lane.selected_candidate_rank
            ),
            "compiler_receipt_sha256": compiler_receipt,
            "capture_receipt_sha256": capture_receipt,
            "target_state_sha256": target_state,
            "parent_sha256": evidence.parent_model_identity_sha256,
            "cache_sha256": evidence.cache_sha256,
            "bridge_sha256": config.expected_bridge_artifact_sha256,
            "corpus_sha256": evidence.corpus_sha256,
            "selective_source_sha256": selective_source,
            "compiler_implementation_sha256": compiler_implementation,
            "runner_source_sha256": runner_source,
            "source_binding_sha256": source_binding,
            "post_capture_source_identities": dict(
                plan.source_identities
            ),
            "final_reload_verification": final_reload_verification,
            "consumer_runtime_guard_receipt": consumer_guard,
            "diagnostic_json": str(lane.paths.diagnostic_json),
            "quarantine_directory": str(
                lane.paths.quarantine_directory
            ),
            "standalone_candidate": True,
            "standalone_reload_verified": True,
            "whole_output_evaluation_captured": True,
            "manual_review_status": "pending",
            "promotable": False,
            "publication_authorized": False,
            "requires_reviewed_rerun": True,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
        }
        supervisor._atomic_new_json(
            lane.paths.execution_receipt_json,
            {
                **receipt_body,
                "receipt_sha256": supervisor._sha(receipt_body),
            },
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    receipt = supervisor.execute_single_ratio(
        plan,
        100,
        subprocess_runner=fake_worker,
    )

    assert len(calls) == 1
    assert "--internal-worker-manifest" in calls[0][0][0]
    assert receipt["promotable"] is False
    assert plan.lane_for_ratio(100).paths.process_manifest_json.is_file()


def test_reviewed_rerun_binds_state_output_and_manual_review_but_not_promotion(
    tmp_path: Path,
    frontier_fixture,
) -> None:
    graph, capture_schedule, bridge_directory = frontier_fixture
    inputs = _inputs(tmp_path)
    reviews = {
        lane.target_plan_fingerprint: _approved(
            lane.target_plan_fingerprint
        )
        for lane in capture_schedule.lanes
    }
    reviewed = authorize_lfm2_cpcf_v5_reviewed_rerun(
        capture_schedule,
        reviews,
    )
    config = _config(
        tmp_path,
        graph=graph,
        schedule=reviewed,
        inputs=inputs,
        lifecycle=Lfm2CpcfV5FrontierLifecycle.REVIEWED_RERUN,
        reviews=reviews,
        bridge_directory=bridge_directory,
    )
    evidence = _evidence(
        cache_sha256=config.expected_cache_sha256,
        corpus=inputs["corpus"],
        envelope=inputs["envelope"],
    )

    plan = supervisor.build_frontier_supervisor_plan(
        reviewed,
        config,
        evidence,
    )
    for lane in plan.lanes:
        approved = reviews[lane.catalog_lane.target_plan_fingerprint]
        template = lane.command_manifest[
            "quarantine_capture_template"
        ]
        bindings = template["resolved_plan_and_input_bindings"]
        assert (
            bindings["expected_model_state_sha256"]
            == approved.approved_candidate_state_sha256
        )
        assert (
            bindings["expected_output_sha256"]
            == approved.approved_candidate_outputs_sha256
        )
        report = lane.as_dict()
        assert report["lifecycle_contract"][
            "approved_review_bindings"
        ]["approved_manual_review_sha256"] == (
            approved.approved_manual_review_sha256
        )
        assert report["promotable"] is False
        assert report["publication_authorized"] is False
        assert lane.execution_ready is False
        assert any(
            "reviewed_rerun" in blocker
            for blocker in report["execution_blockers"]
        )


def test_lane_outputs_are_create_new_and_collision_fails_closed(
    tmp_path: Path,
    frontier_fixture,
) -> None:
    graph, schedule, bridge_directory = frontier_fixture
    inputs = _inputs(tmp_path)
    config = _config(
        tmp_path,
        graph=graph,
        schedule=schedule,
        inputs=inputs,
        bridge_directory=bridge_directory,
    )
    evidence = _evidence(
        cache_sha256=config.expected_cache_sha256,
        corpus=inputs["corpus"],
        envelope=inputs["envelope"],
    )
    first = supervisor.build_frontier_supervisor_plan(
        schedule,
        config,
        evidence,
    )
    first.lanes[0].paths.run_directory.mkdir(parents=True)

    with pytest.raises(
        supervisor.Lfm2CpcfV5FrontierSupervisorError,
        match="already exists",
    ):
        supervisor.build_frontier_supervisor_plan(
            schedule,
            config,
            evidence,
        )

    retry_config = _config(
        tmp_path,
        graph=graph,
        schedule=schedule,
        inputs=inputs,
        bridge_directory=bridge_directory,
        attempt_id="attempt-0002",
    )
    retry = supervisor.build_frontier_supervisor_plan(
        schedule,
        retry_config,
        evidence,
    )
    assert retry.lanes[0].paths.run_directory != (
        first.lanes[0].paths.run_directory
    )
    assert retry.lanes[0].paths.resume_directory == (
        first.lanes[0].paths.resume_directory
    )


def test_authenticated_resume_reopens_plan_store_under_new_attempt(
    tmp_path: Path,
    frontier_fixture,
) -> None:
    graph, schedule, bridge_directory = frontier_fixture
    inputs = _inputs(tmp_path)
    resumes = {
        lane.target_plan_fingerprint: _resume(
            lane.target_plan_fingerprint
        )
        for lane in schedule.lanes
    }
    resumed_schedule = replace(
        schedule,
        lanes=tuple(
            replace(
                lane,
                resume_reference=resumes[lane.target_plan_fingerprint],
            )
            for lane in schedule.lanes
        ),
    )
    evidence = _evidence(
        cache_sha256=_digest("cache"),
        corpus=inputs["corpus"],
        envelope=inputs["envelope"],
    )
    first_config = _config(
        tmp_path,
        graph=graph,
        schedule=resumed_schedule,
        inputs=inputs,
        resumes=resumes,
        bridge_directory=bridge_directory,
        attempt_id="attempt-0001",
    )
    first = supervisor.build_frontier_supervisor_plan(
        resumed_schedule,
        first_config,
        evidence,
    )
    for lane in first.lanes:
        lane.paths.resume_directory.mkdir(parents=True)

    second_config = _config(
        tmp_path,
        graph=graph,
        schedule=resumed_schedule,
        inputs=inputs,
        resumes=resumes,
        bridge_directory=bridge_directory,
        attempt_id="attempt-0002",
    )
    second = supervisor.build_frontier_supervisor_plan(
        resumed_schedule,
        second_config,
        evidence,
    )

    for old, new in zip(first.lanes, second.lanes, strict=True):
        assert new.paths.resume_reopen_allowed is True
        assert new.paths.resume_directory == old.paths.resume_directory
        assert new.paths.run_directory != old.paths.run_directory


def test_early_failure_receipt_does_not_claim_unauthenticated_zero_ops() -> None:
    receipt = supervisor._worker_failure_receipt(
        {
            "ratio": 100,
            "target_plan_fingerprint": _digest("failed-plan"),
        },
        RuntimeError("fixture failure"),
    )

    assert receipt["zero_operation_audit_authenticated"] is False
    assert receipt["operation_counts_claimed"] is False
    assert receipt["training_steps"] is None
    assert receipt["parameters_pruned"] is None
    assert receipt["parameters_quantized"] is None


def test_atomic_plan_output_never_overwrites(tmp_path: Path) -> None:
    output = tmp_path / "plan.json"
    supervisor._atomic_new_json(output, {"schema": "test", "value": 1})
    assert output.read_bytes() == b'{"schema":"test","value":1}\n'

    with pytest.raises(
        supervisor.Lfm2CpcfV5FrontierSupervisorError,
        match="already exists",
    ):
        supervisor._atomic_new_json(output, {"schema": "test", "value": 2})
    assert output.read_bytes() == b'{"schema":"test","value":1}\n'


def test_parent_envelope_may_use_a_distinct_cache_key_when_parent_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import neural_forge.hf_lfm2_intelligence_gate as gate_module
    import neural_forge.hf_lfm2_intelligence_holdout as holdout_module
    from experiments.evaluate_lfm2_intelligence_retention import (
        fingerprint_bound_local_tokenizer,
    )

    inputs = _inputs(tmp_path)
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    parent_identity = _digest("shared-parent-model-identity")
    tokenizer_identity = fingerprint_bound_local_tokenizer(
        inputs["donor_tokenizer"]
    )[0]
    semantic_identity = _digest("distinct-live-tokenizer-semantics")
    assert tokenizer_identity != semantic_identity
    envelope_document = {
        "envelope_sha256": _digest("envelope-seal"),
        "parent_state_fingerprint": _digest("parent-state"),
        "parent_model_fingerprint": parent_identity,
        "tokenizer_fingerprint": tokenizer_identity,
    }
    inputs["envelope"].write_text(
        json.dumps(envelope_document),
        encoding="ascii",
    )
    config = supervisor.FrontierSupervisorConfig(
        parent_model_directory=inputs["parent"],
        donor_tokenizer_directory=inputs["donor_tokenizer"],
        target_tokenizer_directory=inputs["target_tokenizer"],
        bridge_directory=bridge,
        trace_cache_directory=inputs["cache"],
        trajectory_corpus=inputs["corpus"],
        parent_envelope=inputs["envelope"],
        work_directory=tmp_path / "runs",
        expected_parent_graph_fingerprint=_digest("graph"),
        expected_bridge_artifact_sha256=_digest("bridge"),
        expected_cache_sha256=_digest("folding-trace-cache"),
        expected_parent_envelope_file_sha256=_file_sha(
            inputs["envelope"]
        ),
        lifecycle=Lfm2CpcfV5FrontierLifecycle.CAPTURE_QUARANTINE,
    )
    manifest = SimpleNamespace(
        manifest_sha256=_digest("manifest"),
        document_sha256=_digest("manifest-document"),
        cases=tuple(range(64)),
    )
    envelope = SimpleNamespace(
        parent_cache_key=_digest("different-holdout-cache"),
        parent_model_fingerprint=parent_identity,
        tokenizer_fingerprint=tokenizer_identity,
        envelope_sha256=envelope_document["envelope_sha256"],
        document_sha256=_digest("envelope-document"),
    )
    monkeypatch.setattr(
        gate_module,
        "load_lfm2_intelligence_holdout_manifest",
        lambda: manifest,
    )
    monkeypatch.setattr(
        holdout_module,
        "load_lfm2_intelligence_holdout_parent_envelope",
        lambda *args, **kwargs: envelope,
    )

    loaded_manifest, loaded_envelope = supervisor._read_parent_envelope(
        config,
        parent_model_identity_sha256=parent_identity,
    )

    assert loaded_manifest is manifest
    assert loaded_envelope is envelope
    assert envelope.parent_cache_key != config.expected_cache_sha256
    with pytest.raises(
        supervisor.Lfm2CpcfV5FrontierSupervisorError,
        match="same parent model",
    ):
        supervisor._read_parent_envelope(
            config,
            parent_model_identity_sha256=_digest("wrong-parent"),
        )

    (inputs["donor_tokenizer"] / "tokenizer.json").write_text(
        '{"fixture":"changed-donor-tokenizer"}\n',
        encoding="ascii",
    )
    with pytest.raises(
        supervisor.Lfm2CpcfV5FrontierSupervisorError,
        match="package identity",
    ):
        supervisor._read_parent_envelope(
            config,
            parent_model_identity_sha256=parent_identity,
        )
