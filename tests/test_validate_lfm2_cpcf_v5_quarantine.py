from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
import torch

from experiments import validate_lfm2_cpcf_v5_quarantine as validator
from neural_forge.dynamic_candidate_capture import QuarantineEvaluation
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_cpcf_model_fold import (
    independent_parameter_inventory,
)
from neural_forge.hf_lfm2_intelligence_gate import (
    IntelligenceGatePolicy,
    build_manual_review_packet,
    load_lfm2_intelligence_holdout_manifest,
)
from neural_forge.hf_lfm2_intelligence_holdout import (
    baseline_from_lfm2_intelligence_holdout_envelope,
    candidate_outputs_from_lfm2_holdout_rows,
    load_lfm2_intelligence_holdout_parent_envelope,
    seal_lfm2_intelligence_holdout_parent_envelope,
)
from neural_forge.dynamic_lfm2_cpcf_v5_frontier import (
    plan_lfm2_cpcf_v5_frontier,
)
from tests.test_dynamic_lfm2_cpcf_v5_frontier import (
    _real_geometry_graph,
    _sealed_bridge,
)
from tests.test_run_lfm2_cpcf_v5_frontier import (
    _catalog_preparation,
    _config as _supervisor_config,
    _evidence as _supervisor_evidence,
    _inputs as _supervisor_inputs,
    _write_fake_quarantine_receipt,
)


def _digest(label: str) -> str:
    return canonical_sha256({"test": label})


def _validator_sha() -> str:
    return hashlib.sha256(Path(validator.__file__).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def parent_envelope():
    manifest = load_lfm2_intelligence_holdout_manifest()
    rows = [
        {
            "case_id": spec.case_id,
            "input_token_count": 1,
            "output": f"parent response for {spec.case_id}",
            "output_token_ids": [2, 1],
        }
        for spec in manifest.cases
    ]
    sealed = seal_lfm2_intelligence_holdout_parent_envelope(
        manifest,
        rows,
        parent_state_fingerprint=_digest("parent-state"),
        parent_model_fingerprint=_digest("parent-model"),
        tokenizer_fingerprint=_digest("parent-tokenizer"),
        eos_token_ids=(1,),
    )
    return load_lfm2_intelligence_holdout_parent_envelope(
        sealed,
        expected_envelope_sha256=sealed["envelope_sha256"],
        expected_manifest_sha256=manifest.manifest_sha256,
        expected_parent_state_fingerprint=_digest("parent-state"),
        expected_parent_model_fingerprint=_digest("parent-model"),
        expected_tokenizer_fingerprint=_digest("parent-tokenizer"),
    )


def _candidate_rows(envelope: Any) -> list[dict[str, Any]]:
    return [
        {
            "case_id": spec.case_id,
            "input_token_count": 1,
            "output": f"child response for {spec.case_id}",
            "output_token_ids": [2, 1],
            "generation_limit": spec.generation_limit,
            "topical_relevance": None,
        }
        for spec in envelope.manifest.cases
    ]


def _evidence(envelope: Any) -> QuarantineEvaluation:
    rows = _candidate_rows(envelope)
    return QuarantineEvaluation(
        output={
            "schema": validator.WHOLE_RESPONSE_SCHEMA,
            "case_count": len(rows),
            "rows": rows,
            "full_decoded_outputs_retained": True,
            "parent_model_loaded": False,
            "parent_forward_calls": 0,
        },
        evaluation={
            "schema": "fixture-evaluation",
            "manual_review_status": "pending",
            "parent_forward_calls": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
        },
    )


def _guard() -> dict[str, Any]:
    return {
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "autograd_grad_calls": 0,
        "pruning_operations": 0,
        "quantization_operations": 0,
        "clean": True,
    }


def _fake_run(
    tmp_path: Path,
    envelope: Any,
    evidence: QuarantineEvaluation,
) -> validator.AuthenticatedRun:
    run_directory = tmp_path / "capture_quarantine" / "004x-fixture" / "attempt-0002"
    quarantine = run_directory / "quarantine-fixture"
    candidate = quarantine / "candidate"
    candidate.mkdir(parents=True)
    parent_path = tmp_path / "parent-envelope.json"
    parent_path.write_text('{"fixture":true}\n', encoding="ascii")
    donor = tmp_path / "donor-tokenizer"
    donor.mkdir()
    target_state = _digest("target-state")
    plan_sha = _digest("plan")
    lane_sha = _digest("lane")
    command_sha = _digest("command")
    execution_sha = _digest("execution")
    diagnostic_sha = _digest("diagnostic")
    capture_sha = _digest("capture")
    final_verification = {
        "verification_sha256": _digest("original-verification"),
    }
    catalog_preparation = _catalog_preparation()
    return validator.AuthenticatedRun(
        plan=MappingProxyType({"plan_sha256": plan_sha}),
        lane=MappingProxyType({"lane_sha256": lane_sha}),
        manifest=MappingProxyType(
            {
                "command_sha256": command_sha,
                "environment_overrides": {},
                "worker_config": {
                    "parent_envelope": str(parent_path),
                    "donor_tokenizer_directory": str(donor),
                },
                "input_bindings": {
                    "parent_envelope_file_sha256": hashlib.sha256(
                        parent_path.read_bytes()
                    ).hexdigest(),
                },
            }
        ),
        execution=MappingProxyType(
            {
                "receipt_sha256": execution_sha,
                "final_reload_verification": final_verification,
                "target_plan_fingerprint": _digest("target-plan"),
                "target_state_sha256": target_state,
                "target_unique_parameters": 6,
                "selective_catalog_artifact": catalog_preparation,
            }
        ),
        diagnostic=MappingProxyType({"receipt_sha256": diagnostic_sha}),
        quarantine_receipt=MappingProxyType(
            {
                "receipt_sha256": capture_sha,
                "candidate_artifact": {"inventory": {"fixture": True}},
                "bindings": {"output_sha256": evidence.output_sha256},
                "evaluations": {
                    "fresh_reload_output_sha256": evidence.output_sha256,
                    "fresh_reload_evaluation_sha256": (
                        evidence.evaluation_sha256
                    ),
                },
            }
        ),
        artifact_metadata=MappingProxyType({}),
        compiler_guard=MappingProxyType(
            {"report_sha256": _digest("compiler-guard")}
        ),
        consumer_guard=MappingProxyType(
            {"report_sha256": _digest("consumer-guard")}
        ),
        run_directory=run_directory,
        quarantine_directory=quarantine,
        candidate_directory=candidate,
        output_directory=run_directory.parent
        / "post-success-validation-attempt-0002",
    )


def _fake_runtime(envelope: Any) -> validator.ReplayRuntime:
    return validator.ReplayRuntime(
        request=SimpleNamespace(),
        loader=SimpleNamespace(),
        evaluator=SimpleNamespace(),
        parent_envelope=envelope,
        target_tokenizer=SimpleNamespace(),
        target_tokenizer_semantics_sha256=_digest("child-tokenizer"),
        child_eos_token_ids=(1,),
    )


def _fake_cached_evaluation(
    config: Any,
    *,
    envelope: Any,
) -> dict[str, Any]:
    rows = json.loads(config.candidate_report.read_text(encoding="ascii"))["rows"]
    candidates = candidate_outputs_from_lfm2_holdout_rows(
        envelope.manifest,
        rows,
        eos_token_ids=(1,),
    )
    packet = build_manual_review_packet(
        baseline_from_lfm2_intelligence_holdout_envelope(envelope),
        candidates,
        IntelligenceGatePolicy(),
    )
    report_body = {
        "schema": validator.holdout_runner.RUNNER_SCHEMA,
        "status": "rejected",
        "promoted": False,
        "manual_review": {
            "status": "pending",
            "packet_sha256": packet["packet_sha256"],
        },
        "intelligence_retention_claim": {
            "status": "withheld_pending_manual_review",
        },
    }
    report = {
        **report_body,
        "report_sha256": canonical_sha256(report_body),
    }
    validator.supervisor._atomic_new_json(
        config.manual_review_packet_path,
        packet,
    )
    validator.supervisor._atomic_new_json(config.report_path, report)
    return report


def _completed_supervisor_attempt(
    tmp_path: Path,
    *,
    profile_bound: bool = False,
) -> validator.PostSuccessValidationConfig:
    bridge = _sealed_bridge(tmp_path / "bridge")
    graph = _real_geometry_graph(
        tokenizer_fingerprint=bridge.donor_tokenizer_semantics_sha256
    )
    schedule = plan_lfm2_cpcf_v5_frontier(
        parent_graph=graph,
        expected_parent_graph_fingerprint=graph.fingerprint,
        tokenizer_bridge=bridge,
        expected_bridge_artifact_sha256=bridge.artifact_sha256,
        compression_ratios=(4,),
        maximum_candidates_per_ratio=8,
        maximum_parallel_lanes=1,
    )
    inputs = _supervisor_inputs(tmp_path)
    attention_profile = (
        validator.fold_runner.V5AttentionSequentialTransportRunnerProfile()
        if profile_bound
        else None
    )
    config = _supervisor_config(
        tmp_path,
        graph=graph,
        schedule=schedule,
        inputs=inputs,
        ratios=(4,),
        bridge_directory=bridge.root_dir,
        attempt_id="attempt-0002",
        attention_profile=attention_profile,
    )
    evidence = _supervisor_evidence(
        cache_sha256=config.expected_cache_sha256,
        corpus=inputs["corpus"],
        envelope=inputs["envelope"],
    )
    profile_policy_fields: dict[str, Any] = {}
    if profile_bound:
        catalog_runner_config = (
            validator.supervisor._frontier_catalog_runner_config(
                config,
                ratio=4,
                prepare_only=False,
            )
        )
        profile_policy_fields = dict(
            validator.supervisor._runner_policy_identity_fields(
                validator.fold_runner,
                catalog_runner_config,
            )
        )
        evidence = replace(
            evidence,
            selective_catalog_preparation={
                **dict(evidence.selective_catalog_preparation),
                **profile_policy_fields,
            },
        )
    plan = validator.supervisor.build_frontier_supervisor_plan(
        schedule,
        config,
        evidence,
    )
    lane = plan.lane_for_ratio(4)
    lane.paths.run_directory.mkdir(parents=True)
    validator.supervisor._atomic_new_json(
        lane.paths.process_manifest_json,
        lane.command_manifest,
    )
    compiler_receipt = _digest("compiler-receipt")
    catalog_preparation = dict(evidence.selective_catalog_preparation)
    selective_source = catalog_preparation["selective_source_sha256"]
    compiler_implementation_sha = _digest("compiler-implementation")
    runner_source = hashlib.sha256(
        validator.supervisor._DIAGNOSTIC_RUNNER.read_bytes()
    ).hexdigest()
    target_state = _digest("target-state")
    guard_identity = _digest("guard-identity")
    source_binding = validator.fold_runner._source_binding_sha256(
        selective_source_sha256=selective_source,
        compiler_implementation_sha256=compiler_implementation_sha,
        runner_source_sha256=runner_source,
        fold_only_runtime_guard_identity_sha256=guard_identity,
        catalog_preparation=catalog_preparation,
    )
    capture_receipt, final_reload = _write_fake_quarantine_receipt(
        lane=lane,
        evidence=evidence,
        config=config,
        compiler_receipt=compiler_receipt,
        source_binding=source_binding,
        target_state=target_state,
    )
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
        **_guard(),
        "training_steps": 0,
    }
    compiler_guard = {
        **guard_body,
        "report_sha256": validator.supervisor._sha(guard_body),
    }
    consumer_body = {
        "schema": (
            "neural-forge-lfm2-cpcf-v5-consumer-runtime-fold-only-guard-v1"
        ),
        "guard_implementation": dict(guard_body["guard_implementation"]),
        "compiler_runtime_guard_sha256": compiler_guard["report_sha256"],
        "capture_verification_sha256": final_reload[
            "verification_sha256"
        ],
        **_guard(),
        "training_steps": 0,
    }
    consumer_guard = {
        **consumer_body,
        "report_sha256": validator.supervisor._sha(consumer_body),
    }
    diagnostic_body = {
        "schema": validator.fold_runner.RUN_SCHEMA,
        "inputs": {
            "parent_fingerprint": evidence.parent_model_identity_sha256,
            "cache_key": evidence.cache_sha256,
            "bridge_artifact_sha256": (
                config.expected_bridge_artifact_sha256
            ),
            "trajectory_corpus_sha256": evidence.corpus_sha256,
            "selective_catalog_preparation_receipt_sha256": (
                catalog_preparation["preparation_receipt_sha256"]
            ),
            "selective_catalog_binding_sha256": catalog_preparation[
                "binding_sha256"
            ],
            "selective_catalog_artifact_manifest_sha256": (
                catalog_preparation["artifact_manifest_sha256"]
            ),
            "selective_catalog_artifact_sha256": catalog_preparation[
                "artifact_catalog_sha256"
            ],
            "selective_catalog_codec_sha256": catalog_preparation[
                "codec_sha256"
            ],
            "selective_catalog_provider_implementation_sha256": (
                catalog_preparation["provider_implementation_sha256"]
            ),
        },
        "compiled_model_fold": {
            "result_sha256": compiler_receipt,
            "selective_source_sha256": selective_source,
            "selective_catalog_sha256": catalog_preparation[
                "selective_catalog_sha256"
            ],
            "compiler_implementation": {
                "implementation_sha256": compiler_implementation_sha,
            },
        },
        "compiler_runtime_fold_only_guard": compiler_guard,
        "selective_catalog_artifact": catalog_preparation,
        "selective_catalog_post_run_reverification": dict(
            validator.fold_runner
            .selective_catalog_post_run_reverification_record(
                catalog_preparation
            )
        ),
        "resume_progress": {
            "resume_requested": False,
            "requested_admission_sha256": None,
            "invocation_head": None,
        },
    }
    if profile_policy_fields:
        runtime_policy_document = {
            "schema": "fixture-runtime-policy",
            "model_fold_policy_sha256": profile_policy_fields[
                "complete_model_fold_policy_sha256"
            ],
        }
        diagnostic_body["compiled_model_fold"][
            "policy"
        ] = runtime_policy_document
        resume_binding_body = {
            "schema": validator.fold_runner.RESUME_POLICY_BINDING_SCHEMA,
            **profile_policy_fields,
            "runtime_policy_sha256": canonical_sha256(
                runtime_policy_document
            ),
            "runtime_policy_contains_complete_model_fold_policy": True,
            "cross_profile_resume_permitted": False,
        }
        resume_binding = {
            **resume_binding_body,
            "binding_sha256": canonical_sha256(resume_binding_body),
        }
        diagnostic_body.update(profile_policy_fields)
        diagnostic_body["request"] = dict(profile_policy_fields)
        diagnostic_body["resume_policy_binding"] = resume_binding
    validator.supervisor._atomic_new_json(
        lane.paths.diagnostic_json,
        {
            **diagnostic_body,
            "receipt_sha256": validator.supervisor._sha(diagnostic_body),
        },
    )
    receipt_body = {
        "schema": validator.supervisor.EXECUTION_SCHEMA,
        "status": "captured_quarantine_requires_reviewed_rerun",
        "ratio": 4,
        "target_plan_fingerprint": (
            lane.catalog_lane.target_plan_fingerprint
        ),
        "target_unique_parameters": (
            lane.catalog_lane.exact_compression.target_unique_parameters
        ),
        "selected_candidate_fingerprint": (
            lane.catalog_lane.selected_candidate.fingerprint
        ),
        "selected_candidate_rank": lane.catalog_lane.selected_candidate_rank,
        "compiler_receipt_sha256": compiler_receipt,
        "capture_receipt_sha256": capture_receipt,
        "target_state_sha256": target_state,
        "parent_sha256": evidence.parent_model_identity_sha256,
        "cache_sha256": evidence.cache_sha256,
        "bridge_sha256": config.expected_bridge_artifact_sha256,
        "corpus_sha256": evidence.corpus_sha256,
        "selective_source_sha256": selective_source,
        "compiler_implementation_sha256": compiler_implementation_sha,
        "runner_source_sha256": runner_source,
        "source_binding_sha256": source_binding,
        "selective_catalog_artifact": catalog_preparation,
        "selective_catalog_post_run_reverification": dict(
            validator.fold_runner
            .selective_catalog_post_run_reverification_record(
                catalog_preparation
            )
        ),
        "post_capture_source_identities": dict(plan.source_identities),
        "final_reload_verification": final_reload,
        "consumer_runtime_guard_receipt": consumer_guard,
        "diagnostic_json": str(lane.paths.diagnostic_json),
        "quarantine_directory": str(lane.paths.quarantine_directory),
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
    if profile_policy_fields:
        receipt_body.update(profile_policy_fields)
        receipt_body["resume_policy_binding"] = resume_binding
    validator.supervisor._atomic_new_json(
        lane.paths.execution_receipt_json,
        {
            **receipt_body,
            "receipt_sha256": validator.supervisor._sha(receipt_body),
        },
    )
    plan_path = config.work_directory / "execute-4x-plan.json"
    validator.supervisor._atomic_new_json(plan_path, plan.as_dict())
    return validator.PostSuccessValidationConfig(
        plan_path=plan_path,
        ratio=4,
        attempt_id="attempt-0002",
        expected_validator_source_sha256=_validator_sha(),
    )


def _configure_completed_attempt_authentication(
    config: validator.PostSuccessValidationConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    plan = json.loads(config.plan_path.read_text(encoding="ascii"))
    manifest_path = Path(
        plan["lanes"][0]["paths"]["process_manifest_json"]
    )
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    for name, value in manifest["environment_overrides"].items():
        monkeypatch.setenv(name, value)
    catalog_preparation = plan["input_bindings"][
        "selective_catalog_preparation"
    ]
    if "attention_sequential_transport_profile" in plan:
        def authenticate_profile_catalog(
            runner_config: Any,
            *,
            expected_receipt_sha256: str | None = None,
        ) -> MappingProxyType:
            assert expected_receipt_sha256 is None
            validator.fold_runner._authenticate_preparation_policy_identity(
                catalog_preparation,
                config=runner_config,
            )
            return MappingProxyType(dict(catalog_preparation))

        monkeypatch.setattr(
            validator.fold_runner,
            "_read_selective_catalog_preparation",
            authenticate_profile_catalog,
        )
        monkeypatch.setattr(
            validator.fold_runner,
            "authenticate_prepared_selective_catalog",
            lambda **_kwargs: pytest.fail(
                "Profile-bound validation used the legacy catalog seam."
            ),
        )
    else:
        monkeypatch.setattr(
            validator.fold_runner,
            "authenticate_prepared_selective_catalog",
            lambda **_kwargs: MappingProxyType(
                dict(catalog_preparation)
            ),
        )
    return plan


def _rewrite_sealed_json(
    path: Path,
    *,
    seal_field: str,
    mutate: Any,
) -> None:
    value = json.loads(path.read_text(encoding="ascii"))
    value.pop(seal_field)
    mutate(value)
    value[seal_field] = canonical_sha256(value)
    path.write_bytes(validator.supervisor._canonical_json(value) + b"\n")


def _reseal_plan_document(value: dict[str, Any]) -> dict[str, Any]:
    for lane in value["lanes"]:
        command = lane["command_manifest"]
        command.pop("command_sha256", None)
        command["command_sha256"] = canonical_sha256(command)
        lane.pop("lane_sha256", None)
        lane["lane_sha256"] = canonical_sha256(lane)
    value.pop("plan_sha256", None)
    value["plan_sha256"] = canonical_sha256(value)
    return value


def test_output_directory_must_be_create_new_direct_sibling(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text("{}\n", encoding="ascii")
    run = tmp_path / "life" / "lane" / "attempt-0002"
    run.mkdir(parents=True)
    outside = tmp_path / "outside"
    config = validator.PostSuccessValidationConfig(
        plan_path=plan,
        ratio=4,
        attempt_id="attempt-0002",
        expected_validator_source_sha256=_validator_sha(),
        output_directory=outside,
    )

    with pytest.raises(
        validator.Lfm2CpcfV5PostSuccessValidationError,
        match="direct sibling",
    ):
        validator._output_directory(config, run)

    expected = run.parent / "post-success-validation-attempt-0002"
    accepted = validator.PostSuccessValidationConfig(
        plan_path=plan,
        ratio=4,
        attempt_id="attempt-0002",
        expected_validator_source_sha256=_validator_sha(),
        output_directory=expected,
    )
    assert validator._output_directory(accepted, run) == expected


@pytest.mark.parametrize(
    "reserved_name",
    ("attempt-0003", "capture_quarantine", "custom-validation"),
)
def test_output_directory_rejects_reserved_or_unscoped_sibling_names(
    tmp_path: Path,
    reserved_name: str,
) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text("{}\n", encoding="ascii")
    run = tmp_path / "life" / "lane" / "attempt-0002"
    run.mkdir(parents=True)
    config = validator.PostSuccessValidationConfig(
        plan_path=plan,
        ratio=4,
        attempt_id="attempt-0002",
        expected_validator_source_sha256=_validator_sha(),
        output_directory=run.parent / reserved_name,
    )

    with pytest.raises(
        validator.Lfm2CpcfV5PostSuccessValidationError,
        match="reserved post-success-validation",
    ):
        validator._output_directory(config, run)


@pytest.mark.parametrize(
    "protected_kind",
    ("plan", "work", "quarantine", "input", "catalog", "other-lane"),
)
def test_validation_output_rejects_every_protected_path_class(
    tmp_path: Path,
    protected_kind: str,
) -> None:
    work = tmp_path / "work"
    ratio_root = work / "capture_quarantine" / "004x-fixture"
    run = ratio_root / "attempt-0002"
    quarantine = run / "quarantine-fixture"
    output = ratio_root / "post-success-validation-attempt-0002"
    input_root = tmp_path / "input"
    catalog_artifact = tmp_path / "catalog-store" / _digest("binding")
    plan_path = tmp_path / "plans" / "execute-4x-plan.json"
    lane = {
        "paths": {
            "run_directory": str(run),
            "diagnostic_json": str(run / "diagnostic.json"),
            "resume_directory": str(run / "resume"),
            "quarantine_directory": str(quarantine),
            "process_manifest_json": str(run / "process-manifest.json"),
            "execution_receipt_json": str(run / "execution-receipt.json"),
            "resume_reopen_allowed": False,
        },
        "command_manifest": {
            "worker_config": {"work_directory": str(work)}
        },
    }
    plan: dict[str, Any] = {
        "local_inputs": {"parent_model_directory": str(input_root)},
        "lanes": [lane],
    }
    catalog: dict[str, Any] = {"artifact_path": str(catalog_artifact)}

    if protected_kind == "plan":
        plan_path = output
    elif protected_kind == "work":
        lane["command_manifest"]["worker_config"]["work_directory"] = str(
            output
        )
    elif protected_kind == "quarantine":
        quarantine = output
    elif protected_kind == "input":
        plan["local_inputs"]["parent_model_directory"] = str(output)
    elif protected_kind == "catalog":
        catalog["artifact_path"] = str(output)
    else:
        plan["lanes"].append(
            {
                "paths": {
                    "run_directory": str(output),
                    "resume_reopen_allowed": False,
                },
                "command_manifest": {
                    "worker_config": {"work_directory": str(work)}
                },
            }
        )

    with pytest.raises(
        validator.Lfm2CpcfV5PostSuccessValidationError,
        match="overlaps protected|strict evidence descendant",
    ):
        validator._assert_output_disjoint_from_protected_paths(
            output,
            plan_path=plan_path,
            plan=plan,
            run_directory=run,
            quarantine_directory=quarantine,
            catalog_preparation=catalog,
        )


def test_default_validation_output_is_disjoint_from_protected_paths(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    ratio_root = work / "capture_quarantine" / "004x-fixture"
    run = ratio_root / "attempt-0002"
    quarantine = run / "quarantine-fixture"
    output = ratio_root / "post-success-validation-attempt-0002"
    plan = {
        "local_inputs": {
            "parent_model_directory": str(tmp_path / "parent"),
            "selective_catalog_store_directory": str(
                tmp_path / "catalog-store"
            ),
            "selective_catalog_receipt_json": str(
                tmp_path / "catalog-receipt.json"
            ),
        },
        "lanes": [
            {
                "paths": {
                    "run_directory": str(run),
                    "diagnostic_json": str(run / "diagnostic.json"),
                    "resume_directory": str(run / "resume"),
                    "quarantine_directory": str(quarantine),
                    "process_manifest_json": str(
                        run / "process-manifest.json"
                    ),
                    "execution_receipt_json": str(
                        run / "execution-receipt.json"
                    ),
                    "resume_reopen_allowed": False,
                },
                "command_manifest": {
                    "worker_config": {"work_directory": str(work)}
                },
            }
        ],
    }

    validator._assert_output_disjoint_from_protected_paths(
        output,
        plan_path=tmp_path / "plans" / "execute-4x-plan.json",
        plan=plan,
        run_directory=run,
        quarantine_directory=quarantine,
        catalog_preparation={
            "artifact_path": str(tmp_path / "catalog-store" / _digest("binding"))
        },
    )


def test_validator_source_exposes_no_hf_auto_or_training_entrypoint() -> None:
    source = Path(validator.__file__).read_text(encoding="utf-8")

    for forbidden in (
        "AutoModel",
        "AutoTokenizer",
        ".from_pretrained(",
        ".backward(",
        "torch.optim",
        "torch.nn.utils.prune",
        "torchao.quant",
    ):
        assert forbidden not in source


def test_authenticates_complete_supervisor_receipt_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _completed_supervisor_attempt(tmp_path)
    _configure_completed_attempt_authentication(config, monkeypatch)

    run = validator._authenticate_run(config)

    assert run.execution["status"] == (
        "captured_quarantine_requires_reviewed_rerun"
    )
    assert run.execution["ratio"] == 4
    assert run.run_directory.name == "attempt-0002"
    assert run.quarantine_receipt["standalone_candidate"] is True
    assert run.compiler_guard["clean"] is True
    assert run.consumer_guard["clean"] is True
    assert not run.output_directory.exists()


def test_authenticates_profile_bound_supervisor_receipt_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _completed_supervisor_attempt(
        tmp_path,
        profile_bound=True,
    )
    plan = _configure_completed_attempt_authentication(
        config,
        monkeypatch,
    )

    run = validator._authenticate_run(config)

    profile = plan["attention_sequential_transport_profile"]
    scope = f"attention-profile-{profile['profile_sha256'][:16]}"
    policy_names = validator.fold_runner._OPT_IN_POLICY_IDENTITY_FIELDS
    assert run.run_directory.parent.name == scope
    assert run.manifest["attention_sequential_transport_profile"] == profile
    assert all(
        run.execution[name] == run.manifest[name]
        for name in policy_names
    )
    assert run.execution["resume_policy_binding"] == run.diagnostic[
        "resume_policy_binding"
    ]
    assert run.output_directory.parent == run.run_directory.parent
    assert not run.output_directory.exists()

    for tamper in (
        "missing-worker-profile",
        "crossed-capture-profile",
        "profile-path-with-legacy-documents",
        "missing-profile-scope",
        "wrong-profile-scope",
        "extra-profile-scope",
    ):
        changed = json.loads(json.dumps(plan))
        lane = changed["lanes"][0]
        command = lane["command_manifest"]
        worker = command["worker_config"]
        capture = command["quarantine_capture_template"]
        work = Path(worker["work_directory"])
        if tamper == "missing-worker-profile":
            worker.pop("attention_sequential_transport_profile")
        elif tamper == "crossed-capture-profile":
            crossed = dict(
                capture["attention_sequential_transport_profile"]
            )
            crossed["profile_id"] = "crossed-profile"
            crossed.pop("profile_sha256")
            crossed["profile_sha256"] = canonical_sha256(crossed)
            capture["attention_sequential_transport_profile"] = crossed
        elif tamper == "profile-path-with-legacy-documents":
            changed.pop("attention_sequential_transport_profile")
            command.pop("attention_sequential_transport_profile")
            worker.pop("attention_sequential_transport_profile")
            capture.pop("attention_sequential_transport_profile")
        else:
            relative = [
                "capture_quarantine",
                run.run_directory.parents[1].name,
            ]
            if tamper == "missing-profile-scope":
                relative.append(config.attempt_id)
            elif tamper == "wrong-profile-scope":
                relative.extend(
                    ("attention-profile-wrong", config.attempt_id)
                )
            else:
                relative.extend(
                    (scope, "unexpected-extra", config.attempt_id)
                )
            changed_run = work.joinpath(*relative)
            changed_quarantine = changed_run / "quarantine-tampered"
            changed_quarantine.mkdir(parents=True)
            lane["paths"]["run_directory"] = str(changed_run)
            lane["paths"]["quarantine_directory"] = str(
                changed_quarantine
            )
        _reseal_plan_document(changed)
        with pytest.raises(
            validator.Lfm2CpcfV5PostSuccessValidationError,
            match=(
                "profile bindings|unknown or invalid|sealed lifecycle"
            ),
        ):
            validator._plan_lane(
                changed,
                ratio=4,
                attempt_id=config.attempt_id,
            )


def test_plan_file_may_be_outside_sealed_work_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _completed_supervisor_attempt(tmp_path)
    external_root = tmp_path / "external-plans"
    external_root.mkdir()
    external_plan = external_root / "execute-4x-plan.json"
    config.plan_path.replace(external_plan)
    config = replace(config, plan_path=external_plan)
    _configure_completed_attempt_authentication(config, monkeypatch)

    run = validator._authenticate_run(config)

    assert run.run_directory.is_relative_to(
        Path(run.manifest["worker_config"]["work_directory"])
    )
    assert external_plan.parent != Path(
        run.manifest["worker_config"]["work_directory"]
    )


def test_profile_bound_resealed_provenance_tampers_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _completed_supervisor_attempt(
        tmp_path,
        profile_bound=True,
    )
    plan = _configure_completed_attempt_authentication(
        config,
        monkeypatch,
    )
    paths = plan["lanes"][0]["paths"]
    process_manifest_path = Path(paths["process_manifest_json"])
    diagnostic_path = Path(paths["diagnostic_json"])
    execution_path = Path(paths["execution_receipt_json"])
    original_plan = config.plan_path.read_bytes()
    original_manifest = process_manifest_path.read_bytes()
    original_diagnostic = diagnostic_path.read_bytes()
    original_execution = execution_path.read_bytes()

    for tamper in (
        "diagnostic-profile",
        "request-profile",
        "compiled-runtime-policy",
        "resume-runtime-policy",
        "resume-extra-key",
        "execution-profile",
        "execution-resume-binding",
    ):
        diagnostic_path.write_bytes(original_diagnostic)
        execution_path.write_bytes(original_execution)
        target = (
            execution_path
            if tamper.startswith("execution-")
            else diagnostic_path
        )

        def mutate(value: dict[str, Any]) -> None:
            if tamper == "diagnostic-profile":
                value["complete_model_fold_policy_sha256"] = _digest(
                    "crossed-diagnostic-policy"
                )
            elif tamper == "request-profile":
                value["request"][
                    "complete_model_fold_policy_sha256"
                ] = _digest("crossed-request-policy")
            elif tamper == "compiled-runtime-policy":
                value["compiled_model_fold"]["policy"]["schema"] = (
                    "crossed-runtime-policy"
                )
            elif tamper in {
                "resume-runtime-policy",
                "resume-extra-key",
            }:
                binding = value["resume_policy_binding"]
                if tamper == "resume-runtime-policy":
                    binding["runtime_policy_sha256"] = _digest(
                        "crossed-runtime-policy"
                    )
                else:
                    binding["unexpected"] = False
                binding.pop("binding_sha256")
                binding["binding_sha256"] = canonical_sha256(binding)
            elif tamper == "execution-profile":
                value["model_fold_policy_identity_sha256"] = _digest(
                    "crossed-execution-policy"
                )
            else:
                binding = value["resume_policy_binding"]
                binding["runtime_policy_sha256"] = _digest(
                    "crossed-execution-runtime"
                )
                binding.pop("binding_sha256")
                binding["binding_sha256"] = canonical_sha256(binding)

        _rewrite_sealed_json(
            target,
            seal_field="receipt_sha256",
            mutate=mutate,
        )
        with pytest.raises(
            validator.Lfm2CpcfV5PostSuccessValidationError
        ):
            validator._authenticate_run(config)

    diagnostic_path.write_bytes(original_diagnostic)
    execution_path.write_bytes(original_execution)
    changed_plan = json.loads(json.dumps(plan))
    changed_command = changed_plan["lanes"][0]["command_manifest"]
    diagnostic_command = changed_command["diagnostic_runner_command"]
    flag_index = diagnostic_command.index(
        "--attention-sequential-transport-profile"
    )
    del diagnostic_command[flag_index : flag_index + 2]
    _reseal_plan_document(changed_plan)
    config.plan_path.write_bytes(
        validator.supervisor._canonical_json(changed_plan) + b"\n"
    )
    process_manifest_path.write_bytes(
        validator.supervisor._canonical_json(
            changed_plan["lanes"][0]["command_manifest"]
        )
        + b"\n"
    )
    with pytest.raises(
        validator.Lfm2CpcfV5PostSuccessValidationError,
        match="command, config, paths, or attention profile",
    ):
        validator._authenticate_run(config)
    config.plan_path.write_bytes(original_plan)
    process_manifest_path.write_bytes(original_manifest)

    original_catalog = plan["input_bindings"][
        "selective_catalog_preparation"
    ]
    for tamper in ("missing-profile", "crossed-policy"):
        changed_catalog = dict(original_catalog)
        if tamper == "missing-profile":
            changed_catalog.pop(
                "attention_sequential_transport_profile"
            )
        else:
            changed_catalog[
                "complete_model_fold_policy_sha256"
            ] = _digest("crossed-catalog-policy")

        def authenticate_tampered_catalog(
            runner_config: Any,
            *,
            expected_receipt_sha256: str | None = None,
        ) -> MappingProxyType:
            assert expected_receipt_sha256 is None
            validator.fold_runner._authenticate_preparation_policy_identity(
                changed_catalog,
                config=runner_config,
            )
            return MappingProxyType(changed_catalog)

        monkeypatch.setattr(
            validator.fold_runner,
            "_read_selective_catalog_preparation",
            authenticate_tampered_catalog,
        )
        with pytest.raises(
            validator.Lfm2CpcfV5PostSuccessValidationError,
            match="Prepared selective catalog",
        ):
            validator._authenticate_run(config)


def test_rejects_tampered_sealed_worker_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _completed_supervisor_attempt(tmp_path)
    plan = json.loads(config.plan_path.read_text(encoding="ascii"))
    forged_root = tmp_path / "forged-work-root"
    forged_root.mkdir()
    lane = plan["lanes"][0]
    command = lane["command_manifest"]
    command.pop("command_sha256")
    command["worker_config"]["work_directory"] = str(forged_root)
    command["command_sha256"] = canonical_sha256(command)
    lane.pop("lane_sha256")
    lane["lane_sha256"] = canonical_sha256(lane)
    plan.pop("plan_sha256")
    plan["plan_sha256"] = canonical_sha256(plan)
    config.plan_path.write_bytes(
        validator.supervisor._canonical_json(plan) + b"\n"
    )
    _configure_completed_attempt_authentication(config, monkeypatch)

    with pytest.raises(
        validator.Lfm2CpcfV5PostSuccessValidationError,
        match="outside its sealed work directory",
    ):
        validator._authenticate_run(config)


@pytest.mark.parametrize(
    "tamper",
    ("missing-catalog", "legacy-source-binding", "tampered-catalog"),
)
def test_rejects_incomplete_or_legacy_catalog_bound_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    config = _completed_supervisor_attempt(tmp_path)
    plan = _configure_completed_attempt_authentication(
        config,
        monkeypatch,
    )
    paths = plan["lanes"][0]["paths"]
    execution_path = Path(paths["execution_receipt_json"])
    diagnostic_path = Path(paths["diagnostic_json"])

    if tamper == "missing-catalog":
        _rewrite_sealed_json(
            execution_path,
            seal_field="receipt_sha256",
            mutate=lambda value: value.pop(
                "selective_catalog_artifact"
            ),
        )
    elif tamper == "legacy-source-binding":
        execution = json.loads(
            execution_path.read_text(encoding="ascii")
        )
        diagnostic = json.loads(
            diagnostic_path.read_text(encoding="ascii")
        )
        guard_identity = diagnostic[
            "compiler_runtime_fold_only_guard"
        ]["guard_implementation"]["identity_sha256"]
        legacy = validator.supervisor._sha(
            {
                "selective_source_sha256": execution[
                    "selective_source_sha256"
                ],
                "compiler_implementation_sha256": execution[
                    "compiler_implementation_sha256"
                ],
                "runner_source_sha256": execution[
                    "runner_source_sha256"
                ],
                "fold_only_runtime_guard_identity_sha256": (
                    guard_identity
                ),
            }
        )
        _rewrite_sealed_json(
            execution_path,
            seal_field="receipt_sha256",
            mutate=lambda value: value.__setitem__(
                "source_binding_sha256",
                legacy,
            ),
        )
    else:
        _rewrite_sealed_json(
            diagnostic_path,
            seal_field="receipt_sha256",
            mutate=lambda value: value[
                "selective_catalog_artifact"
            ].__setitem__(
                "binding_sha256",
                _digest("tampered-catalog-binding"),
            ),
        )

    with pytest.raises(
        validator.Lfm2CpcfV5PostSuccessValidationError
    ):
        validator._authenticate_run(config)


def test_fresh_replay_verifies_dense_exact_inventory_and_guarded_hashes(
    tmp_path: Path,
    parent_envelope: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence(parent_envelope)
    run = _fake_run(tmp_path, parent_envelope, evidence)
    model = torch.nn.Linear(2, 2)
    inventory = dict(independent_parameter_inventory(model))
    metadata_inventory = dict(inventory)
    metadata_inventory.pop("checkpoint_byte_size_used")
    run = replace(
        run,
        artifact_metadata=MappingProxyType(
            {
                "parameter_inventory_sha256": inventory[
                    "inventory_sha256"
                ],
                "parameter_inventory": metadata_inventory,
            }
        ),
    )
    original = {
        "schema": "reload",
        "capture_receipt_sha256": run.quarantine_receipt["receipt_sha256"],
        "destination_path_sha256": _digest("destination"),
        "model_state_sha256": run.execution["target_state_sha256"],
        "output_sha256": evidence.output_sha256,
        "evaluation_sha256": evidence.evaluation_sha256,
        "artifact_inventory_sha256": _digest("artifact-inventory"),
        "loader_sha256": _digest("loader"),
        "evaluator_sha256": _digest("evaluator"),
        "loader_path_challenge_performed": True,
        "loader_path_challenge_exception_type": "FixtureError",
        "fresh_final_destination_reload_performed": True,
        "fresh_reload_execution": {},
        "promotable": False,
        "publication_authorized": False,
        "requires_reviewed_rerun": True,
        "verification_sha256": _digest("original-verification"),
    }
    run = replace(
        run,
        execution=MappingProxyType(
            {
                **run.execution,
                "target_unique_parameters": inventory["unique_parameters"],
                "final_reload_verification": original,
            }
        ),
    )
    observed = {
        **_guard(),
        "training_steps": 0,
        "parent_forward_calls": 0,
        "parent_inference_calls": 0,
    }
    fresh = {
        **original,
        "capture_observed_audit": observed,
        "verification_sha256": _digest("fresh-verification"),
    }
    fake_loaded = SimpleNamespace(
        path=run.quarantine_directory,
        artifact_path=run.candidate_directory,
        receipt=SimpleNamespace(
            as_dict=lambda: dict(run.quarantine_receipt)
        ),
        model=model,
        verification_receipt=fresh,
        promotable=False,
        publication_authorized=False,
        requires_reviewed_rerun=True,
    )
    monkeypatch.setattr(
        validator,
        "load_quarantined_candidate",
        lambda *args, **kwargs: fake_loaded,
    )
    runtime = _fake_runtime(parent_envelope)
    runtime = replace(
        runtime,
        evaluator=lambda _model: evidence,
    )

    returned_evidence, returned_inventory, audit = (
        validator._fresh_replay(run, runtime)
    )

    assert returned_evidence.output_sha256 == evidence.output_sha256
    assert returned_inventory["unique_parameters"] == inventory[
        "unique_parameters"
    ]
    assert audit["export_guard"]["clean"] is True


@pytest.mark.parametrize("profile_bound", (False, True))
def test_one_shot_validator_publishes_complete_sibling_receipt_last(
    tmp_path: Path,
    parent_envelope: Any,
    monkeypatch: pytest.MonkeyPatch,
    profile_bound: bool,
) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text("{}\n", encoding="ascii")
    config = validator.PostSuccessValidationConfig(
        plan_path=plan,
        ratio=4,
        attempt_id="attempt-0002",
        expected_validator_source_sha256=_validator_sha(),
    )
    evidence = _evidence(parent_envelope)
    run = _fake_run(tmp_path, parent_envelope, evidence)
    profile_policy_fields: dict[str, Any] = {}
    if profile_bound:
        profile_policy_fields = {
            "attention_sequential_transport_profile": (
                validator.fold_runner
                .V5AttentionSequentialTransportRunnerProfile()
                .as_dict()
            ),
            "complete_model_fold_policy_sha256": _digest(
                "complete-model-fold-policy"
            ),
            "model_fold_policy_identity_sha256": _digest(
                "model-fold-policy-identity"
            ),
        }
        resume_binding_body = {
            "schema": validator.fold_runner.RESUME_POLICY_BINDING_SCHEMA,
            **profile_policy_fields,
            "runtime_policy_sha256": _digest("runtime-policy"),
            "runtime_policy_contains_complete_model_fold_policy": True,
            "cross_profile_resume_permitted": False,
        }
        resume_binding = {
            **resume_binding_body,
            "binding_sha256": canonical_sha256(resume_binding_body),
        }
        run = replace(
            run,
            manifest=MappingProxyType(
                {
                    **run.manifest,
                    **profile_policy_fields,
                }
            ),
            execution=MappingProxyType(
                {
                    **run.execution,
                    **profile_policy_fields,
                    "resume_policy_binding": resume_binding,
                }
            ),
        )
    runtime = _fake_runtime(parent_envelope)
    inventory = {
        "method": "fixture",
        "unique_parameters": 6,
        "independent_methods_agree": True,
        "checkpoint_byte_size_used": False,
        "quantization_adjustment_used": False,
        "pruning_mask_adjustment_used": False,
    }
    inventory["inventory_sha256"] = canonical_sha256(inventory)
    replay_audit = {
        "fresh_verification": {
            "verification_sha256": _digest("fresh-verification"),
        },
        "export_guard": _guard(),
    }
    monkeypatch.setattr(validator, "_authenticate_run", lambda _config: run)
    monkeypatch.setattr(
        validator,
        "_build_replay_runtime",
        lambda _run: runtime,
    )
    monkeypatch.setattr(
        validator,
        "_fresh_replay",
        lambda _run, _runtime: (evidence, inventory, replay_audit),
    )
    monkeypatch.setattr(
        validator.holdout_runner,
        "evaluate_lfm2_intelligence_holdout",
        lambda evaluation_config: _fake_cached_evaluation(
            evaluation_config,
            envelope=parent_envelope,
        ),
    )
    monkeypatch.setattr(
        validator.supervisor,
        "_retained_inventory_matches",
        lambda *_args, **_kwargs: True,
    )

    receipt = validator.validate_lfm2_cpcf_v5_quarantine(config)

    root = Path(receipt["output_directory"])
    assert root.parent == run.run_directory.parent
    assert root not in run.quarantine_directory.parents
    assert receipt["status"] == "validated_pending_manual_review"
    assert receipt["case_count"] == 64
    assert receipt["manual_review_case_count"] == 6
    assert receipt["training_steps"] == 0
    assert receipt["parameters_pruned"] == 0
    assert receipt["parameters_quantized"] == 0
    assert receipt["parent_model_loaded"] is False
    if profile_bound:
        assert all(
            receipt[name] == value
            for name, value in profile_policy_fields.items()
        )
        assert receipt["resume_policy_binding"] == resume_binding
    else:
        assert all(
            name not in receipt
            for name in validator.fold_runner._OPT_IN_POLICY_IDENTITY_FIELDS
        )
        assert "resume_policy_binding" not in receipt
    assert set(receipt["files"]) == {
        "automatic_intelligence_report",
        "behavioral_evaluation",
        "behavioral_output",
        "candidate_outputs",
        "fresh_parameter_inventory",
        "manual_review_packet",
        "standalone_reload_verification",
        "whole_response_comparison",
    }
    comparison = json.loads(
        (root / "whole-response-comparison.json").read_text(
            encoding="ascii"
        )
    )
    assert comparison["case_count"] == 64
    assert len(comparison["rows"]) == 64
    assert (root / "post-success-validation-receipt.json").is_file()

    with pytest.raises(
        validator.Lfm2CpcfV5PostSuccessValidationError,
        match="appeared",
    ):
        validator.validate_lfm2_cpcf_v5_quarantine(config)


def test_quarantine_change_withholds_completion_receipt(
    tmp_path: Path,
    parent_envelope: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text("{}\n", encoding="ascii")
    config = validator.PostSuccessValidationConfig(
        plan_path=plan,
        ratio=4,
        attempt_id="attempt-0002",
        expected_validator_source_sha256=_validator_sha(),
    )
    evidence = _evidence(parent_envelope)
    run = _fake_run(tmp_path, parent_envelope, evidence)
    runtime = _fake_runtime(parent_envelope)
    inventory = {
        "inventory_sha256": _digest("inventory"),
        "unique_parameters": 6,
    }
    replay_audit = {
        "fresh_verification": {
            "verification_sha256": _digest("fresh-verification"),
        },
        "export_guard": _guard(),
    }
    monkeypatch.setattr(validator, "_authenticate_run", lambda _config: run)
    monkeypatch.setattr(
        validator,
        "_build_replay_runtime",
        lambda _run: runtime,
    )
    monkeypatch.setattr(
        validator,
        "_fresh_replay",
        lambda _run, _runtime: (evidence, inventory, replay_audit),
    )
    monkeypatch.setattr(
        validator.holdout_runner,
        "evaluate_lfm2_intelligence_holdout",
        lambda evaluation_config: _fake_cached_evaluation(
            evaluation_config,
            envelope=parent_envelope,
        ),
    )
    monkeypatch.setattr(
        validator.supervisor,
        "_retained_inventory_matches",
        lambda *_args, **_kwargs: False,
    )

    with pytest.raises(
        validator.Lfm2CpcfV5PostSuccessValidationError,
        match="changed during",
    ):
        validator.validate_lfm2_cpcf_v5_quarantine(config)

    assert not (
        run.output_directory / "post-success-validation-receipt.json"
    ).exists()
