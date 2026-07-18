from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Thread

import pytest

from neural_forge.dynamic_circuit_ir import (
    CircuitKind,
    CircuitNode,
    CircuitPort,
    ModelCircuitGraph,
    ModalityInterface,
    ParameterSpec,
    PortKind,
    canonical_sha256,
)
from neural_forge.dynamic_fold_artifact import (
    DynamicFoldArtifactCertificate,
    _write_provenance_documents,
    load_dynamic_fold_artifact_certificate,
)
from neural_forge.dynamic_fold_compiler import (
    DynamicFoldCompilation,
    PublishedDynamicFoldCompilation,
    SynthesizedPublishedDynamicFoldCompilation,
)
from neural_forge.dynamic_fold_executor import (
    DynamicFoldRun,
    FoldAttempt,
    FoldDecision,
    fingerprint_model_state,
)
from neural_forge.dynamic_fold_planner import (
    DynamicFoldPlan,
    FoldAction,
    SourceCircuitRef,
)
from neural_forge.dynamic_fold_registry import (
    ClosureLevel,
    ExperimentLedger,
    ExperimentSignature,
    LedgerDecision,
    LedgerRecord,
    RecordStatus,
)
from neural_forge.dynamic_search import (
    AutomaticDynamicFoldSearchSystem,
    DynamicFoldPortfolioRegistry,
    DynamicFoldSearchCandidate,
    DynamicFoldSearchError,
    DynamicFoldSearchObjective,
    FailureDisposition,
)
from neural_forge.dynamic_target_synthesis import (
    COMPILER_BLANK_METHOD,
    CompilerBlankProvenance,
    DynamicTargetRequest,
    DynamicTargetResourceProfile,
    DynamicTargetSynthesisResult,
    StructuralBufferPolicy,
    compiler_authenticated_topology_fingerprint,
    serialized_inference_state_inventory,
    validate_structural_runtime_buffers,
)


torch = pytest.importorskip("torch")


class _FixtureTargetModel(torch.nn.Module):
    def __init__(self, unique_parameters: int) -> None:
        super().__init__()
        self.unit = torch.nn.Module()
        self.unit.weight = torch.nn.Parameter(
            torch.arange(
                1,
                unique_parameters + 1,
                dtype=torch.float16,
            )
        )

    def forward(self):
        return self.unit.weight


class CandidateRejected(RuntimeError):
    pass


class InfrastructureFailure(RuntimeError):
    pass


class SealedCandidateRejected(CandidateRejected):
    def __init__(self, message: str, report: dict) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class UncertifiedCompleteResult:
    complete: bool = True

    def as_dict(self):
        return {"complete": self.complete}


def _graph(
    *,
    model_name: str,
    family: str,
    unique_parameters: int,
    tokenizer_fingerprint: str | None,
) -> ModelCircuitGraph:
    parameter = ParameterSpec(
        path="unit.weight",
        role="normalization.scale",
        shape=(unique_parameters,),
        dtype="float16",
        storage_group=f"{model_name}.unit.weight",
        trainable=True,
    )
    node = CircuitNode(
        node_id="unit",
        kind=CircuitKind.NORM,
        order=0,
        path="unit",
        layer_index=0,
        inputs=(
            CircuitPort(
                name="input",
                kind=PortKind.GENERIC,
                dimensions=(unique_parameters,),
                modality="text",
            ),
        ),
        outputs=(
            CircuitPort(
                name="output",
                kind=PortKind.GENERIC,
                dimensions=(unique_parameters,),
                modality="text",
            ),
        ),
        parameters=(parameter,),
    )
    return ModelCircuitGraph(
        model_name=model_name,
        family=family,
        model_class="TestFoldModel",
        nodes=(node,),
        edges=(),
        interfaces=(
            ModalityInterface(
                interface_id="text.input",
                modality="text",
                direction="input",
                node_id="unit",
                port_name="input",
                representation="test.hidden",
            ),
            ModalityInterface(
                interface_id="text.output",
                modality="text",
                direction="output",
                node_id="unit",
                port_name="output",
                representation="test.hidden",
            ),
        ),
        logical_parameter_count=unique_parameters,
        unique_parameter_count=unique_parameters,
        unique_parameter_bytes=unique_parameters * 2,
        config_fingerprint=canonical_sha256(
            {
                "model_name": model_name,
                "family": family,
                "unique_parameters": unique_parameters,
            }
        ),
        tokenizer_fingerprint=tokenizer_fingerprint,
    )


def _published_result(
    *,
    target_request: DynamicTargetRequest,
    evidence_fingerprint: str,
    destination: Path,
    quality: float,
    certificate_changes: dict | None = None,
    attempt_changes: dict | None = None,
    run_changes: dict | None = None,
    plan_changes: dict | None = None,
    method_id: str = "test.dynamic-fold",
    duplicate_parent_graph: bool = False,
) -> SynthesizedPublishedDynamicFoldCompilation:
    graph = _graph(
        model_name=target_request.child_name,
        family=target_request.target_family,
        unique_parameters=target_request.exact_unique_parameters,
        tokenizer_fingerprint=target_request.tokenizer_fingerprint,
    )
    target_adapter_id = target_request.adapter_id or "auto-target"
    target_model = _FixtureTargetModel(target_request.exact_unique_parameters)
    state_inventory = serialized_inference_state_inventory(target_model)
    structural_policy = StructuralBufferPolicy(
        policy_id="compiler.test.dynamic-search.empty-buffers",
        version=1,
        adapter_id=target_adapter_id,
        adapter_version=1,
        entries=(),
        source="compiler_empty_policy",
        compiler_reviewed=True,
    )
    structural_audit = validate_structural_runtime_buffers(
        target_model,
        structural_policy,
        expected_adapter_id=target_adapter_id,
        expected_adapter_version=1,
    )
    parent_graph = _graph(
        model_name="parent",
        family="qwen3",
        unique_parameters=target_request.exact_unique_parameters + 100,
        tokenizer_fingerprint="b" * 64,
    )
    blank_state_fingerprint = "3" * 64
    final_state_fingerprint = fingerprint_model_state(target_model)
    parent_state_fingerprint = "2" * 64
    signature = ExperimentSignature(
        method_id=method_id,
        method_version=1,
        family_id="test.dynamic-search",
        parent_graph_fingerprints=(parent_graph.fingerprint,),
        parent_state_fingerprints=(("parent", parent_state_fingerprint),),
        child_graph_fingerprint=graph.fingerprint,
        child_state_fingerprint=blank_state_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        source_unit_ids=("parent:unit",),
        target_unit_ids=("unit",),
        topology_transition="one-parent-to-smaller-child",
        selection_policy="test-fixture",
        fit_objective="closed-form-fold",
        promotion_gate="whole-output-and-fresh-reload",
        zero_gradient=True,
        configuration=target_request.configuration,
    )
    action = FoldAction(
        action_id="fold.unit",
        sequence_index=0,
        method_id=method_id,
        sources=(SourceCircuitRef(parent_name="parent", node_ids=("unit",)),),
        target_node_ids=("unit",),
        target_parameter_paths=graph.parameter_paths,
        target_storage_groups=(f"{target_request.child_name}.unit.weight",),
        dependencies=(),
        evidence_requirements=("sealed-parent-evidence",),
        source_logical_parameters=parent_graph.logical_parameter_count,
        target_logical_parameters=graph.logical_parameter_count,
        configuration=target_request.configuration,
        signature=signature,
        ledger_decision=LedgerDecision(
            allowed=True,
            reason="No closed experiment blocks this fixture action.",
        ),
    )
    plan = DynamicFoldPlan(
        parent_graph_fingerprints=(parent_graph.fingerprint,),
        parent_state_fingerprints=(("parent", parent_state_fingerprint),),
        child_graph_fingerprint=graph.fingerprint,
        child_state_fingerprint=blank_state_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        exact_target_unique_parameters=target_request.exact_unique_parameters,
        actions=(action,),
        gaps=(),
        registry_fingerprint="4" * 64,
        ledger_fingerprint="5" * 64,
    )
    plan = replace(plan, **dict(plan_changes or {}))
    evidence_payload_fingerprint = "9" * 64
    evidence_binding_fingerprint = "a" * 64
    attempt_evidence_manifest = {
        "schema": "neural-forge-fold-evidence-derivation-v1",
        "action_id": action.action_id,
        "action_sha256": action.action_sha256,
        "method_id": action.method_id,
        "child_state_fingerprint": blank_state_fingerprint,
        "evidence_fingerprint": evidence_fingerprint,
        "evidence_root_fingerprint": evidence_fingerprint,
        "evidence_payload_fingerprint": evidence_payload_fingerprint,
        "evidence_payload_integrity_mode": "canonical-content",
        "evidence_channels": list(action.evidence_requirements),
        "evidence_binding_fingerprint": evidence_binding_fingerprint,
    }
    attempt = FoldAttempt(
        action_id=action.action_id,
        method_id=method_id,
        execution_signature=signature,
        evidence_fingerprint=evidence_fingerprint,
        evidence_root_fingerprint=evidence_fingerprint,
        incumbent_state_fingerprint=blank_state_fingerprint,
        candidate_state_fingerprint=final_state_fingerprint,
        decision=FoldDecision.accept("fold action passed"),
        transaction_report={
            "committed": True,
            "changed_state_paths": list(graph.parameter_paths),
        },
        rollback_exact=None,
        ledger_record_sha256="6" * 64,
        evidence_payload_fingerprint=evidence_payload_fingerprint,
        evidence_payload_integrity_mode="canonical-content",
        evidence_channels=action.evidence_requirements,
        evidence_binding_fingerprint=evidence_binding_fingerprint,
        evidence_derivation_manifest=attempt_evidence_manifest,
        evidence_derivation_fingerprint=canonical_sha256(attempt_evidence_manifest),
        evidence_payload_integrity_reverified=True,
    )
    attempt = replace(attempt, **dict(attempt_changes or {}))
    final_evidence_manifest = {
        **attempt_evidence_manifest,
        "child_state_fingerprint": final_state_fingerprint,
    }
    run = DynamicFoldRun(
        plan_fingerprint=plan.fingerprint,
        status="complete",
        initial_state_fingerprint=blank_state_fingerprint,
        final_state_fingerprint=final_state_fingerprint,
        bootstrap_report={
            "initialized": True,
            "changed_state_paths": [],
            "non_mutating_initializer_verified": True,
        },
        attempts=(attempt,),
        accepted_action_ids=(action.action_id,),
        completed_target_parameter_paths=graph.parameter_paths,
        final_decision=FoldDecision.accept("behavior passed"),
        final_evidence_fingerprint=evidence_fingerprint,
        final_evidence_root_fingerprint=evidence_fingerprint,
        ledger_fingerprint="8" * 64,
        operator_implementation_fingerprint="b" * 64,
        operator_static_zero_training_audit_passed=True,
        final_evidence_payload_fingerprint=evidence_payload_fingerprint,
        final_evidence_payload_integrity_mode="canonical-content",
        final_evidence_channels=action.evidence_requirements,
        final_evidence_binding_fingerprint=evidence_binding_fingerprint,
        final_evidence_derivation_manifest=final_evidence_manifest,
        final_evidence_derivation_fingerprint=canonical_sha256(final_evidence_manifest),
        final_evidence_payload_integrity_reverified=True,
        observed_parent_module_forward_calls=0,
        observed_parent_root_forward_calls=0,
        observed_final_child_module_forward_calls=1,
        observed_final_child_root_forward_calls=1,
        observed_final_child_inference_entrypoint_calls=1,
    )
    run = replace(run, **dict(run_changes or {}))
    compilation = DynamicFoldCompilation(
        parent_graphs=(
            (parent_graph, parent_graph) if duplicate_parent_graph else (parent_graph,)
        ),
        child_graph=graph,
        plan=plan,
        run=run,
    )
    blank_provenance = CompilerBlankProvenance(
        method=COMPILER_BLANK_METHOD,
        compiler_created_blank=True,
        parent_child_alias_free=True,
        zeroed_under_no_grad=True,
        dense_materialized_parameters=True,
        all_parameters_zero=True,
        parameter_gradients_absent=True,
        runtime_buffers_preserved=True,
        unique_parameter_object_count=1,
        unique_parameter_storage_count=1,
        unique_parameter_element_count=(target_request.exact_unique_parameters),
        unique_parameter_bytes=(target_request.exact_unique_parameters * 2),
        buffer_tensor_count=0,
        parameter_layout_fingerprint="e" * 64,
        buffer_layout_fingerprint="f" * 64,
        buffer_state_fingerprint_before="1" * 64,
        buffer_state_fingerprint_after="1" * 64,
        blank_state_fingerprint=blank_state_fingerprint,
        structural_buffer_policy_digest=structural_policy.digest,
        structural_buffer_audit_digest=structural_audit.digest,
        structural_buffer_policy_verified=True,
        nonzero_structural_buffer_count=structural_audit.nonzero_buffer_count,
    )
    target = DynamicTargetSynthesisResult(
        request=target_request,
        adapter_id=target_adapter_id,
        adapter_version=1,
        model=target_model,
        graph=graph,
        resource_profile=DynamicTargetResourceProfile(
            unique_parameters=target_request.exact_unique_parameters,
            parameter_bytes=target_request.exact_unique_parameters * 2,
            family=target_request.target_family,
            modalities=target_request.target_modalities,
            layer_count=1,
            serialized_state_elements=target_request.exact_unique_parameters,
            serialized_state_bytes=target_request.exact_unique_parameters * 2,
            persistent_buffer_elements=0,
            persistent_buffer_bytes=0,
            runtime_state_elements=state_inventory["runtime_state_elements"],
            runtime_state_bytes=state_inventory["runtime_state_bytes"],
            runtime_buffer_elements=0,
            runtime_buffer_bytes=0,
            nonpersistent_buffer_elements=0,
            nonpersistent_buffer_bytes=0,
            runtime_named_buffer_count=0,
            nonpersistent_named_buffer_count=0,
            runtime_unique_buffer_view_count=0,
            runtime_tensor_state_fingerprint=state_inventory[
                "runtime_tensor_state_fingerprint"
            ],
        ),
        topology_fingerprint=compiler_authenticated_topology_fingerprint(
            request=target_request,
            adapter_id=target_adapter_id,
            adapter_version=1,
            graph=graph,
            blank_provenance=blank_provenance,
        ),
        adapter_report={"quality": quality},
        parent_graph_fingerprints=(("parent", parent_graph.fingerprint),),
        blank_state_fingerprint=blank_state_fingerprint,
        blank_provenance=blank_provenance,
        topology_provenance_verified=True,
        structural_buffer_policy=structural_policy,
        structural_buffer_audit=structural_audit,
    )
    destination.mkdir()
    (destination / "child").mkdir()
    model_payload = b"folded-child"
    (destination / "child" / "model.bin").write_bytes(model_payload)
    provenance = _write_provenance_documents(
        destination,
        compilation,
        dynamic_target=target,
    )
    artifact_tree_sha256 = canonical_sha256(
        (
            {
                "path": "model.bin",
                "size_bytes": len(model_payload),
                "sha256": hashlib.sha256(model_payload).hexdigest(),
            },
        )
    )
    behavior_manifest_body = {
        "schema": "test-behavior-gate-manifest-v1",
        "gate_id": "test.whole-output-gate",
        "gate_version": 1,
        "baseline_fingerprint": "a" * 64,
        "cases": ["test-case"],
    }
    behavior_manifest_fingerprint = canonical_sha256(behavior_manifest_body)
    behavior_manifest = {
        **behavior_manifest_body,
        "manifest_fingerprint": behavior_manifest_fingerprint,
    }
    provisional = DynamicFoldArtifactCertificate(
        path=str(destination.absolute()),
        plan_fingerprint=plan.fingerprint,
        run_fingerprint=run.fingerprint,
        child_graph_fingerprint=graph.fingerprint,
        child_state_fingerprint=run.final_state_fingerprint,
        exact_unique_parameters=target_request.exact_unique_parameters,
        artifact_tree_sha256=artifact_tree_sha256,
        artifact_file_count=1,
        artifact_total_bytes=len(model_payload),
        fresh_reload_decision=FoldDecision.accept("fresh reload passed"),
        saver_id="test-saver",
        loader_id="test-loader",
        compilation_optimizer_constructions=0,
        compilation_optimizer_steps=0,
        compilation_backward_calls=0,
        compilation_autograd_grad_calls=0,
        optimizer_constructions=0,
        optimizer_steps=0,
        backward_calls=0,
        autograd_grad_calls=0,
        certificate_sha256="0" * 64,
        evidence_fingerprint=evidence_fingerprint,
        behavior_gate_id="test.whole-output-gate",
        behavior_gate_version=1,
        behavior_baseline_fingerprint="a" * 64,
        behavior_gate_manifest_fingerprint=(behavior_manifest_fingerprint),
        behavior_gate_manifest=behavior_manifest,
        fresh_reload_state_inventory=state_inventory,
        fresh_reload_module_forward_calls=1,
        fresh_reload_root_forward_calls=1,
        fresh_reload_inference_entrypoint_calls=1,
        preparation_guard_applied=True,
        preparation_optimizer_constructions=0,
        preparation_optimizer_steps=0,
        preparation_backward_calls=0,
        preparation_autograd_grad_calls=0,
        publication_guard_covers_save_load=True,
        publication_optimizer_constructions=0,
        publication_optimizer_steps=0,
        publication_backward_calls=0,
        publication_autograd_grad_calls=0,
        provenance_plan_sha256=provenance["plan_sha256"],
        provenance_run_sha256=provenance["run_sha256"],
        provenance_child_graph_sha256=provenance["child_graph_sha256"],
        provenance_parent_graphs_sha256=provenance["parent_graphs_sha256"],
        provenance_parent_graphs_fingerprint=(provenance["parent_graphs_fingerprint"]),
        provenance_target_sha256=provenance["target_sha256"],
        provenance_target_fingerprint=provenance["target_fingerprint"],
        dynamic_target_provenance_present=True,
        provenance_tree_sha256=provenance["tree_sha256"],
        loader_path_challenge_performed=True,
        loader_path_challenge_exception_type="builtins.FileNotFoundError",
        saver_implementation_fingerprint="c" * 64,
        saver_resource_fingerprint="2" * 64,
        loader_implementation_fingerprint="d" * 64,
        loader_resource_fingerprint="3" * 64,
    )
    changes = dict(certificate_changes or {})
    if {
        "behavior_gate_id",
        "behavior_gate_version",
        "behavior_baseline_fingerprint",
    } & set(changes):
        changed_manifest_body = {
            **behavior_manifest_body,
            "gate_id": changes.get(
                "behavior_gate_id",
                provisional.behavior_gate_id,
            ),
            "gate_version": changes.get(
                "behavior_gate_version",
                provisional.behavior_gate_version,
            ),
            "baseline_fingerprint": changes.get(
                "behavior_baseline_fingerprint",
                provisional.behavior_baseline_fingerprint,
            ),
        }
        changed_manifest_fingerprint = canonical_sha256(changed_manifest_body)
        changes.setdefault(
            "behavior_gate_manifest_fingerprint",
            changed_manifest_fingerprint,
        )
        changes.setdefault(
            "behavior_gate_manifest",
            {
                **changed_manifest_body,
                "manifest_fingerprint": changed_manifest_fingerprint,
            },
        )
    forced_certificate_sha256 = changes.pop("certificate_sha256", None)
    provisional = replace(provisional, **changes)
    artifact = replace(
        provisional,
        certificate_sha256=canonical_sha256(provisional.body()),
    )
    if forced_certificate_sha256 is not None:
        artifact = replace(
            artifact,
            certificate_sha256=forced_certificate_sha256,
        )
    certificate_payload = artifact.as_dict()
    (destination / "certificate.json").write_text(
        json.dumps(certificate_payload, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return SynthesizedPublishedDynamicFoldCompilation(
        target=target,
        published=PublishedDynamicFoldCompilation(
            compilation=compilation,
            artifact=artifact,
        ),
    )


class FakeRunner:
    def __init__(self, outcomes):
        self.outcomes = dict(outcomes)
        self.calls = []

    def run(
        self,
        *,
        parent_models,
        target_request,
        evidence_fingerprint,
        destination,
        parent_tokenizer_fingerprints=None,
    ):
        arm = dict(target_request.configuration)["arm"]
        self.calls.append(
            {
                "arm": arm,
                "parents": tuple(sorted(parent_models)),
                "evidence_fingerprint": evidence_fingerprint,
                "destination": str(destination),
                "parent_tokenizer_fingerprints": (parent_tokenizer_fingerprints),
            }
        )
        outcome = self.outcomes[arm]
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, dict):
            result = _published_result(
                target_request=outcome.get(
                    "target_request",
                    target_request,
                ),
                evidence_fingerprint=outcome.get(
                    "evidence_fingerprint",
                    evidence_fingerprint,
                ),
                destination=Path(destination),
                quality=float(outcome.get("quality", 1.0)),
                certificate_changes=outcome.get("certificate_changes"),
                attempt_changes=outcome.get("attempt_changes"),
                run_changes=outcome.get("run_changes"),
                plan_changes=outcome.get("plan_changes"),
                method_id=outcome.get("method_id", "test.dynamic-fold"),
                duplicate_parent_graph=bool(
                    outcome.get("duplicate_parent_graph", False)
                ),
            )
            post_publication_error = outcome.get("post_publication_error")
            if post_publication_error is not None:
                raise post_publication_error
            return result
        if not isinstance(outcome, (int, float)):
            Path(destination).mkdir()
            return outcome
        return _published_result(
            target_request=target_request,
            evidence_fingerprint=evidence_fingerprint,
            destination=Path(destination),
            quality=float(outcome),
        )


class FakePortfolio:
    portfolio_id = "qwen-repair"
    version = 1
    priority = 10
    target_families = ("qwen3_codebook",)
    target_modalities = ("text",)

    def __init__(self, candidates):
        self.values = tuple(candidates)

    def supports(self, objective):
        return objective.target.target_family == "qwen3_codebook"

    def candidates(self, objective):
        del objective
        return self.values

    def classify_failure(self, candidate, error):
        del candidate
        if isinstance(error, CandidateRejected):
            return FailureDisposition.CANDIDATE_REJECTED
        if isinstance(error, InfrastructureFailure):
            return FailureDisposition.INFRASTRUCTURE
        return FailureDisposition.FATAL

    def score(self, candidate, result):
        del candidate
        return float(result.target.adapter_report["quality"])


class ExplodingScorePortfolio(FakePortfolio):
    def score(self, candidate, result):
        del candidate, result
        raise RuntimeError("scoring implementation failed")


class TamperingRunner(FakeRunner):
    def run(self, **kwargs):
        result = super().run(**kwargs)
        child = Path(kwargs["destination"]) / "child" / "model.bin"
        child.write_bytes(child.read_bytes() + b"-tampered")
        return result


class HiddenInferenceStateRunner(FakeRunner):
    def run(self, **kwargs):
        result = super().run(**kwargs)
        result.target.model.hidden_inference_tensor = torch.ones(1)
        return result


def _target(
    *,
    arm="base",
    exact=100,
    adapter_id=None,
    fold_method_id: str | None = None,
):
    return DynamicTargetRequest(
        child_name=f"child-{arm}",
        target_family="qwen3_codebook",
        target_modalities=("text",),
        exact_unique_parameters=exact,
        tokenizer_fingerprint="a" * 64,
        adapter_id=adapter_id,
        fold_method_id=fold_method_id,
        blank_seed=17,
        configuration=(("arm", arm),),
    )


def _candidate(
    arm,
    *,
    priority,
    adapter_id=None,
    method_id: str | None = None,
):
    metadata = [("repair_arm", arm)]
    if method_id is not None:
        metadata.append(("method_id", method_id))
    return DynamicFoldSearchCandidate(
        candidate_id=arm,
        target=_target(
            arm=arm,
            adapter_id=adapter_id,
            fold_method_id=method_id,
        ),
        evidence_fingerprint=hashlib.sha256(arm.encode()).hexdigest(),
        priority=priority,
        metadata=tuple(metadata),
    )


def _objective(*, stop_after_first_accept=True, adapter_id=None):
    return DynamicFoldSearchObjective(
        target=_target(adapter_id=adapter_id),
        maximum_attempts=8,
        stop_after_first_accept=stop_after_first_accept,
    )


def test_search_continues_after_candidate_rejection_and_stops_on_accept(
    tmp_path,
) -> None:
    runner = FakeRunner(
        {
            "alpha": CandidateRejected("behavior gate rejected alpha"),
            "beta": 0.8,
            "gamma": 0.9,
        }
    )
    portfolio = FakePortfolio(
        (
            _candidate("gamma", priority=10),
            _candidate("alpha", priority=30),
            _candidate("beta", priority=20),
        )
    )
    system = AutomaticDynamicFoldSearchSystem(
        runner,
        DynamicFoldPortfolioRegistry((portfolio,)),
    )

    result = system.run(
        parent_models={"parent": object()},
        objective=_objective(),
        destination_root=tmp_path / "search",
        parent_tokenizer_fingerprints={"parent": "b" * 64},
    )

    assert [value["arm"] for value in runner.calls] == ["alpha", "beta"]
    assert [attempt.status for attempt in result.attempts] == [
        "candidate_rejected",
        "accepted",
    ]
    assert result.complete is True
    assert result.selected_candidate_id == "beta"
    assert result.selected_result.target.request.child_name == "child-beta"
    assert result.attempts[1].artifact_retained is True
    assert len(result.fingerprint) == 64


def test_search_preserves_only_self_sealed_immutable_failure_diagnostic(
    tmp_path,
) -> None:
    diagnostic_body = {
        "schema": "neural-forge-synthetic-fold-failure-v1",
        "layer_result_issued": False,
        "promotion_claimed": False,
        "nested": {
            "behavior_gate": {
                "passed": False,
                "checks": ["validation_endpoint_relative_rmse"],
            }
        },
    }
    diagnostic = {
        **diagnostic_body,
        "report_sha256": canonical_sha256(diagnostic_body),
    }
    failure = SealedCandidateRejected("fold gate rejected", diagnostic)
    runner = FakeRunner({"alpha": failure})
    system = AutomaticDynamicFoldSearchSystem(
        runner,
        DynamicFoldPortfolioRegistry(
            (FakePortfolio((_candidate("alpha", priority=10),)),)
        ),
    )

    result = system.run(
        parent_models={"parent": object()},
        objective=_objective(),
        destination_root=tmp_path / "search",
    )

    attempt = result.attempts[0]
    assert attempt.status == "candidate_rejected"
    assert attempt.failure_diagnostic_sha256 == canonical_sha256(diagnostic)
    assert attempt.as_dict()["failure_diagnostic"] == diagnostic
    diagnostic["nested"]["behavior_gate"]["passed"] = True
    assert (
        attempt.as_dict()["failure_diagnostic"]["nested"]["behavior_gate"]["passed"]
        is False
    )
    with pytest.raises(TypeError):
        attempt.failure_diagnostic["nested"]["behavior_gate"]["passed"] = True


def test_search_drops_forged_success_claiming_failure_report(tmp_path) -> None:
    forged_body = {
        "schema": "neural-forge-synthetic-fold-failure-v1",
        "layer_result_issued": True,
        "promotion_claimed": True,
    }
    forged = {
        **forged_body,
        "report_sha256": canonical_sha256(forged_body),
    }
    failure = SealedCandidateRejected("forged diagnostic", forged)
    system = AutomaticDynamicFoldSearchSystem(
        FakeRunner({"alpha": failure}),
        DynamicFoldPortfolioRegistry(
            (FakePortfolio((_candidate("alpha", priority=10),)),)
        ),
    )

    result = system.run(
        parent_models={"parent": object()},
        objective=_objective(),
        destination_root=tmp_path / "search",
    )

    attempt = result.attempts[0]
    assert attempt.failure_diagnostic is None
    assert attempt.failure_diagnostic_sha256 is None


def test_exhaustive_search_selects_highest_scoring_accepted_candidate(
    tmp_path,
) -> None:
    runner = FakeRunner({"alpha": 0.4, "beta": 0.9, "gamma": 0.6})
    portfolio = FakePortfolio(
        (
            _candidate("alpha", priority=30),
            _candidate("beta", priority=20),
            _candidate("gamma", priority=10),
        )
    )
    system = AutomaticDynamicFoldSearchSystem(
        runner,
        DynamicFoldPortfolioRegistry((portfolio,)),
    )

    result = system.run(
        parent_models={"parent": object()},
        objective=_objective(stop_after_first_accept=False),
        destination_root=tmp_path / "search",
    )

    assert len(result.attempts) == 3
    assert all(attempt.status == "accepted" for attempt in result.attempts)
    assert result.selected_candidate_id == "beta"
    assert result.selected_result.target.adapter_report["quality"] == pytest.approx(0.9)
    assert [attempt.artifact_retained for attempt in result.attempts] == [
        False,
        True,
        False,
    ]
    assert sorted(path.name for path in (tmp_path / "search").iterdir()) == ["beta"]


def test_infrastructure_failure_stops_without_trying_later_candidates(
    tmp_path,
) -> None:
    runner = FakeRunner(
        {
            "alpha": InfrastructureFailure("GPU unavailable"),
            "beta": 0.9,
        }
    )
    portfolio = FakePortfolio(
        (
            _candidate("alpha", priority=20),
            _candidate("beta", priority=10),
        )
    )
    system = AutomaticDynamicFoldSearchSystem(
        runner,
        DynamicFoldPortfolioRegistry((portfolio,)),
    )

    result = system.run(
        parent_models={"parent": object()},
        objective=_objective(),
        destination_root=tmp_path / "search",
    )

    assert [value["arm"] for value in runner.calls] == ["alpha"]
    assert result.complete is False
    assert result.attempts[0].status == "infrastructure"
    assert result.attempts[0].error_type == "InfrastructureFailure"


def test_candidate_cannot_change_exact_objective_contract(tmp_path) -> None:
    invalid = DynamicFoldSearchCandidate(
        candidate_id="wrong-size",
        target=_target(arm="wrong-size", exact=99),
        evidence_fingerprint="c" * 64,
    )
    runner = FakeRunner({"wrong-size": 1.0})
    system = AutomaticDynamicFoldSearchSystem(
        runner,
        DynamicFoldPortfolioRegistry((FakePortfolio((invalid,)),)),
    )

    with pytest.raises(
        DynamicFoldSearchError,
        match="changes objective exact_unique_parameters",
    ):
        system.run(
            parent_models={"parent": object()},
            objective=_objective(),
            destination_root=tmp_path / "search",
        )
    assert runner.calls == []


def test_candidate_cannot_override_explicitly_pinned_adapter(tmp_path) -> None:
    runner = FakeRunner({"alpha": 1.0})
    candidate = _candidate(
        "alpha",
        priority=1,
        adapter_id="different-adapter",
    )
    system = AutomaticDynamicFoldSearchSystem(
        runner,
        DynamicFoldPortfolioRegistry((FakePortfolio((candidate,)),)),
    )

    with pytest.raises(
        DynamicFoldSearchError,
        match="changes objective adapter_id",
    ):
        system.run(
            parent_models={"parent": object()},
            objective=_objective(adapter_id="pinned-adapter"),
            destination_root=tmp_path / "search",
        )
    assert runner.calls == []


def test_complete_but_uncertified_result_is_fatal_not_accepted(tmp_path) -> None:
    runner = FakeRunner({"alpha": UncertifiedCompleteResult()})
    system = AutomaticDynamicFoldSearchSystem(
        runner,
        DynamicFoldPortfolioRegistry(
            (FakePortfolio((_candidate("alpha", priority=1),)),)
        ),
    )

    result = system.run(
        parent_models={"parent": object()},
        objective=_objective(),
        destination_root=tmp_path / "search",
    )

    assert result.complete is False
    assert result.attempts[0].status == "fatal"
    assert "could not be authenticated and safely removed" in (
        result.attempts[0].reason
    )
    assert (tmp_path / "search" / "alpha").is_dir()


def test_post_publication_scoring_failure_prunes_owned_artifact(tmp_path) -> None:
    runner = FakeRunner({"alpha": 1.0})
    portfolio = ExplodingScorePortfolio((_candidate("alpha", priority=1),))
    system = AutomaticDynamicFoldSearchSystem(
        runner,
        DynamicFoldPortfolioRegistry((portfolio,)),
    )
    root = tmp_path / "search"

    result = system.run(
        parent_models={"parent": object()},
        objective=_objective(),
        destination_root=root,
    )

    assert result.complete is False
    assert result.attempts[0].status == "fatal"
    assert result.attempts[0].error_type == "RuntimeError"
    assert list(root.iterdir()) == []


def test_retirement_keeps_artifact_when_pending_ledger_write_fails(tmp_path) -> None:
    class PrepareFailLedger(ExperimentLedger):
        def prepare_supersession(self, *args, **kwargs):
            del args, kwargs
            raise OSError("ledger unavailable")

    candidate = _candidate("alpha", priority=1)
    root = tmp_path / "search"
    root.mkdir()
    destination = root / candidate.candidate_id
    result = _published_result(
        target_request=candidate.target,
        evidence_fingerprint=candidate.evidence_fingerprint,
        destination=destination,
        quality=1.0,
    )
    signature = result.published.compilation.run.attempts[0].execution_signature
    accepted = LedgerRecord(
        signature=signature,
        closure_level=ClosureLevel.EXACT,
        status=RecordStatus.ACCEPTED,
        reason_class="gate",
        reason="provisionally accepted",
    )
    runner = FakeRunner({})
    runner.ledger = PrepareFailLedger((accepted,))
    system = AutomaticDynamicFoldSearchSystem(
        runner,
        DynamicFoldPortfolioRegistry((FakePortfolio((candidate,)),)),
    )
    owned = {destination.resolve(strict=True)}

    with pytest.raises(DynamicFoldSearchError, match="artifact was retained"):
        system._retire_owned_artifact(
            root=root,
            candidate=candidate,
            destination=destination,
            artifact=result.published.artifact,
            owned_destinations=owned,
            record_sha256s=(accepted.record_sha256,),
            reason="test rollback",
        )

    assert destination.is_dir()
    assert owned == {destination.resolve(strict=True)}
    assert runner.ledger.decision(signature).allowed is False


def test_retirement_finalization_failure_leaves_deleted_acceptance_blocked(
    tmp_path,
) -> None:
    class FinalizeFailLedger(ExperimentLedger):
        def supersede_records(self, *args, **kwargs):
            del args, kwargs
            raise OSError("final durable append failed")

    candidate = _candidate("alpha", priority=1)
    root = tmp_path / "search"
    root.mkdir()
    destination = root / candidate.candidate_id
    result = _published_result(
        target_request=candidate.target,
        evidence_fingerprint=candidate.evidence_fingerprint,
        destination=destination,
        quality=1.0,
    )
    signature = result.published.compilation.run.attempts[0].execution_signature
    accepted = LedgerRecord(
        signature=signature,
        closure_level=ClosureLevel.EXACT,
        status=RecordStatus.ACCEPTED,
        reason_class="gate",
        reason="provisionally accepted",
    )
    runner = FakeRunner({})
    runner.ledger = FinalizeFailLedger((accepted,))
    system = AutomaticDynamicFoldSearchSystem(
        runner,
        DynamicFoldPortfolioRegistry((FakePortfolio((candidate,)),)),
    )
    owned = {destination.resolve(strict=True)}

    with pytest.raises(DynamicFoldSearchError, match="safely rollback-pending"):
        system._retire_owned_artifact(
            root=root,
            candidate=candidate,
            destination=destination,
            artifact=result.published.artifact,
            owned_destinations=owned,
            record_sha256s=(accepted.record_sha256,),
            reason="test rollback",
        )

    assert not destination.exists()
    assert owned == set()
    assert runner.ledger.records[-1].status == (RecordStatus.SUPERSESSION_PENDING.value)
    assert runner.ledger.decision(signature).allowed is False


def test_search_rehashes_child_tree_before_promotion(tmp_path) -> None:
    runner = TamperingRunner({"alpha": 1.0})
    system = AutomaticDynamicFoldSearchSystem(
        runner,
        DynamicFoldPortfolioRegistry(
            (FakePortfolio((_candidate("alpha", priority=1),)),)
        ),
    )

    result = system.run(
        parent_models={"parent": object()},
        objective=_objective(),
        destination_root=tmp_path / "search",
    )

    assert result.complete is False
    assert result.attempts[0].status == "fatal"
    assert "could not be authenticated and safely removed" in (
        result.attempts[0].reason
    )
    assert (
        (tmp_path / "search" / "alpha" / "child" / "model.bin")
        .read_bytes()
        .endswith(b"-tampered")
    )


def test_search_rejects_training_attempt_recorded_by_fold_run(tmp_path) -> None:
    runner = FakeRunner(
        {
            "alpha": {
                "quality": 1.0,
                "run_changes": {"backward_calls": 1},
            }
        }
    )
    system = AutomaticDynamicFoldSearchSystem(
        runner,
        DynamicFoldPortfolioRegistry(
            (FakePortfolio((_candidate("alpha", priority=1),)),)
        ),
    )

    result = system.run(
        parent_models={"parent": object()},
        objective=_objective(),
        destination_root=tmp_path / "search",
    )

    assert result.complete is False
    assert result.attempts[0].status == "fatal"
    assert "training operation" in result.attempts[0].reason


@pytest.mark.parametrize(
    "run_changes",
    (
        {
            "observed_parent_module_forward_calls": 1,
            "observed_parent_root_forward_calls": 1,
        },
        {
            "observed_final_child_module_forward_calls": 1,
            "observed_final_child_root_forward_calls": 0,
        },
        {
            "observed_final_child_inference_entrypoint_calls": 0,
        },
    ),
)
def test_search_requires_zero_parent_and_observed_child_forwards(
    tmp_path,
    run_changes,
) -> None:
    runner = FakeRunner({"alpha": {"quality": 1.0, "run_changes": run_changes}})
    system = AutomaticDynamicFoldSearchSystem(
        runner,
        DynamicFoldPortfolioRegistry(
            (FakePortfolio((_candidate("alpha", priority=1),)),)
        ),
    )

    result = system.run(
        parent_models={"parent": object()},
        objective=_objective(),
        destination_root=tmp_path / "search",
    )

    assert result.complete is False
    assert result.attempts[0].status == "fatal"
    assert "zero observed parent forwards" in result.attempts[0].reason
    assert list((tmp_path / "search").iterdir()) == []


@pytest.mark.parametrize(
    "outcome_changes",
    (
        {
            "attempt_changes": {
                "evidence_payload_fingerprint": "0" * 64,
            }
        },
        {
            "attempt_changes": {
                "evidence_payload_integrity_reverified": False,
            }
        },
        {
            "run_changes": {
                "final_evidence_binding_fingerprint": "0" * 64,
            }
        },
        {
            "run_changes": {
                "final_evidence_payload_integrity_reverified": False,
            }
        },
    ),
)
def test_search_requires_compiler_owned_reverified_evidence_provenance(
    tmp_path,
    outcome_changes,
) -> None:
    runner = FakeRunner({"alpha": {"quality": 1.0, **outcome_changes}})
    system = AutomaticDynamicFoldSearchSystem(
        runner,
        DynamicFoldPortfolioRegistry(
            (FakePortfolio((_candidate("alpha", priority=1),)),)
        ),
    )

    result = system.run(
        parent_models={"parent": object()},
        objective=_objective(),
        destination_root=tmp_path / "search",
    )

    assert result.complete is False
    assert result.attempts[0].status == "fatal"
    assert "evidence" in result.attempts[0].reason
    assert list((tmp_path / "search").iterdir()) == []


@pytest.mark.parametrize(
    ("certificate_changes", "reason", "safely_removable"),
    (
        (
            {"certificate_sha256": "f" * 64},
            "could not be authenticated and safely removed",
            False,
        ),
        (
            {"exact_unique_parameters": 99},
            "misses the exact candidate budget",
            True,
        ),
        (
            {"child_graph_fingerprint": "e" * 64},
            "could not be authenticated and safely removed",
            False,
        ),
        (
            {"child_state_fingerprint": "d" * 64},
            "state identity differs from the accepted run",
            True,
        ),
    ),
)
def test_forged_or_mismatched_artifact_certificate_is_never_promoted(
    tmp_path,
    certificate_changes,
    reason,
    safely_removable,
) -> None:
    runner = FakeRunner(
        {
            "alpha": {
                "quality": 1.0,
                "certificate_changes": certificate_changes,
            }
        }
    )
    system = AutomaticDynamicFoldSearchSystem(
        runner,
        DynamicFoldPortfolioRegistry(
            (FakePortfolio((_candidate("alpha", priority=1),)),)
        ),
    )

    result = system.run(
        parent_models={"parent": object()},
        objective=_objective(),
        destination_root=tmp_path / "search",
    )

    assert result.complete is False
    assert result.attempts[0].status == "fatal"
    assert reason in result.attempts[0].reason
    candidate_path = tmp_path / "search" / "alpha"
    assert candidate_path.exists() is (not safely_removable)


def test_artifact_certificate_must_bind_exact_destination(tmp_path) -> None:
    wrong_destination = tmp_path / "someone-elses-artifact"
    wrong_destination.mkdir()
    runner = FakeRunner(
        {
            "alpha": {
                "quality": 1.0,
                "certificate_changes": {"path": str(wrong_destination)},
            }
        }
    )
    system = AutomaticDynamicFoldSearchSystem(
        runner,
        DynamicFoldPortfolioRegistry(
            (FakePortfolio((_candidate("alpha", priority=1),)),)
        ),
    )

    result = system.run(
        parent_models={"parent": object()},
        objective=_objective(),
        destination_root=tmp_path / "search",
    )

    assert result.complete is False
    assert result.attempts[0].status == "fatal"
    assert "could not be authenticated and safely removed" in (
        result.attempts[0].reason
    )
    assert wrong_destination.is_dir()
    assert (tmp_path / "search" / "alpha").is_dir()


def test_published_result_must_bind_exact_candidate_request(tmp_path) -> None:
    runner = FakeRunner(
        {
            "alpha": {
                "quality": 1.0,
                "target_request": _target(arm="not-alpha"),
            }
        }
    )
    system = AutomaticDynamicFoldSearchSystem(
        runner,
        DynamicFoldPortfolioRegistry(
            (FakePortfolio((_candidate("alpha", priority=1),)),)
        ),
    )

    result = system.run(
        parent_models={"parent": object()},
        objective=_objective(),
        destination_root=tmp_path / "search",
    )

    assert result.complete is False
    assert result.attempts[0].status == "fatal"
    assert "exact candidate" in result.attempts[0].reason


def test_mismatched_candidate_evidence_is_rejected_and_cleaned(tmp_path) -> None:
    runner = FakeRunner(
        {
            "alpha": {
                "quality": 1.0,
                "evidence_fingerprint": "e" * 64,
            }
        }
    )
    system = AutomaticDynamicFoldSearchSystem(
        runner,
        DynamicFoldPortfolioRegistry(
            (FakePortfolio((_candidate("alpha", priority=1),)),)
        ),
    )
    root = tmp_path / "search"

    result = system.run(
        parent_models={"parent": object()},
        objective=_objective(),
        destination_root=root,
    )

    assert result.complete is False
    assert result.attempts[0].status == "fatal"
    assert "evidence differs from the search candidate" in (result.attempts[0].reason)
    assert list(root.iterdir()) == []


def test_search_rejects_unregistered_live_target_inference_state(tmp_path) -> None:
    runner = HiddenInferenceStateRunner({"alpha": 1.0})
    system = AutomaticDynamicFoldSearchSystem(
        runner,
        DynamicFoldPortfolioRegistry(
            (FakePortfolio((_candidate("alpha", priority=1),)),)
        ),
    )
    root = tmp_path / "search"

    result = system.run(
        parent_models={"parent": object()},
        objective=_objective(),
        destination_root=root,
    )

    assert result.complete is False
    assert result.attempts[0].status == "fatal"
    assert "complete inference state" in result.attempts[0].reason
    assert list(root.iterdir()) == []


def test_runner_exception_after_publication_leaves_unowned_path(tmp_path) -> None:
    runner = FakeRunner(
        {
            "alpha": {
                "quality": 1.0,
                "post_publication_error": InfrastructureFailure(
                    "worker disappeared after publication"
                ),
            }
        }
    )
    system = AutomaticDynamicFoldSearchSystem(
        runner,
        DynamicFoldPortfolioRegistry(
            (FakePortfolio((_candidate("alpha", priority=1),)),)
        ),
    )
    root = tmp_path / "search"

    result = system.run(
        parent_models={"parent": object()},
        objective=_objective(),
        destination_root=root,
    )

    assert result.complete is False
    assert result.attempts[0].status == "infrastructure"
    assert result.attempts[0].error_type == "InfrastructureFailure"
    artifact_path = root / "alpha"
    artifact = load_dynamic_fold_artifact_certificate(artifact_path)
    assert Path(artifact.path).resolve(strict=True) == artifact_path.resolve(
        strict=True
    )


def test_runner_exception_does_not_adopt_concurrent_valid_artifact(
    tmp_path,
) -> None:
    class ConcurrentPublishingFailureRunner(FakeRunner):
        def run(self, **kwargs):
            target_request = kwargs["target_request"]
            destination = Path(kwargs["destination"])
            errors: list[BaseException] = []

            def publish_concurrently() -> None:
                try:
                    _published_result(
                        target_request=target_request,
                        evidence_fingerprint=kwargs["evidence_fingerprint"],
                        destination=destination,
                        quality=1.0,
                    )
                except BaseException as exc:  # pragma: no cover - assertion aid
                    errors.append(exc)

            publisher = Thread(target=publish_concurrently)
            publisher.start()
            publisher.join()
            assert errors == []
            raise InfrastructureFailure(
                "runner failed while another publisher won the destination"
            )

    runner = ConcurrentPublishingFailureRunner({})
    system = AutomaticDynamicFoldSearchSystem(
        runner,
        DynamicFoldPortfolioRegistry(
            (FakePortfolio((_candidate("alpha", priority=1),)),)
        ),
    )
    root = tmp_path / "search"

    result = system.run(
        parent_models={"parent": object()},
        objective=_objective(),
        destination_root=root,
    )

    assert result.complete is False
    assert result.attempts[0].status == "infrastructure"
    assert result.attempts[0].error_type == "InfrastructureFailure"
    artifact_path = root / "alpha"
    artifact = load_dynamic_fold_artifact_certificate(artifact_path)
    assert Path(artifact.path).resolve(strict=True) == artifact_path.resolve(
        strict=True
    )


def test_exhaustive_search_prunes_provisional_accept_on_later_infrastructure(
    tmp_path,
) -> None:
    runner = FakeRunner(
        {
            "alpha": 0.8,
            "beta": InfrastructureFailure("GPU failed during exhaustive search"),
        }
    )
    portfolio = FakePortfolio(
        (
            _candidate("alpha", priority=20),
            _candidate("beta", priority=10),
        )
    )
    system = AutomaticDynamicFoldSearchSystem(
        runner,
        DynamicFoldPortfolioRegistry((portfolio,)),
    )
    root = tmp_path / "search"

    result = system.run(
        parent_models={"parent": object()},
        objective=_objective(stop_after_first_accept=False),
        destination_root=root,
    )

    assert result.complete is False
    assert [attempt.status for attempt in result.attempts] == [
        "accepted",
        "infrastructure",
    ]
    assert result.attempts[0].artifact_retained is False
    assert "prevented exhaustive completion" in result.attempts[0].reason
    assert list(root.iterdir()) == []


def test_catalog_method_mismatch_is_rejected_and_cleaned(tmp_path) -> None:
    candidate = _candidate(
        "alpha",
        priority=1,
        method_id="catalog.residual-attention",
    )
    runner = FakeRunner(
        {
            "alpha": {
                "quality": 1.0,
                "method_id": "runner.unregistered-fold",
            }
        }
    )
    system = AutomaticDynamicFoldSearchSystem(
        runner,
        DynamicFoldPortfolioRegistry((FakePortfolio((candidate,)),)),
    )
    root = tmp_path / "search"

    result = system.run(
        parent_models={"parent": object()},
        objective=_objective(),
        destination_root=root,
    )

    assert result.complete is False
    assert result.attempts[0].status == "fatal"
    assert "fold method differs from the catalog candidate" in (
        result.attempts[0].reason
    )
    assert list(root.iterdir()) == []


@pytest.mark.parametrize(
    ("outcome_changes", "reason"),
    (
        (
            {"plan_changes": {"actions": ()}},
            "lacks a nonempty ready fold plan",
        ),
        (
            {"run_changes": {"final_state_fingerprint": "3" * 64}},
            "does not prove a changed pre-fold blank state",
        ),
    ),
)
def test_zero_action_or_unchanged_blank_result_is_rejected_and_cleaned(
    tmp_path,
    outcome_changes,
    reason,
) -> None:
    runner = FakeRunner(
        {
            "alpha": {
                "quality": 1.0,
                **outcome_changes,
            }
        }
    )
    system = AutomaticDynamicFoldSearchSystem(
        runner,
        DynamicFoldPortfolioRegistry(
            (FakePortfolio((_candidate("alpha", priority=1),)),)
        ),
    )
    root = tmp_path / "search"

    result = system.run(
        parent_models={"parent": object()},
        objective=_objective(),
        destination_root=root,
    )

    assert result.complete is False
    assert result.attempts[0].status == "fatal"
    assert reason in result.attempts[0].reason
    assert list(root.iterdir()) == []


def test_duplicate_parent_graph_names_fail_before_certification(tmp_path) -> None:
    runner = FakeRunner(
        {
            "alpha": {
                "quality": 1.0,
                "duplicate_parent_graph": True,
            }
        }
    )
    system = AutomaticDynamicFoldSearchSystem(
        runner,
        DynamicFoldPortfolioRegistry(
            (FakePortfolio((_candidate("alpha", priority=1),)),)
        ),
    )
    root = tmp_path / "search"

    result = system.run(
        parent_models={"parent": object()},
        objective=_objective(),
        destination_root=root,
    )

    assert result.complete is False
    assert result.attempts[0].status == "fatal"
    assert "parent graph names must be unique" in result.attempts[0].reason
    assert [path.name for path in root.iterdir()] == ["alpha"]
    assert not (root / "alpha" / "certificate.json").exists()


@pytest.mark.parametrize(
    ("certificate_changes", "reason"),
    (
        (
            {"fresh_reload_decision": FoldDecision.reject("fake reload")},
            "lacks an accepted fresh-reload behavior decision",
        ),
        (
            {"preparation_guard_applied": False},
            "lacks a versioned, sealed behavior and full guard contract",
        ),
        (
            {"publication_guard_covers_save_load": False},
            "lacks a versioned, sealed behavior and full guard contract",
        ),
        (
            {"behavior_gate_id": "unregistered-behavior-gate"},
            "lacks a versioned, sealed behavior and full guard contract",
        ),
        (
            {"behavior_gate_version": 0},
            "lacks a versioned, sealed behavior and full guard contract",
        ),
        (
            {"behavior_baseline_fingerprint": "0" * 64},
            "lacks a versioned, sealed behavior and full guard contract",
        ),
    ),
)
def test_fake_reload_or_incomplete_certificate_contract_is_rejected_and_cleaned(
    tmp_path,
    certificate_changes,
    reason,
) -> None:
    runner = FakeRunner(
        {
            "alpha": {
                "quality": 1.0,
                "certificate_changes": certificate_changes,
            }
        }
    )
    system = AutomaticDynamicFoldSearchSystem(
        runner,
        DynamicFoldPortfolioRegistry(
            (FakePortfolio((_candidate("alpha", priority=1),)),)
        ),
    )
    root = tmp_path / "search"

    result = system.run(
        parent_models={"parent": object()},
        objective=_objective(),
        destination_root=root,
    )

    assert result.complete is False
    assert result.attempts[0].status == "fatal"
    assert reason in result.attempts[0].reason
    assert list(root.iterdir()) == []


def test_registry_rejects_duplicate_portfolios_and_unsafe_candidate_ids() -> None:
    portfolio = FakePortfolio((_candidate("alpha", priority=1),))
    registry = DynamicFoldPortfolioRegistry((portfolio,))
    with pytest.raises(DynamicFoldSearchError, match="already registered"):
        registry.register(portfolio)
    with pytest.raises(DynamicFoldSearchError, match="safe lowercase"):
        DynamicFoldSearchCandidate(
            candidate_id="../escape",
            target=_target(),
            evidence_fingerprint="d" * 64,
        )
