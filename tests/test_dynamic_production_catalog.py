from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from neural_forge.dynamic_circuit_ir import (
    CircuitKind,
    CircuitNode,
    ModelCircuitGraph,
    canonical_sha256,
)
from neural_forge.dynamic_fold_artifact import (
    DynamicFoldArtifactCertificate,
    DynamicFoldArtifactError,
)
from neural_forge.dynamic_fold_compiler import DynamicFoldCompilerError
from neural_forge.dynamic_fold_executor import FoldDecision
from neural_forge.dynamic_fold_registry import ExperimentLedger
from neural_forge.dynamic_model_inspector import InspectorRegistry
from neural_forge.dynamic_production_catalog import (
    CertificateScoreTerm,
    ProductionCandidateRejected,
    ProductionDynamicFoldCatalog,
    ProductionDynamicFoldCatalogError,
    ProductionFoldBundle,
    ProductionFoldCandidateSpec,
    ProductionFoldCapability,
    RegisteredExactTargetPortfolio,
)
from neural_forge.dynamic_search import (
    AutomaticDynamicFoldSearchSystem,
    DynamicFoldSearchObjective,
    FailureDisposition,
)
from neural_forge.dynamic_target_synthesis import DynamicTargetRequest


def test_production_catalog_is_exported_from_package_root() -> None:
    import neural_forge

    assert neural_forge.ProductionDynamicFoldCatalog is (ProductionDynamicFoldCatalog)
    assert neural_forge.DynamicTargetRequest is DynamicTargetRequest


class _TargetAdapter:
    adapter_id = "test.production.target"
    version = 1
    priority = 100
    target_families = ("lfm2",)
    target_modalities = ("text",)

    def supports(self, _request, _parent_graphs):
        return True

    def materialize(self, *_args, **_kwargs):
        raise AssertionError("The assembly test must not materialize a model.")


class _StackAdapter:
    adapter_id = "test.production.stack"
    version = 1
    priority = 100
    target_families = ("lfm2",)
    target_modalities = ("text",)
    method_ids = ("method.alpha", "method.beta", "method.gamma")

    def supports(self, _target):
        return True

    def build(self, _target):
        raise AssertionError("The assembly test must not build a fold stack.")


class _FamilyModel:
    def __init__(self, family: str) -> None:
        self.family = family


class _FamilyInspector:
    name = "test.production.family-inspector"
    priority = 1000

    def supports(self, model):
        return isinstance(model, _FamilyModel)

    def inspect(self, model, *, model_name, tokenizer_fingerprint=None):
        node = CircuitNode(
            node_id="model.unit",
            kind=CircuitKind.NORM,
            order=0,
            path="model",
            layer_index=None,
            inputs=(),
            outputs=(),
        )
        return ModelCircuitGraph(
            model_name=model_name,
            family=model.family,
            model_class="FamilyModel",
            nodes=(node,),
            edges=(),
            interfaces=(),
            logical_parameter_count=0,
            unique_parameter_count=0,
            unique_parameter_bytes=0,
            config_fingerprint=canonical_sha256({"family": model.family}),
            tokenizer_fingerprint=tokenizer_fingerprint,
        )


def _capability(
    *,
    capability_id: str = "lfm2-text-90-110",
    minimum: int = 90,
    maximum: int = 110,
) -> ProductionFoldCapability:
    return ProductionFoldCapability(
        capability_id=capability_id,
        source_families=("lfm2",),
        target_family="lfm2",
        target_modalities=("text",),
        minimum_target_unique_parameters=minimum,
        maximum_target_unique_parameters=maximum,
    )


def _spec(
    candidate_id: str,
    *,
    priority: int,
    method_id: str | None = None,
    configuration=(),
) -> ProductionFoldCandidateSpec:
    return ProductionFoldCandidateSpec(
        candidate_id=candidate_id,
        method_id=method_id or f"method.{candidate_id}",
        evidence_fingerprint=canonical_sha256({"candidate": candidate_id}),
        priority=priority,
        configuration=configuration,
    )


def _bundle(
    *,
    capability: ProductionFoldCapability | None = None,
    maximum_attempts: int = 2,
    score_terms=(),
    candidate_rejection_exceptions=(ProductionCandidateRejected,),
) -> ProductionFoldBundle:
    return ProductionFoldBundle(
        capability=capability or _capability(),
        target_adapters=(_TargetAdapter(),),
        stack_adapters=(_StackAdapter(),),
        candidate_specs=(
            _spec("gamma", priority=10),
            _spec(
                "alpha",
                priority=30,
                configuration=(("fold_variant", "residual_attention"),),
            ),
            _spec(
                "beta",
                priority=20,
                configuration=(("fold_variant", "prototype_repair"),),
            ),
        ),
        maximum_attempts=maximum_attempts,
        score_terms=score_terms,
        candidate_rejection_exceptions=candidate_rejection_exceptions,
    )


def _objective(
    *,
    family: str = "lfm2",
    exact: int = 100,
    maximum_attempts: int = 8,
) -> DynamicFoldSearchObjective:
    return DynamicFoldSearchObjective(
        target=DynamicTargetRequest(
            child_name="production-child",
            target_family=family,
            target_modalities=("text",),
            exact_unique_parameters=exact,
            tokenizer_fingerprint="a" * 64,
            blank_seed=7,
            configuration=(("source_parent", "parent"),),
        ),
        maximum_attempts=maximum_attempts,
        stop_after_first_accept=False,
    )


def _certificate(
    *,
    decision: FoldDecision,
) -> DynamicFoldArtifactCertificate:
    provisional = DynamicFoldArtifactCertificate(
        path="C:/folded/candidate",
        plan_fingerprint="1" * 64,
        run_fingerprint="2" * 64,
        child_graph_fingerprint="3" * 64,
        child_state_fingerprint="4" * 64,
        exact_unique_parameters=100,
        artifact_tree_sha256="5" * 64,
        artifact_file_count=2,
        artifact_total_bytes=1024,
        fresh_reload_decision=decision,
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
    )
    return replace(
        provisional,
        certificate_sha256=canonical_sha256(provisional.body()),
    )


def test_registered_portfolio_generates_bounded_deterministic_exact_targets() -> None:
    bundle = _bundle(maximum_attempts=2)
    portfolio = RegisteredExactTargetPortfolio(bundle)

    candidates = portfolio.candidates(_objective(maximum_attempts=99))

    assert [value.candidate_id for value in candidates] == ["alpha", "beta"]
    assert all(value.target.exact_unique_parameters == 100 for value in candidates)
    assert all(
        value.target.target_family == "lfm2"
        and value.target.target_modalities == ("text",)
        for value in candidates
    )
    assert all(
        value.target.adapter_id == _TargetAdapter.adapter_id for value in candidates
    )
    assert [value.target.fold_method_id for value in candidates] == [
        "method.alpha",
        "method.beta",
    ]
    assert dict(candidates[0].target.configuration) == {
        "fold_variant": "residual_attention",
        "source_parent": "parent",
    }
    assert dict(candidates[0].metadata) == {
        "capability_id": "lfm2-text-90-110",
        "method_id": "method.alpha",
    }
    assert candidates[0].target.child_name == "production-child--alpha"


def test_default_scoring_uses_only_authenticated_fresh_reload_metrics() -> None:
    bundle = _bundle(
        score_terms=(
            CertificateScoreTerm("intelligence_retention", 2.0),
            CertificateScoreTerm("repetition_regressions", -0.5),
        )
    )
    portfolio = RegisteredExactTargetPortfolio(bundle)
    candidate = portfolio.candidates(_objective())[0]
    artifact = _certificate(
        decision=FoldDecision.accept(
            "fresh reload passed",
            {
                "intelligence_retention": 0.9,
                "repetition_regressions": 2,
            },
        )
    )
    result = SimpleNamespace(
        published=SimpleNamespace(artifact=artifact),
        target=SimpleNamespace(adapter_report={"untrusted_quality": 9999}),
    )

    assert portfolio.score(candidate, result) == pytest.approx(1.8)

    forged = replace(artifact, certificate_sha256="f" * 64)
    with pytest.raises(
        ProductionDynamicFoldCatalogError,
        match="unauthenticated certificate",
    ):
        portfolio.score(
            candidate,
            SimpleNamespace(published=SimpleNamespace(artifact=forged)),
        )


def test_failure_classification_is_explicit_and_fails_closed() -> None:
    portfolio = RegisteredExactTargetPortfolio(_bundle())
    candidate = portfolio.candidates(_objective())[0]

    assert (
        portfolio.classify_failure(
            candidate,
            ProductionCandidateRejected("behavior rejected"),
        )
        is FailureDisposition.CANDIDATE_REJECTED
    )
    assert (
        portfolio.classify_failure(
            candidate,
            DynamicFoldArtifactError(
                "Freshly reloaded child failed its behavioral gate: one task lost"
            ),
        )
        is FailureDisposition.CANDIDATE_REJECTED
    )
    assert (
        portfolio.classify_failure(
            candidate,
            DynamicFoldArtifactError("Fresh reload state differs"),
        )
        is FailureDisposition.FATAL
    )
    assert (
        portfolio.classify_failure(candidate, TimeoutError("worker timeout"))
        is FailureDisposition.INFRASTRUCTURE
    )
    assert (
        portfolio.classify_failure(candidate, ValueError("programming defect"))
        is FailureDisposition.FATAL
    )
    assert (
        portfolio.classify_failure(
            candidate,
            DynamicFoldCompilerError(
                "Dynamic folding did not produce a behavior-accepted complete "
                "compilation; no artifact was published."
            ),
        )
        is FailureDisposition.CANDIDATE_REJECTED
    )
    assert (
        portfolio.classify_failure(
            candidate,
            DynamicFoldCompilerError("unknown compiler failure"),
        )
        is FailureDisposition.FATAL
    )

    with pytest.raises(
        ProductionDynamicFoldCatalogError,
        match="unsafe broad",
    ):
        _bundle(candidate_rejection_exceptions=(RuntimeError,))


def test_catalog_assembles_once_and_facade_delegates_neutral_request(
    monkeypatch,
    tmp_path,
) -> None:
    catalog = ProductionDynamicFoldCatalog(
        (_bundle(),),
        inspectors=InspectorRegistry((_FamilyInspector(),)),
        allow_ephemeral_ledger=True,
        require_positive_compaction=False,
    )
    facade = catalog.assemble()

    assert catalog.sealed is True
    assert catalog.assemble() is facade
    assert isinstance(facade.search_system, AutomaticDynamicFoldSearchSystem)
    assert isinstance(
        facade.search_system.folding_system.ledger,
        ExperimentLedger,
    )
    assert facade.search_system.folding_system.ledger is catalog._ledger
    assert facade.capabilities == (_capability(),)
    assert len(facade.fingerprint) == 64
    with pytest.raises(ProductionDynamicFoldCatalogError, match="sealed"):
        catalog.register(_bundle())

    calls = []
    sentinel = SimpleNamespace(complete=True)

    def _record_run(
        _self,
        *,
        parent_models,
        objective,
        destination_root,
        parent_tokenizer_fingerprints=None,
    ):
        calls.append(
            (
                tuple(parent_models),
                objective,
                destination_root,
                parent_tokenizer_fingerprints,
            )
        )
        return sentinel

    monkeypatch.setattr(AutomaticDynamicFoldSearchSystem, "run", _record_run)
    objective = _objective()
    destination = tmp_path / "search"
    result = facade.run(
        parent_models={"parent": _FamilyModel("lfm2")},
        objective=objective,
        destination_root=destination,
        parent_tokenizer_fingerprints={"parent": "b" * 64},
    )

    assert result is sentinel
    assert calls == [
        (
            ("parent",),
            objective,
            destination,
            {"parent": "b" * 64},
        )
    ]


def test_facade_rejects_unknown_target_range_and_source_family(tmp_path) -> None:
    facade = ProductionDynamicFoldCatalog(
        (_bundle(),),
        inspectors=InspectorRegistry((_FamilyInspector(),)),
        allow_ephemeral_ledger=True,
    ).assemble()

    with pytest.raises(
        ProductionDynamicFoldCatalogError,
        match="No single registered production capability",
    ):
        facade.run(
            parent_models={"parent": _FamilyModel("lfm2")},
            objective=_objective(family="unknown"),
            destination_root=tmp_path / "unknown-family",
        )
    with pytest.raises(
        ProductionDynamicFoldCatalogError,
        match="No single registered production capability",
    ):
        facade.run(
            parent_models={"parent": _FamilyModel("lfm2")},
            objective=_objective(exact=111),
            destination_root=tmp_path / "unknown-size",
        )
    with pytest.raises(
        ProductionDynamicFoldCatalogError,
        match="outside the registered source capability",
    ):
        facade.run(
            parent_models={"parent": _FamilyModel("qwen3")},
            objective=_objective(),
            destination_root=tmp_path / "unsupported-source",
        )


def test_catalog_refuses_ambiguous_overlapping_target_capabilities() -> None:
    first = _bundle()
    overlapping = _bundle(
        capability=_capability(
            capability_id="lfm2-text-100-120",
            minimum=100,
            maximum=120,
        )
    )
    catalog = ProductionDynamicFoldCatalog(
        (first,),
        allow_ephemeral_ledger=True,
    )

    with pytest.raises(
        ProductionDynamicFoldCatalogError,
        match="overlap ambiguously",
    ):
        catalog.register(overlapping)


def test_bundle_fingerprint_binds_adapter_versions() -> None:
    class _TargetV2(_TargetAdapter):
        version = 2

    class _StackV2(_StackAdapter):
        version = 2

    original = _bundle()
    revised = ProductionFoldBundle(
        capability=original.capability,
        target_adapters=(_TargetV2(),),
        stack_adapters=(_StackV2(),),
        candidate_specs=original.candidate_specs,
        maximum_attempts=original.maximum_attempts,
        portfolio_id=original.portfolio_id,
        version=original.version,
        priority=original.priority,
    )

    assert revised.fingerprint != original.fingerprint
