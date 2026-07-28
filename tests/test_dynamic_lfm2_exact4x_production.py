from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import neural_forge.dynamic_lfm2_exact4x_production as production
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_lfm2_exact4x_semantic_backend import (
    LFM2_EXACT4X_SEMANTIC_RESOLVER_ID,
    Lfm2Exact4xSemanticBackendConfig,
)
from neural_forge.dynamic_parameter_frontier import DenseParameterBudgetRequest
from neural_forge.dynamic_target_synthesis import DynamicTargetRequest
from neural_forge.lfm2_exact4x_contract import (
    EXPECTED_CHILD_TOPOLOGY,
    EXPECTED_CHILD_UNIQUE_PARAMETERS,
    EXPECTED_PARENT_UNIQUE_PARAMETERS,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _backend(
    tmp_path: Path,
    *,
    model_device: str = "cpu",
    child_dtype: str = "float32",
) -> Lfm2Exact4xSemanticBackendConfig:
    directories = {
        name: tmp_path / name
        for name in (
            "parent",
            "donor-tokenizer",
            "target-tokenizer",
            "bridge",
            "compiler-cache",
            "semantic-cache",
            "selective-catalog-store",
        )
    }
    for path in directories.values():
        path.mkdir()
    files = {
        "trajectory": tmp_path / "trajectory.json",
        "selective_receipt": tmp_path / "selective-receipt.json",
        "prompts": tmp_path / "prompts.json",
        "parent_envelope": tmp_path / "parent-envelope.json",
    }
    for index, path in enumerate(files.values()):
        path.write_text(json.dumps({"index": index}), encoding="utf-8")
    return Lfm2Exact4xSemanticBackendConfig(
        parent_model_directory=directories["parent"],
        donor_tokenizer_directory=directories["donor-tokenizer"],
        target_tokenizer_directory=directories["target-tokenizer"],
        bridge_directory=directories["bridge"],
        expected_bridge_artifact_sha256="1" * 64,
        compiler_cache_directory=directories["compiler-cache"],
        expected_compiler_cache_key="2" * 64,
        semantic_cache_directory=directories["semantic-cache"],
        expected_semantic_cache_key="3" * 64,
        trajectory_corpus=files["trajectory"],
        selective_catalog_store_directory=directories[
            "selective-catalog-store"
        ],
        selective_catalog_receipt_json=files["selective_receipt"],
        expected_selective_catalog_receipt_sha256="4" * 64,
        expected_parent_identity_sha256="5" * 64,
        expected_parent_unique_parameters=EXPECTED_PARENT_UNIQUE_PARAMETERS,
        expected_parent_graph_fingerprint="6" * 64,
        source_parent_name="parent",
        prompts_json=files["prompts"],
        expected_prompts_file_sha256=_sha256(files["prompts"]),
        parent_envelope=files["parent_envelope"],
        expected_parent_envelope_sha256="7" * 64,
        expected_parent_envelope_file_sha256=_sha256(
            files["parent_envelope"]
        ),
        body_recipe="test-complete-body-recipe",
        expected_body_recipe_sha256="8" * 64,
        short_conv_construction=(
            "body-global-coordinate-complete-structural-v1"
        ),
        model_device=model_device,
        fold_compute_device="cpu",
        analytic_solve_device="cpu",
        global_residual_backend="cpu",
        global_residual_basis_rank=128,
        correction_scale=1.0,
        target_dtype_representability_level="requested",
        child_dtype=child_dtype,
        maximum_prompt_tokens=128,
        maximum_new_tokens=64,
        compile_artifact_root=tmp_path / "compile-artifacts",
    )


def _request(
    tmp_path: Path,
    *,
    target_unique_parameters: int = EXPECTED_CHILD_UNIQUE_PARAMETERS,
    match_policy: str = "exact",
) -> production.NeutralFullModelFoldRequest:
    return production.NeutralFullModelFoldRequest(
        request_id="whole-model-exact4x",
        parent_name="parent",
        child_name="folded-child",
        budget=DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            target_unique_parameters=target_unique_parameters,
            target_vocabulary_size=EXPECTED_CHILD_TOPOLOGY["vocab_size"],
            match_policy=match_policy,
            maximum_candidates=32,
        ),
        destination=tmp_path / "published-child",
        blank_seed=0,
    )


def test_sealed_config_loads_typed_neutral_request_and_backend(tmp_path) -> None:
    backend = _backend(tmp_path)
    request = _request(tmp_path)
    document = production.seal_lfm2_exact4x_production_document(
        {
            "schema": production.SCHEMA,
            "request": request.as_dict(),
            "backend": backend.as_dict(),
        }
    )
    config = tmp_path / "production.json"
    config.write_text(json.dumps(document), encoding="utf-8")

    loaded = production.load_lfm2_exact4x_production_registration(config)

    assert loaded.configuration_sha256 == document["configuration_sha256"]
    assert loaded.request == request
    assert loaded.backend.fingerprint == backend.fingerprint
    assert loaded.backend.as_dict()["network_allowed"] is False


def test_sealed_config_rejects_request_drift(tmp_path) -> None:
    document = production.seal_lfm2_exact4x_production_document(
        {
            "schema": production.SCHEMA,
            "request": _request(tmp_path).as_dict(),
            "backend": _backend(tmp_path).as_dict(),
        }
    )
    document["request"]["child_name"] = "changed-after-seal"
    config = tmp_path / "drifted.json"
    config.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(
        production.Lfm2Exact4xProductionError,
        match="seal does not match",
    ):
        production.load_lfm2_exact4x_production_registration(config)


class _FakeInspectors:
    def __init__(self) -> None:
        self.calls = []

    def inspect(self, model, **kwargs):
        self.calls.append((model, kwargs))
        return "inspected-parent-graph"


class _FakeAutomaticSystem:
    def __init__(self) -> None:
        self.inspectors = _FakeInspectors()
        self.plan_calls = []
        self.run_calls = []
        self.materialized_target = object()
        self.materialized_binding = canonical_sha256(
            {"binding": "materialized-target"}
        )

    class _TargetCertificate:
        def __init__(self, expected_target, binding) -> None:
            self.expected_target = expected_target
            self.binding = binding
            self.verify_calls = []

        def verify_materialized_target(self, target):
            self.verify_calls.append(target)
            if target is not self.expected_target:
                raise RuntimeError("materialized target differs from plan")
            return self.binding

    def plan_budgeted_target(self, **kwargs):
        self.plan_calls.append(kwargs)
        budget = kwargs["budget_request"]
        target = DynamicTargetRequest(
            child_name=kwargs["child_name"],
            target_family=budget.target_family,
            target_modalities=budget.target_modalities,
            exact_unique_parameters=budget.target_unique_parameters,
            tokenizer_fingerprint=kwargs["tokenizer_fingerprint"],
            adapter_id=kwargs["adapter_id"],
            fold_method_id=kwargs["fold_method_id"],
            blank_seed=kwargs["blank_seed"],
        )
        certificate = self._TargetCertificate(
            self.materialized_target,
            self.materialized_binding,
        )
        return SimpleNamespace(
            target_request=target,
            fingerprint=canonical_sha256(target.as_dict()),
            target_certificate=certificate,
        )

    def run(self, **kwargs):
        self.run_calls.append(kwargs)
        return SimpleNamespace(
            marker="complete-system-result",
            target=self.materialized_target,
        )


@pytest.mark.parametrize("target_family", ("family-alpha", "family-beta"))
def test_neutral_entrypoint_routes_without_family_dispatch(
    tmp_path,
    target_family,
) -> None:
    system = _FakeAutomaticSystem()
    parent = object()
    entrypoint = production.ConfiguredFullModelFoldEntrypoint(
        system=system,
        parent_sources=(
            production.RegisteredParentSource(
                name="source",
                tokenizer_fingerprint="a" * 64,
                loader=lambda: parent,
            ),
        ),
        budget_parent_name="source",
        child_tokenizer_fingerprint="b" * 64,
        evidence_fingerprint="c" * 64,
        target_adapter_id="configured.target.adapter",
        fold_method_id="configured.whole.model.fold",
    )
    request = production.NeutralFullModelFoldRequest(
        request_id=f"neutral-{target_family}",
        parent_name="source",
        child_name="complete-child",
        budget=DenseParameterBudgetRequest(
            target_family=target_family,
            target_modalities=("text",),
            target_unique_parameters=10,
        ),
        destination=tmp_path / target_family,
    )

    execution = entrypoint.run(request)

    assert execution.result.marker == "complete-system-result"
    assert execution.materialized_target_binding_sha256 == (
        system.materialized_binding
    )
    assert execution.target_plan.target_certificate.verify_calls == [
        system.materialized_target
    ]
    assert len(system.plan_calls) == 1
    assert system.plan_calls[0]["budget_request"] is request.budget
    assert system.plan_calls[0]["parent_graph"] == "inspected-parent-graph"
    assert len(system.run_calls) == 1
    assert system.run_calls[0]["parent_models"] == {"source": parent}
    assert (
        system.run_calls[0]["target_request"].target_family
        == target_family
    )
    assert system.run_calls[0]["destination"] == request.destination
    run_source = inspect.getsource(
        production.ConfiguredFullModelFoldEntrypoint.run
    )
    assert '"lfm2"' not in run_source.lower()
    assert "'lfm2'" not in run_source.lower()
    assert '"qwen"' not in run_source.lower()
    assert "'qwen'" not in run_source.lower()
    assert "layer" not in run_source.lower()


def test_neutral_entrypoint_fails_closed_when_materialized_target_changed(
    tmp_path,
) -> None:
    system = _FakeAutomaticSystem()
    expected_target = system.materialized_target

    def return_changed_target(**kwargs):
        system.run_calls.append(kwargs)
        return SimpleNamespace(target=object())

    system.run = return_changed_target
    entrypoint = production.ConfiguredFullModelFoldEntrypoint(
        system=system,
        parent_sources=(
            production.RegisteredParentSource(
                name="source",
                tokenizer_fingerprint="a" * 64,
                loader=object,
            ),
        ),
        budget_parent_name="source",
        child_tokenizer_fingerprint="b" * 64,
        evidence_fingerprint="c" * 64,
    )
    request = production.NeutralFullModelFoldRequest(
        request_id="tampered-materialization",
        parent_name="source",
        child_name="complete-child",
        budget=DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            target_unique_parameters=10,
        ),
        destination=tmp_path / "tampered",
    )

    with pytest.raises(RuntimeError, match="differs from plan"):
        entrypoint.run(request)

    assert system.materialized_target is expected_target


def test_execution_summary_binds_component_plan_to_materialized_target(
    tmp_path,
) -> None:
    request = _request(tmp_path)
    component_receipt_sha256 = canonical_sha256(
        {"receipt": "component-allocation"}
    )
    topology_certificate_sha256 = canonical_sha256(
        {"certificate": "target-topology"}
    )
    materialized_binding_sha256 = canonical_sha256(
        {"binding": "planned-to-materialized"}
    )
    plan = SimpleNamespace(
        fingerprint=canonical_sha256({"plan": "target"}),
        target_request=SimpleNamespace(
            fingerprint=canonical_sha256({"request": "target"})
        ),
        target_certificate=SimpleNamespace(
            fingerprint=topology_certificate_sha256,
            selection_policy=(
                "architecture_discovered_component_risk_executable_v1"
            ),
            component_allocation_receipt=SimpleNamespace(
                fingerprint=component_receipt_sha256
            ),
        ),
    )
    artifact = SimpleNamespace(
        path=str(tmp_path / "published-child"),
        certificate_sha256=canonical_sha256({"artifact": "child"}),
        exact_unique_parameters=EXPECTED_CHILD_UNIQUE_PARAMETERS,
        fresh_reload_decision=SimpleNamespace(accepted=True),
    )
    result = SimpleNamespace(
        complete=True,
        published=SimpleNamespace(artifact=artifact),
    )

    summary = production.FullModelFoldExecution(
        request=request,
        target_plan=plan,
        result=result,
        materialized_target_binding_sha256=materialized_binding_sha256,
    ).summary()

    assert summary["target_topology_certificate_sha256"] == (
        topology_certificate_sha256
    )
    assert summary["component_allocation_receipt_sha256"] == (
        component_receipt_sha256
    )
    assert summary["materialized_target_binding_sha256"] == (
        materialized_binding_sha256
    )
    assert summary["target_selection_policy"] == (
        "architecture_discovered_component_risk_executable_v1"
    )


class _DummyTargetAdapter:
    adapter_id = "test.registered.target"
    version = 1
    priority = 100
    target_families = ("lfm2",)
    target_modalities = ("text",)

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        type(self).last_kwargs = kwargs

    def supports(self, _request, _parent_graphs) -> bool:
        return True

    def materialize(self, *_args, **_kwargs):
        raise AssertionError("assembly must not materialize a target")


class _DummyCatalog:
    catalog_id = "test.registered.catalog"
    version = 1
    priority = 100

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def supports(self, *_args, **_kwargs) -> bool:
        return True

    def enumerate_frontier(self, *_args, **_kwargs):
        raise AssertionError("assembly must not enumerate a target")


@pytest.mark.parametrize(
    ("target_unique_parameters", "match_policy"),
    (
        (EXPECTED_CHILD_UNIQUE_PARAMETERS, "exact"),
        (57_422_368, "exact"),
        (EXPECTED_PARENT_UNIQUE_PARAMETERS // 4, "at_or_below"),
    ),
)
def test_real_registration_installs_one_resolver_and_direct_local_parent_loader(
    tmp_path,
    monkeypatch,
    target_unique_parameters,
    match_policy,
) -> None:
    backend = _backend(
        tmp_path,
        model_device="cuda:3",
        child_dtype="float32",
    )
    request = _request(
        tmp_path,
        target_unique_parameters=target_unique_parameters,
        match_policy=match_policy,
    )
    bridge = SimpleNamespace(
        donor_tokenizer_semantics_sha256="a" * 64,
        target_tokenizer_semantics_sha256="b" * 64,
        transport_profile=SimpleNamespace(
            target_row_capacity=EXPECTED_CHILD_TOPOLOGY["vocab_size"],
        ),
    )
    monkeypatch.setattr(
        production,
        "load_lfm2_direct_local_tokenizer",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        production,
        "load_strict_tokenizer_bridge_artifact",
        lambda *_args, **_kwargs: bridge,
    )
    monkeypatch.setattr(
        production,
        "Lfm2TargetSynthesisAdapter",
        _DummyTargetAdapter,
    )
    monkeypatch.setattr(
        production,
        "Lfm2DenseTargetTopologyCatalog",
        _DummyCatalog,
    )
    parent_loads = []

    def direct_local_parent(path, *, device, dtype):
        parent_loads.append((path, str(device), str(dtype)))
        return "local-parent"

    monkeypatch.setattr(
        production,
        "load_lfm2_direct_local_model",
        direct_local_parent,
    )
    registration = production.Lfm2Exact4xProductionRegistration(
        request=request,
        backend=backend,
        configuration_sha256="d" * 64,
    )

    entrypoint = production.assemble_lfm2_exact4x_production_entrypoint(
        registration
    )

    assert isinstance(entrypoint.system, production.AutomaticDynamicFoldingSystem)
    assert entrypoint.system.target_adapters.adapter_ids == (
        _DummyTargetAdapter.adapter_id,
    )
    assert entrypoint.system.stack_adapters.adapter_ids == (
        LFM2_EXACT4X_SEMANTIC_RESOLVER_ID,
    )
    assert entrypoint.system.topology_catalogs.catalog_ids == (
        _DummyCatalog.catalog_id,
    )
    assert len(entrypoint.parent_sources) == 1
    assert entrypoint.parent_sources[0].loader() == "local-parent"
    assert parent_loads == [
        (backend.parent_model_directory, "cpu", "torch.bfloat16")
    ]
    assert _DummyTargetAdapter.last_kwargs["terminal_topology"] is None
    module_source = inspect.getsource(production)
    assert "from_pretrained(" not in module_source
    assert "AutoModel" not in module_source
    assert "AutoTokenizer" not in module_source


def test_registration_rejects_target_above_authenticated_quarter_ceiling(
    tmp_path,
) -> None:
    backend = _backend(tmp_path)
    request = _request(
        tmp_path,
        target_unique_parameters=(
            backend.expected_parent_unique_parameters // 4 + 1
        ),
    )
    registration = production.Lfm2Exact4xProductionRegistration(
        request=request,
        backend=backend,
        configuration_sha256="d" * 64,
    )

    with pytest.raises(
        production.Lfm2Exact4xProductionError,
        match="at or below one quarter",
    ):
        production.assemble_lfm2_exact4x_production_entrypoint(registration)


def test_dense_budget_adapter_does_not_require_manual_terminal_topology() -> None:
    adapter = production.Lfm2TargetSynthesisAdapter(
        terminal_topology=None,
        evidence_fingerprint="e" * 64,
    )

    assert adapter.terminal_topology is None
