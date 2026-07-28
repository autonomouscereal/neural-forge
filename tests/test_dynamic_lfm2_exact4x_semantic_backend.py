from __future__ import annotations

import json
import inspect
from types import SimpleNamespace

import pytest

from experiments import run_lfm2_exact4x_composite_fold as exact4x_runner
from neural_forge.dynamic_behavior_execution import (
    resolve_behavior_inference_execution_contract,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import fingerprint_model_state
from neural_forge.dynamic_lfm2_adapter import (
    Lfm2DenseBudgetedSelectedTarget,
    Lfm2TargetSynthesisAdapter,
)
from neural_forge.dynamic_fold_budget import (
    ArchitectureDiscoveredBudgetSelection,
    ExactTopologyBudgetSelection,
)
from neural_forge.dynamic_parameter_frontier import (
    COMPONENT_ALLOCATION_POLICY,
    LFM2_DENSE_TOKEN_INTERFACE_MODE_CONFIG_KEY,
    DenseComponentAllocationReceipt,
)
from neural_forge.dynamic_target_synthesis import DynamicTargetSynthesisResult
from neural_forge import dynamic_lfm2_exact4x_semantic_backend as backend_module
from neural_forge.dynamic_lfm2_exact4x_semantic_backend import (
    Lfm2DecodedManualReviewReadyGate,
    Lfm2Exact4xSemanticCompiler,
    Lfm2Exact4xSemanticResourceResolver,
)
from neural_forge.dynamic_semantic_fold import SemanticWholeModelCompileRequest
from neural_forge import lfm2_exact4x_kernel_bridge as kernel_bridge
from neural_forge.lfm2_exact4x_contract import EXPECTED_PARENT_UNIQUE_PARAMETERS
from neural_forge.lfm2_native_exact4x_target import (
    LFM2_NATIVE_EXACT4X_LAYER_TYPES,
    LFM2_NATIVE_EXACT4X_TARGET_CLASS_ID,
    Lfm2NativeExact4xTargetError,
    derive_lfm2_native_exact4x_target_contract,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _sha(label: str) -> str:
    return canonical_sha256({"label": label})


def _small_complete_lfm2() -> object:
    model = transformers.Lfm2ForCausalLM(
        transformers.Lfm2Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            layer_types=["conv", "full_attention"],
            conv_L_cache=3,
            block_auto_adjust_ff_dim=False,
            tie_word_embeddings=False,
        )
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    model.eval()
    return model


class _FakeKernelConfig:
    def __init__(self, root) -> None:
        self.compile_artifact_root = root
        self.expected_semantic_cache_key = _sha("semantic-cache")
        self.expected_parent_unique_parameters = EXPECTED_PARENT_UNIQUE_PARAMETERS
        self.expected_bridge_artifact_sha256 = _sha("bridge")

    def kernel_arguments(self):
        return {}


class _FakeAssets:
    def __init__(self, root) -> None:
        self.fingerprint = _sha("asset-bundle")
        self.config = _FakeKernelConfig(root)
        self.donor_tokenizer = object()
        self.target_tokenizer = object()
        self.bridge = object()
        self.compiler_cache = object()
        self.semantic_cache = object()

    def verify(self):
        return self


def _selected_stub(source_graph_fingerprint: str):
    selected = object.__new__(Lfm2DenseBudgetedSelectedTarget)
    object.__setattr__(selected, "source_parent", "parent")
    object.__setattr__(
        selected,
        "source_graph_fingerprint",
        source_graph_fingerprint,
    )
    return selected


class _FakeNativeContract:
    target_class_id = LFM2_NATIVE_EXACT4X_TARGET_CLASS_ID
    target_unique_parameters = 100
    component_allocation_receipt_sha256 = _sha("allocation-receipt")
    component_allocation_topology_sha256 = _sha("allocation-topology")
    fingerprint = _sha("native-target-contract")
    topology_dict = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "tie_word_embeddings": False,
        "layer_types": ["conv", "full_attention"],
    }

    def as_dict(self):
        return {
            "target_class_id": self.target_class_id,
            "target_unique_parameters": self.target_unique_parameters,
            "target_topology": self.topology_dict,
            "fingerprint": self.fingerprint,
        }


def test_real_backend_compiler_reuses_one_blank_and_covers_every_component(
    tmp_path,
    monkeypatch,
) -> None:
    child = _small_complete_lfm2()
    parent = torch.nn.Linear(4, 4, bias=False).eval()
    source_graph_fingerprint = _sha("source-graph")
    selected = _selected_stub(source_graph_fingerprint)
    blank_state = fingerprint_model_state(child)
    target = SimpleNamespace(
        model=child,
        graph=SimpleNamespace(),
        adapter_payload=selected,
        blank_state_fingerprint=blank_state,
        request=SimpleNamespace(blank_seed=71),
        fingerprint=_sha("target"),
    )
    context = SimpleNamespace(
        child_model=child,
        parent_models={"parent": parent},
        parent_graphs={
            "parent": SimpleNamespace(fingerprint=source_graph_fingerprint)
        },
    )
    assets = _FakeAssets(tmp_path / "kernel-runs")
    evidence = SimpleNamespace(
        source=assets,
        root_fingerprint=assets.config.expected_semantic_cache_key,
        fingerprint=_sha("semantic-evidence"),
    )
    action = SimpleNamespace(action_sha256=_sha("whole-model-action"))
    request = SemanticWholeModelCompileRequest(
        target=target,
        action=action,
        context=context,
        evidence=evidence,
    )
    seen = {}
    native_contract = _FakeNativeContract()

    def fake_kernel(**kwargs):
        seen.update(kwargs)
        assert kwargs["target"] is target
        assert kwargs["target"].model is child
        assert kwargs["parent_model"] is parent
        assert kwargs["blank_seed"] == 71
        assert kwargs["existing_compiler_cache"] is assets.compiler_cache
        assert kwargs["existing_semantic_cache"] is assets.semantic_cache
        with torch.no_grad():
            for index, parameter in enumerate(child.parameters()):
                parameter.fill_((index + 1) / 100.0)
        body = {
            "complete_child_model_state_sha256": fingerprint_model_state(child),
            "existing_compiler_blank_used": True,
            "publication_deferred_to_outer_compiler": True,
            "preopened_semantic_resources_reused": True,
            "operation_audit": {
                "training_steps": 0,
                "optimizer_constructions": 0,
                "optimizer_steps": 0,
                "autograd_grad_calls": 0,
                "backward_calls": 0,
                "parent_forward_calls": 0,
                "parent_generate_calls": 0,
                "parent_trace_recaptures": 0,
                "parameters_pruned": 0,
                "parameters_quantized": 0,
                "parameters_sparsified": 0,
                "external_adapter_parameters": 0,
                "parent_sidecar_parameters": 0,
                "runtime_shared_parent_parameters": 0,
                "parent_model_loads_inside_kernel": 0,
                "target_model_materializations_inside_kernel": 0,
                "verification_reload_materializations": 0,
                "network_access": False,
            },
            "semantic_layer_stack": {"result_sha256": _sha("layer-stack")},
            "structural_interface_install": {
                "result_sha256": _sha("structural-interface")
            },
            "causal_token_bridge": {
                "result_sha256": _sha("causal-token-bridge")
            },
            "causal_output_install": {
                "result_sha256": _sha("causal-output")
            },
            "deferred_final_w2_install": {
                "result_sha256": _sha("deferred-w2")
            },
            "attention_lineage_sha256": _sha("attention-lineage"),
            "structural_attention_candidate_archive": {
                "manifest_sha256": _sha("attention-archive")
            },
            "short_conv_construction": "coordinate-complete",
            "body_global_coordinate_completion_recipe": (
                "bridge-procrustes-isometric"
            ),
            "neutral_norm_install": {"installed": True},
            "body_global_parent_geometry_sha256": _sha("global-geometry"),
            "parent_identity_sha256": _sha("parent-identity"),
            "parent_unique_parameters": 400,
            "target_unique_parameters": 100,
            "realized_parameter_compaction_ratio": 4.0,
            "native_target_class_id": native_contract.target_class_id,
            "native_target_contract_sha256": native_contract.fingerprint,
            "native_target_contract": native_contract.as_dict(),
            "target_topology": native_contract.topology_dict,
        }
        return {**body, "result_sha256": canonical_sha256(body)}

    monkeypatch.setattr(
        "neural_forge.dynamic_lfm2_exact4x_semantic_backend."
        "verify_compiler_authenticated_target",
        lambda value: value,
    )
    monkeypatch.setattr(
        "neural_forge.dynamic_lfm2_exact4x_semantic_backend."
        "validate_lfm2_dense_budgeted_child",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "neural_forge.dynamic_lfm2_exact4x_semantic_backend."
        "derive_lfm2_native_exact4x_target_contract",
        lambda *args, **kwargs: native_contract,
    )
    monkeypatch.setattr(
        "neural_forge.dynamic_lfm2_exact4x_semantic_backend."
        "_run_existing_exact4x_target",
        fake_kernel,
    )

    result = Lfm2Exact4xSemanticCompiler(assets).compile(request)

    parameter_paths = {
        name
        for name, _parameter in child.named_parameters(
            recurse=True,
            remove_duplicate=False,
        )
    }
    assert set(result.covered_target_parameter_paths) == parameter_paths
    assert {
        receipt.component_id for receipt in result.component_receipts
    } == {
        "token-interface-and-causal-head",
        "residual-attention",
        "short-convolution",
        "gated-mlp",
        "normalization-and-global-residual",
    }
    assert all(receipt.covered_parameter_paths for receipt in result.component_receipts)
    assert result.operation_audit["target_model_materializations_inside_kernel"] == 0
    assert result.operation_audit["parent_model_loads_inside_kernel"] == 0
    assert result.operation_audit["complete_model_compaction"] is True
    assert fingerprint_model_state(child) != blank_state
    assert seen["run_directory"].parent == assets.config.compile_artifact_root


def test_run_existing_target_owns_in_place_and_outer_publication_flags(
    monkeypatch,
) -> None:
    target = object()
    parent = object()
    observed = {}

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(exact4x_runner, "run", fake_run)
    assert exact4x_runner.run_existing_target(
        target=target,
        parent_model=parent,
        run_directory="unused",
    ) == {"ok": True}
    assert observed["existing_target"] is target
    assert observed["existing_parent_model"] is parent
    assert observed["defer_publication_to_caller"] is True


def _native_unique_parameters(
    hidden_size: int,
    intermediate_size: int,
    query_heads: int,
    key_value_heads: int,
) -> int:
    result = hidden_size + 2 * 4_096 * hidden_size
    for layer_type in LFM2_NATIVE_EXACT4X_LAYER_TYPES:
        result += 2 * hidden_size
        if layer_type == "conv":
            result += 4 * hidden_size * hidden_size + 3 * hidden_size
        else:
            result += (
                2
                * hidden_size
                * 64
                * (query_heads + key_value_heads)
                + 2 * 64
            )
        result += 3 * hidden_size * intermediate_size
    return result


def _component_receipt(
    *,
    source_graph_fingerprint: str,
    target_unique_parameters: int,
    topology_sha256: str,
) -> DenseComponentAllocationReceipt:
    option_id = f"native-{target_unique_parameters}"
    topology_selection = ExactTopologyBudgetSelection(
        problem_fingerprint=_sha("topology-problem"),
        selected_option_ids=(option_id,),
        shared_resource_ids=(),
        exact_unique_parameters=target_unique_parameters,
        serialized_bytes=target_unique_parameters * 4,
        total_predicted_risk=0.0,
        states_considered=1,
        peak_retained_states=1,
    )
    allocation = ArchitectureDiscoveredBudgetSelection(
        problem_fingerprint=_sha("allocation-problem"),
        source_graph_fingerprint=source_graph_fingerprint,
        topology_selection=topology_selection,
        rule_retention=(),
        semantic_retention=(),
        discovered_group_kinds=(),
    )
    return DenseComponentAllocationReceipt(
        policy=COMPONENT_ALLOCATION_POLICY,
        topology_option_id=option_id,
        topology_configuration_sha256=topology_sha256,
        source_graph_fingerprint=source_graph_fingerprint,
        source_unique_parameters=EXPECTED_PARENT_UNIQUE_PARAMETERS,
        source_logical_parameters=EXPECTED_PARENT_UNIQUE_PARAMETERS,
        target_unique_parameters=target_unique_parameters,
        target_logical_parameters=target_unique_parameters,
        allocation_selection=allocation,
    )


def _native_target_stub(
    *,
    hidden_size: int,
    intermediate_size: int,
    query_heads: int,
    key_value_heads: int,
):
    source_graph_fingerprint = _sha("native-source-graph")
    target_graph_fingerprint = _sha(
        f"native-target-graph-{hidden_size}-{intermediate_size}"
    )
    topology_sha256 = _sha(
        f"native-topology-{hidden_size}-{intermediate_size}"
    )
    bridge_sha256 = _sha("native-bridge")
    target_unique_parameters = _native_unique_parameters(
        hidden_size,
        intermediate_size,
        query_heads,
        key_value_heads,
    )
    receipt = _component_receipt(
        source_graph_fingerprint=source_graph_fingerprint,
        target_unique_parameters=target_unique_parameters,
        topology_sha256=topology_sha256,
    )
    selected = object.__new__(Lfm2DenseBudgetedSelectedTarget)
    for name, value in {
        "source_parent": "parent",
        "source_graph_fingerprint": source_graph_fingerprint,
        "source_state_fingerprint": _sha("native-parent-state"),
        "parent_tokenizer_semantics_sha256": _sha("parent-tokenizer"),
        "child_tokenizer_semantics_sha256": _sha("child-tokenizer"),
        "tokenizer_bridge_artifact_sha256": bridge_sha256,
        "tokenizer_transport_profile_sha256": _sha("bridge-profile"),
        "catalog_id": "native-test-catalog",
        "component_allocation_policy": COMPONENT_ALLOCATION_POLICY,
        "component_allocation_receipt_sha256": receipt.fingerprint,
        "component_allocation_topology_sha256": topology_sha256,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "query_heads": query_heads,
        "key_value_heads": key_value_heads,
        "head_dim": 64,
        "vocabulary_size": 4_096,
        "fir_length": 3,
        "layer_types": LFM2_NATIVE_EXACT4X_LAYER_TYPES,
        "ordered_source_spans": tuple(
            (index,) for index in range(len(LFM2_NATIVE_EXACT4X_LAYER_TYPES))
        ),
        "tie_word_embeddings": False,
        "exact_target_unique_parameters": target_unique_parameters,
        "exact_target_logical_parameters": target_unique_parameters,
        "configuration": (
            (LFM2_DENSE_TOKEN_INTERFACE_MODE_CONFIG_KEY, "untied"),
        ),
    }.items():
        object.__setattr__(selected, name, value)
    config = SimpleNamespace(
        vocab_size=4_096,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=len(LFM2_NATIVE_EXACT4X_LAYER_TYPES),
        num_attention_heads=query_heads,
        num_key_value_heads=key_value_heads,
        head_dim=64,
        tie_word_embeddings=False,
        layer_types=list(LFM2_NATIVE_EXACT4X_LAYER_TYPES),
    )
    request = SimpleNamespace(
        component_allocation_receipt=receipt,
        fingerprint=_sha(
            f"native-request-{hidden_size}-{intermediate_size}"
        ),
        exact_unique_parameters=target_unique_parameters,
        fold_method_id=None,
    )
    target = object.__new__(DynamicTargetSynthesisResult)
    for name, value in {
        "request": request,
        "adapter_id": Lfm2TargetSynthesisAdapter.adapter_id,
        "adapter_version": Lfm2TargetSynthesisAdapter.version,
        "model": SimpleNamespace(config=config, training=False),
        "graph": SimpleNamespace(
            fingerprint=target_graph_fingerprint,
            unique_parameter_count=target_unique_parameters,
        ),
        "resource_profile": SimpleNamespace(
            unique_parameters=target_unique_parameters,
            family="lfm2",
            modalities=("text",),
            layer_count=len(LFM2_NATIVE_EXACT4X_LAYER_TYPES),
        ),
        "topology_fingerprint": _sha(
            f"compiler-topology-{hidden_size}-{intermediate_size}"
        ),
        "parent_graph_fingerprints": (
            ("parent", source_graph_fingerprint),
        ),
        "adapter_payload": selected,
        "blank_provenance": SimpleNamespace(
            compiler_created_blank=True,
            all_parameters_zero=True,
        ),
    }.items():
        object.__setattr__(target, name, value)
    return target, receipt, bridge_sha256


@pytest.mark.parametrize(
    ("hidden_size", "intermediate_size", "query_heads", "key_value_heads"),
    ((776, 700, 10, 5), (800, 574, 12, 6)),
)
def test_registered_native_exact4x_contract_routes_h776_and_h800(
    hidden_size,
    intermediate_size,
    query_heads,
    key_value_heads,
) -> None:
    target, receipt, bridge_sha256 = _native_target_stub(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
    )
    contract = derive_lfm2_native_exact4x_target_contract(
        target,
        expected_parent_unique_parameters=EXPECTED_PARENT_UNIQUE_PARAMETERS,
        expected_bridge_artifact_sha256=bridge_sha256,
    )

    assert contract.target_class_id == LFM2_NATIVE_EXACT4X_TARGET_CLASS_ID
    assert contract.target_unique_parameters == _native_unique_parameters(
        hidden_size,
        intermediate_size,
        query_heads,
        key_value_heads,
    )
    assert contract.target_unique_parameters <= EXPECTED_PARENT_UNIQUE_PARAMETERS // 4
    assert contract.topology_dict["hidden_size"] == hidden_size
    assert contract.topology_dict["intermediate_size"] == intermediate_size
    assert contract.component_allocation_receipt_sha256 == receipt.fingerprint
    assert contract.as_dict()["full_coverage_gqa"] is True


def test_registered_native_exact4x_contract_rejects_topology_and_count_tamper() -> None:
    target, _receipt, bridge_sha256 = _native_target_stub(
        hidden_size=800,
        intermediate_size=574,
        query_heads=12,
        key_value_heads=6,
    )
    target.model.config.hidden_size = 801
    with pytest.raises(Lfm2NativeExact4xTargetError, match="outside"):
        derive_lfm2_native_exact4x_target_contract(
            target,
            expected_parent_unique_parameters=EXPECTED_PARENT_UNIQUE_PARAMETERS,
            expected_bridge_artifact_sha256=bridge_sha256,
        )

    target.model.config.hidden_size = 800
    target.resource_profile.unique_parameters += 1
    with pytest.raises(Lfm2NativeExact4xTargetError, match="outside"):
        derive_lfm2_native_exact4x_target_contract(
            target,
            expected_parent_unique_parameters=EXPECTED_PARENT_UNIQUE_PARAMETERS,
            expected_bridge_artifact_sha256=bridge_sha256,
        )


def test_registered_native_exact4x_contract_rejects_receipt_tamper() -> None:
    target, receipt, bridge_sha256 = _native_target_stub(
        hidden_size=776,
        intermediate_size=700,
        query_heads=10,
        key_value_heads=5,
    )
    replacement = _component_receipt(
        source_graph_fingerprint=receipt.source_graph_fingerprint,
        target_unique_parameters=receipt.target_unique_parameters,
        topology_sha256=_sha("tampered-topology"),
    )
    target.request.component_allocation_receipt = replacement
    with pytest.raises(Lfm2NativeExact4xTargetError, match="outside"):
        derive_lfm2_native_exact4x_target_contract(
            target,
            expected_parent_unique_parameters=EXPECTED_PARENT_UNIQUE_PARAMETERS,
            expected_bridge_artifact_sha256=bridge_sha256,
        )


@pytest.mark.parametrize(
    ("hidden_size", "intermediate_size", "query_heads", "key_value_heads"),
    ((776, 700, 10, 5), (800, 574, 12, 6)),
)
def test_semantic_resolver_matches_registered_h776_and_h800_native_class(
    hidden_size,
    intermediate_size,
    query_heads,
    key_value_heads,
    monkeypatch,
) -> None:
    target, _receipt, bridge_sha256 = _native_target_stub(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
    )
    config = SimpleNamespace(
        expected_parent_unique_parameters=EXPECTED_PARENT_UNIQUE_PARAMETERS,
        expected_bridge_artifact_sha256=bridge_sha256,
        source_parent_name=None,
        expected_parent_graph_fingerprint=None,
    )
    monkeypatch.setattr(backend_module, "verify_compiler_authenticated_target", lambda _v: None)
    monkeypatch.setattr(backend_module, "validate_lfm2_dense_budgeted_child", lambda *_v: None)

    assert Lfm2Exact4xSemanticResourceResolver._matches(config, target)


def test_legacy_standalone_runner_defaults_keep_original_materialize_publish_path() -> None:
    signature = inspect.signature(exact4x_runner.run)
    assert signature.parameters["existing_target"].default is None
    assert signature.parameters["existing_parent_model"].default is None
    assert signature.parameters["defer_publication_to_caller"].default is False
    source = inspect.getsource(exact4x_runner.run)
    assert "target = DynamicTargetSynthesizer(" in source
    assert 'checkpoint_io.save(child, candidate_checkpoint)' in source
    assert "dossier = review_frontier(" in source
    assert 'target_model_materializations_inside_kernel = 1' in source
    assert 'parent_model_loads_inside_kernel = 1' in source


class _ReviewTokenizer:
    eos_token_id = 9
    pad_token_id = 0

    def __call__(self, _prompt, **_kwargs):
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        return "A coherent decoded response " + " ".join(map(str, token_ids))


class _ReviewModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, **_kwargs):
        return input_ids

    def generate(self, input_ids, **_kwargs):
        self(input_ids=input_ids)
        continuation = torch.tensor([[5, 6, 9]], device=input_ids.device)
        return torch.cat((input_ids, continuation), dim=1)


def test_decoded_gate_publishes_outputs_without_intelligence_verdict(tmp_path) -> None:
    model = _ReviewModel().eval()
    state = fingerprint_model_state(model)
    prompts = (
        {
            "id": "hello",
            "prompt": "Hello",
            "category": "conversation",
            "context": None,
            "parent_response": None,
        },
    )
    gate = Lfm2DecodedManualReviewReadyGate(
        tokenizer=_ReviewTokenizer(),
        prompts=prompts,
        output_root=tmp_path / "reviews",
        maximum_prompt_tokens=32,
        maximum_new_tokens=8,
        baseline_fingerprint=_sha("parent-envelope"),
        prompt_set_fingerprint=canonical_sha256([dict(value) for value in prompts]),
        tokenizer_fingerprint=_sha("target-tokenizer"),
    )

    decision = gate.evaluate(
        SimpleNamespace(
            execution_context=SimpleNamespace(child_model=model),
        )
    )
    fresh_decision = gate(model)

    assert decision.accepted and fresh_decision.accepted
    assert "No automatic intelligence verdict" in decision.reason
    assert dict(decision.metrics)["automatic_behavior_verdict_performed"] is False
    assert dict(decision.metrics)["accepted_means_ready_for_manual_review"] is True
    assert resolve_behavior_inference_execution_contract(gate).entrypoint == "generate"
    dossiers = sorted((tmp_path / "reviews").glob("*.json"))
    assert len(dossiers) == 2
    for path in dossiers:
        dossier = json.loads(path.read_text(encoding="ascii"))
        assert dossier["response_count"] == 1
        assert dossier["automatic_behavior_verdict_performed"] is False
        assert dossier["manual_intelligence_review_required"] is True
        assert dossier["intelligence_success_claimed"] is False
        assert dossier["numeric_diagnostics_have_acceptance_authority"] is False
        assert "automatic_score" not in dossier
        assert "intelligence_accepted" not in dossier
        assert dossier["responses"][0]["response"].startswith(
            "A coherent decoded response"
        )
    assert fingerprint_model_state(model) == state


def test_packaged_backend_import_isolated_from_transitional_experiment_kernel(
    monkeypatch,
) -> None:
    source = inspect.getsource(backend_module)
    assert "from experiments" not in source
    assert "import experiments" not in source
    assert kernel_bridge.TRANSITIONAL_REPOSITORY_RUNTIME_DEPENDENCY is True

    def unavailable(_name):
        raise ModuleNotFoundError("experiment package intentionally absent")

    monkeypatch.setattr(kernel_bridge.importlib, "import_module", unavailable)
    with pytest.raises(
        kernel_bridge.Lfm2Exact4xKernelPackagingError,
        match="transitional Neural Forge repository runtime",
    ):
        kernel_bridge.run_existing_target(target=object(), parent_model=object())
