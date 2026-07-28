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
from neural_forge.dynamic_lfm2_adapter import Lfm2DenseBudgetedSelectedTarget
from neural_forge import dynamic_lfm2_exact4x_semantic_backend as backend_module
from neural_forge.dynamic_lfm2_exact4x_semantic_backend import (
    Lfm2DecodedManualReviewReadyGate,
    Lfm2Exact4xSemanticCompiler,
)
from neural_forge.dynamic_semantic_fold import SemanticWholeModelCompileRequest
from neural_forge import lfm2_exact4x_kernel_bridge as kernel_bridge


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
